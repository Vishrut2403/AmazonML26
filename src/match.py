"""Pair features, LightGBM matcher, decision rule and submission files.

Decision rule: each S2/S3 record (o) is assigned to its single highest-probability S1 candidate if that
probability >= thr and beats the o's second-best candidate by >= margin (look-alike businesses); both are
tuned for macro F0.5 on validation. This uses the dataset property that every o matches at most one S1.

Validation: S1 entities whose numeric id % 100 == 0 (1%). Matching still runs against the full S1 pool, so
competing entities and decoy records are as realistic as on the test set. Training pairs come only from o
records with no validation entity among their candidates.

Usage:
  python src/match.py train    # fit + validate -> work/model.txt, work/thr.json
  python src/match.py predict  # score test -> output/matching_results.tsv, output/candidate_pairs.tsv
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from block import sound
from common import macro_f05, read_tsv
from prep import DATA, ROOT, WORK

TRAIN_O_RATE = 0.30  # fraction of eligible o records sampled for training (~27M pairs at K=10; ~10 GB peak)
# pseudo-labels for a country without training labels: test records the first model is sure about
PSEUDO = WORK / "pseudo.parquet"
PSEUDO_COUNTRY = "France"
PSEUDO_N = 150_000  # o records sampled from the confident positives and decoys, in their natural ratio
PSEUDO_SCORED = 300_000  # o records scored to find them (a uniform sample, so the group ratio is unbiased)

# (feature name, column, scorer) for rapidfuzz pairwise similarities between o and its S1 candidate
SIMS = [
    ("name_ratio", "name", fuzz.ratio),
    ("core_ratio", "core", fuzz.ratio),
    ("core_tset", "core", fuzz.token_set_ratio),
    ("core_tsort", "core", fuzz.token_sort_ratio),
    ("core_partial", "core", fuzz.partial_ratio),
    ("core_jw", "core", JaroWinkler.normalized_similarity),
    ("addr_ratio", "addr", fuzz.ratio),
    ("addr_tset", "addr", fuzz.token_set_ratio),
    ("addr_tsort", "addr", fuzz.token_sort_ratio),
    ("addr_partial", "addr", fuzz.partial_ratio),
    ("nums_tset", "nums", fuzz.token_set_ratio),
    ("snd_ratio", "snd", fuzz.ratio),          # sound keys: transliterated vs English spellings
    ("snd_tset", "snd", fuzz.token_set_ratio),
    ("sq_ratio", "sq", fuzz.ratio),            # spaces removed: website-style names
    ("sq_partial", "sq", fuzz.partial_ratio),
    ("street_tset", "street", fuzz.token_set_ratio),  # address without numbers: judged apart from numbers
    ("street_ratio", "street", fuzz.ratio),
    ("pt_core_tset", "core_p", fuzz.token_set_ratio),  # vs the partner record (see s1_stats)
    ("pt_addr_tset", "addr_p", fuzz.token_set_ratio),
]
FEATS = [n for n, _, _ in SIMS] + [
    "tok", "rank", "tok_rel", "tok_gap", "n_cand", "src",
    "num_first_eq", "num_jacc", "len_core", "len_core_s", "len_addr", "len_addr_s",
    "s_n", "s_top1", "num_o_in_s", "num_s_in_o", "pt_tok",
    "num_diff", "num_up", "o_no_num",
]


def records(split):
    """Normalized records of a split with the same row index block.py used, plus sound keys and squashed name."""
    return pl.read_parquet(WORK / f"{split}.parquet").with_row_index("idx").with_columns(
        snd=sound(pl.col("core")), sq=pl.col("core").str.replace_all(" ", ""),
        street=pl.col("addr").str.replace_all(r"\d+", " ").str.replace_all(" +", " ").str.strip_chars(),
    )


def cand_chunks(split):
    return sorted((WORK / f"{split}_cand").glob("*.parquet"))


def s1_stats(split):
    """Per S1 record: how many o records have it as a candidate (s_n), how many rank it first (s_top1), and
    the two o records with the highest blocking score for it (o1, o2 with scores t1, t2).

    s_n and s_top1 are divided by their split-wide mean, because test has more S2/S3 records per S1 than train.
    o1/o2 give each pair a "partner": the strongest *other* record competing for the same S1. Records of one
    business from S2 and S3 often resemble each other more than they resemble the S1 record.
    """
    st = (
        pl.scan_parquet(cand_chunks(split)).sort("tok", descending=True).group_by("s")
        .agg(
            s_n=pl.len(), s_top1=(pl.col("rank") == 1).sum(),
            o1=pl.col("o").first(), o2=pl.col("o").slice(1, 1).first(),
            t1=pl.col("tok").first(), t2=pl.col("tok").slice(1, 1).first(),
        ).collect()
    )
    return st.with_columns((pl.col(c) / pl.col(c).mean()).cast(pl.Float32) for c in ("s_n", "s_top1"))


def featurize(cand, recs, stats):
    """Attach o-side and S1-side text to candidate pairs (o, s, tok, rank) and compute FEATS."""
    side = recs.select("idx", "src", "name", "core", "addr", "nums", "snd", "sq", "street")
    d = (
        cand.join(stats, on="s", how="left")
        .with_columns(
            partner=pl.when(pl.col("o1") == pl.col("o")).then("o2").otherwise("o1"),
            pt_tok=pl.when(pl.col("o1") == pl.col("o")).then("t2").otherwise("t1"),
        )
        .join(side, left_on="o", right_on="idx")
        .join(side.drop("src"), left_on="s", right_on="idx", suffix="_s")
        .join(recs.select("idx", core_p="core", addr_p="addr"), left_on="partner", right_on="idx", how="left")
        .with_columns(pl.col("core_p", "addr_p").fill_null(""))
        .with_columns(
            tok_top=pl.col("tok").max().over("o"),
            tok_2nd=pl.col("tok").top_k(2).min().over("o"),
            n_cand=pl.len().over("o"),
            n1=pl.col("nums").str.split(" "),
            n2=pl.col("nums_s").str.split(" "),
        )
        .with_columns(
            tok_rel=pl.col("tok") / pl.col("tok_top"),
            # how far this candidate is ahead of the o's runner-up (negative for non-top candidates)
            tok_gap=pl.col("tok") - pl.when(pl.col("rank") == 1).then("tok_2nd").otherwise("tok_top"),
            num_first_eq=(pl.col("n1").list.first() == pl.col("n2").list.first()).cast(pl.Float32),
            num_jacc=pl.col("n1").list.set_intersection("n2").list.len()
            / pl.col("n1").list.set_union("n2").list.len(),
            num_o_in_s=(pl.col("n1").list.set_difference("n2").list.len() == 0).cast(pl.Float32),
            num_s_in_o=(pl.col("n2").list.set_difference("n1").list.len() == 0).cast(pl.Float32),
            len_core=pl.col("core").str.len_chars(), len_core_s=pl.col("core_s").str.len_chars(),
            len_addr=pl.col("addr").str.len_chars(), len_addr_s=pl.col("addr_s").str.len_chars(),
            # signed change of the first number: decoys copy a business and move its number up by 1-60, while
            # true copies keep it or change it at random
            num_diff=(pl.col("n1").list.first().cast(pl.Float32, strict=False)
                      - pl.col("n2").list.first().cast(pl.Float32, strict=False)),
            o_no_num=(pl.col("nums") == "").cast(pl.Float32),
        )
        .with_columns(num_up=((pl.col("num_diff") >= 1) & (pl.col("num_diff") <= 60)).cast(pl.Float32))
    )
    cols = {}
    for name, col, scorer in SIMS:
        a, b = (col[:-2], col) if col.endswith("_p") else (col, col + "_s")
        cols[name] = process.cpdist(d[a].to_list(), d[b].to_list(), scorer=scorer, workers=-1)
    d = d.with_columns(pl.Series(k, v, dtype=pl.Float32) for k, v in cols.items())
    # empty numbers on either side: the comparison is meaningless, let the model see NaN
    d = d.with_columns(
        pl.when((pl.col("nums") == "") | (pl.col("nums_s") == "")).then(None).otherwise(pl.col(c)).alias(c)
        for c in ("num_first_eq", "num_jacc", "nums_tset", "num_o_in_s", "num_s_in_o", "num_diff", "num_up")
    )
    # no partner (the S1 has a single candidate record): partner features are undefined
    d = d.with_columns(
        pl.when(pl.col("partner").is_null()).then(None).otherwise(pl.col(c)).alias(c)
        for c in ("pt_core_tset", "pt_addr_tset", "pt_tok")
    )
    return d.select("o", "s", *(pl.col(f).cast(pl.Float32) for f in FEATS))


def truth_pairs(recs):
    """(o, s_true) index pairs from the training ground truth."""
    ids = recs.select("idx", "entity_id")
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv").drop_nulls()
    return (
        gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
        .join(ids, left_on="source1_entity_id", right_on="entity_id").rename({"idx": "s_true"})
        .join(ids, left_on="matched_entity_ids", right_on="entity_id").select(o="idx", s_true="s_true")
    )


def best_per_o(scored):
    """Highest-probability candidate of each o (the only S1 it may be assigned to), its probability p and
    the probability p2 of the o's second-best candidate (0 when there is none)."""
    return scored.sort("p", descending=True).group_by("o").agg(
        s=pl.col("s").first(), p=pl.col("p").first(), p2=pl.col("p").slice(1, 1).first().fill_null(0.0)
    )


def accept(best, thr, margin):
    """Rows of best_per_o output that get assigned under the decision rule."""
    return best.filter((pl.col("p") >= thr) & (pl.col("p") - pl.col("p2") >= margin))


def val_score(best, val_s, truth, thr, margin):
    """Macro F0.5 over validation S1 entities under the decision rule."""
    pred = accept(best, thr, margin).filter(pl.col("s").is_in(val_s.implode())).group_by("s").agg("o")
    pred = {s: set(os) for s, os in pred.iter_rows()}
    return macro_f05(pred, truth)


def train():
    recs = records("train")
    tp = truth_pairs(recs)
    stats = s1_stats("train")
    s1 = recs.filter(pl.col("src") == 1)
    val_s = s1.filter(pl.col("entity_id").str.slice(3).cast(pl.Int64) % 100 == 0)["idx"]
    print(f"validation S1 entities: {len(val_s):,}")

    # pass 1: choose the o records to featurize (validation-linked ones, plus a sample of the rest for
    # training), so only the text of records that are actually compared stays in memory
    rng = np.random.default_rng(0)
    picks = []
    for f in cand_chunks("train"):
        c = pl.read_parquet(f, columns=["o", "s"]).with_columns(has_val=pl.col("s").is_in(val_s.implode()).any().over("o"))
        o = c.filter(~pl.col("has_val"))["o"].unique()
        picks.append((c.filter("has_val")["o"].unique(), o.filter(pl.Series(rng.random(len(o)) < TRAIN_O_RATE))))
    needed = pl.concat(
        [s1["idx"], stats["o1"].drop_nulls(), stats["o2"].drop_nulls()] + [x for pair in picks for x in pair]
    ).unique()
    recs = recs.filter(pl.col("idx").is_in(needed.implode())).drop("entity_id", "country")
    del s1, needed
    print(f"text kept for {recs.height:,} records")

    # pass 2: features for training pairs and for every candidate of validation-linked records
    parts, val_parts = [], []
    for f, (val_o, keep) in zip(cand_chunks("train"), picks):
        c = pl.read_parquet(f)
        val_parts.append(featurize(c.filter(pl.col("o").is_in(val_o.implode())), recs, stats))
        parts.append(featurize(c.filter(pl.col("o").is_in(keep.implode())), recs, stats))
    del recs, picks  # all text comparisons are done
    val_x = pl.concat(val_parts)
    tr = pl.concat(parts).join(tp.rename({"s_true": "s"}).with_columns(y=pl.lit(1)), on=["o", "s"], how="left")
    del val_parts, parts
    if PSEUDO.exists():  # test pairs labelled by a first model (see pseudo()); validation stays train-only
        ps = pl.read_parquet(PSEUDO)
        print(f"adding {ps.height:,} pseudo-labelled test pairs ({ps['o'].n_unique():,} records)")
        tr = pl.concat([tr.select(*FEATS, pl.col("y").fill_null(0)), ps.select(*FEATS, pl.col("y").cast(pl.Int32))], rechunk=False)
    y = tr["y"].fill_null(0).to_numpy()
    x = tr.select(FEATS).to_numpy()
    del tr
    print(f"training on {len(y):,} pairs, {y.mean():.3f} positive")

    # early stopping on all candidates of 200k validation-linked records (~2M pairs)
    es_o = val_x["o"].unique()
    es_o = es_o.sample(min(200_000, len(es_o)), seed=0)
    es = val_x.filter(pl.col("o").is_in(es_o.implode()))
    es_y = es.join(tp.rename({"s_true": "s"}).with_columns(y=pl.lit(1)), on=["o", "s"], how="left")["y"].fill_null(0)
    train_set = lgb.Dataset(x, y, feature_name=FEATS)
    model = lgb.train(
        {"objective": "binary", "learning_rate": 0.02, "num_leaves": 255, "min_data_in_leaf": 200,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1},
        train_set,
        num_boost_round=5000,
        valid_sets=[lgb.Dataset(es.select(FEATS).to_numpy(), es_y.to_numpy(), reference=train_set)],
        callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(100)],
    )
    del x, train_set, es
    print(f"best iteration {model.best_iteration}")
    model.save_model(str(WORK / "model.txt"))
    imp = sorted(zip(FEATS, model.feature_importance("gain")), key=lambda t: -t[1])
    print("top features:", [n for n, _ in imp[:10]])

    # score every candidate of every o that could be assigned to a validation entity
    scored = val_x.select("o", "s", p=pl.Series(model.predict(val_x.select(FEATS).to_numpy())))
    scored.write_parquet(WORK / "val_scored.parquet")  # kept for error analysis
    best = best_per_o(scored)
    truth = {s: set() for s in val_s}
    for o, s in tp.filter(pl.col("s_true").is_in(val_s.implode())).iter_rows():
        truth[s].add(o)
    results = {
        (thr, m): val_score(best, val_s, truth, thr, m)
        for thr in np.arange(0.40, 0.91, 0.05).round(2) for m in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
    }
    for (thr, m), f in results.items():
        if m in (0.0, 0.1):
            print(f"thr {thr:.2f} margin {m:.2f}  val macro F0.5 {f:.4f}")
    thr, m = max(results, key=results.get)
    (WORK / "thr.json").write_text(json.dumps({"thr": float(thr), "margin": m, "val_f05": results[thr, m]}))
    print(f"best thr {thr} margin {m} -> val macro F0.5 {results[thr, m]:.4f}")


def write_lists(pairs, recs, col, path, parts=10):
    """Write one row per S1 entity (empty list when none) with its comma-joined S2/S3 ids, tab-separated.

    pairs: LazyFrame of (o, s) row indices. S1 records occupy row indices 0..n1-1 of the split parquet, so the
    file is written in `parts` slices of S1 to keep memory flat even for ~200M candidate pairs.
    """
    ids = recs["entity_id"]
    s1 = recs.filter(pl.col("src") == 1).select(s="idx", source1_entity_id="entity_id")
    step = -(-s1.height // parts)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for lo in range(0, s1.height, step):
            part = pairs.filter(pl.col("s").is_between(lo, lo + step - 1)).collect()
            lists = (
                part.with_columns(eid=ids.gather(part["o"])).group_by("s")
                .agg(pl.col("eid").unique().sort().str.join(",").alias(col))
            )
            s1.slice(lo, step).join(lists, on="s", how="left") \
                .select("source1_entity_id", pl.col(col).fill_null("")) \
                .write_csv(fh, separator="\t", quote_style="never", include_header=False)


def predict():
    recs = records("test")
    model = lgb.Booster(model_file=str(WORK / "model.txt"))
    rule = json.loads((WORK / "thr.json").read_text())
    thr, margin = rule["thr"], rule.get("margin", 0.0)
    stats = s1_stats("test")
    matched = []
    sc_dir = WORK / "test_scored"  # every candidate's score, kept for blending with other models
    sc_dir.mkdir(exist_ok=True)
    for f in cand_chunks("test"):
        # each chunk holds every candidate of its o records, so the best-per-o decision is local to the chunk
        x = featurize(pl.read_parquet(f), recs, stats)
        scored = x.select("o", "s", p=pl.Series(model.predict(x.select(FEATS).to_numpy())))
        scored.write_parquet(sc_dir / f.name)
        matched.append(accept(best_per_o(scored), thr, margin))
        print(f.name, flush=True)
    matched = pl.concat(matched)
    matched.write_parquet(WORK / "test_matched.parquet")
    out = ROOT / "output"
    write_lists(matched.lazy().select("o", "s"), recs, "matched_entity_ids", out / "matching_results.tsv")
    write_lists(pl.scan_parquet(cand_chunks("test")).select("o", "s"), recs, "candidate_entity_ids",
                out / "candidate_pairs.tsv")
    print(f"{matched.height:,} matched records over {matched['s'].n_unique():,} S1 entities -> {out}")


def pseudo():
    """Label test candidates of PSEUDO_COUNTRY with the current model (work/model.txt) -> work/pseudo.parquet.

    Confident positives: best p >= 0.95 and ahead of the runner-up by >= 0.5 -> best pair 1, the o's other pairs 0.
    Decoys: best p < 0.05 -> all pairs 0. PSEUDO_N o records are sampled from both groups together, out of
    PSEUDO_SCORED records scored.
    """
    recs = records("test")
    model = lgb.Booster(model_file=str(WORK / "model.txt"))
    stats = s1_stats("test")
    target = recs.filter((pl.col("country") == PSEUDO_COUNTRY) & (pl.col("src") != 1))["idx"]
    target = target.sample(min(PSEUDO_SCORED, len(target)), seed=0)
    kept = []
    for f in cand_chunks("test"):
        c = pl.read_parquet(f)
        c = c.filter(pl.col("o").is_in(target.implode()))
        if c.height == 0:
            continue
        x = featurize(c, recs, stats)
        x = x.with_columns(p=pl.Series(model.predict(x.select(FEATS).to_numpy())))
        best = best_per_o(x.select("o", "s", "p"))
        pos = best.filter((pl.col("p") >= 0.95) & (pl.col("p") - pl.col("p2") >= 0.5)).select("o", s_pos="s")
        neg = best.filter(pl.col("p") < 0.05).select("o")
        x = x.join(pl.concat([pos, neg.with_columns(s_pos=pl.lit(None, pl.UInt32))]), on="o")
        kept.append(x.with_columns(y=(pl.col("s") == pl.col("s_pos")).fill_null(False).cast(pl.Int32)).drop("s_pos"))
        print(f.name, f"{best.height:,} o: {pos.height:,} positive, {neg.height:,} decoy", flush=True)
    kept = pl.concat(kept)
    groups = kept.group_by("o").agg(pos=pl.col("y").max())
    n_pos, n_neg = groups["pos"].sum(), groups.height - groups["pos"].sum()
    pick = groups.sample(min(PSEUDO_N, groups.height), seed=0)
    kept = kept.join(pick.select("o"), on="o")
    kept.write_parquet(PSEUDO)
    print(f"{PSEUDO_COUNTRY}: scored {len(target):,} o; {n_pos:,} confident positive, {n_neg:,} decoy; sampled {pick.height:,} "
          f"({pick['pos'].sum():,} positive) -> {kept.height:,} pairs, {kept['y'].mean():.3f} positive -> {PSEUDO}")


if __name__ == "__main__":
    {"train": train, "predict": predict, "pseudo": pseudo}[sys.argv[1]]()
