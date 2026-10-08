"""Second pass: re-score candidate pairs using how the other records competing for the same S1 entity scored.

First pass: two LightGBM models on the match.py features, each trained on half of the sampled training
records (fold = record index parity). Every candidate pair gets a first-pass probability p1: sampled
training records get the other half's model (so p1 is never predicted by a model that saw its label), all
other records and every test record get the average of both models.

Second pass: the match.py features plus, for a pair (o, s):
  p1, p1_rank / p1_gap  where the pair stands among o's candidates
  s_other_p            best p1 of any other record for the same S1 entity
  s_n_hi, s_sum_other  how many other records score above 0.5 for it, and their summed p1
  pp_*                 similarity of o to that best other record ("p1 partner"), and whether it is the same source
Records of one business in Source 2 and Source 3 often look more alike than either looks like Source 1, and a
record competing with a very confident one for the same entity is less likely to be the match.

The decision rule is the same as match.py (best candidate, probability threshold, runner-up margin), tuned on
the same validation entities. Work files go to work/stage2/ and are processed one candidate chunk at a time.
Usage: python src/stage2.py train     (after block.py train)  -> work/stage2/*.txt, thr_stage2.json
       python src/stage2.py predict   (after block.py test)   -> output/matching_results.tsv, candidate_pairs.tsv,
                                                              work/stage2/test_scored/ (every pair, for stack.py)
"""
import gc
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from block import sound
from match import FEATS, accept, best_per_o, cand_chunks, featurize, s1_stats, truth_pairs, val_score, write_lists
from prep import ROOT, WORK

W = WORK / "stage2"
RATE = 0.15  # fraction of eligible training records sampled (as in match.py)
BATCH = 100_000  # rows per split text file
FOLD_PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
               "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}
FOLD_ROUNDS = 400
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}
NEW = ["p1", "p1_rank", "p1_gap", "s_other_p", "s_n_hi", "s_sum_other", "pp_core_tset", "pp_addr_tset",
       "pp_name_ratio", "pp_same_src"]


def derive(df):
    """Same derived text columns as match.records()."""
    return df.with_columns(
        snd=sound(pl.col("core")), sq=pl.col("core").str.replace_all(" ", ""),
        street=pl.col("addr").str.replace_all(r"\d+", " ").str.replace_all(" +", " ").str.strip_chars(),
    )


class Text:
    """Record text of a split without holding all of it: the split parquet is streamed once into 100k-row
    files; S1 records and every blocking partner stay in memory, a chunk's own records are read on demand."""

    def __init__(self, split, stats):
        import pyarrow.parquet as pq
        self.dir = W / "text" / split
        self.dir.mkdir(parents=True, exist_ok=True)
        start = 0
        cols = ["src", "name", "core", "addr", "nums"]
        for b, batch in enumerate(pq.ParquetFile(WORK / f"{split}.parquet").iter_batches(BATCH, columns=cols)):
            df = pl.from_arrow(batch).with_row_index("idx", offset=start)
            derive(df).write_parquet(self.dir / f"{b:05d}.parquet")
            start += df.height
        partners = pl.concat([stats["o1"], stats["o2"]]).drop_nulls().unique()
        self.base = pl.concat([
            pl.read_parquet(f).filter((pl.col("src") == 1) | pl.col("idx").is_in(partners.implode()))
            for f in sorted(self.dir.glob("*.parquet"))
        ])
        self.stats = stats

    def rows(self, lo, hi):
        files = [self.dir / f"{b:05d}.parquet" for b in range(lo // BATCH, hi // BATCH + 1)]
        return pl.concat([pl.read_parquet(f) for f in files]).filter(pl.col("idx").is_between(lo, hi))

    def for_chunk(self, c):
        """Text needed by featurize for candidate pairs c (one chunk's records, their S1s, those S1s' partners)."""
        part = self.stats.filter(pl.col("s").is_in(c["s"].unique().implode()))
        need = pl.concat([c["s"], part["o1"].drop_nulls(), part["o2"].drop_nulls()]).unique()
        own = self.rows(int(c["o"].min()), int(c["o"].max()))
        return pl.concat([self.base.filter(pl.col("idx").is_in(need.implode())), own]).unique("idx")

    def names(self):
        """src/name/core/addr of every record, for the p1-partner similarities."""
        return pl.concat([pl.read_parquet(f, columns=["idx", "src", "name", "core", "addr"])
                          for f in sorted(self.dir.glob("*.parquet"))])


def aggregates(p1_dir):
    """Per S1: the two records with the highest p1 (and their p1), how many exceed 0.5, and the p1 sum."""
    return (
        pl.scan_parquet(sorted(p1_dir.glob("*.parquet"))).sort("p1", descending=True).group_by("s")
        .agg(q1_o=pl.col("o").first(), q1_p=pl.col("p1").first(), q2_o=pl.col("o").slice(1, 1).first(),
             q2_p=pl.col("p1").slice(1, 1).first(), n_hi=(pl.col("p1") > 0.5).sum(), sum_p=pl.col("p1").sum())
        .collect()
    )


def second_pass_features(x, agg, names):
    """Add NEW to featurized pairs that carry p1 (every candidate of each o present). Returns numbers only."""
    x = (
        x.join(agg, on="s", how="left")
        .with_columns(
            p1_rank=pl.col("p1").rank("ordinal", descending=True).over("o").cast(pl.Float32),
            o_best=pl.col("p1").max().over("o"),
            o_2nd=pl.col("p1").top_k(2).min().over("o"),
            own=pl.col("q1_o") == pl.col("o"),
        )
        .with_columns(
            p1_gap=pl.when(pl.col("p1_rank") == 1).then(pl.col("p1") - pl.col("o_2nd"))
            .otherwise(pl.col("p1") - pl.col("o_best")),
            s_other_p=pl.when("own").then("q2_p").otherwise("q1_p"),
            partner=pl.when("own").then("q2_o").otherwise("q1_o"),
            s_n_hi=(pl.col("n_hi") - (pl.col("p1") > 0.5).cast(pl.UInt32)).cast(pl.Float32),
            s_sum_other=(pl.col("sum_p") - pl.col("p1")).cast(pl.Float32),
        )
        .join(names.rename(lambda c: c + "_a"), left_on="o", right_on="idx_a")
        .join(names.rename(lambda c: c + "_b"), left_on="partner", right_on="idx_b", how="left")
        .with_columns(pl.col("name_b", "core_b", "addr_b").fill_null(""))
    )
    sims = {"pp_core_tset": ("core", fuzz.token_set_ratio), "pp_addr_tset": ("addr", fuzz.token_set_ratio),
            "pp_name_ratio": ("name", fuzz.ratio)}
    x = x.with_columns(
        pl.Series(k, process.cpdist(x[c + "_a"].to_list(), x[c + "_b"].to_list(), scorer=sc, workers=-1),
                  dtype=pl.Float32)
        for k, (c, sc) in sims.items()
    ).with_columns(pp_same_src=(pl.col("src_a") == pl.col("src_b")).cast(pl.Float32))
    x = x.with_columns(pl.when(pl.col("partner").is_null()).then(None).otherwise(pl.col(c)).alias(c)
                       for c in ("pp_core_tset", "pp_addr_tset", "pp_name_ratio", "pp_same_src"))
    keep = ["y"] if "y" in x.columns else []
    return x.select("o", "s", *keep, *(pl.col(c).cast(pl.Float32) for c in FEATS + NEW))


def dirs(*names):
    for n in names:
        (W / n).mkdir(parents=True, exist_ok=True)
    return [W / n for n in names]


def train():
    s1feat, p1_dir, valfeat, s2train, s2val = dirs("s1feat", "p1_train", "valfeat", "s2train", "s2val")
    ids = pl.read_parquet(WORK / "train.parquet", columns=["entity_id", "src"]).with_row_index("idx")
    tp = truth_pairs(ids)
    lab = tp.rename({"s_true": "s"}).with_columns(y=pl.lit(1, pl.Int8))
    val_s = ids.filter((pl.col("src") == 1) & (pl.col("entity_id").str.slice(3).cast(pl.Int64) % 100 == 0))["idx"]
    del ids
    stats = s1_stats("train")
    text = Text("train", stats)
    print(f"text ready: {text.base.height:,} resident records", flush=True)

    # first-pass features for the sampled training records
    rng = np.random.default_rng(0)
    train_o, val_o = [], []
    for f in cand_chunks("train"):
        c = pl.read_parquet(f).with_columns(has_val=pl.col("s").is_in(val_s.implode()).any().over("o"))
        val_o.append(c.filter("has_val")["o"].unique())
        o = c.filter(~pl.col("has_val"))["o"].unique()
        keep = o.filter(pl.Series(rng.random(len(o)) < RATE))
        train_o.append(keep)
        ck = c.filter(pl.col("o").is_in(keep.implode())).drop("has_val")
        featurize(ck, text.for_chunk(ck), stats).join(lab, on=["o", "s"], how="left") \
            .with_columns(pl.col("y").fill_null(0)).write_parquet(s1feat / f.name)
    train_o, val_o = pl.concat(train_o), pl.concat(val_o)
    print("first-pass training features written", flush=True)

    models = []
    for k in (0, 1):
        d = pl.scan_parquet(sorted(s1feat.glob("*.parquet"))).filter(pl.col("o") % 2 == k).select(*FEATS, "y").collect()
        models.append(lgb.train(FOLD_PARAMS, lgb.Dataset(d.select(FEATS).to_numpy(), d["y"].to_numpy()), FOLD_ROUNDS))
        models[-1].save_model(str(W / f"fold_{k}.txt"))
        del d
        gc.collect()
    print("first-pass fold models trained", flush=True)

    # out-of-sample p1 for every training candidate; keep the validation-linked records' features
    for f in cand_chunks("train"):
        c = pl.read_parquet(f)
        x = featurize(c, text.for_chunk(c), stats)
        xs = x.select(FEATS).to_numpy()
        pa, pb = models[0].predict(xs), models[1].predict(xs)
        o = x["o"].to_numpy()
        in_tr = x["o"].is_in(train_o.implode()).to_numpy()
        p1 = np.where(in_tr & (o % 2 == 0), pb, np.where(in_tr & (o % 2 == 1), pa, (pa + pb) / 2))
        x = x.with_columns(p1=pl.Series(p1, dtype=pl.Float32))
        x.select("o", "s", "p1").write_parquet(p1_dir / f.name)
        x.filter(pl.col("o").is_in(val_o.implode())).write_parquet(valfeat / f.name)
    del x, xs
    names = text.names()
    del text
    gc.collect()
    agg = aggregates(p1_dir)
    print("p1 and S1 aggregates ready", flush=True)

    for f in cand_chunks("train"):
        t = pl.read_parquet(s1feat / f.name)
        if t.height:
            second_pass_features(t.join(pl.read_parquet(p1_dir / f.name), on=["o", "s"]), agg, names) \
                .write_parquet(s2train / f.name)
        v = pl.read_parquet(valfeat / f.name)
        if v.height:
            second_pass_features(v, agg, names).write_parquet(s2val / f.name)
    del names, agg
    gc.collect()
    print("second-pass features written", flush=True)

    cols = FEATS + NEW
    tr = pl.scan_parquet(sorted(s2train.glob("*.parquet"))).select(*cols, "y").collect()
    va = pl.scan_parquet(sorted(s2val.glob("*.parquet"))).collect()
    es_o = va["o"].unique()
    es_o = es_o.sample(min(200_000, len(es_o)), seed=0)
    es = va.filter(pl.col("o").is_in(es_o.implode()))
    es_y = es.join(tp.rename({"s_true": "s"}).with_columns(y=pl.lit(1)), on=["o", "s"], how="left")["y"].fill_null(0)
    train_set = lgb.Dataset(tr.select(cols).to_numpy(), tr["y"].to_numpy(), feature_name=cols)
    print(f"second-pass training on {tr.height:,} pairs, {tr['y'].mean():.3f} positive", flush=True)
    del tr
    model = lgb.train(PARAMS, train_set, num_boost_round=3000,
                      valid_sets=[lgb.Dataset(es.select(cols).to_numpy(), es_y.to_numpy(), reference=train_set)],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(100)])
    model.save_model(str(W / "model_stage2.txt"))
    imp = sorted(zip(cols, model.feature_importance("gain")), key=lambda t: -t[1])
    print(f"best iteration {model.best_iteration}; top features {[n for n, _ in imp[:12]]}", flush=True)

    truth = {s: set() for s in val_s}
    for o, s in tp.filter(pl.col("s_true").is_in(val_s.implode())).iter_rows():
        truth[s].add(o)
    scored = va.select("o", "s", "p1", p=pl.Series(model.predict(va.select(cols).to_numpy())))
    scored.write_parquet(W / "val_scored.parquet")
    for name, col in (("first pass (fold average)", "p1"), ("second pass", "p")):
        best = best_per_o(scored.select("o", "s", p=col))
        res = {(t, m): val_score(best, val_s, truth, t, m)
               for t in np.arange(0.40, 0.91, 0.05).round(2) for m in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)}
        thr, m = max(res, key=res.get)
        print(f"{name}: best thr {thr} margin {m} -> val macro F0.5 {res[thr, m]:.4f}", flush=True)
    (W / "thr_stage2.json").write_text(json.dumps({"thr": float(thr), "margin": m, "val_f05": res[thr, m]}))


def predict():
    (p1_dir,) = dirs("p1_test")
    models = [lgb.Booster(model_file=str(W / f"fold_{k}.txt")) for k in (0, 1)]
    model = lgb.Booster(model_file=str(W / "model_stage2.txt"))
    rule = json.loads((W / "thr_stage2.json").read_text())
    stats = s1_stats("test")
    text = Text("test", stats)
    feat_dir = W / "feat_test"
    feat_dir.mkdir(exist_ok=True)
    for f in cand_chunks("test"):
        c = pl.read_parquet(f)
        x = featurize(c, text.for_chunk(c), stats)
        xs = x.select(FEATS).to_numpy()
        x = x.with_columns(p1=pl.Series((models[0].predict(xs) + models[1].predict(xs)) / 2, dtype=pl.Float32))
        x.select("o", "s", "p1").write_parquet(p1_dir / f.name)
        x.write_parquet(feat_dir / f.name)
    names = text.names()
    del text
    gc.collect()
    agg = aggregates(p1_dir)
    cols = FEATS + NEW
    matched = []
    (W / "test_scored").mkdir(exist_ok=True)
    for f in cand_chunks("test"):
        x = second_pass_features(pl.read_parquet(feat_dir / f.name), agg, names)
        scored = x.select("o", "s", p=pl.Series(model.predict(x.select(cols).to_numpy())))
        scored.write_parquet(W / "test_scored" / f.name)
        matched.append(accept(best_per_o(scored), rule["thr"], rule["margin"]))
    matched = pl.concat(matched)
    matched.write_parquet(W / "test_matched.parquet")
    recs = pl.read_parquet(WORK / "test.parquet", columns=["entity_id", "src"]).with_row_index("idx")
    out = ROOT / "output"
    write_lists(matched.lazy().select("o", "s"), recs, "matched_entity_ids", out / "matching_results.tsv")
    write_lists(pl.scan_parquet(cand_chunks("test")).select("o", "s"), recs, "candidate_entity_ids",
                out / "candidate_pairs.tsv")
    print(f"{matched.height:,} matched records over {matched['s'].n_unique():,} S1 entities -> {out}")


if __name__ == "__main__":
    {"train": train, "predict": predict}[sys.argv[1]]()
