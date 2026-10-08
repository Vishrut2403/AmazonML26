"""Final model on top of the pair scores: LightGBM over the cross-encoder score, the stage-2 LightGBM score and
"competition" features (how the pair ranks among the record's candidates, how many records pick the same S1).

Inputs per split: cross-encoder scores (o, s, ce) merged over blocking, dense search and translit.py, and stage-2
scores (o, s, p) from stage2.py, which exist only for blocking candidates (null otherwise).
  val VAL_CE VAL_S2    5-fold (by o) out-of-fold scores on the validation records -> tune threshold/margin and report
                       val F0.5; then fit on all validation records -> work/stack/model.txt, work/stack/thr.json
  test CE_DIR TEST_S2 [FR_S2 RULE...]
                       score the test pairs (CE_DIR: per-chunk merged scores from dense.py final). France has no
                       training labels and the cross-encoders transfer badly there (leaderboard: France ~0.88 with
                       the cross-encoder, ~0.94 with stage 2), so France records keep stage 2's decision from FR_S2
                       (default TEST_S2) under each RULE "thr/margin" -> output/matching_results_stack_fr<rule>.tsv
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from ce_predict import val_truth
from match import accept, best_per_o, cand_chunks, truth_pairs, val_score, write_lists
from common import read_tsv
from prep import DATA, ROOT, WORK

OUT = WORK / "stack"
# raw-text features: the sources edit the raw S1 text, so punctuation, legal-form spelling and house numbers that
# normalization removes still tell near-duplicate S1 records apart
RAW = ["r_name", "r_name_sort", "r_addr", "lf_raw_eq", "lf_eq", "lf_o_n", "lf_s_n", "num_eq", "num_in", "num_diff", "o_addr_empty"]
FEATS = ["ce", "s2", "has_s2", "ce_rank", "ce_gap", "ce_2nd", "n_o", "s2_rank", "s2_gap", "src",
         "s_n", "s_top", "s_ce_sum", "s_top_src", "s_top_other"] + RAW
LF = ["inc", "incorporated", "llc", "llp", "corp", "corporation", "co", "company", "ltd", "limited", "pvt", "private",
      "pte", "plc", "lp", "pc", "pllc", "sarl", "sas", "sa", "eurl", "sasu", "sci", "snc", "ei"]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.9, "verbose": -1}
ROUNDS = 400
# unmatched ("decoy") records are ~26% of validation but ~40% of test; weighting their pairs by the odds ratio
# trains the model for the test's mix (STACK_DECOY_W=1 turns it off)
DECOY_W = float(__import__("os").environ.get("STACK_DECOY_W", 1.0))
FR_THR, FR_MARGIN = 0.723, 0.3  # stage 2's strict rule, as uploaded (leaderboard 0.972)


def raw_table(split, keep=None):
    """Lowercased raw name/address per record (row index = split parquet index), its legal-form words and numbers."""
    raw = pl.concat(read_tsv(DATA / split / f"{split}_source{i}.tsv").select("business_name", "business_address")
                    for i in (1, 2, 3)).with_row_index("idx").with_columns(pl.col("idx").cast(pl.UInt32))
    if keep is not None:
        raw = raw.filter(pl.col("idx").is_in(keep.implode()))
    rn = pl.col("business_name").fill_null("").str.to_lowercase()
    lf_tok = rn.str.split(" ").list.eval(pl.element().filter(pl.element().str.replace_all(r"[^a-z]", "").is_in(LF)))
    return raw.select("idx", rn=rn, ra=pl.col("business_address").fill_null("").str.to_lowercase(),
                      lf_raw=lf_tok.list.join(" "), lf=lf_tok.list.eval(pl.element().str.replace_all(r"[^a-z]", "")).list.join(""),
                      nums=pl.col("business_address").fill_null("").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")))


def raw_level(sc, raw):
    """Adds RAW: raw name/address similarity, legal-form agreement, first house number agreement."""
    j = sc.select("o", "s").join(raw.rename(lambda c: c if c == "idx" else c + "_o"), left_on="o", right_on="idx") \
        .join(raw.rename(lambda c: c if c == "idx" else c + "_s"), left_on="s", right_on="idx")
    sim = lambda a, b, f: pl.Series(process.cpdist(j[a].to_list(), j[b].to_list(), scorer=f, workers=-1), dtype=pl.Float32)
    n_o, n_s = pl.col("nums_o").list.first(), pl.col("nums_s").list.first()
    f = j.select(
        "o", "s", r_name=sim("rn_o", "rn_s", fuzz.ratio), r_name_sort=sim("rn_o", "rn_s", fuzz.token_sort_ratio),
        r_addr=sim("ra_o", "ra_s", fuzz.token_set_ratio),
        lf_raw_eq=(pl.col("lf_raw_o") == pl.col("lf_raw_s")).cast(pl.Int8), lf_eq=(pl.col("lf_o") == pl.col("lf_s")).cast(pl.Int8),
        lf_o_n=pl.col("lf_o").str.len_chars(), lf_s_n=pl.col("lf_s").str.len_chars(),
        num_eq=(n_o == n_s).cast(pl.Int8),
        num_in=(pl.col("nums_o").list.set_intersection(pl.col("nums_s").list.head(1)).list.len() > 0).cast(pl.Int8),
        num_diff=(n_o.cast(pl.Int64, strict=False) - n_s.cast(pl.Int64, strict=False)).abs().log1p().cast(pl.Float32),
        o_addr_empty=(pl.col("ra_o") == "").cast(pl.Int8))
    return sc.join(f, on=["o", "s"], how="left")


def o_level(sc, src):
    """sc: o, s, ce, s2 (nullable); src: o -> 2 or 3. Per-record features (a record's candidates are all in sc)."""
    sc = sc.join(src, on="o", how="left").with_columns(
        has_s2=pl.col("s2").is_not_null().cast(pl.Int8),
        ce_rank=pl.col("ce").rank("ordinal", descending=True).over("o").cast(pl.Int16),
        n_o=pl.len().over("o").cast(pl.Int16),
        s2_rank=pl.col("s2").rank("ordinal", descending=True).over("o").cast(pl.Int16),
    )
    # best score among the record's other candidates
    top2 = sc.group_by("o").agg(c1=pl.col("ce").max(), c2=pl.col("ce").sort(descending=True).slice(1, 1).first().fill_null(0.0),
                                t1=pl.col("s2").max(), t2=pl.col("s2").sort(descending=True, nulls_last=True).slice(1, 1).first())
    return sc.join(top2, on="o").with_columns(
        ce_2nd=pl.when(pl.col("ce_rank") == 1).then(pl.col("c2")).otherwise(pl.col("c1")),
        s2_gap=pl.col("s2") - pl.when(pl.col("s2_rank") == 1).then(pl.col("t2")).otherwise(pl.col("t1")),
    ).with_columns(ce_gap=pl.col("ce") - pl.col("ce_2nd")).drop("c1", "c2", "t1", "t2")


def s_partial(sc):
    """Per (S1, source) counts over o-level rows; sums of these over any split of the records are exact."""
    sure = (pl.col("ce_rank") == 1) & (pl.col("ce") >= 0.5)
    return sc.group_by("s", "src").agg(n=pl.len(), top=sure.sum(), ce_sum=pl.col("ce").sum())


def s_level(part):
    """S1 competition features from summed s_partial rows: records scoring it, records whose confident best it is
    (all, same source as the row's record, other source)."""
    part = part.group_by("s", "src").agg(pl.col("n", "top", "ce_sum").sum())
    tot = part.group_by("s").agg(s_n=pl.col("n").sum(), s_top=pl.col("top").sum(), s_ce_sum=pl.col("ce_sum").sum())
    return part.select("s", "src", s_top_src="top").join(tot, on="s")


def add_s_level(sc, s_tab):
    return sc.join(s_tab, on=["s", "src"], how="left").with_columns(
        pl.col("s_top_src").fill_null(0), s_top_other=pl.col("s_top") - pl.col("s_top_src").fill_null(0))


def matrix(df):
    return df.select(pl.col(FEATS).cast(pl.Float32)).to_numpy()


def src_of(split):
    return pl.read_parquet(WORK / f"{split}.parquet", columns=["src"]).with_row_index("o").select(pl.col("o").cast(pl.UInt32), "src")


def val(ce_path, s2_path):
    val_s, truth = val_truth()
    ids = pl.read_parquet(WORK / "train.parquet", columns=["entity_id"]).with_row_index("idx")
    tp = truth_pairs(ids).select(pl.col("o", "s_true").cast(pl.UInt32))
    sc = pl.read_parquet(ce_path).select(pl.col("o", "s").cast(pl.UInt32), "ce").join(
        pl.read_parquet(s2_path).select(pl.col("o", "s").cast(pl.UInt32), s2="p"), on=["o", "s"], how="left")
    sc = o_level(sc, src_of("train"))
    sc = add_s_level(sc, s_level(s_partial(sc)))
    sc = raw_level(sc, raw_table("train", pl.concat([sc["o"], sc["s"]]).unique()))
    sc = sc.join(tp.rename({"s_true": "s"}).with_columns(y=pl.lit(1, pl.Int8)), on=["o", "s"], how="left") \
        .with_columns(pl.col("y").fill_null(0), fold=(pl.col("o").hash(7) % 5).cast(pl.Int8))
    x, y, fold = matrix(sc), sc["y"].to_numpy(), sc["fold"].to_numpy()
    decoy = ~sc["o"].is_in(tp["o"].implode()).to_numpy()
    w = np.where(decoy, DECOY_W, 1.0)
    oof = np.zeros(len(y))
    for k in range(5):
        m = lgb.train(PARAMS, lgb.Dataset(x[fold != k], y[fold != k], weight=w[fold != k]), ROUNDS)
        oof[fold == k] = m.predict(x[fold == k])
    OUT.mkdir(exist_ok=True)
    best = best_per_o(sc.select("o", "s", p=pl.Series(oof)))
    res = {(t, mg): val_score(best, val_s, truth, t, mg) for t in np.arange(0.3, 0.96, 0.05).round(2) for mg in (0.0, 0.1, 0.2, 0.3, 0.5)}
    t, mg = max(res, key=res.get)
    ce_best = best_per_o(sc.select("o", "s", p="ce"))
    print(f"cross-encoder alone: {max(val_score(ce_best, val_s, truth, a, 0.3) for a in (0.3, 0.45, 0.6)):.4f}; "
          f"stack (out-of-fold): best thr {t} margin {mg} -> {res[t, mg]:.4f}; "
          f"at thr {round(min(t + 0.1, 0.95), 2)}: {res.get((round(min(t + 0.1, 0.95), 2), mg), float('nan')):.4f}", flush=True)
    m = lgb.train(PARAMS, lgb.Dataset(x, y, weight=w, feature_name=FEATS), ROUNDS)
    m.save_model(str(OUT / "model.txt"))
    imp = sorted(zip(FEATS, m.feature_importance("gain")), key=lambda z: -z[1])
    print("top features:", [n for n, _ in imp[:8]])
    (OUT / "thr.json").write_text(json.dumps({"thr": float(t), "margin": mg, "val_f05": res[t, mg]}))


def test(ce_dir, s2_path, fr_path=None, *fr_rules):
    """fr_path: France stage-2 scores (default s2_path); fr_rules: "thr/margin" strings (default FR_THR/FR_MARGIN).
    Writes output/matching_results_stack_fr<thr>_<margin>.tsv per France rule."""
    ce_dir = Path(ce_dir)
    recs = pl.read_parquet(WORK / "test.parquet", columns=["entity_id", "src", "country"]).with_row_index("idx")
    france = recs.filter(pl.col("country") == "France")["idx"].cast(pl.UInt32)
    src = src_of("test")
    s2 = pl.scan_parquet(s2_path).select(pl.col("o", "s").cast(pl.UInt32), s2="p")
    model = lgb.Booster(model_file=str(OUT / "model.txt"))
    rule = json.loads((OUT / "thr.json").read_text())
    tmp = OUT / f"test_o_level_{ce_dir.name}"
    tmp.mkdir(exist_ok=True)
    raw = raw_table("test")
    # pass 1: record-level features per chunk (India/US records only), and the S1 competition counts
    parts = []
    for f in cand_chunks("test"):
        g = tmp / f.name
        if not g.exists():
            sc = pl.read_parquet(ce_dir / f.name).select(pl.col("o", "s").cast(pl.UInt32), "ce") \
                .filter(~pl.col("o").is_in(france.implode()))
            sc = sc.join(s2.filter(pl.col("o").is_in(sc["o"].unique().implode())).collect(), on=["o", "s"], how="left")
            raw_level(o_level(sc, src), raw).write_parquet(g)
        parts.append(s_partial(pl.read_parquet(g)))
    s_tab = s_level(pl.concat(parts))
    # pass 2: score, pick the best candidate per record
    best = []
    for f in cand_chunks("test"):
        sc = add_s_level(pl.read_parquet(tmp / f.name), s_tab)
        best.append(best_per_o(sc.select("o", "s", p=pl.Series(model.predict(matrix(sc))))))
    iu = accept(pl.concat(best), rule["thr"], rule["margin"]).select("o", "s")
    fr_best = best_per_o(pl.scan_parquet(fr_path or s2_path).select(pl.col("o", "s").cast(pl.UInt32), "p")
                         .filter(pl.col("o").is_in(france.implode())).collect())
    out = ROOT / "output"
    for fr_rule in fr_rules or [f"{FR_THR}/{FR_MARGIN}"]:
        t, m = (float(x) for x in fr_rule.split("/"))
        fr = accept(fr_best, t, m).select("o", "s")
        name = f"matching_results_stack_fr{t}_{m}.tsv"
        write_lists(pl.concat([iu, fr]).lazy(), recs, "matched_entity_ids", out / name)
        print(f"{name}: India/US {iu.height:,} matches (thr {rule['thr']}, margin {rule['margin']}); "
              f"France {fr.height:,} (thr {t}, margin {m}); {(out / name).stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    {"val": val, "test": test}[sys.argv[1]](*sys.argv[2:])
