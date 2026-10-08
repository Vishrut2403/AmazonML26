"""Cross-encoder decision rule and test predictions.

  tune:    pick threshold/margin for the cross-encoder scores on validation -> work/ce/thr.json
  predict: score each test record's top candidates (resumable, one file per candidate chunk), apply the rule and
           write output/matching_results.tsv and output/candidate_pairs.tsv (the pairs the model scored).
           Also writes matching_results_strict.tsv at threshold STRICT.
Usage: python src/ce_predict.py tune | predict [TOP]
"""
import json
import sys

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ce_data import text
from ce_train import OUT, score
from match import accept, best_per_o, cand_chunks, truth_pairs, val_score, write_lists
from prep import ROOT, WORK

STRICT = 0.60  # stricter threshold for the second file


def val_truth():
    ids = pl.read_parquet(WORK / "train.parquet", columns=["entity_id", "src"]).with_row_index("idx")
    tp = truth_pairs(ids)
    val_s = ids.filter((pl.col("src") == 1) & (pl.col("entity_id").str.slice(3).cast(pl.Int64) % 100 == 0))["idx"]
    truth = {s: set() for s in val_s}
    for o, s in tp.filter(pl.col("s_true").is_in(val_s.implode())).iter_rows():
        truth[s].add(o)
    return val_s, truth


def tune():
    val_s, truth = val_truth()
    best = best_per_o(pl.read_parquet(OUT / "val_scored.parquet").select("o", "s", p="ce"))
    res = {(t, m): val_score(best, val_s, truth, t, m)
           for t in np.arange(0.30, 0.96, 0.05).round(2) for m in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)}
    for (t, m), f in sorted(res.items()):
        if m in (0.0, 0.1, 0.3):
            print(f"thr {t:.2f} margin {m:.2f}  val macro F0.5 {f:.4f}")
    t, m = max(res, key=res.get)
    (OUT / "thr.json").write_text(json.dumps({"thr": float(t), "margin": m, "val_f05": res[t, m]}))
    print(f"best thr {t} margin {m} -> val macro F0.5 {res[t, m]:.4f}")


def predict(top):
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(OUT / "model")
    model = AutoModelForSequenceClassification.from_pretrained(OUT / "model").to(dev)
    if dev == "cuda":
        model = model.half()
    recs = pl.read_parquet(WORK / "test.parquet", columns=["entity_id", "src", "name", "addr"]).with_row_index("idx")
    txt = text(recs)
    sc_dir = OUT / "test_scored"
    sc_dir.mkdir(exist_ok=True)
    for f in cand_chunks("test"):
        if (sc_dir / f.name).exists():
            continue  # already scored (resumable)
        c = pl.read_parquet(f, columns=["o", "s", "rank"]).filter(pl.col("rank") <= top)
        c = c.join(txt.rename({"t": "a"}), left_on="o", right_on="idx").join(txt.rename({"t": "b"}), left_on="s", right_on="idx")
        ce = score(model, tok, c["a"].to_list(), c["b"].to_list(), dev)
        c.select("o", "s", ce=pl.Series(ce)).write_parquet(sc_dir / f.name)
        print(f.name, c.height, flush=True)
    scored = pl.read_parquet(sorted(sc_dir.glob("*.parquet")))
    rule = json.loads((OUT / "thr.json").read_text())
    best = best_per_o(scored.select("o", "s", p="ce"))
    out = ROOT / "output"
    # the tuned rule, plus a stricter copy: the leaderboard rewarded a higher threshold than validation did
    for name, thr in (("matching_results.tsv", rule["thr"]), ("matching_results_strict.tsv", STRICT)):
        write_lists(accept(best, thr, rule["margin"]).lazy().select("o", "s"), recs, "matched_entity_ids", out / name)
        print(f"{name}: thr {thr}, margin {rule['margin']}, {(out / name).stat().st_size:,} bytes", flush=True)
    write_lists(scored.lazy().select("o", "s"), recs, "candidate_entity_ids", out / "candidate_pairs.tsv")


def mix(lgb, ce, w):
    """Weighted average of the LightGBM and cross-encoder scores; candidates the cross-encoder did not score
    (below its top-k) count as 0 for it."""
    return lgb.join(ce.select("o", "s", "ce"), on=["o", "s"], how="left").select(
        "o", "s", p=w * pl.col("ce").fill_null(0.0) + (1 - w) * pl.col("p"))


def blend():
    """Tune weight/threshold/margin of the LightGBM + cross-encoder average on validation, then write
    output/matching_results_blend.tsv and a stricter copy (threshold +0.15)."""
    val_s, truth = val_truth()
    lgb, ce = pl.read_parquet(WORK / "val_scored.parquet"), pl.read_parquet(OUT / "val_scored.parquet")
    res = {}
    for w in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        best = best_per_o(mix(lgb, ce, w))
        for t in np.arange(0.30, 0.91, 0.05).round(2):
            for m in (0.0, 0.1, 0.2, 0.3):
                res[w, t, m] = val_score(best, val_s, truth, t, m)
        print(f"w {w}: best val {max(v for k, v in res.items() if k[0] == w):.4f}", flush=True)
    w, t, m = max(res, key=res.get)
    (OUT / "blend.json").write_text(json.dumps({"w": w, "thr": float(t), "margin": m, "val_f05": res[w, t, m]}))
    print(f"best w {w} thr {t} margin {m} -> val macro F0.5 {res[w, t, m]:.4f}", flush=True)
    del lgb, ce
    recs = pl.read_parquet(WORK / "test.parquet", columns=["entity_id", "src"]).with_row_index("idx")
    best = pl.concat(  # chunk by chunk: every candidate of an o is in the same chunk
        best_per_o(mix(pl.read_parquet(f), pl.read_parquet(OUT / "test_scored" / f.name), w))
        for f in sorted((WORK / "test_scored").glob("*.parquet")))
    out = ROOT / "output"
    for name, thr in (("matching_results_blend.tsv", t), ("matching_results_blend_strict.tsv", min(t + 0.15, 0.95))):
        write_lists(accept(best, thr, m).lazy().select("o", "s"), recs, "matched_entity_ids", out / name)
        print(f"{name}: thr {thr}, margin {m}, {(out / name).stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    predict(int(sys.argv[2]) if len(sys.argv) > 2 else 3) if cmd == "predict" else blend() if cmd == "blend" else tune()
