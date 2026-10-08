"""Rescore only the uncertain records with a second (bigger) cross-encoder in CE_DIR, keeping the first
model's scores (work/ce) for the rest. A record is uncertain unless the first model is sure it matches its best
candidate (p >= SURE, runner-up <= 1 - SURE) or sure it matches none (p <= 1 - SURE); about 8% of records.
Writes CE_DIR/val_scored.parquet and CE_DIR/test_scored/*.parquet in the same format as the first model,
so ce_predict.py tune / predict / blend work on them unchanged.
Usage: CE_DIR=ce_base python src/ce_rescore.py val|test
"""
import sys

import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ce_data import OUT, text
from ce_train import score
from match import best_per_o, cand_chunks
from prep import WORK

FIRST = WORK / "ce"
SURE = 0.99


def uncertain(sc):
    b = best_per_o(sc.select("o", "s", p="ce"))
    sure = ((pl.col("p") >= SURE) & (pl.col("p2") <= 1 - SURE)) | (pl.col("p") <= 1 - SURE)
    return b.filter(~sure)["o"]


def rescore(sc, model, tok, dev):
    """sc: first-model scores with text columns a, b; returns o, s, ce with uncertain records rescored."""
    u = sc.filter(pl.col("o").is_in(uncertain(sc).implode()))
    new = score(model, tok, u["a"].to_list(), u["b"].to_list(), dev)
    u = u.select("o", "s", ce2=pl.Series(new))
    return sc.join(u, on=["o", "s"], how="left").select("o", "s", ce=pl.coalesce("ce2", "ce")), u.height


def main(split):
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(OUT / "model")
    model = AutoModelForSequenceClassification.from_pretrained(OUT / "model").to(dev)
    if dev == "cuda":
        model = model.half()
    if split == "val":
        va = pl.read_parquet(FIRST / "val.parquet").join(
            pl.read_parquet(FIRST / "val_scored.parquet").select("o", "s", "ce"), on=["o", "s"])
        out, n = rescore(va, model, tok, dev)
        va.select("o", "s", "rank", "y").join(out, on=["o", "s"]).write_parquet(OUT / "val_scored.parquet")
        print(f"val: rescored {n:,} of {va.height:,} pairs", flush=True)
        return
    recs = pl.read_parquet(WORK / "test.parquet", columns=["name", "addr"]).with_row_index("idx")
    txt = text(recs)
    sc_dir = OUT / "test_scored"
    sc_dir.mkdir(exist_ok=True)
    for f in cand_chunks("test"):
        if (sc_dir / f.name).exists():
            continue
        c = pl.read_parquet(FIRST / "test_scored" / f.name)
        c = c.join(txt.rename({"t": "a"}), left_on="o", right_on="idx").join(txt.rename({"t": "b"}), left_on="s", right_on="idx")
        out, n = rescore(c, model, tok, dev)
        out.write_parquet(sc_dir / f.name)
        print(f.name, n, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
