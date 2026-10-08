"""Pairs for the cross-encoder: each record's top candidates (by blocking rank) with their text and label.

Training pairs come from a sample of training records that are not linked to a validation entity; validation
pairs are all top candidates of validation-linked records (same validation entities as match.py).
Output: work/ce/{train,val}.parquet with o, s, rank, a (S2/S3 text), b (S1 text), y.
Usage: python src/ce_data.py [TOP] [RATE]
"""
import os
import sys

import numpy as np
import polars as pl

from match import cand_chunks, truth_pairs
from prep import WORK

OUT = WORK / os.environ.get("CE_DIR", "ce")  # a second model can use its own folder


def text(df):
    """One string per record: normalized name and address, the way the model sees it."""
    return df.select("idx", t=pl.lit("name: ") + pl.col("name") + pl.lit(" | address: ") + pl.col("addr"))


def main(TOP=3, RATE=0.10):
    recs = pl.read_parquet(WORK / "train.parquet", columns=["entity_id", "src", "name", "addr"]).with_row_index("idx")
    tp = truth_pairs(recs.select("idx", "entity_id"))
    lab = tp.rename({"s_true": "s"}).with_columns(y=pl.lit(1, pl.Int8))
    val_s = recs.filter((pl.col("src") == 1) & (pl.col("entity_id").str.slice(3).cast(pl.Int64) % 100 == 0))["idx"]
    txt = text(recs)
    del recs
    rng = np.random.default_rng(0)
    tr, va = [], []
    for f in cand_chunks("train"):
        c = pl.read_parquet(f, columns=["o", "s", "rank"]).with_columns(
            has_val=pl.col("s").is_in(val_s.implode()).any().over("o"))
        va.append(c.filter("has_val" & (pl.col("rank") <= TOP)).drop("has_val"))
        o = c.filter(~pl.col("has_val"))["o"].unique()
        keep = o.filter(pl.Series(rng.random(len(o)) < RATE))
        tr.append(c.filter(pl.col("o").is_in(keep.implode()) & (pl.col("rank") <= TOP)).drop("has_val"))
    OUT.mkdir(exist_ok=True)
    for name, parts in (("train", tr), ("val", va)):
        d = (
            pl.concat(parts).join(lab, on=["o", "s"], how="left").with_columns(pl.col("y").fill_null(0))
            .join(txt.rename({"t": "a"}), left_on="o", right_on="idx")
            .join(txt.rename({"t": "b"}), left_on="s", right_on="idx")
        )
        d.write_parquet(OUT / f"{name}.parquet")
        print(f"{name}: {d.height:,} pairs, {d['o'].n_unique():,} records, {d['y'].mean():.3f} positive")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3, float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
