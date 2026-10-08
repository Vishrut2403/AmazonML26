"""Validation scores of CE_DIR merged like the test side (dense-search extras, Indian-script overrides), the input
of stack.py val. Usage: CE_DIR=ce_base3 python src/tools/val_merge.py OUT_PARQUET"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ce_data import OUT  # noqa: E402
from dense import DEN, SMALL, new_pairs, override, val_os  # noqa: E402
from match import cand_chunks  # noqa: E402
from prep import WORK  # noqa: E402

old = pl.read_parquet(OUT / "val_scored.parquet", columns=["o", "s", "ce"]).with_columns(pl.col("o", "s").cast(pl.UInt32))
first, vo = pl.read_parquet(SMALL / "val_scored.parquet", columns=["o", "ce"]), val_os()
rule = pl.concat(new_pairs(f, "train", vo, first) for f in cand_chunks("train"))
new = pl.read_parquet(DEN / "train_scored" / "*.parquet").join(rule, on=["o", "s"], how="semi")
sc = override(pl.concat([old, new]).unique(["o", "s"], keep="first"), pl.read_parquet(WORK / "translit" / "train" / "*.parquet"))
sc.write_parquet(sys.argv[1])
print(f"{sc.height:,} val pairs -> {sys.argv[1]}")
