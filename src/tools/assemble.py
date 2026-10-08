"""Assemble a matching file from a base file plus pair sets: each OVERRIDE set replaces the base pick of its records,
each ADD set only fills records the base (and earlier sets) leave unmatched.
Usage: python src/tools/assemble.py BASE_TSV OUT_TSV [override:PARQUET ...] [add:PARQUET ...]   (parquets hold o, s)
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_tsv  # noqa: E402
from match import write_lists  # noqa: E402
from prep import WORK  # noqa: E402


def base_pairs(tsv, recs):
    b = read_tsv(tsv).drop_nulls("matched_entity_ids").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    ids = recs.select(pl.col("idx").cast(pl.UInt32), "entity_id")
    return b.join(ids.rename({"entity_id": "source1_entity_id", "idx": "s"}), on="source1_entity_id") \
        .join(ids.rename({"entity_id": "matched_entity_ids", "idx": "o"}), on="matched_entity_ids").select("o", "s")


def main(base, out, *sets):
    recs = pl.read_parquet(WORK / "test.parquet", columns=["entity_id", "src"]).with_row_index("idx")
    pairs = base_pairs(base, recs)
    n0 = pairs.height
    for spec in sets:
        kind, path = spec.split(":", 1)
        p = pl.read_parquet(path).select(pl.col("o", "s").cast(pl.UInt32)).unique("o")
        if kind == "override":
            pairs = pl.concat([pairs.filter(~pl.col("o").is_in(p["o"].implode())), p])
        else:
            pairs = pl.concat([pairs, p.filter(~pl.col("o").is_in(pairs["o"].implode()))])
        print(f"{kind} {Path(path).name}: {pairs.height - n0:+,} pairs vs base", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_lists(pairs.lazy(), recs, "matched_entity_ids", out)
    print(f"{pairs.height:,} pairs -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
