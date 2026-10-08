"""Write output/candidate_pairs.tsv (every pair any final model or rule scored) and copy the final matches next to it.

Candidates = pairs the stack scored (work/merged/translit) + France stage-2 top 3 per record (work/fr_ce_scores.parquet)
+ initials/website pairs (work/initials_pairs_test.parquet). Fails if a final match is not among them.
Usage: python src/tools/candidates.py FINAL_MATCHING_TSV
"""
import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assemble import base_pairs  # noqa: E402
from match import write_lists  # noqa: E402
from prep import ROOT, WORK  # noqa: E402


def main(final):
    recs = pl.read_parquet(WORK / "test.parquet", columns=["entity_id", "src"]).with_row_index("idx")
    cand = pl.concat([pl.scan_parquet(WORK / "merged" / "translit" / "*.parquet").select("o", "s"),
                      pl.scan_parquet(WORK / "fr_ce_scores.parquet").select("o", "s"),
                      pl.scan_parquet(WORK / "initials_pairs_test.parquet").select("o", "s")]) \
        .select(pl.col("o", "s").cast(pl.UInt32)).unique().collect()
    missing = base_pairs(final, recs).join(cand, on=["o", "s"], how="anti").height
    assert missing == 0, f"{missing} final matches are not candidates"
    out = ROOT / "output"
    write_lists(cand.lazy(), recs, "candidate_entity_ids", out / "candidate_pairs.tsv")
    shutil.copyfile(final, out / "matching_results.tsv")
    print(f"{cand.height:,} candidate pairs -> {out / 'candidate_pairs.tsv'}; matches copied to {out / 'matching_results.tsv'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
