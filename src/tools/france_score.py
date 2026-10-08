"""Score France candidate pairs with the France cross-encoder (work/ce_fr/model, trained on tools/france_ce.py data)
and write France-rule variants of a matching file.

  score FR_S2          top-3 pairs per France record from stage 2 scores FR_S2 (o, s, p), scored with the France
                       cross-encoder -> work/fr_ce_scores.parquet
  variants IN_TSV OUT_PREFIX S2_THR S2_MARGIN
                       France rows of IN_TSV replaced by: frce (France cross-encoder alone, 0.6/0.3) and agree
                       (stage 2 at S2_THR/S2_MARGIN, kept only if the France cross-encoder gives that pair >= 0.5)
"""
import os
import sys
from pathlib import Path

import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ce_data import text  # noqa: E402
from ce_train import score  # noqa: E402
from match import accept, best_per_o, write_lists  # noqa: E402
from prep import WORK  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
FR_CE = os.environ.get("FR_CE", "ce_fr")  # France cross-encoder folder under work/
SCORES = WORK / ("fr_ce_scores.parquet" if FR_CE == "ce_fr" else f"fr_ce_scores_{FR_CE}.parquet")


def france_o(recs):
    return recs.filter((pl.col("country") == "France") & (pl.col("src") > 1))["idx"].cast(pl.UInt32)


def run_score(fr_s2):
    recs = pl.read_parquet(WORK / "test.parquet", columns=["src", "country", "name", "addr"]).with_row_index("idx")
    s2 = pl.read_parquet(fr_s2).select(pl.col("o", "s").cast(pl.UInt32), s2="p").filter(pl.col("o").is_in(france_o(recs).implode()))
    p = s2.filter(pl.col("s2").rank("ordinal", descending=True).over("o") <= 3)
    txt = text(recs)["t"]
    tok = AutoTokenizer.from_pretrained(WORK / FR_CE / "model")
    m = AutoModelForSequenceClassification.from_pretrained(WORK / FR_CE / "model").to(DEV)
    if DEV == "cuda":
        m = m.half()
    fce = score(m, tok, txt.gather(p["o"]).to_list(), txt.gather(p["s"]).to_list(), DEV)
    p.with_columns(fce=pl.Series(fce)).write_parquet(SCORES)
    print(f"{p.height:,} France pairs scored -> {SCORES}")


def replace_france(base_tsv, acc, recs, out):
    tmp = str(out) + ".fr.tsv"
    write_lists(acc.lazy().select("o", "s"), recs, "matched_entity_ids", tmp)
    rd = lambda f: pl.read_csv(f, separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.String}).with_columns(pl.col("matched_entity_ids").fill_null(""))
    fr, base = rd(tmp), rd(base_tsv)
    fr_ids = recs.filter((pl.col("country") == "France") & (pl.col("src") == 1))["entity_id"]
    base.join(fr.rename({"matched_entity_ids": "fr"}), on="source1_entity_id", how="left").with_columns(
        matched_entity_ids=pl.when(pl.col("source1_entity_id").is_in(fr_ids.implode())).then(pl.col("fr")).otherwise(pl.col("matched_entity_ids"))
    ).select(base.columns).write_csv(out, separator="\t", quote_style="never")
    Path(tmp).unlink()


def variants(base_tsv, out_prefix, s2_thr, s2_margin):
    recs = pl.read_parquet(WORK / "test.parquet", columns=["entity_id", "src", "country"]).with_row_index("idx")
    sc = pl.read_parquet(SCORES)
    frce = accept(best_per_o(sc.select("o", "s", p="fce")), 0.6, 0.3)
    s2 = accept(best_per_o(sc.select("o", "s", p="s2")), float(s2_thr), float(s2_margin))
    agree = s2.join(sc.select("o", "s", "fce"), on=["o", "s"]).filter(pl.col("fce") >= 0.5)
    for name, acc in (("frce", frce), ("agree", agree)):
        out = Path(f"{out_prefix}_{name}") / "upload" / "matching_results.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        replace_france(base_tsv, acc, recs, out)
        print(f"{name}: {acc.height:,} France matches (stage 2 alone: {s2.height:,}) -> {out}", flush=True)


if __name__ == "__main__":
    run_score(sys.argv[2]) if sys.argv[1] == "score" else variants(*sys.argv[2:6])
