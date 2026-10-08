"""Records whose name is only initials ("VJ" for "Vieilles Jeunes SAS") or a website/handle form
("100francecentre.com", "#globalpediatrics"): the sources write true matches this way (training data: 99.1-99.9% of
initials-only records are true matches), but no name word survives, so blocking misses them. They keep the address,
so the match is the S1 record at the same address (house number, street, city; French regions ignored) whose name
initials (or squashed name) fit, when exactly one S1 there fits.
  pairs(split) -> (o, s) for such records
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_tsv  # noqa: E402
from prep import DATA, FR_REGIONS, LEGAL, WORK  # noqa: E402

INITIALS = r"^[A-Z]{2,4}$"
WEB = r"(?i)^[a-z0-9]+\.(com|fr|net|org|in|co)$|^[#@][A-Za-z0-9]+$"


def addr_key(expr):
    for reg in sorted(FR_REGIONS, key=len, reverse=True):
        expr = expr.str.replace_all(rf"\b{reg}\b", " ")
    return expr.str.split(" ").list.sort().list.join(" ").str.replace_all(" +", " ").str.strip_chars()


def pairs(split):
    r = pl.read_parquet(WORK / f"{split}.parquet", columns=["entity_id", "src", "country", "name", "core", "addr"]).with_row_index("i") \
        .with_columns(pl.col("i").cast(pl.UInt32), key=addr_key(pl.col("addr")))
    raw = pl.concat(read_tsv(DATA / split / f"{split}_source{k}.tsv").select("entity_id", "business_name") for k in (2, 3))
    o = r.filter(pl.col("src") > 1).join(raw, on="entity_id").filter(
        (pl.col("business_name").str.contains(INITIALS) | pl.col("business_name").str.contains(WEB)) & (pl.col("key") != ""))
    o = o.with_columns(ini=pl.when(pl.col("business_name").str.contains(INITIALS)).then(pl.col("business_name").str.to_lowercase()),
                       sq=pl.col("business_name").str.to_lowercase().str.replace_all(r"^[#@]|\.(com|fr|net|org|in|co)$", "").str.replace_all(r"[^a-z0-9]", ""))
    s1 = r.filter(pl.col("src") == 1).with_columns(
        ini_s=pl.col("name").str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL))).str.slice(0, 1)).list.join(""),
        ini_c=pl.col("core").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join(""),
        sq_s=pl.col("name").str.replace_all(" ", ""), sq_c=pl.col("core").str.replace_all(" ", ""))
    j = o.select("i", "country", "key", "ini", "sq").join(s1.select(s="i", country="country", key="key", ini_s="ini_s", ini_c="ini_c", sq_s="sq_s", sq_c="sq_c"),
                                                            on=["country", "key"])
    fit = pl.when(pl.col("ini").is_not_null()).then(pl.col("ini_s").str.starts_with(pl.col("ini")) | pl.col("ini_c").str.starts_with(pl.col("ini"))
                                                   | (pl.col("ini_s") == pl.col("ini")) | (pl.col("ini_c") == pl.col("ini"))) \
        .otherwise((pl.col("sq") == pl.col("sq_s")) | (pl.col("sq") == pl.col("sq_c")) | pl.col("sq").str.contains(pl.col("sq_c"), literal=True))
    j = j.filter(fit).with_columns(n=pl.len().over("i")).filter(pl.col("n") == 1)
    return j.select(o="i", s="s")


if __name__ == "__main__":
    p = pairs(sys.argv[1])
    p.write_parquet(WORK / f"initials_pairs_{sys.argv[1]}.parquet")
    print(f"{p.height:,} pairs -> work/initials_pairs_{sys.argv[1]}.parquet")
