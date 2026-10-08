"""Normalize all three sources of a split into one parquet: work/{split}.parquet.

Columns: entity_id, src (1/2/3), country, name, core, addr, nums
  name  - ascii, lowercase, alnum-only name
  core  - name without legal suffixes / filler words (inc, pvt, limited, the, ...)
  addr  - ascii, lowercase address with street/state abbreviations canonicalized
  nums  - space-joined digit runs from the address (house / unit / pin numbers)
Usage: python src/prep.py train|test
"""
import sys
from pathlib import Path

import polars as pl
from anyascii import anyascii

from common import read_tsv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "student_resource" / "dataset"
WORK = ROOT / "work"

# legal forms and filler words that sources add/drop freely; stripped for `core`
LEGAL = {
    "inc", "incorporated", "llc", "llp", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "praivet", "pte", "plc", "lp", "the", "and", "of", "sarl", "sas", "sa", "eurl", "sasu", "sci",
    "snc", "ei", "com", "www", "net", "org",
}
# France only: words that Source 2/3 add to French names (counts from test data, 10k-44k each) and that carry
# no identity; stripped from the French core name
FR_FILLER = {
    "developpement", "groupe", "participations", "holding", "distribution", "international", "fils", "associes",
    "services", "france", "compagnie",
}
FR_NAME = {"st": "saint", "ste": "sainte", "cb": "club", "ctre": "centre", "et": "and"}
# France only: Source 1 writes the region, Sources 2/3 the department or nothing. These are the 7 region and
# department names that occur as whole address parts in the data; such parts are dropped (a street like
# "rue du nord" is a different part and stays)
FR_REGIONS = ["hauts de france", "nouvelle aquitaine", "pays de la loire", "nord", "pas de calais", "gironde",
              "loire atlantique"]

# name spellings to unify before the legal words are stripped (dotted and OCR-garbled legal forms, short forms)
NAME_FIX = [
    (r"\bs a s u\b", "sasu"), (r"\bs a r l\b", "sarl"), (r"\be u r l\b", "eurl"), (r"\bs a s\b", "sas"),
    (r"\bs c i\b", "sci"), (r"\bs n c\b", "snc"), (r"\bs a\b", "sa"),
    (r"\b5arl\b", "sarl"), (r"\b5asu\b", "sasu"), (r"\b5as\b", "sas"),
    (r"\bcie\b", "compagnie"), (r"\bfrs\b", "freres"),
]
# "<invented brand> dba <real name>": everything up to and including the alias marker is dropped
ALIAS = r"^.*?\b(?:dba|d b a|doing business as|trading as|formerly known as|formerly|f k a|fka|a k a|aka|nee)\b\s*"

# word-level address canonicalization (token -> canonical token)
ADDR_ABBR = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane",
    "blvd": "boulevard", "bd": "boulevard", "ct": "court", "cir": "circle", "pl": "place", "hwy": "highway",
    "pkwy": "parkway", "trl": "trail", "ter": "terrace", "sq": "square", "mt": "mount", "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "apt": "unit", "apartment": "unit", "ste": "unit", "suite": "unit",
    "bldg": "building", "flr": "floor", "fl": "floor", "nagr": "nagar", "opp": "opposite", "nr": "near",
    "distt": "district", "dist": "district",
    # noise tokens carrying no location information
    "null": "", "no": "", "h": "", "door": "", "flat": "", "plot": "", "po": "", "box": "", "ndeg": "",
}
# France: st/ste are saint/sainte, n is the start of "n°", plus French street types
FR_ABBR = {
    **{k: v for k, v in ADDR_ABBR.items() if k not in ("n", "s", "e", "w", "ne", "nw", "se", "sw", "ter")},
    "st": "saint", "ste": "sainte", "n": "", "r": "rue", "ave": "avenue", "av": "avenue", "all": "allee",
    "bd": "boulevard", "blvd": "boulevard", "imp": "impasse", "rte": "route", "ch": "chemin", "pl": "place",
    "crs": "cours", "q": "quai", "res": "residence", "psg": "passage", "pass": "passage", "bld": "boulevard",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "tamil nadu": "tn", "tamilnadu": "tn", "west bengal": "wb", "uttar pradesh": "up", "madhya pradesh": "mp",
    "andhra pradesh": "ap", "himachal pradesh": "hp", "arunachal pradesh": "arp", "maharashtra": "mh",
    "karnataka": "ka", "kerala": "kl", "gujarat": "gj", "rajasthan": "rj", "haryana": "hr", "punjab": "pb",
    "telangana": "tg", "odisha": "od", "orissa": "od", "bihar": "br", "jharkhand": "jh", "assam": "as",
    "chhattisgarh": "cg", "uttarakhand": "uk", "goa": "ga", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
# multi-word names first so "west virginia" wins over "virginia"
STATES = sorted({**US_STATES, **IN_STATES}.items(), key=lambda kv: -len(kv[0]))


def to_ascii(col):
    """Transliterate non-ascii strings (accents, Devanagari, Bengali, ...) to ascii; ascii rows skip the Python call."""
    return (
        pl.when(pl.col(col).str.contains(r"[^\x00-\x7F]"))
        .then(pl.col(col).map_elements(anyascii, return_dtype=pl.String))
        .otherwise(pl.col(col))
    )


def clean(expr):
    """Lowercase, '&' -> 'and', every non-alphanumeric run -> single space, trimmed."""
    return (
        expr.fill_null("").str.to_lowercase().str.replace_all("&", " and ", literal=True)
        .str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
    )


def map_tokens(expr, mapping):
    """Replace whole words via a dict; words mapped to '' are dropped."""
    return (
        expr.str.split(" ").list.eval(pl.element().replace(mapping)).list.eval(pl.element().filter(pl.element() != ""))
        .list.join(" ")
    )


def fix_name(expr):
    """Unify legal-form spellings and drop '<brand> dba' alias prefixes (kept as is if nothing would remain)."""
    for a, b in NAME_FIX:
        expr = expr.str.replace_all(a, b)
    stripped = expr.str.replace(ALIAS, "")
    return pl.when(stripped.str.len_chars() > 0).then(stripped).otherwise(expr)


def drop_regions(expr):
    """Remove comma-separated address parts that are a French region or department name."""
    part = pl.element().str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
    return expr.fill_null("").str.split(",").list.eval(pl.element().filter(~part.is_in(FR_REGIONS))).list.join(",")


def normalize(df):
    """Add the normalized columns described in the module docstring to a raw source frame."""
    france = pl.col("country") == "France"
    raw_addr = pl.when(france).then(drop_regions(to_ascii("business_address"))).otherwise(to_ascii("business_address"))
    df = df.with_columns(name=fix_name(clean(to_ascii("business_name"))), addr=clean(raw_addr))
    # sources garble ordinals (24rd / 42th / 138st), pad numbers with zeros (005937) and write n° as ndeg5:
    # keep the bare number
    addr = (
        pl.col("addr").str.replace_all(r"\b(\d+)(?:st|nd|rd|th)\b", "$1").str.replace_all(r"\b0+(\d)", "$1")
        .str.replace_all(r"\bndeg(\d)", "$1")
    )
    # French house-number suffixes: 82bis / 82 b -> 82 bis, 459t / 459 ter -> 459 ter
    addr = pl.when(france).then(
        addr.str.replace_all(r"\b(\d+) ?(?:bis|b)\b", "$1 bis").str.replace_all(r"\b(\d+) ?(?:ter|t)\b", "$1 ter")
    ).otherwise(addr)
    df = df.with_columns(addr=addr)
    padded = (" " + pl.col("addr") + " ").str.replace_many(
        [f" {k} " for k, _ in STATES], [f" {v} " for _, v in STATES]
    )
    df = df.with_columns(name=pl.when(france).then(map_tokens(pl.col("name"), FR_NAME)).otherwise(pl.col("name")))
    return df.with_columns(
        core=pl.when(france).then(map_tokens(pl.col("name"), {w: "" for w in LEGAL | FR_FILLER}))
        .otherwise(map_tokens(pl.col("name"), {w: "" for w in LEGAL})),
        addr=pl.when(france).then(map_tokens(padded.str.strip_chars(), FR_ABBR))
        .otherwise(map_tokens(padded.str.strip_chars(), ADDR_ABBR)),
        nums=pl.col("addr").str.extract_all(r"\d+").list.join(" "),
    ).select("entity_id", "src", "country", "name", "core", "addr", "nums")


def main(split):
    WORK.mkdir(exist_ok=True)
    parts = [
        normalize(read_tsv(DATA / split / f"{split}_source{s}.tsv").with_columns(src=pl.lit(s, pl.Int8)))
        for s in (1, 2, 3)
    ]
    out = pl.concat(parts)
    out.write_parquet(WORK / f"{split}.parquet")
    print(split, out.height, "rows ->", WORK / f"{split}.parquet")


if __name__ == "__main__":
    main(sys.argv[1])
