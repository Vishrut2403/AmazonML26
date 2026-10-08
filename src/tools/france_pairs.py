"""Build likely S1 <-> S2/S3 pairs for France from the test TSVs (France has no labels): work/france_pairs.parquet.

Two sets, each keeping only S1 keys that are unique within France:
  a_name_linked : same name key (name words minus legal forms and filler words), first house number agrees
  b_addr_linked : same address key (house number, street words, city; street types and regions canonicalized),
                  and the names share at least one word
Output columns: o_id (S2/S3 entity_id), s1_id (S1 entity_id), set.
Usage: python src/tools/france_pairs.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_tsv  # noqa: E402
from prep import DATA, WORK, clean, to_ascii  # noqa: E402

# legal forms, filler words and dotted-form letters that sources add or drop freely
NAME_NOISE = """sarl sas sasu eurl sa sci snc ei s a r l u e i c com groupe developpement international participations
distribution holding services associes et and france frnce de du des la le les d cie fils freres frs dba aka fka formerly
doing business as known nee trading enterprises labs sys co one 5arl 5as societe ste ets etablissements compagnie""".split()

# street types, number prefixes/suffixes and articles, mapped for the address key ("" = dropped)
ADDR_MAP = {
    "rue": "rue", "r": "rue", "avenue": "avenue", "av": "avenue", "ave": "avenue", "allee": "allee", "all": "allee",
    "boulevard": "boulevard", "bd": "boulevard", "blvd": "boulevard", "impasse": "impasse", "imp": "impasse",
    "route": "route", "rte": "route", "chemin": "chemin", "ch": "chemin", "place": "place", "pl": "place",
    "cours": "cours", "crs": "cours", "quai": "quai", "q": "quai", "st": "saint", "ste": "sainte",
    "no": "", "ndeg": "", "n": "", "b": "bis", "t": "ter",
    "de": "", "du": "", "des": "", "la": "", "le": "", "les": "", "d": "", "l": "",
}
# region and department words: present, swapped or dropped at random by S2/S3, so left out of the key
REGION_WORDS = ["hauts", "france", "nouvelle", "aquitaine", "pays", "loire", "nord", "gironde", "atlantique", "pas", "calais"]


def load_france():
    """France rows of the three test sources with cleaned name (n) and address (a)."""
    parts = []
    for s in (1, 2, 3):
        df = read_tsv(DATA / "test" / f"test_source{s}.tsv").filter(pl.col("country") == "France")
        parts.append(df.with_columns(src=pl.lit(s, pl.Int8), n=clean(to_ascii("business_name")),
                                     a=clean(to_ascii("business_address"))))
    return pl.concat(parts)


def name_key(col):
    """Sorted unique name words without legal forms and filler words."""
    return (pl.col(col).str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(NAME_NOISE) & (pl.element() != "")))
            .list.unique().list.sort().list.join(" "))


def house_number(col):
    """First house number, ignoring ndeg/no prefixes and leading zeros."""
    return pl.col(col).str.extract(r"(?:^|\s)(?:ndeg|no)?0*(\d+)", 1)


def addr_key(col):
    """Sorted unique address words with numbers split from prefixes/suffixes, street types mapped, regions dropped."""
    t = (pl.col(col).str.replace_all(r"\b(ndeg|no)(\d)", "$2").str.replace_all(r"\b0+(\d)", "$1")
         .str.replace_all(r"\b(\d+)(bis|ter|b|t)\b", "$1 $2").str.split(" "))
    t = t.list.eval(pl.element().replace(ADDR_MAP)).list.eval(
        pl.element().filter((pl.element() != "") & ~pl.element().is_in(REGION_WORDS)))
    return t.list.unique().list.sort().list.join(" ")


def unique_s1(s1, key):
    """S1 rows whose key occurs exactly once in S1."""
    return s1.join(s1.group_by(key).len().filter(pl.col("len") == 1).select(key), on=key)


def name_linked(fr):
    """Pairs sharing a unique S1 name key (at least 6 characters) and the first house number."""
    fr = fr.with_columns(nk=name_key("n"), hn=house_number("a"))
    s1 = unique_s1(fr.filter(pl.col("src") == 1), "nk").filter(pl.col("nk").str.len_chars() >= 6)
    p = fr.filter(pl.col("src") != 1).join(s1, on="nk", suffix="_1").filter(pl.col("hn") == pl.col("hn_1"))
    return p.select(o_id="entity_id", s1_id="entity_id_1", set=pl.lit("a_name_linked"))


def addr_linked(fr):
    """Pairs sharing a unique S1 address key (with a number, at least 12 characters) and at least one name word."""
    fr = fr.with_columns(ak=addr_key("a"))
    fr = fr.filter(pl.col("ak").str.contains(r"\d") & (pl.col("ak").str.len_chars() >= 12))
    s1 = unique_s1(fr.filter(pl.col("src") == 1), "ak")
    p = fr.filter(pl.col("src") != 1).join(s1, on="ak", suffix="_1")
    p = p.filter(pl.col("n").str.split(" ").list.set_intersection(pl.col("n_1").str.split(" ")).list.len() >= 1)
    return p.select(o_id="entity_id", s1_id="entity_id_1", set=pl.lit("b_addr_linked"))


def main():
    fr = load_france()
    out = pl.concat([name_linked(fr), addr_linked(fr)]).unique()
    WORK.mkdir(exist_ok=True)
    out.write_parquet(WORK / "france_pairs.parquet")
    print(out.group_by("set").len().sort("set"))


if __name__ == "__main__":
    main()
