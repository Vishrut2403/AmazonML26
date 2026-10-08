"""Candidate generation by rare-token blocking: work/{split}.parquet -> work/{split}_cand/*.parquet.

Every S2/S3 record ("other", o) is matched to at most one S1 record, and matches never cross country,
so blocking runs from each o towards S1 within its country:
  1. tokens of a record (see `tokens`): name words, address words, number+street pairs, adjacent word pairs,
     sound keys of name words, the first 6 letters of the name with spaces removed and the consonant
     skeleton of that squashed name
  2. keep o's M rarest tokens whose S1 document frequency (per country) is <= CAP
  3. S1 records sharing any kept token are scored by summed IDF (tok); the top POOL by tok are re-ranked by
     name + address similarity (rs = token set ratio of core names + of addresses, tok breaks ties) and the
     top K become the candidates. The re-rank fixes the main blocking miss: an unrelated S1 record that
     shares one very rare token outscoring the true one.
Tokens are stored as 64-bit hashes of "country|token", which keeps memory low and makes the joins fast.
Output: one parquet per chunk in work/{split}_cand/, columns o (row index into the split parquet),
s (row index of the S1 record), tok (IDF score), rs (re-rank similarity), rank (1 = best by rs).
Usage: python src/block.py train|test [K]
"""
import string
import sys

import polars as pl
from rapidfuzz import fuzz, process

from prep import WORK

M = 8        # rarest tokens used per o record
CAP = 500    # tokens shared by more S1 records than this are too common to block on
CHUNK = 100_000  # o records per chunk; keeps peak memory around 3 GB
POOL = 100   # candidates per o re-ranked by text similarity before keeping the top K

LOW = list(string.ascii_lowercase)
DBL = [c * 2 for c in LOW]
# spelling -> sound rules, applied in order, so English and transliterated Hindi spellings meet
SOUND = [
    ("tion", "shn"), ("x", "ks"), ("ph", "f"), ("gh", ""), ("ck", "k"), ("q", "k"),
    ("c([eiy])", "s$1"), ("c", "k"), ("g([eiy])", "j$1"), ("sh", "s"), ("w", "v"), ("z", "j"),
    ("m([^aeiouyhbpm ])", "n$1"), ("h", ""), ("[aeiouy]", ""),
]


def sound(expr):
    """Consonant skeleton of every word: 'indian enterprises' and 'imdiyn emtrpraiss' -> 'ndn ntrpr'."""
    for a, b in SOUND:
        expr = expr.str.replace_all(a, b)
    expr = expr.str.replace_many(DBL, LOW).str.replace_many(DBL, LOW)  # twice: collapses up to 4 repeats
    return expr.str.replace_all(r"s( |$)", "$1")  # plural s


def words(expr, min_len=2):
    return expr.str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= min_len))


def bigrams(expr):
    """All adjacent word pairs 'a_b': two non-overlapping passes, the second shifted by one word."""
    pair = r"[a-z0-9]+ [a-z0-9]+"
    return pl.concat_list(
        expr.str.extract_all(pair), expr.str.replace(r"^[a-z0-9]+ ", "").str.extract_all(pair)
    ).list.eval(pl.element().str.replace(" ", "_"))


def tokens(df):
    """(idx, key) rows with key = hash("country|token"). Token kinds are prefixed so they never collide."""
    core, addr = pl.col("core"), pl.col("addr")
    squashed = core.str.replace_all(" ", "")
    toks = pl.concat_list(
        words(core),
        words(addr),
        addr.str.extract_all(r"\d+ [a-z]+").list.eval(pl.element().str.replace(" ", "_")),
        bigrams(core).list.eval("n:" + pl.element()),
        bigrams(addr).list.eval("a:" + pl.element()),
        words(sound(core), 3).list.eval("~" + pl.element()),
        pl.when(squashed.str.len_chars() >= 6).then(pl.concat_list("^" + squashed.str.slice(0, 6)))
        .otherwise(pl.lit([], dtype=pl.List(pl.String))),
        # consonant skeleton of the squashed name's start: website-style names often drop accented letters
        # ("collgejean com" vs "college jean"), which are mostly vowels
        pl.when(squashed.str.len_chars() >= 6).then(pl.concat_list("~^" + sound(squashed).str.slice(0, 5)))
        .otherwise(pl.lit([], dtype=pl.List(pl.String))),
    ).list.unique()
    return (
        df.select("idx", "country", tok=toks).explode("tok").drop_nulls("tok")
        .select("idx", key=(pl.col("country") + "|" + pl.col("tok")).hash())
    )


def index(df):
    """Blocking index over the S1 rows of df (idx, src, country, core, addr): blockable S1 tokens, their idf, S1 text."""
    s1 = df.filter(pl.col("src") == 1)
    s1_tok = tokens(s1)
    # country is inside the key, so the idf denominator is the whole S1 size; only relative idf matters
    dfreq = (
        s1_tok.group_by("key").len("df").filter(pl.col("df") <= CAP)
        .with_columns(idf=(s1.height / pl.col("df")).log().cast(pl.Float32))
    )
    s1_tok = s1_tok.join(dfreq.select("key"), on="key")  # only blockable tokens
    return s1_tok, dfreq, s1.select("idx", core_s="core", addr_s="addr")


def candidates(others, s1_tok, dfreq, s1_txt, k):
    """Top-k S1 candidates for o records (idx, country, core, addr)."""
    o_tok = (
        tokens(others).join(dfreq, on="key")
        .sort("df").group_by("idx", maintain_order=True).head(M)  # M rarest per record
    )
    pool = (
        o_tok.join(s1_tok, on="key", suffix="_s")
        .group_by("idx", "idx_s").agg(tok=pl.col("idf").sum())
        .filter(pl.col("tok").rank("ordinal", descending=True).over("idx") <= POOL)
        .join(others.select("idx", "core", "addr"), on="idx")
        .join(s1_txt, left_on="idx_s", right_on="idx")
    )
    rs = (
        process.cpdist(pool["core"].to_list(), pool["core_s"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        + process.cpdist(pool["addr"].to_list(), pool["addr_s"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    )
    return (
        pool.select("idx", "idx_s", "tok").with_columns(rs=pl.Series(rs, dtype=pl.Float32))
        .sort(["idx", "rs", "tok"], descending=[False, True, True])
        .with_columns(rank=(pl.int_range(pl.len()).over("idx") + 1).cast(pl.Int16))
        .filter(pl.col("rank") <= k)
        .select(o=pl.col("idx").cast(pl.UInt32), s=pl.col("idx_s").cast(pl.UInt32), tok="tok", rs="rs", rank="rank")
    )


def main(split, k):
    df = pl.read_parquet(WORK / f"{split}.parquet", columns=["src", "country", "core", "addr"]).with_row_index("idx")
    s1_tok, dfreq, s1_txt = index(df)
    others = df.filter(pl.col("src") != 1).select("idx", "country", "core", "addr")
    out_dir = WORK / f"{split}_cand"
    out_dir.mkdir(exist_ok=True)
    total = 0
    for start in range(0, others.height, CHUNK):
        cand = candidates(others.slice(start, CHUNK), s1_tok, dfreq, s1_txt, k)
        cand.write_parquet(out_dir / f"{start // CHUNK:04d}.parquet")
        total += cand.height
        print(f"{min(start + CHUNK, others.height):>10,} / {others.height:,} others, {total:,} candidates", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
