"""Indian-script names: learn a word dictionary from the training pairs, then re-block and re-score those records.

About 18% of India's S2/S3 names are written in an Indian script (Devanagari, Tamil, Telugu, ...). anyascii turns
"பாலாஜி பிசினஸ் பிரைவேட் லிமிடெட்" into "palaji picins piraivet limitet", which shares no word with the S1 record
"Balaji Business Private Limited", so blocking and the cross-encoders miss many of these records. The names come
from a fixed vocabulary, so a word dictionary learned from the training pairs (transliterated word -> S1 word,
aligned by position when both names have the same number of words) maps them back. On validation the share of
these records whose name matches the true S1 name (token set ratio >= 90) goes from 6.5% to 94.2%.

  learn      -> work/translit.json (validation entities are left out, so validation stays honest)
  run SPLIT  mapped name -> new blocking candidates (top NEW_K) plus the pairs already scored for these records;
             scored by the first cross-encoder, then the uncertain records again by the CE_DIR model
             -> work/translit/SPLIT/NNNN.parquet (o, s, ce). train: validation records only. Resumable.
"""
import json
import sys
from collections import Counter, defaultdict

import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from block import candidates, index
from ce_data import OUT, text
from ce_rescore import rescore
from ce_train import score
from match import truth_pairs
from prep import DATA, LEGAL, WORK, map_tokens

NEW_K = 5  # new blocking candidates per record
PART = 100_000  # records per output part
DICT = WORK / "translit.json"
SMALL = WORK / "ce"
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def indic_ids(split):
    """entity_ids of S2/S3 records whose raw name contains letters outside Latin."""
    raw = pl.concat(pl.read_csv(DATA / split / f"{split}_source{i}.tsv", separator="\t", quote_char=None,
                                columns=["entity_id", "business_name"]) for i in (2, 3))
    return raw.filter(pl.col("business_name").str.contains(r"[^\x00-\x7FÀ-ɏ]"))["entity_id"]


def learn():
    recs = pl.read_parquet(WORK / "train.parquet", columns=["entity_id", "name"]).with_row_index("idx")
    p = (truth_pairs(recs.select("idx", "entity_id"))
         .join(recs.select(o="idx", o_id="entity_id", o_name="name"), on="o")
         .join(recs.select(s_true="idx", s_id="entity_id", s_name="name"), on="s_true")
         .filter(pl.col("o_id").is_in(indic_ids("train").implode())
                 & (pl.col("s_id").str.slice(3).cast(pl.Int64) % 100 != 0)))
    cnt = defaultdict(Counter)
    for a, b in p.select("o_name", "s_name").iter_rows():
        a, b = a.split(), b.split()
        if len(a) == len(b):
            for x, y in zip(a, b):
                cnt[x][y] += 1
    m = {}
    for x, c in cnt.items():
        y, n = c.most_common(1)[0]
        if n >= 2 and n / sum(c.values()) >= 0.5 and x != y:
            m[x] = y
    DICT.write_text(json.dumps(m, sort_keys=True))
    print(f"{p.height:,} training pairs -> {len(m):,} words -> {DICT}")


def scored_pairs(split):
    """(o, s) pairs already scored for a split: blocking top-3 and the dense-search extras."""
    if split == "train":
        parts = [pl.scan_parquet(SMALL / "val.parquet"), pl.scan_parquet(WORK / "dense" / "train_scored" / "*.parquet")]
    else:
        parts = [pl.scan_parquet(SMALL / "test_scored" / "*.parquet"), pl.scan_parquet(WORK / "dense" / "test_scored" / "*.parquet")]
    return pl.concat([p.select(pl.col("o", "s").cast(pl.UInt32)) for p in parts]).unique().collect()


def run(split):
    m = json.loads(DICT.read_text())
    recs = pl.read_parquet(WORK / f"{split}.parquet", columns=["entity_id", "src", "country", "name", "core", "addr"]).with_row_index("idx")
    sel = recs.filter(pl.col("entity_id").is_in(indic_ids(split).implode()))["idx"]
    if split == "train":
        from dense import val_os
        sel = sel.filter(sel.cast(pl.UInt32).is_in(val_os().implode()))
    o = (recs.filter(pl.col("idx").is_in(sel.implode()))
         .with_columns(name=map_tokens(pl.col("name"), m)).with_columns(core=map_tokens(pl.col("name"), {w: "" for w in LEGAL}))
         .select("idx", "country", "name", "core", "addr"))
    print(f"{split}: {o.height:,} records with Indian-script names", flush=True)
    s1_index = index(recs.select("idx", "src", "country", "core", "addr"))
    txt = text(recs)["t"]
    old = scored_pairs(split)
    del recs
    tok = AutoTokenizer.from_pretrained(SMALL / "model")
    small = AutoModelForSequenceClassification.from_pretrained(SMALL / "model").to(DEV)
    big_tok = AutoTokenizer.from_pretrained(OUT / "model")
    big = AutoModelForSequenceClassification.from_pretrained(OUT / "model").to(DEV)
    if DEV == "cuda":
        small, big = small.half(), big.half()
    out = WORK / "translit" / split
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, o.height, PART):
        f = out / f"{i // PART:04d}.parquet"
        if f.exists():
            continue
        part = o.slice(i, PART)
        new = candidates(part.select("idx", "country", "core", "addr"), *s1_index, NEW_K).select("o", "s")
        pairs = pl.concat([new, old.filter(pl.col("o").is_in(part["idx"].cast(pl.UInt32).implode()))]).unique()
        # scored with the record's original text: the cross-encoders already read these transliterations well
        # (the mapped name only finds the candidates); mapped names also lose the legal form's evidence
        pairs = pairs.with_columns(a=txt.gather(pairs["o"]))
        pairs = pairs.with_columns(b=txt.gather(pairs["s"]))
        pairs = pairs.with_columns(ce=pl.Series(score(small, tok, pairs["a"].to_list(), pairs["b"].to_list(), DEV)))
        res, n = rescore(pairs, big, big_tok, DEV)
        res.write_parquet(f)
        print(f"{f.name}: {part.height:,} records, {pairs.height:,} pairs, {n:,} rescored", flush=True)
    (out / "DONE").touch()


if __name__ == "__main__":
    learn() if sys.argv[1] == "learn" else run(sys.argv[2])
