"""Edit-explanation features: which generator edits turn the S1 record into this S2/S3 record.

The sources make true matches and decoys with different edits: true matches get typos, case and accents, dropped
or moved legal forms, reformatted numbers, dropped or reordered address parts; decoys get their house number moved
up by 1-60, a legal form swapped or added, a filler added, a core word added or replaced. Similarity scores mix
these; here each pair is described by the edits that explain it, with typo-level word changes (character edits)
kept apart from real word replacements.
  features(pairs, recs) -> pairs with EDIT_FEATS (recs: idx, country, name, addr, nums from prep.py)
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prep import FR_REGIONS  # noqa: E402

LEGAL = {"inc", "incorporated", "llc", "llp", "corp", "corporation", "co", "company", "ltd", "limited", "pvt", "private",
         "pte", "plc", "lp", "pc", "pllc", "sarl", "sas", "sa", "eurl", "sasu", "sci", "snc", "ei"}
FILLER = {"group", "groupe", "services", "international", "enterprises", "developpement", "development", "compagnie",
          "associes", "associates", "fils", "sons", "freres", "brothers", "holding", "holdings", "participations",
          "distribution", "france", "trust", "partners", "center", "centre", "solutions", "global", "industries", "trading",
          "sri", "shri", "mr", "ms", "cie", "and", "et"}
GENERIC = {"club", "ecole", "amicale", "comite", "sportive", "amis", "parents", "union", "primaire", "college", "maison",
           "societe", "pharmacie", "fetes", "loisirs", "federation", "anciens", "sante", "culture", "culturel", "culturelle",
           "jeunes", "association", "art", "sport", "elementaire", "maternelle", "residence", "groupement", "lycee", "foyer",
           "institut", "cercle", "atelier", "musique", "gestion", "solidarite", "entraide", "etablissements", "ets"}
STOP = {"the", "of", "de", "du", "des", "la", "le", "les", "d", "l"}
EDIT_FEATS = ["n_o", "n_s", "n_common", "typo", "add_core", "add_generic", "add_filler", "add_legal", "drop_core",
              "drop_generic", "drop_filler", "drop_legal", "legal_swap", "order_changed", "o_initials", "o_web",
              "a_common", "a_typo", "a_add", "a_drop", "o_addr_empty", "s_addr_empty", "num_rel", "num_diff",
              "o_num_extra", "s_num_missing"]


def _words(s):
    return [w for w in (s or "").split() if w not in STOP]


def _align(a, b):
    """(common, typo, a_only_unmatched, b_only_unmatched) for word lists, typo = char-level near match."""
    sa, sb = set(a), set(b)
    ao, bo = [w for w in a if w not in sb], [w for w in b if w not in sa]
    typo, used, a_left = 0, set(), []
    for w in ao:
        best, bi = 0, -1
        for i, v in enumerate(bo):
            if i in used:
                continue
            r = fuzz.ratio(w, v)
            if r > best:
                best, bi = r, i
        if best >= 75 and (len(w) >= 3 or best == 100):
            typo += 1
            used.add(bi)
        else:
            a_left.append(w)
    b_left = [v for i, v in enumerate(bo) if i not in used]
    return len(sa & sb), typo, a_left, b_left


def _row(on, sn, oa, sa):
    a, b = _words(on), _words(sn)
    common, typo, ao, bo = _align(a, b)
    kind = lambda ws, st: sum(w in st for w in ws)
    core_add = sum(w not in LEGAL | FILLER | GENERIC for w in ao)
    core_drop = sum(w not in LEGAL | FILLER | GENERIC for w in bo)
    la, lb = [w for w in a if w in LEGAL], [w for w in b if w in LEGAL]
    ca, cb = [w for w in a if w in set(b)], [w for w in b if w in set(a)]
    ta, tb = [w for w in (oa or "").split() if not w.isdigit()], [w for w in (sa or "").split() if not w.isdigit()]
    acommon, atypo, aao, abo = _align(ta, tb)
    return (len(a), len(b), common, typo, core_add, kind(ao, GENERIC), kind(ao, FILLER), kind(ao, LEGAL), core_drop,
            kind(bo, GENERIC), kind(bo, FILLER), kind(bo, LEGAL), int(bool(la) and bool(lb) and sorted(la) != sorted(lb)),
            int(ca != cb), acommon, atypo, len(aao), len(abo))


def features(pairs, recs):
    a = pl.col("addr")
    for reg in sorted(FR_REGIONS, key=len, reverse=True):
        a = a.str.replace_all(rf"\b{reg}\b", " ")
    r = recs.select(pl.col("idx").cast(pl.UInt32), "name", addr=a.str.replace_all(" +", " ").str.strip_chars(),
                    n1=pl.col("nums").str.extract(r"^(\d+)").cast(pl.Int64, strict=False), nums=pl.col("nums").str.split(" "))
    p = pairs.select(pl.col("o", "s").cast(pl.UInt32)).join(r.rename(lambda c: c if c == "idx" else "o_" + c), left_on="o", right_on="idx") \
        .join(r.rename(lambda c: c if c == "idx" else "s_" + c), left_on="s", right_on="idx")
    rows = [_row(*t) for t in zip(p["o_name"].to_list(), p["s_name"].to_list(), p["o_addr"].to_list(), p["s_addr"].to_list())]
    cols = ["n_o", "n_s", "n_common", "typo", "add_core", "add_generic", "add_filler", "add_legal", "drop_core", "drop_generic",
            "drop_filler", "drop_legal", "legal_swap", "order_changed", "a_common", "a_typo", "a_add", "a_drop"]
    f = pl.DataFrame(np.array(rows, dtype=np.int16), schema=cols, orient="row")
    d = pl.col("o_n1") - pl.col("s_n1")
    p = pl.concat([p, f], how="horizontal").with_columns(
        o_initials=pl.col("o_name").str.contains(r"^[a-z]{2,4}$").cast(pl.Int8),
        o_web=pl.col("o_name").str.contains(r"^[a-z0-9]+ (com|fr|net|org|in|co)$").cast(pl.Int8),
        o_addr_empty=(pl.col("o_addr") == "").cast(pl.Int8), s_addr_empty=(pl.col("s_addr") == "").cast(pl.Int8),
        num_rel=pl.when(pl.col("o_n1").is_null() & pl.col("s_n1").is_null()).then(0).when(pl.col("o_n1").is_null()).then(1)
        .when(pl.col("s_n1").is_null()).then(2).when(d == 0).then(3).when(d.is_between(1, 60)).then(4)
        .when(d.is_between(-60, -1)).then(5).otherwise(6).cast(pl.Int8),
        num_diff=d.clip(-1000, 1000).cast(pl.Float32),
        o_num_extra=pl.col("o_nums").list.set_difference("s_nums").list.len().cast(pl.Int16),
        s_num_missing=pl.col("s_nums").list.set_difference("o_nums").list.len().cast(pl.Int16))
    return p.select("o", "s", *EDIT_FEATS)
