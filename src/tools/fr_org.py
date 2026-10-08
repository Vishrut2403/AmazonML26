"""France org-word rules, applied to a finished matching file (each step was checked on the leaderboard).

France S1 names look like "<core words> <org word> <legal form>" (e.g. "lille amicale sarl"). Decoy records swap
the org word for another one (amicale -> club) or add one, most of them at the same address, so the models
(trained on India/US) accept them. True variants may drop the org word and add a filler (amicale -> & fils).
  1. remove France pairs whose differing core words are all org/filler words, with at least one added;
  2. put back the pairs where the record only adds fillers; remove pairs where both names differ by org words;
  3. repeat 2 with the larger org/industry vocabulary learned from shifted-number decoys (typo-aware), also
     removing records that add an org word;
  4. give records that are still unmatched their only candidate at the same address (same street words and
     house number) when the names differ by typos only, the record is a one-word brand name, or the France
     cross-encoder is sure (>= 0.9);
  5. remove pairs where the house number moved up by 1 to 60 and the record adds a decoy filler or a legal form
     (the decoy recipe: true variants move the number up for under 1% of records, decoys for almost all).
Usage: python src/tools/fr_org.py BASE_TSV OUT_TSV   (needs work/test.parquet and work/fr_ce_scores.parquet)
"""
import sys
from collections import Counter
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assemble import base_pairs  # noqa: E402
from edits import LEGAL, STOP, features  # noqa: E402
from match import write_lists  # noqa: E402
from prep import WORK  # noqa: E402

ORG = ["club", "ecole", "amicale", "comite", "sportive", "amis", "centre", "parents", "union", "primaire", "college",
       "maison", "societe", "pharmacie", "fetes", "loisirs", "federation", "anciens", "sante", "culture", "culturel",
       "culturelle", "jeunes", "association", "art", "sport", "elementaire", "maternelle", "residence", "groupement",
       "lycee", "foyer", "institut", "cercle", "atelier", "musique", "gestion", "solidarite", "entraide",
       "etablissements", "ets"]
FILL = ["groupe", "developpement", "france", "services", "associes", "fils", "and", "et", "holding", "participations",
        "distribution", "international", "compagnie", "cie", "freres", "de", "du", "des", "la", "le", "les", "d", "l"]
GEN = ORG[:35] + ["groupe", "developpement", "compagnie", "services", "associes", "fils", "freres", "holding",
                  "participations", "distribution", "international", "france", "and", "de", "du", "des", "la", "le",
                  "les", "d", "l", "et"]  # step 1 list (the org words up to "atelier", fillers, articles)
# words that replace a known org word in shifted-number decoys at least ~15 times (typo forms left out)
NEW = ["section", "collectif", "service", "patrimoine", "danse", "ehpad", "theatre", "conseil", "soins", "sportif",
       "medico", "agricole", "ateliers", "auto", "clinique", "cafe", "energie", "concept", "etablissement", "medical",
       "hopital", "automobile", "restaurant", "travaux", "transports", "eau", "batiment", "garage", "restauration",
       "bar", "immobiliere", "transport", "hotel", "construction", "peinture", "coiffure", "immobilier", "gaz",
       "logistique", "finance", "finances", "industrie", "elevage", "btp", "immo", "menuiserie"]
VOCAB, SKIP = set(ORG) | set(NEW), set(FILL) | LEGAL | STOP
DROP_KINDS = {"swap", "org added", "org added (other drop)", "org dropped + filler (other add)"}
DECOY_ADD = {"groupe", "developpement", "france", "participations", "distribution", "international", "holding",
             "sarl", "sas", "sa", "sasu", "eurl", "sci", "snc", "ei"}
EDITS = ["add_core", "drop_core", "add_generic", "drop_generic", "add_filler", "drop_filler", "add_legal", "drop_legal",
         "legal_swap"]


def kind(ow, sw):
    """How the org words differ between record words ow and S1 words sw (typo pairs matched up first)."""
    norm = lambda w: w.replace("1", "l").replace("0", "o")
    so, oo = set(sw), set(ow)
    a = [norm(w) for w in ow if w not in so and w not in SKIP]
    b = [norm(w) for w in sw if w not in oo and w not in SKIP]
    filler_added = any(w in FILL for w in ow if w not in so)
    for x in list(a):
        for y in b:
            if fuzz.ratio(x, y) >= 75:
                a.remove(x)
                b.remove(y)
                break
    oa, ob = [x for x in a if x in VOCAB], [y for y in b if y in VOCAB]
    if oa and ob:
        return "swap"
    if oa:
        return "org added" + (" (other drop)" if b else "")
    if ob:
        return "org dropped" + (" + filler" if filler_added else "") + (" (other add)" if a else "")
    return "none"


def repeats_word(on, sn):
    """The record repeats a word the S1 name already has ("x france club" -> "x france club france")."""
    co, cs = Counter(on.split()), Counter(sn.split())
    return any(c > cs[w] > 0 for w, c in co.items())


def main(base, out):
    recs = pl.read_parquet(WORK / "test.parquet", columns=["entity_id", "src", "country", "name", "core", "addr", "nums"]) \
        .with_row_index("idx")
    ids = recs.select("idx", "entity_id")
    r = recs.select(pl.col("idx").cast(pl.UInt32), "country", "name", ct=pl.col("core").str.split(" "),
                    w=pl.col("name").str.split(" "))
    pairs = base_pairs(base, ids)
    fr = lambda p: p.join(r.select(o="idx", country="country", oct="ct", ow="w"), on="o") \
        .join(r.select(s="idx", sct="ct", sw="w"), on="s").filter(pl.col("country") == "France")
    only = lambda a, b: pl.col(a).list.set_difference(b)
    within = lambda c, words: pl.col(c).list.eval(pl.element().is_in(words).not_()).list.sum() == 0
    has = lambda c: pl.col(c).list.eval(pl.element().is_in(ORG)).list.any()

    # 1-2
    q = fr(pairs).with_columns(o_only=only("oct", "sct"), s_only=only("sct", "oct"))
    step1 = q.filter(within("o_only", GEN) & within("s_only", GEN) & (pl.col("o_only").list.len() > 0))
    restore = step1.filter(within("o_only", FILL)).select("o", "s")
    kept = pairs.join(step1.select("o", "s"), on=["o", "s"], how="anti")
    mixed = fr(kept).with_columns(o_only=only("ow", "sw"), s_only=only("sw", "ow")).filter(has("o_only") & has("s_only"))
    pairs = pl.concat([kept.join(mixed.select("o", "s"), on=["o", "s"], how="anti"), restore]).unique()
    print(f"step 1: -{step1.height:,}; step 2: +{restore.height:,} -{mixed.height:,}")

    # 3
    q = fr(pairs)
    q = q.with_columns(k=pl.Series([kind(a, b) for a, b in zip(q["ow"].to_list(), q["sw"].to_list())], dtype=pl.String))
    drop = q.filter(pl.col("k").is_in(DROP_KINDS)).select("o", "s")
    pairs = pairs.join(drop, on=["o", "s"], how="anti")
    print(f"step 3: -{drop.height:,}")

    # 4
    sc = pl.read_parquet(WORK / "fr_ce_scores.parquet")
    cand = pl.concat([sc.select("o", "s"), fr(base_pairs(base, ids)).select("o", "s")]).unique()
    ef = features(cand, recs).join(sc, on=["o", "s"], how="left")
    same = (pl.col("num_rel") == 3) & (pl.col("a_add") == 0) & (pl.col("a_drop") == 0) & (pl.col("o_addr_empty") == 0)
    ef = ef.with_columns(n_same=same.sum().over("o")).filter(same & (pl.col("n_same") == 1) & ~pl.col("o").is_in(pairs["o"].implode()))
    ef = ef.join(r.select(o="idx", o_name="name", ow="w"), on="o").join(r.select(s="idx", s_name="name", sw="w"), on="s")
    ef = ef.with_columns(k=pl.Series([kind(a, b) for a, b in zip(ef["ow"].to_list(), ef["sw"].to_list())], dtype=pl.String),
                         rep=pl.Series([repeats_word(a, b) for a, b in zip(ef["o_name"].to_list(), ef["s_name"].to_list())]))
    typo = (pl.col("typo") > 0) & (pl.sum_horizontal(EDITS) == 0)
    brand = (pl.col("n_common") == 0) & (pl.col("o_initials") == 0) & pl.col("o_name").str.contains(r"^[a-z]{5,}( (labs|co|one|sys))?$")
    add = ef.filter(pl.col("k").is_in(["none", "org dropped", "org dropped (other add)", "org dropped + filler"]) & ~pl.col("rep")
                    & (typo | brand | (pl.col("fce") >= 0.9))).select("o", "s").unique("o")
    pairs = pl.concat([pairs, add])
    print(f"step 4: +{add.height:,}")

    # 5
    n1 = recs.select(pl.col("idx").cast(pl.UInt32), n1=pl.col("nums").str.extract(r"^(\d+)").cast(pl.Int64, strict=False))
    q = fr(pairs).join(n1.rename({"idx": "o", "n1": "on"}), on="o").join(n1.rename({"idx": "s", "n1": "sn"}), on="s") \
        .filter((pl.col("on") - pl.col("sn")).is_between(1, 60))
    decoy = [any(w in DECOY_ADD for w in Counter(a) - Counter(b)) for a, b in zip(q["ow"].to_list(), q["sw"].to_list())]
    drop = q.filter(pl.Series(decoy, dtype=pl.Boolean)).select("o", "s")
    pairs = pairs.join(drop, on=["o", "s"], how="anti")
    print(f"step 5: -{drop.height:,}")
    assert pairs["o"].n_unique() == pairs.height
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_lists(pairs.lazy(), recs, "matched_entity_ids", out)
    print(f"{pairs.height:,} pairs -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
