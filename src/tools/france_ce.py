"""France check and France-adapted fine-tuning data for the cross-encoder.

France has no training labels, so two label sources stand in for them:
  heuristic pairs  work/france_pairs.parquet (tools/france_pairs.py): name-linked and address-linked S1 records
  sure matches     France records whose best cross-encoder score is >= 0.99 with the runner-up <= 0.01
S1 ids are split 80/20 by a hash of the id; the 20% are held out for evaluation.

  bundle <scores>           write <scores>: France rows of work/ce and work/ce_base2 test scores
  analyze <scores>          sure / uncertain shares, heuristic-pair agreement, uncertain examples
  data <scores> [N]         work/ce_fr/train.parquet (o, s, rank, a, b, y): France pairs + N sampled work/ce/train.parquet
  eval <scores> <model...>  held-out heuristic agreement and India/US validation sample, per model
<scores>: parquet with o, s, ce_small, ce_base2 for the top-3 candidates of every France S2/S3 test record.
Usage: python src/tools/france_ce.py bundle|analyze|data|eval ...
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ce_data import text  # noqa: E402
from match import accept, best_per_o, cand_chunks  # noqa: E402
from prep import WORK  # noqa: E402

THR, MARGIN = 0.60, 0.30  # decision rule of the uploaded cross-encoder
FR = WORK / "ce_fr"
SURE_HI, SURE_LO = 0.99, 0.01
TEXT_SRC = WORK / os.environ.get("FR_TEXT", "test.parquet")  # parquet the model text comes from
N_FRANCE = 350_000  # France training pairs (heuristic + sure matches), before the India/US sample


def held_out(ids):
    """True for S1 entity ids in the 20% evaluation split (stable hash of the id)."""
    return pl.Series([int(hashlib.md5(i.encode()).hexdigest(), 16) % 5 == 0 for i in ids])


def load(scores_path):
    """France top-3 scores with blocking rank, the test entity ids and each record's best/runner-up per model."""
    ids = pl.read_parquet(WORK / "test.parquet", columns=["entity_id"]).with_row_index("idx")
    sc = pl.read_parquet(scores_path).with_columns(pl.col("o", "s").cast(pl.UInt32))
    rank = pl.scan_parquet(sorted(str(p) for p in cand_chunks("test"))).select("o", "s", "rank") \
        .join(sc.select("o").unique().lazy(), on="o").collect()
    sc = sc.join(rank, on=["o", "s"], how="left")
    return ids, sc


def best(sc, col):
    """Per o: top candidate, its score p and the runner-up p2 for one model column."""
    return best_per_o(sc.select("o", "s", p=col))


def heuristic(ids):
    """Heuristic pairs as test row indices, one S1 per o (name-linked preferred), with the 80/20 split."""
    fp = pl.read_parquet(WORK / "france_pairs.parquet")
    fp = fp.sort("set").unique("o_id", keep="first")  # a_name_linked sorts before b_addr_linked
    fp = fp.join(ids.rename({"entity_id": "o_id", "idx": "o"}), on="o_id") \
        .join(ids.rename({"entity_id": "s1_id", "idx": "h"}), on="s1_id")
    return fp.with_columns(held=held_out(fp["s1_id"].to_list())).select("o", "h", "set", "held")


def analyze(scores_path):
    ids, sc = load(scores_path)
    n_o = sc["o"].n_unique()
    print(f"France records scored: {n_o:,}; pairs {sc.height:,}")
    for col in ("ce_base2", "ce_small"):
        b = best(sc, col)
        sure_m = ((b["p"] >= SURE_HI) & (b["p2"] <= SURE_LO)).sum()
        sure_n = (b["p"] <= SURE_LO).sum()
        acc = accept(b, THR, MARGIN).height
        print(f"{col}: sure match {sure_m / n_o:.1%}, sure none {sure_n / n_o:.1%}, "
              f"uncertain {(n_o - sure_m - sure_n) / n_o:.1%}; accepted at thr {THR} margin {MARGIN}: {acc / n_o:.1%}")
    h = heuristic(ids)
    b = best(sc, "ce_base2")
    acc = accept(b, THR, MARGIN).select("o", pick="s")
    top3 = sc.group_by("o").agg(cand=pl.col("s"))
    r = h.join(acc, on="o", how="left").join(top3, on="o", how="left").with_columns(
        in_top3=pl.col("cand").list.contains(pl.col("h")).fill_null(False))
    r = r.with_columns(out=pl.when(pl.col("pick") == pl.col("h")).then(pl.lit("picked heuristic S1"))
                       .when(pl.col("pick").is_not_null()).then(pl.lit("picked another S1"))
                       .when(pl.col("in_top3")).then(pl.lit("not accepted, heuristic S1 in top 3"))
                       .otherwise(pl.lit("not accepted, heuristic S1 not in top 3")))
    with pl.Config(tbl_rows=20):
        print(r.group_by("set", "out").len().with_columns(
            share=(pl.col("len") / pl.col("len").sum().over("set")).round(4)).sort("set", "out"))
    return ids, sc


def examples(scores_path, n=40, seed=0):
    """Uncertain France records (ce_base2) with their top-3 candidates' text and both models' scores."""
    ids, sc = load(scores_path)
    t = text(pl.read_parquet(TEXT_SRC, columns=["name", "addr"]).with_row_index("idx"))
    b = best(sc, "ce_base2")
    unc = b.filter(~(((pl.col("p") >= SURE_HI) & (pl.col("p2") <= SURE_LO)) | (pl.col("p") <= SURE_LO)))
    pick = unc.sample(n, seed=seed)["o"]
    rows = sc.filter(pl.col("o").is_in(pick.implode())).join(t.rename({"idx": "s", "t": "tb"}), on="s") \
        .join(t.rename({"idx": "o", "t": "ta"}), on="o").sort("o", "rank")
    for o, g in rows.group_by("o", maintain_order=True):
        print(f"\nO: {g['ta'][0]}")
        for r in g.iter_rows(named=True):
            print(f"   r{r['rank']} base2 {r['ce_base2']:.3f} small {r['ce_small']:.3f}  S1: {r['tb']}")


def bundle(out):
    """France rows of the small (work/ce) and base (work/ce_base2) cross-encoder test scores -> out, the <scores> input."""
    fr = pl.read_parquet(WORK / "test.parquet", columns=["country"]).with_row_index("o") \
        .filter(pl.col("country") == "France")["o"]
    rd = lambda d, col: pl.read_parquet(WORK / d / "test_scored" / "*.parquet").with_columns(pl.col("o", "s").cast(pl.UInt32)) \
        .filter(pl.col("o").is_in(fr.implode())).select("o", "s", pl.col("ce").alias(col))
    d = rd("ce", "ce_small").join(rd("ce_base2", "ce_base2"), on=["o", "s"], how="full", coalesce=True)
    d.write_parquet(out)
    print(f"{d.height:,} France pairs -> {out}")


def data(scores_path, n_orig=250_000):
    from rapidfuzz import fuzz, process
    ids, sc = load(scores_path)
    t = text(pl.read_parquet(TEXT_SRC, columns=["name", "addr"]).with_row_index("idx"))
    h = heuristic(ids).filter(~pl.col("held"))
    # address-linked pairs include different businesses at the same address: keep those with similar names only
    names = pl.read_parquet(WORK / "test.parquet", columns=["name"]).with_row_index("idx")
    h = h.join(names.rename({"idx": "o", "name": "na"}), on="o").join(names.rename({"idx": "h", "name": "nb"}), on="h")
    h = h.with_columns(sim=pl.Series(process.cpdist(h["na"].to_list(), h["nb"].to_list(),
                                                    scorer=fuzz.token_set_ratio, workers=-1)))
    kept = h.filter((pl.col("set") == "a_name_linked") | (pl.col("sim") >= 70))
    print(f"heuristic training pairs: {h.height:,}; kept {kept.height:,} "
          f"(address-linked with name token_set_ratio >= 70, all name-linked)")
    h = kept.drop("na", "nb", "sim")
    b = best(sc, "ce_base2")
    sure = b.filter((pl.col("p") >= SURE_HI) & (pl.col("p2") <= SURE_LO)).select("o", h="s")
    # keep held-out S1 entities out of the sure-match set too
    held_s1 = heuristic(ids).filter("held")["h"]
    sure = sure.filter(~pl.col("h").is_in(held_s1.implode()) & ~pl.col("o").is_in(h["o"].implode()))
    rng = np.random.default_rng(0)
    parts = []
    for name, src in (("heuristic", h.select("o", "h")), ("sure", sure)):
        n_o = min(src.height, N_FRANCE // 2 // 3)  # about 3 pairs per record (1 positive, up to 2 negatives)
        src = src.sample(n_o, seed=int(rng.integers(1 << 30)))
        neg = sc.join(src, on="o").filter(pl.col("s") != pl.col("h")).select("o", "s", "rank", y=pl.lit(0, pl.Int8))
        pos = src.join(sc.select("o", "s", "rank"), left_on=["o", "h"], right_on=["o", "s"], how="left") \
            .select("o", s="h", rank=pl.col("rank").fill_null(99).cast(pl.Int16), y=pl.lit(1, pl.Int8))
        d = pl.concat([pos, neg.with_columns(pl.col("rank").cast(pl.Int16))])
        print(f"{name}: {n_o:,} records -> {d.height:,} pairs, {d['y'].mean():.3f} positive")
        parts.append(d)
    fr = pl.concat(parts).join(t.rename({"idx": "o", "t": "a"}), on="o").join(t.rename({"idx": "s", "t": "b"}), on="s")
    orig = pl.read_parquet(WORK / "ce" / "train.parquet")
    orig = orig.sample(min(n_orig, orig.height), seed=0).select("o", "s", pl.col("rank").cast(pl.Int16), "a", "b",
                                                               pl.col("y").cast(pl.Int8))
    # test and train row indices overlap; o/s are only used for bookkeeping here, the model reads a and b
    out = pl.concat([fr.select("o", "s", "rank", "a", "b", "y"), orig]).sample(fraction=1.0, shuffle=True, seed=0)
    FR.mkdir(exist_ok=True)
    out.write_parquet(FR / "train.parquet")
    print(f"work/ce_fr/train.parquet: {out.height:,} pairs ({fr.height:,} France, {orig.height:,} India/US), "
          f"{out['y'].mean():.3f} positive")


def evaluate(scores_path, models, n_val=100_000):
    """Held-out heuristic agreement (France) and India/US validation log loss / best-pick accuracy, per model."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from ce_train import score
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ids, sc = load(scores_path)
    t = text(pl.read_parquet(TEXT_SRC, columns=["name", "addr"]).with_row_index("idx"))
    h = heuristic(ids).filter("held")
    fr = sc.join(h.select("o", "h", "set"), on="o").join(t.rename({"idx": "o", "t": "a"}), on="o") \
        .join(t.rename({"idx": "s", "t": "b"}), on="s")
    # same-name France pairs whose first house number agrees vs differs: a differing number must stay a negative
    raw = pl.read_parquet(TEXT_SRC, columns=["core", "addr"]).with_row_index("idx").with_columns(
        num=pl.col("addr").str.extract(r"(\d+)", 1))
    pr = sc.join(raw.rename({"idx": "o", "core": "co", "num": "no"}).drop("addr"), on="o") \
        .join(raw.rename({"idx": "s", "core": "cs", "num": "ns"}).drop("addr"), on="s") \
        .filter((pl.col("co") == pl.col("cs")) & pl.col("no").is_not_null() & pl.col("ns").is_not_null())
    pr = pr.join(t.rename({"idx": "o", "t": "a"}), on="o").join(t.rename({"idx": "s", "t": "b"}), on="s")
    numchk = {"same number": pr.filter(pl.col("no") == pl.col("ns")), "different number": pr.filter(pl.col("no") != pl.col("ns"))}
    numchk = {k: v.sample(min(20_000, v.height), seed=0) for k, v in numchk.items()}
    va = pl.read_parquet(WORK / "ce" / "val.parquet")
    va = va.join(va.select("o").unique().sample(n_val, seed=0), on="o")
    print(f"held-out France: {fr['o'].n_unique():,} records, {fr.height:,} pairs; "
          f"India/US val sample: {va['o'].n_unique():,} records, {va.height:,} pairs")
    for m in models:
        tok = AutoTokenizer.from_pretrained(m)
        model = AutoModelForSequenceClassification.from_pretrained(m).to(dev)
        if Path(m).name == "ce_small_model":  # the bundle already holds this model's France scores
            p = fr["ce_small"].to_numpy()
        else:
            p = score(model, tok, fr["a"].to_list(), fr["b"].to_list(), dev)
        FR.mkdir(exist_ok=True)
        fr.select("o", "s", p=pl.Series(p)).write_parquet(FR / f"heldout_{Path(m).name}.parquet")
        b = best_per_o(fr.select("o", "s", p=pl.Series(p)))
        acc = accept(b, THR, MARGIN).select("o", pick="s")
        r = h.join(acc, on="o", how="left")
        for st in ("a_name_linked", "b_addr_linked"):
            x = r.filter(pl.col("set") == st)
            print(f"{Path(m).name} France {st}: picked heuristic S1 {(x['pick'] == x['h']).fill_null(False).mean():.4f}, "
                  f"picked another {(x['pick'].is_not_null() & (x['pick'] != x['h'])).mean():.4f}, "
                  f"not accepted {x['pick'].is_null().mean():.4f}  ({x.height:,} records)")
        for label, g in numchk.items():
            if Path(m).name == "ce_small_model":
                pg = g["ce_small"].to_numpy()
            else:
                pg = score(model, tok, g["a"].to_list(), g["b"].to_list(), dev)
            print(f"{Path(m).name} France same name, {label}: {g.height:,} pairs, mean p {pg.mean():.3f}, "
                  f"p >= 0.5 {(pg >= 0.5).mean():.3f}")
        q = score(model, tok, va["a"].to_list(), va["b"].to_list(), dev).clip(1e-6, 1 - 1e-6)
        y = va["y"].to_numpy()
        ll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        v = va.select("o", "y", q=pl.Series(q)).sort("q", descending=True).group_by("o").agg(
            top_y=pl.col("y").first(), any_y=pl.col("y").max(), top_q=pl.col("q").first())
        withm = v.filter(pl.col("any_y") == 1)
        nom = v.filter(pl.col("any_y") == 0)
        print(f"{Path(m).name} India/US val: log loss {ll:.4f}; best pick correct {withm['top_y'].mean():.4f} "
              f"({withm.height:,} records with a match in top 3); no-match records scored < {THR}: "
              f"{(nom['top_q'] < THR).mean():.4f} ({nom.height:,})")


def compare(scores_path, n_ex=30, s2_rule=(0.723, 0.3), ce_rule=(THR, MARGIN)):
    """Stage 2 (work/stage2_test_scored.parquet) vs the cross-encoder (ce_base2) on France: where they disagree,
    which side the heuristic pairs support, and examples of each disagreement."""
    from rapidfuzz import fuzz, process
    ids, sc = load(scores_path)
    fr_o = sc.select("o").unique()
    s2 = pl.scan_parquet(WORK / "stage2_test_scored.parquet").join(fr_o.lazy(), on="o").collect()
    b2 = best_per_o(s2.select("o", "s", "p"))
    bc = best(sc, "ce_base2")
    p2 = accept(b2, *s2_rule).select("o", s2="s", p_s2="p")
    pc = accept(bc, *ce_rule).select("o", ce="s", p_ce="p")
    d = fr_o.join(p2, on="o", how="left").join(pc, on="o", how="left").with_columns(
        case=pl.when(pl.col("s2").is_null() & pl.col("ce").is_null()).then(pl.lit("neither"))
        .when(pl.col("s2") == pl.col("ce")).then(pl.lit("both, same S1"))
        .when(pl.col("s2").is_not_null() & pl.col("ce").is_not_null()).then(pl.lit("both, different S1"))
        .when(pl.col("s2").is_not_null()).then(pl.lit("stage2 only")).otherwise(pl.lit("CE only")))
    print(f"France records {d.height:,}: stage2 accepts {d['s2'].is_not_null().sum():,}, "
          f"CE accepts {d['ce'].is_not_null().sum():,}")
    h = heuristic(ids)
    names = pl.read_parquet(WORK / "test.parquet", columns=["name"]).with_row_index("idx")
    h = h.join(names.rename({"idx": "o", "name": "na"}), on="o").join(names.rename({"idx": "h", "name": "nb"}), on="h")
    h = h.with_columns(sim=pl.Series(process.cpdist(h["na"].to_list(), h["nb"].to_list(),
                                                    scorer=fuzz.token_set_ratio, workers=-1)))
    h = h.with_columns(grp=pl.when(pl.col("set") == "a_name_linked").then(pl.lit("name-linked"))
                       .when(pl.col("sim") >= 70).then(pl.lit("addr-linked sim>=70"))
                       .otherwise(pl.lit("addr-linked other"))).select("o", "h", "grp")
    r = d.join(h, on="o", how="left").with_columns(
        support=pl.when(pl.col("h").is_null()).then(pl.lit("no heuristic pair"))
        .when(pl.col("s2") == pl.col("h")).then(pl.when(pl.col("ce") == pl.col("h")).then(pl.lit("both = heuristic"))
                                                 .otherwise(pl.lit("stage2 = heuristic")))
        .when(pl.col("ce") == pl.col("h")).then(pl.lit("CE = heuristic"))
        .otherwise(pl.lit("neither = heuristic")))
    with pl.Config(tbl_rows=40, tbl_width_chars=160):
        print(r.group_by("case").len().with_columns(share=(pl.col("len") / d.height).round(4)).sort("len", descending=True))
        print(r.filter(pl.col("h").is_not_null()).group_by("case", "support").len().sort("case", "len", descending=[False, True]))
        for g in ("name-linked", "addr-linked sim>=70"):
            x = r.filter(pl.col("grp") == g)
            print(f"{g} ({x.height:,}): stage2 hit {(x['s2'] == x['h']).fill_null(False).mean():.4f} "
                  f"other {(x['s2'].is_not_null() & (x['s2'] != x['h'])).mean():.4f} | "
                  f"CE hit {(x['ce'] == x['h']).fill_null(False).mean():.4f} "
                  f"other {(x['ce'].is_not_null() & (x['ce'] != x['h'])).mean():.4f}")
    t = text(pl.read_parquet(TEXT_SRC, columns=["name", "addr"]).with_row_index("idx"))
    tt = dict(zip(t["idx"].to_list(), t["t"].to_list()))
    top = sc.sort("o", "rank").group_by("o", maintain_order=True).agg("s", "ce_base2")
    top = dict((o, list(zip(s, c))) for o, s, c in top.iter_rows())
    s2p = s2.sort("p", descending=True).group_by("o").agg(pl.col("s").head(3), pl.col("p").head(3))
    s2p = dict((o, list(zip(s, p))) for o, s, p in s2p.iter_rows())
    for case in ("stage2 only", "CE only", "both, different S1"):
        ex = r.filter(pl.col("case") == case).sample(min(n_ex, r.filter(pl.col("case") == case).height), seed=0)
        print(f"\n===== {case}: {r.filter(pl.col('case') == case).height:,} records; examples =====")
        for row in ex.iter_rows(named=True):
            o = row["o"]
            print(f"\nO: {tt[o]}   [heuristic: {row['support']}]")
            print("   stage2 top: " + " || ".join(f"{p:.3f} {tt[s][:90]}" for s, p in s2p.get(o, [])))
            print("   CE top:     " + " || ".join(f"{p:.3f} {tt[s][:90]}" for s, p in top.get(o, [])))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "bundle":
        bundle(args[0])
    elif cmd == "analyze":
        analyze(args[0])
    elif cmd == "examples":
        examples(args[0], int(args[1]) if len(args) > 1 else 40)
    elif cmd == "data":
        data(args[0], int(args[1]) if len(args) > 1 else 250_000)
    elif cmd == "compare":
        compare(args[0], int(args[1]) if len(args) > 1 else 30)
    elif cmd == "eval":
        evaluate(args[0], args[1:])
