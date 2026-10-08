"""Dense candidate search, to catch matches the token blocking (block.py) misses.

Records are embedded with multilingual-e5-small as it comes (no fine-tuning); each o record's nearest S1 records in
the same country (cosine similarity) become extra candidates. The first cross-encoder (work/ce) then scores the
extra candidates, plus blocking ranks 4..R, for the o records it did not already match confidently.

  search SPLIT  nearest K S1 records per o -> work/dense/SPLIT/NNNN.parquet (o, s, drank, dsim)
                (train: only the o records used for validation)
  recall        validation: how many true pairs blocking and the dense search find, and the extra pairs needed
  score SPLIT   cross-encoder scores for the extra pairs -> work/dense/SPLIT_scored/NNNN.parquet (o, s, ce)
  final [translit]  merge with the existing scores in CE_DIR (and translit.py's), tune on validation, write
                output/matching_results_dense.tsv and matching_results_dense_strict.tsv
All steps are resumable: finished chunks are skipped.
"""
import json
import sys

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from ce_data import OUT, text
from ce_predict import STRICT, val_truth
from ce_train import score
from match import accept, best_per_o, cand_chunks, val_score, write_lists
from prep import ROOT, WORK

EMB = "intfloat/multilingual-e5-small"
K = 5  # dense candidates per o
R = 6  # blocking ranks 4..R are scored too
# validation recall of true pairs: blocking top-3 97.13%; + dense top-3 98.16%; + dense top-5 and ranks 4-6 98.48%
U = 0.9  # o records whose best first-pass score is >= U are not expanded
FR_THR = 0.85  # stricter threshold for France (no labels there; chosen on likely pairs from src/tools/france_pairs.py)
DEN = WORK / "dense"
SMALL = WORK / "ce"  # first cross-encoder: its scores pick the records to expand, its model scores the new pairs
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def embed(texts, model, tok, n=512):
    """Unit-length mean-pooled e5 embeddings (float16), in input order."""
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
    order = np.argsort([len(t) for t in texts], kind="stable")
    with torch.no_grad():
        for i in range(0, len(order), n):
            j = order[i:i + n]
            enc = tok(["query: " + texts[k] for k in j], truncation=True, max_length=64, padding=True,
                      return_tensors="pt").to(DEV)
            h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            out[j] = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).half().cpu().numpy()
    return out


def val_os():
    """o records of the validation set: those scored by the cross-encoder plus every true match of a val entity."""
    _, truth = val_truth()
    tv = pl.Series("o", [o for os in truth.values() for o in os], dtype=pl.UInt32)
    return pl.concat([pl.read_parquet(SMALL / "val.parquet", columns=["o"])["o"].cast(pl.UInt32), tv]).unique()


def search(split):
    recs = pl.read_parquet(WORK / f"{split}.parquet", columns=["src", "country", "name", "addr"]).with_row_index("idx")
    txt, country = text(recs)["t"], recs["country"].to_numpy()
    n1 = recs.filter(pl.col("src") == 1).height  # S1 records are rows 0..n1-1
    del recs
    tok = AutoTokenizer.from_pretrained(EMB)
    model = AutoModel.from_pretrained(EMB).to(DEV).eval()
    if DEV == "cuda":
        model = model.half()
    (DEN / split).mkdir(parents=True, exist_ok=True)
    f1 = DEN / f"{split}_s1.npy"
    if not f1.exists():
        np.save(f1, embed(txt[:n1].to_list(), model, tok))
    e1 = np.load(f1)
    s1 = {c: np.flatnonzero(country[:n1] == c) for c in np.unique(country[:n1])}
    e1 = {c: torch.from_numpy(e1[g]).to(DEV) for c, g in s1.items()}  # per country, on the GPU
    keep = val_os() if split == "train" else None
    for f in cand_chunks(split):
        if (DEN / split / f.name).exists():
            continue
        o = pl.read_parquet(f, columns=["o"])["o"].unique().sort()
        if keep is not None:
            o = o.filter(o.is_in(keep.implode()))
        o = o.to_numpy()
        eo = torch.from_numpy(embed(txt.gather(o).to_list(), model, tok)).to(DEV)
        res = []
        for c, g in s1.items():
            sel = np.flatnonzero(country[o] == c)
            for i in range(0, len(sel), 256):
                b = sel[i:i + 256]
                sim, top = (eo[b] @ e1[c].T).float().topk(K, dim=1)
                res.append(pl.DataFrame({
                    "o": np.repeat(o[b], K).astype(np.uint32), "s": g[top.cpu().numpy()].ravel().astype(np.uint32),
                    "drank": np.tile(np.arange(1, K + 1, dtype=np.uint8), len(b)), "dsim": sim.cpu().numpy().ravel()}))
        pl.concat(res).write_parquet(DEN / split / f.name)
        print(f.name, len(o), flush=True)


def new_pairs(f, split, keep=None, first=None):
    """Pairs to score for the o records of candidate chunk f: dense top-K and blocking ranks 4..R, minus blocking
    top-3, for o records whose best first-pass score is below U."""
    c = pl.read_parquet(f, columns=["o", "s", "rank"]).with_columns(pl.col("o", "s").cast(pl.UInt32))
    if keep is not None:
        c = c.filter(pl.col("o").is_in(keep.implode()))
    if first is None:
        first = pl.read_parquet(SMALL / "test_scored" / f.name)
    sure = first.filter(pl.col("ce") >= U)["o"].cast(pl.UInt32).unique()
    d = pl.read_parquet(DEN / split / f.name, columns=["o", "s", "drank"]).filter(pl.col("drank") <= K).drop("drank")
    new = pl.concat([c.filter(pl.col("rank").is_between(4, R)).select("o", "s"), d]).unique()
    new = new.filter(~pl.col("o").is_in(sure.implode()))
    return new.join(c.filter(pl.col("rank") <= 3).select("o", "s"), on=["o", "s"], how="anti")


def recall():
    val_s, truth = val_truth()
    tv = pl.DataFrame({"o": [o for os in truth.values() for o in os],
                       "s": [s for s, os in truth.items() for _ in os]}).with_columns(pl.col("o", "s").cast(pl.UInt32))
    keep = tv["o"].unique()
    cand = pl.concat(pl.read_parquet(f, columns=["o", "s", "rank"]).with_columns(pl.col("o", "s").cast(pl.UInt32))
                     .filter(pl.col("o").is_in(keep.implode())) for f in cand_chunks("train"))
    dn = pl.read_parquet(DEN / "train" / "*.parquet").filter(pl.col("o").is_in(keep.implode()))
    t = tv.join(cand, on=["o", "s"], how="left").join(dn, on=["o", "s"], how="left")
    n = t.height
    for name, e in (("blocking top-3", pl.col("rank") <= 3), ("blocking top-10", pl.col("rank") <= 10),
                    (f"blocking top-{R}", pl.col("rank") <= R)) + tuple(
            (f"dense top-{k}", pl.col("drank") <= k) for k in (1, 3, K)) + (
            (f"blocking top-3 + dense top-{K}", (pl.col("rank") <= 3) | (pl.col("drank") <= K)),
            (f"blocking top-{R} + dense top-{K}", (pl.col("rank") <= R) | (pl.col("drank") <= K))):
        print(f"{name:32s} {t.filter(e.fill_null(False)).height / n * 100:6.2f}% of {n:,} true pairs")
    first = pl.read_parquet(SMALL / "val_scored.parquet", columns=["o", "ce"])
    vo = val_os()
    extra = sum(new_pairs(f, "train", vo, first).height for f in cand_chunks("train"))
    print(f"extra pairs to score on validation: {extra:,} for {vo.len():,} o records")


def score_new(split):
    tok = AutoTokenizer.from_pretrained(SMALL / "model")
    model = AutoModelForSequenceClassification.from_pretrained(SMALL / "model").to(DEV)
    if DEV == "cuda":
        model = model.half()
    recs = pl.read_parquet(WORK / f"{split}.parquet", columns=["name", "addr"]).with_row_index("idx")
    txt = text(recs)["t"]
    del recs
    keep = first = None
    if split == "train":
        keep, first = val_os(), pl.read_parquet(SMALL / "val_scored.parquet", columns=["o", "ce"])
    out = DEN / f"{split}_scored"
    out.mkdir(exist_ok=True)
    for f in cand_chunks(split):
        if (out / f.name).exists():
            continue
        p = new_pairs(f, split, keep, first)
        ce = score(model, tok, txt.gather(p["o"]).to_list(), txt.gather(p["s"]).to_list(), DEV)
        p.with_columns(ce=pl.Series(ce)).write_parquet(out / f.name)
        print(f.name, p.height, flush=True)


def override(sc, tr):
    """Replace the scores of the records in tr (translit.py output) with tr's."""
    return pl.concat([sc.filter(~pl.col("o").is_in(tr["o"].unique().implode())), tr.select(sc.columns)])


def final(variant="dense"):
    """OUT (CE_DIR) holds the scores the new pairs are merged into; existing scores win on overlap.
    variant "translit" also replaces the Indian-script records' scores with translit.py's."""
    tr_dir = WORK / "translit"
    if variant == "translit":
        assert (tr_dir / "train" / "DONE").exists() and (tr_dir / "test" / "DONE").exists(), "run translit.py first"
    val_s, truth = val_truth()
    old = pl.read_parquet(OUT / "val_scored.parquet", columns=["o", "s", "ce"]).with_columns(pl.col("o", "s").cast(pl.UInt32))
    first, vo = pl.read_parquet(SMALL / "val_scored.parquet", columns=["o", "ce"]), val_os()
    rule = pl.concat(new_pairs(f, "train", vo, first) for f in cand_chunks("train"))  # same selection as test
    new = pl.read_parquet(DEN / "train_scored" / "*.parquet").join(rule, on=["o", "s"], how="semi")
    sc = pl.concat([old, new]).unique(["o", "s"], keep="first")
    if variant == "translit":
        sc = override(sc, pl.read_parquet(tr_dir / "train" / "*.parquet"))
    best = best_per_o(sc.select("o", "s", p="ce"))
    base = json.loads((OUT / "thr.json").read_text())
    res = {(t, m): val_score(best, val_s, truth, t, m)
           for t in np.arange(0.30, 0.96, 0.05).round(2) for m in (0.0, 0.1, 0.2, 0.3, 0.5)}
    t, m = max(res, key=res.get)
    print(f"{OUT.name} alone: val {base['val_f05']:.4f}; {variant}: best thr {t} margin {m} -> "
          f"val {res[t, m]:.4f}; at strict {STRICT}: {res[STRICT, m]:.4f}", flush=True)
    (DEN / f"thr_{variant}.json").write_text(json.dumps({"ce_dir": OUT.name, "thr": float(t), "margin": m, "val_f05": res[t, m]}))
    recs = pl.read_parquet(WORK / "test.parquet", columns=["entity_id", "src", "country"]).with_row_index("idx")
    tr = pl.read_parquet(tr_dir / "test" / "*.parquet") if variant == "translit" else None
    best, merged = [], WORK / "merged" / variant
    for f in cand_chunks("test"):
        sc = pl.concat([pl.read_parquet(OUT / "test_scored" / f.name).with_columns(pl.col("o", "s").cast(pl.UInt32)),
                        pl.read_parquet(DEN / "test_scored" / f.name)]).unique(["o", "s"], keep="first")
        if tr is not None:
            sc = override(sc, tr.filter(pl.col("o").is_in(sc["o"].unique().implode())))
        merged.mkdir(parents=True, exist_ok=True)
        sc.select("o", "s", "ce").write_parquet(merged / f.name)  # input of stack.py
        best.append(best_per_o(sc.select("o", "s", p="ce")))
    best = pl.concat(best)
    france = recs.filter(pl.col("country") == "France")["idx"].cast(pl.UInt32)
    out = ROOT / "output"
    for name, thr, thr_fr in ((f"matching_results_{variant}.tsv", t, t), (f"matching_results_{variant}_strict.tsv", STRICT, STRICT),
                              (f"matching_results_{variant}_strict_fr.tsv", STRICT, FR_THR)):
        acc = pl.concat([accept(best.filter(~pl.col("o").is_in(france.implode())), thr, m),
                         accept(best.filter(pl.col("o").is_in(france.implode())), thr_fr, m)])
        write_lists(acc.lazy().select("o", "s"), recs, "matched_entity_ids", out / name)
        print(f"{name}: thr {thr} (France {thr_fr}), margin {m}, {(out / name).stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"search": lambda: search(sys.argv[2]), "recall": recall, "score": lambda: score_new(sys.argv[2]),
     "final": lambda: final(*sys.argv[2:])}[cmd]()
