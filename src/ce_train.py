"""Fine-tune a small multilingual text model as a cross-encoder: it reads (S2/S3 record, S1 candidate) together
and outputs a match score. Trains on work/ce/train.parquet, then scores work/ce/val.parquet.
Model: intfloat/multilingual-e5-small (MIT license, 118M parameters).
Usage: python src/ce_train.py [--no-val]  -> work/ce/model/, work/ce/val_scored.parquet
Env: CE_DIR (output folder under work/), CE_BASE, CE_BATCH, CE_LR, CE_FREEZE_EMB=1.
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ce_data import OUT

BASE = os.environ.get("CE_BASE", "intfloat/multilingual-e5-small")
MAX_LEN = 96
BATCH = int(os.environ.get("CE_BATCH", 128))
LR = float(os.environ.get("CE_LR", 5e-5))
EPOCHS = 1


def batches(a, b, n, shuffle, seed=0):
    idx = np.random.default_rng(seed).permutation(len(a)) if shuffle else np.arange(len(a))
    for i in range(0, len(idx), n):
        j = idx[i:i + n]
        yield j, [a[k] for k in j], [b[k] for k in j]


def score(model, tok, a, b, dev, n=512):
    """Match scores (probabilities) for text pairs, in input order."""
    model.eval()
    out = np.empty(len(a), dtype=np.float32)
    # batches of similar length waste less compute on padding
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        for i in range(0, len(order), n):
            j = order[i:i + n]
            enc = tok([a[k] for k in j], [b[k] for k in j], truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to(dev)
            out[j] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tr = pl.read_parquet(OUT / "train.parquet")
    a, b, y = tr["a"].to_list(), tr["b"].to_list(), tr["y"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(dev)
    if os.environ.get("CE_FREEZE_EMB") == "1":  # the word embeddings are most of the weights; frozen, a bigger model fits in 6 GB
        model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
    steps = EPOCHS * -(-len(a) // BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print(f"device {dev}; {len(a):,} training pairs, {y.mean():.3f} positive, {steps:,} steps", flush=True)
    t0, step, run = time.time(), 0, 0.0
    model.train()
    for epoch in range(EPOCHS):
        for j, xa, xb in batches(a, b, BATCH, shuffle=True, seed=epoch):
            enc = tok(xa, xb, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = loss_fn(model(**enc).logits.squeeze(-1).float(), torch.from_numpy(y[j]).to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 200 == 0 or step == steps:
                el = time.time() - t0
                print(f"step {step:,}/{steps:,}  loss {run:.4f}  {step * BATCH / el:,.0f} pairs/s  "
                      f"eta {(steps - step) * el / step / 60:.0f} min", flush=True)
    model.save_pretrained(OUT / "model")
    tok.save_pretrained(OUT / "model")
    if "--no-val" in sys.argv:
        return
    va = pl.read_parquet(OUT / "val.parquet")
    t1 = time.time()
    ce = score(model, tok, va["a"].to_list(), va["b"].to_list(), dev)
    print(f"validation: {va.height:,} pairs scored at {va.height / (time.time() - t1):,.0f} pairs/s", flush=True)
    va.select("o", "s", "rank", "y", ce=pl.Series(ce)).write_parquet(OUT / "val_scored.parquet")


if __name__ == "__main__":
    main()
