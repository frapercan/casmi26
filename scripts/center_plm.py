"""Centre PLM embeddings: x <- normalize(x - mean(train molecules)).

Writes <out>/plm.npy and <out>/plm_mean.npy; the mean is reused at prediction
time (PLMEmbedder(mean=...)) so candidates get the identical transform.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--emb", default="/home/user/work/emb")
ap.add_argument("--out", default="")
a = ap.parse_args()
out = a.out or a.emb
x = np.load(f"{a.emb}/plm.npy").astype(np.float32)
mols = pd.read_parquet(f"{a.emb}/mol_table.parquet")
mean = x[(mols.source == "train").to_numpy()].mean(0)
rng = np.random.default_rng(0)
s = rng.choice(len(x), 2000, replace=False)
print(f"mean cosine before: {(x[s] @ x[s].T).mean():.3f}", flush=True)
x -= mean
x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
print(f"mean cosine after:  {(x[s] @ x[s].T).mean():.3f}", flush=True)
os.makedirs(out, exist_ok=True)
dst = f"{out}/plm.npy"
if os.path.islink(dst):
    os.remove(dst)
np.save(dst, x.astype(np.float16))
np.save(f"{out}/plm_mean.npy", mean)
print("saved", dst)
