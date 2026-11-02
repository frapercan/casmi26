"""Train the top-k SAE on all PLM embeddings (train molecules + candidates)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.sae import train_sae  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/home/user/work")
ap.add_argument("--n-latent", type=int, default=4096)
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--out", default="/home/user/work/emb/sae.pt")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
x = torch.from_numpy(np.load(f"{a.work}/emb/plm.npy"))
sae = train_sae(x, a.n_latent, a.k, a.steps, device=dev, log=lambda s: print(s, flush=True))
torch.save({"state": sae.state_dict(), "n_latent": a.n_latent, "k": a.k, "d_in": x.shape[1]}, a.out)
print("saved", a.out)
