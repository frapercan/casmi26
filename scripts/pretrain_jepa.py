"""JEPA pretraining of the spectrum encoder on library spectra (labels unused).

Held-out and np-examples molecules' spectra are excluded so evaluation stays
honest. Saves the context encoder to <out>/encoder.pt for train.py --init-encoder.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.data import SpectraStore  # noqa: E402
from casmi.jepa import JEPA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/home/user/work")
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--d", type=int, default=256)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--out", default="/home/user/work/runs/jepa")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

st = SpectraStore(f"{a.work}/lib", f"{a.work}/emb")
hold = set(pd.read_parquet(f"{a.work}/holdout.parquet").inchikey14)
hold |= set(st.spec.loc[st.spec.ingest_lib == "enveda-np-examples", "inchikey14"])
pool = np.flatnonzero(~st.spec.inchikey14.isin(hold).to_numpy() & st.spec.mol_row.notna().to_numpy())
print(f"pretraining spectra {len(pool)}  device {dev}", flush=True)

m = JEPA(d=a.d, layers=a.layers).to(dev)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.04)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=max(0.05, 3 / a.steps))
rng = np.random.default_rng(0)
amp = dev == "cuda"
amp_dt = torch.bfloat16 if amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
scaler = torch.amp.GradScaler(enabled=amp and amp_dt == torch.float16)
t0 = time.time()
for step in range(1, a.steps + 1):
    b = st.collate(rng.choice(pool, a.batch, replace=False), device=dev)
    with torch.autocast(dev, dtype=amp_dt, enabled=amp):
        loss, info = m.loss(b)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    m.update_target(0.996 + (1 - 0.996) * step / a.steps)  # EMA momentum -> 1
    if step % 200 == 0:
        print(f"step {step}/{a.steps} {time.time()-t0:.0f}s loss={loss.item():.4f} " +
              " ".join(f"{k}={v:.3f}" for k, v in info.items()), flush=True)
    if step % 5000 == 0 or step == a.steps:
        torch.save(m.tgt.state_dict(), f"{a.out}/encoder.pt")  # EMA target is the smoother encoder
print("saved", f"{a.out}/encoder.pt")
