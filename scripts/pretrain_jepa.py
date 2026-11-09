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
from casmi.gpustore import GPUStore  # noqa: E402
from casmi.valset import excluded_spectra, load_exclude, load_exclude_all  # noqa: E402
from casmi.jepa import JEPA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/home/user/work")
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--d", type=int, default=256)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--out", default="/home/user/work/runs/jepa")
ap.add_argument("--no-gpu-data", action="store_true", help="collate on CPU instead of GPUStore")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--val", default="", help="validation dir from make_valset.py (its spectra are hidden)")
ap.add_argument("--query", default="mz", choices=["mz", "none"], help="mz = v2 masked queries with m/z; none = v1 (buggy)")
ap.add_argument("--ctx-free", action="store_true", help="shortcut control: predictor sees queries but no context")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)

spec_cols = pd.read_parquet(f"{a.work}/lib/spectra.parquet", columns=["ingest_lib", "inchikey14"])
val_ex = excluded_spectra(spec_cols, load_exclude(a.val), load_exclude_all(a.val))
print(f"validation set {a.val or '-'}: {val_ex.sum()} spectra hidden from training", flush=True)
st = SpectraStore(f"{a.work}/lib", f"{a.work}/emb", exclude=val_ex)
hold = set(pd.read_parquet(f"{a.work}/holdout.parquet").inchikey14)
hold |= set(st.spec.loc[st.spec.ingest_lib == "enveda-np-examples", "inchikey14"])
pool = np.flatnonzero(~st.spec.inchikey14.isin(hold).to_numpy() & st.spec.mol_row.notna().to_numpy())
print(f"pretraining spectra {len(pool)}  device {dev}", flush=True)

m = JEPA(d=a.d, layers=a.layers, query=a.query, ctx_free=a.ctx_free).to(dev)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.04)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=max(0.05, 3 / a.steps))
rng = np.random.default_rng(0)
gpu_data = dev == "cuda" and not a.no_gpu_data
if gpu_data:
    gs = GPUStore(st, dev)
    gpool = torch.as_tensor(pool, device=dev)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    next_batch = lambda: gs.collate(gs.sample_spectra(a.batch, gen, gpool))
else:  # with replacement: O(batch), not a 1.9M-element permutation per step
    next_batch = lambda: st.collate(pool[rng.integers(len(pool), size=a.batch)], device=dev)
loss_fn = torch.compile(m.loss) if a.compile else m.loss
amp = dev == "cuda"
amp_dt = torch.bfloat16 if amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
scaler = torch.amp.GradScaler(enabled=amp and amp_dt == torch.float16)
t0 = t_log = time.time()
for step in range(1, a.steps + 1):
    b = next_batch()
    with torch.autocast(dev, dtype=amp_dt, enabled=amp):
        loss, info = loss_fn(b)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    m.update_target(0.996 + (1 - 0.996) * step / a.steps)  # EMA momentum -> 1
    if step % 200 == 0:
        print(f"step {step}/{a.steps} {time.time()-t0:.0f}s loss={loss.item():.4f} {200/(time.time()-t_log):.1f} it/s " +
              " ".join(f"{k}={float(v):.3f}" for k, v in info.items()), flush=True)
        t_log = time.time()
    if step % 5000 == 0 or step == a.steps:
        torch.save(m.tgt.state_dict(), f"{a.out}/encoder.pt")  # EMA target is the smoother encoder
print("saved", f"{a.out}/encoder.pt")
