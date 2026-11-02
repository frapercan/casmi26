"""Class-2 proxy: rank mass-matched candidates with no spectral model.

Pool = external candidate DB (train molecules excluded) + held-out molecules.
Reports candidate-count stats and MRR@25 for simple priors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PPM = 10.0
db = pq.read_table("/home/user/work/candidates.parquet",
                   columns=["inchikey14", "exact_mass", "in_coconut", "pcl_entries"]).to_pandas()
hold = pd.read_parquet("/home/user/work/holdout.parquet")
pcl = pd.read_parquet("/home/user/data/cands/pcl.parquet", columns=["key", "n_entries"])
hold["pcl_entries"] = hold.inchikey14.map(pcl.set_index("key").n_entries).fillna(0)

mass = db.exact_mass.to_numpy()
pop = db.pcl_entries.to_numpy().astype(float)
coco = db.in_coconut.to_numpy()
rng = np.random.default_rng(0)

ncand, rr = [], {"random": [], "popularity": [], "coconut>pop": []}
for m, hp in zip(hold.exact_mass, hold.pcl_entries):
    tol = max(m * PPM * 1e-6, 0.003)
    lo, hi = np.searchsorted(mass, m - tol), np.searchsorted(mass, m + tol, side="right")
    n = hi - lo + 1  # + the true molecule
    ncand.append(n)
    # rank of the true molecule (index n-1) under each scoring; ties -> average
    def rank(score_c, score_t):
        better = (score_c > score_t).sum()
        ties = (score_c == score_t).sum()
        return better + 1 + ties / 2
    r = {
        "random": (n + 1) / 2,
        "popularity": rank(pop[lo:hi], hp),
        "coconut>pop": rank(coco[lo:hi] * 1e9 + pop[lo:hi], hp),  # holdout not flagged coconut
    }
    for k, v in r.items():
        rr[k].append(1 / v if v <= 25 else 0.0)

nc = np.array(ncand)
print(f"candidates per molecule @{PPM:.0f}ppm: median {np.median(nc):.0f}  p10 {np.percentile(nc,10):.0f}  p90 {np.percentile(nc,90):.0f}")
print(f"molecules with <=25 candidates: {(nc <= 25).mean():.2%}")
for k, v in rr.items():
    print(f"{k:12s} MRR@25 = {np.mean(v):.4f}")
print(f"holdout molecules found in PubChemLite: {(hold.pcl_entries > 0).mean():.2%}")
