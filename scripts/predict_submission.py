"""Full predictor (library search + model over DB candidates) -> submission.csv."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.plm import PLMEmbedder  # noqa: E402
from casmi.predict import DEFAULT_W, CandidateDB, MolFeaturizer, candidate_table, fuse, load_model  # noqa: E402
from casmi.search import Library, prep_query  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/home/user/work")
ap.add_argument("--test", default="/home/user/data/test.parquet")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--sae", default="")
ap.add_argument("--weights", default="")
ap.add_argument("--plm", default="", help="local PLM dir (offline Kaggle)")
ap.add_argument("--ppm", type=float, default=10.0)
ap.add_argument("--out", default="submission.csv")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
w = {**DEFAULT_W, **(json.load(open(a.weights)) if a.weights else {})}

lib = Library.load(f"{a.work}/lib"); lib.build_index()
db = CandidateDB(f"{a.work}/candidates.parquet")
model, sae = load_model(a.ckpt, a.sae, dev)
feat = MolFeaturizer(PLMEmbedder(a.plm, device=dev) if a.plm else PLMEmbedder(device=dev), dev)
test = pd.read_parquet(a.test)
q = prep_query(test)
q["ce_mean"] = test.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)

rows, t0 = [], time.time()
for n, (mid, g) in enumerate(q.groupby("molecule_id", sort=False)):
    t = candidate_table(g, lib, db, feat, model, sae, dev, a.ppm)
    smis = []
    if len(t):
        t["s"] = fuse(t, w)
        smis = t.sort_values("s", ascending=False).drop_duplicates("ik").smiles.head(25).tolist()
    if not smis:  # nothing within ppm: nearest-mass DB structures
        m = float(np.median(g.mass))
        i = np.searchsorted(db.mass, m)
        smis = db.smiles[max(0, i - 12):i + 13].tolist()
    rows.append((mid, ";".join(smis[:25])))
    if (n + 1) % 50 == 0:
        print(f"{n+1} molecules  {time.time()-t0:.0f}s", flush=True)
sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
assert sub.smiles.str.len().gt(0).all() and sub.molecule_id.is_unique
sub.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(sub)} molecules in {time.time()-t0:.0f}s")
