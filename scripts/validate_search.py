"""Validate library search on enveda-np-examples (timsTOF, same pipeline as test).

Queries: np-examples spectra grouped by inchikey14 (one query molecule each).
Library: every other spectrum (np-examples removed), so this measures the
"class 1" scenario: the molecule has public reference spectra from other labs.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.metric import reciprocal_rank_ik  # noqa: E402
from casmi.search import Library, search_molecule  # noqa: E402
from casmi.spectra import neutral_mass  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lib", default="/home/user/work/lib")
ap.add_argument("--ppm", type=float, default=10.0)
ap.add_argument("--tol", type=float, default=0.01)
ap.add_argument("--exclude-libs", default="", help="comma list of libs to drop from the reference side")
a = ap.parse_args()

full = Library.load(a.lib)
is_q = (full.spectra.ingest_lib == "enveda-np-examples").to_numpy()
excl = set(filter(None, a.exclude_libs.split(",")))
ref = full.subset(~is_q & ~full.spectra.ingest_lib.isin(excl).to_numpy())
ref.build_index()

qs = full.spectra[is_q].copy()
qi = np.flatnonzero(is_q)
qs["mz"] = [full.mz[full.off[i]:full.off[i + 1]].astype(np.float64) for i in qi]
qs["it"] = [full.it[full.off[i]:full.off[i + 1]].astype(np.float64) for i in qi]
qs["mass"] = [neutral_mass(p, ad) for p, ad in zip(qs.precursor_mz, qs.adduct)]

t0 = time.time()
rows = []
for ik, g in qs.groupby("inchikey14"):
    res = search_molecule(ref, g, ppm=a.ppm, tol=a.tol)
    rr = reciprocal_rank_ik([r[0] for r in res], ik)
    in_lib = ik in ref._spec_by_mol
    rows.append((ik, len(g), len(res), in_lib, rr))
r = pd.DataFrame(rows, columns=["ik14", "n_spec", "n_cand", "in_lib", "rr"])
print(f"molecules {len(r)}  in reference lib {r.in_lib.mean():.2%}  time {time.time()-t0:.0f}s")
print(f"MRR@25 = {r.rr.mean():.4f}   top1 = {(r.rr == 1).mean():.3f}   hit@25 = {(r.rr > 0).mean():.3f}")
print("median #mass candidates", r.n_cand.median())
