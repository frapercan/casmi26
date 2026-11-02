"""Validate the full predictor on a class-1 + class-2 mix and tune fusion weights.

class 1: enveda-np-examples molecules (timsTOF); reference library = everything
         except np-examples and holdout spectra (answers exist in other libraries).
class 2: holdout molecules; their spectra are removed from the library and the
         answer is injected into the external-DB candidate pool.
Dumps per-candidate tables to <out>/cands.parquet and prints tuned weights.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.plm import DEFAULT_PLM, PLMEmbedder  # noqa: E402
from casmi.predict import DEFAULT_W, CandidateDB, MolFeaturizer, candidate_table, fuse, load_model  # noqa: E402
from casmi.search import Library  # noqa: E402
from casmi.spectra import neutral_mass  # noqa: E402


class Fast:
    """All candidate tables stacked into numpy arrays for fast weight search."""

    def __init__(self, tables):
        tables = [t for t in tables if len(t)]
        self.n = len(tables)
        self.g = np.concatenate([np.full(len(t), i) for i, t in enumerate(tables)])
        cat = pd.concat(tables, ignore_index=True)
        self.cols = {k: cat[k].to_numpy(np.float64) for k in ("nce", "jepa", "fp", "sae", "L") if k in cat}
        self.in_lib = cat.in_lib.to_numpy(np.float64)
        self.lpop = np.log1p(cat.pcl.to_numpy(np.float64))
        self.true = cat.is_true.to_numpy()
        self.Lmax = np.zeros(self.n); np.maximum.at(self.Lmax, self.g, self.cols["L"])

    def mrr(self, w):
        c = self.cols
        s = sum(w.get(k, 0) * c[k] for k in ("nce", "jepa", "fp", "sae") if k in c)
        s = s + w["L"] * c["L"] + w["lib"] * self.in_lib + w["pop"] * self.lpop
        s = s + (self.Lmax[self.g] >= w["tau"]) * w["boost"] * c["L"] * self.in_lib
        ts = np.full(self.n, np.nan); ts[self.g[self.true]] = s[self.true]  # one true per table
        above = np.zeros(self.n); np.add.at(above, self.g, s > ts[self.g])
        r = above + 1
        rr = np.where(np.isnan(ts) | (r > 25), 0.0, 1 / r)
        return float(rr.sum() / self.n)


def mrr_of(tables, w):
    return Fast(tables).mrr(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/home/user/work")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sae", default="")
    ap.add_argument("--n-class2", type=int, default=400)
    ap.add_argument("--n-class1", type=int, default=0, help="limit np-examples molecules (0 = all)")
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--plm", default="", help="local PLM dir")
    ap.add_argument("--out", default="/home/user/work/fusion")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    full = Library.load(f"{a.work}/lib")
    hold = pd.read_parquet(f"{a.work}/holdout.parquet")
    hold_ik = set(hold.inchikey14)
    is_np = (full.spectra.ingest_lib == "enveda-np-examples").to_numpy()
    is_hold = full.spectra.inchikey14.isin(hold_ik).to_numpy()
    ref = full.subset(~is_np & ~is_hold)
    ref.build_index()
    db = CandidateDB(f"{a.work}/candidates.parquet")
    model, sae = load_model(a.ckpt, a.sae, dev)
    mp = f"{a.work}/emb/plm_mean.npy"
    mean = np.load(mp) if os.path.exists(mp) else None
    feat = MolFeaturizer(PLMEmbedder(a.plm or DEFAULT_PLM, device=dev, mean=mean), dev)

    def queries(mask):
        q = full.spectra[mask].copy()
        idx = np.flatnonzero(mask)
        q["mz"] = [full.mz[full.off[i]:full.off[i + 1]].astype(np.float64) for i in idx]
        q["it"] = [full.it[full.off[i]:full.off[i + 1]].astype(np.float64) for i in idx]
        q["mass"] = [neutral_mass(p, ad) for p, ad in zip(q.precursor_mz, q.adduct)]
        return q

    tabs, t0 = [], time.time()
    smi = hold.set_index("inchikey14").smiles
    if a.n_class1:
        keep = set(pd.unique(full.spectra.inchikey14[is_np])[:a.n_class1])
        is_np = is_np & full.spectra.inchikey14.isin(keep).to_numpy()
    c2 = set(hold.inchikey14.iloc[:a.n_class2])
    for cls, mask in (("1", is_np), ("2", is_hold & full.spectra.inchikey14.isin(c2).to_numpy())):
        for ik, g in queries(mask).groupby("inchikey14"):
            t = candidate_table(g, ref, db, feat, model, sae, dev, a.ppm,
                                extra=(ik, smi[ik]) if cls == "2" else None)
            t["is_true"] = t.ik == ik
            t["qid"], t["cls"] = ik, cls
            tabs.append(t)
        print(f"class {cls} done  {time.time()-t0:.0f}s", flush=True)
    allc = pd.concat(tabs, ignore_index=True)
    allc.drop(columns=["smiles"]).to_parquet(f"{a.out}/cands.parquet", index=False)

    by = {c: Fast([t for t in tabs if len(t) and t.cls.iloc[0] == c]) for c in ("1", "2")}
    def report(w, tag):
        m1, m2 = by["1"].mrr(w), by["2"].mrr(w)
        print(f"{tag}: class1 {m1:.4f}  class2 {m2:.4f}  mix50/50 {(m1+m2)/2:.4f}", flush=True)
        return (m1 + m2) / 2
    report(DEFAULT_W, "default")
    heads = ("nce", "jepa", "fp", "sae")
    for k in heads:
        if k in allc:
            report({**DEFAULT_W, **{x: 0 for x in heads}, k: 1.0}, f"only {k}")
    best, bw = -1, None
    grid = {"nce": [0.5, 1], "jepa": [0, 0.5, 1], "fp": [0, 0.5], "sae": [0, 0.5, 1] if "sae" in allc else [0],
            "L": [2, 4, 8], "lib": [0, -1, -2, -4, -8], "pop": [0, 0.1, 0.3], "tau": [0.6, 0.7, 0.8]}
    for vals in itertools.product(*grid.values()):
        w = {**DEFAULT_W, **dict(zip(grid, vals))}
        m = (by["1"].mrr(w) + by["2"].mrr(w)) / 2
        if m > best:
            best, bw = m, w
    report(bw, "tuned")
    print("weights", json.dumps(bw), flush=True)
    report({**bw, "lib": 0}, "tuned w/o lib penalty")
    json.dump(bw, open(f"{a.out}/weights.json", "w"), indent=1)


if __name__ == "__main__":
    main()
