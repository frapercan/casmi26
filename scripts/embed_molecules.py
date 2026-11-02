"""Embed molecules with the frozen PLM and compute Morgan fingerprints.

Inputs: train molecules + holdout candidate pools (DB rows within --ppm of a
holdout mass). Output (in --out): mol_table.parquet (row order), plm.npy
(float16, L2-normalised), fp.npy (packed bits, 2048).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FP_BITS = 2048


def morgan(smi: str) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    out = np.zeros(FP_BITS // 8, np.uint8)
    if m is None:
        return out
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    return np.packbits(g.GetFingerprintAsNumPy(m).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/home/user/work")
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--out", default="/home/user/work/emb")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train = pd.read_parquet(f"{a.work}/lib/molecules.parquet", columns=["inchikey14", "smiles", "exact_mass"])
    train["source"] = "train"
    hold = pd.read_parquet(f"{a.work}/holdout.parquet")
    db = pq.read_table(f"{a.work}/candidates.parquet",
                       columns=["inchikey14", "smiles", "exact_mass"]).to_pandas()
    mass = db.exact_mass.to_numpy()
    keep = np.zeros(len(db), bool)
    for m in hold.exact_mass:
        tol = max(m * a.ppm * 1e-6, 0.003)
        keep[np.searchsorted(mass, m - tol):np.searchsorted(mass, m + tol, side="right")] = True
    cand = db[keep].copy()
    cand["source"] = "db"
    del db
    tab = pd.concat([train, cand], ignore_index=True)
    print(f"train {len(train)}  holdout-candidates {len(cand)}  total {len(tab)}", flush=True)
    tab.to_parquet(f"{a.out}/mol_table.parquet", index=False)

    t = time.time()
    with Pool(os.cpu_count()) as p:
        fp = np.stack(p.map(morgan, tab.smiles, chunksize=2000))
    np.save(f"{a.out}/fp.npy", fp)
    print(f"fingerprints {fp.shape} in {time.time()-t:.0f}s", flush=True)

    from casmi.plm import PLMEmbedder
    emb = PLMEmbedder()
    t = time.time()
    chunks = []
    step = 50_000
    for i in range(0, len(tab), step):
        chunks.append(emb.embed(tab.smiles.iloc[i:i + step].tolist()))
        print(f"plm {i+len(chunks[-1])}/{len(tab)}  {time.time()-t:.0f}s", flush=True)
    np.save(f"{a.out}/plm.npy", np.concatenate(chunks))


if __name__ == "__main__":
    main()
