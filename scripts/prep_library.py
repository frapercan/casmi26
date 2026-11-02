"""Stream train.parquet into a compact spectral library.

Outputs (in --out):
  molecules.parquet : one row per inchikey14 (smiles, formula, exact mass)
  spectra.parquet   : per-spectrum metadata (row-aligned with peaks offsets)
  peaks.npz         : flattened cleaned peaks (mz, intensity) + offsets
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.spectra import TEST_ADDUCTS, clean_peaks  # noqa: E402

META = ["ingest_lib", "inchikey14", "normalized_smiles", "molecular_formula",
        "ionization_mode", "adduct", "precursor_mz", "precursor_error_ppm",
        "collision_energy_ev", "base_peak_intensity"]


def exact_mass(smi: str) -> float:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    return float(ExactMolWt(m)) if m is not None else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/home/user/data/train.parquet")
    ap.add_argument("--out", default="/home/user/work/lib")
    ap.add_argument("--max-ppm", type=float, default=20.0)
    ap.add_argument("--min-peaks", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pf = pq.ParquetFile(a.train)
    metas, mzs, its, offs = [], [], [], [0]
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=META + ["ms2_mzs", "ms2_normalized_intensities"]).to_pandas()
        ok = t.adduct.isin(TEST_ADDUCTS) & (t.precursor_error_ppm.abs() <= a.max_ppm)
        t = t[ok]
        keep = np.zeros(len(t), dtype=bool)
        for i, (mz, it, pm) in enumerate(zip(t.ms2_mzs, t.ms2_normalized_intensities, t.precursor_mz)):
            cm, ci = clean_peaks(mz, it, pm, top_n=a.top_n)
            if cm.size >= a.min_peaks:
                keep[i] = True
                mzs.append(cm); its.append(ci); offs.append(offs[-1] + cm.size)
        m = t.loc[keep, META].copy()
        m["ce_mean"] = m.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
        metas.append(m.drop(columns=["collision_energy_ev"]))
        print(f"rg {rg+1}/{pf.num_row_groups}: kept {keep.sum()}/{len(keep)}", flush=True)
        del t

    meta = pd.concat(metas, ignore_index=True)
    mol = (meta.groupby("inchikey14", as_index=False)
               .agg(smiles=("normalized_smiles", "first"), formula=("molecular_formula", "first")))
    with Pool(os.cpu_count()) as p:
        mol["exact_mass"] = p.map(exact_mass, mol.smiles, chunksize=2000)
    mol = mol.dropna(subset=["exact_mass"]).reset_index(drop=True)
    mol.to_parquet(f"{a.out}/molecules.parquet", index=False)

    meta = meta.drop(columns=["normalized_smiles", "molecular_formula"])
    meta.to_parquet(f"{a.out}/spectra.parquet", index=False)
    np.savez(f"{a.out}/peaks.npz", mz=np.concatenate(mzs), it=np.concatenate(its),
             off=np.asarray(offs, dtype=np.int64))
    print(f"spectra {len(meta)}  molecules {len(mol)}  peaks {offs[-1]}")


if __name__ == "__main__":
    main()
