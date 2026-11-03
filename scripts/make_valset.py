"""Build a validation set shaped like the hidden test.

Hidden test: ~400 molecules, Enveda timsTOF spectra (1-9 per molecule), electron-mass
corrected in positive mode, novelty classes 1/2/3. The train library `enveda-180`
is the same instrument and source, so queries come from it:

  class 1  molecule also has spectra in other libraries -> the library can find it
  class 2  molecule only in enveda-180 -> once its enveda spectra are removed it is
           absent from the library; evaluation injects the answer into the DB pool
  class 3  no need to simulate: the answer is unreachable, so it always scores 0
           (handled analytically by the class mix)

Writes <out>/queries.parquet (test.parquet columns), answers.parquet
(molecule_id, inchikey14, smiles, cls, mass) and exclude_ik.txt: the molecules whose
enveda-180 spectra must be removed from library AND training (SpectraStore exclude).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ELECTRON = 0.000548  # Da; applied to the hidden test only (positive mode), per the hosts
# spectra per molecule in the visible test (same size as the hidden one)
SPEC_PER_MOL = {1: 55, 2: 97, 3: 99, 4: 106, 5: 24, 6: 14, 7: 2, 8: 2, 9: 1}
TEST_COLS = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities", "base_peak_intensity",
             "adduct", "ionization_mode", "instrument_type", "precursor_mz", "collision_energy_orig",
             "collision_energy_ev", "collision_energy_orig_units"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/home/user/data/train.parquet")
    ap.add_argument("--work", default="/home/user/work")
    ap.add_argument("--out", default="")
    ap.add_argument("--n1", type=int, default=400, help="class-1 molecules")
    ap.add_argument("--n2", type=int, default=400, help="class-2 molecules")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or f"{a.work}/val"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    d = ds.dataset(a.train)
    meta = d.to_table(columns=["inchikey14", "ingest_lib"]).to_pandas()
    lib_ik = set(pd.read_parquet(f"{a.work}/lib/spectra.parquet", columns=["inchikey14"]).inchikey14)
    is_env = meta.ingest_lib == "enveda-180"
    env_ik = set(meta.inchikey14[is_env])
    other_ik = set(meta.inchikey14[~meta.ingest_lib.isin(["enveda-180", "enveda-np-examples"])])
    taboo = set(meta.inchikey14[meta.ingest_lib == "enveda-np-examples"])  # class-1 benchmark, keep apart
    hold = f"{a.work}/holdout.parquet"
    if os.path.exists(hold):
        taboo |= set(pd.read_parquet(hold).inchikey14)  # old class-2 holdout, keep apart
    cand = (env_ik & lib_ik) - taboo  # must survive prep_library (test adducts, clean peaks)
    pool1 = np.array(sorted(cand & other_ik))
    pool2 = np.array(sorted(cand - other_ik))
    print(f"enveda-180 molecules {len(env_ik)}: class-1 pool {len(pool1)}, class-2 pool {len(pool2)}", flush=True)
    pick = {1: rng.choice(pool1, min(a.n1, len(pool1)), replace=False),
            2: rng.choice(pool2, min(a.n2, len(pool2)), replace=False)}
    chosen = set(pick[1]) | set(pick[2])

    t = d.to_table(filter=(ds.field("ingest_lib") == "enveda-180") & ds.field("inchikey14").isin(list(chosen))).to_pandas()
    t = t[t.adduct.isin(pd.read_parquet(f"{a.work}/lib/spectra.parquet", columns=["adduct"]).adduct.unique())]
    t = t[t.ms2_mzs.map(len) >= 3]
    k_vals = np.array(list(SPEC_PER_MOL)); k_p = np.array(list(SPEC_PER_MOL.values()), float)
    k_p /= k_p.sum()
    qs, ans = [], []
    for cls in (1, 2):
        for i, ik in enumerate(pick[cls]):
            g = t[t.inchikey14 == ik]
            if g.empty:
                continue
            k = min(int(rng.choice(k_vals, p=k_p)), len(g))
            g = g.iloc[rng.choice(len(g), k, replace=False)].copy()
            mid = f"val{cls}_{i:04d}"
            g["molecule_id"] = mid
            qs.append(g)
            ans.append((mid, ik, g.normalized_smiles.iloc[0], cls))
    q = pd.concat(qs, ignore_index=True)
    q["spectrum_id"] = [f"vs{i:06d}" for i in range(len(q))]
    pos = q.ionization_mode == "positive"
    q.loc[pos, "precursor_mz"] = q.loc[pos, "precursor_mz"] - ELECTRON
    q.loc[pos, "ms2_mzs"] = q.loc[pos, "ms2_mzs"].map(lambda m: (np.asarray(m) - ELECTRON).tolist())
    q["instrument_type"] = "timsTOF"
    q[TEST_COLS].to_parquet(f"{out}/queries.parquet", index=False)
    A = pd.DataFrame(ans, columns=["molecule_id", "inchikey14", "smiles", "cls"])
    A.to_parquet(f"{out}/answers.parquet", index=False)
    with open(f"{out}/exclude_ik.txt", "w") as f:
        f.write("\n".join(sorted(chosen)))
    print(f"wrote {len(A)} molecules ({(A.cls == 1).sum()} class-1, {(A.cls == 2).sum()} class-2), "
          f"{len(q)} spectra -> {out}", flush=True)
    print("spectra/molecule:", q.groupby("molecule_id").size().value_counts().sort_index().to_dict())
    print("adducts:", q.adduct.value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
