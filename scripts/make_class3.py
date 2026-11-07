"""Build an honest class-3 cell: natural products whose structure the models never see.

Picks natural-product-like train molecules (RDKit NP-likeness score >= --np-min; the COCONUT copy we have
excludes train structures, and enveda-180 holds only synthetic Enamine compounds, so the queries come from
the other libraries: GNPS, MassBank, MoNA, ...), and writes:
  queries.parquet       their enveda-180 spectra, shaped like the hidden test (as make_valset.py)
  answers.parquet       molecule_id, inchikey14, smiles, cls=3, mass
  exclude_all_ik.txt    every spectrum of these molecules is hidden from the library and from training,
                        and the structures must be dropped from every candidate pool at evaluation
  exclude_ik.txt        copied from --base-val, so the class-1/2 validation stays hidden too

  python scripts/make_class3.py --out /home/user/work/val3 --n 500
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from make_valset import SPEC_PER_MOL, TEST_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/home/user/data/train.parquet")
    ap.add_argument("--work", default="/home/user/work")
    ap.add_argument("--base-val", default="/home/user/work/val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--np-min", type=float, default=1.0, help="RDKit NP-likeness threshold (natural products ~ >1)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    d = ds.dataset(a.train)
    meta = d.to_table(columns=["inchikey14", "ingest_lib"]).to_pandas()
    env_ik = set(meta.inchikey14[meta.ingest_lib == "enveda-180"])
    lib = pd.read_parquet(f"{a.work}/lib/spectra.parquet", columns=["inchikey14", "adduct"])
    taboo = set(open(f"{a.base_val}/exclude_ik.txt").read().split()) if os.path.exists(f"{a.base_val}/exclude_ik.txt") else set()
    taboo |= set(meta.inchikey14[meta.ingest_lib == "enveda-np-examples"])
    if os.path.exists(f"{a.work}/holdout.parquet"):
        taboo |= set(pd.read_parquet(f"{a.work}/holdout.parquet").inchikey14)
    import gzip, pickle, sys as _s
    from rdkit import Chem, RDLogger
    from rdkit.Chem import RDConfig
    RDLogger.DisableLog("rdApp.*")
    _s.path.append(os.path.join(RDConfig.RDContribDir, "NP_Score"))
    import npscorer
    npm = pickle.load(gzip.open(os.path.join(RDConfig.RDContribDir, "NP_Score", "publicnp.model.gz")))
    mols = pd.read_parquet(f"{a.work}/lib/molecules.parquet")
    smi_col = "smiles" if "smiles" in mols else [c for c in mols if "smiles" in c][0]
    cand = mols[mols.inchikey14.isin(set(lib.inchikey14) - taboo - env_ik)]
    cand = cand.sample(min(len(cand), 40 * a.n), random_state=a.seed)
    def nps(smi):
        m = Chem.MolFromSmiles(smi)
        return npscorer.scoreMol(m, npm) if m is not None else -9
    cand = cand.assign(np_score=cand[smi_col].map(nps))
    pool = np.array(sorted(cand.inchikey14[cand.np_score >= a.np_min]))
    print(f"NP-like (score >= {a.np_min}) library molecules: {len(pool)} of {len(cand)} scored", flush=True)
    pick = rng.choice(pool, min(a.n, len(pool)), replace=False)
    t = d.to_table(filter=ds.field("inchikey14").isin(list(pick))).to_pandas()
    t = t[t.adduct.isin(["[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+"]) & (t.ms2_mzs.map(len) >= 3)]
    k_vals = np.array(list(SPEC_PER_MOL)); k_p = np.array(list(SPEC_PER_MOL.values()), float); k_p /= k_p.sum()
    qs, ans = [], []
    for i, ik in enumerate(pick):
        g = t[t.inchikey14 == ik]
        if g.empty:
            continue
        k = min(int(rng.choice(k_vals, p=k_p)), len(g))
        g = g.iloc[rng.choice(len(g), k, replace=False)].copy()
        mid = f"val3_{i:04d}"
        g["molecule_id"] = mid
        qs.append(g)
        ans.append((mid, ik, g.normalized_smiles.iloc[0], 3))
    q = pd.concat(qs, ignore_index=True)
    q["spectrum_id"] = [f"v3s{i:06d}" for i in range(len(q))]
    pos = q.ionization_mode == "positive"
    q[TEST_COLS].to_parquet(f"{a.out}/queries.parquet", index=False)
    A = pd.DataFrame(ans, columns=["molecule_id", "inchikey14", "smiles", "cls"])
    A.to_parquet(f"{a.out}/answers.parquet", index=False)
    open(f"{a.out}/exclude_all_ik.txt", "w").write("\n".join(sorted(A.inchikey14)))
    if os.path.exists(f"{a.base_val}/exclude_ik.txt"):
        shutil.copyfile(f"{a.base_val}/exclude_ik.txt", f"{a.out}/exclude_ik.txt")
    print(f"wrote {len(A)} class-3 molecules, {len(q)} spectra -> {a.out}")
    print("adducts:", q.adduct.value_counts(normalize=True).round(3).to_dict())
    print("instruments:", q.instrument_type.value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
