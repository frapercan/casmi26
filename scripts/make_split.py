"""Hold out natural-product-like molecules to simulate novelty classes 2/3.

Held-out molecules are removed from the reference library AND from model
training (by InChIKey14). Their spectra become queries. Each one is labelled
by whether it is reachable in the candidate DB (class-2 proxy) or not
(class-3 proxy).

Output: <out>/holdout.parquet  (inchikey14, smiles, exact_mass, in_db, in_coconut)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

NP_LIBS = ["riken", "gnps", "massbank", "mona", "msdial", "spectraverse", "masaryk"]

ap = argparse.ArgumentParser()
ap.add_argument("--lib", default="/home/user/work/lib")
ap.add_argument("--db", default="/home/user/work/candidates.parquet")
ap.add_argument("--out", default="/home/user/work")
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

spec = pd.read_parquet(a.lib + "/spectra.parquet", columns=["inchikey14", "ingest_lib"])
mol = pd.read_parquet(a.lib + "/molecules.parquet")
db = pd.read_parquet(a.db, columns=["inchikey14", "in_coconut"])

libs = spec.groupby("inchikey14").ingest_lib.agg(set)
# NP-like: measured only in NP-oriented libraries (never in enveda-180 / drug sets),
# and not one of the np-examples (kept for the class-1 timsTOF validation)
ok = libs.map(lambda s: s <= set(NP_LIBS))
pool = mol[mol.inchikey14.isin(ok[ok].index)]
pool = pool[(pool.exact_mass >= 150) & (pool.exact_mass <= 1200)]
pool = pool.merge(db, on="inchikey14", how="left")
pool["in_db"] = pool.in_coconut.notna()
pool["in_coconut"] = pool.in_coconut.fillna(False).astype(bool)

rng = np.random.default_rng(a.seed)
hold = pool.iloc[rng.choice(len(pool), size=min(a.n, len(pool)), replace=False)]
hold = hold[["inchikey14", "smiles", "exact_mass", "in_db", "in_coconut"]].reset_index(drop=True)
hold.to_parquet(a.out + "/holdout.parquet", index=False)
print(f"pool {len(pool)}  holdout {len(hold)}")
print(f"in candidate DB: {hold.in_db.mean():.2%}   in COCONUT: {hold.in_coconut.mean():.2%}")
