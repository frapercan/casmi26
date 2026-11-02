"""Library-search submission for test.parquet -> submission.csv."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.search import Library, prep_query, search_molecule  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lib", default="/home/user/work/lib")
ap.add_argument("--test", default="/home/user/data/test.parquet")
ap.add_argument("--out", default="submission.csv")
ap.add_argument("--ppm", type=float, default=10.0)
a = ap.parse_args()

lib = Library.load(a.lib)
lib.build_index()
q = prep_query(pd.read_parquet(a.test))

rows = []
for mid, g in q.groupby("molecule_id", sort=False):
    res = search_molecule(lib, g, ppm=a.ppm)
    smis, seen = [r[1] for r in res], {r[0] for r in res}
    if len(smis) < 25:  # pad with nearest-mass library molecules
        mass = float(np.median(g.mass))
        for ci in np.argsort(np.abs(lib._mass - mass))[:200]:
            row = lib.molecules.iloc[ci]
            if row.inchikey14 not in seen:
                smis.append(row.smiles); seen.add(row.inchikey14)
            if len(smis) == 25:
                break
    rows.append((mid, ";".join(smis[:25])))
sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
sub.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(sub)} molecules")
