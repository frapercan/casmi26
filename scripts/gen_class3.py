"""Class-3 generator v0: propose structures that are in no database, starting from spectral analogs.

For each query molecule of a class-3 cell (make_class3.py):
  1. analogs: library molecules whose spectra look alike (binned fragment + neutral-loss cosine), with the
     cell's own molecules removed from the library;
  2. edits: one or two chemically plausible transformations of each analog (hydroxylation, methylation,
     glycosylation, reduction, ...), written as RDKit reaction SMARTS;
  3. formula filter: keep products whose exact mass is within --ppm of the query's neutral mass.
Reports how often the true structure is generated (recall) and how large the candidate sets are, plus the
MRR of a naive ranking (analog similarity), which the world model will have to beat.

  python scripts/gen_class3.py --cell /home/user/work/val3 --out /home/user/work/val3/gen
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator  # noqa: E402
from rdkit.DataStructs import BulkTanimotoSimilarity  # noqa: E402

from casmi.metric import ik14  # noqa: E402
from casmi.search import Library, prep_query  # noqa: E402

RDLogger.DisableLog("rdApp.*")
from casmi import gen3 as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--lib", default="/home/user/work/lib")
    ap.add_argument("--out", required=True)
    ap.add_argument("--analogs", type=int, default=15)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--max-mols", type=int, default=0)
    ap.add_argument("--nl-weight", type=float, default=0.5)
    ap.add_argument("--start", type=int, default=0, help="chunk of the cell's molecules (outputs get a _<start> suffix)")
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--edits", default="v0", choices=["v0", "v1"])
    ap.add_argument("--max-products", type=int, default=20)
    ap.add_argument("--no-prune", action="store_true", help="expand every edit (reference for the pruning check)")
    ap.add_argument("--sample", default="", help="comma-free file of molecule_ids to run on (overrides start/end)")
    a = ap.parse_args()
    G.set_edits(a.edits, a.max_products)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    A = pd.read_parquet(f"{a.cell}/answers.parquet")
    Q = pd.read_parquet(f"{a.cell}/queries.parquet")
    hide = set(open(f"{a.cell}/exclude_all_ik.txt").read().split())
    lib = Library.load(a.lib)
    lib = lib.subset(~lib.spectra.inchikey14.isin(hide).to_numpy())
    assert not (set(lib.molecules.inchikey14) & hide), "class-3 answers leaked into the library"
    print(f"library without the cell: {len(lib.spectra)} spectra, {len(lib.molecules)} molecules [{time.time()-t0:.0f}s]", flush=True)
    F, N = G.library_matrix(lib)
    print(f"binned library matrices [{time.time()-t0:.0f}s]", flush=True)
    if not a.no_prune:
        G.edit_deltas(lib.molecules.smiles.tolist())
        print(f"edit mass deltas known for {len(G.DELTA)}/{len(G.RXN)} edits (pruning {'on' if len(G.DELTA) == len(G.RXN) else 'off'})", flush=True)
    spec_ik = lib.spectra.inchikey14.to_numpy()
    spec_mode = lib.spectra.ionization_mode.to_numpy()
    smi_of = dict(zip(lib.molecules.inchikey14, lib.molecules.smiles))
    q = prep_query(Q)
    rows, stats = [], []
    mids = A.molecule_id.tolist()[: a.max_mols or None]
    if a.end:
        mids = mids[a.start:a.end]
    sfx = f"_{a.start:04d}" if a.end else ""
    if a.sample:
        keep = set(open(a.sample).read().split()); mids = [m for m in A.molecule_id if m in keep]; sfx = "_sample"
    for k, mid in enumerate(mids):
        g = q[q.molecule_id == mid]
        truth = A.set_index("molecule_id").inchikey14[mid]
        cands, parents = G.generate(g, F, N, spec_ik, spec_mode, smi_of, a.analogs, a.depth, a.ppm, a.nl_weight, not a.no_prune)
        # naive score: parent spectral similarity x Tanimoto(product, parent)
        nv = G.naive_scores(cands, smi_of)
        for key, (smi, pr, ps, path) in cands.items():
            rows.append(dict(molecule_id=mid, ik=key, smiles=smi, parent=pr, parent_sim=ps, tanimoto=nv[key][0],
                             edits=path, n_edits=len(path.split("+")) if path else 0, hit=key == truth,
                             naive=nv[key][1]))
        got = truth in cands
        stats.append(dict(molecule_id=mid, n_cand=len(cands), generated=got, truth_is_parent=truth in parents))
        if (k + 1) % 25 == 0:
            st = pd.DataFrame(stats)
            print(f"{k+1}/{len(mids)} recall {st.generated.mean():.3f} median cands {st.n_cand.median():.0f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    R, St = pd.DataFrame(rows), pd.DataFrame(stats)
    R.to_parquet(f"{a.out}/candidates{sfx}.parquet", index=False)
    St.to_parquet(f"{a.out}/stats{sfx}.parquet", index=False)
    rr = []
    for mid in St.molecule_id:
        d = R[R.molecule_id == mid].sort_values("naive", ascending=False) if len(R) else R
        h = np.flatnonzero(d.hit.to_numpy()) if len(d) else []
        rr.append(1 / (h[0] + 1) if len(h) and h[0] < 25 else 0.0)
    print(f"GEN v0: {len(St)} molecules | truth generated {St.generated.mean():.3f} | "
          f"median candidates {St.n_cand.median():.0f} (mean {St.n_cand.mean():.0f}) | "
          f"naive MRR@25 {np.mean(rr):.4f} | top-1 {np.mean([r == 1 for r in rr]):.3f} [{time.time()-t0:.0f}s]")
    if len(R):
        e = R[R.hit].edits.str.split("+").explode().value_counts().head(10)
        print("edits that produced true structures:", e.to_dict())


if __name__ == "__main__":
    main()
