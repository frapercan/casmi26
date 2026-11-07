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
# (name, reaction SMARTS) — common phase-I/II and biosynthetic tailoring steps of natural products
EDITS = [
    ("hydroxylation_arom", "[cH:1]>>[c:1]O"),
    ("hydroxylation_aliph", "[CH2;!R:1]>>[CH1:1]O"),
    ("hydroxylation_ring", "[CH2;R:1]>>[CH1:1]O"),
    ("dehydroxylation", "[C,c:1][OH1]>>[C,c:1]"),
    ("O-methylation", "[c,C:1][OH1:2]>>[c,C:1][O:2]C"),
    ("O-demethylation", "[c,C:1][O:2][CH3]>>[c,C:1][O:2]"),
    ("N-methylation", "[N;H1,H2;!$(N-C=O):1]>>[N:1]C"),
    ("N-demethylation", "[N:1][CH3]>>[N:1]"),
    ("C-methylation_arom", "[cH:1]>>[c:1]C"),
    ("glycosylation_O", "[c,C:1][OH1:2]>>[c,C:1][O:2]C1OC(CO)C(O)C(O)C1O"),
    ("deglycosylation", "[c,C:1][O:2]C1OC(CO)C(O)C(O)C1O>>[c,C:1][O:2]"),
    ("rhamnosylation_O", "[c,C:1][OH1:2]>>[c,C:1][O:2]C1OC(C)C(O)C(O)C1O"),
    ("acetylation_O", "[C,c:1][OH1:2]>>[C,c:1][O:2]C(C)=O"),
    ("deacetylation", "[C,c:1][O:2]C(=O)[CH3]>>[C,c:1][O:2]"),
    ("hydrogenation", "[C:1]=[C:2]>>[C:1][C:2]"),
    ("dehydrogenation", "[CH1:1][CH1:2]>>[C:1]=[C:2]"),
    ("ketone_reduction", "[C:1]=[O:2]>>[C:1][O:2]"),
    ("alcohol_oxidation", "[CH1:1][OH1:2]>>[C:1]=[O:2]"),
    ("epoxidation", "[C:1]=[C:2]>>[C:1]1O[C:2]1"),
    ("sulfation_O", "[c,C:1][OH1:2]>>[c,C:1][O:2]S(=O)(=O)O"),
    ("prenylation_arom", "[cH:1]>>[c:1]CC=C(C)C"),
    ("hydration", "[C:1]=[C:2]>>[C:1][C:2]O"),
    ("dehydration", "[CH1:1][C:2][OH1]>>[C:1]=[C:2]"),
    ("methoxy_to_methylenedioxy", "[c:1]([O][CH3])[c:2][OH1]>>[c:1]1[c:2]OCO1"),
]
RXN = [(n, AllChem.ReactionFromSmarts(s)) for n, s in EDITS]


def binned(mz, it, prec, bins=1500, res=1.0):
    """Fragment and neutral-loss vectors (nominal-mass bins), each L2-normalised."""
    f = np.zeros(bins, np.float32); n = np.zeros(bins, np.float32)
    for m, i in zip(mz, it):
        b = int(m / res)
        if 0 <= b < bins:
            f[b] = max(f[b], i)
        l = int((prec - m) / res)
        if 0 < l < bins:
            n[l] = max(n[l], i)
    return f / (np.linalg.norm(f) + 1e-9), n / (np.linalg.norm(n) + 1e-9)


def library_matrix(lib: Library, bins=1500):
    rows, cols, vals, nrows, ncols, nvals = [], [], [], [], [], []
    prec = lib.spectra.precursor_mz.to_numpy()
    for i in range(len(lib.spectra)):
        mz, it = lib.peaks(i)
        f, n = binned(mz, it, prec[i], bins)
        nz = np.flatnonzero(f); rows += [i] * len(nz); cols += nz.tolist(); vals += f[nz].tolist()
        nz = np.flatnonzero(n); nrows += [i] * len(nz); ncols += nz.tolist(); nvals += n[nz].tolist()
    S = len(lib.spectra)
    return (sp.csr_matrix((vals, (rows, cols)), shape=(S, bins)),
            sp.csr_matrix((nvals, (nrows, ncols)), shape=(S, bins)))


def apply_edits(mol, depth):
    """All products of up to `depth` edits (sanitised, canonical SMILES -> edit path)."""
    out = {}
    frontier = {Chem.MolToSmiles(mol): ()}
    for _ in range(depth):
        nxt = {}
        for smi, path in frontier.items():
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            for name, rxn in RXN:
                try:
                    prods = rxn.RunReactants((m,), maxProducts=20)
                except Exception:
                    continue
                for (p,) in prods:
                    try:
                        Chem.SanitizeMol(p)
                        ps = Chem.MolToSmiles(p)
                    except Exception:
                        continue
                    if ps not in out and ps not in nxt:
                        nxt[ps] = path + (name,)
        out.update(nxt)
        frontier = nxt
    return out


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
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    A = pd.read_parquet(f"{a.cell}/answers.parquet")
    Q = pd.read_parquet(f"{a.cell}/queries.parquet")
    hide = set(open(f"{a.cell}/exclude_all_ik.txt").read().split())
    lib = Library.load(a.lib)
    lib = lib.subset(~lib.spectra.inchikey14.isin(hide).to_numpy())
    assert not (set(lib.molecules.inchikey14) & hide), "class-3 answers leaked into the library"
    print(f"library without the cell: {len(lib.spectra)} spectra, {len(lib.molecules)} molecules [{time.time()-t0:.0f}s]", flush=True)
    F, N = library_matrix(lib)
    print(f"binned library matrices [{time.time()-t0:.0f}s]", flush=True)
    spec_ik = lib.spectra.inchikey14.to_numpy()
    spec_mode = lib.spectra.ionization_mode.to_numpy()
    smi_of = dict(zip(lib.molecules.inchikey14, lib.molecules.smiles))
    q = prep_query(Q)
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows, stats = [], []
    mids = A.molecule_id.tolist()[: a.max_mols or None]
    for k, mid in enumerate(mids):
        g = q[q.molecule_id == mid]
        truth = A.set_index("molecule_id").inchikey14[mid]
        mass = float(np.median(g.mass))
        # 1) analogs: best cosine over the molecule's spectra, same polarity
        sim = np.zeros(len(spec_ik), np.float32)
        for mz, it, prec, mode in zip(g.mz, g.it, g.precursor_mz, g.ionization_mode):
            f, n = binned(mz, it, prec)
            s = F @ f + a.nl_weight * (N @ n)
            s[spec_mode != mode] = 0
            sim = np.maximum(sim, s)
        best = pd.Series(sim).groupby(spec_ik).max().sort_values(ascending=False)
        parents = best.index[: a.analogs].tolist()
        # 2-3) edits + formula filter
        tol = mass * a.ppm * 1e-6
        cands = {}
        for pr in parents:
            pm = Chem.MolFromSmiles(smi_of[pr])
            if pm is None:
                continue
            prods = apply_edits(pm, a.depth)
            prods[Chem.MolToSmiles(pm)] = ()
            for smi, path in prods.items():
                m = Chem.MolFromSmiles(smi)
                if m is None or abs(Descriptors.ExactMolWt(m) - mass) > tol:
                    continue
                key = ik14(smi)
                if key and (key not in cands or best[pr] > cands[key][2]):
                    cands[key] = (smi, pr, float(best[pr]), "+".join(path))
        # naive score: parent spectral similarity x Tanimoto(product, parent)
        for key, (smi, pr, ps, path) in cands.items():
            m, pmol = Chem.MolFromSmiles(smi), Chem.MolFromSmiles(smi_of[pr])
            tan = BulkTanimotoSimilarity(fpg.GetFingerprint(m), [fpg.GetFingerprint(pmol)])[0]
            rows.append(dict(molecule_id=mid, ik=key, smiles=smi, parent=pr, parent_sim=ps, tanimoto=tan,
                             edits=path, n_edits=len(path.split("+")) if path else 0, hit=key == truth,
                             naive=ps * tan))
        got = truth in cands
        stats.append(dict(molecule_id=mid, n_cand=len(cands), generated=got, truth_is_parent=truth in parents))
        if (k + 1) % 25 == 0:
            st = pd.DataFrame(stats)
            print(f"{k+1}/{len(mids)} recall {st.generated.mean():.3f} median cands {st.n_cand.median():.0f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    R, St = pd.DataFrame(rows), pd.DataFrame(stats)
    R.to_parquet(f"{a.out}/candidates.parquet", index=False)
    St.to_parquet(f"{a.out}/stats.parquet", index=False)
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
