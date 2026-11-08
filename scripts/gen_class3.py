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
# v1 additions, from the miss diagnostic (formula gaps truth - nearest analog: O+1 at sp3 CH/CH3, CH2 on
# aliphatic C, pentose / glucuronide / acyl sugars, and positional isomers of the same formula)
EDITS_V1 = [
    ("hydroxylation_tert", "[CH1;!R;!$(C=*);!$(C[O,N]):1]>>[C:1]O"),
    ("hydroxylation_ringCH", "[CH1;R;!$(C=*);!$(C[O,N]):1]>>[C:1]O"),
    ("hydroxylation_methyl", "[CH3;$(C[#6]):1]>>[CH2:1]O"),
    ("C-methylation_aliph", "[CH2;!$(C[O,N]);!$(C=*):1]>>[CH1:1]C"),
    ("pentosylation_O", "[c,C:1][OH1:2]>>[c,C:1][O:2]C1OCC(O)C(O)C1O"),
    ("glucuronidation_O", "[c,C:1][OH1:2]>>[c,C:1][O:2]C1OC(C(=O)O)C(O)C(O)C1O"),
    ("malonylation_O", "[C,c:1][OH1:2]>>[C,c:1][O:2]C(=O)CC(=O)O"),
    ("coumaroylation_O", "[C:1][OH1:2]>>[C:1][O:2]C(=O)C=Cc1ccc(O)cc1"),
    ("galloylation_O", "[C:1][OH1:2]>>[C:1][O:2]C(=O)c1cc(O)c(O)c(O)c1"),
    ("dehydroxylation_arom", "[c:1][OH1]>>[cH1:1]"),
    ("carboxyl_to_methylester", "[C:1](=[O:2])[OH1]>>[C:1](=[O:2])OC"),
    ("lactone_hydrolysis", "[C;R:1](=[O:2])[O;R:3][C;R:4]>>[C:1](=[O:2])O.[O:3][C:4]"),
    ("decarboxylation", "[#6:1][CX3](=O)[OH1]>>[#6:1]"),
    ("N-oxidation", "[NX3;H0;!$(N-C=O);!$(N-a):1]>>[N+:1][O-]"),
]
RXN = [(n, AllChem.ReactionFromSmarts(s)) for n, s in EDITS]
MAX_PRODUCTS = 20


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


DELTA = {}   # edit name -> monoisotopic mass change (constant per reaction; measured on library molecules)


def edit_deltas(smiles, n_probe=3000):
    """Measure each edit's mass change on library molecules; edits never seen to fire stay unknown."""
    for smi in smiles[:n_probe]:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for name, rxn in RXN:
            if name in DELTA:
                continue
            try:
                prods = rxn.RunReactants((m,), maxProducts=1)
                p = prods[0][0] if prods and len(prods[0]) == 1 else None
                if p is not None:
                    Chem.SanitizeMol(p)
                    DELTA[name] = Descriptors.ExactMolWt(p) - Descriptors.ExactMolWt(m)
            except Exception:
                continue
        if len(DELTA) == len(RXN):
            break


def _can_reach(mass_now, remaining, target, tol):
    """Can <= `remaining` further edits (any combination of known deltas) bring mass_now to target?
    Exact pruning for the final mass filter; disabled when some edit's delta is unknown."""
    if target is None or len(DELTA) < len(RXN):
        return True
    d = target - mass_now
    if abs(d) <= tol:
        return True
    if remaining == 0:
        return False
    return any(_can_reach(mass_now + v, remaining - 1, target, tol) for v in DELTA.values())


def apply_edits(mol, depth, target=None, tol=0.0):
    """All products of up to `depth` edits (sanitised, canonical SMILES -> edit path). With a target mass,
    products that can no longer reach it within the remaining edits are not expanded (same final set after
    the mass filter, much less work)."""
    out = {}
    frontier = {Chem.MolToSmiles(mol): (Descriptors.ExactMolWt(mol), ())}
    for level in range(depth):
        remaining = depth - level - 1
        nxt = {}
        for smi, (mw, path) in frontier.items():
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            for name, rxn in RXN:
                if name in DELTA and not _can_reach(mw + DELTA[name], remaining, target, tol):
                    continue
                try:
                    prods = rxn.RunReactants((m,), maxProducts=MAX_PRODUCTS)
                except Exception:
                    continue
                for pp in prods:
                    p = pp[0] if len(pp) == 1 else None
                    if p is None:                      # ring-opening edits give 2 fragments of one molecule
                        try:
                            p = Chem.MolFromSmiles(".".join(Chem.MolToSmiles(x) for x in pp))
                            if p is None or "." in Chem.MolToSmiles(p):
                                continue
                        except Exception:
                            continue
                    try:
                        Chem.SanitizeMol(p)
                        ps = Chem.MolToSmiles(p)
                    except Exception:
                        continue
                    if ps not in out and ps not in nxt:
                        nxt[ps] = (Descriptors.ExactMolWt(p), path + (name,))
        out.update({k: v[1] for k, v in nxt.items()})
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
    ap.add_argument("--start", type=int, default=0, help="chunk of the cell's molecules (outputs get a _<start> suffix)")
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--edits", default="v0", choices=["v0", "v1"])
    ap.add_argument("--max-products", type=int, default=20)
    ap.add_argument("--no-prune", action="store_true", help="expand every edit (reference for the pruning check)")
    ap.add_argument("--sample", default="", help="comma-free file of molecule_ids to run on (overrides start/end)")
    a = ap.parse_args()
    global RXN, MAX_PRODUCTS
    MAX_PRODUCTS = a.max_products
    if a.edits == "v1":
        RXN = RXN + [(n, AllChem.ReactionFromSmarts(x)) for n, x in EDITS_V1]
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
    if not a.no_prune:
        edit_deltas(lib.molecules.smiles.tolist())
        print(f"edit mass deltas known for {len(DELTA)}/{len(RXN)} edits (pruning {'on' if len(DELTA) == len(RXN) else 'off'})", flush=True)
    spec_ik = lib.spectra.inchikey14.to_numpy()
    spec_mode = lib.spectra.ionization_mode.to_numpy()
    smi_of = dict(zip(lib.molecules.inchikey14, lib.molecules.smiles))
    q = prep_query(Q)
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
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
            prods = apply_edits(pm, a.depth, None if a.no_prune else mass, tol)
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
