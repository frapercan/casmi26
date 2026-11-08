"""Class-3 candidate generator core: spectral analogs -> NP tailoring edits -> exact-mass filter.

Shared by scripts/gen_class3.py (honest cell) and the submission layer (hybrid/class3_sub_cell.py).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity

from .metric import ik14

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


def library_matrix(lib, bins=1500):
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




def set_edits(version="v0", max_products=20):
    global RXN, MAX_PRODUCTS
    RXN = [(n, AllChem.ReactionFromSmarts(s)) for n, s in EDITS + (EDITS_V1 if version == "v1" else [])]
    MAX_PRODUCTS = max_products
    DELTA.clear()


_FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def analog_sims(g, F, N, spec_ik, spec_mode, nl_weight=0.5):
    """Best fragment + neutral-loss cosine of each library molecule to the query spectra (same polarity)."""
    sim = np.zeros(len(spec_ik), np.float32)
    for mz, it, prec, mode in zip(g.mz, g.it, g.precursor_mz, g.ionization_mode):
        f, n = binned(mz, it, prec)
        s = F @ f + nl_weight * (N @ n)
        s[spec_mode != mode] = 0
        sim = np.maximum(sim, s)
    return pd.Series(sim).groupby(spec_ik).max().sort_values(ascending=False)


def generate(g, F, N, spec_ik, spec_mode, smi_of, analogs=30, depth=2, ppm=10.0, nl_weight=0.5, prune=True):
    """Candidates for one query molecule g (prep_query rows): {ik14: (smiles, parent, parent_sim, edit path)}."""
    mass = float(np.median(g.mass))
    best = analog_sims(g, F, N, spec_ik, spec_mode, nl_weight)
    parents = best.index[:analogs].tolist()
    tol = mass * ppm * 1e-6
    cands = {}
    for pr in parents:
        pm = Chem.MolFromSmiles(smi_of[pr])
        if pm is None:
            continue
        prods = apply_edits(pm, depth, mass if prune else None, tol)
        prods[Chem.MolToSmiles(pm)] = ()
        for smi, path in prods.items():
            m = Chem.MolFromSmiles(smi)
            if m is None or abs(Descriptors.ExactMolWt(m) - mass) > tol:
                continue
            key = ik14(smi)
            if key and (key not in cands or best[pr] > cands[key][2]):
                cands[key] = (smi, pr, float(best[pr]), "+".join(path))
    return cands, parents


def naive_scores(cands, smi_of):
    """Baseline order: parent spectral similarity x Tanimoto(product, parent)."""
    out = {}
    for key, (smi, pr, ps, path) in cands.items():
        tan = BulkTanimotoSimilarity(_FPG.GetFingerprint(Chem.MolFromSmiles(smi)), [_FPG.GetFingerprint(Chem.MolFromSmiles(smi_of[pr]))])[0]
        out[key] = (tan, ps * tan)
    return out
