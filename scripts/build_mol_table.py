"""J2: format-clean training table: train molecules + database negatives for EVERY train molecule.

Fixes two inflations of the old table (scripts/embed_molecules.py):
  * format: positives were RDKit-canonical train strings, negatives raw database strings (PubChem is Kekule);
    the PLM saw the format. Here every SMILES (train and database) is RDKit-canonical, stereo-free.
  * memorised distractors: database rows existed only around holdout masses, so training hard negatives WERE the
    evaluation distractors. Here database structures are split by a hash of their InChIKey-14 into two disjoint
    halves: "db" (training negatives, around every train molecule) and "dist" (evaluation distractors, around
    the evaluation molecules only).
Validation molecules (holdout, class-3 cell, enveda-np-examples, --exclude files) never appear as negatives.

Sources: the pipeline's NP-flavoured pool (v4g casmi26-v2-pool, non-train rows) and the PubChem tier arrays.

  python scripts/build_mol_table.py --out work/emb_v3
  then: python scripts/build_mol_table.py --out work/emb_v3 --embed      (PLM + Morgan, GPU)
"""
from __future__ import annotations

import argparse, hashlib, os, sys, time
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

V4G = "/kaggle/input/casmi26-attributed-v4g-reproduction-assets"
PC = "/kaggle/input/casmi26-pubchem-popularity-prior/casmi26-pubchem-tier/"
G = {}


def canon_info(s):
    """(canonical stereo-free SMILES, InChIKey-14, formula, exact mass) or None."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        return (Chem.MolToSmiles(m, isomericSmiles=False), Chem.MolToInchiKey(m)[:14],
                rdMolDescriptors.CalcMolFormula(m), rdMolDescriptors.CalcExactMolWt(m))
    except Exception:
        return None


def half(ik):
    return hashlib.sha1(ik.encode()).digest()[0] & 1        # 0 -> "db" (train negatives), 1 -> "dist" (eval)


def formula_of(s):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(s)
    return rdMolDescriptors.CalcMolFormula(m) if m is not None else None


def pick(args):
    """Negatives (or distractors) for one anchor molecule: same formula first, from both sources."""
    ik, mass, formula, want_half, k_pool, k_pc, seed = args
    rng = np.random.default_rng(seed)
    tol = max(mass * 10e-6, 0.003)
    out = []
    for name, k, draw in (("pool", k_pool, 4 * k_pool), ("pubchem", k_pc, 4 * k_pc)):
        if k == 0:
            continue
        if name == "pool":
            lo, hi = np.searchsorted(G["pool_mass"], mass - tol), np.searchsorted(G["pool_mass"], mass + tol, side="right")
            get = lambda i: G["pool_smiles"][i]
        else:
            lo, hi = np.searchsorted(G["pc_mass"], mass - tol), np.searchsorted(G["pc_mass"], mass + tol, side="right")
            get = lambda i: bytes(G["pc_smiles"][G["pc_off"][i]:G["pc_off"][i + 1]]).decode()
        if hi <= lo:
            continue
        idx = rng.choice(np.arange(lo, hi), min(draw, hi - lo), replace=False)
        cand = [(get(i), name) for i in idx]
        fs = [formula_of(s) for s, _ in cand]
        order = sorted(range(len(cand)), key=lambda j: fs[j] != formula)      # same formula first
        kept = 0
        for j in order:
            if fs[j] is None:
                continue
            info = canon_info(cand[j][0])
            if info is None or info[1] == ik or info[1] in G["exclude"] or half(info[1]) != want_half:
                continue
            out.append((*info, name, fs[j] == formula))
            kept += 1
            if kept >= k:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="work/lib")
    ap.add_argument("--out", default="work/emb_v3")
    ap.add_argument("--k-pool", type=int, default=8, help="negatives per train molecule from the NP pool")
    ap.add_argument("--k-pc", type=int, default=8, help="negatives per train molecule from PubChem")
    ap.add_argument("--k-dist", type=int, default=100, help="distractors per evaluation molecule and source")
    ap.add_argument("--holdout", default="/kaggle/input/casmi26-train-v2/casmi26/splits/holdout.parquet")
    ap.add_argument("--exclude", action="append", default=["work/c3data/exclude_all_ik.txt"])
    ap.add_argument("--eval-answers", action="append", default=["work/c3data/answers.parquet"])
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many train molecules")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.embed:
        return embed(a)
    t0 = time.time()
    lib_mols = pd.read_parquet(f"{a.lib}/molecules.parquet", columns=["inchikey14", "smiles", "exact_mass"])
    spec = pd.read_parquet(f"{a.lib}/spectra.parquet", columns=["ingest_lib", "inchikey14"])
    hold = pd.read_parquet(a.holdout)
    excl = set(hold.inchikey14) | set(spec.inchikey14[spec.ingest_lib == "enveda-np-examples"])
    for f in a.exclude:
        excl |= set(open(f).read().split()) if os.path.exists(f) else set()
    evals = [pd.read_parquet(f) for f in a.eval_answers if os.path.exists(f)]
    G["exclude"] = excl
    pool = pd.read_parquet(f"{V4G}/casmi26-v2-pool/pool_meta.parquet", columns=["smiles", "mass", "train_sid"])
    pool = pool[pool.train_sid.isna() | (pool.train_sid < 0)] if pool.train_sid.notna().any() else pool
    pool = pool.sort_values("mass")
    G["pool_mass"], G["pool_smiles"] = pool.mass.to_numpy(), pool.smiles.to_numpy()
    G["pc_mass"] = np.load(PC + "pc_mass.npy", mmap_mode="r")
    G["pc_off"] = np.load(PC + "pc_off.npy", mmap_mode="r")
    G["pc_smiles"] = np.load(PC + "pc_smiles.npy", mmap_mode="r")
    assert bool(np.all(np.diff(np.asarray(G["pc_mass"][:1_000_000])) >= 0)), "pc_mass not sorted (corrupt copy?)"
    print(f"NP pool (non-train) {len(pool)} | PubChem {len(G['pc_mass'])} | excluded structures {len(excl)} [{time.time() - t0:.0f}s]", flush=True)

    with get_context("fork").Pool(a.workers) as P:
        tr = list(P.imap(canon_info, lib_mols.smiles, chunksize=2000))
    train = pd.DataFrame([x if x else (None,) * 4 for x in tr], columns=["smiles", "ik_check", "formula", "mass_calc"])
    train["inchikey14"], train["exact_mass"], train["source"], train["origin"], train["same_formula"] = \
        lib_mols.inchikey14.to_numpy(), lib_mols.exact_mass.to_numpy(), "train", "train", True
    train["smiles"] = train.smiles.fillna(pd.Series(lib_mols.smiles.to_numpy()))
    print(f"train molecules {len(train)} canonicalised; key mismatches (lib ik14 vs RDKit) {(train.ik_check != train.inchikey14).sum()} [{time.time() - t0:.0f}s]", flush=True)
    anchors = train if not a.limit else train.iloc[:a.limit]
    anchors = anchors[~anchors.inchikey14.isin(excl)]
    jobs = [(ik, m, f, 0, a.k_pool, a.k_pc, i) for i, (ik, m, f) in enumerate(zip(anchors.inchikey14, anchors.exact_mass, anchors.formula))]
    ev = pd.concat(evals, ignore_index=True) if evals else pd.DataFrame(columns=["inchikey14", "smiles"])
    ev_mols = pd.concat([hold[["inchikey14", "smiles"]], ev[["inchikey14", "smiles"]]], ignore_index=True).drop_duplicates("inchikey14")
    ev_info = [canon_info(s) for s in ev_mols.smiles]
    jobs += [(ik, x[3], x[2], 1, a.k_dist, a.k_dist, 10_000_000 + i) for i, (ik, x) in enumerate(zip(ev_mols.inchikey14, ev_info)) if x]
    print(f"jobs: {len(anchors)} train anchors (negatives) + {len(jobs) - len(anchors)} evaluation molecules (distractors)", flush=True)
    rows = []
    with get_context("fork").Pool(a.workers) as P:
        for k, r in enumerate(P.imap(pick, jobs, chunksize=64)):
            src = "db" if jobs[k][3] == 0 else "dist"
            rows += [(*x, src) for x in r]
            if (k + 1) % 20000 == 0:
                print(f"  {k + 1}/{len(jobs)} anchors, {len(rows)} rows [{time.time() - t0:.0f}s]", flush=True)
    D = pd.DataFrame(rows, columns=["smiles", "inchikey14", "formula", "exact_mass", "origin", "same_formula", "source"])
    D = D[~D.inchikey14.isin(set(train.inchikey14))].drop_duplicates("inchikey14")
    tab = pd.concat([train[["inchikey14", "smiles", "exact_mass", "formula", "source", "origin", "same_formula"]], D[["inchikey14", "smiles", "exact_mass", "formula", "source", "origin", "same_formula"]]],
                    ignore_index=True)
    assert not (set(D.inchikey14[D.source == "db"]) & set(D.inchikey14[D.source == "dist"]))
    assert not (set(D.inchikey14) & excl)
    tab.to_parquet(f"{a.out}/mol_table.parquet", index=False)
    print(f"mol_table {len(tab)} rows: {tab.source.value_counts().to_dict()} | origin {D.origin.value_counts().to_dict()} | "
          f"same-formula share of negatives {D.same_formula.mean():.2f} [{time.time() - t0:.0f}s]", flush=True)


def embed(a):
    from casmi.plm import PLMEmbedder
    sys.path.insert(0, os.path.dirname(__file__))
    from embed_molecules import morgan
    tab = pd.read_parquet(f"{a.out}/mol_table.parquet")
    t = time.time()
    with get_context("fork").Pool(a.workers) as P:
        fp = np.stack(P.map(morgan, tab.smiles, chunksize=4000))
    np.save(f"{a.out}/fp.npy", fp); print(f"fingerprints {fp.shape} [{time.time() - t:.0f}s]", flush=True)
    emb = PLMEmbedder(device="cuda")
    out = np.zeros((len(tab), emb.model.config.hidden_size), np.float16)
    step = 100_000
    for i in range(0, len(tab), step):
        out[i:i + step] = emb.embed(tab.smiles.iloc[i:i + step].tolist(), batch_size=1024)
        print(f"plm {min(i + step, len(tab))}/{len(tab)} [{time.time() - t:.0f}s]", flush=True)
    np.save(f"{a.out}/plm.npy", out)
    tr = out[(tab.source == "train").to_numpy()].astype(np.float32)
    np.save(f"{a.out}/plm_mean.npy", tr.mean(0))
    print("done", flush=True)


if __name__ == "__main__":
    main()
