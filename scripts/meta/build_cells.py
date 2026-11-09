"""Test-like validation cells on natural products, with the full PubChem candidate window.

The hidden test is natural products from biological extracts (157-1159 Da, median 348) while every earlier cell
was built from Enveda-180 (drug-like screening chemistry, 235-499 Da). Cells here (work/meta/v1/molecules.parquet):

  np_tims  enveda-np-examples structures, their timsTOF spectra as queries (the closest proxy to the test;
           already excluded from all training: spectra, anchors and negatives)
  np_c2    natural products (COCONUT / LOTUS / NPAtlas) with public spectra, sampled so the mass histogram
           matches enveda-np-examples; class 2 is simulated by hiding ALL their spectra from training
           (exclude_all_ik.txt), the structure stays in PubChem

Candidates per molecule: every PubChem tier structure within 10 ppm of the neutral mass (floor 3 mDa), deduplicated
by InChIKey-14 (features = max over the rows of a key), RDKit-canonical stereo-free SMILES (format-clean), with the
tier's documentation features aligned. Structures are stored once per cell (windows of isomers overlap).

  python scripts/meta/build_cells.py                 # CPU: cells, query spectra, candidate windows
  python scripts/meta/build_cells.py --embed         # GPU: Morgan + centred PLM of each cell's structures
Then scripts/meta/eval_cells.py scores a checkpoint and the documentation prior on every cell.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
POP = "/kaggle/input/casmi26-pubchem-popularity-prior/"
TIER = POP + "casmi26-pubchem-tier/"
FEATS = {"lsid": "pc_lsid", "lpmid": "pc_lpmid", "lpatent": "pc_lpatent", "lsyn": "pc_lsynonyms", "ndep": "pc_n_depositors",
         "np": "pc_np", "src_mask": "pc_src_mask", "lotus": "pc_lotus_refs", "nvend": "pc_n_vendors", "lmesh": "pc_lmesh"}
MASS_EDGES = [157, 200, 250, 300, 350, 400, 450, 500, 600, 800, 1160]
t0 = time.time()


def log(s):
    print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)


def canon(s):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None, None
    try:
        return Chem.MolToSmiles(m, isomericSmiles=False), rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        return None, None


def mass_sample(U, target_mass, n, rng):
    """Sample n rows of U whose mass histogram follows target_mass (MASS_EDGES bins, +1 smoothing)."""
    tb = np.histogram(target_mass, MASS_EDGES)[0] + 1.0
    want = np.round(n * tb / tb.sum()).astype(int)
    b = np.digitize(U.exact_mass, MASS_EDGES) - 1
    out = []
    for k, w in enumerate(want):
        idx = np.flatnonzero(b == k)
        out.append(rng.choice(idx, min(w, len(idx)), replace=False))
    return U.iloc[np.sort(np.concatenate(out))]


def windows(cell, pm):
    tol = np.maximum(cell.exact_mass.to_numpy() * 1e-5, 0.003)
    lo = np.searchsorted(pm, cell.exact_mass.to_numpy() - tol)
    hi = np.searchsorted(pm, cell.exact_mass.to_numpy() + tol, side="right")
    return lo, hi


def build_candidates(cell, out, workers):
    pm = np.load(TIER + "pc_mass.npy", mmap_mode="r")
    assert bool(np.all(np.diff(np.asarray(pm[:1_000_000])) >= 0)), "pc_mass not sorted (corrupt copy?)"
    pik = np.load(POP + "pc_ik14.npy", mmap_mode="r")
    lo, hi = windows(cell, pm)
    rows = np.unique(np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]))
    log(f"{len(cell)} molecules: {int((hi - lo).sum())} window rows, {len(rows)} unique tier rows")
    T = pd.DataFrame({"row": rows, "ik": np.asarray(pik[rows]).astype("U14")})
    for k, f in FEATS.items():
        T[k] = np.asarray(np.load(POP + f + ".npy", mmap_mode="r")[rows]).astype(np.float32)
    T["doc"] = T.lsid + T.lpmid
    T = T.sort_values(["ik", "doc"], ascending=[True, False])
    agg = T.groupby("ik", sort=False).agg({"row": "first", **{k: "max" for k in FEATS}})
    S = agg.reset_index()
    off = np.load(TIER + "pc_off.npy", mmap_mode="r")
    smi = np.load(TIER + "pc_smiles.npy", mmap_mode="r")
    raw = [bytes(smi[off[r]:off[r + 1]]).decode() for r in S.row.to_numpy()]
    log(f"{len(S)} unique structures; canonicalising")
    with get_context("fork").Pool(workers) as P:
        cf = P.map(canon, raw, chunksize=4000)
    S["smiles"] = [c[0] for c in cf]
    S["formula"] = [c[1] for c in cf]
    S["injected"] = False
    # truths missing from their window (mass/key mismatch): inject from the meta table
    key_rows = {ik: i for i, ik in enumerate(S.ik)}
    miss = cell[~cell.ik14_raw.isin(key_rows) & ~cell.ik14_lib.isin(key_rows)]
    if len(miss):
        add = pd.DataFrame({"ik": miss.ik14_raw.fillna(miss.ik14_lib).to_numpy(), "row": -1, "smiles": miss.smiles_canon.to_numpy(),
                            "formula": miss.formula.to_numpy(), "injected": True})
        for k in FEATS:
            add[k] = 0.0
        S = pd.concat([S, add], ignore_index=True)
        log(f"injected {len(miss)} truths absent from their window")
    S = S[S.smiles.notna()].reset_index(drop=True)
    S.index.name = "sid"
    sid_of = pd.Series(S.index.to_numpy(), index=S.ik)
    sid_of = sid_of[~sid_of.index.duplicated()]
    win_ik = np.asarray(pik[rows]).astype("U14")
    C = []
    for mi, (a, b, r) in enumerate(zip(lo, hi, cell.itertuples())):
        ks = np.unique(win_ik[np.searchsorted(rows, a):np.searchsorted(rows, b)])
        sids = sid_of.reindex(ks).dropna().astype(np.int64).to_numpy()
        tk = r.ik14_raw if r.ik14_raw in sid_of.index else r.ik14_lib
        t = int(sid_of[tk]) if tk in sid_of.index else -1
        if t >= 0 and t not in set(sids):
            sids = np.append(sids, t)
        C.append(pd.DataFrame({"mol_id": r.mol_id, "sid": sids, "is_truth": sids == t}))
    C = pd.concat(C, ignore_index=True)
    truth_formula = dict(zip(cell.mol_id, cell.formula))
    C["same_formula"] = S.formula.to_numpy()[C.sid.to_numpy()] == C.mol_id.map(truth_formula).to_numpy()
    S.reset_index().to_parquet(f"{out}/structs.parquet", index=False)
    C.to_parquet(f"{out}/cands.parquet", index=False)
    n = C.groupby("mol_id").agg(n=("sid", "size"), nf=("same_formula", "sum"), t=("is_truth", "sum"))
    log(f"cands {len(C)} | per molecule median {n.n.median():.0f} (same formula {n.nf.median():.0f}) | "
        f"truth present {int((n.t > 0).sum())}/{len(n)}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="work/meta/v1/molecules.parquet")
    ap.add_argument("--lib", default="work/lib")
    ap.add_argument("--out", default="work/cells")
    ap.add_argument("--holdout", default="/kaggle/input/casmi26-train-v2/casmi26/splits/holdout.parquet")
    ap.add_argument("--c3", default="work/c3data", help="existing Enveda-180 class-3 cell (exclusions are merged)")
    ap.add_argument("--n-np", type=int, default=800)
    ap.add_argument("--max-spec", type=int, default=6, help="query spectra per np_c2 molecule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--plm-mean", default="work/emb_v3/plm_mean.npy")
    a = ap.parse_args()
    if a.embed:
        return embed(a)
    rng = np.random.default_rng(a.seed)
    M = pd.read_parquet(a.meta)
    spec = pd.read_parquet(f"{a.lib}/spectra.parquet", columns=["ingest_lib", "inchikey14", "ionization_mode", "adduct"])
    hold = set(pd.read_parquet(a.holdout).inchikey14)
    c3x = set(open(f"{a.c3}/exclude_all_ik.txt").read().split()) if os.path.exists(f"{a.c3}/exclude_all_ik.txt") else set()
    c3e = set(open(f"{a.c3}/exclude_ik.txt").read().split()) if os.path.exists(f"{a.c3}/exclude_ik.txt") else set()
    is_npx = M.n_spec_enveda_np_examples.fillna(0) > 0
    clean = (M.n_frag == 1) & (M.formal_charge == 0)

    cells = {}
    T = M[is_npx & M.in_pubchem_tier & clean].copy()
    cells["np_tims"] = T
    used = set(M.ik14_lib[is_npx]) | hold | c3x | c3e
    U = M[(M.in_coconut | M.in_lotus | M.in_npatlas) & (M.n_public_libs > 0) & M.in_test_mass_range & clean
          & M.in_pubchem_tier & ~M.ik14_lib.isin(used) & ~M.ik14_raw.isin(used)]
    cells["np_c2"] = mass_sample(U, T.exact_mass.to_numpy(), a.n_np, rng)
    log(f"NP universe {len(U)} -> np_c2 {len(cells['np_c2'])}; np_tims {len(T)}")

    by_ik = spec.groupby("inchikey14").indices
    excl_all = set(c3x)
    reg = []
    for name, cell in cells.items():
        out = f"{a.out}/{name}"
        os.makedirs(out, exist_ok=True)
        cell = cell.copy()
        cell["mol_id"] = [f"{name}_{i:04d}" for i in range(len(cell))]
        cell["doc"] = cell.pc_lsid.fillna(0) + cell.pc_lpmid.fillna(0)
        q = []
        for mid, ik in zip(cell.mol_id, cell.ik14_lib):
            idx = by_ik.get(ik, np.array([], int))
            libs = spec.ingest_lib.to_numpy()[idx]
            idx = idx[libs == "enveda-np-examples"] if name == "np_tims" else idx[libs != "enveda-180"]
            if name != "np_tims" and len(idx) > a.max_spec:
                idx = np.sort(rng.choice(idx, a.max_spec, replace=False))
            q += [(mid, int(i)) for i in idx]
        Q = pd.DataFrame(q, columns=["mol_id", "spec_idx"])
        Q = Q.join(spec[["ingest_lib", "ionization_mode", "adduct"]], on="spec_idx")
        cell = cell[cell.mol_id.isin(set(Q.mol_id))]
        Q.to_parquet(f"{out}/queries.parquet", index=False)
        keep = ["mol_id", "ik14_lib", "ik14_raw", "skey", "smiles_canon", "formula", "exact_mass", "mass_bin", "np_score",
                "doc", "pc_lsid", "pc_lpmid", "pc_formula_isomers", "n_tier_10ppm", "in_coconut", "in_lotus", "in_npatlas",
                "has_timstof", "chemotype", "n_public_libs"]
        cell[keep].to_parquet(f"{out}/molecules.parquet", index=False)
        n = build_candidates(cell, out, a.workers)
        ex = sorted(set(cell.ik14_lib) | set(cell.ik14_raw.dropna()) | set(cell.skey.dropna()))
        open(f"{out}/exclude_all_ik.txt", "w").write("\n".join(ex) + "\n")
        excl_all |= set(ex)
        reg.append({"cell": name, "n_molecules": len(cell), "n_queries": len(Q), "median_cands": float(n.n.median()),
                    "median_same_formula": float(n.nf.median()), "truth_present": int((n.t > 0).sum()),
                    "mass_median": float(cell.exact_mass.median()), "doc_median": float(cell.doc.median()),
                    "query_libs": Q.ingest_lib.value_counts().to_dict(), "pos_share": float((Q.ionization_mode == "positive").mean())})
        log(json.dumps(reg[-1]))
    # union registry for training: every cell structure hidden from every library; Enveda-180 cell spectra too
    open(f"{a.out}/exclude_all_ik.txt", "w").write("\n".join(sorted(excl_all)) + "\n")
    open(f"{a.out}/exclude_ik.txt", "w").write("\n".join(sorted(c3e)) + "\n")
    json.dump({"cells": reg, "seed": a.seed, "excluded_structures": len(excl_all)}, open(f"{a.out}/registry.json", "w"), indent=1)
    log(f"registry: {len(excl_all)} structures hidden from training -> {a.out}/exclude_all_ik.txt")


def embed(a):
    from casmi.plm import PLMEmbedder
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from embed_molecules import morgan
    emb = PLMEmbedder(device="cuda", mean=np.load(a.plm_mean))
    for name in sorted(os.listdir(a.out)):
        p = f"{a.out}/{name}/structs.parquet"
        if not os.path.exists(p) or os.path.exists(f"{a.out}/{name}/plm.npy"):
            continue
        S = pd.read_parquet(p, columns=["smiles"])
        with get_context("fork").Pool(a.workers) as P:
            fp = np.stack(P.map(morgan, S.smiles, chunksize=4000))
        np.save(f"{a.out}/{name}/fp.npy", fp)
        out = np.zeros((len(S), emb.model.config.hidden_size), np.float16)
        for i in range(0, len(S), 200_000):
            out[i:i + 200_000] = emb.embed(S.smiles.iloc[i:i + 200_000].tolist(), batch_size=1024)
            log(f"{name}: plm {min(i + 200_000, len(S))}/{len(S)}")
        np.save(f"{a.out}/{name}/plm.npy", out)


if __name__ == "__main__":
    main()
