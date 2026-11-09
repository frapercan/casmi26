"""Meta-classified molecule table (v1): one row per training structure, keyed by the official InChIKey-14.

Groups (docs/research_plan.md §3.4, first cut):
  identity   skey (official, tautomer-canonical ik14), ik14_raw, smiles_canon, formula, exact_mass, n_heavy,
             formal_charge, n_frag, elem_class, has_N/S/P/hal
  provenance n_spec_<lib> per training library, n_libs, n_public_libs (non-Enveda), has_timstof, n_pos/n_neg,
             n_adducts, n_inst_types
  novelty    c1_train_public (spectra in a non-Enveda training library), c1_mbh (MassBank-harmonised), c1_extlib,
             c1_enveda_only, in_pubchem (tier | delta | growth), in_coconut, in_lotus, in_npatlas, in_pcl,
             nov_true in {1, 2, 3, 3?} (3? = not a clean single neutral CHNOPS+hal 150-1250 Da structure)
  popularity pc_cid, pc_lsid, pc_lpmid, pc_lpatent, pc_lsyn, pc_ndep, pc_src_mask, pc_formula_isomers
  chemistry  np_score (Ertl), chemotype flags (steroid core, secondary amide, ester, glycoside), murcko
  density    n_tier_10ppm, n_pool_10ppm, n_coco_10ppm (structures of the PubChem tier / pool / COCONUT within 10 ppm)
  test range mass_bin, in_test_mass_range (157-1159 Da)

  python scripts/meta/build_molecules.py --out work/meta/v1
"""
from __future__ import annotations

import argparse, os, sys, time
from multiprocessing import get_context

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

COMP = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra"
POP = "/kaggle/input/casmi26-pubchem-popularity-prior/"
V4G = "/kaggle/input/casmi26-attributed-v4g-reproduction-assets/"
MBH = "/kaggle/input/casmi-2026-spectral-library-massbankharmonized/"
EXT = "/kaggle/input/casmi26-extlib/"
ENVEDA = {"enveda-180", "enveda-np-examples"}
SMARTS = {
    "has_steroid_core": "[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]~1)~[#6]~[#6]~[#6]1~[#6]~2~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~1~2",
    "has_sec_amide": "[CX3](=O)[NX3;H1][#6]",
    "has_ester": "[CX3](=O)[OX2][#6]",
    "has_glycoside": "[OX2;R1]1[CX4;R1][CX4;R1]([OX2])[CX4;R1]([OX2])[CX4;R1]([OX2])[CX4;R1]1",
    "has_carboxylic_acid": "[CX3](=O)[OX2H1]",
    "has_phenol": "c[OX2H1]",
}
_NP = None
_PAT = None


def chem(smi):
    """Per-structure RDKit descriptors (worker)."""
    global _NP, _PAT
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors, Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    if _NP is None:
        from rdkit.Contrib.NP_Score import npscorer
        _NP = (npscorer, npscorer.readNPModel())
        _PAT = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
    from casmi.metric import ik14
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    el = {a.GetSymbol() for a in m.GetAtoms()}
    hal = bool(el & {"F", "Cl", "Br", "I"})
    base = el - {"C", "H"}
    elem = "CHO" if base <= {"O"} else "CHNO" if base <= {"O", "N"} else "CHNOPS" if base <= {"O", "N", "P", "S"} else "other"
    if hal and base - {"F", "Cl", "Br", "I"} <= {"O", "N", "P", "S"}:
        elem = "hal"
    try:
        sk = ik14(smi)
    except Exception:
        sk = None
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        scaf = ""
    out = dict(skey=sk, ik14_raw=Chem.MolToInchiKey(m)[:14], smiles_canon=Chem.MolToSmiles(m, isomericSmiles=False),
               formula=rdMolDescriptors.CalcMolFormula(m), mass_calc=rdMolDescriptors.CalcExactMolWt(m),
               n_heavy=m.GetNumHeavyAtoms(), formal_charge=Chem.GetFormalCharge(m), n_frag=len(Chem.GetMolFrags(m)),
               elem_class=elem, has_N="N" in el, has_S="S" in el, has_P="P" in el, has_hal=hal,
               np_score=float(_NP[0].scoreMol(m, _NP[1])), murcko=scaf, n_rings=rdMolDescriptors.CalcNumRings(m),
               logp=float(Descriptors.MolLogP(m)))
    for k, p in _PAT.items():
        out[k] = m.HasSubstructMatch(p)
    return out


def lookup(sorted_keys, order, q):
    """Row index in the original array of each query key (first match) or -1."""
    pos = np.searchsorted(sorted_keys, q)
    pos = np.clip(pos, 0, len(sorted_keys) - 1)
    hit = sorted_keys[pos] == q
    return np.where(hit, order[pos], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="work/lib")
    ap.add_argument("--out", default="work/meta/v1")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"{s} [{time.time() - t0:.0f}s]", flush=True)
    mols = pd.read_parquet(f"{a.lib}/molecules.parquet", columns=["inchikey14", "smiles", "exact_mass"])
    with get_context("fork").Pool(a.workers) as P:
        rows = list(P.imap(chem, mols.smiles, chunksize=500))
    C = pd.DataFrame([r if r else {} for r in rows])
    M = pd.concat([mols.rename(columns={"inchikey14": "ik14_lib"}).reset_index(drop=True), C], axis=1)
    log(f"chemistry for {len(M)} structures; RDKit failures {C.skey.isna().sum()}; "
        f"lib-key vs raw-InChIKey mismatches {(M.ik14_lib != M.ik14_raw).sum()}; skey != raw {(M.skey != M.ik14_raw).sum()}")

    # provenance from train.parquet (instrument, adducts and modes per structure)
    tr = ds.dataset(f"{COMP}/train.parquet").to_table(columns=["inchikey14", "ingest_lib", "instrument_type", "adduct", "ionization_mode"]).to_pandas()
    cnt = tr.groupby(["inchikey14", "ingest_lib"]).size().unstack(fill_value=0)
    cnt.columns = [f"n_spec_{c.replace('-', '_')}" for c in cnt.columns]
    g = tr.groupby("inchikey14")
    prov = pd.DataFrame({"n_libs": g.ingest_lib.nunique(),
                         "n_public_libs": tr[~tr.ingest_lib.isin(ENVEDA)].groupby("inchikey14").ingest_lib.nunique(),
                         "has_timstof": g.instrument_type.agg(lambda s: (s == "timsTOF").any()),
                         "n_inst_types": g.instrument_type.nunique(), "n_adducts": g.adduct.nunique(),
                         "n_pos": g.ionization_mode.agg(lambda s: int((s == "positive").sum())),
                         "n_neg": g.ionization_mode.agg(lambda s: int((s == "negative").sum()))}).join(cnt)
    prov["n_public_libs"] = prov.n_public_libs.fillna(0).astype(int)
    M = M.merge(prov, left_on="ik14_lib", right_index=True, how="left")
    del tr
    log("provenance")

    # novelty: public spectra (non-Enveda training libraries, MassBank-harmonised, extlib)
    mbh = pd.read_parquet(f"{MBH}/structures_with_fingerprints.parquet", columns=["inchikey"]).inchikey.str[:14]
    ext = pd.read_parquet(f"{EXT}/ext_meta.parquet", columns=["inchikey14"]).inchikey14
    M["c1_train_public"] = M.n_public_libs > 0
    M["c1_mbh"] = M.ik14_raw.isin(set(mbh)) | M.ik14_lib.isin(set(mbh))
    M["c1_extlib"] = M.ik14_raw.isin(set(ext)) | M.ik14_lib.isin(set(ext))
    M["c1_any"] = M.c1_train_public | M.c1_mbh | M.c1_extlib
    M["c1_enveda_only"] = ~M.c1_any

    # databases: PubChem tier (+ delta, growth) by raw InChIKey-14, COCONUT, PubChemLite
    pc_ik = np.load(f"{POP}/pc_ik14.npy", mmap_mode="r")
    order = np.argsort(pc_ik, kind="stable"); sk = np.asarray(pc_ik)[order]
    q = M.ik14_raw.fillna(M.ik14_lib).to_numpy().astype("S14")
    row = lookup(sk, order, q)
    q2 = M.ik14_lib.to_numpy().astype("S14")
    row = np.where(row >= 0, row, lookup(sk, order, q2))
    del sk, order
    extra = set()
    for pre in ("delta", "growth"):
        extra |= {x.decode() for x in np.load(f"{POP}/{pre}_pc_ik14.npy")}
    M["in_pubchem_tier"] = row >= 0
    M["in_pubchem"] = M.in_pubchem_tier | M.ik14_raw.isin(extra) | M.ik14_lib.isin(extra)
    ok = row >= 0
    for col, f in (("pc_cid", "pc_cid"), ("pc_lsid", "pc_lsid"), ("pc_lpmid", "pc_lpmid"), ("pc_lpatent", "pc_lpatent"),
                   ("pc_lsyn", "pc_lsynonyms"), ("pc_ndep", "pc_n_depositors"), ("pc_src_mask", "pc_src_mask"),
                   ("pc_np", "pc_np"), ("pc_formula_isomers", "pc_formula_isomers")):
        arr = np.load(f"{POP}/{f}.npy", mmap_mode="r")
        v = np.full(len(M), np.nan); v[ok] = np.asarray(arr[row[ok]], dtype=np.float64); M[col] = v
    coco = set(pd.read_parquet(f"{EXT}/coconut_index.parquet", columns=["inchikey14"]).inchikey14)
    pool = pd.read_parquet(f"{V4G}/casmi26-v2-pool/pool_meta.parquet", columns=["key", "src", "mass"])
    coco |= set(pool.key[pool.src == 2])
    pcl = set(pd.read_parquet(f"{POP}/pubchemlite_annotations.parquet", columns=["ik14"]).ik14)
    npbits = M.pc_np.fillna(0).astype(int)
    M["in_coconut"] = M.ik14_raw.isin(coco) | M.ik14_lib.isin(coco) | ((npbits & 1) > 0)
    M["in_lotus"] = (npbits & 2) > 0
    M["in_npatlas"] = (npbits & 4) > 0
    M["in_pcl"] = M.ik14_raw.isin(pcl) | M.ik14_lib.isin(pcl)
    clean = (M.n_frag == 1) & (M.formal_charge == 0) & (M.elem_class != "other") & M.exact_mass.between(150, 1250)
    M["nov_true"] = np.select([M.c1_any, M.in_pubchem | M.in_coconut | M.in_pcl, clean], ["1", "2", "3"], "3?")
    log(f"novelty {M.nov_true.value_counts().to_dict()}")

    # candidate density around each structure (10 ppm)
    pm = np.load(f"{POP}/casmi26-pubchem-tier/pc_mass.npy", mmap_mode="r")
    assert bool(np.all(np.diff(np.asarray(pm[:1_000_000])) >= 0)), "pc_mass not sorted (corrupt copy?)"
    tol = np.maximum(M.exact_mass.to_numpy() * 1e-5, 0.003)
    lo, hi = M.exact_mass.to_numpy() - tol, M.exact_mass.to_numpy() + tol
    M["n_tier_10ppm"] = np.searchsorted(pm, hi, side="right") - np.searchsorted(pm, lo)
    for name, mass in (("pool", np.sort(pool.mass.to_numpy())), ("coco", np.sort(pool.mass[pool.src == 2].to_numpy()))):
        M[f"n_{name}_10ppm"] = np.searchsorted(mass, hi, side="right") - np.searchsorted(mass, lo)
    M["mass_bin"] = pd.cut(M.exact_mass, [0, 157, 235, 348, 500, 800, 1159, 1e9],
                           labels=["<157", "157-235", "235-348", "348-500", "500-800", "800-1159", ">1159"]).astype(str)
    M["in_test_mass_range"] = M.exact_mass.between(157, 1159)
    M["chemotype"] = np.select([M.has_steroid_core & (M.has_sec_amide | M.has_ester), M.has_sec_amide & ~M.has_steroid_core,
                                M.has_glycoside], ["steroid_conj", "acyl_amide", "glycoside"], "other")
    M.to_parquet(f"{a.out}/molecules.parquet", index=False)
    log(f"wrote {a.out}/molecules.parquet {M.shape}")
    E = M[M.n_spec_enveda_180.fillna(0) > 0]
    print("\nEnveda-180 structures in the test mass range, by novelty class:")
    print(E[E.in_test_mass_range].nov_true.value_counts().to_string())
    print("\nnovelty x mass bin (all structures):")
    print(pd.crosstab(M.mass_bin, M.nov_true).to_string())


if __name__ == "__main__":
    main()
