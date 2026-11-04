# ── CASMI26 cell-aware ranker: train on class 1 + class 2 together, gate per cell ──
# Class 1 (structure in the library, new spectrum): validation molecules of the train-based harness. Leakage
# does not distort this cell, because in the hidden test the models have also seen class-1 structures.
# Class 2 (structure only in databases): the leak-free Enveda-180 set.
# A change ships only if class 2 improves (CI > 0, >= 8/10 folds) AND class 1 does not degrade.
import glob, json, os
import numpy as np, pandas as pd
import lightgbm as lgb
SRCS = __SRCS__          # {cell: (kernel substring, keep only this answers.cls or None)}
FOLDS, SEED, N_BOOT = 10, 0, 5000
MIX = {"public mix (c1 .18, c2 .07)": {1: .18, 2: .07}, "c2-heavy (c1 .15, c2 .25)": {1: .15, 2: .25}}
rng = np.random.default_rng(SEED)
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
_te = rdMolStandardize.TautomerEnumerator(); _ikc = {}
def ik14(s):
    if s not in _ikc:
        try:
            m = Chem.MolFromSmiles(s); Chem.RemoveStereochemistry(m)
            _ikc[s] = Chem.MolToInchiKey(_te.Canonicalize(m))[:14] or None
        except Exception:
            _ikc[s] = None
    return _ikc[s]

def load(cell, key, keep_cls):
    R = os.path.dirname([p for p in glob.glob("/kaggle/input/**/val_rerank.parquet", recursive=True) if key in p][0])
    T = pd.read_parquet(f"{R}/val_rerank.parquet")
    A = pd.read_parquet(f"{R}/val_answers.parquet")
    if "inchikey14" not in A:
        A["inchikey14"] = A.smiles.map(ik14)
    if keep_cls is not None:
        A = A[A.cls == keep_cls]
    T = T[T.molecule_id.isin(set(A.molecule_id))].drop(columns=[c for c in ["nce", "sae", "jepa", "fwd"] if c in T])
    true = dict(zip(A.molecule_id, A.inchikey14))
    T["ik"] = T.smiles.map(ik14)
    T["hit"] = (T.ik == T.molecule_id.map(true)).astype(int)
    g = T.groupby("molecule_id")
    T["fp_z"] = (T.fp - g.fp.transform("mean")) / (g.fp.transform("std") + 1e-6)
    T["fp_rk"] = g.fp.rank(ascending=False, method="first")
    T["fp_gap"] = T.fp - g.fp.transform("max")
    T["inv_base"] = 1 / T.base_rank
    T["n_cand"] = g.smiles.transform("size")
    for p in sorted(glob.glob(f"{R}/*.csv") + glob.glob(f"{R}/variants/*.csv")):
        s = pd.read_csv(p, dtype=str, keep_default_na=False)
        if not {"molecule_id", "smiles"} <= set(s.columns) or not set(true) <= set(s.molecule_id):
            continue
        name = "v_" + os.path.splitext(os.path.relpath(p, R))[0].replace("/", "_")
        pos = {}
        for m, x in zip(s.molecule_id, s.smiles):
            if m in true:
                for i, sm in enumerate([y for y in x.split(";") if y.strip()][:25]):
                    pos.setdefault((m, ik14(sm)), i + 1)
        T[name] = [pos.get((m, k), 26) for m, k in zip(T.molecule_id, T.ik)]
    Q = pd.read_parquet(glob.glob(f"{R}/**/test.parquet", recursive=True)[0])
    T = T.join(Q.groupby("molecule_id").agg(mass=("precursor_mz", "median"), n_spec=("precursor_mz", "size")), on="molecule_id")
    T["cell"] = cell
    T["molecule_id"] = f"{cell}:" + T.molecule_id
    print(f"cell {cell}: {T.molecule_id.nunique()} molecules from {R}; truth in top-25 {T.groupby('molecule_id').hit.max().mean():.3f}")
    return T

T = pd.concat([load(c, k, kc) for c, (k, kc) in SRCS.items()], ignore_index=True)
vcols = sorted(c for c in T if c.startswith("v_") and T[c].notna().all())
T["v_mean"] = T[vcols].mean(axis=1); T["v_min"] = T[vcols].min(axis=1); T["v_top1_votes"] = (T[vcols] == 1).sum(axis=1)
FEATS = ["base_rank", "inv_base", "n_cand", "v_mean", "v_min", "v_top1_votes", "mass", "n_spec", "fp", "fp_z", "fp_rk", "fp_gap"] + vcols
print("features", len(FEATS), "| variant columns shared by every cell:", len(vcols))

def mrr_of(col, df, asc):
    out = {}
    for m, d in df.groupby("molecule_id"):
        hit = np.flatnonzero(d.sort_values(col, ascending=asc, kind="stable").hit.to_numpy())
        out[m] = 1 / (hit[0] + 1) if len(hit) else 0.0
    return pd.Series(out)

mols = T[["molecule_id", "cell"]].drop_duplicates()
fold = {}
for c, d in mols.groupby("cell"):          # folds stratified by cell
    for m, f in zip(d.molecule_id, rng.permutation(len(d)) % FOLDS):
        fold[m] = f
T["fold"] = T.molecule_id.map(fold)
cell_of = dict(zip(mols.molecule_id, mols.cell))
base = mrr_of("base_rank", T, True)

def fit_predict(train_cells, keep1=False, name=""):
    col = f"s_{name}"
    T[col] = np.nan
    for f in range(FOLDS):
        tr = T[(T.fold != f) & T.cell.isin(train_cells)].sort_values("molecule_id")
        te = T[T.fold == f]
        rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15,
                            min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            lambdarank_truncation_level=25, random_state=SEED, verbose=-1)
        rk.fit(tr[FEATS], tr.hit, group=tr.groupby("molecule_id", sort=True).size().to_numpy())
        T.loc[te.index, col] = rk.predict(te[FEATS])
    if keep1:
        T[col] = np.where(T.base_rank == 1, np.inf, T[col])
    return mrr_of(col, T, False)

def report(rr, name):
    d = rr - base
    out = {"arm": name}
    line = f"{name:<40s}"
    for c in sorted(T.cell.unique()):
        dc = d[[cell_of[m] == c for m in d.index]]
        idx = rng.integers(0, len(dc), (N_BOOT, len(dc)))
        lo, hi = np.percentile(dc.to_numpy()[idx].mean(1), [2.5, 97.5])
        fpos = sum(dc[[fold[m] == f for m in dc.index]].mean() > 0 for f in range(FOLDS))
        out[c] = dict(mrr=float(rr[dc.index].mean()), delta=float(dc.mean()), ci=[float(lo), float(hi)], folds_pos=int(fpos))
        line += f" | {c}: {rr[dc.index].mean():.4f} ({dc.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {fpos}/10)"
    c1, c2 = out.get("class1"), out.get("class2")
    ok = bool(c2 and c2["ci"][0] > 0 and c2["folds_pos"] >= 8 and (c1 is None or (c1["delta"] >= -0.002 and c1["ci"][0] > -0.01)))
    out["gate"] = ok
    for mn, w in MIX.items():
        line += f" | LB-est {mn}: {sum(w[int(c[-1])] * out[c]['delta'] for c in out if c.startswith('class')):+.4f}"
    print(line + f" | GATE {'PASS' if ok else 'fail'}", flush=True)
    return out

res = [report(base, "base")]
res.append(report(fit_predict(["class2"], name="c2only"), "ranker trained on class 2 only"))
res.append(report(fit_predict(["class2"], keep1=True, name="c2keep"), "  ... rank 1 kept"))
res.append(report(fit_predict(["class1", "class2"], name="both"), "ranker trained on class 1 + class 2"))
res.append(report(fit_predict(["class1", "class2"], keep1=True, name="bothkeep"), "  ... rank 1 kept"))
json.dump(res, open("/kaggle/working/cell_ranker.json", "w"), indent=1)
print("CELL RANKER DONE")
