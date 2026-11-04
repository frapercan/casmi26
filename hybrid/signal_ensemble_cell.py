# ── CASMI26 candidate-level signal table + learned ensemble (cross-fitted, gated) ──
# For every query molecule of a finished honest-evaluation run, each candidate in the pipeline's top-25 gets
# a row of signals: its position in the final list and in every pipeline variant, our model's heads (raw,
# z-score and rank within the molecule) and molecule-level context. A ranker learns to order the candidates;
# it is trained on disjoint molecule folds and applied blind to the held-out fold, then judged by the gate.
import glob, json, os
import numpy as np, pandas as pd
SRC = "__SRC__"
FOLDS, SEED, N_BOOT = 10, 0, 5000
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

R = os.path.dirname([p for p in glob.glob("/kaggle/input/**/val_rerank.parquet", recursive=True) if SRC in p][0])
T = pd.read_parquet(f"{R}/val_rerank.parquet")
A = pd.read_parquet(f"{R}/val_answers.parquet")
if "inchikey14" not in A:
    A["inchikey14"] = A.smiles.map(ik14)
TRUE = dict(zip(A.molecule_id, A.inchikey14))
HEADS_FROM = "__HEADS__"            # optional parquet whose head scores replace ours (e.g. the spectrum-blind control)
if HEADS_FROM and not HEADS_FROM.startswith("__"):
    Bf = pd.read_parquet(HEADS_FROM)
    hcols = [c for c in ["nce", "fp", "sae", "jepa", "fwd"] if c in Bf]
    T = T.drop(columns=hcols).merge(Bf[["molecule_id", "base_rank"] + hcols], on=["molecule_id", "base_rank"])
    print("head scores replaced from", HEADS_FROM)
FEATSET = "__FEATSET__"             # all | fp (pipeline + fp head only) | debias (heads minus their spectrum-blind score)
BLIND = "__BLIND__"
if FEATSET == "debias":
    Bb = pd.read_parquet(BLIND)
    hb = [c for c in ["nce", "fp", "sae", "jepa", "fwd"] if c in Bb and c in T]
    T = T.merge(Bb[["molecule_id", "base_rank"] + hb].rename(columns={h: h + "_blind" for h in hb}), on=["molecule_id", "base_rank"])
    for h in hb:
        T[h] = T[h] - T[h + "_blind"]       # spectrum-specific part of the score
    T = T.drop(columns=[h + "_blind" for h in hb])
    print("heads debiased with", BLIND)
if FEATSET == "fp":
    T = T.drop(columns=[c for c in ["nce", "sae", "jepa", "fwd"] if c in T])
T["ik"] = T.smiles.map(ik14)
T["hit"] = (T.ik == T.molecule_id.map(TRUE)).astype(int)
print(f"{T.molecule_id.nunique()} molecules, {len(T)} candidates, truth in list for {T.groupby('molecule_id').hit.max().mean():.3f}")

# ---- signals ----
heads = [h for h in ["nce", "fp", "sae", "jepa", "fwd"] if h in T]
g = T.groupby("molecule_id")
for h in heads:
    T[f"{h}_z"] = (T[h] - g[h].transform("mean")) / (g[h].transform("std") + 1e-6)
    T[f"{h}_rk"] = g[h].rank(ascending=False, method="first")
    T[f"{h}_gap"] = T[h] - g[h].transform("max")
T["inv_base"] = 1 / T.base_rank
T["n_cand"] = g.smiles.transform("size")
VAR = {}
for p in sorted(glob.glob(f"{R}/*.csv") + glob.glob(f"{R}/variants/*.csv")):
    s = pd.read_csv(p, dtype=str, keep_default_na=False)
    if not {"molecule_id", "smiles"} <= set(s.columns) or not set(TRUE) <= set(s.molecule_id):
        continue
    name = "v_" + os.path.splitext(os.path.relpath(p, R))[0].replace("/", "_")
    pos = {}
    for m, x in zip(s.molecule_id, s.smiles):
        if m in TRUE:
            for i, sm in enumerate([y for y in x.split(";") if y.strip()][:25]):
                pos.setdefault((m, ik14(sm)), i + 1)
    T[name] = [pos.get((m, k), 26) for m, k in zip(T.molecule_id, T.ik)]
    VAR[name] = p
vcols = list(VAR)
T["v_mean"] = T[vcols].mean(axis=1); T["v_min"] = T[vcols].min(axis=1); T["v_top1_votes"] = (T[vcols] == 1).sum(axis=1)
Q = pd.read_parquet(glob.glob(f"{R}/**/test.parquet", recursive=True)[0])
mq = Q.groupby("molecule_id").agg(mass=("precursor_mz", "median"), n_spec=("precursor_mz", "size"))
T = T.join(mq, on="molecule_id")
FEATS = (["base_rank", "inv_base", "n_cand", "v_mean", "v_min", "v_top1_votes", "mass", "n_spec"] + vcols
         + [f"{h}{s}" for h in heads for s in ["", "_z", "_rk", "_gap"]])
print("features:", len(FEATS))
T.to_parquet("/kaggle/working/signal_table.parquet", index=False)

def mrr_of(order_col, df, ascending=True):
    out = {}
    for m, d in df.groupby("molecule_id"):
        d = d.sort_values(order_col, ascending=ascending, kind="stable")
        hit = np.flatnonzero(d.hit.to_numpy())
        out[m] = 1 / (hit[0] + 1) if len(hit) else 0.0
    return pd.Series(out)

APPLY = "__APPLY__"                 # kernel holding a trained ranker.txt: apply it here (transfer test), then stop
if APPLY and not APPLY.startswith("__"):
    import lightgbm as lgb
    _rk = [p for p in glob.glob("/kaggle/input/**/ranker.txt", recursive=True) if APPLY in p][0]
    _meta = json.load(open(os.path.join(os.path.dirname(_rk), "ranker_meta.json")))
    for c in _meta["feats"]:
        if c not in T:
            print("feature missing here, filled with 26:", c); T[c] = 26
    T["app"] = lgb.Booster(model_file=_rk).predict(T[_meta["feats"]])
    T["app_keep1"] = np.where(T.base_rank == 1, np.inf, T.app)
    cls = A.set_index("molecule_id").cls
    for name, col, asc in [("base", "base_rank", True), ("ranker", "app", False), ("ranker, rank 1 kept", "app_keep1", False)]:
        r = mrr_of(col, T, ascending=asc)
        print(f"TRANSFER {name:<22s} all {r.mean():.4f}  " + "  ".join(f"class{k} {v:.4f}" for k, v in r.groupby(cls.reindex(r.index)).mean().items()))
    raise SystemExit
base = mrr_of("base_rank", T)
mols = np.array(sorted(T.molecule_id.unique()))
fold_of = dict(zip(mols, rng.permutation(len(mols)) % FOLDS))
T["fold"] = T.molecule_id.map(fold_of)

def gate(rr, name):
    d = (rr - base.reindex(rr.index)).to_numpy()
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    fd = [d[np.array([fold_of[m] == f for m in rr.index])].mean() for f in range(FOLDS)]
    pos = sum(x > 0 for x in fd)
    ok = pos >= 8 and lo > 0
    print(f"{name:<34s} MRR {rr.mean():.4f}  delta {d.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  folds+ {pos}/10  "
          f"GATE {'PASS' if ok else 'fail'}", flush=True)
    return dict(arm=name, mrr=float(rr.mean()), delta=float(d.mean()), ci=[float(lo), float(hi)], folds_pos=int(pos), gate=bool(ok))

res = [gate(base, "base (pipeline order)")]
# 1) cross-fitted linear blend: base RRF + w * fp rank (w chosen on the other folds)
W = [0, .1, .25, .5, .75, 1]
def blend(df, w, h="fp"):
    return 1 / (3 + df.base_rank) + w / (3 + df[f"{h}_rk"])
for h in [x for x in ["fp", "nce"] if x in heads]:
    pick = {}
    for f in range(FOLDS):
        tr = T[T.fold != f]
        sc = {w: mrr_of("s", tr.assign(s=blend(tr, w, h)), ascending=False).mean() for w in W}
        pick[f] = max(sc, key=sc.get)
    T["s"] = [blend(r, pick[r.fold], h) for _, r in T[["base_rank", f"{h}_rk", "fold"]].iterrows()]
    res.append(gate(mrr_of("s", T, ascending=False), f"cross-fitted RRF blend ({h})"))
    print("   weights chosen per fold:", list(pick.values()))
# 2) learned ranker (LambdaRank), cross-fitted
import lightgbm as lgb
T["lgb"] = np.nan
imp = np.zeros(len(FEATS))
for f in range(FOLDS):
    tr, te = T[T.fold != f].sort_values("molecule_id"), T[T.fold == f]
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15,
                        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                        lambdarank_truncation_level=25, random_state=SEED, verbose=-1)
    rk.fit(tr[FEATS], tr.hit, group=tr.groupby("molecule_id", sort=True).size().to_numpy())
    T.loc[te.index, "lgb"] = rk.predict(te[FEATS])
    imp += rk.booster_.feature_importance("gain")
res.append(gate(mrr_of("lgb", T, ascending=False), "LambdaRank on all signals"))
# leader-safe: the pipeline's rank 1 stays, the ranker only re-orders ranks 2-25 (class-1 library hits are safe)
T["lgb_keep1"] = np.where(T.base_rank == 1, np.inf, T.lgb)
res.append(gate(mrr_of("lgb_keep1", T, ascending=False), "LambdaRank, rank 1 kept"))
print("top features (gain):", [FEATS[i] for i in np.argsort(-imp)[:12]])
# 3) same ranker without our model's heads: how much do they add?
T["lgb0"] = np.nan
F0 = [c for c in FEATS if not any(c.startswith(h) for h in heads)]
for f in range(FOLDS):
    tr, te = T[T.fold != f].sort_values("molecule_id"), T[T.fold == f]
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15,
                        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                        random_state=SEED, verbose=-1)
    rk.fit(tr[F0], tr.hit, group=tr.groupby("molecule_id", sort=True).size().to_numpy())
    T.loc[te.index, "lgb0"] = rk.predict(te[F0])
res.append(gate(mrr_of("lgb0", T, ascending=False), "LambdaRank, pipeline signals only"))
# 4) control: same ranker trained on labels shuffled within each molecule must not beat the base
T["hit_perm"] = T.groupby("molecule_id").hit.transform(lambda x: rng.permutation(x.to_numpy()))
T["lgbp"] = np.nan
for f in range(FOLDS):
    tr, te = T[T.fold != f].sort_values("molecule_id"), T[T.fold == f]
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15,
                        min_child_samples=40, random_state=SEED, verbose=-1)
    rk.fit(tr[FEATS], tr.hit_perm, group=tr.groupby("molecule_id", sort=True).size().to_numpy())
    T.loc[te.index, "lgbp"] = rk.predict(te[FEATS])
res.append(gate(mrr_of("lgbp", T, ascending=False), "CONTROL: shuffled labels"))
json.dump(res, open("/kaggle/working/signal_ensemble.json", "w"), indent=1)
# final ranker on every molecule, for the submission notebook (same features, same hyper-parameters)
fin = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15,
                     min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                     lambdarank_truncation_level=25, random_state=SEED, verbose=-1)
Ts = T.sort_values("molecule_id")
fin.fit(Ts[FEATS], Ts.hit, group=Ts.groupby("molecule_id", sort=True).size().to_numpy())
fin.booster_.save_model("/kaggle/working/ranker.txt")
json.dump({"feats": FEATS, "vcols": vcols, "heads": heads, "featset": FEATSET}, open("/kaggle/working/ranker_meta.json", "w"))
print("FINAL ranker saved:", len(FEATS), "features")
print("SIGNAL ENSEMBLE DONE")
