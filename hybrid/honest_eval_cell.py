# ── CASMI26 honest evaluation: strata, oracle ceiling, router (CPU, minutes) ──
# Inputs: finished harness runs (casmi26-eval-ext*: pipeline variant CSVs, val_rerank.parquet,
# val_answers.parquet, valcomp/test.parquet), attached as kernel sources.
# Protocol (after PROTEA / cafaeval): every arm is scored per molecule with the official MRR@25
# rule; strata use only test-time observables (routing-eligible) or the true class (evaluation
# only); differences are paired bootstrap CIs; a router is cross-fitted on disjoint molecule
# folds and must pass a gate (positive in >= 8/10 folds and CI lower bound > 0) to count.
import glob, json, os, sys
import numpy as np, pandas as pd
ROOTS = __ROOTS__                       # {label: substring identifying the kernel source}
BASE = "__BASE__"                       # arm everything is compared against
N_BOOT, SEED, FOLDS = 5000, 0, 10
rng = np.random.default_rng(SEED)
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
_te = rdMolStandardize.TautomerEnumerator()
_ikc = {}
def ik14(s):                            # official canonicalisation (metric.py)
    if s not in _ikc:
        try:
            m = Chem.MolFromSmiles(s); Chem.RemoveStereochemistry(m)
            _ikc[s] = Chem.MolToInchiKey(_te.Canonicalize(m))[:14] or None
        except Exception:
            _ikc[s] = None
    return _ikc[s]

def rr_of(lists, true):
    out = {}
    for m, lst in lists.items():
        lst = [s.strip() for s in lst if s.strip()][:25]
        r = next((i + 1 for i, s in enumerate(lst) if ik14(s) == true[m]), 0)
        out[m] = 1 / r if r else 0.0
    return pd.Series(out)

# ---------- load arms ----------
ANS, TEST, RR, TOP1 = None, None, {}, {}
for lab, key in ROOTS.items():
    hits = [p for p in glob.glob("/kaggle/input/**/val_answers.parquet", recursive=True) if key in p]
    if not hits:
        print("missing source", lab, key); continue
    R = os.path.dirname(hits[0])
    a = pd.read_parquet(f"{R}/val_answers.parquet")
    if "inchikey14" not in a:
        a["inchikey14"] = a.smiles.map(ik14)
    if ANS is None:
        ANS, TEST = a, pd.read_parquet(glob.glob(f"{R}/**/test.parquet", recursive=True)[0])
    true = dict(zip(a.molecule_id, a.inchikey14))
    for p in sorted(glob.glob(f"{R}/*.csv") + glob.glob(f"{R}/variants/*.csv")):
        s = pd.read_csv(p, dtype=str, keep_default_na=False)
        if not {"molecule_id", "smiles"} <= set(s.columns) or not set(true) <= set(s.molecule_id):
            continue
        L = {m: x.split(";") for m, x in zip(s.molecule_id, s.smiles) if m in true}
        name = f"{lab}:{os.path.relpath(p, R)}"
        RR[name] = rr_of(L, true)
        TOP1[name] = pd.Series({m: ik14(l[0]) if l and l[0] else None for m, l in L.items()})
    vr = f"{R}/val_rerank.parquet"
    if os.path.exists(vr):                      # our heads alone, re-ordering the pipeline's top-25
        T = pd.read_parquet(vr)
        for k in [c for c in ["nce", "fp", "sae", "jepa", "fwd"] if c in T]:
            L = {m: g.sort_values(k, ascending=False).smiles.tolist() for m, g in T.groupby("molecule_id")}
            RR[f"{lab}:ours-{k}"] = rr_of(L, true)
RR = pd.DataFrame(RR).fillna(0.0)
print(f"{len(RR)} molecules x {RR.shape[1]} arms; base = {BASE}")
assert BASE in RR, sorted(RR)

# ---------- strata ----------
q = TEST.copy()
q["npk"] = q.ms2_mzs.map(len)
g = q.groupby("molecule_id")
S = pd.DataFrame({
    "mass": g.precursor_mz.median(),
    "n_spec": g.size(),
    "neg": g.ionization_mode.agg(lambda x: (x == "negative").any()),
    "stepped_ce": g.collision_energy_ev.agg(lambda x: any(len(v) > 1 for v in x)),
    "peaks": g.npk.median(),
}).reindex(RR.index)
S["mass_bin"] = pd.qcut(S.mass, 4, labels=["Q1 light", "Q2", "Q3", "Q4 heavy"])
S["peaks_bin"] = pd.qcut(S.peaks.rank(method="first"), 3, labels=["few peaks", "mid", "many peaks"])
S["spec_bin"] = pd.cut(S.n_spec, [0, 1, 3, 1e9], labels=["1 spec", "2-3", "4+"])
_t1 = pd.DataFrame(TOP1).reindex(RR.index)
S["agree"] = _t1.eq(_t1[BASE], axis=0).mean(axis=1)          # share of arms agreeing with the base top-1
S["agree_bin"] = pd.cut(S.agree, [-1, .5, .9, 2], labels=["experts disagree", "partial", "consensus"])
S["true_cls"] = ANS.set_index("molecule_id").cls.reindex(RR.index)   # evaluation-only

def boot_ci(d):
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    m = d.to_numpy()[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

# ---------- 1. arms vs base, paired ----------
print("\n## Arms vs base (paired bootstrap 95% CI of the MRR difference)")
rows = []
for a_ in RR:
    d = RR[a_] - RR[BASE]
    lo, hi = boot_ci(d)
    rows.append(dict(arm=a_, mrr=RR[a_].mean(), delta=d.mean(), ci_lo=lo, ci_hi=hi,
                     wins=(d > 0).sum(), losses=(d < 0).sum()))
A = pd.DataFrame(rows).sort_values("mrr", ascending=False)
print(A.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------- 2. oracle ceiling ----------
print("\n## Oracle ceiling (best arm per molecule)")
orc = RR.max(axis=1)
print(f"base {RR[BASE].mean():.4f}  oracle(all arms) {orc.mean():.4f}  "
      f"(+{orc.mean() - RR[BASE].mean():.4f})  molecules where some arm beats base: {(orc > RR[BASE]).sum()}/{len(RR)}")
pair = {a_: np.maximum(RR[a_], RR[BASE]).mean() - RR[BASE].mean() for a_ in RR if a_ != BASE}
print("best partners for a base+1 oracle:", {k: round(v, 4) for k, v in sorted(pair.items(), key=lambda x: -x[1])[:6]})

# ---------- 3. per stratum ----------
top_arms = [BASE] + [a_ for a_ in A.arm if a_ != BASE][:4]
for col in ["true_cls", "mass_bin", "peaks_bin", "spec_bin", "neg", "stepped_ce", "agree_bin"]:
    print(f"\n## Stratum: {col}" + ("  (evaluation-only)" if col == "true_cls" else "  (observable -> routing)"))
    t = RR[top_arms].groupby(S[col], observed=True).mean()
    t.insert(0, "n", S[col].value_counts())
    t["oracle"] = orc.groupby(S[col], observed=True).mean()
    print(t.to_string(float_format=lambda x: f"{x:.3f}"))

# ---------- 4. cross-fitted router with a gate ----------
print("\n## Router (cross-fitted, gated)")
from sklearn.tree import DecisionTreeRegressor
X = pd.get_dummies(S[["mass", "n_spec", "neg", "stepped_ce", "peaks", "agree"]].astype(float))
cand = top_arms
fold = rng.permutation(len(RR)) % FOLDS
gain, picks = np.zeros(len(RR)), pd.Series(BASE, index=RR.index)
fold_delta = []
for f in range(FOLDS):
    tr, te = fold != f, fold == f
    pred = {}
    for a_ in cand:  # predicted advantage of each arm over the base, from observables only
        y = (RR[a_] - RR[BASE]).to_numpy()
        m = DecisionTreeRegressor(max_depth=2, min_samples_leaf=max(10, tr.sum() // 20), random_state=SEED)
        pred[a_] = m.fit(X[tr], y[tr]).predict(X[te])
    P = pd.DataFrame(pred, index=RR.index[te])
    choice = P.idxmax(axis=1).where(P.max(axis=1) > 0, BASE)
    picks[te] = choice
    got = np.array([RR.loc[m, c] for m, c in choice.items()])
    fold_delta.append(float((got - RR[BASE][te]).mean()))
routed = pd.Series([RR.loc[m, c] for m, c in picks.items()], index=RR.index)
lo, hi = boot_ci(routed - RR[BASE])
pos = sum(d > 0 for d in fold_delta)
gate = pos >= 8 and lo > 0
print(f"routed {routed.mean():.4f} vs base {RR[BASE].mean():.4f}: delta {routed.mean() - RR[BASE].mean():+.4f} "
      f"CI [{lo:+.4f}, {hi:+.4f}]  folds positive {pos}/{FOLDS} (sd {np.std(fold_delta):.4f})  GATE {'PASS' if gate else 'FAIL'}")
print("arm usage:", picks.value_counts().to_dict())
json.dump({"arms": A.to_dict("records"), "oracle": float(orc.mean()), "base": float(RR[BASE].mean()),
           "router": {"mrr": float(routed.mean()), "ci": [lo, hi], "folds_positive": pos, "gate": bool(gate)}},
          open("/kaggle/working/honest_eval.json", "w"), indent=1, default=str)
RR.to_parquet("/kaggle/working/honest_eval_rr.parquet"); S.to_parquet("/kaggle/working/honest_eval_strata.parquet")
print("\nHONEST EVAL DONE")
