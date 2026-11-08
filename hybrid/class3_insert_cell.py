# ── CASMI26 class 3: insertion policy for generated candidates, simulated on the class-3 cell ──
# Runs after class3_score_cell.py (S: generated candidates with model heads). The pipeline's lists come from
# casmi26-eval-c3 (class-3 structures hidden from the library, but still present in the external candidate
# databases). Two scenarios per molecule:
#   DB  - as observed: the truth may be in the pipeline list (it is, for most of these NP molecules)
#   C3  - true class 3: the truth is removed from the pipeline list (it is in no database)
# A policy must help C3 without costing DB. Scores: pipeline top-25 and generated candidates are scored by the
# class-3-safe model together, so their heads are on one per-molecule z scale.
import glob
from casmi.metric import ik14 as _ik14
_ev = sorted([p for p in glob.glob("/kaggle/input/**/submission.csv", recursive=True) if "casmi26-eval-c3" in p], key=len)[0]
_evd = os.path.dirname(_ev)
VA = pd.read_parquet(glob.glob(f"{_evd}/**/val_answers.parquet", recursive=True)[0] if glob.glob(f"{_evd}/**/val_answers.parquet", recursive=True) else f"{_evd}/val_answers.parquet")
_sub = pd.read_csv(_ev, dtype=str, keep_default_na=False)
_m2v = dict(zip(VA.molecule_id, VA.val3_id))
TRUE3 = {v: _ik14(x) for v, x in zip(VA.val3_id, VA.smiles)}   # official key (see key aliases)
PL = {_m2v[m]: [s for s in x.split(";") if s.strip()][:25] for m, x in zip(_sub.molecule_id, _sub.smiles) if m in _m2v}
_kc = {}
def K(s):
    if s not in _kc:
        try: _kc[s] = _ik14(s)
        except Exception: _kc[s] = None
    return _kc[s]
mids3 = [m for m in PL if m in qg]
print(f"INSERT: {len(mids3)} molecules with a pipeline list and queries; generated sets for {S.molecule_id.nunique()}", flush=True)
rows = []
with torch.no_grad():
    for k, m in enumerate(mids3):
        pl = PL[m]; pk = [K(s) for s in pl]
        g = S[S.molecule_id == m]
        gs, gk, seen = [], [], set()   # dedup against the pipeline list happens per scenario
        for x in g.smiles:
            kx = K(x)
            if kx is not None and kx not in seen:
                seen.add(kx); gs.append(x); gk.append(kx)
        smi = pl + gs; iks = pk + gk
        if not smi:
            continue
        sc = {}
        for i in range(0, len(smi), 8192):
            part = model_scores(model, sae, query_batch(qg[m], dev), *feat(iks[i:i+8192], smi[i:i+8192]))
            for h, v in part.items(): sc.setdefault(h, []).append(v)
        sc = {h: np.concatenate(v) for h, v in sc.items()}
        for i in range(len(smi)):
            rows.append(dict(m=m, src="pl" if i < len(pl) else "gen", pos=i + 1 if i < len(pl) else 0, ik=iks[i],
                             hit=iks[i] == TRUE3[m], fwd=float(sc["fwd"][i]), fp=float(sc["fp"][i])))
        if (k + 1) % 100 == 0: print(f"  {k+1} [{time.time()-t0:.0f}s]", flush=True)
U = pd.DataFrame(rows)
g_ = U.groupby("m")
for h in ["fwd", "fp"]:
    U[f"z{h}"] = (U[h] - g_[h].transform("mean")) / (g_[h].transform("std").fillna(0) + 1e-6)
U["s"] = U.zfwd + U.zfp
U.to_parquet("/kaggle/working/class3_union.parquet", index=False)

def rr(lst_hits):
    h = np.flatnonzero(np.asarray(lst_hits[:25], bool)); return 1 / (h[0] + 1) if len(h) else 0.0
def evaluate(policy):
    out = {"DB": [], "C3": []}
    for m, d in U.groupby("m", sort=False):
        P = d[d.src == "pl"].sort_values("pos"); G = d[d.src == "gen"].sort_values("s", ascending=False)
        for scen in ("DB", "C3"):
            P_ = P if scen == "DB" else P[~P.hit]
            out[scen].append(rr(policy(P_, G[~G.ik.isin(set(P_.ik))])))
    return {k: np.array(v) for k, v in out.items()}
def ins(p, k):
    return lambda P, G: list(P.hit[:p - 1]) + list(G.hit[:k]) + list(P.hit[p - 1:])
def margin(t, p, k):
    def f(P, G):
        if len(G) and len(P) and G.s.iloc[0] > P.s.max() + t:
            return list(G.hit[:1]) + list(P.hit[:p - 2]) + list(G.hit[1:k]) + list(P.hit[p - 2:])
        return ins(p, k)(P, G)
    return f
def union_tail(q):
    def f(P, G):
        rest = pd.concat([P.iloc[q:], G]).sort_values("s", ascending=False)
        return list(P.hit[:q]) + list(rest.hit)
    return f
POL = {"pipeline": lambda P, G: list(P.hit)}
for p in (2, 3, 5, 10, 20):
    for k in (1, 3, 5):
        POL[f"ins p={p} k={k}"] = ins(p, k)
for q in (1, 3, 5, 10):
    POL[f"union_tail q={q}"] = union_tail(q)
for t in (0.0, 0.5, 1.0, 2.0):
    POL[f"margin t={t} p=3 k=3"] = margin(t, 3, 3)
base = evaluate(POL["pipeline"])
rng = np.random.default_rng(0)
print(f"\n{'policy':<24s}  DB MRR   dDB [95% CI]              C3 MRR   dC3")
res3 = {}
for name, pol in POL.items():
    r = evaluate(pol); d = r["DB"] - base["DB"]
    lo, hi = np.percentile(d[rng.integers(0, len(d), (4000, len(d)))].mean(1), [2.5, 97.5])
    res3[name] = dict(db=float(r["DB"].mean()), ddb=float(d.mean()), ci=[float(lo), float(hi)], c3=float(r["C3"].mean()))
    print(f"{name:<24s}  {r['DB'].mean():.4f}  {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]   {r['C3'].mean():.4f}  {r['C3'].mean() - base['C3'].mean():+.4f}", flush=True)
# how separable is "truth absent from the pipeline list" by our model's view of the list?
_pk = set(zip(U.m[U.src == "pl"], U.ik[U.src == "pl"]))
_gn = U[(U.src == "gen") & ~pd.Series(list(zip(U.m, U.ik)), index=U.index).isin(_pk)]
P1 = U[U.src == "pl"].groupby("m").s.max(); G1 = _gn.groupby("m").s.max()
gap = (G1 - P1).dropna()
inl = U[U.src == "pl"].groupby("m").hit.any()
print("\nbest-generated minus best-pipeline score (higher = model prefers a generated one):")
print("  truth in pipeline list:", gap[inl.reindex(gap.index).fillna(False)].describe()[["count", "mean", "50%"]].round(3).to_dict())
print("  truth NOT in list     :", gap[~inl.reindex(gap.index).fillna(False)].describe()[["count", "mean", "50%"]].round(3).to_dict())
json.dump(res3, open("/kaggle/working/class3_insert.json", "w"), indent=1)
print(f"INSERT DONE [{time.time()-t0:.0f}s]")
