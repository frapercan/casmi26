# ── casmi26 validation harness: score the pipeline's lists, then sweep our re-rank weight ──
import base64, glob, io, itertools, json, os, sys, tarfile, time, traceback
import numpy as np, pandas as pd
_t0 = time.time()
_code = "/kaggle/working/casmi26_ours"
tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__"))).extractall(_code)
sys.path.insert(0, _code + "/src")
from casmi.metric import ik14
ANS = pd.read_parquet("/kaggle/working/val_answers.parquet")
TRUE = dict(zip(ANS.molecule_id, ANS.inchikey14)); CLS = dict(zip(ANS.molecule_id, ANS.cls))
_ikc = {}
def _ik(s):
    if s not in _ikc:
        _ikc[s] = ik14(s) if s else None
    return _ikc[s]
def hit_rank(lst, mid):  # official scorer: rank = position among non-empty guesses, first hit counts
    lst = [s.strip() for s in lst if s.strip()][:25]
    for i, s in enumerate(lst):
        if _ik(s) == TRUE[mid]:
            return i + 1
    return 0
def report(name, ranks):
    r = pd.Series(ranks)
    rr = r.map(lambda x: 1 / x if x else 0.0)
    c = r.index.map(CLS)
    per = {int(k): round(float(rr[c == k].mean()), 4) for k in sorted(set(c))}
    print(f"VALMRR {name:<34s} all {rr.mean():.4f}  " + "  ".join(f"class{k} {v:.4f}" for k, v in per.items())
          + f"  top1 {(r == 1).mean():.3f}  in25 {(r > 0).mean():.3f}", flush=True)
    return per
LISTS = {}
for p in sorted(set(glob.glob("/kaggle/working/*.csv") + glob.glob("/kaggle/working/**/submission_*.csv", recursive=True))):
    try:
        s = pd.read_csv(p, dtype=str, keep_default_na=False)
        if not {"molecule_id", "smiles"} <= set(s.columns) or not set(s.molecule_id) >= set(TRUE):
            continue
        L = {m: x.split(";") for m, x in zip(s.molecule_id, s.smiles) if m in TRUE}
        report(os.path.relpath(p, "/kaggle/working"), {m: hit_rank(l, m) for m, l in L.items()})
        if os.path.basename(p) == "submission.csv":
            LISTS = L
    except Exception as e:
        print("skip", p, e)
print(f"scored pipeline files in {time.time() - _t0:.0f}s", flush=True)

try:
    import torch
    from casmi.plm import PLMEmbedder
    from casmi.predict import MolFeaturizer, load_model, model_scores, query_batch
    from casmi.search import prep_query
    _ck = sorted(glob.glob("/kaggle/input/**/work/runs/run/best.pt", recursive=True))
    _ck = [h for h in _ck if "casmi26-train-v3" in h] or _ck
    _W = _ck[0][:-len("/runs/run/best.pt")]
    _plm = [os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/work/plm/config.json", recursive=True))]
    _mean = [p for p in [f"{_W}/emb/plm_mean.npy"] + sorted(glob.glob("/kaggle/input/**/work/emb/plm_mean.npy", recursive=True)) if os.path.exists(p)]
    print("OURS ckpt", _W, "| plm", _plm[0], "| mean", _mean[0], flush=True)
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    _model, _sae = load_model(f"{_W}/runs/run/best.pt", f"{_W}/runs/sae.pt" if os.path.exists(f"{_W}/runs/sae.pt") else "", _dev)
    _feat = MolFeaturizer(PLMEmbedder(_plm[0], device=_dev, mean=np.load(_mean[0])), _dev)
    _test = pd.read_parquet("/kaggle/working/valcomp/test.parquet")
    _q = prep_query(_test)
    _q["ce_mean"] = _test.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
    _qg = dict(tuple(_q.groupby("molecule_id", sort=False)))
    rows = []
    for mid, lst in LISTS.items():
        lst = [s for s in lst if s][:25]
        g = _qg.get(mid)
        if g is None or not lst:
            continue
        sc = model_scores(_model, _sae, query_batch(g, _dev), *_feat(lst, lst))
        for i, s in enumerate(lst):
            rows.append(dict(molecule_id=mid, base_rank=i + 1, smiles=s, ik=_ik(s), hit=_ik(s) == TRUE[mid],
                             **{k: float(v[i]) for k, v in sc.items()}))
    T = pd.DataFrame(rows)
    T.to_parquet("/kaggle/working/val_rerank.parquet", index=False)
    print(f"OURS scored {T.molecule_id.nunique()} lists in {time.time() - _t0:.0f}s", flush=True)
    # our model alone on the pipeline's top-25, per head
    for k in ["nce", "fp", "sae", "jepa"]:
        if k in T:
            print(f"OURS-ALONE {k:<5s} hit@1 {T.loc[T.groupby('molecule_id')[k].idxmax()].hit.mean():.3f}")

    def rerank(T, hw, w, keep_top):
        out = {}
        for mid, g in T.groupby("molecule_id", sort=False):
            g = g.sort_values("base_rank")
            s = sum(v * g[k].to_numpy() for k, v in hw.items() if k in g)
            br = g.base_rank.to_numpy()
            if keep_top:
                s, br, head = s[1:], br[1:], [g.smiles.iloc[0]]
                tail = g.smiles.iloc[1:].tolist()
            else:
                head, tail = [], g.smiles.tolist()
            orr = np.empty(len(s)); orr[np.argsort(-s)] = np.arange(1, len(s) + 1)
            f = 1 / (3 + br) + w * 1 / (3 + orr)
            out[mid] = head + [tail[i] for i in np.argsort(-f, kind="stable")]
        return out
    HWS = {"n.5f.5s1": {"nce": .5, "fp": .5, "sae": 1}, "nce": {"nce": 1}, "fp": {"fp": 1}, "sae": {"sae": 1},
           "n1f1": {"nce": 1, "fp": 1}, "all": {"nce": .5, "fp": .5, "sae": 1, "jepa": .5}}
    res = []
    for (hn, hw), w, kt in itertools.product(HWS.items(), [0, .1, .25, .5, .75, 1, 1.5, 2, 4], [True, False]):
        if w == 0 and (hn != "nce" or not kt):
            continue
        L = rerank(T, hw, w, kt)
        per = report(f"{hn} w={w} {'keep1' if kt else 'all25'}", {m: hit_rank(l, m) for m, l in L.items()})
        res.append(dict(hw=hn, w=w, keep_top=kt, **{f"c{k}": v for k, v in per.items()}))
    R = pd.DataFrame(res)
    R.to_csv("/kaggle/working/val_sweep.csv", index=False)
    base = R[(R.w == 0)].iloc[0]
    for c in [c for c in R.columns if c.startswith("c")]:
        b = R.loc[R[c].idxmax()]
        print(f"BEST {c}: {b.hw} w={b.w} keep_top={b.keep_top} -> {b[c]:.4f} (base {base[c]:.4f}, delta {b[c] - base[c]:+.4f})")
except Exception:
    traceback.print_exc()
print(f"VALIDATION HARNESS DONE in {time.time() - _t0:.0f}s")
