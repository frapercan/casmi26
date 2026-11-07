# ── casmi26: learned candidate-level ranker over the pipeline's top-25 (trained on the leak-free honest set) ──
# Features are rebuilt exactly as in hybrid/signal_ensemble_cell.py: position in the final list and in every
# pipeline variant file, our model's heads over the 25 candidates, molecule context. Any failure keeps the
# upstream submission.
CKPT_KERNEL = __CKPT__
RANKER_KERNEL = __RANKER__
import base64, glob, io, json, os, shutil, sys, tarfile, time, traceback
import numpy as np, pandas as pd
_t0 = time.time()
try:
    _sub = pd.read_csv("/kaggle/working/submission.csv", dtype=str, keep_default_na=False)
    _sub.to_csv("/kaggle/working/submission_upstream.csv", index=False)
    _rk = [p for p in glob.glob("/kaggle/input/**/ranker.txt", recursive=True) if RANKER_KERNEL in p][0]
    _meta = json.load(open(os.path.join(os.path.dirname(_rk), "ranker_meta.json")))
    import lightgbm as lgb
    _booster = lgb.Booster(model_file=_rk)
    _code = "/kaggle/working/casmi26_ours"
    tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__"))).extractall(_code)
    sys.path.insert(0, _code + "/src")
    import torch
    from casmi.metric import ik14
    from casmi.plm import PLMEmbedder
    from casmi.predict import MolFeaturizer, load_model, model_scores, query_batch
    from casmi.search import prep_query
    _w = [h for h in glob.glob("/kaggle/input/**/work/runs/run/best.pt", recursive=True) if CKPT_KERNEL in h][0]
    _W = _w[:-len("/runs/run/best.pt")]
    _any = lambda rel: [p for p in [f"{_W}/{rel}"] + sorted(glob.glob(f"/kaggle/input/**/work/{rel}", recursive=True)) if os.path.exists(p)][0]
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    _model, _sae = load_model(_w, _any("runs/sae.pt"), _dev)
    _feat = MolFeaturizer(PLMEmbedder(os.path.dirname(_any("plm/config.json")), device=_dev, mean=np.load(_any("emb/plm_mean.npy"))), _dev)
    _test = pd.read_parquet(glob.glob("/kaggle/input/**/enveda-CASMI26-molecule-id-mass-spectra/test.parquet", recursive=True)[0])
    _q = prep_query(_test)
    _q["ce_mean"] = _test.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
    _qg = dict(tuple(_q.groupby("molecule_id", sort=False)))
    _ik = {}
    def _k(s):
        if s not in _ik:
            try:
                _ik[s] = ik14(s)
            except Exception:
                _ik[s] = None
        return _ik[s]
    # positions in every pipeline variant file the ranker was trained with
    _vpos, _missing = {}, []
    for v in _meta["vcols"]:
        p = "/kaggle/working/" + v[2:].replace("variants_", "variants/", 1) + ".csv"
        if not os.path.exists(p):
            _missing.append(v); continue
        s = pd.read_csv(p, dtype=str, keep_default_na=False)
        pos = {}
        for m, x in zip(s.molecule_id, s.smiles):
            for i, sm in enumerate([y for y in x.split(";") if y.strip()][:25]):
                pos.setdefault((m, _k(sm)), i + 1)
        _vpos[v] = pos
    print("RANKER variant files missing (filled with 26):", _missing, flush=True)
    rows = []
    for mid, smis in zip(_sub.molecule_id, _sub.smiles):
        lst = [x for x in smis.split(";") if x.strip()][:25]
        g = _qg.get(mid)
        if g is None or len(lst) < 2:
            continue
        sc = model_scores(_model, _sae, query_batch(g, _dev), *_feat(lst, lst))
        for i, s in enumerate(lst):
            rows.append({"molecule_id": mid, "base_rank": i + 1, "smiles": s, "ik": _k(s),
                         **{h: float(sc[h][i]) for h in _meta["heads"] if h in sc}})
    T = pd.DataFrame(rows)
    g = T.groupby("molecule_id")
    for h in _meta["heads"]:
        T[f"{h}_z"] = (T[h] - g[h].transform("mean")) / (g[h].transform("std") + 1e-6)
        T[f"{h}_rk"] = g[h].rank(ascending=False, method="first")
        T[f"{h}_gap"] = T[h] - g[h].transform("max")
    T["inv_base"] = 1 / T.base_rank
    T["n_cand"] = g.smiles.transform("size")
    for v in _meta["vcols"]:
        T[v] = [_vpos.get(v, {}).get((m, k), 26) for m, k in zip(T.molecule_id, T.ik)]
    vc = _meta["vcols"]
    T["v_mean"] = T[vc].mean(axis=1); T["v_min"] = T[vc].min(axis=1); T["v_top1_votes"] = (T[vc] == 1).sum(axis=1)
    mq = _test.groupby("molecule_id").agg(mass=("precursor_mz", "median"), n_spec=("precursor_mz", "size"))
    T = T.join(mq, on="molecule_id")
    T["score"] = _booster.predict(T[_meta["feats"]])
    if _meta.get("keep_top"):        # leader-safe: the pipeline's rank 1 stays first
        T.loc[T.base_rank == 1, "score"] = np.inf
    new = {m: ";".join(d.sort_values("score", ascending=False, kind="stable").smiles) for m, d in T.groupby("molecule_id")}
    _out = [new.get(m, s) for m, s in zip(_sub.molecule_id, _sub.smiles)]
    _new = _sub.assign(smiles=_out)
    _n = lambda col: col.map(lambda x: sorted(y for y in x.split(";") if y.strip())[:25])
    assert (_new.molecule_id == _sub.molecule_id).all()
    assert (_n(_new.smiles) == _sub.smiles.map(lambda x: sorted([y for y in x.split(";") if y.strip()][:25]))).all()
    _moved = int((_new.smiles.str.split(";").str[0] != _sub.smiles.str.split(";").str[0]).sum())
    _new.to_csv("/kaggle/working/submission.csv", index=False)
    print(f"OURS: re-ranked by learned ranker ({_meta['featset']}, {len(_meta['feats'])} features); "
          f"top-1 changed on {_moved}/{len(_new)} molecules in {time.time()-_t0:.0f}s")
except Exception:
    traceback.print_exc()
    if os.path.exists("/kaggle/working/submission_upstream.csv"):
        shutil.copyfile("/kaggle/working/submission_upstream.csv", "/kaggle/working/submission.csv")
    print("OURS: failed, upstream submission kept")
