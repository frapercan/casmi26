# ── casmi26 hybrid layer: our JEPA+SAE spectrum model re-ranks the upstream top-25 ──
# score = 1/(K + base_rank) + OUR_W/(K + our_rank); KEEP_TOP pins rank 1. Any failure keeps the upstream submission.
OUR_W = __OUR_W__
KEEP_TOP = __KEEP_TOP__
CKPT_KERNEL = __CKPT__
_HW = __HW__  # head weights, tuned on our validation run of the upstream pipeline
K_RRF = 3.0
import base64, glob, io, os, sys, tarfile, time, traceback
import numpy as np, pandas as pd
_t0 = time.time()
try:
    if OUR_W > 0:
        _code = "/kaggle/working/casmi26_ours"
        tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__"))).extractall(_code)
        sys.path.insert(0, _code + "/src")
        import torch
        from casmi.plm import PLMEmbedder
        from casmi.predict import MolFeaturizer, load_model, model_scores, query_batch
        from casmi.search import prep_query
        _w = [h for h in glob.glob("/kaggle/input/**/work/runs/run/best.pt", recursive=True) if CKPT_KERNEL in h][0]
        _W = _w[:-len("/runs/run/best.pt")]
        _any = lambda rel: [p for p in [f"{_W}/{rel}"] + sorted(glob.glob(f"/kaggle/input/**/work/{rel}", recursive=True)) if os.path.exists(p)][0]
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _sae = load_model(_w, _any("runs/sae.pt"), _dev)
        _feat = MolFeaturizer(PLMEmbedder(os.path.dirname(_any("plm/config.json")), device=_dev, mean=np.load(_any("emb/plm_mean.npy"))), _dev)
        print("OURS ckpt", _w, "sae", _any("runs/sae.pt"), flush=True)
        _test = pd.read_parquet(glob.glob("/kaggle/input/**/enveda-CASMI26-molecule-id-mass-spectra/test.parquet", recursive=True)[0])
        _q = prep_query(_test)
        _q["ce_mean"] = _test.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
        _qg = dict(tuple(_q.groupby("molecule_id", sort=False)))
        _sub = pd.read_csv("/kaggle/working/submission.csv", dtype=str, keep_default_na=False)
        _sub.to_csv("/kaggle/working/submission_upstream.csv", index=False)
        _out, _moved = [], 0
        for mid, smis in zip(_sub.molecule_id, _sub.smiles):
            lst = [x for x in smis.split(";") if x.strip()]
            g = _qg.get(mid)
            if g is None or len(lst) < 3:
                _out.append(smis); continue
            tail = lst[1:] if KEEP_TOP else lst
            iks = list(tail)  # featurizer cache keyed by SMILES string
            plm, fp = _feat(iks, tail)
            sc = model_scores(_model, _sae, query_batch(g, _dev), plm, fp)
            s = sum(w * sc[k] for k, w in _HW.items() if k in sc)
            our_rank = np.empty(len(tail)); our_rank[np.argsort(-s)] = np.arange(1, len(tail) + 1)
            base_rank = np.arange(2, len(tail) + 2) if KEEP_TOP else np.arange(1, len(tail) + 1)
            fused = 1 / (K_RRF + base_rank) + OUR_W / (K_RRF + our_rank)
            new = [tail[i] for i in np.argsort(-fused, kind="stable")]
            _moved += new != tail
            _out.append(";".join(([lst[0]] if KEEP_TOP else []) + new))
        _new = _sub.assign(smiles=_out)
        assert _new.molecule_id.tolist() == _sub.molecule_id.tolist()
        _n = lambda col: col.map(lambda x: sorted(y for y in x.split(";") if y.strip()))
        assert (_n(_new.smiles) == _n(_sub.smiles)).all()  # same guesses, only reordered
        if KEEP_TOP:
            assert (_new.smiles.str.split(";").str[0] == _sub.smiles.str.split(";").str[0]).all()
        _new.to_csv("/kaggle/working/submission.csv", index=False)
        print(f"OURS: re-ranked {_moved}/{len(_new)} molecules (w={OUR_W}, keep_top={KEEP_TOP}, hw={_HW}) in {time.time()-_t0:.0f}s")
except Exception:
    traceback.print_exc()
    if os.path.exists("/kaggle/working/submission_upstream.csv"):
        import shutil; shutil.copyfile("/kaggle/working/submission_upstream.csv", "/kaggle/working/submission.csv")
    print("OURS: failed, upstream submission kept")
