# ── casmi26 class-3 layer: generated structures (in no database) inserted from rank INS_POS of the upstream lists ──
# For every test molecule: spectral analogs in the training library -> NP tailoring edits -> exact-mass filter
# (casmi.gen3), then the generated structures not already in the list are ranked by our model (z(fwd) + z(fp),
# scored together with the list) and the best INS_K are inserted at rank INS_POS. Ranks 1..INS_POS-1 never move.
# Honest evaluation (class3_insert_cell.py, insert_harm_cell.py): -0.001 worst case where the truth is in a
# database, +0.026 MRR on true class-3 molecules. Any failure keeps the upstream submission.
INS_POS = __INS_POS__
INS_K = __INS_K__
ANALOGS = 30
CKPT_KERNEL = "__CKPT__"
DRY = __DRY__                 # True: validation run on the class-3 cell (library without its structures, answers known)
TIME_BUDGET = __BUDGET__      # seconds for the whole layer; molecules not reached keep their upstream list
import base64, glob, io, os, sys, tarfile, time, traceback
import numpy as np, pandas as pd
_t0 = time.time()
try:   # never push the session past Kaggle's 9 h limit: the kernel process is as old as the session
    _age = float(open("/proc/uptime").read().split()[0]) - int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[19]) / os.sysconf("SC_CLK_TCK")
except Exception:
    _age = 0.0
TIME_BUDGET = min(TIME_BUDGET, 8.4 * 3600 - _age)
print(f"C3 session age {_age / 3600:.2f} h -> layer budget {TIME_BUDGET / 60:.0f} min", flush=True)
_W = "/kaggle/working"
try:
    assert TIME_BUDGET > 900, "less than 15 min left in the session"
    _code = f"{_W}/casmi26_c3"
    tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__"))).extractall(_code)
    sys.path.insert(0, _code + "/src")
    from casmi import gen3 as G
    from casmi.metric import ik14 as _ik14
    from casmi.search import Library, prep_query
    if DRY:
        _ev = sorted([p for p in glob.glob("/kaggle/input/**/submission.csv", recursive=True) if "casmi26-eval-c3" in p], key=len)[0]
        _evd = os.path.dirname(_ev)
        _sub = pd.read_csv(_ev, dtype=str, keep_default_na=False)
        _test = pd.read_parquet(f"{_W}/valcomp/test.parquet")
        _hide = set(open(glob.glob("/kaggle/**/exclude_all_ik.txt", recursive=True)[0]).read().split())
        _vap = glob.glob(f"{_evd}/**/val_answers.parquet", recursive=True)
        _VA = pd.read_parquet(_vap[0] if _vap else f"{_evd}/val_answers.parquet")
    else:
        _sub = pd.read_csv(f"{_W}/submission.csv", dtype=str, keep_default_na=False)
        _test = pd.read_parquet(glob.glob("/kaggle/input/**/enveda-CASMI26-molecule-id-mass-spectra/test.parquet", recursive=True)[0])
        _hide = set()
    _sub.to_csv(f"{_W}/submission_before_c3.csv", index=False)
    _lp = os.path.dirname(glob.glob("/kaggle/input/**/casmi26-speclib/**/peaks.npz", recursive=True)[0]) \
        if glob.glob("/kaggle/input/**/casmi26-speclib/**/peaks.npz", recursive=True) else \
        os.path.dirname(glob.glob("/kaggle/input/**/peaks.npz", recursive=True)[0])
    LIB = Library.load(_lp)
    if _hide:
        LIB = LIB.subset(~LIB.spectra.inchikey14.isin(_hide).to_numpy())
    G.set_edits("v0", 20)
    G.edit_deltas(LIB.molecules.smiles.tolist())
    F, N = G.library_matrix(LIB)
    SPEC_IK = LIB.spectra.inchikey14.to_numpy(); SPEC_MODE = LIB.spectra.ionization_mode.to_numpy()
    SMI_OF = dict(zip(LIB.molecules.inchikey14, LIB.molecules.smiles))
    _q = prep_query(_test)
    _q["ce_mean"] = _test.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
    QG = dict(tuple(_q.groupby("molecule_id", sort=False)))
    print(f"C3 library {len(LIB.spectra)} spectra, {len(SMI_OF)} molecules | deltas {len(G.DELTA)}/{len(G.RXN)} | "
          f"{len(QG)} query molecules [{time.time() - _t0:.0f}s]", flush=True)

    def _gen(mid):
        try:
            c, _ = G.generate(QG[mid], F, N, SPEC_IK, SPEC_MODE, SMI_OF, ANALOGS)
            return mid, [v[0] for v in c.values()]
        except Exception:
            return mid, []
    import multiprocessing as _mp
    GEN = {}
    _mids = [m for m in _sub.molecule_id if m in QG]
    _gen_budget = 0.6 * TIME_BUDGET
    STRAGGLER = 240               # seconds to wait for the next result before giving up on the slow tail
    with _mp.get_context("fork").Pool(max(1, os.cpu_count())) as _pool:
        _it = _pool.imap_unordered(_gen, _mids, chunksize=1)
        for k in range(len(_mids)):
            try:
                mid, smis = _it.next(timeout=max(1.0, min(STRAGGLER, _gen_budget - (time.time() - _t0))))
            except _mp.TimeoutError:
                print(f"  C3 generation stopped at {k}/{len(_mids)} (slow tail or budget) [{time.time() - _t0:.0f}s]", flush=True)
                _pool.terminate(); break
            GEN[mid] = smis
            if (k + 1) % 50 == 0:
                print(f"  C3 generated {k + 1}/{len(_mids)} [{time.time() - _t0:.0f}s]", flush=True)
    pd.to_pickle(GEN, f"{_W}/c3_generated.pkl")
    print(f"C3 generated for {len(GEN)} molecules, median {np.median([len(v) for v in GEN.values()]) if GEN else 0:.0f} "
          f"candidates [{time.time() - _t0:.0f}s]", flush=True)

    import torch
    from casmi.plm import PLMEmbedder
    from casmi.predict import MolFeaturizer, load_model, model_scores, query_batch
    _ck = [h for h in glob.glob("/kaggle/input/**/work/runs/run/best.pt", recursive=True) if CKPT_KERNEL in h][0]
    _any = lambda rel: sorted(glob.glob(f"/kaggle/input/**/work/{rel}", recursive=True))[0]
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    _model, _sae = load_model(_ck, _any("runs/sae.pt"), _dev)
    _feat = MolFeaturizer(PLMEmbedder(os.path.dirname(_any("plm/config.json")), device=_dev, mean=np.load(_any("emb/plm_mean.npy"))), _dev)
    _kc = {}
    def _K(s):
        if s not in _kc:
            try: _kc[s] = _ik14(s)
            except Exception: _kc[s] = None
        return _kc[s]
    _T = {m: _ik14(x) for m, x in zip(_VA.molecule_id, _VA.smiles)} if DRY else {}
    _out, _n_ins, _C3 = [], 0, {}
    with torch.no_grad():
        for mid, smis in zip(_sub.molecule_id, _sub.smiles):
            lst = [x for x in smis.split(";") if x.strip()]
            gen = GEN.get(mid) or []
            if not gen or mid not in QG or time.time() - _t0 > TIME_BUDGET:
                _out.append(smis); continue
            pk = [_K(s) for s in lst[:25]]
            gs, gk, seen = [], [], set(k for k in pk if k) - {_T.get(mid)}   # DRY: keep a generated truth for the class-3 view
            for s in gen:
                k = _K(s)
                if k and k not in seen:
                    seen.add(k); gs.append(s); gk.append(k)
            if not gs:
                _out.append(smis); continue
            allsmi = lst[:25] + gs; allik = [k or s for k, s in zip(pk, lst[:25])] + gk
            sc = {}
            for i in range(0, len(allsmi), 8192):
                part = model_scores(_model, _sae, query_batch(QG[mid], _dev), *_feat(allik[i:i + 8192], allsmi[i:i + 8192]))
                for h, v in part.items():
                    sc.setdefault(h, []).append(np.asarray(v, np.float64))
            z = 0.0
            for h in ("fwd", "fp"):
                v = np.concatenate(sc[h]); z = z + (v - v.mean()) / (v.std() + 1e-6)
            order = [j for j in np.argsort(-z[len(lst[:25]):], kind="stable")]
            best = [gs[j] for j in order if gk[j] not in set(pk)][:INS_K]
            new = lst[:INS_POS - 1] + best + lst[INS_POS - 1:]
            if DRY:   # true class 3: the truth is in no database, so it is not in the upstream list
                t = _T[mid]; l3 = [x for x, k in zip(lst, pk + [None] * len(lst)) if k != t]
                _C3[mid] = l3[:INS_POS - 1] + [gs[j] for j in order][:INS_K] + l3[INS_POS - 1:]
            _out.append(";".join(new[:max(len(lst), 25)])); _n_ins += 1
    _new = pd.DataFrame({"molecule_id": _sub.molecule_id, "smiles": _out})
    assert len(_new) == len(_sub) and (_new.molecule_id == _sub.molecule_id).all()
    _nb = lambda x: [y for y in x.split(";") if y.strip()]
    assert all(_nb(b)[:len(_nb(a)[:INS_POS - 1])] == _nb(a)[:INS_POS - 1] for a, b in zip(_sub.smiles, _new.smiles))
    _new.to_csv(f"{_W}/submission.csv" if not DRY else f"{_W}/submission_c3.csv", index=False)
    print(f"C3 LAYER: inserted generated structures in {_n_ins}/{len(_sub)} lists at rank {INS_POS} "
          f"[{time.time() - _t0:.0f}s]", flush=True)
    if DRY:   # answers known: before/after, and the true-class-3 view (truth removed from the upstream list)
        def _rr(lst, t):
            for i, x in enumerate([x for x in lst if x.strip()][:25]):
                if _K(x) == t: return 1.0 / (i + 1)
            return 0.0
        b = np.array([_rr(x.split(";"), _T[m]) for m, x in zip(_sub.molecule_id, _sub.smiles)])
        a = np.array([_rr(x.split(";"), _T[m]) for m, x in zip(_new.molecule_id, _new.smiles)])
        c = np.array([_rr(_C3[m], _T[m]) if m in _C3 else 0.0 for m in _sub.molecule_id])
        print(f"C3 DRY: MRR upstream {b.mean():.4f} -> layer {a.mean():.4f} (delta {a.mean() - b.mean():+.4f}) | "
              f"true-class-3 view {c.mean():.4f} | generated truth in {sum(_T[m] in {_K(x) for x in GEN.get(m, [])} for m in _T)} "
              f"of {len(_T)} molecules", flush=True)
except Exception:
    traceback.print_exc()
    print("C3 LAYER FAILED -> upstream submission kept", flush=True)
    if os.path.exists(f"{_W}/submission_before_c3.csv") and not DRY:
        pd.read_csv(f"{_W}/submission_before_c3.csv", dtype=str, keep_default_na=False).to_csv(f"{_W}/submission.csv", index=False)
