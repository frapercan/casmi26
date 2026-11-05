# ── Re-score the honest-evaluation candidate lists with a newly trained checkpoint (CPU, minutes) ──
# For each cell source (finished harness outputs), every candidate in the pipeline's top-25 is scored with the
# new model's heads; the files needed by cell_ranker_cell.py are copied to /kaggle/working/rescored/<cell>/.
import base64, glob, io, os, shutil, sys, tarfile, time
import numpy as np, pandas as pd
SRCS = __SRCS__            # {cell: (kernel substring, keep cls or None)}
CKPT_KERNEL = "__CKPT__"
_code = "/kaggle/working/casmi26_ours"
tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__"))).extractall(_code)
sys.path.insert(0, _code + "/src")
import torch
from casmi.plm import PLMEmbedder
from casmi.predict import MolFeaturizer, load_model, model_scores, query_batch
from casmi.search import prep_query
t0 = time.time()
ck = [h for h in glob.glob("/kaggle/input/**/work/runs/run/best.pt", recursive=True) if CKPT_KERNEL in h][0]
W = ck[:-len("/runs/run/best.pt")]
anyp = lambda rel: [p for p in [f"{W}/{rel}"] + sorted(glob.glob(f"/kaggle/input/**/work/{rel}", recursive=True)) if os.path.exists(p)][0]
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, sae = load_model(ck, anyp("runs/sae.pt"), dev)
feat = MolFeaturizer(PLMEmbedder(os.path.dirname(anyp("plm/config.json")), device=dev, mean=np.load(anyp("emb/plm_mean.npy"))), dev)
print("RESCORE with", ck, "fwd head" if getattr(model, "has_fwd", False) else "", flush=True)
for cell, (key, _) in SRCS.items():
    R = os.path.dirname([p for p in glob.glob("/kaggle/input/**/val_rerank.parquet", recursive=True) if key in p][0])
    O = f"/kaggle/working/rescored/{cell}"
    os.makedirs(f"{O}/variants", exist_ok=True)
    for p in glob.glob(f"{R}/*.csv") + glob.glob(f"{R}/variants/*.csv") + [f"{R}/val_answers.parquet"]:
        shutil.copyfile(p, os.path.join(O, os.path.relpath(p, R)))
    tp = glob.glob(f"{R}/**/test.parquet", recursive=True)[0]
    os.makedirs(f"{O}/valcomp", exist_ok=True); shutil.copyfile(tp, f"{O}/valcomp/test.parquet")
    T = pd.read_parquet(f"{R}/val_rerank.parquet")
    test = pd.read_parquet(tp)
    q = prep_query(test)
    q["ce_mean"] = test.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
    qg = dict(tuple(q.groupby("molecule_id", sort=False)))
    rows = []
    for m, g in T.groupby("molecule_id", sort=False):
        g = g.sort_values("base_rank")
        lst = g.smiles.tolist()
        sc = model_scores(model, sae, query_batch(qg[m], dev), *feat(lst, lst))
        for i, (_, r) in enumerate(g.iterrows()):
            rows.append({"molecule_id": m, "base_rank": r.base_rank, "smiles": r.smiles, **{k: float(v[i]) for k, v in sc.items()}})
    N = pd.DataFrame(rows)
    N.to_parquet(f"{O}/val_rerank.parquet", index=False)
    print(f"RESCORE {cell}: {N.molecule_id.nunique()} molecules, heads {[c for c in N if c not in ('molecule_id', 'base_rank', 'smiles')]} [{time.time()-t0:.0f}s]", flush=True)
