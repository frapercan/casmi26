# ── Spectrum-blind control: score every molecule's candidates with ANOTHER molecule's spectra ──
# If a ranker using these blind scores still gains, the gain comes from what the model thinks of the
# candidates alone (a prior that may only reflect how this query set was assembled), not from the spectrum.
import base64, glob, io, os, sys, tarfile, time
import numpy as np, pandas as pd
SRC = "__SRC__"
_code = "/kaggle/working/casmi26_ours"
tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__"))).extractall(_code)
sys.path.insert(0, _code + "/src")
import torch
from casmi.plm import PLMEmbedder
from casmi.predict import MolFeaturizer, load_model, model_scores, query_batch
from casmi.search import prep_query
t0 = time.time()
R = os.path.dirname([p for p in glob.glob("/kaggle/input/**/val_rerank.parquet", recursive=True) if SRC in p][0])
T = pd.read_parquet(f"{R}/val_rerank.parquet")
test = pd.read_parquet(glob.glob(f"{R}/**/test.parquet", recursive=True)[0])
_ck = sorted(glob.glob("/kaggle/input/**/work/runs/run/best.pt", recursive=True))
_ck = [h for h in _ck if "casmi26-train-v3" in h] or _ck
W = _ck[0][:-len("/runs/run/best.pt")]
plm = [os.path.dirname(p) for p in sorted(glob.glob("/kaggle/input/**/work/plm/config.json", recursive=True))][0]
mean = [p for p in [f"{W}/emb/plm_mean.npy"] + sorted(glob.glob("/kaggle/input/**/work/emb/plm_mean.npy", recursive=True)) if os.path.exists(p)][0]
sae = [p for p in [f"{W}/runs/sae.pt"] + sorted(glob.glob("/kaggle/input/**/work/runs/sae.pt", recursive=True)) if os.path.exists(p)][0]
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, sae_m = load_model(f"{W}/runs/run/best.pt", sae, dev)
feat = MolFeaturizer(PLMEmbedder(plm, device=dev, mean=np.load(mean)), dev)
q = prep_query(test)
q["ce_mean"] = test.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
qg = dict(tuple(q.groupby("molecule_id", sort=False)))
mids = np.array(sorted(T.molecule_id.unique()))
rng = np.random.default_rng(0)
perm = rng.permutation(len(mids))
while (perm == np.arange(len(mids))).any():                    # derangement: nobody keeps its own spectra
    perm = rng.permutation(len(mids))
donor = dict(zip(mids, mids[perm]))
rows = []
for m, g in T.groupby("molecule_id", sort=False):
    g = g.sort_values("base_rank")
    lst = g.smiles.tolist()
    sc = model_scores(model, sae_m, query_batch(qg[donor[m]], dev), *feat(lst, lst))
    for i, (_, r) in enumerate(g.iterrows()):
        rows.append({"molecule_id": m, "base_rank": r.base_rank, "smiles": r.smiles, **{k: float(v[i]) for k, v in sc.items()}})
B = pd.DataFrame(rows)
B.to_parquet("/kaggle/working/val_rerank_blind.parquet", index=False)
print(f"BLIND scores for {B.molecule_id.nunique()} molecules in {time.time() - t0:.0f}s on {dev}")
