# ── CASMI26 class 3: score generated candidates (structures in no database) with a class-3-safe model ──
# The model (CKPT kernel) was trained with every spectrum of the class-3 molecules hidden (exclude_all_ik.txt),
# and the generator built candidates from a library without them, so nothing here has seen the answers.
# Arms are within-molecule rankings of the generated set; the naive arm (analog spectral similarity x Tanimoto
# to the analog) is the baseline to beat. MRR@25 over ALL molecules (recall caps it) and over generated ones.
import base64, glob, io, os, sys, tarfile, time
import numpy as np, pandas as pd
CKPT_KERNEL = "__CKPT__"
t0 = time.time()
_code = "/kaggle/working/casmi26_ours"
tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__"))).extractall(_code)
_q3 = glob.glob("/kaggle/input/**/casmi26-class3-data/**/queries.parquet", recursive=True)
if _q3:
    D = os.path.dirname(_q3[0])
else:                                 # archive not extracted by Kaggle
    D = "/kaggle/tmp_c3"
    tarfile.open(glob.glob("/kaggle/input/**/c3data.tgz", recursive=True)[0]).extractall(D)
print("class-3 data from", D, sorted(os.listdir(D)), flush=True)
sys.path.insert(0, _code + "/src")
import torch
from casmi.plm import PLMEmbedder
from casmi.predict import MolFeaturizer, load_model, model_scores, query_batch
from casmi.search import prep_query
ck = [h for h in glob.glob("/kaggle/input/**/work/runs/run/best.pt", recursive=True) if CKPT_KERNEL in h][0]
anyp = lambda rel: sorted(glob.glob(f"/kaggle/input/**/work/{rel}", recursive=True))[0]
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, sae = load_model(ck, anyp("runs/sae.pt"), dev)
feat = MolFeaturizer(PLMEmbedder(os.path.dirname(anyp("plm/config.json")), device=dev, mean=np.load(anyp("emb/plm_mean.npy"))), dev)
print("CLASS3 scoring with", ck, "on", dev, flush=True)
C = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{D}/candidates_*.parquet"))], ignore_index=True)
St = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{D}/stats_*.parquet"))], ignore_index=True)
Q = pd.read_parquet(f"{D}/queries.parquet")
hide = set(open(f"{D}/exclude_all_ik.txt").read().split())
q = prep_query(Q)
q["ce_mean"] = Q.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
qg = dict(tuple(q.groupby("molecule_id", sort=False)))
print(f"{St.molecule_id.nunique()} molecules, {len(C)} candidates, recall {St.generated.mean():.3f}", flush=True)
out = []
with torch.no_grad():
    for k, (m, g) in enumerate(C.groupby("molecule_id", sort=False)):
        g = g.copy()
        sc = {}
        for i in range(0, len(g), 8192):
            part = model_scores(model, sae, query_batch(qg[m], dev), *feat(g.ik.tolist()[i:i+8192], g.smiles.tolist()[i:i+8192]))
            for h, v in part.items():
                sc.setdefault(h, []).append(v)
        # heads were z-scored per chunk by model_scores; re-z over the whole set below
        for h, v in sc.items():
            g[h] = np.concatenate(v)
        out.append(g)
        if (k + 1) % 50 == 0:
            print(f"  {k+1} molecules [{time.time()-t0:.0f}s]", flush=True)
S = pd.concat(out, ignore_index=True)
heads = [h for h in ["nce", "fwd", "fp", "jepa", "sae"] if h in S]
gr = S.groupby("molecule_id")
for h in heads + ["naive", "parent_sim", "tanimoto"]:
    S[f"z_{h}"] = (S[h] - gr[h].transform("mean")) / (gr[h].transform("std").fillna(0) + 1e-6)
S["z_edits"] = -S.n_edits.astype(float)
ARMS = {"naive": ["naive"], "parent_sim": ["parent_sim"], **{h: [h] for h in heads},
        "fwd+fp": ["fwd", "fp"], "fwd+nce": ["fwd", "nce"], "fwd+fp+nce": ["fwd", "fp", "nce"],
        "fwd+naive": ["fwd", "naive"], "fwd+fp+naive": ["fwd", "fp", "naive"], "fwd+fp+nce+naive": ["fwd", "fp", "nce", "naive"],
        "fwd+fp+naive+edits": ["fwd", "fp", "naive", "edits"]}
mids = St.molecule_id.tolist(); gen = set(St.molecule_id[St.generated])
rng = np.random.default_rng(0)
def rr_of(cols):
    S["_s"] = sum(S[f"z_{c}"] for c in cols)
    r = {}
    for m, d in S.groupby("molecule_id"):
        h = np.flatnonzero(d.sort_values("_s", ascending=False, kind="stable").hit.to_numpy())
        r[m] = 1 / (h[0] + 1) if len(h) and h[0] < 25 else 0.0
    return pd.Series(r).reindex(mids).fillna(0.0)
base = rr_of(ARMS["naive"])
print(f"\n{'arm':<22s} MRR@25 all  top1   MRR|gen   delta vs naive [95% CI]")
res = {}
for name, cols in ARMS.items():
    r = rr_of(cols); d = (r - base).to_numpy()
    lo, hi = np.percentile(d[rng.integers(0, len(d), (5000, len(d)))].mean(1), [2.5, 97.5])
    res[name] = dict(mrr=float(r.mean()), top1=float((r == 1).mean()), mrr_gen=float(r[list(gen)].mean()), delta=float(d.mean()), ci=[float(lo), float(hi)])
    print(f"{name:<22s} {r.mean():.4f}     {(r == 1).mean():.3f}  {r[list(gen)].mean():.4f}   {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)
print(f"ceiling (truth generated): {St.generated.mean():.3f} | median candidates {St.n_cand.median():.0f}")
S.drop(columns=["_s"]).to_parquet("/kaggle/working/class3_scored.parquet", index=False)
import json; json.dump(res, open("/kaggle/working/class3_arms.json", "w"), indent=1)
print(f"CLASS3 DONE [{time.time()-t0:.0f}s]")
