"""J1b: the internal-evaluation setting. Truth (train format) among up to 200 same-mass PubChem candidates (raw PubChem
format, mostly Kekulé) -> truth MRR per head; then every string canonicalised; then blind (another molecule's spectra).

  python scripts/audit_j1b.py --ckpt /kaggle/input/casmi26-train-wm7c3/work/runs/run/best.pt
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scripts.audit_j1 import Feat, canon, EV, V2  # noqa: E402
from casmi.metric import ik14  # noqa: E402
from casmi.plm import PLMEmbedder  # noqa: E402
from casmi.predict import load_model, model_scores, query_batch  # noqa: E402
from casmi.search import prep_query  # noqa: E402

PC = "/kaggle/input/casmi26-pubchem-popularity-prior/casmi26-pubchem-tier/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/kaggle/input/casmi26-train-wm7c3/work/runs/run/best.pt")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--out", default="work/audit/j1b.json")
    a = ap.parse_args()
    from rdkit import Chem
    from rdkit.Chem.Descriptors import ExactMolWt
    dev = "cuda"
    model, sae = load_model(a.ckpt, f"{V2}/runs/sae.pt", dev)
    feat = Feat(PLMEmbedder(f"{V2}/plm", device=dev, mean=np.load(f"{V2}/emb/plm_mean.npy")), dev)
    raw = pd.read_parquet(f"{EV}/effective_query_spectra.parquet"); q = prep_query(raw)
    q["ce_mean"] = raw.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan).to_numpy()
    QG = dict(tuple(q.groupby("molecule_id", sort=False)))
    ans = pd.read_parquet(f"{EV}/val_answers.parquet")
    pm = np.load(PC + "pc_mass.npy", mmap_mode="r"); po = np.load(PC + "pc_off.npy", mmap_mode="r"); ps = np.load(PC + "pc_smiles.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    mids = [m for m in ans.molecule_id if m in QG]; donor = dict(zip(mids, rng.permutation(mids)))
    rows = []
    for mid, ts in zip(ans.molecule_id, ans.smiles):
        if mid not in QG:
            continue
        mass = ExactMolWt(Chem.MolFromSmiles(ts)); tol = mass * 10e-6; t = ik14(ts)
        lo, hi = np.searchsorted(pm, mass - tol), np.searchsorted(pm, mass + tol, side="right")
        idx = np.arange(lo, hi)
        if len(idx) > a.k * 2:
            idx = np.sort(rng.choice(idx, a.k * 2, replace=False))
        cands = []
        for i in idx:
            s = bytes(ps[po[i]:po[i + 1]]).decode()
            if Chem.MolFromSmiles(s) is not None and ik14(s) != t:
                cands.append(s)
            if len(cands) >= a.k:
                break
        if len(cands) < 10:
            continue
        conds = {"raw": [ts] + cands, "all_canon": [canon(ts)] + [canon(x) for x in cands], "truth_kekule": [Chem.MolToSmiles(Chem.MolFromSmiles(ts), kekuleSmiles=True)] + cands}
        r = {"molecule_id": mid, "n": len(cands) + 1}
        for spec, sm in (("real", mid), ("blind", donor[mid])):
            b = query_batch(QG[sm], dev)
            for cn, cs in conds.items():
                sc = model_scores(model, sae, b, *feat(cs))
                for h, v in sc.items():
                    r[f"{spec}:{cn}:{h}"] = 1.0 / (1 + int((v[1:] > v[0]).sum()))
        rows.append(r)
    R = pd.DataFrame(rows); heads = sorted({c.split(":")[2] for c in R.columns if c.count(":") == 2})
    rand = float(np.mean([np.mean(1.0 / np.arange(1, n + 1)) for n in R.n]))
    print(f"{len(R)} molecules, median {int(R.n.median())} candidates, random MRR {rand:.3f}\n" + " " * 19 + "  ".join(f"{h:>6s}" for h in heads))
    out = {"n": len(R), "random": rand, "mrr": {}}
    for spec in ("real", "blind"):
        for cn in ("raw", "truth_kekule", "all_canon"):
            v = [R[f"{spec}:{cn}:{h}"].mean() for h in heads]; out["mrr"][f"{spec}:{cn}"] = dict(zip(heads, map(float, v)))
            print(f"{spec:5s} {cn:12s} " + "  ".join(f"{x:6.3f}" for x in v))
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
