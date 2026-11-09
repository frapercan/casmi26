"""J1 audit: does our model recognise the *format* of the true SMILES instead of the spectrum?

Training positives are RDKit-canonical, stereo-free strings (train.parquet normalized_smiles); database
candidates come in their own formats (stereo, bracket atoms, non-canonical order). The PLM embeds the raw
string, so a model can learn "train-style string = positive". This script scores the pipeline's top-25 lists of
the class-3 validation cell (truth inside the list) under controlled conditions and reports the truth's MRR:

  deployed   : candidates as the pipeline wrote them (truth in database format)
  truth_train: only the truth's string swapped to the train format
  all_canon  : every candidate canonicalised (stereo-free), i.e. format removed
  blind      : the same three, but each molecule scored with another molecule's spectra

  python scripts/audit_j1.py --ckpt /kaggle/input/casmi26-train-wm7c3/work/runs/run/best.pt
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.metric import ik14  # noqa: E402
from casmi.plm import PLMEmbedder  # noqa: E402
from casmi.predict import load_model, model_scores, query_batch  # noqa: E402
from casmi.search import prep_query  # noqa: E402

EV = "/kaggle/input/casmi26-eval-c3"
V2 = "/kaggle/input/casmi26-train-v2/work"


def canon(s):
    from rdkit import Chem
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else s


class Feat:
    """PLM + Morgan features keyed by the exact string (format matters here)."""

    def __init__(self, emb, dev):
        from rdkit.Chem import rdFingerprintGenerator
        self.emb, self.dev, self.c = emb, dev, {}
        self.gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def __call__(self, smiles):
        from rdkit import Chem
        todo = [s for s in dict.fromkeys(smiles) if s not in self.c]
        if todo:
            for s, v in zip(todo, self.emb.embed(todo)):
                m = Chem.MolFromSmiles(s)
                fp = self.gen.GetFingerprintAsNumPy(m).astype(np.float32) if m is not None else np.zeros(2048, np.float32)
                self.c[s] = (v.astype(np.float32), fp)
        t = lambda i: torch.as_tensor(np.stack([self.c[s][i] for s in smiles]), device=self.dev)
        return t(0), t(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/kaggle/input/casmi26-train-wm7c3/work/runs/run/best.pt")
    ap.add_argument("--sub", default=f"{EV}/submission.csv")
    ap.add_argument("--out", default="work/audit/j1.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, sae = load_model(a.ckpt, f"{V2}/runs/sae.pt", dev)
    feat = Feat(PLMEmbedder(f"{V2}/plm", device=dev, mean=np.load(f"{V2}/emb/plm_mean.npy")), dev)
    raw = pd.read_parquet(f"{EV}/effective_query_spectra.parquet")
    q = prep_query(raw)
    q["ce_mean"] = raw.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan).to_numpy()
    QG = dict(tuple(q.groupby("molecule_id", sort=False)))
    ans = pd.read_parquet(f"{EV}/val_answers.parquet")
    truth_s = dict(zip(ans.molecule_id, ans.smiles))
    sub = pd.read_csv(a.sub, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(a.seed)
    mids = [m for m in sub.molecule_id if m in QG and m in truth_s]
    donor = dict(zip(mids, rng.permutation(mids)))
    rows = []
    for mid, lst in zip(sub.molecule_id, sub.smiles):
        if mid not in mids:
            continue
        P = [x for x in lst.split(";") if x.strip()][:25]
        t = ik14(truth_s[mid]); keys = [ik14(x) for x in P]
        if t not in keys:
            continue
        j = keys.index(t)
        conds = {"deployed": P, "truth_train": P[:j] + [truth_s[mid]] + P[j + 1:], "all_canon": [canon(x) for x in P]}
        r = {"molecule_id": mid, "n": len(P), "truth_fmt_differs": P[j] != truth_s[mid]}
        for spec_name, sm in (("real", mid), ("blind", donor[mid])):
            b = query_batch(QG[sm], dev)
            for cname, cands in conds.items():
                plm, fp = feat(cands)
                sc = model_scores(model, sae, b, plm, fp)
                for h, v in sc.items():
                    r[f"{spec_name}:{cname}:{h}"] = 1.0 / (1 + int((v > v[j]).sum()))
        rows.append(r)
    R = pd.DataFrame(rows)
    heads = sorted({c.split(":")[2] for c in R.columns if c.count(":") == 2})
    rand = float(np.mean([np.mean(1.0 / np.arange(1, n + 1)) for n in R.n]))
    out = {"n_molecules": len(R), "truth_format_differs": int(R.truth_fmt_differs.sum()), "random_mrr": rand, "mrr": {}}
    print(f"{len(R)} molecules with the truth in the pipeline top-25; truth string differs from train format in "
          f"{int(R.truth_fmt_differs.sum())}; random MRR {rand:.3f}")
    for sub_name, S in (("all", R), ("fmt_differs", R[R.truth_fmt_differs])):
        print(f"\n[{sub_name}] n={len(S)}   " + "  ".join(f"{h:>6s}" for h in heads))
        for spec_name in ("real", "blind"):
            for cname in ("deployed", "truth_train", "all_canon"):
                vals = [S[f"{spec_name}:{cname}:{h}"].mean() for h in heads]
                out["mrr"][f"{sub_name}:{spec_name}:{cname}"] = dict(zip(heads, map(float, vals)))
                print(f"{spec_name:5s} {cname:12s} " + "  ".join(f"{v:6.3f}" for v in vals))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    R.to_parquet(a.out.replace(".json", ".parquet"), index=False)


if __name__ == "__main__":
    main()
