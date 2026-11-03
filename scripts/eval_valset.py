"""Evaluate fusion configs on the hidden-test-like validation set and calibrate it to the LB.

1. Builds every query molecule's candidate table once (library candidates from the library
   minus the validation spectra, DB candidates, class-2 answers injected), with model scores
   when --ckpt is given. Saved to <out>/cands.parquet.
2. Scores each weights config per class (exact official InChIKey14 on the top-25 SMILES).
3. With --lb name=score anchors (public LB of submitted configs), fits the class mix
   p1, p2 (p3 = 1 - p1 - p2, class 3 always scores 0) so that p1*m1 + p2*m2 ~ LB, then
   predicts the LB of every config. Residuals tell how much the validation can be trusted.

  python scripts/eval_valset.py --val /home/user/work/val \\
      --config lib=configs/weights_libonly.json --lb lib=0.152            # CPU, library only
  python scripts/eval_valset.py --ckpt best.pt --sae sae.pt --plm plm/ \\
      --config fused=configs/weights_v2.json --config lib=... --config model=... \\
      --lb fused=0.179 --lb lib=0.152 --lb model=0.122
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.metric import ik14  # noqa: E402
from casmi.predict import DEFAULT_W, CandidateDB, candidate_table, fuse  # noqa: E402
from casmi.search import Library, prep_query  # noqa: E402
from casmi.valset import excluded_spectra, load_exclude  # noqa: E402


def official_rr(smis: list[str], true_smiles: str) -> float:
    t = ik14(true_smiles)
    for i, s in enumerate(smis[:25]):
        if ik14(s) == t:
            return 1 / (i + 1)
    return 0.0


def fit_mix(m1: dict, m2: dict, lb: dict, step=0.005):
    best = None
    for p1 in np.arange(0, 1 + 1e-9, step):
        for p2 in np.arange(0, 1 - p1 + 1e-9, step):
            err = sum((p1 * m1[k] + p2 * m2[k] - v) ** 2 for k, v in lb.items())
            if best is None or err < best[0]:
                best = (err, p1, p2)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/home/user/work")
    ap.add_argument("--val", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--sae", default="")
    ap.add_argument("--plm", default="")
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--config", action="append", default=[], help="name=weights.json (repeatable)")
    ap.add_argument("--lb", action="append", default=[], help="name=public LB score (repeatable)")
    ap.add_argument("--cands", default="", help="reuse a saved cands.parquet instead of rebuilding")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    val = a.val or f"{a.work}/val"
    out = a.out or f"{val}/eval"
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = pd.read_parquet(f"{val}/answers.parquet").set_index("molecule_id")

    if a.cands:
        allc = pd.read_parquet(a.cands)
    else:
        full = Library.load(f"{a.work}/lib")
        ref = full.subset(~excluded_spectra(full.spectra, load_exclude(val)))
        ref.build_index()
        db = CandidateDB(f"{a.work}/candidates.parquet")
        model = sae = feat = None
        if a.ckpt:
            from casmi.plm import DEFAULT_PLM, PLMEmbedder
            from casmi.predict import MolFeaturizer, load_model
            model, sae = load_model(a.ckpt, a.sae, dev)
            mp = f"{a.work}/emb/plm_mean.npy"
            feat = MolFeaturizer(PLMEmbedder(a.plm or DEFAULT_PLM, device=dev,
                                             mean=np.load(mp) if os.path.exists(mp) else None), dev)
            excl = load_exclude(val)
            ck_val = torch.load(a.ckpt, map_location="cpu")["args"].get("val", "")
            if not ck_val:
                print("WARNING: checkpoint was trained without --val: model scores on this set are leaky "
                      "(it saw these spectra). Library-only numbers are still valid.", flush=True)
            del excl
        test = pd.read_parquet(f"{val}/queries.parquet")
        q = prep_query(test)
        q["ce_mean"] = test.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
        tabs, t0 = [], time.time()
        for n, (mid, g) in enumerate(q.groupby("molecule_id", sort=False)):
            r = A.loc[mid]
            t = candidate_table(g, ref, db, feat, model, sae, dev, a.ppm,
                                extra=(r.inchikey14, r.smiles) if r.cls == 2 else None)
            t["is_true"] = t.ik == r.inchikey14
            t["qid"], t["cls"] = mid, int(r.cls)
            tabs.append(t)
            if (n + 1) % 100 == 0:
                print(f"{n+1}/{len(A)} molecules {time.time()-t0:.0f}s", flush=True)
        allc = pd.concat(tabs, ignore_index=True)
        allc.to_parquet(f"{out}/cands.parquet", index=False)

    confs = {"default": DEFAULT_W}
    for c in a.config:
        k, p = c.split("=", 1)
        confs[k] = {**DEFAULT_W, **json.load(open(p))}
    res = {}
    groups = dict(tuple(allc.groupby("qid", sort=False)))
    for name, w in confs.items():
        rr = {1: [], 2: []}
        for mid, r in A.iterrows():
            t = groups.get(mid)
            smis = []
            if t is not None and len(t):
                t = t.assign(s=fuse(t, w)).sort_values("s", ascending=False).drop_duplicates("ik")
                smis = t.smiles.head(25).tolist()
            rr[int(r.cls)].append(official_rr(smis, r.smiles))
        res[name] = {"m1": float(np.mean(rr[1])), "m2": float(np.mean(rr[2]))}
        print(f"{name:12s} class1 {res[name]['m1']:.4f}  class2 {res[name]['m2']:.4f}", flush=True)
    recall = allc.groupby("qid").is_true.any()
    print("answer in candidate pool: class1 %.3f  class2 %.3f" % tuple(
        recall[A.index[A.cls == c]].mean() for c in (1, 2)))

    lb = dict((k, float(v)) for k, v in (x.split("=") for x in a.lb))
    lb = {k: v for k, v in lb.items() if k in res}
    if lb:
        p1, p2 = fit_mix({k: res[k]["m1"] for k in res}, {k: res[k]["m2"] for k in res}, lb)
        print(f"\nfitted class mix: p1={p1:.3f} p2={p2:.3f} p3={1-p1-p2:.3f}  (anchors: {lb})")
        for k, r in res.items():
            pred = p1 * r["m1"] + p2 * r["m2"]
            res[k]["pred_lb"] = pred
            print(f"  {k:12s} predicted LB {pred:.3f}" + (f"   actual {lb[k]:.3f}  resid {pred-lb[k]:+.3f}" if k in lb else ""))
        res["_mix"] = {"p1": p1, "p2": p2, "p3": 1 - p1 - p2}
    json.dump(res, open(f"{out}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
