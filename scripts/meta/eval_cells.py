"""Score validation cells (scripts/meta/build_cells.py) with a checkpoint and the PubChem documentation prior.

For every cell molecule all candidates of its 10 ppm PubChem window are ranked ("mass" setting) and, separately,
only those with the truth's formula ("formula" setting: an oracle formula, isolates structure ranking).
Variants: each model head (z-scored within the molecule's list), the head ensemble, the documentation prior
doc = ln(1+substances) + ln(1+PubMed) alone, and z(model) + w * doc for a grid of w. MRR@25, ties broken by a fixed
random order. Strata: mass bin, documentation tertile of the truth, list length.

  python scripts/meta/eval_cells.py --ckpt work/runs/wmcanon_s0/best.pt --tag wmcanon_s0
  python scripts/meta/eval_cells.py --tag prior                       # prior only, no model
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
W_GRID = [0.05, 0.1, 0.15, 0.25, 0.5, 1.0]


def rr_at25(score, truth, jitter):
    s = score + jitter
    t = s[truth].max()
    r = int((s > t).sum()) + 1
    return 1.0 / r if r <= 25 else 0.0


@torch.no_grad()
def head_scores(model, b, plm, fp, chunk=8192):
    from casmi.predict import cond_vec, fwd_emb
    h = model.spec(b)
    se = model.spec_emb(h)
    pr = F.normalize(model.pred(h), dim=-1)
    p = torch.sigmoid(model.fp(h)).mean(0).clamp(1e-4, 1 - 1e-4)
    cv = cond_vec(model, b) if model.has_fwd else None
    out = {k: [] for k in ("nce", "jepa", "fp") + (("fwd",) if model.has_fwd else ())}
    for i in range(0, len(plm), chunk):
        P = plm[i:i + chunk].float(); Fp = fp[i:i + chunk].float()
        out["nce"].append((se @ model.mol_emb(P, Fp).T).mean(0))
        out["jepa"].append((pr @ P.T).mean(0))
        out["fp"].append(Fp @ p.log() + (1 - Fp) @ (1 - p).log())
        if model.has_fwd:
            out["fwd"].append(torch.einsum("se,sme->sm", se, fwd_emb(model, P, Fp, cv)).mean(0))
    return {k: torch.cat(v).float().cpu().numpy() for k, v in out.items()}


def z(x):
    return (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="work/cells")
    ap.add_argument("--only", default="", help="comma-separated cell names")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--sae", default="")
    ap.add_argument("--work", default="work/w3", help="lib + emb for query spectra")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-mols", type=int, default=0)
    ap.add_argument("--dump", action="store_true", help="also save candidate-level z-scores (for fusion experiments)")
    ap.add_argument("--shift-pos", type=float, default=0.0, help="add this to positive-mode query m/z (electron-convention test)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = st = None
    if a.ckpt:   # several comma-separated checkpoints = ensemble (per-model z-scores averaged)
        from casmi.predict import load_model
        from casmi.data import SpectraStore
        models = [load_model(c, a.sae, dev)[0] for c in a.ckpt.split(",")]
        model = models[0]
        st = SpectraStore(f"{a.work}/lib", f"{a.work}/emb")
    names = a.only.split(",") if a.only else sorted(d for d in os.listdir(a.cells) if os.path.exists(f"{a.cells}/{d}/cands.parquet"))
    os.makedirs(f"{a.cells}/eval", exist_ok=True)
    summary = {"tag": a.tag, "ckpt": a.ckpt, "cells": {}}
    for name in names:
        d = f"{a.cells}/{name}"
        t0 = time.time()
        mols = pd.read_parquet(f"{d}/molecules.parquet")
        S = pd.read_parquet(f"{d}/structs.parquet", columns=["sid", "lsid", "lpmid"])
        doc_all = (S.lsid + S.lpmid).to_numpy(np.float32)
        C = pd.read_parquet(f"{d}/cands.parquet")
        Q = pd.read_parquet(f"{d}/queries.parquet")
        qby = Q.groupby("mol_id").spec_idx.apply(np.array).to_dict()
        if model is not None:
            plm_all = torch.as_tensor(np.load(f"{d}/plm.npy", mmap_mode="r")[:], device="cpu")
            fp_all = np.load(f"{d}/fp.npy", mmap_mode="r")
        rows, dumps = [], []
        for k, (mid, g) in enumerate(C.groupby("mol_id", sort=False)):
            if a.max_mols and k >= a.max_mols:
                break
            sid = g.sid.to_numpy(); truth = g.is_truth.to_numpy(); sf = g.same_formula.to_numpy()
            if not truth.any():
                continue
            doc = doc_all[sid]
            jit = np.random.default_rng(hash(mid) % 2**32).random(len(sid)) * 1e-6
            V = {"prior": doc.astype(np.float64)}
            if model is not None:
                b = st.collate(qby[mid], device=dev)
                if a.shift_pos:
                    pos = (b["mode"] == 1)[:, None] & ~b["pad"]
                    b["mz"] = torch.where(pos, b["mz"] + a.shift_pos, b["mz"])
                    b["prec"] = torch.where(b["mode"] == 1, b["prec"] + a.shift_pos, b["prec"])
                plm = plm_all[sid].to(dev)
                fp = torch.as_tensor(np.unpackbits(fp_all[sid], axis=1), device=dev)
                Hs = [head_scores(m_, b, plm, fp) for m_ in models]
                H = {h: np.mean([z(x[h]) for x in Hs], 0) if len(Hs) > 1 else Hs[0][h] for h in Hs[0]}
                V.update(H)
                if a.dump:
                    dumps.append(pd.DataFrame({"mol_id": mid, "sid": sid, "is_truth": truth, "same_formula": sf,
                                               **{f"z_{h}": z(v).astype(np.float16) for h, v in H.items()}}))
                ens = z(H["nce"]) + z(H["jepa"]) + 0.5 * z(H["fp"]) + (z(H["fwd"]) if "fwd" in H else 0)
                V["ens"] = ens
                if "fwd" in H:   # the deployed combination (hybrid/ours_neural_scores.py)
                    V["nf"] = (z(H["nce"]) + z(H["fwd"])) / 2
            r = {"mol_id": mid, "n": len(sid), "n_formula": int(sf.sum())}
            for setting, m in (("mass", np.ones(len(sid), bool)), ("formula", sf)):
                if not truth[m].any():
                    continue
                for v, s in V.items():
                    r[f"{setting}:{v}"] = rr_at25(s[m] if v == "prior" else z(s[m]), truth[m], jit[m])
                if model is not None:
                    for v in ("nce", "fwd", "ens", "nf"):
                        if v not in V:
                            continue
                        zz = z(V[v][m])
                        for w in W_GRID:
                            r[f"{setting}:{v}+{w}doc"] = rr_at25(zz + w * doc[m], truth[m], jit[m])
            rows.append(r)
        if dumps:
            pd.concat(dumps, ignore_index=True).to_parquet(f"{a.cells}/eval/{a.tag}__{name}__cands.parquet", index=False)
        R = pd.DataFrame(rows).merge(mols[["mol_id", "mass_bin", "doc", "exact_mass", "pc_formula_isomers"]], on="mol_id")
        R.to_parquet(f"{a.cells}/eval/{a.tag}__{name}.parquet", index=False)
        cols = [c for c in R.columns if ":" in c]
        R["doc_t"] = pd.qcut(R.doc.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
        res = {"n": len(R), "mrr": R[cols].mean().round(4).to_dict(),
               "by_doc": R.groupby("doc_t")[cols].mean().round(4).to_dict(orient="index"),
               "by_mass": R.groupby("mass_bin")[cols].mean().round(4).to_dict(orient="index")}
        summary["cells"][name] = res
        print(f"\n== {name}  n={len(R)}  [{time.time() - t0:.0f}s]")
        best = sorted(((v, c) for c, v in res["mrr"].items()), reverse=True)
        for setting in ("mass", "formula"):
            line = [f"{c.split(':', 1)[1]}={v:.3f}" for v, c in best if c.startswith(setting + ":")]
            print(f"  {setting:8s} " + "  ".join(line[:14]))
    json.dump(summary, open(f"{a.cells}/eval/{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
