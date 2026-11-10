"""Learned score fusion on the validation cells: a linear conditional-logit (softmax over each molecule's candidate list)
on our model's z-scores and the tier's documentation / natural-product features, against the hand rule
z(model) + w * doc. Fit on one cell, test on the other (and 5-fold by molecule inside a cell), so the
learned weights are judged on molecules and selection rules they were not fitted on.

Needs candidate dumps: scripts/meta/eval_cells.py --ckpt ... --tag T --dump
  python scripts/meta/fusion_cells.py --tag w4base_s0_dump
"""
from __future__ import annotations

import argparse, json

import numpy as np
import pandas as pd
import torch

BASE = ["z_fwd", "z_nce", "z_fp"]
DOC = ["doc", "lsyn", "ndep", "lpatent", "lotus", "nvend", "lmesh", "np_coco", "np_lotus", "np_npatlas", "srcbits"]


def load(cells, tag, name):
    C = pd.read_parquet(f"{cells}/eval/{tag}__{name}__cands.parquet")
    S = pd.read_parquet(f"{cells}/{name}/structs.parquet",
                        columns=["sid", "lsid", "lpmid", "lpatent", "lsyn", "ndep", "np", "src_mask", "lotus", "nvend", "lmesh"])
    C = C.join(S.set_index("sid"), on="sid")
    C["doc"] = C.lsid + C.lpmid
    C["ndep"] = np.log1p(C.ndep); C["lotus"] = np.log1p(C.lotus); C["nvend"] = np.log1p(C.nvend)
    npb = C.np.fillna(0).astype(int)
    C["np_coco"], C["np_lotus"], C["np_npatlas"] = (npb & 1) > 0, (npb & 2) > 0, (npb & 4) > 0
    C["srcbits"] = C.src_mask.fillna(0).astype(int).map(lambda v: bin(v).count("1"))
    g = C.groupby("mol_id", sort=False)
    C["doc_rel"] = C.doc - g.doc.transform("max")            # documentation relative to the best of the list
    C["docmax"] = g.doc.transform("max")
    C["mol"] = pd.factorize(C.mol_id)[0]
    return C.reset_index(drop=True)


def design(C, feats, inter):
    X = [C[f].to_numpy(np.float32) for f in feats]
    names = list(feats)
    if inter:   # list-level context: how documented the best candidate is, interacting with model and doc
        L = ((C.docmax - 10) / 5).to_numpy(np.float32)
        for f in ("z_fwd", "doc"):
            if f in feats:
                X.append(C[f].to_numpy(np.float32) * L); names.append(f"{f}*docmax")
    return np.stack(X, 1), names


def mrr(C, s, mask=None):
    mol = C.mol.to_numpy(); t = C.is_truth.to_numpy()
    if mask is not None:
        mol, t, s = mol[mask], t[mask], s[mask]
    s = s + np.random.default_rng(0).random(len(s)) * 1e-7
    nm = mol.max() + 1
    st = np.full(nm, -np.inf); np.maximum.at(st, mol[t], s[t])
    rank = np.bincount(mol, weights=(s > st[mol]).astype(float), minlength=nm) + 1
    has = np.bincount(mol[t], minlength=nm) > 0
    rr = np.where(has & (rank <= 25), 1 / rank, 0.0)
    return float(rr[np.unique(mol)].mean())


def fit(X, mol, t, l2=1e-3, iters=200):
    X = torch.as_tensor(X); mol = torch.as_tensor(mol); t = torch.as_tensor(t)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True)
    nm = int(mol.max()) + 1
    opt = torch.optim.LBFGS([w], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = Xn @ w
        mx = torch.full((nm,), -1e9).scatter_reduce(0, mol, s.detach(), "amax")
        lse = torch.log(torch.zeros(nm).index_add(0, mol, torch.exp(s - mx[mol]))) + mx
        st = torch.zeros(nm).index_add(0, mol[t], s[t])
        loss = (lse - st).mean() + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return (w.detach() / sd).numpy(), float(-(mu / sd) @ w.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="work/cells")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    D = {n: load(a.cells, a.tag, n) for n in ("np_c2", "np_tims")}
    out = {}
    for n, C in D.items():
        r = {"z_fwd": mrr(C, C.z_fwd.to_numpy()), "doc": mrr(C, C.doc.to_numpy())}
        for w in (0.1, 0.25, 0.5):
            r[f"hand z_fwd+{w}doc"] = mrr(C, C.z_fwd.to_numpy() + w * C.doc.to_numpy())
        out[n] = r
    configs = {"[z_fwd,doc]": (["z_fwd", "doc"], False),
               "[heads,doc]": (BASE + ["doc"], False),
               "[heads,doc,+NP,+docfeats]": (BASE + DOC, False),
               "[heads,doc,+NP,+docfeats,+doc_rel]": (BASE + DOC + ["doc_rel"], False),
               "[... + docmax interactions]": (BASE + DOC + ["doc_rel"], True)}
    weights = {}
    for cname, (feats, inter) in configs.items():
        for tr, te in (("np_c2", "np_tims"), ("np_tims", "np_c2")):
            Xtr, names = design(D[tr], feats, inter)
            w, _ = fit(Xtr, D[tr].mol.to_numpy(), D[tr].is_truth.to_numpy())
            Xte, _ = design(D[te], feats, inter)
            out[te][f"learned on {tr} {cname}"] = mrr(D[te], Xte @ w)
            weights[f"{cname} fit {tr}"] = dict(zip(names, np.round(w, 3).tolist()))
        # 5-fold by molecule inside np_c2
        C = D["np_c2"]; X, _ = design(C, feats, inter)
        fold = C.mol.to_numpy() % 5; s = np.zeros(len(C))
        for k in range(5):
            w, _ = fit(X[fold != k], pd.factorize(C.mol[fold != k])[0], C.is_truth.to_numpy()[fold != k])
            s[fold == k] = X[fold == k] @ w
        out["np_c2"][f"5-fold CV {cname}"] = mrr(C, s)
    for n, r in out.items():
        print(f"\n== {n}")
        for k, v in sorted(r.items(), key=lambda kv: -kv[1]):
            print(f"  {v:.4f}  {k}")
    print("\nweights (per raw feature unit):")
    for k, v in weights.items():
        print(f"  {k}: {v}")
    json.dump({"mrr": out, "weights": weights}, open(f"{a.cells}/eval/fusion_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
