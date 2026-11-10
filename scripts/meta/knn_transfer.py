"""E002 (pre-registered in registry/explorations/E002.json): KNN transfer from a vector index of library spectra.

The index holds every library spectrum the model trained on, embedded with the model's spectrum head (cosine).
For each cell molecule, the k nearest index spectra of each of its query spectra are pooled by library structure
(max cosine), weighted by softmax(cos / tau), and every candidate gets knn = sum_n w_n * Tanimoto(candidate, n)
(Morgan r2 2048). Arms: A0 = z(nf) + 0.25 doc, A1 = A0 + beta * z(knn), beta chosen on the other cell.
Writes work/cells/eval/e002_A{0,1}__<cell>.parquet (column mass:score) for scripts/meta/registry.py.

  python scripts/meta/knn_transfer.py --ckpt work/runs/w4base_s0/best.pt --dump-tag w4base_s0_dump
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from casmi.data import SpectraStore  # noqa: E402
from casmi.predict import load_model  # noqa: E402
from casmi.valset import excluded_spectra, load_exclude, load_exclude_all  # noqa: E402

BETAS = [0.25, 0.5, 1.0, 2.0]


@torch.no_grad()
def embed(model, st, idx, dev, bs=2048):
    out = np.zeros((len(idx), model.spec_emb.out_features if hasattr(model.spec_emb, "out_features") else 256), np.float16)
    first = True
    for i in range(0, len(idx), bs):
        b = st.collate(idx[i:i + bs], device=dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            e = F.normalize(model.spec_emb(model.spec(b)).float(), dim=-1)
        if first and e.shape[1] != out.shape[1]:
            out = np.zeros((len(idx), e.shape[1]), np.float16)
        first = False
        out[i:i + bs] = e.cpu().numpy().astype(np.float16)
    return out


@torch.no_grad()
def topk(q, index, k, dev, chunk=250_000, qb=1024):
    """Exact cosine top-k of every row of q among the index rows (both unit-norm float16)."""
    vals = np.full((len(q), k), -2.0, np.float32); ids = np.zeros((len(q), k), np.int64)
    for j in range(0, len(index), chunk):
        X = torch.as_tensor(index[j:j + chunk], device=dev)
        for i in range(0, len(q), qb):
            Q = torch.as_tensor(q[i:i + qb], device=dev)
            v, t = (Q @ X.T).float().topk(min(k, len(X)), dim=1)
            V = np.concatenate([vals[i:i + qb], v.cpu().numpy()], 1)
            I = np.concatenate([ids[i:i + qb], t.cpu().numpy() + j], 1)
            o = np.argsort(-V, 1)[:, :k]
            vals[i:i + qb], ids[i:i + qb] = np.take_along_axis(V, o, 1), np.take_along_axis(I, o, 1)
    return vals, ids


def tanimoto(C, N):
    """C (c, bits) and N (n, bits) as float32 0/1 -> (c, n)."""
    inter = C @ N.T
    return inter / (C.sum(1, keepdims=True) + N.sum(1)[None, :] - inter + 1e-9)


def rr(score, truth, jit):
    s = score + jit
    r = int((s > s[truth].max()).sum()) + 1
    return 1.0 / r if r <= 25 else 0.0


def z(x):
    return (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work/runs/w4base_s0/best.pt")
    ap.add_argument("--work", default="work/w4")
    ap.add_argument("--val", default="work/cells")
    ap.add_argument("--cells", default="work/cells")
    ap.add_argument("--dump-tag", default="w4base_s0_dump")
    ap.add_argument("--out", default="work/knn/w4base_s0")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--holdout", default="/kaggle/input/casmi26-train-v2/casmi26/splits/holdout.parquet")
    a = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(a.ckpt, "", dev)
    st = SpectraStore(f"{a.work}/lib", f"{a.work}/emb")
    os.makedirs(a.out, exist_ok=True)
    # the index = exactly the spectra the model trained on (scripts/train.py)
    val_ex = excluded_spectra(st.spec[["ingest_lib", "inchikey14"]], load_exclude(a.val), load_exclude_all(a.val))
    hold = set(pd.read_parquet(a.holdout).inchikey14) | set(st.spec.inchikey14[st.spec.ingest_lib == "enveda-np-examples"])
    vis = st.spec.mol_row.notna().to_numpy() & ~val_ex & ~st.spec.inchikey14.isin(hold).to_numpy()
    idx = np.flatnonzero(vis)
    idx_ik = set(st.spec.inchikey14.to_numpy()[idx])
    for c in ("np_c2", "np_tims"):
        m = pd.read_parquet(f"{a.cells}/{c}/molecules.parquet")
        bad = idx_ik & (set(m.ik14_lib) | set(m.ik14_raw.dropna()) | set(m.skey.dropna()))
        assert not bad, f"index holds spectra of {len(bad)} {c} structures"
    print(f"index: {len(idx)} spectra of {len(idx_ik)} structures, free of cell structures [{time.time() - t0:.0f}s]", flush=True)
    p_emb = f"{a.out}/index_emb.npy"
    if os.path.exists(p_emb):
        E = np.load(p_emb)
    else:
        E = embed(model, st, idx, dev); np.save(p_emb, E)
    rows = st.a_row[idx]
    print(f"index embedded {E.shape} [{time.time() - t0:.0f}s]", flush=True)

    summary = {}
    per_cell = {}
    for c in ("np_c2", "np_tims"):
        Q = pd.read_parquet(f"{a.cells}/{c}/queries.parquet")
        qe = embed(model, st, Q.spec_idx.to_numpy(), dev)
        vals, ids = topk(qe, E, a.k, dev)
        D = pd.read_parquet(f"work/cells/eval/{a.dump_tag}__{c}__cands.parquet")
        S = pd.read_parquet(f"{a.cells}/{c}/structs.parquet", columns=["sid", "lsid", "lpmid"])
        fp_c = np.load(f"{a.cells}/{c}/fp.npy", mmap_mode="r")
        doc_all = (S.lsid + S.lpmid).to_numpy(np.float32)
        qpos = Q.groupby("mol_id").indices
        res = []
        for mid, g in D.groupby("mol_id", sort=False):
            qi = qpos[mid]
            nb = pd.DataFrame({"row": rows[ids[qi].ravel()], "cos": vals[qi].ravel()}).groupby("row").cos.max()
            w = np.exp((nb.to_numpy() - nb.max()) / a.tau); w /= w.sum()
            Nf = np.unpackbits(st.fp[nb.index.to_numpy()], axis=1).astype(np.float32)
            sid = g.sid.to_numpy()
            Cf = np.unpackbits(fp_c[sid], axis=1).astype(np.float32)
            knn = tanimoto(Cf, Nf) @ w
            nf = z((g.z_nce.to_numpy(np.float32) + g.z_fwd.to_numpy(np.float32)) / 2)
            truth = g.is_truth.to_numpy(); doc = doc_all[sid]
            jit = np.random.default_rng(hash(mid) % 2**32).random(len(sid)) * 1e-6
            r = {"mol_id": mid, "A0": rr(nf + 0.25 * doc, truth, jit), "knn_alone": rr(knn, truth, jit),
                 "doc+knn": rr(0.25 * doc + z(knn), truth, jit), "nn_cos_max": float(nb.max()),
                 "truth_knn_pct": float((knn < knn[truth].max()).mean())}
            for b in BETAS:
                r[f"A1_beta{b}"] = rr(nf + 0.25 * doc + b * z(knn), truth, jit)
            res.append(r)
        R = pd.DataFrame(res)
        per_cell[c] = R
        summary[c] = {k: float(R[k].mean()) for k in R.columns if k != "mol_id"}
        print(f"{c}: " + "  ".join(f"{k}={v:.4f}" for k, v in summary[c].items()) + f" [{time.time() - t0:.0f}s]", flush=True)
    other = {"np_c2": "np_tims", "np_tims": "np_c2"}
    for c, R in per_cell.items():
        bstar = max(BETAS, key=lambda b: summary[other[c]][f"A1_beta{b}"])
        summary[c]["beta_used"] = bstar
        R[["mol_id"]].assign(**{"mass:score": R.A0}).to_parquet(f"work/cells/eval/e002_A0__{c}.parquet", index=False)
        R[["mol_id"]].assign(**{"mass:score": R[f"A1_beta{bstar}"]}).to_parquet(f"work/cells/eval/e002_A1__{c}.parquet", index=False)
        print(f"{c}: beta {bstar} chosen on {other[c]} -> A0 {R.A0.mean():.4f}  A1 {R[f'A1_beta{bstar}'].mean():.4f}")
    json.dump(summary, open(f"{a.out}/e002_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
