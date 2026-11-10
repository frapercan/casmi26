"""E004 (pre-registered in registry/explorations/E004.json; supersedes E003): KNN transfer with a model-free
spectral similarity. Same index, pooling, Tanimoto transfer and cross-cell beta choice as E002, but the neighbours
come from a classic cosine: sqrt relative intensity of the 50 most intense peaks, fragment bins of 0.01 Da plus
neutral-loss bins of 0.01 Da (losses >= 1.5 Da), each block L2-normalised and weighted 1/sqrt(2).
Writes work/cells/eval/e004_A{0,1}__<cell>.parquet (column mass:score).

  python scripts/meta/knn_cosine.py
"""
from __future__ import annotations

import argparse, json, os, sys, time
from multiprocessing import get_context

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from casmi.data import SpectraStore  # noqa: E402
from casmi.valset import excluded_spectra, load_exclude, load_exclude_all  # noqa: E402
from knn_transfer import BETAS, embed, rr, tanimoto, topk, z  # noqa: E402

BW, TOP, NL_MIN, MZ_MAX = 0.01, 50, 1.5, 2000.0
NF = int(MZ_MAX / BW) + 1
G = {}


def sparse_spectra(st, idx):
    """CSR (len(idx), 2*NF): fragment block (+) neutral-loss block, sqrt intensities, each block unit-norm / sqrt(2)."""
    idx = np.asarray(idx)
    starts = st.off[idx].astype(np.int64); n = (st.off[idx + 1] - st.off[idx]).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(n)[:-1]])
    seg = np.repeat(np.arange(len(idx)), n)
    pos = np.repeat(starts - cum, n) + np.arange(int(n.sum()))
    mz, it = st.mz[pos].astype(np.float64), st.it[pos].astype(np.float64)
    o = np.lexsort((-it, seg))
    keep = o[(np.arange(len(o)) - cum[seg[o]]) < TOP]
    seg, mz, v = seg[keep], mz[keep], np.sqrt(np.clip(it[keep], 0, None))
    prec = st.a_prec[idx].astype(np.float64)[seg]
    blocks = []
    for cols, ok in ((np.rint(mz / BW), mz < MZ_MAX), (np.rint((prec - mz) / BW), (prec - mz >= NL_MIN) & (prec - mz < MZ_MAX))):
        r, c, d = seg[ok], cols[ok].astype(np.int64), v[ok]
        M = sp.csr_matrix((d, (r, c)), shape=(len(idx), NF), dtype=np.float32)
        M.sum_duplicates()
        nrm = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()); nrm[nrm == 0] = 1
        blocks.append(sp.diags((1 / (nrm * np.sqrt(2))).astype(np.float32)) @ M)
    return sp.hstack(blocks, format="csr")


def _topk_block(i):
    Q = G["Q"][i:i + G["qb"]]
    k = G["k"]
    best_v = np.full((Q.shape[0], k), -1.0, np.float32); best_i = np.zeros((Q.shape[0], k), np.int64)
    for off, LT in G["LT"]:
        S = (Q @ LT).toarray()
        kk = min(k, S.shape[1])
        t = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        V = np.concatenate([best_v, np.take_along_axis(S, t, 1)], 1); I = np.concatenate([best_i, t + off], 1)
        o = np.argsort(-V, 1)[:, :k]
        best_v, best_i = np.take_along_axis(V, o, 1), np.take_along_axis(I, o, 1)
    return i, best_v, best_i


def cos_topk(Q, L, k, workers, chunk=200_000, qb=128):
    G.update(Q=Q, k=k, qb=qb, LT=[(j, L[j:j + chunk].T.tocsr()) for j in range(0, L.shape[0], chunk)])
    vals = np.zeros((Q.shape[0], k), np.float32); ids = np.zeros((Q.shape[0], k), np.int64)
    with get_context("fork").Pool(workers) as P:
        for i, v, t in P.imap_unordered(_topk_block, range(0, Q.shape[0], qb)):
            vals[i:i + qb], ids[i:i + qb] = v, t
    return vals, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work/runs/w4base_s0/best.pt")
    ap.add_argument("--work", default="work/w4")
    ap.add_argument("--val", default="work/cells")
    ap.add_argument("--cells", default="work/cells")
    ap.add_argument("--dump-tag", default="w4base_s0_dump")
    ap.add_argument("--emb-index", default="work/knn/w4base_s0/index_emb.npy", help="E002 index, for the secondary arm")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--holdout", default="/kaggle/input/casmi26-train-v2/casmi26/splits/holdout.parquet")
    a = ap.parse_args()
    t0 = time.time()
    st = SpectraStore(f"{a.work}/lib", f"{a.work}/emb")
    val_ex = excluded_spectra(st.spec[["ingest_lib", "inchikey14"]], load_exclude(a.val), load_exclude_all(a.val))
    hold = set(pd.read_parquet(a.holdout).inchikey14) | set(st.spec.inchikey14[st.spec.ingest_lib == "enveda-np-examples"])
    idx = np.flatnonzero(st.spec.mol_row.notna().to_numpy() & ~val_ex & ~st.spec.inchikey14.isin(hold).to_numpy())
    idx_ik = set(st.spec.inchikey14.to_numpy()[idx])
    for c in ("np_c2", "np_tims"):
        m = pd.read_parquet(f"{a.cells}/{c}/molecules.parquet")
        assert not idx_ik & (set(m.ik14_lib) | set(m.ik14_raw.dropna()) | set(m.skey.dropna())), f"index holds {c} structures"
    L = sparse_spectra(st, idx)
    rows = st.a_row[idx]
    print(f"index {L.shape[0]} spectra, {L.nnz} non-zeros, free of cell structures [{time.time() - t0:.0f}s]", flush=True)
    import torch
    from casmi.predict import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(a.ckpt, "", dev)
    E_emb = np.load(a.emb_index)
    summary, per_cell = {}, {}
    for c in ("np_c2", "np_tims"):
        Qd = pd.read_parquet(f"{a.cells}/{c}/queries.parquet")
        Qs = sparse_spectra(st, Qd.spec_idx.to_numpy())
        cv, ci = cos_topk(Qs, L, a.k, a.workers)
        ev, ei = topk(embed(model, st, Qd.spec_idx.to_numpy(), dev), E_emb, a.k, dev)
        print(f"{c}: neighbours found [{time.time() - t0:.0f}s]", flush=True)
        D = pd.read_parquet(f"work/cells/eval/{a.dump_tag}__{c}__cands.parquet")
        S = pd.read_parquet(f"{a.cells}/{c}/structs.parquet", columns=["sid", "lsid", "lpmid"])
        fp_c = np.load(f"{a.cells}/{c}/fp.npy", mmap_mode="r")
        doc_all = (S.lsid + S.lpmid).to_numpy(np.float32)
        qpos = Qd.groupby("mol_id").indices
        res = []

        def transfer(vals, ids, qi, Cf):
            nb = pd.DataFrame({"row": rows[ids[qi].ravel()], "s": vals[qi].ravel()}).groupby("row").s.max()
            w = np.exp((nb.to_numpy() - nb.max()) / a.tau); w /= w.sum()
            return tanimoto(Cf, np.unpackbits(st.fp[nb.index.to_numpy()], axis=1).astype(np.float32)) @ w, float(nb.max())

        for mid, g in D.groupby("mol_id", sort=False):
            qi = qpos[mid]; sid = g.sid.to_numpy()
            Cf = np.unpackbits(fp_c[sid], axis=1).astype(np.float32)
            kc, cmax = transfer(cv, ci, qi, Cf)
            ke, _ = transfer(ev, ei, qi, Cf)
            nf = z((g.z_nce.to_numpy(np.float32) + g.z_fwd.to_numpy(np.float32)) / 2)
            truth = g.is_truth.to_numpy(); doc = doc_all[sid]
            jit = np.random.default_rng(hash(mid) % 2**32).random(len(sid)) * 1e-6
            r = {"mol_id": mid, "A0": rr(nf + 0.25 * doc, truth, jit), "knn_cos_alone": rr(kc, truth, jit),
                 "doc+knn_cos": rr(0.25 * doc + z(kc), truth, jit), "cos_max": cmax,
                 "corr_knn_cos_emb": float(np.corrcoef(kc, ke)[0, 1]) if len(sid) > 2 else np.nan}
            for b in BETAS:
                r[f"A1_beta{b}"] = rr(nf + 0.25 * doc + b * z(kc), truth, jit)
                r[f"S_beta{b}"] = rr(nf + 0.25 * doc + b * z(kc) + b * z(ke), truth, jit)
            res.append(r)
        R = pd.DataFrame(res)
        per_cell[c] = R
        summary[c] = {k: float(R[k].mean()) for k in R.columns if k != "mol_id"}
        print(f"{c}: " + "  ".join(f"{k}={v:.4f}" for k, v in summary[c].items()) + f" [{time.time() - t0:.0f}s]", flush=True)
    other = {"np_c2": "np_tims", "np_tims": "np_c2"}
    for c, R in per_cell.items():
        b = max(BETAS, key=lambda x: summary[other[c]][f"A1_beta{x}"])
        bs = max(BETAS, key=lambda x: summary[other[c]][f"S_beta{x}"])
        summary[c].update(beta_used=b, beta_secondary=bs, secondary=float(R[f"S_beta{bs}"].mean()))
        R[["mol_id"]].assign(**{"mass:score": R.A0}).to_parquet(f"work/cells/eval/e004_A0__{c}.parquet", index=False)
        R[["mol_id"]].assign(**{"mass:score": R[f"A1_beta{b}"]}).to_parquet(f"work/cells/eval/e004_A1__{c}.parquet", index=False)
        print(f"{c}: beta {b} (chosen on {other[c]}) A0 {R.A0.mean():.4f} A1 {R[f'A1_beta{b}'].mean():.4f} | "
              f"secondary emb+cos beta {bs}: {R[f'S_beta{bs}'].mean():.4f}")
    os.makedirs("work/knn", exist_ok=True)
    json.dump(summary, open("work/knn/e004_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
