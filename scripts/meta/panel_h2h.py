"""Head-to-head on the public natural-product panel (250 Enveda timsTOF NPs = enveda-np-examples).

Same candidates (the public pipeline's PubChem window, casmi26-natural-product-panel-benchmark/pubchem_candidates)
and the same metric as that dataset's quick start (in-pool rows dropped except the truth, one guess per scored
identity), so our model is compared with the public FPNet scores cached there:
  ab      public FPNet A+B of the v4g pipeline (trained on these molecules: optimistic)
  fold0   fold-safe FPNet, never saw the panel (honest)
  ours    our checkpoint (np-examples excluded from its training: honest)
Each spectral score is z-scored within the molecule and combined with doc = ln(1+substances) + ln(1+PubMed).

  python scripts/meta/panel_h2h.py --ckpt work/runs/w4base_s0/best.pt --tag w4base_s0
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from eval_cells import head_scores  # noqa: E402

D = "/kaggle/input/casmi26-natural-product-panel-benchmark/"
TIER = "/kaggle/input/casmi26-pubchem-popularity-prior/casmi26-pubchem-tier/"
W = [0.1, 0.15, 0.25, 0.5]


def mrr25(c, score, mols):
    """Replica of the panel quick-start metric on a fixed molecule list (missing molecules count 0)."""
    out = {}
    for m, g in c.assign(s=score).groupby("molecule", observed=True):
        g = g[~g.in_pool | g.is_truth].sort_values("s", ascending=False, kind="stable")
        ident = np.where(g.scorer_key != "", g.scorer_key, g.connectivity)
        first = ~pd.Series(ident).duplicated().values
        rank = np.flatnonzero(g.is_truth.values[first])
        out[m] = 1 / (rank[0] + 1) if len(rank) and rank[0] < 25 else 0.0
    return pd.Series([out.get(m, 0.0) for m in mols], index=mols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cell", default="work/cells/np_tims")
    ap.add_argument("--work", default="work/w4")
    ap.add_argument("--plm-mean", default="work/emb_v4/plm_mean.npy")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = pd.read_parquet(D + "pubchem_candidates.parquet",
                        columns=["molecule", "tier_row", "connectivity", "scorer_key", "is_truth", "in_pool", "log_substances", "log_pubmed"])
    f = pd.read_parquet(D + "panel_fpnet_scores.parquet", columns=["ab_full", "fold0_full"])
    c["ab"], c["fold0"] = f.ab_full.to_numpy(), f.fold0_full.to_numpy()
    c["molecule"] = c.molecule.astype(str); c["connectivity"] = c.connectivity.astype(str); c["scorer_key"] = c.scorer_key.astype(str)
    c = c[np.isfinite(c.ab) & np.isfinite(c.fold0)].reset_index(drop=True)
    c["doc"] = (c.log_substances + c.log_pubmed).astype(np.float64)

    M = pd.read_parquet(f"{a.cell}/molecules.parquet")
    key2mid = {}
    for col in ("skey", "ik14_lib", "ik14_raw"):
        key2mid.update({k: m for k, m in zip(M[col], M.mol_id) if isinstance(k, str)})
    c["mol_id"] = c.molecule.map(key2mid)
    mols = sorted(set(c.molecule[c.mol_id.notna()]))
    print(f"panel molecules with candidates {c.molecule.nunique()}, matched to the cell {len(mols)}", flush=True)
    c = c[c.mol_id.notna()].reset_index(drop=True)

    # features per connectivity: the cell's structures, plus the rows outside our window embedded here
    S = pd.read_parquet(f"{a.cell}/structs.parquet", columns=["sid", "ik"])
    plm = np.load(f"{a.cell}/plm.npy", mmap_mode="r"); fp = np.load(f"{a.cell}/fp.npy", mmap_mode="r")
    sid_of = dict(zip(S.ik, S.sid))
    miss = c.loc[~c.connectivity.isin(sid_of)].drop_duplicates("connectivity")
    print(f"rows covered by the cell {c.connectivity.isin(sid_of).mean():.3f}; embedding {len(miss)} extra structures", flush=True)
    from build_cells import canon
    from casmi.plm import PLMEmbedder
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from embed_molecules import morgan
    off = np.load(TIER + "pc_off.npy", mmap_mode="r"); smi = np.load(TIER + "pc_smiles.npy", mmap_mode="r")
    raw = [bytes(smi[off[r]:off[r + 1]]).decode() for r in miss.tier_row.to_numpy()]
    can = [canon(s)[0] or s for s in raw]
    emb = PLMEmbedder(device=dev, mean=np.load(a.plm_mean))
    x_plm = emb.embed(can, batch_size=256).astype(np.float16)
    x_fp = np.stack([morgan(s) for s in can])
    extra = {k: i for i, k in enumerate(miss.connectivity)}

    from casmi.predict import load_model
    from casmi.data import SpectraStore
    model, _ = load_model(a.ckpt, "", dev)
    st = SpectraStore(f"{a.work}/lib", f"{a.work}/emb")
    Q = pd.read_parquet(f"{a.cell}/queries.parquet")
    qby = Q.groupby("mol_id").spec_idx.apply(np.array).to_dict()
    heads = {}
    for mid, g in c.groupby("mol_id", sort=False):
        ks = g.connectivity.unique()
        P = np.stack([plm[sid_of[k]] if k in sid_of else x_plm[extra[k]] for k in ks])
        Fp = np.stack([fp[sid_of[k]] if k in sid_of else x_fp[extra[k]] for k in ks])
        with torch.no_grad():
            H = head_scores(model, st.collate(qby[mid], device=dev), torch.as_tensor(P, device=dev),
                            torch.as_tensor(np.unpackbits(Fp, axis=1), device=dev))
        for h, v in H.items():
            heads.setdefault(h, {}).update({(mid, k): s for k, s in zip(ks, v)})
    for h in heads:
        c[h] = [heads[h][(m, k)] for m, k in zip(c.mol_id, c.connectivity)]
    z = lambda col: c.groupby("molecule", observed=True)[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    Z = {k: z(k) for k in ("ab", "fold0", "nce", "fwd")}
    rules = {"doc": c.doc, "ab": Z["ab"], "fold0": Z["fold0"], "ours_nce": Z["nce"], "ours_fwd": Z["fwd"]}
    for w in W:
        rules[f"ab+{w}doc"] = Z["ab"] + w * c.doc
        rules[f"fold0+{w}doc"] = Z["fold0"] + w * c.doc
        rules[f"ours_fwd+{w}doc"] = Z["fwd"] + w * c.doc
        rules[f"ours_nce+{w}doc"] = Z["nce"] + w * c.doc
        rules[f"fold0+ours_fwd+{w}doc"] = (Z["fold0"] + Z["fwd"]) / np.sqrt(2) + w * c.doc
    R = pd.DataFrame({k: mrr25(c, v, mols) for k, v in rules.items()})
    os.makedirs("work/cells/eval", exist_ok=True)
    R.to_parquet(f"work/cells/eval/panel_h2h_{a.tag}.parquet")
    res = R.mean().round(4).sort_values(ascending=False)
    json.dump({"tag": a.tag, "n": len(mols), "mrr": res.to_dict()}, open(f"work/cells/eval/panel_h2h_{a.tag}.json", "w"), indent=1)
    print(res.to_string())
    # paired bootstrap: ours vs fold-safe, both with their best weight
    rng = np.random.default_rng(0)
    for x, y in (("ours_fwd+0.25doc", "fold0+0.25doc"), ("fold0+ours_fwd+0.25doc", "fold0+0.25doc"), ("ours_fwd", "fold0")):
        d = (R[x] - R[y]).to_numpy()
        bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(2000)]
        print(f"{x} - {y}: {d.mean():+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")


if __name__ == "__main__":
    main()
