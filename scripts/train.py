"""Train the spectrum->PLM-latent model and evaluate on the class-2 holdout.

Held-out molecules (holdout.parquet) and enveda-np-examples are excluded from
training. Evaluation ranks, per held-out molecule, all DB candidates within
--ppm of its precursor-derived mass plus the true structure.
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
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.data import MassLocalSampler, SpectraStore  # noqa: E402
from casmi.model import SpecMolModel, losses, sae_pred  # noqa: E402
from casmi.sae import TopKSAE  # noqa: E402
from casmi.spectra import neutral_mass  # noqa: E402


@torch.no_grad()
def evaluate(model, st: SpectraStore, hold_ik: list[str], ppm: float, device, max_mols=None, sae=None):
    model.eval()
    db = st.mols[st.mols.source == "db"]
    db_mass, db_rows = db.exact_mass.to_numpy(), db.index.to_numpy()
    o = np.argsort(db_mass)
    db_mass, db_rows = db_mass[o], db_rows[o]
    rr = {"nce": [], "jepa": [], "fp": [], "ens": []} | ({"sae": []} if sae is not None else {})
    for ik in hold_ik[:max_mols]:
        true_row = int(st.row_of_ik[ik])
        sidx = st.spec_by_row.get(true_row)
        if sidx is None:
            continue
        b = st.collate(sidx, device=device)
        mass = float(np.median([neutral_mass(p, a) for p, a in
                                zip(st.spec.precursor_mz.iloc[sidx], st.spec.adduct.iloc[sidx])]))
        tol = max(mass * ppm * 1e-6, 0.003)
        lo, hi = np.searchsorted(db_mass, mass - tol), np.searchsorted(db_mass, mass + tol, side="right")
        rows = np.concatenate([[true_row], db_rows[lo:hi]])
        plm = torch.as_tensor(st.plm[rows].astype(np.float32), device=device)
        fp = torch.as_tensor(np.unpackbits(st.fp[rows], axis=1).astype(np.float32), device=device)
        h = model.spec(b)
        s_nce = (model.spec_emb(h) @ model.mol_emb(plm, fp).T).mean(0)
        s_jepa = (F.normalize(model.pred(h), dim=-1) @ plm.T).mean(0)
        p = torch.sigmoid(model.fp(h)).mean(0).clamp(1e-4, 1 - 1e-4)
        s_fp = fp @ p.log() + (1 - fp) @ (1 - p).log()
        z = lambda s: (s - s.mean()) / (s.std() + 1e-6)
        scores = {"nce": s_nce, "jepa": s_jepa, "fp": s_fp, "ens": z(s_nce) + z(s_jepa) + 0.5 * z(s_fp)}
        if sae is not None:
            codes = F.normalize(sae.encode(plm), dim=-1)
            s_sae = (F.normalize(sae_pred(model, h), dim=-1) @ codes.T).mean(0)
            scores["sae"] = s_sae
            scores["ens"] = scores["ens"] + z(s_sae)
        for k, s in scores.items():
            rank = int((s > s[0]).sum()) + 1  # true is row 0
            rr[k].append(1 / rank if rank <= 25 else 0.0)
    model.train()
    return {k: float(np.mean(v)) for k, v in rr.items()} | {"n": len(rr["nce"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/home/user/work")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-mols", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", default="/home/user/work/runs/v1")
    ap.add_argument("--sae", default="", help="path to sae.pt (adds sparse-code head)")
    ap.add_argument("--init-encoder", default="", help="JEPA-pretrained encoder.pt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    st = SpectraStore(f"{a.work}/lib", f"{a.work}/emb")
    hold = pd.read_parquet(f"{a.work}/holdout.parquet").inchikey14.tolist()
    excl = set(hold) | set(st.spec.loc[st.spec.ingest_lib == "enveda-np-examples", "inchikey14"])
    tr_rows = np.array([r for r, ik in zip(st.row_of_ik.values, st.row_of_ik.index)
                        if ik not in excl and r in st.spec_by_row])
    print(f"train molecules {len(tr_rows)}  holdout {len(hold)}  device {dev}", flush=True)

    sae = None
    if a.sae:
        ck = torch.load(a.sae, map_location=dev)
        sae = TopKSAE(ck["d_in"], ck["n_latent"], ck["k"]).to(dev).eval().requires_grad_(False)
        sae.load_state_dict(ck["state"])
    model = SpecMolModel(d=a.d, layers=a.layers, sae_latent=sae.enc.out_features if sae else 0).to(dev)
    if a.init_encoder:
        model.enc.load_state_dict(torch.load(a.init_encoder, map_location=dev))
        print("encoder initialised from", a.init_encoder, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sampler = MassLocalSampler(st, tr_rows, a.batch)
    total = a.max_steps or int(a.epochs * len(sampler))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=max(0.05, 3 / total))
    use_amp = dev == "cuda"
    step, t0, best, log = 0, time.time(), -1, []
    while step < total:
        for idx in sampler:
            b = st.collate(idx, device=dev)
            if sae is not None:
                b["sae"] = sae.encode(b["plm"])
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
                loss, info = losses(model, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(f"step {step}/{total} {time.time()-t0:.0f}s " +
                      " ".join(f"{k}={v:.3f}" for k, v in info.items()), flush=True)
            if step % a.eval_every == 0 or step == total:
                ev = evaluate(model, st, hold, a.ppm, dev, a.eval_mols, sae)
                print(f"EVAL step {step}: " + json.dumps(ev), flush=True)
                log.append({"step": step, **ev})
                if ev["ens"] > best:
                    best = ev["ens"]
                    torch.save({"model": model.state_dict(), "args": vars(a)}, f"{a.out}/best.pt")
            if step >= total:
                break
    model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=dev)["model"])
    ev = evaluate(model, st, hold, a.ppm, dev, sae=sae)
    print("FINAL holdout (all):", json.dumps(ev), flush=True)
    json.dump(log + [{"final": ev}], open(f"{a.out}/log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
