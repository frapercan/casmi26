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
from casmi.gpustore import GPUStore  # noqa: E402
from casmi.valset import excluded_spectra, load_exclude, load_exclude_all  # noqa: E402
from casmi.model import SpecMolModel, cond_vec, fwd_emb, losses, sae_pred  # noqa: E402
from casmi.sae import TopKSAE  # noqa: E402
from casmi.spectra import neutral_mass  # noqa: E402


@torch.no_grad()
def evaluate(model, st: SpectraStore, hold_ik: list[str], ppm: float, device, max_mols=None, sae=None):
    model.eval()
    # distractors disjoint from the training negatives when the table has them (scripts/build_mol_table.py)
    db = st.mols[st.mols.source == ("dist" if (st.mols.source == "dist").any() else "db")]
    db_mass, db_rows = db.exact_mass.to_numpy(), db.index.to_numpy()
    o = np.argsort(db_mass)
    db_mass, db_rows = db_mass[o], db_rows[o]
    rr = ({"nce": [], "jepa": [], "fp": [], "ens": []} | ({"sae": []} if sae is not None else {})
          | ({"fwd": []} if model.has_fwd else {}))
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
        if model.has_fwd:
            s_fwd = torch.einsum("se,sme->sm", model.spec_emb(h), fwd_emb(model, plm, fp, cond_vec(model, b))).mean(0)
            scores["fwd"] = s_fwd
            scores["ens"] = scores["ens"] + z(s_fwd)
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
    # fast-iteration knobs (big GPUs)
    ap.add_argument("--gpu-data", action="store_true", help="keep the training set on the GPU (GPUStore)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    ap.add_argument("--local-frac", type=float, default=0.5, help="share of each batch from one mass window")
    ap.add_argument("--w-nce", type=float, default=1.0)
    ap.add_argument("--w-pred", type=float, default=1.0)
    ap.add_argument("--w-fp", type=float, default=0.5)
    ap.add_argument("--w-sae", type=float, default=1.0)
    ap.add_argument("--fwd", action="store_true", help="forward world-model head (molecule + conditions -> spectrum latent)")
    ap.add_argument("--hard-k", type=int, default=0, help="database same-mass negatives per spectrum (needs --gpu-data)")
    ap.add_argument("--hard-ppm", type=float, default=10.0)
    ap.add_argument("--hard-pool", type=int, default=0, help="draw this many same-mass candidates, keep the hard-k most Tanimoto-similar")
    ap.add_argument("--w-hard", type=float, default=1.0)
    ap.add_argument("--w-fwd", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val", default="", help="validation dir from make_valset.py (its spectra are hidden)")
    ap.add_argument("--tag", default="", help="experiment name for the results log")
    ap.add_argument("--results", default="", help="append final metrics as one JSON line here")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    spec_cols = pd.read_parquet(f"{a.work}/lib/spectra.parquet", columns=["ingest_lib", "inchikey14"])
    val_ex = excluded_spectra(spec_cols, load_exclude(a.val), load_exclude_all(a.val))
    print(f"validation set {a.val or '-'}: {val_ex.sum()} spectra hidden from training", flush=True)
    st = SpectraStore(f"{a.work}/lib", f"{a.work}/emb", exclude=val_ex)
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
    model = SpecMolModel(d=a.d, layers=a.layers, drop=a.drop, sae_latent=sae.enc.out_features if sae else 0, fwd=a.fwd).to(dev)
    if a.init_encoder:
        model.enc.load_state_dict(torch.load(a.init_encoder, map_location=dev))
        print("encoder initialised from", a.init_encoder, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, fused=dev == "cuda")
    if a.gpu_data:
        gs = GPUStore(st, dev, mol_rows=tr_rows)
        gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
        steps_per_epoch = gs.epoch_steps(a.batch, a.local_frac)

        def batches():
            while True:
                b = gs.collate(gs.sample(a.batch, gen, a.local_frac))
                yield gs.hard_negatives(b, a.hard_k, gen, a.hard_ppm, a.hard_pool) if a.hard_k else b
        print(f"GPUStore: {gs.n_spec} spectra, {gs.n_mol} train molecules on {dev}", flush=True)
    else:
        sampler = MassLocalSampler(st, tr_rows, a.batch, seed=a.seed)
        steps_per_epoch = len(sampler)

        def batches():
            while True:
                for idx in sampler:
                    yield st.collate(idx, device=dev)
    total = a.max_steps or int(a.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=max(0.05, 3 / total))
    use_amp = dev == "cuda" and a.amp != "off"
    bf16_ok = dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    amp_dt = torch.bfloat16 if (a.amp == "bf16" or (a.amp == "auto" and bf16_ok)) else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dt == torch.float16)
    assert not a.hard_k or a.gpu_data, "--hard-k needs --gpu-data"
    lw = dict(w_nce=a.w_nce, w_pred=a.w_pred, w_fp=a.w_fp, w_sae=a.w_sae, w_hard=a.w_hard, w_fwd=a.w_fwd)
    step_fn = losses
    if a.compile:
        step_fn = torch.compile(losses, dynamic=True)  # batch size varies (unique molecules)
    print(f"steps {total} ({steps_per_epoch}/epoch)  amp {use_amp and amp_dt}  compile {a.compile}", flush=True)
    step, t0, best, log = 0, time.time(), -1, []
    t_log, n_log = time.time(), 0
    for b in batches():
        if sae is not None:
            with torch.no_grad():
                b["sae"] = sae.encode(b["plm"])
        with torch.autocast(dev, dtype=amp_dt, enabled=use_amp):
            loss, info = step_fn(model, b, **lw)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        n_log += len(b["mz"])
        if step % a.log_every == 0:
            dt = time.time() - t_log
            print(f"step {step}/{total} {time.time()-t0:.0f}s {a.log_every/dt:.1f} it/s {n_log/dt:.0f} spec/s " +
                  " ".join(f"{k}={float(v):.3f}" for k, v in info.items()), flush=True)
            t_log, n_log = time.time(), 0
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
    if a.results:
        import subprocess
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                    cwd=os.path.dirname(__file__)).stdout.strip()
        except Exception:
            commit = ""
        rec = {"tag": a.tag, "commit": commit, "time_s": round(time.time() - t0), "steps": total,
               "device": torch.cuda.get_device_name() if dev == "cuda" else "cpu",
               "args": vars(a), "final": ev, "best_partial": best}
        with open(a.results, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
