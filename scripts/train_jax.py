"""JAX/Flax counterpart of scripts/train.py (same data, objective, evaluation and outputs).

Runs on TPU (Kaggle TPU VM), GPU or CPU. Data parallel over every local device through GSPMD:
the batch is sharded along its first axis, parameters are replicated, the step is one jit.

  python scripts/train_jax.py --work /kaggle/working/work --batch 2048 --fwd --hard-k 16 --out runs/jax1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax import serialization  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from casmi.data import SpectraStore  # noqa: E402
from casmi.spectra import neutral_mass  # noqa: E402
from casmi.valset import excluded_spectra, load_exclude, load_exclude_all  # noqa: E402
from casmi_jax.data import JaxStore, sample_batch, unpack_bits  # noqa: E402
from casmi_jax.model import SpecMolModel, l2n, losses  # noqa: E402

SMAX = 16  # spectra per molecule at evaluation (more are truncated)


def pow2(n, lo=64):
    return max(lo, 1 << int(np.ceil(np.log2(max(n, 1)))))


def make_scorer(model):
    @jax.jit
    def score(params, b, smask, plm, fp, mmask):
        m = model.bind({"params": params})
        h = m.spec(b)
        w = smask / smask.sum()
        zs = m.spec_emb(h)
        out = {"nce": w @ (zs @ m.mol_emb(plm, fp).T),
               "jepa": w @ (l2n(m.plm_pred(h)) @ l2n(plm).T)}
        p = jnp.clip(jax.nn.sigmoid(m.fp_logits(h)).T @ w, 1e-4, 1 - 1e-4)
        out["fp"] = fp @ jnp.log(p) + (1 - fp) @ jnp.log(1 - p)
        if model.fwd:
            out["fwd"] = w @ jnp.einsum("se,sme->sm", zs, m.fwd_emb(plm, fp, b))
        def z(s):
            mu = (s * mmask).sum() / mmask.sum()
            sd = jnp.sqrt((((s - mu) * mmask) ** 2).sum() / mmask.sum()) + 1e-6
            return (s - mu) / sd
        ens = z(out["nce"]) + z(out["jepa"]) + 0.5 * z(out["fp"]) + (z(out["fwd"]) if model.fwd else 0)
        out["ens"] = ens
        return {k: jnp.where(mmask > 0, v, -jnp.inf) for k, v in out.items()}
    return score


def evaluate(score, params, st: SpectraStore, hold_ik, ppm, max_mols=None):
    db = st.mols[st.mols.source == "db"]
    o = np.argsort(db.exact_mass.to_numpy())
    db_mass, db_rows = db.exact_mass.to_numpy()[o], db.index.to_numpy()[o]
    rr = {}
    for ik in hold_ik[:max_mols]:
        true_row = int(st.row_of_ik[ik])
        sidx = st.spec_by_row.get(true_row)
        if sidx is None:
            continue
        sidx = np.asarray(sidx)[:SMAX]
        b = st.collate(np.pad(sidx, (0, SMAX - len(sidx)), mode="edge"))
        b = {k: jnp.asarray(v.numpy()) for k, v in b.items() if k in
             ("mz", "it", "pad", "prec", "adduct", "mode", "ce")}
        smask = jnp.asarray((np.arange(SMAX) < len(sidx)).astype(np.float32))
        mass = float(np.median([neutral_mass(p, a) for p, a in
                                zip(st.spec.precursor_mz.iloc[sidx], st.spec.adduct.iloc[sidx])]))
        tol = max(mass * ppm * 1e-6, 0.003)
        lo, hi = np.searchsorted(db_mass, mass - tol), np.searchsorted(db_mass, mass + tol, side="right")
        rows = np.concatenate([[true_row], db_rows[lo:hi]])
        M = pow2(len(rows))
        rp = np.pad(rows, (0, M - len(rows)), constant_values=true_row)
        mmask = jnp.asarray((np.arange(M) < len(rows)).astype(np.float32))
        plm = jnp.asarray(st.plm[rp].astype(np.float32))
        fp = unpack_bits(jnp.asarray(st.fp[rp]))
        for k, s in score(params, b, smask, plm, fp, mmask).items():
            s = np.asarray(s)
            rank = int((s > s[0]).sum()) + 1
            rr.setdefault(k, []).append(1 / rank if rank <= 25 else 0.0)
    return {k: float(np.mean(v)) for k, v in rr.items()} | {"n": len(rr.get("nce", []))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/home/user/work")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch", type=int, default=1024, help="global batch (split across devices)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-mols", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", default="/home/user/work/runs/jax1")
    ap.add_argument("--local-frac", type=float, default=0.5)
    ap.add_argument("--w-nce", type=float, default=1.0)
    ap.add_argument("--w-pred", type=float, default=1.0)
    ap.add_argument("--w-fp", type=float, default=0.5)
    ap.add_argument("--w-hard", type=float, default=1.0)
    ap.add_argument("--w-fwd", type=float, default=1.0)
    ap.add_argument("--fwd", action="store_true")
    ap.add_argument("--hard-k", type=int, default=0)
    ap.add_argument("--hard-ppm", type=float, default=10.0)
    ap.add_argument("--hard-pool", type=int, default=0)
    ap.add_argument("--neg-all", action="store_true", help="negatives from every molecule of the same mass (database + train)")
    ap.add_argument("--self-mine", type=int, default=0, help="score --hard-pool same-mass candidates with the current model "
                    "(no gradient) and keep the K it confuses most with the answer as hard negatives")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true", help="bf16 compute (fp32 params)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--results", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    devs = jax.devices()
    print(f"JAX {jax.__version__} backend {jax.default_backend()} devices {len(devs)}: {devs[0].device_kind}", flush=True)
    assert a.batch % len(devs) == 0, "global batch must divide the device count"

    spec_cols = pd.read_parquet(f"{a.work}/lib/spectra.parquet", columns=["ingest_lib", "inchikey14"])
    val_ex = excluded_spectra(spec_cols, load_exclude(a.val), load_exclude_all(a.val))
    st = SpectraStore(f"{a.work}/lib", f"{a.work}/emb", exclude=val_ex)
    hold = pd.read_parquet(f"{a.work}/holdout.parquet").inchikey14.tolist()
    excl = set(hold) | set(st.spec.loc[st.spec.ingest_lib == "enveda-np-examples", "inchikey14"])
    tr_rows = np.array([r for r, ik in zip(st.row_of_ik.values, st.row_of_ik.index)
                        if ik not in excl and r in st.spec_by_row])
    t0 = time.time()
    js = JaxStore(st, mol_rows=tr_rows)
    n_mol = js.n_mol
    print(f"train molecules {len(tr_rows)} on device in {time.time()-t0:.0f}s; holdout {len(hold)}", flush=True)

    model = SpecMolModel(d=a.d, layers=a.layers, drop=a.drop, fwd=a.fwd,
                         dtype=jnp.bfloat16 if a.bf16 else jnp.float32)
    key = jax.random.PRNGKey(a.seed)
    key, kinit, kb = jax.random.split(key, 3)
    b0 = sample_batch(js.a, kb, 8, a.local_frac, a.hard_k, a.hard_ppm, 0 if a.self_mine else a.hard_pool, a.neg_all)
    params = model.init({"params": kinit, "dropout": kinit}, b0)["params"]
    print("params", sum(x.size for x in jax.tree_util.tree_leaves(params)), flush=True)
    steps_per_epoch = max(1, int(n_mol / (a.batch * a.local_frac)))
    total = a.max_steps or int(a.epochs * steps_per_epoch)
    sched = optax.cosine_onecycle_schedule(total, a.lr, pct_start=max(0.05, 3 / total))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=a.wd))
    opt_state = opt.init(params)

    mesh = Mesh(np.array(devs), ("data",))
    shard = NamedSharding(mesh, P("data"))
    repl = NamedSharding(mesh, P())
    params, opt_state = jax.device_put((params, opt_state), repl)
    store = jax.device_put(js.a, repl)
    del js
    lw = dict(w_nce=a.w_nce, w_pred=a.w_pred, w_fp=a.w_fp, w_hard=a.w_hard, w_fwd=a.w_fwd)

    @jax.jit  # the store is an argument, never a closure: closed-over arrays become compile-time constants
    def step(params, opt_state, store, key):
        kb, kd = jax.random.split(key)
        if a.self_mine:   # draw the whole pool, let the current model pick the negatives it confuses most
            b = sample_batch(store, kb, a.batch, a.local_frac, max(a.hard_pool, a.self_mine), a.hard_ppm, 0, a.neg_all)
            b = jax.lax.with_sharding_constraint(b, shard)
            m = model.bind({"params": jax.lax.stop_gradient(params)})
            zs = m.spec_emb(m.spec(b))
            zn = m.mol_emb(b["neg_plm"], b["neg_fp"])
            sim = jnp.where(b["neg_ok"], jnp.einsum("be,bke->bk", zs, zn), -2.0)
            top = jax.lax.top_k(sim, a.self_mine)[1]
            pick = lambda x: jnp.take_along_axis(x, top[..., None] if x.ndim == 3 else top, 1)
            b = {**b, "neg_plm": pick(b["neg_plm"]), "neg_fp": pick(b["neg_fp"]), "neg_ok": pick(b["neg_ok"])}
        else:
            b = sample_batch(store, kb, a.batch, a.local_frac, a.hard_k, a.hard_ppm, a.hard_pool, a.neg_all)
        b = jax.lax.with_sharding_constraint(b, shard)
        (loss, info), g = jax.value_and_grad(lambda p: losses(model, p, b, kd, **lw), has_aux=True)(params)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, info

    score = make_scorer(model)
    print(f"steps {total} ({steps_per_epoch}/epoch) global batch {a.batch}", flush=True)
    best, log, t_log = -1, [], time.time()
    for s in range(1, total + 1):
        key, k = jax.random.split(key)
        params, opt_state, info = step(params, opt_state, store, k)
        if s == 1:
            jax.block_until_ready(params)
            print(f"first step (compile) {time.time()-t0:.0f}s since start", flush=True)
            t_log = time.time()
        if s % a.log_every == 0:
            info = jax.device_get(info)
            dt = time.time() - t_log
            print(f"step {s}/{total} {a.log_every/dt:.1f} it/s {a.log_every*a.batch/dt:.0f} spec/s " +
                  " ".join(f"{k}={float(v):.3f}" for k, v in info.items()), flush=True)
            t_log = time.time()
        if s % a.eval_every == 0 or s == total:
            ev = evaluate(score, params, st, hold, a.ppm, a.eval_mols)
            print(f"EVAL step {s}: " + json.dumps(ev), flush=True)
            log.append({"step": s, **ev})
            if ev["ens"] > best:
                best = ev["ens"]
                open(f"{a.out}/best.msgpack", "wb").write(serialization.to_bytes(jax.device_get(params)))
            t_log = time.time()
    params = serialization.from_bytes(jax.device_get(params), open(f"{a.out}/best.msgpack", "rb").read())
    ev = evaluate(score, params, st, hold, a.ppm)
    print("FINAL holdout (all):", json.dumps(ev), flush=True)
    json.dump({"args": vars(a), "log": log, "final": ev}, open(f"{a.out}/log.json", "w"), indent=1)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"))
    if a.results:
        with open(a.results, "a") as f:
            f.write(json.dumps({"tag": a.tag, "framework": "jax", "device": devs[0].device_kind, "n_dev": len(devs),
                                "time_s": round(time.time() - t0), "steps": total, "args": vars(a), "final": ev}) + "\n")


if __name__ == "__main__":
    main()
