"""Dry run of the version-2 exploration registry on an E006-like exploration, inside a throwaway sandbox (fake cells,
library, molecule table, weights rows, per-member eval files, arm-level audit files and a git repository):
cells freeze -> explore check -> a FragWM draft refused on the reserved ID -> explore freeze -> report refused before the
commit -> git commit -> explore report. Nothing outside the sandbox is written; the only repository files read are
scripts/meta/registry.py and registry/reserved_ids.json.

  nice -n 19 env CUDA_VISIBLE_DEVICES="" .venv/bin/python tests/registry_dryrun.py [--dir DIR]
"""
from __future__ import annotations

import argparse, contextlib, importlib.util, json, os, shutil, subprocess, tempfile, uuid

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESERVED = os.path.join(REPO, "registry", "reserved_ids.json")


def load_registry():
    """A fresh copy of scripts/meta/registry.py (empty caches)."""
    spec = importlib.util.spec_from_file_location(f"registry_{uuid.uuid4().hex[:8]}", os.path.join(REPO, "scripts", "meta", "registry.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(root, *args):
    r = subprocess.run(["git", "-c", "user.name=sandbox", "-c", "user.email=sandbox@example.invalid", "-c", "commit.gpgsign=false",
                        *args], cwd=root, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr}")
    return r.stdout


def git_init(root):
    git(root, "init", "-q")
    git(root, "commit", "-q", "--allow-empty", "-m", "sandbox root")


def commit_all(root, msg):
    git(root, "add", "-A", "registry")
    git(root, "commit", "-q", "-m", msg)


@contextlib.contextmanager
def inside(root):
    """cwd = sandbox, and git never looks above it."""
    old, ceil = os.getcwd(), os.environ.get("GIT_CEILING_DIRECTORIES")
    os.environ["GIT_CEILING_DIRECTORIES"] = os.path.dirname(os.path.abspath(root))
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old)
        if ceil is None:
            os.environ.pop("GIT_CEILING_DIRECTORIES", None)
        else:
            os.environ["GIT_CEILING_DIRECTORIES"] = ceil


def key(prefix, i):
    return f"{prefix}{i:0{14 - len(prefix)}d}"


def write_cell(root, name, keys, extra=None):
    d = os.path.join(root, "work", "cells", name)
    os.makedirs(d, exist_ok=True)
    n = len(keys)
    m = pd.DataFrame({"mol_id": [f"{name}_{i:04d}" for i in range(n)], "ik14_lib": keys, "ik14_raw": keys, "skey": keys,
                      "doc": np.linspace(1, 9, n), "exact_mass": np.linspace(160, 900, n)})
    for c, v in (extra or {}).items():
        m[c] = v
    m.to_parquet(f"{d}/molecules.parquet", index=False)
    pd.DataFrame({"mol_id": m.mol_id, "spec_idx": range(n)}).to_parquet(f"{d}/queries.parquet", index=False)
    pd.DataFrame({"sid": range(3 * n)}).to_parquet(f"{d}/cands.parquet", index=False)
    return m


def weights_row(name, exclude="work/cells/exclude_all_ik.txt", init=None, kind="spectrum_model"):
    return {"name": name, "kind": kind, "path": f"/sandbox/{name}.pt", "sha256": "0" * 64, "init_encoder": init,
            "hidden": {"val_dir": os.path.dirname(exclude) if exclude else "", "exclude_all": exclude, "always": ["holdout", "enveda-np-examples"],
                       "emb": "emb_t", "emb_exclusion": [exclude] if exclude else []}}


def build_world(root, n=(("np_nit", 60), ("np_c2", 80), ("np_tims", 40)), weights=()):
    """Fake library (train structures, hidden-cell structures, np-examples, holdout), library meta table, molecule table,
    hidden and absent cells, and weights rows. np_nit structures are absent from the library; three of them are database
    negatives of the molecule table (emb_db_exposure 3)."""
    n = dict(n)
    os.makedirs(os.path.join(root, "registry"), exist_ok=True)
    for d in ("lib", "meta/v1", "emb_t", "cells/eval"):
        os.makedirs(os.path.join(root, "work", d), exist_ok=True)
    train = [key("TRAIN", i) for i in range(200)]
    c2 = [key("CTWO", i) for i in range(n["np_c2"])]
    tims = [key("TIMS", i) for i in range(n["np_tims"])]
    nit = [key("NIT", i) for i in range(n["np_nit"])]
    hold = [key("HOLD", i) for i in range(5)]
    spec = pd.DataFrame({"ingest_lib": ["gnps"] * (len(train) + len(c2) + len(hold)) + ["enveda-np-examples"] * len(tims),
                         "inchikey14": train + c2 + hold + tims})
    spec.to_parquet(f"{root}/work/lib/spectra.parquet", index=False)
    lib = sorted(set(spec.inchikey14))
    pd.DataFrame({"ik14_lib": lib, "ik14_raw": lib, "skey": lib}).to_parquet(f"{root}/work/meta/v1/molecules.parquet", index=False)
    pd.DataFrame({"inchikey14": hold}).to_parquet(f"{root}/holdout.parquet", index=False)
    emb = pd.DataFrame({"inchikey14": train + nit[:3] + [key("PCDB", i) for i in range(50)] + nit[3:5],
                        "source": ["train"] * len(train) + ["db"] * 53 + ["dist"] * 2})
    emb.to_parquet(f"{root}/work/emb_t/mol_table.parquet", index=False)
    open(f"{root}/work/cells/exclude_all_ik.txt", "w").write("\n".join(sorted(c2 + tims)) + "\n")
    cells = {"np_c2": write_cell(root, "np_c2", c2), "np_tims": write_cell(root, "np_tims", tims), "np_nit": write_cell(root, "np_nit", nit)}
    for c in ("np_c2", "np_tims"):
        open(f"{root}/work/cells/{c}/exclude_all_ik.txt", "w").write("\n".join(sorted(cells[c].ik14_lib)) + "\n")
    with open(f"{root}/registry/weights.jsonl", "w") as f:
        for w in weights:
            f.write(json.dumps(w if isinstance(w, dict) else weights_row(w)) + "\n")
    shutil.copy(RESERVED, f"{root}/registry/reserved_ids.json")
    return cells


# --------------------------------------------------------------------------------------------- the E006-like run

SEEDS = {"A0": ["w4base_t4s0", "w4base_t4s1", "w4base_t4s2"], "A1": ["w4mnpoe_t4s0", "w4mnpoe_t4s1"],
         "A2": ["w4mnt_t4s0", "w4mnt_t4s1"]}
ARMS = {**SEEDS, "A0r": [f"{w}~mir" for w in SEEDS["A0"]], "A0q": [f"{w}~bankq" for w in SEEDS["A0"]]}
REFERENCE = {"A0_3060": ["w4base_s0"]}
EFFECT = {"A0": 0.0, "A1": 0.025, "A2": 0.02, "A0r": 0.02, "A0q": 0.01, "A0_3060": -0.02}   # log-rank shift: + is better
LIFT = {"A0": 0.27, "A1": 0.03, "A2": 0.05, "A0r": 0.004, "A0q": 0.06, "A0_3060": 0.29}
UNFINISHED = {("w4mnt_t4s1", "np_tims")}   # A2's second seed never reached np_tims: A2 is short of min_seeds there


def ranks_to_rr(r):
    r = np.maximum(1, np.round(r))
    return np.where(r <= 25, 1 / r, 0.0)


def write_eval(root, cells, write, seed=7):
    """Per-member eval files (mol_id + metric columns, w_mass on np_nit) and arm-level audit files (lift_fe with 2000
    bootstrap replicates on np_c2 / np_tims, lift_fe_hidden / lift_fe_sf on the np_pool pseudo-cell) of the arms in
    `write`. Every call draws the same numbers, so arms written in different phases stay paired."""
    rng = np.random.default_rng(seed)
    E = f"{root}/work/cells/eval"
    for cell, m in cells.items():
        k = len(m)
        hard = rng.normal(1.6, 1.0, k)                       # per-molecule log-rank of the truth under A0
        in_p25 = rng.random(k) > 0.1                          # truth outside the p25 list -> p25 undefined
        has_train = rng.random(k) > 0.25                      # no train decoy -> mrrT undefined
        w_mass = rng.gamma(2.0, 0.5, k)
        arms = {**ARMS, **({} if cell == "np_nit" else REFERENCE)}
        for arm, members in arms.items():
            for j, mem in enumerate(members):
                if (mem, cell) in UNFINISHED:
                    continue
                noise = rng.normal(0, 0.25, k)
                lr = hard - EFFECT[arm] * 4 + noise
                df = pd.DataFrame({"mol_id": m.mol_id, "p25:nf": np.where(in_p25, ranks_to_rr(np.exp(lr) * 0.6), np.nan),
                                   "mass:nf": ranks_to_rr(np.exp(lr + 0.4)),
                                   "mrrT:nf": np.where(has_train, ranks_to_rr(np.exp(lr - 1.0)), np.nan),
                                   "blind:nf": ranks_to_rr(np.exp(hard + 2.0 + rng.normal(0, 0.5, k))),
                                   "docgrad:nf": rng.normal(0.1 if arm in ("A0", "A0_3060") else 0.02, 0.05, k),
                                   "doc": m.doc, "exact_mass": m.exact_mass})
                if cell == "np_nit":
                    df["w_mass"] = w_mass
                if arm in write:
                    df.to_parquet(f"{E}/{mem}__{cell}.parquet", index=False)
    for arm, members in {**ARMS, **REFERENCE}.items():
        done = {c: [mem for mem in members if (mem, c) not in UNFINISHED] for c in ("np_c2", "np_tims")}
        files = {}
        for cell in ("np_c2", "np_tims"):
            stat = lambda est, se: {"est": est, "boot": list(rng.normal(est, se, 2000)), "n": len(cells[cell])}
            files[f"{'+'.join(done[cell])}__{cell}.arm.json"] = {"lift_fe": stat(LIFT[arm] + rng.normal(0, 0.01), 0.03),
                                                                 "lift_fe_np": stat(LIFT[arm] * 1.1, 0.04)}
        pooled = [mem for mem in members if all(mem in done[c] for c in done)]
        files[f"{'+'.join(pooled)}__np_pool.arm.json"] = {
            "lift_fe_hidden": {"est": LIFT[arm] * 1.2, "ci": {"0.95": [LIFT[arm] * 1.2 - 0.09, LIFT[arm] * 1.2 + 0.09]},
                               "n": len(cells["np_c2"]) + len(cells["np_tims"])},
            "lift_fe_sf": {"est": LIFT[arm] * 0.9, "boot": list(rng.normal(LIFT[arm] * 0.9, 0.05, 2000)), "n": 120}}
        for f, content in files.items():
            if arm in write:
                json.dump(content, open(f"{E}/{f}", "w"))


def lift_block(arm):
    out = []
    for cell in ("np_c2", "np_tims"):
        out += [{"arm": arm, "cell": cell, "metric": "lift_fe", "test": "abs_mean_lt", "value": 0.10},
                {"arm": arm, "cell": cell, "metric": "lift_fe", "test": "ci_high_lt", "value": 0.15, "level": 0.95, "sided": "upper"}]
    return out


def skill_block(arm, ref, value, level, sided, test):
    return [{"contrast": [arm, ref], "cell": "np_nit", "metric": "p25:nf", "weights": "w_mass", "test": test, "value": value,
             "level": level, "sided": sided}]


def draft_E006():
    V = {}
    for a in ("A1", "A2"):
        V[f"N1_{a}"] = lift_block(a)
        V[f"N2_{a}"] = [{"arm": a, "cell": "np_pool", "metric": "lift_fe_hidden", "test": "abs_mean_lt", "value": 0.10}]
        V[f"N3_{a}"] = [{"arm": a, "cell": "np_c2", "metric": "lift_fe_np", "test": "mean_lt", "value": 0.15}]
        V[f"S1_{a}"] = skill_block(a, "A0r", -0.03, 0.95, "lower", "ci_low_gt")
        V[f"S2_{a}"] = [{"contrast": [a, "A0"], "cell": c, "metric": "mrrT:nf", "test": "ci_low_gt", "value": -0.03,
                         "level": 0.95, "sided": "lower"} for c in ("np_c2", "np_tims")]
        V[f"skill_lost_{a}_nit"] = skill_block(a, "A0r", 0.0, 0.975, "upper", "ci_high_lt")
        V[f"skill_lost_{a}_mrrT"] = [{"contrast": [a, "A0"], "cell": "np_c2", "metric": "mrrT:nf", "test": "ci_high_lt",
                                      "value": -0.03, "level": 0.975, "sided": "upper"}]
        V[f"skill_lost_{a}"] = {"any_of": [f"skill_lost_{a}_nit", f"skill_lost_{a}_mrrT"]}
        V[f"{a}_admit"] = {"all_of": [f"N1_{a}", f"N2_{a}", f"N3_{a}", f"S1_{a}", f"S2_{a}", {"not": f"skill_lost_{a}"}]}
        V[f"superiority_{a}"] = [{"contrast": [a, "A0"], "cell": "np_nit", "metric": "p25:nf", "test": "ci_low_gt", "value": 0.0,
                                  "level": 0.975}]
        V[f"{a}_adopt"] = {"all_of": [f"{a}_admit", f"superiority_{a}"]}
    V["A0r_noninferior"] = skill_block("A0r", "A0", -0.01, 0.95, "lower", "ci_low_gt")
    V["A0r_ship"] = {"all_of": [{"not": "A1_admit"}, {"not": "A2_admit"}, "A0r_noninferior"]}
    return {
        "id": "E006", "slug": "sr1-neutral-negatives",
        "title": "Membership-neutral hard negatives for the w4 bi-encoder (sandbox dry run, fake data)",
        "kind": "one_axis", "axis": "hard-negative construction only (fake)",
        "question": "Can the membership lift be removed without losing isomer skill where the truth is not a training structure?",
        "hypothesis": "A1 has |lift_fe| < 0.10 on np_c2 and np_tims and is non-inferior to A0r on np_nit p25w.",
        "arms": ARMS, "reference_arms": REFERENCE, "candidate_arms": ["A1", "A2"],
        "cells": ["np_nit", "np_c2", "np_tims"], "pseudo_cells": {"np_pool": ["np_c2", "np_tims"]},
        "arms_with_results_at_freeze": ["A0_3060 (w4base_s0; np_c2 and np_tims, before this exploration)"],
        "rules": {
            "version": 2,
            "primary": {"contrast": ["A1", "A0r"], "cell": "np_nit", "metric": "p25:nf", "weights": "w_mass"},
            "metrics": ["p25:nf", "mass:nf", "mrrT:nf", "blind:nf", "docgrad:nf", "pipe25:nf"],
            "arm_metrics": ["lift_fe", "lift_fe_np", "lift_fe_hidden", "lift_fe_sf"],
            "metric_definitions": {"p25:nf": "RR of the truth in the documentation top-25", "w_mass": "mass weight to np-examples",
                                   "lift_fe": "fixed-effect coefficient on is_train among wrong p25 candidates (audit tool)",
                                   "lift_fe_hidden": "lift_fe of never-seen train structures, np_c2+np_tims pooled"},
            "contrasts": [["A1", "A0"], ["A2", "A0"], ["A1", "A2"], ["A1", "A0r"], ["A2", "A0r"], ["A0r", "A0"], ["A0q", "A0"],
                          ["A0", "A0_3060"]],
            "arm_stats": [{"arm": ["A0", "A1", "A2", "A0r", "A0q", "A0_3060"], "cell": ["np_c2", "np_tims"], "metric": "lift_fe"},
                          {"arm": ["A0", "A1", "A2"], "cell": "np_pool", "metric": ["lift_fe_hidden", "lift_fe_sf"]},
                          {"arm": ["A0", "A1", "A0r"], "cell": "np_nit", "metric": "p25:nf", "weights": "w_mass"}],
            "report_weights": {"np_nit": "w_mass"},
            "bootstrap": {"resamples": 2000, "seed": 0, "paired_by": "mol_id", "level": 0.95},
            "min_seeds": 2,
            "verdict": V,
            "consequence": "A1_admit -> A1 negatives enter the deploy recipe; A0r_ship -> residualisation ships (fake)",
        },
        "state": "draft",
    }


def all_weights():
    return sorted({m.split("~")[0] for v in {**ARMS, **REFERENCE}.values() for m in v})


def main(root=None):
    root = os.path.abspath(root or tempfile.mkdtemp(prefix="registry_dryrun_"))
    os.makedirs(root, exist_ok=True)
    cells = build_world(root, weights=all_weights())
    write_eval(root, cells, write=set(REFERENCE))      # seen before the freeze, and declared
    os.makedirs(f"{root}/drafts", exist_ok=True)
    json.dump(draft_E006(), open(f"{root}/drafts/E006.json", "w"), indent=1)
    frag = {**draft_E006(), "title": "FragWM draft that also used E006"}
    frag.pop("slug")
    json.dump(frag, open(f"{root}/drafts/E006_fragwm.json", "w"), indent=1)
    reg = load_registry()
    reg.HOLDOUT = f"{root}/holdout.parquet"
    with inside(root):
        print(f"== sandbox {root}")
        git_init(root)
        reg.main(["cells", "freeze"])
        commit_all(root, "sandbox: cells, weights, reserved IDs")
        print("== explore check drafts/E006.json")
        reg.main(["explore", "check", "drafts/E006.json"])
        print("== explore freeze drafts/E006_fragwm.json (same ID, no slug)")
        try:
            reg.main(["explore", "freeze", "drafts/E006_fragwm.json"])
        except SystemExit as e:
            print(f"refused: {e}")
        print("== explore freeze drafts/E006.json")
        reg.main(["explore", "freeze", "drafts/E006.json"])
        write_eval(root, cells, write=set(ARMS))            # the training and evaluation jobs run after the freeze
        print("== explore report E006 (not committed yet)")
        try:
            reg.main(["explore", "report", "E006"])
        except SystemExit as e:
            print(f"refused: {e}")
        commit_all(root, "sandbox: freeze E006")
        print("== explore report E006")
        reg.main(["explore", "report", "E006"])
        E = json.load(open("registry/explorations/E006.json"))
    return root, E


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="sandbox directory (default: a new temporary directory)")
    root, E = main(ap.parse_args().dir)
    print(f"== state {E['state']}; verdict {json.dumps(E['verdict'])}")
