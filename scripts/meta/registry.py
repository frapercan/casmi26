"""Evaluation registry, after PROTEA's ADRs D40 (leakage-free protocol), D54 (model-weights registry) and
D55 (evaluated explorations): frozen cells, registered weights, leakage checked by construction, and
pre-registered explorations whose verdicts are computed from rows, never read off by eye.

  registry/cells.json               frozen cell manifests: role, population, known biases, leak rule, counts, sha256 of every file
  registry/weights.jsonl            one row per checkpoint / encoder: kind, sha256, recipe, what its training hid
  registry/explorations/E###.json   hypothesis, arms, cells, rules, frozen with a sha256 of the whole object BEFORE results; report, verdict
  registry/reserved_ids.json        exploration IDs a plan assigned before their draft exists; a draft claims one with its "slug"
  registry/results.jsonl            one row per (weights, cell, metric): mean, paired-bootstrap CI, leak flags, cell sha, commit

  python scripts/meta/registry.py cells freeze | verify
  python scripts/meta/registry.py weights add PATH --name N --kind spectrum_model|jepa_encoder [--encoder E] [--val DIR] [--work DIR]
  python scripts/meta/registry.py leak N
  python scripts/meta/registry.py score N               (rows from work/cells/eval/N__<cell>.parquet)
  python scripts/meta/registry.py explore check FILE | freeze FILE | report ID | archive ID --reason R
  python scripts/meta/registry.py table [--metric mass:nf+0.25doc,mass:nf]

Cell directories can be redirected (sandboxes): CASMI_CELL_DIRS='{"np_nit": "/elsewhere/np_nit"}'.

Leak rules. "hidden" cells (np_tims, np_c2, np_like): their structures are in a training's exclusion lists, so a structure
leaks unless one of its keys is hidden from the weights' spectra (and from its init encoder's) and, for a spectrum model,
excluded from the molecule table it trained against. "absent" cells (np_nit): their structures are simply absent from
training, so a structure leaks if one of its keys (ik14_lib, ik14_raw, skey = official key, ik14_official if present)
belongs to a library structure whose spectra the weights saw (work/lib minus its hidden sets, with raw and official
aliases from work/meta/v1). Presence in the molecule table as a database negative is not a leak: emb_db_exposure.

Exploration rules, version 2 (rules.version = 2; E001-E005 keep version 1 and its report code). Example:
  "arms": {"A0": ["w4base_t4s0", "w4base_t4s1", "w4base_t4s2"], "A1": ["w4mnpoe_t4s0", "w4mnpoe_t4s1"],
           "A0r": ["w4base_t4s0~mir", "w4base_t4s1~mir", "w4base_t4s2~mir"]},
  "reference_arms": {"A0_3060": ["w4base_s0"]},             reported (contrasts, arm_stats), never a candidate or a verdict input
  "candidate_arms": ["A1"], "cells": ["np_nit", "np_c2", "np_tims"],
  "pseudo_cells": {"np_pool": ["np_c2", "np_tims"]},        pooled cells: arm-level metrics only, leaks checked on the constituents
  "arms_with_results_at_freeze": ["A0_3060 (w4base_s0; np_c2, np_tims)"],   declared by the author, kept as written
  "rules": {"version": 2, "metrics": ["p25:nf", "mrrT:nf"], "arm_metrics": ["lift_fe", "lift_fe_hidden"],
            "contrasts": [["A1", "A0"], ["A1", "A0r"], ["A0", "A0_3060"]],
            "arm_stats": [{"arm": ["A0", "A1"], "cell": ["np_c2", "np_tims"], "metric": "lift_fe"}],
            "report_weights": {"np_nit": "w_mass"}, "bootstrap": {"resamples": 4000, "seed": 0, "level": 0.95}, "min_seeds": 2,
            "verdict": {"N1": [{"arm": "A1", "cell": "np_c2", "metric": "lift_fe", "test": "abs_mean_lt", "value": 0.10},
                               {"arm": "A1", "cell": "np_c2", "metric": "lift_fe", "test": "ci_high_lt", "value": 0.15, "sided": "upper"}],
                        "S1": [{"contrast": ["A1", "A0r"], "cell": "np_nit", "metric": "p25:nf", "weights": "w_mass",
                                "test": "ci_low_gt", "value": -0.03, "level": 0.95, "sided": "lower"}],
                        "skill_lost": [{"contrast": ["A1", "A0r"], "cell": "np_nit", "metric": "p25:nf", "weights": "w_mass",
                                        "test": "ci_high_lt", "value": 0, "level": 0.975, "sided": "upper"}],
                        "A1_admit": {"all_of": ["N1", "S1", {"not": "skill_lost"}]}}}
  Members. An arm is a seed set. A member is a registered weights name or "<weights>~<transform>" (a scoring transform
    of those weights, e.g. ~mir residualisation, ~bankq bank normalisation) with eval files of its own; leaks are always
    checked on the weights.
  Per-molecule metrics (rules.metrics) are columns of {EVAL}/{member}__{cell}.parquet (mol_id + columns). An arm's value
    for a molecule is the mean over the members that have it (member counts are reported), and the report gives the
    between-seed SD of every metric: the mean over molecules of the per-molecule SD, and the SD of member means.
  Arm-level metrics (rules.arm_metrics, e.g. lift_fe from the audit tool on seed-averaged scores) are read from
    {EVAL}/{set}__{cell}.arm.json, set = "+".join(the arm's members with results on the cell, in declared order: all of
    them once every member has results; on a pseudo-cell, results on every constituent cell), holding
    {"<metric>": {"est": x, "boot": [replicates], "ci": {"0.95": [lo, hi]}, "n": k}}. Intervals come from "boot" at any
    level, else from a stored two-sided "ci" at the needed level, else the condition is undetermined.
  Statistics. A contrast ["A1", "A0"] is the paired difference A1 - A0 over the molecules both arms have. "weights": "<column>"
    (a per-molecule column, e.g. w_mass) makes it a weighted mean. The bootstrap resamples molecules (in mol_id order) with
    rules.bootstrap.resamples draws from a fresh generator seeded with rules.bootstrap.seed for every statistic, and
    recomputes the (weighted) mean. Reported: every contrast in rules.contrasts on every cell and metric (and weighted on the
    cells of rules.report_weights) and every arm statistic in rules.arm_stats (cross product of arm, cell, metric): estimate,
    two-sided CI at rules.bootstrap.level, n, and up/down molecule counts for contrasts. Missing ones are listed, not fatal.
  Conditions. {"arm": A, ...} tests an arm statistic, {"contrast": [A, B], ...} a contrast; "cell", "metric", "test", "value",
    and optionally "level" (default rules.bootstrap.level), "sided", "weights", "note".
    Tests on the estimate x: mean_gt (x > v), mean_lt (x < v), abs_mean_lt (|x| < v).
    Tests on the interval [lo, hi]: ci_low_gt (lo > v), ci_high_lt (hi < v), abs_ci_lt (max(|lo|, |hi|) < v).
    "sided": "two" (default): level 0.95 -> lo, hi = the 2.5th and 97.5th percentiles of the replicates.
             "lower": level 0.95 -> lo = the 5th percentile, a one-sided 95% lower bound (ci_low_gt only).
             "upper": level 0.95 -> hi = the 95th percentile, a one-sided 95% upper bound (ci_high_lt only).
    A condition is undetermined, never true, when an input is missing: no eval or arm file, no column, no interval at the
    level, or an arm with fewer than rules.min_seeds members with results (that arm is still reported).
  Verdicts. A block is a list of conditions (a conjunction). A composite is {"all_of": [...]}, {"any_of": [...]} or
    {"not": x}, over verdict names or nested composites, without cycles. Values are true, false or "undetermined" with
    Kleene logic: all_of is false once an operand is false, any_of true once an operand is true, otherwise undetermined if
    an operand is; not undetermined is undetermined.
  Freeze refuses invalid rules, an ID that exists (on disk or in git history) or is reserved for another draft, version-1
  rules and unfrozen cells. It keeps the declared arms_with_results_at_freeze, adds arms_with_results_detected (per member,
  derived names included, plus arm-level files) and sets post_hoc if a candidate arm appears in either. frozen_sha256 covers
  the whole frozen object except what report and archive add (POST_FREEZE). Report refuses unless the frozen file is
  committed (HEAD holds it byte-identical and the commit that added it, an ancestor of HEAD, has the same frozen content),
  the hash still matches, and the cells are still the frozen ones; any leak of an arm member voids the exploration (a
  leaking reference arm is flagged LEAKY). Reporting again needs --again --reason; earlier verdicts stay in report_history.
"""
from __future__ import annotations

import argparse, datetime, functools, hashlib, json, math, os, re, subprocess, sys

import numpy as np
import pandas as pd

REG = "registry"
EVAL = "work/cells/eval"
LIB_SPECTRA = "work/lib/spectra.parquet"
LIB_META = "work/meta/v1/molecules.parquet"   # provided, raw and official (skey) keys of every library structure
KEY_COLS = ["ik14_lib", "ik14_raw", "skey", "ik14_official"]   # skey is the official tautomer-canonical key (build_molecules)
CELLS = {
    "np_tims": dict(dir="work/cells/np_tims", role="valid", leak="hidden",
                    simulates="class 2, timsTOF natural products (Enveda's own panel)",
                    population="enveda-np-examples structures in the PubChem tier, one neutral species; their timsTOF spectra "
                               "as queries; excluded from every training (spectra, anchors, negatives) since the start",
                    biases=["far better documented than the typical NP (doc median 13.0): the documentation prior is optimistic",
                            "the prior weight 0.25 was chosen on it on 2026-11-09"]),
    "np_c2": dict(dir="work/cells/np_c2", role="valid", leak="hidden", simulates="class 2, natural products without reference spectra",
                  population="COCONUT/LOTUS/NPAtlas structures with public spectra, 157-1159 Da, mass histogram matched to "
                             "np-examples (n 800, seed 0); all their spectra hidden from training (work/cells registry)",
                  biases=["selected by NP-database membership: NP-membership priors are circular here",
                          "queries are Orbitrap/QTOF library spectra, not timsTOF"]),
    "np_like": dict(dir="work/cells_x/np_like", role="prior_only", leak="hidden",
                    simulates="class 2, NP-like structures, database membership free",
                    population="Ertl NP-score > 1, public spectra, mass-matched (n 800, seed 0), disjoint from the work/cells "
                               "registry; 82% carry an NP-database bit",
                    biases=["its spectra are NOT hidden from any training so far: model results on it are leaky until a model "
                            "is trained with work/cells_x hidden"]),
    "np_nit": dict(dir="work/cells/np_nit", role="decision", leak="absent",
                   simulates="class 2, structures absent from training (half natural products), reference-standard queries",
                   population="MassBank-harmonized records plus external NP reference spectra whose structures are absent from "
                              "train by provided, official and raw key (and skeleton key); high-resolution LC-ESI only (no "
                              "QQ/QQQ/EI/GC); about 480 structures in the test mass range, truth inside the documentation top-25 "
                              "(p25) for 283-287; primary statistics mass-reweighted to the np-examples histogram (w_mass), "
                              "unweighted as sensitivity (docs/model_proposal.md 3.1)",
                   biases=["48% of the p25 truths are Eawag records (cyanopeptides, PFAS; median 704 Da); p25 median mass 507 Da "
                           "against 354 for np-examples: effective size about 165 (75 NP) after mass reweighting",
                           "only 4.1% of the wrong p25 candidates are train structures (56% of lists have none)",
                           "reference standards with a library look (median blind percentile 0.75 on the np_tims proxy); "
                           "Orbitrap/QTOF queries, no timsTOF",
                           "licences CC BY-SA 73%, CC BY 24%, NC 2.7%: evaluation use only",
                           "structures present in the molecule table as database negatives are a stratum (emb_db_exposure), "
                           "not a leak"]),
}
FILES = ["molecules.parquet", "queries.parquet", "cands.parquet", "structs.parquet", "exclude_all_ik.txt", "plm.npy", "fp.npy"]
# what each molecule table hid from anchors and negatives (scripts/build_mol_table.py invocations)
EMB_EXCLUSION = {"emb_v3": ["work/c3data/exclude_all_ik.txt"],
                 "emb_v4": ["work/c3data/exclude_all_ik.txt", "work/cells/exclude_all_ik.txt"]}
HOLDOUT = "/kaggle/input/casmi26-train-v2/casmi26/splits/holdout.parquet"
# written after the freeze (report, archive): everything else of a frozen exploration is covered by frozen_sha256
POST_FREEZE = ("frozen_sha256", "state", "report", "verdict", "report_sha256", "reported_at", "report_commit", "report_history",
               "void_reason", "leaks", "archived_at", "archived_reason", "results_existed")
TESTS = ("mean_gt", "mean_lt", "abs_mean_lt", "ci_low_gt", "ci_high_lt", "abs_ci_lt")
SIDES = {"ci_low_gt": ("two", "lower"), "ci_high_lt": ("two", "upper"), "abs_ci_lt": ("two",)}
COND_KEYS = {"arm", "contrast", "cell", "metric", "weights", "test", "value", "level", "sided", "note"}
MEMBER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(~[A-Za-z0-9][A-Za-z0-9_.-]*)?")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def commit():
    return git("rev-parse", "--short", "HEAD") or ""


def sha_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_set(p):
    return set(open(p).read().split()) if p and os.path.exists(p) else set()


def jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def cell_dir(name):
    o = os.environ.get("CASMI_CELL_DIRS")
    return (json.loads(o) if o else {}).get(name, CELLS[name]["dir"])


def weights_of(member):
    """Registered weights behind an arm member: 'w4base_s0~mir' -> 'w4base_s0'."""
    return member.split("~", 1)[0]


def cell_manifest(name):
    c = CELLS[name]; d = cell_dir(name)
    m = pd.read_parquet(f"{d}/molecules.parquet"); q = pd.read_parquet(f"{d}/queries.parquet", columns=["mol_id"])
    n_c = len(pd.read_parquet(f"{d}/cands.parquet", columns=["sid"]))
    return {"name": name, **c, "dir": d, "n_molecules": len(m), "n_queries": len(q), "n_candidate_rows": n_c,
            "doc_median": float(m.doc.median()), "mass_median": float(m.exact_mass.median()),
            "sha256": {f: sha_file(f"{d}/{f}") for f in FILES if os.path.exists(f"{d}/{f}")}}


def cells_cmd(a):
    path = f"{REG}/cells.json"
    old = json.load(open(path)) if os.path.exists(path) else {}
    if a.action == "freeze":
        for name in CELLS:
            if not os.path.exists(f"{cell_dir(name)}/cands.parquet"):
                continue
            m = cell_manifest(name)
            if name in old and old[name]["sha256"] != m["sha256"] and not a.force:
                sys.exit(f"{name}: files differ from the frozen manifest; a changed cell is a new cell (use a new name)")
            if name not in old:
                m["frozen_at"], m["commit"] = now(), commit()
                old[name] = m
                print(f"froze {name}: {m['n_molecules']} molecules, {m['n_queries']} queries")
        os.makedirs(REG, exist_ok=True)
        json.dump(old, open(path, "w"), indent=1)
    else:
        bad = 0
        for name, m in old.items():
            d = cell_dir(name) if name in CELLS else m["dir"]
            for f, h in m["sha256"].items():
                ok = sha_file(f"{d}/{f}") == h
                bad += not ok
                if not ok:
                    print(f"MISMATCH {name}/{f}")
        print("cells verified" if not bad else f"{bad} files differ")


def weights_cmd(a):
    import torch
    rows = jsonl(f"{REG}/weights.jsonl")
    if any(r["name"] == a.name for r in rows):
        sys.exit(f"{a.name} already registered")
    if not MEMBER.fullmatch(a.name) or "~" in a.name or "__" in a.name:
        sys.exit(f"{a.name}: weights names use letters, digits, '_', '.', '-' (no '~', which marks derived members, no '__')")
    r = {"name": a.name, "kind": a.kind, "path": a.path, "sha256": sha_file(a.path), "size": os.path.getsize(a.path),
         "registered_at": now(), "commit": commit()}
    if a.kind == "spectrum_model":
        args = torch.load(a.path, map_location="cpu", weights_only=False)["args"]
        r["recipe"] = args
        work, val = args["work"], args.get("val", "")
        r["init_encoder"] = a.encoder or (args.get("init_encoder") or None)
    else:
        work, val = a.work, a.val
        r["recipe"] = {"cmd": a.recipe}
    emb = os.path.basename(os.path.realpath(f"{work}/emb"))
    r["hidden"] = {"val_dir": val, "exclude_all": f"{val}/exclude_all_ik.txt" if val else None,
                   "exclude_all_sha256": sha_file(f"{val}/exclude_all_ik.txt") if val and os.path.exists(f"{val}/exclude_all_ik.txt") else None,
                   "always": ["holdout", "enveda-np-examples"], "emb": emb, "emb_exclusion": EMB_EXCLUSION.get(emb)}
    os.makedirs(REG, exist_ok=True)
    open(f"{REG}/weights.jsonl", "a").write(json.dumps(r) + "\n")
    print("registered", a.name, r["sha256"][:12], "hidden:", r["hidden"]["val_dir"], r["hidden"]["emb"])


@functools.lru_cache(maxsize=None)
def lib_keys(spectra):
    """Structures with library spectra, and those of enveda-np-examples."""
    s = pd.read_parquet(spectra, columns=["ingest_lib", "inchikey14"])
    return frozenset(s.inchikey14.unique()), frozenset(s.inchikey14[s.ingest_lib == "enveda-np-examples"].unique())


@functools.lru_cache(maxsize=None)
def always_hidden(spectra, holdout):
    """Hidden from every training: the holdout and the enveda-np-examples structures."""
    return frozenset(set(pd.read_parquet(holdout).inchikey14) | lib_keys(spectra)[1])


@functools.lru_cache(maxsize=None)
def visible_keys(spectra, meta, holdout, exclude_files):
    """Every key of every library structure whose spectra a training saw (cached per hidden set): the library minus the
    always-hidden structures and the training's exclusion lists, plus the raw and official aliases of what is left."""
    hid = set(always_hidden(spectra, holdout))
    for f in exclude_files:
        hid |= read_set(f)
    vis = set(lib_keys(spectra)[0] - hid)
    if os.path.exists(meta):
        M = pd.read_parquet(meta, columns=["ik14_lib", "ik14_raw", "skey"])
        M = M[M.ik14_lib.isin(vis)]
        vis |= set(M.ik14_raw.dropna()) | set(M.skey.dropna())
    return frozenset(vis)


@functools.lru_cache(maxsize=None)
def emb_rows(emb):
    """Keys of a molecule table (work/<emb>/mol_table.parquet) by source: train, db (negatives), dist (never trained)."""
    p = f"work/{emb}/mol_table.parquet"
    if not emb or not os.path.exists(p):
        return None
    t = pd.read_parquet(p, columns=["inchikey14", "source"])
    return {s: frozenset(g.unique()) for s, g in t.groupby("source").inchikey14}


def cell_keys(cname):
    p = f"{cell_dir(cname)}/molecules.parquet"
    m = pd.read_parquet(p)
    cols = [c for c in KEY_COLS if c in m.columns]
    if not cols:
        sys.exit(f"{p} has none of the key columns {KEY_COLS}")
    return [{k for k in row if isinstance(k, str) and k} for row in zip(*(m[c] for c in cols))]


def leak_report(name, verbose=True, cells=None):
    """Leak flags of the weights behind `name` (a derived member is checked on its weights) on every built cell
    (or on `cells`), by each cell's leak rule (module docstring)."""
    W = {r["name"]: r for r in jsonl(f"{REG}/weights.jsonl")}
    r = W[weights_of(name)]
    chain = [r] + ([W[r["init_encoder"]]] if r.get("init_encoder") in W else [])
    out = {}
    for cname in CELLS if cells is None else cells:
        if not os.path.exists(f"{cell_dir(cname)}/molecules.parquet"):
            continue
        keys = cell_keys(cname)
        res = {}
        if CELLS[cname].get("leak", "hidden") == "absent":
            for w in chain:
                vis = visible_keys(LIB_SPECTRA, LIB_META, HOLDOUT, tuple(filter(None, [w["hidden"].get("exclude_all")])))
                res[f"spectra_leak:{w['name']}"] = int(sum(bool(k & vis) for k in keys))
            rows = emb_rows(r["hidden"].get("emb"))
            res["emb_db_exposure"] = None if rows is None else int(sum(bool(k & rows.get("db", frozenset())) for k in keys))
        else:
            always = always_hidden(LIB_SPECTRA, HOLDOUT)
            for w in chain:
                hid = always | read_set(w["hidden"].get("exclude_all"))
                res[f"spectra_leak:{w['name']}"] = int(sum(not (k & hid) for k in keys))
            emb_ex = set(always)
            for f in r["hidden"].get("emb_exclusion") or []:
                emb_ex |= read_set(f)
            res["anchor_leak"] = int(sum(not (k & emb_ex) for k in keys)) if r["kind"] == "spectrum_model" else 0
        res["leak_free"] = all(v == 0 for k, v in res.items() if k.endswith("leak") or k.startswith("spectra_leak"))
        out[cname] = res
        if verbose:
            print(f"{name} on {cname}: " + ", ".join(f"{k}={v}" for k, v in res.items()))
    return out


def boot(x, rng, n=4000, level=0.95):
    bs = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(bs, round(50 * (1 - level), 9))), float(np.percentile(bs, round(50 * (1 + level), 9)))


def score_cmd(a):
    leaks = leak_report(a.name, verbose=False)
    cells = json.load(open(f"{REG}/cells.json"))
    rng = np.random.default_rng(0)
    n = 0
    with open(f"{REG}/results.jsonl", "a") as out:
        for cname in cells:
            p = f"{EVAL}/{a.name}__{cname}.parquet"
            if not os.path.exists(p):
                continue
            R = pd.read_parquet(p)
            for col in [c for c in R.columns if ":" in c]:
                x = R[col].dropna().to_numpy(float)
                lo, hi = boot(x, rng, 2000)
                out.write(json.dumps({"weights": a.name, "cell": cname, "cell_sha": cells[cname]["sha256"]["cands.parquet"][:16],
                                      "metric": col, "mean": float(x.mean()), "ci95": [lo, hi], "n": len(x),
                                      "leak_free": leaks[cname]["leak_free"], "time": now(), "commit": commit()}) + "\n")
                n += 1
    print(f"{n} rows for {a.name}; leak-free cells: {[c for c, v in leaks.items() if v['leak_free']]}")


# ----------------------------------------------------------------------------------------------- explorations

def rules_version(E):
    R = E.get("rules")
    return R.get("version", 1) if isinstance(R, dict) else 1


def frozen_sha(E):
    return sha_obj({k: v for k, v in E.items() if k not in POST_FREEZE})


def all_arms(E):
    """Arm -> member list, reference arms included (version 1: arm -> its eval name)."""
    A = {**E.get("arms", {}), **E.get("reference_arms", {})}
    return {k: (list(v) if isinstance(v, list) else [v]) for k, v in A.items()}


def real_cells(E):
    """The exploration's cells plus the constituents of its pseudo-cells."""
    out = list(E.get("cells", []))
    for cs in E.get("pseudo_cells", {}).values():
        out += [c for c in cs if c not in out]
    return out


def declared_arms(entry, E):
    """Arms a declared arms_with_results_at_freeze entry names: its leading word, an arm or a member ('A0 (w4base_s0
    only; np_c2)', 'w4base_s0 on np_c2', or {"arm": "A0", ...})."""
    s = entry.get("arm") if isinstance(entry, dict) else entry
    m = re.match(r"\s*([A-Za-z0-9_.~-]+)", s) if isinstance(s, str) else None
    if not m:
        return set()
    A = all_arms(E)
    return {m.group(1)} if m.group(1) in A else {k for k, v in A.items() if m.group(1) in v}


def expand(entry):
    """Cross product of an arm_stats entry's arm, cell and metric (each a name or a list)."""
    f = lambda v: v if isinstance(v, list) else [v]
    return [(a, c, m) for a in f(entry["arm"]) for c in f(entry["cell"]) for m in f(entry["metric"])]


def composite_refs(x, where, names, err):
    """Verdict names a composite refers to (through nested composites); appends structural errors."""
    if isinstance(x, str):
        if x not in names:
            err.append(f"{where}: unknown verdict name {x!r}")
        return {x}
    ops = [k for k in x if k in ("all_of", "any_of", "not")] if isinstance(x, dict) else []
    if len(ops) != 1 or set(x) - {ops[0], "note"}:
        err.append(f"{where}: a composite is {{'all_of': [...]}}, {{'any_of': [...]}} or {{'not': name}}, got {x!r}")
        return set()
    items = [x["not"]] if ops[0] == "not" else x[ops[0]]
    if not isinstance(items, list) or not items:
        err.append(f"{where}: {ops[0]} needs a non-empty list")
        return set()
    return set().union(*(composite_refs(i, where, names, err) for i in items))


def find_cycle(graph):
    state = {}

    def visit(n, path):
        if state.get(n) == 1:
            return path[path.index(n):] + [n]
        if state.get(n) == 2 or n not in graph:
            return None
        state[n] = 1
        for m in sorted(graph[n]):
            c = visit(m, path + [n])
            if c:
                return c
        state[n] = 2
    for n in graph:
        c = visit(n, [])
        if c:
            return c
    return None


def isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def isname(v):
    return isinstance(v, str) and bool(v)


def validate(E):
    """Errors that make a version-2 draft unfreezable (an empty list: valid)."""
    err = []
    R = E.get("rules")
    if not isinstance(R, dict):
        return ["rules must be an object"]
    arms = {}
    for key, need in (("arms", True), ("reference_arms", False)):
        A = E.get(key, {} if not need else None)
        if not isinstance(A, dict) or (need and not A):
            err.append(f"{key} must map arm names to lists of member names")
            continue
        for k, v in A.items():
            if k in arms:
                err.append(f"{k} is both an arm and a reference arm")
            if not isinstance(v, list) or not v or not all(isinstance(m, str) for m in v):
                err.append(f"{key}.{k}: members must be a non-empty list of names ('<weights>' or '<weights>~<transform>'), got {v!r}")
                continue
            if len(set(v)) != len(v):
                err.append(f"{key}.{k}: duplicated members")
            err += [f"{key}.{k}: bad member name {m!r} (letters, digits, '_', '.', '-', one '~'; no '__')"
                    for m in v if not MEMBER.fullmatch(m) or "__" in m]
            arms[k] = (v, key == "reference_arms")
    main = {k for k, (_, ref) in arms.items() if not ref}
    if "arm_weights" in E:
        err.append("arm_weights is version-1 bookkeeping: a version-2 member names its weights ('<weights>' or '<weights>~<transform>')")
    cand = E.get("candidate_arms", [])
    if not isinstance(cand, list) or any(not isname(k) or k not in main for k in cand):
        err.append(f"candidate_arms {cand!r} must list arms (reference arms are never candidates)")
    cells = E.get("cells")
    if not isinstance(cells, list) or not cells or not all(isinstance(c, str) for c in cells):
        err.append("cells must be a non-empty list of cell names")
        cells = []
    err += [f"unknown cell {c!r} (cells: {sorted(CELLS)})" for c in cells if c not in CELLS]
    pseudo = E.get("pseudo_cells", {})
    if not isinstance(pseudo, dict):
        err.append("pseudo_cells must map a name to a list of cells")
        pseudo = {}
    for p, cs in pseudo.items():
        if p in CELLS or p in cells:
            err.append(f"pseudo-cell {p!r} shadows a cell")
        if not isinstance(cs, list) or not cs or not all(isname(c) and c in CELLS for c in cs) or len(set(cs)) != len(cs):
            err.append(f"pseudo-cell {p!r}: constituents must be distinct registered cells, got {cs!r}")
    metrics, arm_metrics = R.get("metrics", []), R.get("arm_metrics", [])
    for k, v in (("metrics", metrics), ("arm_metrics", arm_metrics)):
        if not isinstance(v, list) or not all(isinstance(m, str) for m in v):
            err.append(f"rules.{k} must be a list of names")
    if set(metrics) & set(arm_metrics):
        err.append(f"{sorted(set(metrics) & set(arm_metrics))} are both per-molecule and arm-level metrics")

    def cell_metric(where, cell, metric, contrast=False, weights=False):
        if not isname(cell) or not isname(metric):
            return err.append(f"{where}: cell and metric are names, got {cell!r} and {metric!r}")
        if cell in pseudo:
            if metric not in arm_metrics:
                err.append(f"{where}: pseudo-cell {cell} holds arm-level metrics only {arm_metrics}, not {metric!r}")
        elif cell not in cells:
            err.append(f"{where}: cell {cell!r} is neither one of the exploration's cells {cells} nor a declared pseudo-cell {sorted(pseudo)}")
        if metric in arm_metrics:
            if contrast:
                err.append(f"{where}: {metric} is an arm-level metric: test it per arm")
            if weights:
                err.append(f"{where}: weights apply to per-molecule metrics, not to the arm-level {metric}")
        elif metric not in metrics:
            err.append(f"{where}: unknown metric {metric!r} (rules.metrics {metrics}, rules.arm_metrics {arm_metrics})")

    def condition(c, where):
        if not isinstance(c, dict):
            return err.append(f"{where}: a condition is an object, got {c!r}")
        if set(c) - COND_KEYS:
            err.append(f"{where}: unknown keys {sorted(set(c) - COND_KEYS)}")
        if ("arm" in c) == ("contrast" in c):
            err.append(f"{where}: give exactly one of 'arm' or 'contrast'")
        named = [c["arm"]] if "arm" in c else c.get("contrast")
        if not isinstance(named, list) or len(named) != (1 if "arm" in c else 2) or not all(isname(k) for k in named):
            err.append(f"{where}: 'arm' is an arm name and 'contrast' is [arm, arm]")
        else:
            for k in named:
                if k not in arms:
                    err.append(f"{where}: unknown arm {k!r}")
                elif arms[k][1]:
                    err.append(f"{where}: {k} is a reference arm (reported only, never in a verdict)")
            if "contrast" in c and len(set(named)) < 2:
                err.append(f"{where}: a contrast needs two different arms")
        cell_metric(where, c.get("cell"), c.get("metric"), "contrast" in c, "weights" in c)
        if "weights" in c and not (isinstance(c["weights"], str) and c["weights"]):
            err.append(f"{where}: weights names a per-molecule column")
        t, v = c.get("test"), c.get("value")
        if not isname(t) or t not in TESTS:
            old = re.fullmatch(r"(mean|abs_mean|ci_low|ci_high|abs_ci)([<>])(-?[0-9.]+)", t) if isname(t) else None
            hint = f'; version 2 writes "test": "{old[1]}_{"gt" if old[2] == ">" else "lt"}", "value": {old[3]}' if old else ""
            err.append(f"{where}: unknown test {t!r} (one of {', '.join(TESTS)}){hint}")
            t = None
        if not isnum(v):
            err.append(f"{where}: value must be a number, got {v!r}")
        elif t in ("abs_mean_lt", "abs_ci_lt") and v <= 0:
            err.append(f"{where}: {t} needs a positive value")
        if t in TESTS[:3] and ("level" in c or "sided" in c):
            err.append(f"{where}: {t} tests the estimate; level and sided do not apply")
        if "level" in c and not (isnum(c["level"]) and 0.5 < c["level"] < 1):
            err.append(f"{where}: level must be in (0.5, 1)")
        s = c.get("sided", "two")
        if s not in ("two", "lower", "upper"):
            err.append(f"{where}: sided is two, lower or upper, got {s!r}")
        elif t in SIDES and s not in SIDES[t]:
            err.append(f"{where}: {t} cannot use a {s} bound (allowed: {', '.join(SIDES[t])})")

    for k in ("contrasts", "arm_stats"):
        if not isinstance(R.get(k, []), list):
            err.append(f"rules.{k} must be a list")
            R = {**R, k: []}
    for i, p in enumerate(R.get("contrasts", [])):
        if not (isinstance(p, list) and len(p) == 2 and all(isname(k) and k in arms for k in p) and p[0] != p[1]):
            err.append(f"rules.contrasts[{i}] must be [arm, arm] with two different arms (reference arms allowed), got {p!r}")
    for i, s in enumerate(R.get("arm_stats", [])):
        where = f"rules.arm_stats[{i}]"
        if not isinstance(s, dict) or set(s) - {"arm", "cell", "metric", "weights", "note"} or not {"arm", "cell", "metric"} <= set(s):
            err.append(f"{where}: an entry is {{'arm': A or [..], 'cell': C or [..], 'metric': M or [..]}}, got {s!r}")
            continue
        for a_, c_, m_ in expand(s):
            if not isname(a_) or a_ not in arms:
                err.append(f"{where}: unknown arm {a_!r}")
            cell_metric(where, c_, m_, weights="weights" in s)
    b = R.get("bootstrap")
    if not isinstance(b, dict):
        err.append("rules.bootstrap must be {resamples, seed, level}")
    else:
        lv = b.get("level", b.get("ci", 0.95))
        if not (isinstance(b.get("resamples"), int) and b["resamples"] > 0):
            err.append("rules.bootstrap.resamples must be a positive integer")
        if not (isinstance(b.get("seed"), int) and b["seed"] >= 0):
            err.append("rules.bootstrap.seed must be a non-negative integer")
        if not (isnum(lv) and 0.5 < lv < 1):
            err.append("rules.bootstrap.level must be in (0.5, 1)")
        if b.get("paired_by", "mol_id") != "mol_id":
            err.append("rules.bootstrap.paired_by: molecules are paired by mol_id")
    ms = R.get("min_seeds", 1)
    if not (isinstance(ms, int) and not isinstance(ms, bool) and ms >= 1):
        err.append("rules.min_seeds must be a positive integer")
    rw = R.get("report_weights", {})
    if not isinstance(rw, dict) or any(c not in cells or not isinstance(w, str) for c, w in rw.items()):
        err.append(f"rules.report_weights maps cells of the exploration to a weight column, got {rw!r}")
    P = R.get("primary")
    if P is not None:
        if not isinstance(P, dict) or ("arm" in P) == ("contrast" in P):
            err.append("rules.primary is {cell, metric, contrast | arm[, weights]}")
        else:
            named = [P["arm"]] if "arm" in P else P["contrast"]
            if not isinstance(named, list) or any(not isname(k) or k not in main for k in named):
                err.append(f"rules.primary names unknown arms {named!r}")
            cell_metric("rules.primary", P.get("cell"), P.get("metric"), "contrast" in P, "weights" in P)
    V = R.get("verdict")
    if not isinstance(V, dict) or not V:
        err.append("rules.verdict must map names to condition lists or composites")
        V = {}
    graph = {}
    for name, blk in V.items():
        where = f"verdict {name}"
        if isinstance(blk, list):
            if not blk:
                err.append(f"{where}: empty block")
            for i, c in enumerate(blk):
                condition(c, f"{where}[{i}]")
        elif isinstance(blk, dict) and "test" in blk:
            err.append(f"{where}: a single condition still goes inside a list")
        elif isinstance(blk, dict):
            graph[name] = composite_refs(blk, where, set(V), err)
        else:
            err.append(f"{where}: is a {type(blk).__name__}; a block is a list of conditions or a composite {{all_of|any_of|not}}")
    cyc = find_cycle(graph)
    if cyc:
        err.append("verdict composites form a cycle: " + " -> ".join(cyc))
    D = E.get("arms_with_results_at_freeze")
    if not isinstance(D, list):
        err.append("declare arms_with_results_at_freeze: the arms whose results you have seen (an empty list if none)")
    else:
        err += [f"arms_with_results_at_freeze entry {x!r} names no arm or member" for x in D if not declared_arms(x, E)]
    if "slug" in E and not isinstance(E["slug"], str):
        err.append("slug must be a string")
    late = [k for k in POST_FREEZE if k in E and k != "state"]
    if late:
        err.append(f"a draft carries no report or archive fields: {late}")
    return err


def reserved_ids():
    p = f"{REG}/reserved_ids.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def next_free_id():
    used = set(reserved_ids())
    for d in ("explorations", "proposals"):
        if os.path.isdir(f"{REG}/{d}"):
            used |= {f[:-5] for f in os.listdir(f"{REG}/{d}") if re.fullmatch(r"E\d{3}[a-z]?\.json", f)}
    return f"E{max([int(u[1:]) for u in used if re.fullmatch(r'E\d{3}', u)] + [0]) + 1:03d}"


def id_errors(E):
    eid = E["id"]
    path = f"{REG}/explorations/{eid}.json"
    if os.path.exists(path):
        return [f"{eid} is already frozen ({path}); a changed exploration is a new exploration with a new ID ({next_free_id()})"]
    if git("log", "--all", "--format=%h", "-1", "--", path):
        return [f"{eid} existed in git history ({path}); IDs are never reused (next free: {next_free_id()})"]
    res = reserved_ids().get(eid)
    if res and E.get("slug") != res.get("slug"):
        return [f"{eid} is reserved for {res.get('for', '?')!r}; this draft ({E.get('title', '')[:70]!r}, slug {E.get('slug')!r}) "
                f"does not claim it (a draft that IS that exploration carries \"slug\": {res.get('slug')!r}). {res.get('note', '')} "
                f"Otherwise renumber the draft: the next free ID is {next_free_id()}."]
    return []


def detect_results(E):
    """Eval files that exist for every member (derived names included) on the exploration's cells, and arm-level files
    of any member set."""
    files = set(os.listdir(EVAL)) if os.path.isdir(EVAL) else set()
    cells = real_cells(E)
    every = cells + list(E.get("pseudo_cells", {}))
    out = {}
    for arm, members in all_arms(E).items():
        got = {m: [c for c in cells if f"{m}__{c}.parquet" in files] for m in members}
        got = {m: cs for m, cs in got.items() if cs}
        sets = sorted(f for f in files for c in every if f.endswith(f"__{c}.arm.json")
                      and set(f[:-len(f"__{c}.arm.json")].split("+")) & set(members))
        if got or sets:
            out[arm] = {"members": got, **({"arm_files": sets} if sets else {})}
    return out


def explore_freeze(path, write=True):
    E = json.load(open(path))
    eid = E.get("id")
    if not isinstance(eid, str) or not re.fullmatch(r"E\d{3}[a-z]?", eid):
        sys.exit(f"{path}: id must look like E006 (or E006b), got {eid!r}")
    if rules_version(E) != 2:
        sys.exit(f"{eid}: rules.version must be 2 (version-1 rules are readable for E001-E005 but no longer frozen; "
                 f"see the module docstring)")
    err = id_errors(E) + validate(E)
    C = json.load(open(f"{REG}/cells.json")) if os.path.exists(f"{REG}/cells.json") else {}
    if not err:
        unfrozen = [c for c in real_cells(E) if c not in C]
        if unfrozen:
            err.append(f"cells {unfrozen} are not frozen in {REG}/cells.json: freeze and commit them first")
    if err:
        sys.exit(f"{eid} cannot be frozen:\n  - " + "\n  - ".join(err))
    detected = detect_results(E)
    declared = set().union(*(declared_arms(x, E) for x in E["arms_with_results_at_freeze"]))
    seen = declared | set(detected)
    post_hoc = bool(E.get("post_hoc")) or any(k in seen for k in E.get("candidate_arms", []))
    E.update(frozen_at=now(), commit=commit(), rules_sha256=sha_obj(E["rules"]), arms_with_results_detected=detected,
             post_hoc=post_hoc, cell_manifests={c: {"sha256": sha_obj(C[c]["sha256"]), "frozen_at": C[c].get("frozen_at")}
                                                for c in real_cells(E)}, state="frozen")
    E["frozen_sha256"] = frozen_sha(E)
    undeclared = sorted(set(detected) - declared)
    print(f"{eid}: rules version 2 valid; declared arms_with_results_at_freeze {E['arms_with_results_at_freeze']} (kept)")
    for arm, d in detected.items():
        print(f"  detected {arm}: " + "; ".join([f"{m} on {cs}" for m, cs in d["members"].items()] + d.get("arm_files", [])))
    if undeclared:
        print(f"  WARNING: results exist for arms not declared: {undeclared} (recorded in arms_with_results_detected)")
    print(f"  post_hoc={post_hoc} (candidate arms {E.get('candidate_arms', [])})")
    if not write:
        return E
    os.makedirs(f"{REG}/explorations", exist_ok=True)
    out = f"{REG}/explorations/{eid}.json"
    json.dump(E, open(out, "w"), indent=1)
    print(f"froze {eid}: frozen_sha256 {E['frozen_sha256'][:12]}, rules {E['rules_sha256'][:12]} -> {out}\n"
          f"  commit it now: report refuses until git holds this file unchanged")
    return E


def require_committed(path):
    """Report reads a frozen file only when git holds it: HEAD has it byte-identical, and the commit that added it (an
    ancestor of HEAD) has the same frozen content."""
    head = git("rev-parse", "--verify", "--quiet", f"HEAD:./{path}")
    if not head:
        sys.exit(f"{path} is not committed: commit the frozen exploration before any report")
    if git("hash-object", path) != head:
        sys.exit(f"{path} differs from its committed version (HEAD): a frozen exploration is reported as committed")
    adds = (git("log", "--no-renames", "--diff-filter=A", "--format=%H", "--", path) or "").split()
    first = json.loads(git("show", f"{adds[-1]}:./{path}") or "{}") if adds else {}
    if frozen_sha(first) != frozen_sha(json.load(open(path))):
        sys.exit(f"{path}: its frozen content differs from the commit that added it ({adds[-1][:9] if adds else '?'}); "
                 f"a changed exploration is a new exploration")


def explore_cmd(a):
    if a.action in ("check", "freeze"):
        explore_freeze(a.target, write=a.action == "freeze")
        return
    path = f"{REG}/explorations/{a.target}.json"
    if not os.path.exists(path):
        sys.exit(f"{a.target}: no frozen exploration at {path}")
    E = json.load(open(path))
    if a.action == "archive":   # never deleted: archived with its reason (and whether any result existed)
        assert a.reason, "--reason is required"
        E.update(state="archived", archived_at=now(), archived_reason=a.reason,
                 results_existed=any(os.path.exists(f"{EVAL}/{w}__{c}.parquet") for v in all_arms(E).values() for w in v
                                     for c in real_cells(E)))
        json.dump(E, open(path, "w"), indent=1); print(f"archived {a.target}: {a.reason}"); return
    if E.get("state") == "archived":
        sys.exit(f"{a.target} is archived: {E.get('archived_reason', '')}")
    require_committed(path)
    v = rules_version(E)
    if v == 1:
        report_v1(E, path)
    elif v == 2:
        report_v2(E, path, again=a.again, reason=a.reason)
    else:
        sys.exit(f"{a.target}: unknown rules version {v!r}")


def report_v1(E, path):
    """Version-1 report (E001-E005), unchanged: single-weights arms, ci_low>0 / mean>-0.01 / ci_high<0 tests."""
    assert sha_obj(E["rules"]) == E["rules_sha256"], "rules changed after freezing"
    rules = E["rules"]
    rng = np.random.default_rng(rules["bootstrap"]["seed"])
    for k, w in E.get("arm_weights", E["arms"]).items():   # arms may be scoring rules over registered weights
        lk = leak_report(w, verbose=False, cells=E["cells"])
        for c in E["cells"]:
            if not lk[c]["leak_free"]:
                E.update(state="void", verdict=f"arm {k} ({w}) leaks on {c}: {lk[c]}")
                json.dump(E, open(path, "w"), indent=1); print(E["verdict"]); return
    def per(w, c):
        return pd.read_parquet(f"{EVAL}/{w}__{c}.parquet").set_index("mol_id")
    rep = {}
    for x, y in rules["contrasts"]:
        for c in E["cells"]:
            A, B = per(E["arms"][x], c), per(E["arms"][y], c)
            for metric in rules["metrics"]:
                d = (A[metric] - B.reindex(A.index)[metric]).dropna().to_numpy(float)
                lo, hi = boot(d, rng, rules["bootstrap"]["resamples"])
                rep[f"{x}-{y}|{c}|{metric}"] = {"mean": float(d.mean()), "ci95": [lo, hi], "n": len(d),
                                                 "up": int((d > 0).sum()), "down": int((d < 0).sum())}
    def test(t):
        r = rep[f"{t['contrast'][0]}-{t['contrast'][1]}|{t['cell']}|{t['metric']}"]
        return {"ci_low>0": r["ci95"][0] > 0, "mean>-0.01": r["mean"] > -0.01, "ci_high<0": r["ci95"][1] < 0}[t["test"]]
    verdict = {name: all(test(t) for t in conds) for name, conds in rules["verdict"].items()}
    E.update(report=rep, verdict=verdict, state="reported", reported_at=now(), report_sha256=sha_obj(rep))
    json.dump(E, open(path, "w"), indent=1)
    for k, v in rep.items():
        print(f"  {k:55s} {v['mean']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  up {v['up']} down {v['down']}")
    print("verdict:", verdict)


def read_eval(p):
    R = pd.read_parquet(p)
    if "mol_id" not in R.columns or R.mol_id.duplicated().any():
        sys.exit(f"{p}: an eval file has one row per mol_id")
    R = R.set_index("mol_id")
    return R[[c for c in R.columns if pd.api.types.is_numeric_dtype(R[c]) or pd.api.types.is_bool_dtype(R[c])]].astype(float)


class SeedSet:
    """One arm on one cell: its members' per-molecule eval rows aligned by mol_id. The arm's value for a molecule is the
    mean over the members that have it."""

    def __init__(self, declared, frames):
        self.members = [m for m in declared if m in frames]
        self.missing = [m for m in declared if m not in frames]
        idx = sorted(set().union(*(f.index for f in frames.values())))
        cols = sorted(set().union(*(f.columns for f in frames.values())))
        self.cols = {c: pd.DataFrame({m: frames[m][c] for m in self.members if c in frames[m].columns}, index=idx) for c in cols}
        self.n_molecules = len(idx)

    def mean(self, col):
        return self.cols[col].mean(axis=1)

    def with_values(self, col, index):
        """Members with at least one value of col among the molecules in index."""
        M = self.cols[col].loc[index]
        return [m for m in M.columns if M[m].notna().any()]

    def counts(self, col, index):
        """How many molecules were averaged over how many members."""
        return {str(k): int(v) for k, v in self.cols[col].loc[index].notna().sum(axis=1).value_counts().sort_index().items()}

    def spread(self, col):
        """Between-seed SD: mean over molecules of the per-molecule SD, and SD of the member means over common molecules."""
        M = self.cols[col]
        if M.shape[1] < 2:
            return None
        sd, common = M.std(axis=1, ddof=1).dropna(), M.dropna()
        return {"members": M.shape[1], "mean_mol_sd": float(sd.mean()) if len(sd) else None, "n_mol_sd": len(sd),
                "sd_member_means": float(common.mean().std(ddof=1)) if len(common) else None, "n_common": len(common)}


def interval(st, level, sided):
    """((lo, hi), None) or (None, why): from bootstrap replicates when the statistic has them, else from a stored two-sided
    interval (a one-sided bound at level L is an end of the two-sided interval at 2L - 1). A one-sided interval leaves the
    other end None."""
    if st.get("reps") is not None:
        r = st["reps"][np.isfinite(st["reps"])]
        if not len(r):
            return None, "no finite bootstrap replicate"
        q = {"two": (50 * (1 - level), 50 * (1 + level)), "lower": (100 * (1 - level), None), "upper": (None, 100 * level)}[sided]
        return tuple(None if p is None else float(np.percentile(r, round(p, 9))) for p in q), None
    two = round(level if sided == "two" else 2 * level - 1, 9)
    got = next((v for k, v in st.get("ci", {}).items() if abs(k - two) < 1e-9), None)
    if got is None:
        need = {"two": f"a two-sided {level:g} interval", "lower": f"a one-sided {level:g} lower bound",
                "upper": f"a one-sided {level:g} upper bound"}[sided]
        return None, f"{st.get('source')}: no bootstrap replicates and no stored {two:g} two-sided interval (needed for {need})"
    lo, hi = float(got[0]), float(got[1])
    return (None if sided == "upper" else lo, None if sided == "lower" else hi), None


def all3(v):
    return False if any(x is False for x in v) else (None if any(x is None for x in v) else True)


def any3(v):
    return True if any(x is True for x in v) else (None if any(x is None for x in v) else False)


def show(v):
    return "undetermined" if v is None else v


def expr(x):
    if isinstance(x, str):
        return x
    op = next(k for k in x if k != "note")
    return f"not {expr(x[op])}" if op == "not" else f"{op}[{', '.join(expr(i) for i in x[op])}]"


class Analysis:
    """What a version-2 report computes from the frozen exploration and the eval files of its members."""

    def __init__(self, E):
        self.E, self.R = E, E["rules"]
        self.arms, self.ref = all_arms(E), set(E.get("reference_arms", {}))
        self.cells, self.pseudo = E["cells"], E.get("pseudo_cells", {})
        self.arm_metrics = set(self.R.get("arm_metrics", []))
        b = self.R["bootstrap"]
        self.B, self.seed, self.level = int(b["resamples"]), int(b["seed"]), float(b.get("level", b.get("ci", 0.95)))
        self.min_seeds = int(self.R.get("min_seeds", 1))
        self.usable, self.leaky, self._per, self._stat = {}, set(), {}, {}

    def check_leaks(self):
        """Leak flags of every member's weights on every real cell. Refuses results of unregistered weights or on cells
        whose molecule table is missing. Returns (leaks, reasons to void): a main-arm leak voids, a reference-arm leak is
        flagged."""
        W = {r["name"]: r for r in jsonl(f"{REG}/weights.jsonl")}
        real = real_cells(self.E)
        built = [c for c in real if os.path.exists(f"{cell_dir(c)}/molecules.parquet")]
        leaks, problems, void = {}, [], []
        for arm, members in self.arms.items():
            for m in members:
                has = [c for c in real if os.path.exists(f"{EVAL}/{m}__{c}.parquet")]
                w = weights_of(m)
                if w not in W:
                    if has:
                        problems.append(f"{m} ({arm}) has results on {has} but {w} is not in {REG}/weights.jsonl: register it "
                                        f"(weights add) so its leaks can be checked")
                    self.usable[m] = False
                    continue
                if set(has) - set(built):
                    problems.append(f"{m} ({arm}) has results on {sorted(set(has) - set(built))}, whose molecule table is "
                                    f"missing: its leaks cannot be checked")
                if w not in leaks:
                    leaks[w] = leak_report(w, verbose=False, cells=built)
                self.usable[m] = True
                for c, r in leaks[w].items():
                    if not r["leak_free"]:
                        if arm in self.ref:
                            self.leaky.add(m)
                        else:
                            void.append(f"arm {arm} member {m} ({w}) leaks on {c}: {r}")
        if problems:
            sys.exit("cannot report:\n  - " + "\n  - ".join(problems))
        return leaks, void

    def per(self, arm, cell):
        if (arm, cell) not in self._per:
            frames = {m: read_eval(f"{EVAL}/{m}__{cell}.parquet") for m in self.arms[arm]
                      if self.usable.get(m) and os.path.exists(f"{EVAL}/{m}__{cell}.parquet")}
            self._per[arm, cell] = SeedSet(self.arms[arm], frames)
        return self._per[arm, cell]

    def with_results(self, arm, cell):
        cs = self.pseudo.get(cell, [cell])
        return [m for m in self.arms[arm] if all(m in self.per(arm, c).members for c in cs)]

    def lacks(self, arm, S, cell, metric):
        if not S.members:
            return f"{arm} has no results on {cell}"
        if metric not in S.cols:
            return f"{arm}: no {metric} column on {cell} ({', '.join(S.members)})"
        return None

    def weight(self, col, index, cell, *sets):
        """Per-molecule weights over index (molecules with a weight), consistent across members and arms."""
        if col is None:
            return None, None
        have = [S for S in sets if col in S.cols]
        if not have:
            return None, f"no {col} column on {cell}"
        w = None
        for S in have:
            M = S.cols[col]
            if ((M.max(axis=1) - M.min(axis=1)) > 1e-9).any():
                return None, f"{col} differs between members on {cell}"
            v = S.mean(col)
            if w is not None and ((w - v).abs() > 1e-9).any():
                return None, f"{col} differs between arms on {cell}"
            w = v if w is None else w.combine_first(v)
        w = w.reindex(index).dropna()
        if (w < 0).any() or w.sum() <= 0:
            return None, f"{col} on {cell} must be non-negative with a positive sum"
        return w, None

    def boot(self, x, w=None):
        """Estimate and bootstrap replicates of the mean (or w-weighted mean) of x over molecules in mol_id order: B
        resamples of the molecules from a fresh generator seeded with rules.bootstrap.seed."""
        idx = np.random.default_rng(self.seed).integers(len(x), size=(self.B, len(x)))
        if w is None:
            return float(x.mean()), x[idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float((w * x).sum() / w.sum()), (w[idx] * x[idx]).sum(axis=1) / w[idx].sum(axis=1)

    def summary(self, d, w, members, counts):
        est, reps = self.boot(d.to_numpy(float), None if w is None else w.to_numpy(float))
        out = dict(est=est, reps=reps, n=len(d), members=members, member_counts=counts)
        if w is not None:
            out["n_eff"] = float(w.sum() ** 2 / (w ** 2).sum())
        return out

    def contrast(self, x, y, cell, metric, weights=None):
        """Paired x - y of the seed-averaged metric over the molecules both arms have (a dict, or why it is missing)."""
        key = ("contrast", x, y, cell, metric, weights)
        if key not in self._stat:
            A, B = self.per(x, cell), self.per(y, cell)
            why = self.lacks(x, A, cell, metric) or self.lacks(y, B, cell, metric)
            if not why:
                d = (A.mean(metric) - B.mean(metric)).dropna().sort_index()
                w, why = self.weight(weights, d.index, cell, A, B)
                d = d if w is None else d.loc[w.index]
                if not why and not len(d):
                    why = f"no molecule of {cell} has {metric} for both {x} and {y}"
            if why:
                self._stat[key] = why
            else:
                st = self.summary(d, w, {x: A.with_values(metric, d.index), y: B.with_values(metric, d.index)},
                                  {x: A.counts(metric, d.index), y: B.counts(metric, d.index)})
                self._stat[key] = dict(st, up=int((d > 0).sum()), down=int((d < 0).sum()))
        return self._stat[key]

    def arm_stat(self, arm, cell, metric, weights=None):
        key = ("arm", arm, cell, metric, weights)
        if key not in self._stat:
            self._stat[key] = self.arm_level(arm, cell, metric) if metric in self.arm_metrics else self.arm_mean(arm, cell, metric, weights)
        return self._stat[key]

    def arm_mean(self, arm, cell, metric, weights):
        S = self.per(arm, cell)
        why = self.lacks(arm, S, cell, metric)
        if why:
            return why
        x = S.mean(metric).dropna().sort_index()
        w, why = self.weight(weights, x.index, cell, S)
        if why:
            return why
        x = x if w is None else x.loc[w.index]
        if not len(x):
            return f"{arm} has no {metric} value on {cell}"
        return self.summary(x, w, {arm: S.with_values(metric, x.index)}, {arm: S.counts(metric, x.index)})

    def arm_level(self, arm, cell, metric):
        """An arm-level statistic from {EVAL}/{set}__{cell}.arm.json, set = members with results joined by '+'."""
        members = self.with_results(arm, cell)
        if not members:
            return f"{arm} has no results on {cell}"
        p = f"{EVAL}/{'+'.join(members)}__{cell}.arm.json"
        if not os.path.exists(p):
            others = sorted(f for f in (os.listdir(EVAL) if os.path.isdir(EVAL) else []) if f.endswith(f"__{cell}.arm.json")
                            and set(f[:-len(f"__{cell}.arm.json")].split("+")) & set(self.arms[arm]))
            return f"missing {p}" + (f" (found {others})" if others else "")
        s = json.load(open(p)).get(metric)
        if s is None:
            return f"{p} has no {metric}"
        bad = (not isinstance(s, dict) or not isnum(s.get("est")) or ("boot" in s and not (isinstance(s["boot"], list) and s["boot"]))
               or not isinstance(s.get("ci", {}), dict))
        if bad:
            return f"{p}: {metric} is not {{'est': x, 'boot': [...], 'ci': {{level: [lo, hi]}}, 'n': k}}"
        return dict(est=float(s["est"]), reps=np.asarray(s["boot"], float) if "boot" in s else None, n=s.get("n"),
                    ci={float(k): v for k, v in s.get("ci", {}).items()}, members={arm: members}, source=os.path.basename(p))

    def condition(self, c):
        """(True | False | None, detail) for one condition."""
        d = {k: v for k, v in c.items() if k != "note"}
        if "arm" in c:
            arms, st = [c["arm"]], self.arm_stat(c["arm"], c["cell"], c["metric"], c.get("weights"))
        else:
            arms, st = list(c["contrast"]), self.contrast(*c["contrast"], c["cell"], c["metric"], c.get("weights"))
        if isinstance(st, str):
            return None, dict(d, result="undetermined", why=st)
        t, v = c["test"], float(c["value"])
        d.update(estimate=st["est"], n=st["n"], members=st["members"])
        if t in TESTS[:3]:
            ok = {"mean_gt": st["est"] > v, "mean_lt": st["est"] < v, "abs_mean_lt": abs(st["est"]) < v}[t]
        else:
            lvl, sided = float(c.get("level", self.level)), c.get("sided", "two")
            b, why = interval(st, lvl, sided)
            if why:
                return None, dict(d, result="undetermined", why=why)
            d.update(bounds=list(b), level=lvl, sided=sided)
            lo, hi = b
            ok = {"ci_low_gt": lambda: lo > v, "ci_high_lt": lambda: hi < v, "abs_ci_lt": lambda: max(abs(lo), abs(hi)) < v}[t]()
        short = [f"{a} has {len(st['members'][a])} member(s) with results, min_seeds {self.min_seeds}"
                 for a in arms if len(st["members"][a]) < self.min_seeds]
        if short:
            return None, dict(d, result="undetermined", why="; ".join(short))
        return bool(ok), dict(d, result=bool(ok))

    def verdicts(self):
        V, out, detail, busy = self.R["verdict"], {}, {}, set()

        def ev(name):
            if name not in out:
                if name in busy:
                    sys.exit(f"verdict composites form a cycle through {name}")
                busy.add(name)
                if isinstance(V[name], list):
                    res = [self.condition(c) for c in V[name]]
                    detail[name] = [r[1] for r in res]
                    out[name] = all3([r[0] for r in res])
                else:
                    out[name] = item(V[name])
                    detail[name] = {"expr": expr(V[name]), "result": show(out[name])}
            return out[name]

        def item(x):
            if isinstance(x, str):
                return ev(x)
            op = next(k for k in x if k != "note")
            if op == "not":
                v = item(x["not"])
                return None if v is None else not v
            return (all3 if op == "all_of" else any3)([item(i) for i in x[op]])

        for name in V:
            ev(name)
        return out, detail

    def report(self):
        rep = {"version": 2, "level": self.level, "resamples": self.B, "seed": self.seed, "min_seeds": self.min_seeds,
               "results": {}, "seed_sd": {}, "contrasts": {}, "arm_stats": {}, "missing": []}
        for arm in self.arms:
            for cell in self.cells:
                S = self.per(arm, cell)
                rep["results"].setdefault(arm, {})[cell] = {"members": S.members, "missing": S.missing, "n_molecules": S.n_molecules}
                sd = {m: S.spread(m) for m in self.R.get("metrics", []) if m in S.cols}
                sd = {m: v for m, v in sd.items() if v}
                if sd:
                    rep["seed_sd"].setdefault(arm, {})[cell] = sd
            for cell in self.pseudo:
                rep["results"].setdefault(arm, {})[cell] = {"members": self.with_results(arm, cell)}
        P = self.R.get("primary")
        if P:
            st = (self.arm_stat(P["arm"], P["cell"], P["metric"], P.get("weights")) if "arm" in P
                  else self.contrast(*P["contrast"], P["cell"], P["metric"], P.get("weights")))
            rep["primary"] = dict(self.entry(st, [P["arm"]] if "arm" in P else P["contrast"], "primary", rep), spec=P)
        rw = self.R.get("report_weights", {})
        for x, y in self.R.get("contrasts", []):
            for cell in self.cells:
                for metric in self.R.get("metrics", []):
                    for w in [None] + ([rw[cell]] if cell in rw else []):
                        key = f"{x}-{y}|{cell}|{metric}" + (f"|w:{w}" if w else "")
                        rep["contrasts"][key] = self.entry(self.contrast(x, y, cell, metric, w), [x, y], key, rep)
        for s in self.R.get("arm_stats", []):
            for arm, cell, metric in expand(s):
                key = f"{arm}|{cell}|{metric}" + (f"|w:{s['weights']}" if s.get("weights") else "")
                rep["arm_stats"][key] = self.entry(self.arm_stat(arm, cell, metric, s.get("weights")), [arm], key, rep)
        return rep

    def entry(self, st, arms, key, rep):
        if isinstance(st, str):
            rep["missing"].append(f"{key}: {st}")
            return {"missing": st}
        b, why = interval(st, self.level, "two")
        e = {"mean": st["est"], "ci": list(b) if b else None, "n": st["n"], "members": st["members"]}
        e.update({k: st[k] for k in ("up", "down", "n_eff", "member_counts", "source") if k in st})
        if why:
            e["ci_missing"] = why
        short = [a for a in arms if len(st["members"].get(a, [])) < self.min_seeds]
        if short:
            e["below_min_seeds"] = short
        if any(m in self.leaky for a in arms for m in st["members"].get(a, [])):
            e["leaky"] = True
        return e


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def stat_line(k, v):
    if "missing" in v:
        return f"  {k:44s} missing: {v['missing']}"
    ci = v["ci"] or [None, None]
    extra = (f"  up {v['up']} down {v['down']}" if "up" in v else "") + (f"  n_eff {v['n_eff']:.1f}" if "n_eff" in v else "")
    flags = (f"  [{', '.join(v['below_min_seeds'])} below min_seeds]" if v.get("below_min_seeds") else "") + ("  LEAKY" if v.get("leaky") else "")
    return f"  {k:44s} {fmt(v['mean'])} [{fmt(ci[0])}, {fmt(ci[1])}]  n {v['n']}{extra}{flags}"


def print_report(E, rep, out, detail):
    A = all_arms(E)
    print(f"{E['id']} report: rules version 2, frozen {E.get('frozen_at')} at {E.get('commit')} (sha {E['frozen_sha256'][:12]}), "
          f"{rep['resamples']} resamples, seed {rep['seed']}, CI level {rep['level']:g}, min_seeds {rep['min_seeds']}")
    print("members with results:")
    for arm, cells in rep["results"].items():
        tag = " (reference arm)" if arm in E.get("reference_arms", {}) else ""
        print(f"  {arm:10s}" + "  ".join(f"{c} {len(v['members'])}/{len(A[arm])}" + (f" (missing {', '.join(v['missing'])})"
                                                                                    if 0 < len(v.get("missing", [])) < len(A[arm]) else "")
                                         for c, v in cells.items()) + tag)
    for w, cells in rep["leaks"].items():
        print(f"  leaks {w}: " + "; ".join(f"{c} " + ("leak-free" if r["leak_free"] else "LEAK") +
                                          (f" (emb_db_exposure {r['emb_db_exposure']})" if "emb_db_exposure" in r else "")
                                          for c, r in cells.items()))
    print("between-seed SD (mean per-molecule SD / SD of member means):")
    for arm, cells in rep["seed_sd"].items():
        for cell, ms in cells.items():
            print(f"  {arm:10s}{cell:9s}" + "  ".join(f"{m} {fmt(v['mean_mol_sd'])[1:]}/{fmt(v['sd_member_means'])[1:]}"
                                                     for m, v in ms.items()))
    if "primary" in rep:
        P = rep["primary"]["spec"]
        print("primary:", stat_line(f"{P.get('arm') or '-'.join(P['contrast'])}|{P['cell']}|{P['metric']}"
                                    + (f"|w:{P['weights']}" if P.get("weights") else ""), rep["primary"]).strip())
    for title, part in (("contrasts", "contrasts"), ("arm statistics", "arm_stats")):
        print(f"{title} (two-sided {rep['level']:g}):")
        for k, v in rep[part].items():
            print(stat_line(k, v))
    print("verdict:")
    for name, d in detail.items():
        if isinstance(d, dict):
            print(f"  {name} = {d['expr']} -> {d['result']}")
            continue
        print(f"  {name} -> {show(out[name])}")
        for c in d:
            who = c.get("arm") or "-".join(c["contrast"])
            what = f"{who} {c['cell']} {c['metric']}" + (f" w:{c['weights']}" if c.get("weights") else "")
            got = f"est {fmt(c['estimate'])}" if "estimate" in c else ""
            if "bounds" in c:
                lo, hi = c["bounds"]
                got += {"two": f", CI [{fmt(lo)}, {fmt(hi)}] ({c['level']:g})", "lower": f", lower bound {fmt(lo)} (one-sided {c['level']:g})",
                        "upper": f", upper bound {fmt(hi)} (one-sided {c['level']:g})"}[c["sided"]]
            why = f"  ({c['why']})" if "why" in c else ""
            print(f"      {what:40s} {c['test']} {c['value']:g}: {got} -> {c['result']}{why}")


def report_v2(E, path, again=False, reason=""):
    """Report a frozen exploration. A reported one is reported again only with --again and a --reason; the earlier
    verdicts stay in report_history (a void or archived exploration is final)."""
    eid = E["id"]
    if E.get("state") == "reported" and again:
        if not reason:
            sys.exit("--again needs a --reason (it is kept with the superseded report)")
        E["report_history"] = E.get("report_history", []) + [
            {k: E.get(k) for k in ("reported_at", "report_commit", "report_sha256", "verdict")} | {"superseded_because": reason}]
    elif E.get("state") != "frozen":
        sys.exit(f"{eid} is {E.get('state')!r}: only a frozen exploration is reported"
                 + (" (to report it again: --again --reason R; the earlier verdict is kept)" if E.get("state") == "reported" else ""))
    if frozen_sha(E) != E.get("frozen_sha256"):
        sys.exit(f"{eid}: the frozen object changed after freezing (frozen_sha256 {str(E.get('frozen_sha256'))[:12]} != "
                 f"{frozen_sha(E)[:12]}); a changed exploration is a new exploration")
    C = json.load(open(f"{REG}/cells.json")) if os.path.exists(f"{REG}/cells.json") else {}
    for c, m in E["cell_manifests"].items():
        if c not in C or sha_obj(C[c]["sha256"]) != m["sha256"]:
            sys.exit(f"{eid}: cell {c} is no longer the one frozen with the exploration ({REG}/cells.json)")
        p = f"{cell_dir(c)}/molecules.parquet"
        if os.path.exists(p) and sha_file(p) != C[c]["sha256"].get("molecules.parquet"):
            sys.exit(f"{eid}: {p} differs from the frozen manifest of {c}")
    A = Analysis(E)
    leaks, void = A.check_leaks()
    if void:
        E.update(state="void", verdict="; ".join(void), void_reason=void, leaks=jsonable(leaks), reported_at=now(),
                 report_commit=commit())
        json.dump(E, open(path, "w"), indent=1)
        print(f"{eid} is void:", *void, sep="\n  ")
        return
    rep = A.report()
    out, detail = A.verdicts()
    rep.update(leaks=leaks, conditions=detail)
    rep = jsonable(rep)
    E.update(report=rep, verdict={k: show(v) for k, v in out.items()}, state="reported", reported_at=now(),
             report_commit=commit(), report_sha256=sha_obj(rep))
    json.dump(E, open(path, "w"), indent=1)
    print_report(E, rep, out, rep["conditions"])
    print(f"reported {eid} (report {E['report_sha256'][:12]}) -> {path}; commit it")


def table_cmd(a):
    R = pd.DataFrame(jsonl(f"{REG}/results.jsonl"))
    R = R[R.metric.isin(a.metric.split(","))].drop_duplicates(["weights", "cell", "metric"], keep="last")
    R["v"] = [f"{m:.3f} [{c[0]:.3f},{c[1]:.3f}]" + ("" if lf else " LEAKY") for m, c, lf in zip(R["mean"], R.ci95, R.leak_free)]
    print(R.pivot_table(index=["weights", "metric"], columns="cell", values="v", aggfunc="first").to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cells"); c.add_argument("action", choices=["freeze", "verify"]); c.add_argument("--force", action="store_true")
    w = sp.add_parser("weights"); w.add_argument("action", choices=["add"]); w.add_argument("path")
    w.add_argument("--name", required=True); w.add_argument("--kind", required=True, choices=["spectrum_model", "jepa_encoder"])
    w.add_argument("--encoder", default=""); w.add_argument("--val", default=""); w.add_argument("--work", default="work/w4")
    w.add_argument("--recipe", default="")
    l = sp.add_parser("leak"); l.add_argument("name")
    s = sp.add_parser("score"); s.add_argument("name")
    e = sp.add_parser("explore"); e.add_argument("action", choices=["check", "freeze", "report", "archive"]); e.add_argument("target")
    e.add_argument("--reason", default=""); e.add_argument("--again", action="store_true", help="report a reported exploration again")
    t = sp.add_parser("table"); t.add_argument("--metric", default="mass:nf+0.25doc,mass:nf")
    a = ap.parse_args(argv)
    {"cells": cells_cmd, "weights": weights_cmd, "leak": lambda a: leak_report(a.name), "score": score_cmd,
     "explore": explore_cmd, "table": table_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
