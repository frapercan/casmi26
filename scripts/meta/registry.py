"""Evaluation registry, after PROTEA's ADRs D40 (leakage-free protocol), D54 (model-weights registry) and
D55 (evaluated explorations): frozen cells, registered weights, leakage checked by construction, and
pre-registered explorations whose verdicts are computed from rows, never read off by eye.

  registry/cells.json               frozen cell manifests: role, population, known biases, counts, sha256 of every file
  registry/weights.jsonl            one row per checkpoint / encoder: kind, sha256, recipe, what its training hid
  registry/explorations/E###.json   hypothesis, arms, cells, structured rules + sha256 frozen BEFORE results, report, verdict
  registry/results.jsonl            one row per (weights, cell, metric): mean, paired-bootstrap CI, leak flags, cell sha, commit

  python scripts/meta/registry.py cells freeze | verify
  python scripts/meta/registry.py weights add PATH --name N --kind spectrum_model|jepa_encoder [--encoder E] [--val DIR] [--work DIR]
  python scripts/meta/registry.py leak N
  python scripts/meta/registry.py score N               (rows from work/cells/eval/N__<cell>.parquet)
  python scripts/meta/registry.py explore freeze FILE | report ID
  python scripts/meta/registry.py table [--metric mass:nf+0.25doc,mass:nf]
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, os, subprocess, sys

import numpy as np
import pandas as pd

REG = "registry"
EVAL = "work/cells/eval"
CELLS = {
    "np_tims": dict(dir="work/cells/np_tims", role="valid", simulates="class 2, timsTOF natural products (Enveda's own panel)",
                    population="enveda-np-examples structures in the PubChem tier, one neutral species; their timsTOF spectra "
                               "as queries; excluded from every training (spectra, anchors, negatives) since the start",
                    biases=["far better documented than the typical NP (doc median 13.0): the documentation prior is optimistic",
                            "the prior weight 0.25 was chosen on it on 2026-11-09"]),
    "np_c2": dict(dir="work/cells/np_c2", role="valid", simulates="class 2, natural products without reference spectra",
                  population="COCONUT/LOTUS/NPAtlas structures with public spectra, 157-1159 Da, mass histogram matched to "
                             "np-examples (n 800, seed 0); all their spectra hidden from training (work/cells registry)",
                  biases=["selected by NP-database membership: NP-membership priors are circular here",
                          "queries are Orbitrap/QTOF library spectra, not timsTOF"]),
    "np_like": dict(dir="work/cells_x/np_like", role="prior_only", simulates="class 2, NP-like structures, database membership free",
                    population="Ertl NP-score > 1, public spectra, mass-matched (n 800, seed 0), disjoint from the work/cells "
                               "registry; 82% carry an NP-database bit",
                    biases=["its spectra are NOT hidden from any training so far: model results on it are leaky until a model "
                            "is trained with work/cells_x hidden"]),
}
FILES = ["molecules.parquet", "queries.parquet", "cands.parquet", "structs.parquet", "exclude_all_ik.txt", "plm.npy", "fp.npy"]
# what each molecule table hid from anchors and negatives (scripts/build_mol_table.py invocations)
EMB_EXCLUSION = {"emb_v3": ["work/c3data/exclude_all_ik.txt"],
                 "emb_v4": ["work/c3data/exclude_all_ik.txt", "work/cells/exclude_all_ik.txt"]}
HOLDOUT = "/kaggle/input/casmi26-train-v2/casmi26/splits/holdout.parquet"


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


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


def cell_manifest(name):
    c = CELLS[name]; d = c["dir"]
    m = pd.read_parquet(f"{d}/molecules.parquet"); q = pd.read_parquet(f"{d}/queries.parquet", columns=["mol_id"])
    n_c = len(pd.read_parquet(f"{d}/cands.parquet", columns=["sid"]))
    return {"name": name, **c, "n_molecules": len(m), "n_queries": len(q), "n_candidate_rows": n_c,
            "doc_median": float(m.doc.median()), "mass_median": float(m.exact_mass.median()),
            "sha256": {f: sha_file(f"{d}/{f}") for f in FILES if os.path.exists(f"{d}/{f}")}}


def cells_cmd(a):
    path = f"{REG}/cells.json"
    old = json.load(open(path)) if os.path.exists(path) else {}
    if a.action == "freeze":
        for name in CELLS:
            if not os.path.exists(f"{CELLS[name]['dir']}/cands.parquet"):
                continue
            m = cell_manifest(name)
            if name in old and old[name]["sha256"] != m["sha256"] and not a.force:
                sys.exit(f"{name}: files differ from the frozen manifest; a changed cell is a new cell (use a new name)")
            if name not in old:
                m["frozen_at"], m["commit"] = now(), commit()
                old[name] = m
                print(f"froze {name}: {m['n_molecules']} molecules, {m['n_queries']} queries")
        json.dump(old, open(path, "w"), indent=1)
    else:
        bad = 0
        for name, m in old.items():
            for f, h in m["sha256"].items():
                ok = sha_file(f"{m['dir']}/{f}") == h
                bad += not ok
                if not ok:
                    print(f"MISMATCH {name}/{f}")
        print("cells verified" if not bad else f"{bad} files differ")


def weights_cmd(a):
    import torch
    rows = jsonl(f"{REG}/weights.jsonl")
    if any(r["name"] == a.name for r in rows):
        sys.exit(f"{a.name} already registered")
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


def leak_report(name, verbose=True):
    """Spectra leak: cell structures whose spectra the weights (or their init encoder) trained on.
    Anchor leak: cell structures that were anchors / negatives of the molecule table the model trained against."""
    W = {r["name"]: r for r in jsonl(f"{REG}/weights.jsonl")}
    r = W[name]
    chain = [r] + ([W[r["init_encoder"]]] if r.get("init_encoder") in W else [])
    spec = pd.read_parquet("work/lib/spectra.parquet", columns=["ingest_lib", "inchikey14"])
    always = set(pd.read_parquet(HOLDOUT).inchikey14) | set(spec.inchikey14[spec.ingest_lib == "enveda-np-examples"])
    out = {}
    for cname, c in CELLS.items():
        if not os.path.exists(f"{c['dir']}/molecules.parquet"):
            continue
        m = pd.read_parquet(f"{c['dir']}/molecules.parquet")
        keys = [set(filter(None, (a, b, s))) for a, b, s in zip(m.ik14_lib, m.ik14_raw, m.skey)]
        res = {}
        for w in chain:
            hid = always | read_set(w["hidden"]["exclude_all"])
            res[f"spectra_leak:{w['name']}"] = int(sum(not (k & hid) for k in keys))
        emb_ex = always.copy()
        for f in r["hidden"]["emb_exclusion"] or []:
            emb_ex |= read_set(f)
        res["anchor_leak"] = int(sum(not (k & emb_ex) for k in keys)) if r["kind"] == "spectrum_model" else 0
        res["leak_free"] = all(v == 0 for k, v in res.items() if k.endswith("leak") or k.startswith("spectra_leak"))
        out[cname] = res
        if verbose:
            print(f"{name} on {cname}: " + ", ".join(f"{k}={v}" for k, v in res.items()))
    return out


def boot(x, rng, n=4000):
    bs = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


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


def explore_cmd(a):
    os.makedirs(f"{REG}/explorations", exist_ok=True)
    if a.action == "freeze":
        E = json.load(open(a.target))
        path = f"{REG}/explorations/{E['id']}.json"
        if os.path.exists(path):
            sys.exit(f"{E['id']} is already frozen; a changed exploration is a new exploration")
        existing = [k for k, w in E["arms"].items() if any(os.path.exists(f"{EVAL}/{w}__{c}.parquet") for c in E["cells"])]
        E.update(frozen_at=now(), commit=commit(), rules_sha256=sha_obj(E["rules"]), arms_with_results_at_freeze=existing,
                 post_hoc=any(k in existing for k in E.get("candidate_arms", [])), state="frozen")
        json.dump(E, open(path, "w"), indent=1)
        print(f"froze {E['id']} rules {E['rules_sha256'][:12]}; arms with results already: {existing}; post_hoc={E['post_hoc']}")
        return
    path = f"{REG}/explorations/{a.target}.json"
    E = json.load(open(path))
    assert sha_obj(E["rules"]) == E["rules_sha256"], "rules changed after freezing"
    rules = E["rules"]
    rng = np.random.default_rng(rules["bootstrap"]["seed"])
    for k, w in E.get("arm_weights", E["arms"]).items():   # arms may be scoring rules over registered weights
        lk = leak_report(w, verbose=False)
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


def table_cmd(a):
    R = pd.DataFrame(jsonl(f"{REG}/results.jsonl"))
    R = R[R.metric.isin(a.metric.split(","))].drop_duplicates(["weights", "cell", "metric"], keep="last")
    R["v"] = [f"{m:.3f} [{c[0]:.3f},{c[1]:.3f}]" + ("" if lf else " LEAKY") for m, c, lf in zip(R["mean"], R.ci95, R.leak_free)]
    print(R.pivot_table(index=["weights", "metric"], columns="cell", values="v", aggfunc="first").to_string())


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cells"); c.add_argument("action", choices=["freeze", "verify"]); c.add_argument("--force", action="store_true")
    w = sp.add_parser("weights"); w.add_argument("action", choices=["add"]); w.add_argument("path")
    w.add_argument("--name", required=True); w.add_argument("--kind", required=True, choices=["spectrum_model", "jepa_encoder"])
    w.add_argument("--encoder", default=""); w.add_argument("--val", default=""); w.add_argument("--work", default="work/w4")
    w.add_argument("--recipe", default="")
    l = sp.add_parser("leak"); l.add_argument("name")
    s = sp.add_parser("score"); s.add_argument("name")
    e = sp.add_parser("explore"); e.add_argument("action", choices=["freeze", "report"]); e.add_argument("target")
    t = sp.add_parser("table"); t.add_argument("--metric", default="mass:nf+0.25doc,mass:nf")
    a = ap.parse_args()
    {"cells": cells_cmd, "weights": weights_cmd, "leak": lambda a: leak_report(a.name), "score": score_cmd,
     "explore": explore_cmd, "table": table_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
