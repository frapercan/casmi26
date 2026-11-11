"""Offline replay of the 0.480 notebook's whole post-pipeline stack on a cached pipeline run, for any neural input.

replay_consensus.py replays only the consensus order. This runs the notebook's own code, extracted from the notebook
at run time, stage by stage, exactly as cells 14-18 chain them:
  consensus.build_variants (V0-V4, CUSTOM)  ->  CUSTOM copied to submission.csv
  parallax_physical.apply (fragment check on V2; may move rank 1 and swap neighbours)  ->  submission.csv
  cell 18 (Variant M: rank 1 from the fragment check, ranks 2-25 from CUSTOM + documentation prior + dedup + fill)
and scores every stage file against the run's answers with the official metric key.

Arms (each one a neural_candidate_features + outsider_neural_features pair; everything else is the cached run):
  public       the run's own files (the replay must reproduce the run's hashes)
  b0repro      what the B0-repro cell builds: neural_score 0.5 z(public) + 0.5 z(ours); neural_cos / neural_fp /
               neural_std averaged with ours; neural_ok AND; outside candidates scored by our model alone
  b0_pubgates  b0repro, but neural_cos / neural_fp / neural_std left public (the fragment check's absolute gates)
  b0_pubout    b0repro, but outside candidates scored by the public model

  python hybrid/replay_stack.py --run /kaggle/input/casmi26-eval-c3 --ours work/replay_c3/neural_ours_chk.parquet \
      --ours-out work/replay_c3/outsider_ours_w4base.parquet --work work/replay_stack
"""
from __future__ import annotations

import argparse, ast, hashlib, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract(nb_path, code_dir):
    """Write the notebook's pinned modules and the fragment-check sources; return cell 18's source."""
    nb = json.load(open(nb_path))
    cells = ["".join(c["source"]) for c in nb["cells"]]
    for node in ast.parse(cells[11]).body:
        if isinstance(node, ast.Assign):
            t = node.targets[0]
            if isinstance(t, ast.Subscript) and getattr(t.value, "id", "") == "MODULES":
                (code_dir / ast.literal_eval(t.slice)).write_text(ast.literal_eval(node.value))
    src13 = {}
    for node in ast.parse(cells[13]).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("_physical", "_post"):
            src13[node.targets[0].id] = ast.literal_eval(node.value)
    (code_dir / "frag_rescore.py").write_text(src13["_physical"])
    (code_dir / "parallax_physical.py").write_text(src13["_post"])
    c18 = next(c for c in cells if "def pxm_merge" in c)
    cfg = next(c for c in cells if c.startswith("# ") and "VARIANT = " in c and "POP_W" in c)
    return c18, cfg


def finite_compat(x, default=np.nan):
    """parallax_physical.finite as it behaves under the Kaggle image's numpy: float() of a one-element array is that
    element (collision_energy_ev holds one-element arrays); numpy >= 2.5 raises instead, which sends every spectrum
    of a molecule to the default 30 eV bucket and caps the fragment check at 2 spectra."""
    try:
        a = np.asarray(x, dtype=float)
        t = float(a.reshape(-1)[0]) if a.size == 1 else float(x)
        return t if np.isfinite(t) else default
    except (ValueError, TypeError):
        return default


def z(s):
    g = s.groupby(level=0)
    return (s - g.transform("mean")) / g.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)


def b0repro_features(pub, ours, keep_public_gates=False):
    """The B0-repro cell's transformation (hybrid/build_b0repro.py CELL14), optionally leaving the gate columns public."""
    a = pub.set_index(["molecule_id", "smiles"]).copy()
    b = ours.set_index(["molecule_id", "smiles"]).reindex(a.index)
    a["neural_score"] = 0.5 * z(a["neural_score"].fillna(0.0)) + 0.5 * z(b["neural_score"].fillna(0.0))
    if not keep_public_gates:
        for c in ("neural_cos", "neural_fp", "neural_std"):
            a[c] = 0.5 * a[c] + 0.5 * b[c].fillna(0.0)
    a["neural_ok"] = a["neural_ok"].astype(bool) & b["neural_ok"].fillna(False).astype(bool)
    return a.reset_index()


def mrr_of(path, truth, key_of):
    S = pd.read_csv(path, dtype=str, keep_default_na=False)
    rr, top = {}, {}
    for m, s in zip(S.molecule_id, S.smiles):
        L = []
        for x in s.split(";"):
            k = key_of(x)
            if k not in L:
                L.append(k)
        L = L[:25]
        t = truth.get(m)
        rr[m] = 1 / (L.index(t) + 1) if t in L else 0.0
        top[m] = s.split(";")[0]
    return rr, top


def run_arm(name, run, work, code_dir, c18, cfg_src, nf, of, pop_csv):
    root = work / name
    if root.exists():
        shutil.rmtree(root)
    (root / "valcomp").mkdir(parents=True)
    for f in ("effective_query_spectra.parquet", "e004_candidate_inputs.json", "submission_baseline0465.csv"):
        os.symlink(run / f, root / f)
    os.symlink(run / "valcomp" / "test.parquet", root / "valcomp" / "test.parquet")
    (root / "input").mkdir()
    os.symlink(pop_csv, root / "input" / "pool_popularity.csv")
    nf.to_parquet(root / "neural_candidate_features.parquet", index=False)
    of.to_parquet(root / "outsider_neural_features.parquet", index=False)
    G = {}
    exec(cfg_src, G)                                   # the notebook's configuration cell (VARIANT, CUSTOM, POP_W, ...)
    import consensus, parallax_physical                # noqa: E401  (from code_dir, on sys.path)
    parallax_physical.finite = finite_compat
    presets = dict(consensus.PRESETS)
    presets["CUSTOM"] = dict(w_top=float(G["CUSTOM"]["w_top"]), w_mid=float(G["CUSTOM"]["w_mid"]),
                             rescue=int(G["CUSTOM"].get("rescue", 0)), label="Custom weights")
    baseline = root / "submission_baseline0465.csv"
    report, _ = consensus.build_variants(root / "neural_candidate_features.parquet", baseline, root / "variants", presets,
                                         G["RRF_OFFSET"], root / "outsider_neural_features.parquet")
    shutil.copyfile(root / f"variants/submission_{G['VARIANT']}.csv", root / "submission.csv")   # choose_and_submit
    shutil.copyfile(root / "submission.csv", root / "submission_control.csv")
    diag = parallax_physical.apply(root, root / "valcomp", max_seconds=1500, allow_tail=True, allow_top=True)
    shutil.copyfile(root / "submission.csv", root / "submission_physical.csv")
    sample_ids = pd.read_csv(baseline, dtype=str, keep_default_na=False).molecule_id.tolist()

    def identity_map():
        keys = {}
        for p in (root / "neural_candidate_features.parquet", root / "outsider_neural_features.parquet"):
            if p.is_file():
                f = pd.read_parquet(p, columns=["smiles", "metric_key"])
                keys.update({s: k for s, k in zip(f.smiles, f.metric_key) if isinstance(k, str)})
        return keys

    G.update(ROOT=root, INPUT=root / "input", SAMPLE_IDS=sample_ids, identity_map=identity_map, PIPELINE_DONE=True,
             IS_RERUN=True, PARALLAX_DIAGNOSTICS=diag)
    exec(c18, G)                                        # cell 18 as written; it rewrites submission.csv
    return root, report, diag, G.get("MERGE_REPORT", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/kaggle/input/casmi26-eval-c3")
    ap.add_argument("--nb", default="nb/base480.ipynb")
    ap.add_argument("--ours", required=True, help="our scores on the run's frozen_candidate_evidence")
    ap.add_argument("--ours-out", required=True, help="our scores on the run's outsider_evidence")
    ap.add_argument("--pop", default="/kaggle/input/casmi26-pubchem-popularity-prior/pool_popularity.csv")
    ap.add_argument("--work", default="work/replay_stack")
    ap.add_argument("--arms", default="public,b0repro,b0_pubgates,b0_pubout")
    a = ap.parse_args()
    run, work = Path(a.run), Path(a.work).resolve()
    code_dir = work / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    c18, cfg_src = extract(a.nb, code_dir)
    sys.path.insert(0, str(code_dir))
    from casmi.metric import ik14
    ans = pd.read_parquet(run / "val_answers.parquet")
    truth = {m: ik14(s) for m, s in zip(ans.molecule_id, ans.smiles)}
    pub = pd.read_parquet(run / "neural_candidate_features.parquet")
    pub_out = pd.read_parquet(run / "outsider_neural_features.parquet")
    ours = pd.read_parquet(a.ours)
    ours_out = pd.read_parquet(a.ours_out)
    keys = {}
    for f in (pub, pub_out):
        keys.update({s: k for s, k in zip(f.smiles, f.metric_key) if isinstance(k, str) and k})

    def key_of(x):
        if x not in keys:
            try:
                keys[x] = ik14(x)
            except Exception:
                keys[x] = x
        return keys[x]

    arms = {
        "public": (pub, pub_out),
        "b0repro": (b0repro_features(pub, ours), ours_out),
        "b0_pubgates": (b0repro_features(pub, ours, keep_public_gates=True), ours_out),
        "b0_pubout": (b0repro_features(pub, ours), pub_out),
    }
    stages = [("V0", "variants/submission_V0.csv"), ("V1", "variants/submission_V1.csv"),
              ("V2", "variants/submission_V2.csv"), ("CUSTOM", "variants/submission_CUSTOM.csv"),
              ("physical", "submission_physical.csv"), ("final", "submission.csv")]
    per, rows, tops = {}, [], {}
    for name in a.arms.split(","):
        t0 = time.time()
        nf, of = arms[name]
        root, report, diag, merge = run_arm(name, run, work, code_dir, c18, cfg_src, nf, of, a.pop)
        for st, f in stages:
            rr, top = mrr_of(root / f, truth, key_of)
            per[(name, st)] = rr
            tops[(name, st)] = top
            rows.append({"arm": name, "stage": st, "mrr": np.mean(list(rr.values())), "n": len(rr)})
        print(f"{name}: physical top1 changes {diag.get('top1_changed')}, swaps {diag.get('adjacent_swaps')}, "
              f"tail {diag.get('tail_changed')}; merge leader moves {merge.get('leader_moves')}, old leader pinned "
              f"{merge.get('old_leader_pinned')}, outsider fills {merge.get('outsider_fills')} [{time.time() - t0:.0f}s]",
              flush=True)
        if name == "public":
            for st, mine, theirs in (("V2", "variants/submission_V2.csv", "variants/submission_V2.csv"),
                                     ("CUSTOM", "variants/submission_CUSTOM.csv", "variants/submission_CUSTOM.csv"),
                                     ("final", "submission.csv", "submission.csv")):
                print(f"  reproduces the run's {st} file: {sha(root / mine) == sha(run / theirs)}", flush=True)
    R = pd.DataFrame(rows).pivot(index="stage", columns="arm", values="mrr").loc[[s for s, _ in stages]]
    print(R.round(4).to_string())
    rng = np.random.default_rng(0)
    mids = sorted(per[("public", "final")])
    for name in [x for x in a.arms.split(",") if x != "public"]:
        for st in ("V2", "CUSTOM", "physical", "final"):
            d = np.array([per[(name, st)][m] - per[("public", st)][m] for m in mids])
            bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(4000)]
            lead = sum(tops[(name, st)][m] != tops[("public", st)][m] for m in mids)
            print(f"{name:12s} - public  {st:8s} {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
                  f"  up {(d > 0).sum()} down {(d < 0).sum()}  rank-1 differs {lead}")
    # where the final rank 1 changed against the public arm: who was right
    for name in [x for x in a.arms.split(",") if x != "public"]:
        moved = [m for m in mids if tops[(name, "final")][m] != tops[("public", "final")][m]]
        good = sum(key_of(tops[(name, "final")][m]) == truth.get(m) for m in moved)
        bad = sum(key_of(tops[("public", "final")][m]) == truth.get(m) for m in moved)
        print(f"{name}: final rank 1 differs from public on {len(moved)} molecules; right after the change {good}, "
              f"right before {bad}")


if __name__ == "__main__":
    main()
