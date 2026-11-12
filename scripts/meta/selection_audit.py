"""E013: selection audit of the public 0.480 lineage on the leak-free Enveda-180 frame (registry/drafts/E013.json).

Scores every variant file of the two F_ext executions with the official key (casmi.metric.ik14), then reports
Q1 (MRR and paired differences against final), Q2 (run-to-run agreement), Q3 (documented LB steps against the frame),
Q4 (public/private split simulation of best-of-K selection) and Q5 (paired SE of a difference on 125 molecules).

  python scripts/meta/selection_audit.py [--design registry/drafts/E013.json] [--out work/e013] [--procs 3]
"""
from __future__ import annotations

import argparse, json, os, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from casmi.metric import ik14  # noqa: E402

K = 25


def keys_for(smiles: list[str], procs: int, cache: str) -> dict[str, str | None]:
    have = pd.read_parquet(cache) if os.path.exists(cache) else pd.DataFrame({"smiles": [], "ik14": []})
    known = dict(zip(have.smiles, have.ik14))
    todo = sorted(set(smiles) - set(known))
    if todo:
        with Pool(procs) as p:
            got = p.map(ik14, todo, chunksize=256)
        known.update(zip(todo, got))
        pd.DataFrame({"smiles": list(known), "ik14": list(known.values())}).to_parquet(cache, index=False)
    return known


def boot_ci(d: np.ndarray, rng, n=4000):
    bs = d[rng.integers(len(d), size=(n, len(d)))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(np.percentile(bs, 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="registry/drafts/E013.json")
    ap.add_argument("--root", default="/home/xaxi/casmi")
    ap.add_argument("--out", default="work/e013")
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    D = json.load(open(a.design))
    os.makedirs(a.out, exist_ok=True)
    runs = {r: os.path.join(a.root, p) for r, p in D["data"]["runs"].items()}
    truth = pd.read_parquet(f"{runs['run1']}/val_answers.parquet").set_index("molecule_id").smiles
    t2 = pd.read_parquet(f"{runs['run2']}/val_answers.parquet").set_index("molecule_id").smiles
    assert truth.sort_index().equals(t2.sort_index()), "runs disagree on the truth"
    mids = truth.index.to_numpy()
    lists = {}
    for r, d in runs.items():
        for v, f in D["variants"].items():
            s = pd.read_csv(f"{d}/{f}").set_index("molecule_id").smiles.reindex(mids)
            lists[r, v] = [[g for g in str(x).split(";") if g] if isinstance(x, str) else [] for x in s]
    allsmi = list(truth) + [g for L in lists.values() for row in L for g in row[:K]]
    key = keys_for(allsmi, a.procs, f"{a.out}/ik14_cache.parquet")
    tk = np.array([key[s] for s in truth])
    RR, TOP = {}, {}
    for (r, v), L in lists.items():
        rr, top = np.zeros(len(mids)), []
        for i, row in enumerate(L):
            ks = [key[g] for g in row[:K]]
            top.append(ks[0] if ks else None)
            for j, k in enumerate(ks):
                if k is not None and k == tk[i]:
                    rr[i] = 1.0 / (j + 1)
                    break
        RR[r, v], TOP[r, v] = rr, top
    V = list(D["variants"])
    pd.DataFrame({f"{r}:{v}": RR[r, v] for r in runs for v in V}, index=mids).to_parquet(f"{a.out}/rr.parquet")
    rng = np.random.default_rng(0)
    mean2 = {v: (RR["run1", v] + RR["run2", v]) / 2 for v in V}
    out = {"n_molecules": len(mids), "Q1": {}, "Q2": {}, "Q3": {}, "Q4": {}, "Q5": {}}
    for v in V:
        q = {"mrr_run1": RR["run1", v].mean(), "mrr_run2": RR["run2", v].mean(), "mrr_mean": mean2[v].mean()}
        for tag, d in (("run1", RR["run1", v] - RR["run1", "final"]), ("run2", RR["run2", v] - RR["run2", "final"]),
                       ("mean", mean2[v] - mean2["final"])):
            lo, hi, lo1 = boot_ci(d, rng)
            q[f"d_{tag}"], q[f"ci_{tag}"], q[f"lo1_{tag}"] = float(d.mean()), [lo, hi], lo1
        out["Q1"][v] = q
        out["Q2"][v] = {"mrr_diff": float(RR["run1", v].mean() - RR["run2", v].mean()),
                        "share_rr_differs": float((RR["run1", v] != RR["run2", v]).mean()),
                        "share_top1_differs": float(np.mean([x != y for x, y in zip(TOP["run1", v], TOP["run2", v])]))}
    lb = D["lb_steps"]["scores"]
    num = {"all": [0.0, 0.0], "early": [0.0, 0.0], "late": [0.0, 0.0]}
    for i, (x, y) in enumerate(D["lb_steps"]["steps"]):
        d = mean2[y] - mean2[x]
        lo, hi, _ = boot_ci(d, rng)
        dl = lb[y][0] - lb[x][0]
        out["Q3"][f"{x}->{y}"] = {"d_lb": dl, "d_fext": float(d.mean()), "ci": [lo, hi], "ratio": float(d.mean() / dl) if dl else None}
        for g in ("all", "early" if i < 2 else "late"):
            num[g][0] += float(d.mean()); num[g][1] += dl
    out["Q3"]["shrinkage"] = {g: (s / l if l else None) for g, (s, l) in num.items()}
    sets = {"K_all": V, "K_cons": ["V0", "V1", "V2", "V3", "V4", "CUSTOM"], "K_late": ["V2", "CUSTOM", "final"]}
    exp_priv = {v: mean2[v].mean() for v in V}
    n, npub = len(mids), 125
    for name, S in list(sets.items()) + [("resubmit_final", None)]:
        infl, regret, picks = [], [], {}
        for _ in range(4000):
            perm = rng.permutation(n)
            pub, prv = perm[:npub], perm[npub:]
            if S is None:   # the same notebook submitted twice: best public of its two executions
                cand = [("final", r) for r in ("run1", "run2")]
            else:
                cand = [(v, "run1" if rng.random() < 0.5 else "run2") for v in S]
            ps = [RR[r, v][pub].mean() for v, r in cand]
            j = int(np.argmax(ps))
            v, r = cand[j]
            pv = RR[r, v][prv].mean()
            infl.append(ps[j] - pv)
            best = max(exp_priv[c] for c, _ in cand)
            regret.append(best - pv)
            picks[v] = picks.get(v, 0) + 1
        out["Q4"][name] = {"inflation_mean": float(np.mean(infl)), "inflation_p90": float(np.percentile(infl, 90)),
                           "regret_mean": float(np.mean(regret)), "picks": {k: c / 4000 for k, c in sorted(picks.items())}}
    ses = {}
    for i, x in enumerate(V):
        for y in V[i + 1:]:
            ses[f"{x}-{y}"] = float(np.std(mean2[x] - mean2[y], ddof=1) / np.sqrt(npub))
    out["Q5"] = {"median_se_125": float(np.median(list(ses.values()))),
                 "steps": {f"{x}-{y}": ses.get(f"{x}-{y}", ses.get(f"{y}-{x}")) for x, y in D["lb_steps"]["steps"]},
                 "max_se_125": float(max(ses.values()))}
    rule = [v for v in V if v != "final" and out["Q1"][v]["lo1_run1"] > 0.005 and out["Q1"][v]["lo1_run2"] > 0.005]
    out["pick_A_rule"] = {"variants_passing": rule, "decision": "switch" if rule else "keep public final"}
    json.dump(out, open(f"{a.out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4))[:12000])


if __name__ == "__main__":
    main()
