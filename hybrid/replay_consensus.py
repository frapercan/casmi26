"""Offline replay of the public pipeline's consensus stage with any neural score, on a cached pipeline run.

Re-implements ranking.rrf_order / consensus.leader_safe_order (rank 1 from 1/(k+b) + w_top/(k+n), ranks 2+ from
1/(k+b) + w_mid/(k+n); b baseline rank, n neural rank) and the 480 notebook's ensemble
neural = 0.5 z(public) + 0.5 z(second scorer), then scores MRR@25 with the official metric key against the
cell's answers. Physical fragment check and the final layer are not replayed.

  python hybrid/replay_consensus.py --run /kaggle/input/casmi26-eval-c3 --ours work/replay_c3/neural_ours_new.parquet
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def stable_rank(v):
    v = np.asarray(v, float)
    o = np.argsort(-v, kind="stable"); r = np.empty(len(v), int); r[o] = np.arange(1, len(v) + 1)
    return r


def rrf_order(s, w, k=3.0):
    n = len(s)
    if w == 0:
        return np.arange(n)
    return np.argsort(-(1 / (k + np.arange(1, n + 1)) + w / (k + stable_rank(s))), kind="stable")


def leader_safe(s, w_top, w_mid, k=3.0):
    first = rrf_order(s, w_top, k)
    if w_mid == w_top or len(s) == 0:
        return first
    lead = int(first[0])
    return np.array([lead] + [int(i) for i in rrf_order(s, w_mid, k) if int(i) != lead])


def z(s):
    g = s.groupby(level=0)
    return (s - g.transform("mean")) / g.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/kaggle/input/casmi26-eval-c3")
    ap.add_argument("--ours", action="append", default=[], help="NAME=path of a neural feature file (our scorer)")
    ap.add_argument("--w-ours", default="0.5", help="comma list of z-weights of the second scorer (0.5 = the 480 notebook)")
    a = ap.parse_args()
    W_OURS = [float(x) for x in a.w_ours.split(",")]
    from casmi.metric import ik14
    ans = pd.read_parquet(f"{a.run}/val_answers.parquet")
    truth = {m: ik14(s) for m, s in zip(ans.molecule_id, ans.smiles)}
    pub = pd.read_parquet(f"{a.run}/neural_public.parquet").sort_values(["molecule_id", "baseline_rank"], kind="stable")
    key = ["molecule_id", "smiles"]
    P = pub.set_index(key)
    scorers = {"run_second": pd.read_parquet(f"{a.run}/neural_ours.parquet").set_index(key).reindex(P.index)}
    for spec in a.ours:
        name, path = spec.split("=")
        scorers[name] = pd.read_parquet(path).set_index(key).reindex(P.index)
    neural = {"public": (P.neural_score.fillna(0), P.neural_ok.astype(bool))}
    for name, S in scorers.items():
        for wo in W_OURS:
            neural[f"{1 - wo:g}pub+{wo:g}{name}"] = ((1 - wo) * z(P.neural_score.fillna(0)) + wo * z(S.neural_score.fillna(0)),
                                                    P.neural_ok.astype(bool) & S.neural_ok.fillna(False).astype(bool))
        neural[f"{name}_only"] = (S.neural_score.fillna(0), S.neural_ok.fillna(False).astype(bool))
    mk = P.metric_key.astype(str)
    proc = P.processed.astype(bool)
    rows, per = [], {}
    for nname, (score, ok) in neural.items():
        for w_top, w_mid in ((0.0, 0.0), (0.25, 0.75), (0.25, 0.5), (0.25, 1.0)):
            if w_top == 0 and nname != "public":
                continue
            rr = []
            for mid in P.index.get_level_values(0).unique():
                s = score.loc[mid].to_numpy(float); keys = mk.loc[mid].to_numpy()
                active = bool(proc.loc[mid].iloc[0]) and bool(ok.loc[mid].all())
                order = leader_safe(s, w_top, w_mid) if active else np.arange(len(s))
                L = list(dict.fromkeys(keys[order]))[:25]
                t = truth.get(mid)
                rr.append(1 / (L.index(t) + 1) if t in L else 0.0)
            rows.append({"neural": nname if w_top else "baseline (V0)", "w_top": w_top, "w_mid": w_mid,
                         "mrr": float(np.mean(rr)), "n": len(rr)})
            per[(rows[-1]["neural"], w_top, w_mid)] = np.array(rr)
    R = pd.DataFrame(rows).drop_duplicates(["neural", "w_top", "w_mid"]).sort_values("mrr", ascending=False)
    print(R.to_string(index=False))
    rng = np.random.default_rng(0)
    for name in scorers:
        for wo in W_OURS:
            for w in (0.5, 0.75):
                lab = f"{1 - wo:g}pub+{wo:g}{name}"
                a_, b_ = per.get((lab, 0.25, w)), per.get(("public", 0.25, w))
                if a_ is None:
                    continue
                d = a_ - b_; bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(4000)]
                print(f"{lab} - public (w_mid {w}): {d.mean():+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, "
                      f"{np.percentile(bs, 97.5):+.4f}]  molecules up {(d > 0).sum()} down {(d < 0).sum()}")


if __name__ == "__main__":
    main()
