"""Estimate the test's class-1 fraction without labels.

Class 1 = the structure has public reference spectra (outside Enveda-180). For every test molecule and for
validation molecules of known class we compute label-free library-match features; the test's feature
distribution is then fitted as a mixture of the validation class-1 and non-class-1 distributions.

Validation molecules are Enveda-180 timsTOF molecules in the test's mass range. Their own Enveda-180
spectra are hidden from the library while they are searched (as a test molecule's are), everything else stays.

  python scripts/class_mix.py --lib work/lib --out work/classmix --n-val 3000
"""
from __future__ import annotations

import argparse, os, sys, time
from multiprocessing import get_context

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.search import Library, cosine_greedy, prep_query  # noqa: E402

COMP = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra"
VAL_LIB = "enveda-180"
LIB = None


def features(args):
    """Library-match features of one query molecule; hide_ik's Enveda-180 spectra are skipped."""
    mid, q, hide_ik = args
    L = LIB
    mass = float(np.median(q.mass))
    cands = L.mass_candidates(mass, 10.0)
    scores, src = [], []
    for ci in cands:
        ik = L.molecules.inchikey14.iat[ci]
        sidx = L._spec_by_mol.get(ik, [])
        tot, best_src = 0.0, ""
        for qm, qi, qmode in zip(q.mz, q.it, q.ionization_mode):
            best = 0.0
            for s in sidx:
                if L._mode[s] != qmode or (ik == hide_ik and L._lib[s] == VAL_LIB):
                    continue
                rm, ri = L.peaks(s)
                v = cosine_greedy(qm, qi, rm.astype(np.float64), ri.astype(np.float64), 0.01)
                if v > best:
                    best, best_src = v, L._lib[s]
            tot += best
        scores.append(tot / len(q)); src.append(best_src)
    o = np.argsort(scores)[::-1]
    s = np.asarray(scores)[o] if len(o) else np.zeros(0)
    true_rank = next((r + 1 for r, j in enumerate(o) if L.molecules.inchikey14.iat[cands[j]] == hide_ik), 0) if hide_ik else 0
    return dict(molecule_id=mid, n_cand=len(cands), top1=s[0] if len(s) else 0.0, top2=s[1] if len(s) > 1 else 0.0,
                top1_src=src[o[0]] if len(o) else "", n_spec=len(q), mass=mass, true_rank=true_rank)


def val_queries(lib, test, n_val, seed):
    """Enveda-180 molecules in the test's mass window, spectra per molecule drawn like the test's."""
    rng = np.random.default_rng(seed)
    sp = lib.spectra
    m = lib.molecules.set_index("inchikey14").exact_mass
    lo, hi = test.mass.min() - 1, test.mass.max() + 1
    env = sp.loc[sp.ingest_lib == VAL_LIB, "inchikey14"].unique()
    env = [k for k in env if k in m.index and lo <= m[k] <= hi]
    pick = set(rng.choice(env, size=min(n_val, len(env)), replace=False))
    t = ds.dataset(f"{COMP}/train.parquet").to_table(
        columns=["inchikey14", "ingest_lib", "ms2_mzs", "ms2_normalized_intensities", "adduct", "ionization_mode",
                 "precursor_mz", "instrument_type"],
        filter=(ds.field("ingest_lib") == VAL_LIB) & ds.field("inchikey14").isin(list(pick))).to_pandas()
    t = t[t.adduct.isin(set(test.adduct))]
    want = test.groupby("molecule_id").size().to_numpy()
    parts = []
    for ik, g in t.groupby("inchikey14"):
        k = min(len(g), int(rng.choice(want)))
        parts.append(g.sample(k, random_state=int(rng.integers(1 << 30))).assign(molecule_id="v_" + ik))
    v = pd.concat(parts, ignore_index=True)
    v["spectrum_id"] = np.arange(len(v)).astype(str)
    return v


def mixture(test_x, x1, x0, n_boot=200, seed=0):
    """ML class-1 fraction from binned densities; bootstrap over test molecules and validation sets."""
    edges = [np.quantile(np.concatenate([x1[:, j], x0[:, j]]), np.linspace(0, 1, 9)) for j in range(x1.shape[1])]

    def bins(x):
        return np.ravel_multi_index(tuple(np.clip(np.searchsorted(e[1:-1], x[:, j]), 0, 7) for j, e in enumerate(edges)),
                                    (8,) * x.shape[1])

    def fit(t, a, b):
        nb = 8 ** x1.shape[1]
        f1 = (np.bincount(bins(a), minlength=nb) + 0.5) / (len(a) + 0.5 * nb)
        f0 = (np.bincount(bins(b), minlength=nb) + 0.5) / (len(b) + 0.5 * nb)
        bt = bins(t); grid = np.linspace(0, 1, 1001)
        ll = [np.log(p * f1[bt] + (1 - p) * f0[bt]).sum() for p in grid]
        return grid[int(np.argmax(ll))]

    rng = np.random.default_rng(seed)
    est = fit(test_x, x1, x0)
    bs = [fit(test_x[rng.integers(0, len(test_x), len(test_x))], x1[rng.integers(0, len(x1), len(x1))],
              x0[rng.integers(0, len(x0), len(x0))]) for _ in range(n_boot)]
    return est, np.quantile(bs, [0.025, 0.975])


def main():
    global LIB
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="work/lib")
    ap.add_argument("--out", default="work/classmix")
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    LIB = Library.load(a.lib); LIB.build_index(); LIB._lib = LIB.spectra.ingest_lib.to_numpy()
    other = set(LIB.spectra.loc[LIB.spectra.ingest_lib != VAL_LIB, "inchikey14"])
    test = prep_query(pd.read_parquet(f"{COMP}/test.parquet"))
    val = prep_query(val_queries(LIB, test, a.n_val, a.seed))
    print(f"library {len(LIB.spectra)} spectra | test {test.molecule_id.nunique()} | val {val.molecule_id.nunique()} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    jobs = [(m, g, "") for m, g in test.groupby("molecule_id", sort=False)] + \
           [(m, g, m[2:]) for m, g in val.groupby("molecule_id", sort=False)]
    rows = []
    with get_context("fork").Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(features, jobs, chunksize=4)):
            rows.append(r)
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(jobs)} [{time.time() - t0:.0f}s]", flush=True)
    F = pd.DataFrame(rows)
    F["set"] = np.where(F.molecule_id.str.startswith("v_"), "val", "test")
    F["cls1"] = F.molecule_id.str[2:].isin(other) & (F.set == "val")
    F["gap"] = F.top1 - F.top2
    F.to_parquet(f"{a.out}/features.parquet", index=False)
    V, T = F[F.set == "val"], F[F.set == "test"]
    print(f"\nval class-1 share among Enveda-180 molecules in the test mass window: {V.cls1.mean():.3f} (n={len(V)})")
    print("val class-1: true structure at library rank 1 in "
          f"{(V[V.cls1].true_rank == 1).mean():.3f}, in top 25 {V[V.cls1].true_rank.between(1, 25).mean():.3f}")
    for name, g in (("test", T), ("val class 1", V[V.cls1]), ("val not class 1", V[~V.cls1])):
        print(f"{name:16s} top1 q25/50/75 {np.quantile(g.top1, [.25, .5, .75]).round(3)} | top1>0.7 {(g.top1 > .7).mean():.3f} "
              f"| gap median {g.gap.median():.3f} | n_cand median {g.n_cand.median():.0f}")
    for cols in (["top1"], ["top1", "gap"]):
        est, ci = mixture(T[cols].to_numpy(), V[V.cls1][cols].to_numpy(), V[~V.cls1][cols].to_numpy())
        print(f"class-1 fraction of the test from {cols}: {est:.3f}  (95% CI {ci[0]:.3f}-{ci[1]:.3f})")
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
