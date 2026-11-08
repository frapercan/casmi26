# ── worst-case cost of inserting k generated candidates at position p, on cells where the truth is in a database ──
# Upper bound: every inserted candidate is wrong, so a truth at pipeline rank >= p moves down by k (out at > 25).
import glob
import numpy as np, pandas as pd
for f in sorted(glob.glob("/kaggle/input/**/val_rerank.parquet", recursive=True)):
    T = pd.read_parquet(f)
    src = f.split("/")[-3] if "/work/" not in f else f
    ranks = T[T.hit].groupby("molecule_id").base_rank.min().reindex(T.molecule_id.unique()).fillna(0).astype(int)
    rr = lambda r: np.where(r > 0, 1.0 / np.maximum(r, 1), 0.0)
    base = rr(ranks.to_numpy())
    print(f"\n{f}\n  molecules {len(ranks)} MRR {base.mean():.4f} in25 {(ranks > 0).mean():.3f}")
    for p in (3, 5, 10, 15, 20):
        for k in (1, 3, 5):
            r = ranks.to_numpy().copy()
            mv = r >= p
            r[mv] = r[mv] + k
            r[r > 25] = 0
            print(f"  ins p={p:<2d} k={k}  worst-case dMRR {rr(r).mean() - base.mean():+.4f}")
