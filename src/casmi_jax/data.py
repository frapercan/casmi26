"""Device-resident training data for JAX (same content and sampling as casmi.gpustore.GPUStore).

Every array lives on the accelerator; a batch is pure gathers inside the jitted step, so the host only
passes integer indices. Shapes are fixed: one spectrum per sampled molecule, no de-duplication.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from casmi.data import SpectraStore


class JaxStore:
    def __init__(self, st: SpectraStore, max_peaks: int = 64, mol_rows: np.ndarray | None = None):
        start = st.off[:-1]
        n = np.minimum(st.off[1:] - start, max_peaks)
        ar = np.arange(max_peaks)
        valid = ar[None, :] < n[:, None]
        gi = np.where(valid, start[:, None] + ar[None, :], 0)
        self.a = {
            "mz": jnp.asarray(np.where(valid, st.mz[gi], 0), jnp.float32),
            "it": jnp.asarray(np.where(valid, st.it[gi], 0), jnp.bfloat16),
            "pad": jnp.asarray(~valid),
            "prec": jnp.asarray(st.a_prec, jnp.float32),
            "adduct": jnp.asarray(st.a_add, jnp.int32),
            "mode": jnp.asarray(st.a_mode, jnp.int32),
            "ce": jnp.asarray(st.a_ce, jnp.float32),
            "row": jnp.asarray(st.a_row, jnp.int32),
            "plm": jnp.asarray(st.plm, jnp.bfloat16),
            "fp": jnp.asarray(st.fp, jnp.uint8),
        }
        del gi, valid
        rows = st.a_row
        keep = rows >= 0
        if mol_rows is not None:
            keep &= np.isin(rows, mol_rows)
        sid = np.flatnonzero(keep)
        sid = sid[np.argsort(rows[sid], kind="stable")]
        uniq, first, cnt = np.unique(rows[sid], return_index=True, return_counts=True)
        o = np.argsort(st.mols.exact_mass.to_numpy()[uniq])        # molecules by mass -> mass-local windows
        self.a["spec_ids"] = jnp.asarray(sid, jnp.int32)
        self.a["m_first"] = jnp.asarray(first[o], jnp.int32)
        self.a["m_cnt"] = jnp.asarray(cnt[o], jnp.int32)
        mols = st.mols
        db = np.flatnonzero((mols.source == "db").to_numpy())
        dm = mols.exact_mass.to_numpy()[db]
        so = np.argsort(dm)
        # masses in mDa as int64 would need x64; float32 of (mass - 0) loses ~3e-5 Da at 500 Da: fine for a 10 ppm window
        self.a["db_rows"] = jnp.asarray(db[so], jnp.int32)
        self.a["db_mass"] = jnp.asarray(dm[so], jnp.float32)
        self.a["mol_mass"] = jnp.asarray(mols.exact_mass.to_numpy(), jnp.float32)
        self.a["ik_id"] = jnp.asarray(pd.factorize(mols.inchikey14)[0], jnp.int32)
        self.n_mol = len(uniq)

    def epoch_steps(self, batch: int, local_frac: float = 0.5) -> int:
        return max(1, int(self.n_mol / (batch * local_frac)))


def unpack_bits(x):
    bits = jnp.array([128, 64, 32, 16, 8, 4, 2, 1], jnp.uint8)
    return ((x[..., None] & bits) > 0).reshape(*x.shape[:-1], -1).astype(jnp.float32)


def sample_batch(a: dict, key, batch: int, local_frac: float, hard_k: int = 0, hard_ppm: float = 10.0) -> dict:
    """Jittable: one spectrum for each of `batch` molecules (local_frac from one mass window), plus
    hard_k database negatives within hard_ppm of each molecule."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n_mol = a["m_first"].shape[0]
    nl = int(batch * local_frac)
    s0 = jax.random.randint(k1, (), 0, max(1, n_mol - nl))
    m = jnp.concatenate([s0 + jnp.arange(nl), jax.random.randint(k2, (batch - nl,), 0, n_mol)])
    pick = (jax.random.uniform(k3, (batch,)) * a["m_cnt"][m]).astype(jnp.int32)
    sidx = a["spec_ids"][a["m_first"][m] + pick]
    r = jnp.maximum(a["row"][sidx], 0)
    b = {"mz": a["mz"][sidx], "it": a["it"][sidx].astype(jnp.float32), "pad": a["pad"][sidx],
         "prec": a["prec"][sidx], "adduct": a["adduct"][sidx], "mode": a["mode"][sidx], "ce": a["ce"][sidx],
         "mol_id": a["row"][sidx], "plm": a["plm"][r].astype(jnp.float32), "fp": unpack_bits(a["fp"][r])}
    if hard_k:
        mm = a["mol_mass"][r]
        tol = jnp.maximum(mm * hard_ppm * 1e-6, 0.003)
        lo = jnp.searchsorted(a["db_mass"], mm - tol)
        hi = jnp.searchsorted(a["db_mass"], mm + tol, side="right")
        n = jnp.maximum(hi - lo, 0)
        nb = (lo[:, None] + (jax.random.uniform(k4, (batch, hard_k)) * n[:, None]).astype(jnp.int32))
        rows = a["db_rows"][jnp.minimum(nb, a["db_rows"].shape[0] - 1)]
        b["neg_plm"] = a["plm"][rows].astype(jnp.float32)
        b["neg_fp"] = unpack_bits(a["fp"][rows])
        b["neg_ok"] = (n > 0)[:, None] & (a["ik_id"][rows] != a["ik_id"][r][:, None])
    return b
