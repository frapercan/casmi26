"""Training data: spectra paired with molecule PLM embeddings / fingerprints.

Batches mix mass-local molecules (hard negatives: near-isobaric structures)
with random ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .model import ADDUCT_IDX


def ce_value(v) -> float:
    return float(v) if v == v else 30.0  # NaN -> typical CE


class SpectraStore:
    """Library spectra (from prep_library) + molecule features (from embed_molecules)."""

    def __init__(self, lib: str, emb: str):
        p = np.load(f"{lib}/peaks.npz")
        self.mz, self.it, self.off = p["mz"], p["it"], p["off"]
        self.spec = pd.read_parquet(f"{lib}/spectra.parquet")
        self.mols = pd.read_parquet(f"{emb}/mol_table.parquet")
        self.plm = np.load(f"{emb}/plm.npy", mmap_mode="r")
        self.fp = np.load(f"{emb}/fp.npy", mmap_mode="r")
        tr = self.mols[self.mols.source == "train"]
        self.row_of_ik = pd.Series(tr.index.values, index=tr.inchikey14)
        self.spec["mol_row"] = self.spec.inchikey14.map(self.row_of_ik).astype("Int64")
        self.spec_by_row = self.spec.groupby("mol_row").indices

    def collate(self, spec_idx: np.ndarray, max_peaks: int = 64, device="cpu") -> dict:
        B = len(spec_idx)
        mz = np.zeros((B, max_peaks), np.float32)
        it = np.zeros((B, max_peaks), np.float32)
        pad = np.ones((B, max_peaks), bool)
        for b, s in enumerate(spec_idx):
            a, e = self.off[s], min(self.off[s + 1], self.off[s] + max_peaks)
            n = e - a
            mz[b, :n], it[b, :n], pad[b, :n] = self.mz[a:e], self.it[a:e], False
        sp = self.spec.iloc[spec_idx]
        rows = sp.mol_row.to_numpy(dtype=np.int64)
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)
        return {
            "mz": t(mz), "it": t(it), "pad": t(pad, torch.bool),
            "prec": t(sp.precursor_mz.to_numpy()),
            "adduct": t(sp.adduct.map(ADDUCT_IDX).fillna(0).to_numpy(), torch.long),
            "mode": t((sp.ionization_mode == "positive").to_numpy(), torch.long),
            "ce": t(sp.ce_mean.map(ce_value).to_numpy()),
            "mol_id": t(rows, torch.long),
            "plm": t(self.plm[rows].astype(np.float32)),
            "fp": t(np.unpackbits(self.fp[rows], axis=1).astype(np.float32)),
        }


class MassLocalSampler:
    """Yields spectrum-index batches: half from a narrow mass window, half random."""

    def __init__(self, store: SpectraStore, mol_rows: np.ndarray, batch: int = 256, seed: int = 0):
        self.s, self.batch = store, batch
        m = store.mols.exact_mass.to_numpy()[mol_rows]
        self.rows = mol_rows[np.argsort(m)]
        self.rng = np.random.default_rng(seed)

    def one_spec(self, row):
        idx = self.s.spec_by_row[row]
        return idx[self.rng.integers(len(idx))]

    def __iter__(self):
        n, half = len(self.rows), self.batch // 2
        starts = self.rng.permutation(np.arange(0, n - half, half))
        for st in starts:
            local = self.rows[st:st + half]
            rnd = self.rows[self.rng.integers(n, size=self.batch - half)]
            rows = np.unique(np.concatenate([local, rnd]))
            yield np.array([self.one_spec(r) for r in rows])

    def __len__(self):
        return max(1, len(self.rows) // (self.batch // 2))
