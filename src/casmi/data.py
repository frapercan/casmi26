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

    def __init__(self, lib: str, emb: str, exclude: np.ndarray | None = None):
        """exclude: bool mask of spectra hidden from training (validation queries)."""
        p = np.load(f"{lib}/peaks.npz")
        self.mz, self.it, self.off = p["mz"], p["it"], p["off"]
        self.spec = pd.read_parquet(f"{lib}/spectra.parquet")
        self.mols = pd.read_parquet(f"{emb}/mol_table.parquet")
        self.plm = np.load(f"{emb}/plm.npy")  # in RAM: random gathers from mmap are slow
        self.fp = np.load(f"{emb}/fp.npy")
        tr = self.mols[self.mols.source == "train"]
        self.row_of_ik = pd.Series(tr.index.values, index=tr.inchikey14)
        self.spec["mol_row"] = self.spec.inchikey14.map(self.row_of_ik).astype("Int64")
        if exclude is not None and exclude.any():
            self.spec.loc[exclude, "mol_row"] = pd.NA
        self.spec_by_row = self.spec.groupby("mol_row").indices
        # per-spectrum numeric arrays so collate is pure numpy gathers
        sp = self.spec
        self.a_prec = sp.precursor_mz.to_numpy(np.float32)
        self.a_add = sp.adduct.map(ADDUCT_IDX).fillna(0).to_numpy(np.int64)
        self.a_mode = (sp.ionization_mode == "positive").to_numpy(np.int64)
        self.a_ce = sp.ce_mean.map(ce_value).to_numpy(np.float32)
        self.a_row = sp.mol_row.fillna(-1).to_numpy(np.int64)

    def collate(self, spec_idx: np.ndarray, max_peaks: int = 64, device="cpu") -> dict:
        spec_idx = np.asarray(spec_idx)
        start = self.off[spec_idx]
        n = np.minimum(self.off[spec_idx + 1] - start, max_peaks)
        ar = np.arange(max_peaks)
        valid = ar[None, :] < n[:, None]
        gi = np.where(valid, start[:, None] + ar[None, :], 0)
        mz = np.where(valid, self.mz[gi], 0).astype(np.float32)
        it = np.where(valid, self.it[gi], 0).astype(np.float32)
        rows = self.a_row[spec_idx]
        r = np.maximum(rows, 0)
        nb = device != "cpu"
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt).to(device, non_blocking=nb)
        return {
            "mz": t(mz), "it": t(it), "pad": t(~valid, torch.bool),
            "prec": t(self.a_prec[spec_idx]),
            "adduct": t(self.a_add[spec_idx], torch.long),
            "mode": t(self.a_mode[spec_idx], torch.long),
            "ce": t(self.a_ce[spec_idx]),
            "mol_id": t(rows, torch.long),
            "plm": t(np.asarray(self.plm[r], dtype=np.float32)),
            "fp": t(np.unpackbits(self.fp[r], axis=1).astype(np.float32)),
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
