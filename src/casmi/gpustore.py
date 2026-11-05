"""GPU-resident training data: every batch is pure tensor indexing (no Python per item).

On a large GPU (B200/H100) the whole training set fits in device memory:
  peaks  1.95M x 64 x (mz fp32 + it fp16)  ~ 0.75 GB
  plm    1.7M  x 384 fp16                  ~ 1.3 GB
  fp     1.7M  x 256 uint8 (packed bits)   ~ 0.45 GB
so the CPU only launches kernels and the GPU never waits on collate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import SpectraStore

_BITS = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)


class GPUStore:
    def __init__(self, st: SpectraStore, device="cuda", max_peaks: int = 64, mol_rows: np.ndarray | None = None):
        self.dev, self.P = device, max_peaks
        S = len(st.off) - 1
        start = st.off[:-1]
        n = np.minimum(st.off[1:] - start, max_peaks)
        ar = np.arange(max_peaks)
        valid = ar[None, :] < n[:, None]
        gi = np.where(valid, start[:, None] + ar[None, :], 0)
        t = lambda x, dt: torch.as_tensor(np.ascontiguousarray(x)).to(device=device, dtype=dt)
        self.mz = t(np.where(valid, st.mz[gi], 0), torch.float32)
        self.it = t(np.where(valid, st.it[gi], 0), torch.float16)
        self.pad = t(~valid, torch.bool)
        del gi, valid
        self.prec = t(st.a_prec, torch.float32)
        self.add = t(st.a_add, torch.long)
        self.mode = t(st.a_mode, torch.long)
        self.ce = t(st.a_ce, torch.float32)
        self.row = t(st.a_row, torch.long)
        self.plm = t(st.plm, torch.float16)
        self.fp = t(st.fp, torch.uint8)
        self.bits = _BITS.to(device)
        # CSR of spectra per molecule row, restricted to the training rows
        rows = st.a_row
        keep = rows >= 0
        if mol_rows is not None:
            keep &= np.isin(rows, mol_rows)
        sid = np.flatnonzero(keep)
        o = np.argsort(rows[sid], kind="stable")
        sid = sid[o]
        r_sorted = rows[sid]
        uniq, first, cnt = np.unique(r_sorted, return_index=True, return_counts=True)
        mass = st.mols.exact_mass.to_numpy()[uniq]
        m_order = np.argsort(mass)  # molecules sorted by mass -> mass-local windows
        self.spec_ids = t(sid, torch.long)
        self.m_first = t(first[m_order], torch.long)
        self.m_cnt = t(cnt[m_order], torch.long)
        self.n_mol = len(uniq)
        self.n_spec = S
        # structure-database molecules sorted by mass: explicit same-mass negatives
        mols = st.mols
        db = np.flatnonzero((mols.source == "db").to_numpy())
        dm = mols.exact_mass.to_numpy()[db]
        o = np.argsort(dm)
        self.db_rows = t(db[o], torch.long)
        self.db_mass = t(dm[o], torch.float64)
        self.mol_mass = t(mols.exact_mass.to_numpy(), torch.float64)
        self.ik_id = t(pd.factorize(mols.inchikey14)[0], torch.long)

    def unpack_fp(self, rows):
        return ((self.fp[rows].unsqueeze(-1) & self.bits) > 0).flatten(1).to(torch.float32)

    def collate(self, sidx: torch.Tensor) -> dict:
        r = self.row[sidx].clamp(min=0)
        return {"mz": self.mz[sidx], "it": self.it[sidx].float(), "pad": self.pad[sidx],
                "prec": self.prec[sidx], "adduct": self.add[sidx], "mode": self.mode[sidx],
                "ce": self.ce[sidx], "mol_id": self.row[sidx],
                "plm": self.plm[r].float(), "fp": self.unpack_fp(r)}

    def hard_negatives(self, b: dict, k: int, gen: torch.Generator, ppm: float = 10.0, pool: int = 0) -> dict:
        """k database molecules within ppm of each spectrum's molecule (same structure masked out).
        pool > k: draw `pool` same-mass candidates and keep the k most similar to the true structure
        (Tanimoto on Morgan bits), i.e. the near-isomers a strong retrieval pipeline puts in its top-25."""
        r = b["mol_id"].clamp(min=0)
        m = self.mol_mass[r]
        tol = torch.clamp(m * ppm * 1e-6, min=0.003)
        lo = torch.searchsorted(self.db_mass, m - tol)
        hi = torch.searchsorted(self.db_mass, m + tol, right=True)
        n = (hi - lo).clamp(min=0)
        draw = max(k, pool)
        pick = lo.unsqueeze(1) + (torch.rand(len(r), draw, device=self.dev, generator=gen) * n.unsqueeze(1)).long()
        pick = pick.clamp(max=len(self.db_rows) - 1)
        rows = self.db_rows[pick]                                                   # (B,draw)
        ok = (n > 0).unsqueeze(1) & (self.ik_id[rows] != self.ik_id[r].unsqueeze(1))
        if draw > k:
            fq = b["fp"].half()                                                     # (B,F) true structure
            fc = self.unpack_fp(rows.flatten()).view(len(r), draw, -1).half()       # (B,draw,F)
            inter = torch.einsum("bf,bdf->bd", fq, fc).float()
            tani = inter / (fq.float().sum(-1, keepdim=True) + fc.float().sum(-1) - inter + 1e-6)
            tani = tani.masked_fill(~ok, -1.0)
            top = tani.topk(k, dim=1).indices
            rows, ok = rows.gather(1, top), ok.gather(1, top)
        b["neg_plm"] = self.plm[rows].float()
        b["neg_fp"] = self.unpack_fp(rows.flatten()).view(len(r), k, -1)
        b["neg_ok"] = ok
        return b

    def sample(self, batch: int, gen: torch.Generator, local_frac: float = 0.5) -> torch.Tensor:
        """One spectrum per molecule; local_frac of molecules from one contiguous mass window."""
        nl = int(batch * local_frac)
        s0 = torch.randint(0, max(1, self.n_mol - nl), (1,), device=self.dev, generator=gen)
        loc = s0 + torch.arange(nl, device=self.dev)
        rnd = torch.randint(0, self.n_mol, (batch - nl,), device=self.dev, generator=gen)
        m = torch.unique(torch.cat([loc, rnd]))
        pick = (torch.rand(len(m), device=self.dev, generator=gen) * self.m_cnt[m]).long()
        return self.spec_ids[self.m_first[m] + pick]

    def sample_spectra(self, batch: int, gen: torch.Generator, pool: torch.Tensor) -> torch.Tensor:
        """Uniform spectra from a pool (JEPA pretraining)."""
        return pool[torch.randint(0, len(pool), (batch,), device=self.dev, generator=gen)]

    def epoch_steps(self, batch: int, local_frac: float = 0.5) -> int:
        return max(1, int(self.n_mol / (batch * local_frac)))
