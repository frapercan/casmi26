"""GPU-resident training data: every batch is pure tensor indexing (no Python per item).

On a large GPU (B200/H100) the whole training set fits in device memory:
  peaks  1.95M x 64 x (mz fp32 + it fp16)  ~ 0.75 GB
  plm    1.7M  x 384 fp16                  ~ 1.3 GB
  fp     1.7M  x 256 uint8 (packed bits)   ~ 0.45 GB
so the CPU only launches kernels and the GPU never waits on collate.
"""
from __future__ import annotations

import numpy as np
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

    def unpack_fp(self, rows):
        return ((self.fp[rows].unsqueeze(-1) & self.bits) > 0).flatten(1).to(torch.float32)

    def collate(self, sidx: torch.Tensor) -> dict:
        r = self.row[sidx].clamp(min=0)
        return {"mz": self.mz[sidx], "it": self.it[sidx].float(), "pad": self.pad[sidx],
                "prec": self.prec[sidx], "adduct": self.add[sidx], "mode": self.mode[sidx],
                "ce": self.ce[sidx], "mol_id": self.row[sidx],
                "plm": self.plm[r].float(), "fp": self.unpack_fp(r)}

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
