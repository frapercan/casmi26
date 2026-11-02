"""Spectral library search baseline.

Candidates = library molecules whose exact mass matches the query's neutral
mass (from precursor m/z + adduct). Each candidate is scored by spectral
similarity between the query molecule's spectra and the candidate's library
spectra, aggregated per molecule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd

from .spectra import clean_peaks, neutral_mass


@numba.njit(cache=True)
def cosine_greedy(mz1, it1, mz2, it2, tol):
    """matchms-style greedy cosine; inputs sorted by mz and L2-normalised."""
    n1, n2 = mz1.size, mz2.size
    if n1 == 0 or n2 == 0:
        return 0.0
    # collect candidate pairs within tolerance
    cap = n1 * 4 + 16
    sc = np.empty(cap, np.float64)
    pi = np.empty(cap, np.int64)
    pj = np.empty(cap, np.int64)
    k = 0
    lo = 0
    for i in range(n1):
        while lo < n2 and mz2[lo] < mz1[i] - tol:
            lo += 1
        j = lo
        while j < n2 and mz2[j] <= mz1[i] + tol:
            if k == cap:
                break
            sc[k] = it1[i] * it2[j]
            pi[k] = i
            pj[k] = j
            k += 1
            j += 1
    if k == 0:
        return 0.0
    order = np.argsort(-sc[:k])
    u1 = np.zeros(n1, np.bool_)
    u2 = np.zeros(n2, np.bool_)
    s = 0.0
    for o in order:
        a, b = pi[o], pj[o]
        if not u1[a] and not u2[b]:
            u1[a] = True
            u2[b] = True
            s += sc[o]
    return s


@dataclass
class Library:
    spectra: pd.DataFrame      # per-spectrum meta, row i <-> peaks[off[i]:off[i+1]]
    molecules: pd.DataFrame    # inchikey14, smiles, formula, exact_mass
    mz: np.ndarray
    it: np.ndarray
    off: np.ndarray

    @classmethod
    def load(cls, path: str) -> "Library":
        p = np.load(f"{path}/peaks.npz")
        return cls(pd.read_parquet(f"{path}/spectra.parquet"),
                   pd.read_parquet(f"{path}/molecules.parquet"),
                   p["mz"], p["it"], p["off"])

    def subset(self, spec_mask: np.ndarray) -> "Library":
        """Restrict to a subset of spectra (and the molecules they cover)."""
        idx = np.flatnonzero(spec_mask)
        spectra = self.spectra.iloc[idx].reset_index(drop=True)
        lens = self.off[idx + 1] - self.off[idx]
        off = np.concatenate([[0], np.cumsum(lens)])
        # vectorised gather of each kept spectrum's peak range
        sel = np.repeat(self.off[idx] - off[:-1], lens) + np.arange(off[-1])
        mols = self.molecules[self.molecules.inchikey14.isin(set(spectra.inchikey14))].reset_index(drop=True)
        return Library(spectra, mols, self.mz[sel], self.it[sel], off)

    def build_index(self):
        m = self.molecules.sort_values("exact_mass").reset_index(drop=True)
        self.molecules = m
        self._mass = m.exact_mass.to_numpy()
        self._spec_by_mol = self.spectra.groupby("inchikey14").indices
        self._mode = self.spectra.ionization_mode.to_numpy()
        self._adduct = self.spectra.adduct.to_numpy()

    def mass_candidates(self, mass: float, ppm: float = 10.0, min_da: float = 0.003) -> np.ndarray:
        tol = max(mass * ppm * 1e-6, min_da)
        lo = np.searchsorted(self._mass, mass - tol)
        hi = np.searchsorted(self._mass, mass + tol, side="right")
        return np.arange(lo, hi)

    def peaks(self, i):
        return self.mz[self.off[i]:self.off[i + 1]], self.it[self.off[i]:self.off[i + 1]]


def prep_query(df: pd.DataFrame, top_n: int = 64) -> pd.DataFrame:
    """Clean raw query spectra (test.parquet schema)."""
    out = df[["molecule_id", "spectrum_id", "adduct", "ionization_mode", "precursor_mz"]].copy()
    cl = [clean_peaks(m, i, p, top_n=top_n) for m, i, p in
          zip(df.ms2_mzs, df.ms2_normalized_intensities, df.precursor_mz)]
    out["mz"] = [c[0].astype(np.float64) for c in cl]
    out["it"] = [c[1].astype(np.float64) for c in cl]
    out["mass"] = [neutral_mass(p, a) for p, a in zip(df.precursor_mz, df.adduct)]
    return out


def search_molecule(lib: Library, q: pd.DataFrame, ppm: float = 10.0, tol: float = 0.01,
                    k: int = 25, same_adduct_bonus: float = 0.05) -> list[tuple[str, str, float]]:
    """Rank library molecules for one query molecule (all its spectra)."""
    mass = float(np.median(q.mass))
    cands = lib.mass_candidates(mass, ppm)
    if cands.size == 0:
        return []
    res = []
    for ci in cands:
        row = lib.molecules.iloc[ci]
        sidx = lib._spec_by_mol.get(row.inchikey14, [])
        tot = 0.0
        for qm, qi, qmode, qadd in zip(q.mz, q.it, q.ionization_mode, q.adduct):
            best = 0.0
            for s in sidx:
                if lib._mode[s] != qmode:
                    continue
                rm, ri = lib.peaks(s)
                v = cosine_greedy(qm, qi, rm.astype(np.float64), ri.astype(np.float64), tol)
                if lib._adduct[s] == qadd:
                    v += same_adduct_bonus
                if v > best:
                    best = v
            tot += best
        res.append((row.inchikey14, row.smiles, tot / len(q)))
    res.sort(key=lambda r: -r[2])
    return res[:k]
