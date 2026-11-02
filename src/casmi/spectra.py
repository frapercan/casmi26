"""Spectrum preprocessing and adduct mass arithmetic."""
from __future__ import annotations

import numpy as np

PROTON = 1.007276467
H2O = 18.010564684
# adduct -> (n_M, charge, mass shift added to n_M*M for the ion, before /|z|)
ADDUCTS: dict[str, tuple[int, int, float]] = {
    "[M+H]+": (1, 1, PROTON),
    "[M+NH4]+": (1, 1, 18.033823),
    "[M-H2O+H]+": (1, 1, PROTON - H2O),
    "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M+Na]+": (1, 1, 22.989218),
    "[M+K]+": (1, 1, 38.963158),
    "[M-H]-": (1, -1, -PROTON),
    "[M-H2O-H]-": (1, -1, -PROTON - H2O),
    "[M+CH2O2-H]-": (1, -1, 44.998201),
    "[M+Cl]-": (1, -1, 34.969402),
}
TEST_ADDUCTS = set(ADDUCTS)


def neutral_mass(precursor_mz: float, adduct: str) -> float:
    n, z, shift = ADDUCTS[adduct]
    return (precursor_mz * abs(z) - shift) / n


def clean_peaks(
    mzs,
    ints,
    precursor_mz: float,
    rel_floor: float = 0.005,
    top_n: int = 64,
    above_prec: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop low/above-precursor peaks, keep top_n, sqrt-scale, L2-normalise."""
    mz = np.asarray(mzs, dtype=np.float64)
    it = np.asarray(ints, dtype=np.float64)
    if mz.size == 0:
        return mz.astype(np.float32), it.astype(np.float32)
    keep = (it >= rel_floor * it.max()) & (mz <= precursor_mz + above_prec)
    mz, it = mz[keep], it[keep]
    if mz.size > top_n:
        idx = np.argpartition(-it, top_n)[:top_n]
        mz, it = mz[idx], it[idx]
    order = np.argsort(mz)
    mz, it = mz[order], np.sqrt(it[order])
    nrm = np.linalg.norm(it)
    if nrm > 0:
        it = it / nrm
    return mz.astype(np.float32), it.astype(np.float32)
