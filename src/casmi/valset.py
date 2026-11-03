"""Hidden-test-like validation set (built by scripts/make_valset.py): shared helpers."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

VAL_LIB = "enveda-180"


def load_exclude(val_dir: str) -> set[str]:
    p = os.path.join(val_dir, "exclude_ik.txt")
    return set(open(p).read().split()) if val_dir and os.path.exists(p) else set()


def excluded_spectra(spec: pd.DataFrame, exclude_ik: set[str]) -> np.ndarray:
    """Bool mask of library spectra hidden by the validation set: the enveda-180 spectra of
    validation molecules (class-1 molecules keep their spectra from other libraries)."""
    if not exclude_ik:
        return np.zeros(len(spec), bool)
    return ((spec.ingest_lib == VAL_LIB) & spec.inchikey14.isin(exclude_ik)).to_numpy()
