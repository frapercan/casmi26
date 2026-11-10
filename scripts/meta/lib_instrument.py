"""Instrument class of every library spectrum, row-aligned with work/lib/spectra.parquet.

scripts/prep_library.py dropped train.parquet's instrument_type. This replays its row filter exactly (test adducts,
|precursor error| <= 20 ppm, >= 3 peaks after clean_peaks with top_n 64) and verifies the alignment row by row
(ingest_lib, inchikey14, precursor_mz) before writing work/lib/instrument.parquet.

Classes: tims (timsTOF), orbitrap (Orbitrap / FT / QFT / ITFT / HCD), qtof (QTOF / TOF), trap (ion trap, QQQ / QQ),
other.

  python scripts/meta/lib_instrument.py
"""
from __future__ import annotations

import argparse, os, re, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from casmi.spectra import TEST_ADDUCTS, clean_peaks  # noqa: E402

TRAIN = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
CLASSES = ["tims", "orbitrap", "qtof", "trap", "other"]


def inst_class(s):
    s = str(s).lower()
    if "tims" in s:
        return "tims"
    if re.search(r"orbitrap|q-?ft|itft|hybrid ft|\bft\b|hcd|q exactive|exploris|fusion|lumos", s):
        return "orbitrap"
    if re.search(r"q-?tof|\btof\b|tof|maxis|impact|synapt|xevo|tripletof", s):
        return "qtof"
    if re.search(r"trap|\bit\b|qqq|\bqq\b|qit|lc-esi-it|triple quad", s):
        return "trap"
    return "other"


def rg_keep(args):
    path, rg, max_ppm, min_peaks, top_n = args
    t = pq.ParquetFile(path).read_row_group(rg, columns=["ingest_lib", "inchikey14", "adduct", "precursor_mz",
                                                         "precursor_error_ppm", "instrument_type", "ms2_mzs",
                                                         "ms2_normalized_intensities"]).to_pandas()
    ok = t.adduct.isin(TEST_ADDUCTS) & (t.precursor_error_ppm.abs() <= max_ppm)
    t = t[ok]
    keep = np.zeros(len(t), dtype=bool)
    for i, (mz, it, pm) in enumerate(zip(t.ms2_mzs, t.ms2_normalized_intensities, t.precursor_mz)):
        cm, _ = clean_peaks(mz, it, pm, top_n=top_n)
        keep[i] = cm.size >= min_peaks
    return rg, t.loc[keep, ["ingest_lib", "inchikey14", "precursor_mz", "instrument_type"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--lib", default="work/lib")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    n_rg = pq.ParquetFile(a.train).num_row_groups
    with Pool(a.workers) as P:
        parts = dict(P.imap_unordered(rg_keep, [(a.train, rg, 20.0, 3, 64) for rg in range(n_rg)]))
    X = pd.concat([parts[rg] for rg in range(n_rg)], ignore_index=True)
    S = pd.read_parquet(f"{a.lib}/spectra.parquet", columns=["ingest_lib", "inchikey14", "precursor_mz"])
    assert len(X) == len(S), f"row count differs: replay {len(X)} vs library {len(S)}"
    for c in ("ingest_lib", "inchikey14"):
        assert (X[c].astype(str).to_numpy() == S[c].astype(str).to_numpy()).all(), f"{c} misaligned"
    assert np.allclose(X.precursor_mz.to_numpy(float), S.precursor_mz.to_numpy(float), atol=1e-6), "precursor misaligned"
    X["instrument"] = X.instrument_type.map(inst_class)
    X[["instrument_type", "instrument"]].to_parquet(f"{a.lib}/instrument.parquet", index=False)
    print("aligned with", len(S), "library spectra; classes:", X.instrument.value_counts().to_dict())
    print(pd.crosstab(X.ingest_lib, X.instrument).to_string())


if __name__ == "__main__":
    main()
