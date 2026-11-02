# CASMI 2026 (Enveda × Kaggle)

Molecule identification from MS/MS spectra. Metric: MRR@25 on InChIKey14.

## Layout
- `src/casmi/metric.py` – MRR@25 replicating the competition metric (tautomer canonicalisation + InChIKey14)
- `src/casmi/spectra.py` – adduct arithmetic, peak cleaning
- `src/casmi/search.py` – spectral library search (mass filter + greedy cosine, numba)
- `scripts/prep_library.py` – stream `train.parquet` into a compact library (`molecules/spectra.parquet`, `peaks.npz`)
- `scripts/validate_search.py` – validation: `enveda-np-examples` (timsTOF) queries vs all other libraries
- `scripts/make_submission.py` – library-search `submission.csv`

## Results
| Method | Validation | MRR@25 | top1 |
|---|---|---|---|
| Library search (10 ppm, cosine tol 0.01) | np-examples vs other libs (class 1) | 0.924 | 0.884 |

Competition data is CC BY-NC 4.0 and must not be committed.
