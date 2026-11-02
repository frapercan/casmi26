# CASMI 2026 (Enveda × Kaggle)

Molecule identification from MS/MS spectra. Metric: MRR@25 on InChIKey14 (tautomer-canonical).

## Approach

```
spectra ─► SpectrumEncoder (peak Transformer; JEPA-pretrained on unlabeled spectra)
               │
     ┌─────────┼───────────────┬──────────────┐
  proj_s     pred            fp head       SAE head
 (InfoNCE)  (≈ PLM latent)  (Morgan bits)  (top-k sparse codes of PLM latent)
     │         │               │              │
     └──── scored against candidates' frozen PLM embedding (ChemBERTa-77M-MTR) ───┘

candidates = library molecules (spectral match L) ∪ external DB (COCONUT + PubChem subset + PubChemLite)
final rank = fuse(model scores, L, popularity)   [weights tuned on class-1 + class-2 validation]
```

- **Class 1** (structure has public spectra): spectral library search.
- **Class 2** (in PubChem/COCONUT, no spectra): rank mass-matched DB candidates with the model.
- **Class 3** (novel): not addressed yet (de novo generation is future work).

## Layout
- `src/casmi/metric.py` – MRR@25 replicating the competition metric
- `src/casmi/spectra.py` – adduct arithmetic, peak cleaning
- `src/casmi/search.py` – spectral library search (mass filter + greedy cosine, numba)
- `src/casmi/model.py` – spectrum encoder + retrieval / latent-prediction / FP / SAE heads, losses
- `src/casmi/jepa.py` – JEPA self-supervised pretraining (masked m/z blocks, EMA target)
- `src/casmi/sae.py` – top-k sparse autoencoder over PLM embeddings
- `src/casmi/plm.py` – frozen chemical LM embeddings
- `src/casmi/data.py` – training store and mass-local hard-negative sampler
- `src/casmi/predict.py` – end-to-end candidate table + fusion
- `scripts/` – prep_library, build_candidates, make_split, embed_molecules, train_sae,
  pretrain_jepa, train, eval_fusion, predict_submission, validate_search, eval_prior
- `tools/kaggle_run.py` – push/monitor private Kaggle GPU kernels running the pipeline

## Validation
- class 1: `enveda-np-examples` (250 timsTOF molecules) vs all other libraries
- class 2: 1,500 held-out NP-library molecules (`splits/holdout.parquet`, not committed:
  derived from competition data), removed from library and training, injected into the DB pool.
  Median 628 candidates within 10 ppm.

## Results
| Method | Validation | MRR@25 |
|---|---|---|
| Library search (10 ppm, cosine tol 0.01) | class 1 | 0.924 |
| Random | class 2 | 0.023 |
| PubChemLite popularity prior | class 2 | 0.037 |

Competition data is CC BY-NC 4.0 and must not be committed.
