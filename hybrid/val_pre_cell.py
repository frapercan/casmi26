# ── casmi26 validation harness: point the public pipeline at our hidden-test-like validation ──
# Builds /kaggle/working/valcomp = {test.parquet: validation queries, train.parquet: train minus
# the enveda-180 spectra of every validation molecule, sample_submission.csv}. The pipeline's
# path discovery is patched to read it instead of the competition folder.
import glob, os
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.parquet as pq
VC = "/kaggle/working/valcomp"
os.makedirs(VC, exist_ok=True)
N_CLS = __N_CLS__  # molecules per class taken from the validation set
_vals = sorted(glob.glob("/kaggle/input/**/work/val/answers.parquet", recursive=True))
_vals = [p for p in _vals if "casmi26-train-v3" in p] or _vals
VALDIR = os.path.dirname(_vals[0])
_A = pd.read_parquet(f"{VALDIR}/answers.parquet")
_Q = pd.read_parquet(f"{VALDIR}/queries.parquet")
_A = pd.concat([_A[_A.cls == c].head(n) for c, n in N_CLS.items()], ignore_index=True)
_rng = np.random.default_rng(0)
_mid = dict(zip(_A.molecule_id, [f"m_{x:06x}" for x in _rng.choice(16 ** 6, len(_A), replace=False)]))
_Q = _Q[_Q.molecule_id.isin(_mid)].copy()
_Q["molecule_id"] = _Q.molecule_id.map(_mid)
_Q["spectrum_id"] = [f"s_{x:08x}" for x in _rng.choice(16 ** 8, len(_Q), replace=False)]
_A["molecule_id"] = _A.molecule_id.map(_mid)
_Q.to_parquet(f"{VC}/test.parquet", index=False)
_A.to_parquet("/kaggle/working/val_answers.parquet", index=False)
_real = os.path.dirname(glob.glob("/kaggle/input/**/enveda-CASMI26-molecule-id-mass-spectra/train.parquet", recursive=True)[0])
_tmpl = pd.read_csv(f"{_real}/sample_submission.csv", dtype=str).smiles.iloc[0]
pd.DataFrame({"molecule_id": list(dict.fromkeys(_Q.molecule_id)), "smiles": _tmpl}).to_csv(f"{VC}/sample_submission.csv", index=False)
_excl = sorted(set(open(f"{VALDIR}/exclude_ik.txt").read().split()))
if not os.path.exists(f"{VC}/train.parquet"):
    _d = ds.dataset(f"{_real}/train.parquet")
    _flt = ~((ds.field("ingest_lib") == "enveda-180") & ds.field("inchikey14").isin(_excl))
    _n = 0
    with pq.ParquetWriter(f"{VC}/train.parquet", _d.schema) as _w:
        for _b in _d.to_batches(filter=_flt, batch_size=65536):
            _w.write_batch(_b); _n += _b.num_rows
    print(f"VALCOMP train rows kept {_n}")
print(f"VALCOMP {len(_A)} molecules {dict(_A.cls.value_counts())}, {len(_Q)} spectra, "
      f"{len(_excl)} molecules hidden from the library, from {VALDIR}")
