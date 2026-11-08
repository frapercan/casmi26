# ── casmi26 honest evaluation, class 3: natural products whose structure is in no library ──
# Queries = the class-3 cell built by scripts/make_class3.py (private dataset casmi26-class3-data). Every
# train.parquet row of these structures is removed from the pipeline's library. External candidate
# databases are left as they are, so a list can still contain the truth through them; the analysis
# (hybrid/class3_insert_cell.py) treats the truth as absent from the pipeline's pool, as in true class 3.
import glob, os, tarfile
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.parquet as pq
VC = "/kaggle/working/valcomp"
os.makedirs(VC, exist_ok=True)
_q3 = glob.glob("/kaggle/input/**/casmi26-class3-data/**/queries.parquet", recursive=True)
if _q3:
    D3 = os.path.dirname(_q3[0])
else:
    D3 = "/kaggle/tmp_c3"
    tarfile.open(glob.glob("/kaggle/input/**/c3data.tgz", recursive=True)[0]).extractall(D3)
_real = os.path.dirname(glob.glob("/kaggle/input/**/enveda-CASMI26-molecule-id-mass-spectra/test.parquet", recursive=True)[0])
_vis = pd.read_parquet(f"{_real}/test.parquet")
_Q = pd.read_parquet(f"{D3}/queries.parquet")
_A = pd.read_parquet(f"{D3}/answers.parquet")
# shaped like the hidden test (as ext_pre_cell.py): only adducts the test uses, and every molecule has an
# [M+H]+ or [M-H]- spectrum (the pipeline's engine 2 does not cover molecules seen only as adducts)
_Q = _Q[_Q.adduct.isin(set(_vis.adduct))]
_Q = _Q[_Q.molecule_id.isin(set(_Q.molecule_id[_Q.adduct.isin(["[M+H]+", "[M-H]-"])]))]
# engine 2 silently skips a molecule with no pool structure within 30 ppm (then cell 48 asserts); these three
# have none (hybrid/c3_pool_diag_cell.py, casmi26-c3-pooldiag v2)
_Q = _Q[~_Q.molecule_id.isin(["val3_0052", "val3_0109", "val3_0122"])]
_A = _A[_A.molecule_id.isin(set(_Q.molecule_id))].reset_index(drop=True)
_rng = np.random.default_rng(0)
_mid = dict(zip(_A.molecule_id, [f"m_{x:06x}" for x in _rng.choice(16 ** 6, len(_A), replace=False)]))
_Q["molecule_id"] = _Q.molecule_id.map(_mid)
_Q["spectrum_id"] = [f"s_{x:08x}" for x in _rng.choice(16 ** 8, len(_Q), replace=False)]
_Q = _Q.sort_values("molecule_id", kind="stable")
_Q[list(_vis.columns)].astype(_vis.dtypes.to_dict()).to_parquet(f"{VC}/test.parquet", index=False)
_A["val3_id"] = _A.molecule_id
_A["molecule_id"] = _A.molecule_id.map(_mid)
_A.to_parquet("/kaggle/working/val_answers.parquet", index=False)
_tmpl = pd.read_csv(f"{_real}/sample_submission.csv", dtype=str).smiles.iloc[0]
pd.DataFrame({"molecule_id": list(dict.fromkeys(_Q.molecule_id)), "smiles": _tmpl}).to_csv(f"{VC}/sample_submission.csv", index=False)
_excl = sorted(set(open(f"{D3}/exclude_all_ik.txt").read().split()))
if not os.path.exists(f"{VC}/train.parquet"):
    _d = ds.dataset(f"{_real}/train.parquet")
    _n = 0
    with pq.ParquetWriter(f"{VC}/train.parquet", _d.schema) as _w:
        for _b in _d.to_batches(filter=~ds.field("inchikey14").isin(_excl), batch_size=65536):
            _w.write_batch(_b); _n += _b.num_rows
    print(f"VALCOMP train rows kept {_n}")
print(f"VALCOMP class 3: {len(_A)} molecules, {len(_Q)} spectra, {len(_excl)} structures hidden from every library row")
