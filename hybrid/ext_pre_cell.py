# ── casmi26 honest evaluation: queries = external spectra of structures absent from train.parquet ──
# No model or prior of the public pipeline has seen these structures (they are not in train), so its
# score here is not inflated by leakage. Source: Enveda-180 Zenodo release (e180_extra.parquet, built
# by hybrid/e180_cell.py). The library is the real train, unchanged.
import glob, os
import numpy as np, pandas as pd
VC = "/kaggle/working/valcomp"
os.makedirs(VC, exist_ok=True)
N_MOL = __N_MOL__
_real = os.path.dirname(glob.glob("/kaggle/input/**/enveda-CASMI26-molecule-id-mass-spectra/test.parquet", recursive=True)[0])
_vis = pd.read_parquet(f"{_real}/test.parquet")
_X = pd.read_parquet(glob.glob("/kaggle/input/**/e180_extra.parquet", recursive=True)[0])
_X = _X[_X.adduct.isin(set(_vis.adduct))]                    # only adducts the test uses
_X = _X[_X.inchikey14.isin(set(_X.inchikey14[_X.adduct.isin(["[M+H]+", "[M-H]-"])]))]  # like the test: mostly protonated/deprotonated
_rng = np.random.default_rng(0)
_iks = sorted(set(_X.inchikey14))
_iks = list(_rng.choice(_iks, min(N_MOL, len(_iks)), replace=False))
_X = _X[_X.inchikey14.isin(_iks)].copy()
_mid = dict(zip(_iks, [f"m_{x:06x}" for x in _rng.choice(16 ** 6, len(_iks), replace=False)]))
_X["molecule_id"] = _X.inchikey14.map(_mid)
_X["spectrum_id"] = [f"s_{x:08x}" for x in _rng.choice(16 ** 8, len(_X), replace=False)]
_X = _X.sample(frac=1, random_state=0).sort_values("molecule_id", kind="stable")
# the hidden test's positive-mode m/z were corrected by -0.000548 Da (one electron, hosts' note); train and the
# Zenodo release were not, so apply it here or every validation query sits ~+1.6 ppm off the hidden test
_pos = _X.ionization_mode == "positive"
_X.loc[_pos, "precursor_mz"] = _X.loc[_pos, "precursor_mz"] - 0.000548
_X.loc[_pos, "ms2_mzs"] = _X.loc[_pos, "ms2_mzs"].map(lambda m: (np.asarray(m, float) - 0.000548).tolist())
_X[list(_vis.columns)].astype(_vis.dtypes.to_dict()).to_parquet(f"{VC}/test.parquet", index=False)
_A = _X.drop_duplicates("molecule_id")[["molecule_id", "normalized_smiles"]].rename(columns={"normalized_smiles": "smiles"})
_A["cls"] = 0
_A.to_parquet("/kaggle/working/val_answers.parquet", index=False)
_tmpl = pd.read_csv(f"{_real}/sample_submission.csv", dtype=str).smiles.iloc[0]
pd.DataFrame({"molecule_id": list(dict.fromkeys(_X.molecule_id)), "smiles": _tmpl}).to_csv(f"{VC}/sample_submission.csv", index=False)
if not os.path.exists(f"{VC}/train.parquet"):
    os.symlink(f"{_real}/train.parquet", f"{VC}/train.parquet")
print(f"VALCOMP external: {len(_A)} molecules, {len(_X)} spectra, adducts {_X.adduct.value_counts().to_dict()}")
