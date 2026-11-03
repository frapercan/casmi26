# ── Enveda-180 (Zenodo 21346580, CC-BY-4.0; host-approved, thread 745148) -> extra library rows ──
# Downloads the unfiltered release, keeps spectra of structures absent from train.parquet whose
# adduct is in train's adduct vocabulary, and writes them in train.parquet's schema to
# /kaggle/working/e180_extra.parquet (intensities normalised to sum 1, as train's enveda-180 rows).
import glob, os, subprocess, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
t0 = time.time()
URL = "https://zenodo.org/records/21346580/files/enveda-180.parquet.gz?download=1"
RAW = "/tmp/e180/enveda-180.parquet"
os.makedirs("/tmp/e180", exist_ok=True)
if not os.path.exists(RAW):
    subprocess.run(f"curl -sSL --retry 5 '{URL}' | gunzip -c > {RAW}", shell=True, check=True)
print(f"downloaded {os.path.getsize(RAW) / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)
TR = glob.glob("/kaggle/input/**/enveda-CASMI26-molecule-id-mass-spectra/train.parquet", recursive=True)[0]
tr = pd.read_parquet(TR, columns=["ingest_lib", "inchikey14", "adduct", "adduct_orig"])
known = set(tr.inchikey14.dropna())
tr = tr[tr.ingest_lib == "enveda-180"]               # only the adducts the competition kept for this library
amap = dict(zip(tr.adduct_orig, tr.adduct))          # release spelling -> competition spelling
amap.update({a: a for a in set(tr.adduct)})
print("train structures", len(known), "| adduct vocab", sorted(set(tr.adduct)), flush=True)
schema = pq.ParquetFile(TR).schema_arrow
del tr

def fnum(x, d=np.nan):
    try:
        return float(x)
    except Exception:
        return d

out, smi_cache = [], {}
pf = pq.ParquetFile(RAW)
COLS = ["inchikey", "adduct", "mzs", "intensities", "smiles", "collision_energies", "formula", "ionmode",
        "pepmass", "precursor_ppm"]
for g, b in enumerate(pf.iter_batches(batch_size=20000, columns=COLS)):
    b = b.to_pandas()
    b["ik14"] = b.inchikey.str[:14]
    b = b[~b.ik14.isin(known) & b.adduct.isin(amap)]
    for r in b.itertuples():
        mz = np.asarray(r.mzs, dtype=float); it = np.asarray(r.intensities, dtype=float)
        ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
        mz, it = mz[ok], it[ok]
        if len(mz) == 0:
            continue
        o = np.argsort(mz); mz, it = mz[o], it[o]
        if r.smiles not in smi_cache:
            m = Chem.MolFromSmiles(r.smiles)
            smi_cache[r.smiles] = Chem.MolToSmiles(m) if m is not None else None
        if smi_cache[r.smiles] is None:
            continue
        ces = [abs(fnum(c)) for c in str(r.collision_energies).split(",") if np.isfinite(fnum(c))]
        out.append(dict(ingest_lib="enveda-180", normalized_smiles=smi_cache[r.smiles], inchikey=r.inchikey,
                        inchikey14=r.ik14, molecular_formula=r.formula,
                        ionization_mode="positive" if "pos" in str(r.ionmode).lower() else "negative",
                        instrument_type="timsTOF", adduct=amap[r.adduct], adduct_orig=r.adduct,
                        precursor_mz=fnum(r.pepmass), precursor_error_ppm=fnum(r.precursor_ppm),
                        ms2_mzs=mz.tolist(), ms2_normalized_intensities=(it / it.sum()).tolist(),
                        num_peaks=len(mz), base_peak_intensity=float(it.max()),
                        collision_energy_ev=ces, collision_energy_orig=str(r.collision_energies),
                        collision_energy_orig_units="eV"))
    if g % 10 == 0: print(f"batch {g}: kept {len(out)} spectra  [{time.time() - t0:.0f}s]", flush=True)
X = pd.DataFrame(out)
tbl = pa.Table.from_pandas(X, preserve_index=False)
tbl = pa.Table.from_arrays([tbl[c].cast(schema.field(c).type) for c in schema.names], schema=schema)
pq.write_table(tbl, "/kaggle/working/e180_extra.parquet")
print(f"E180 EXTRA: {len(X)} spectra, {X.inchikey14.nunique()} structures, adducts {X.adduct.value_counts().to_dict()}")
print(f"done in {time.time() - t0:.0f}s")
