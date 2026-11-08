# ── which class-3 molecules does engine 2's candidate pool miss (no structure within 10/30 ppm)? ──
import glob, os, pickle, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog("rdApp.*")
def find(n):
    h = glob.glob(f"/kaggle/input/**/{n}", recursive=True); return sorted(h, key=len)[0]
d = os.path.dirname(find("coco_fp.npy")); keys = list(pickle.load(open(f"{d}/coco_meta.pkl", "rb"))["keys"])
mass = [np.load(f"{d}/coco_mass.npy")]
for sub, pre in [("casmi-chebi-lmsd-structures", "bio"), ("casmi26-pubchemlite-cc0-candidate-fingerprints", "pubchemlite")]:
    try:
        bd = os.path.dirname(find(f"{sub}/{pre}_mass.npy")); bk = np.asarray(pickle.load(open(f"{bd}/{pre}_meta.pkl", "rb"))["keys"], object)
        new = ~pd.Series(bk).isin(set(keys)).values; mass.append(np.load(f"{bd}/{pre}_mass.npy")[new]); keys += list(bk[new])
        print(pre, int(new.sum()))
    except Exception as e:
        print(pre, "missing", repr(e)[:120])
tr = pd.read_parquet(f"{VC}/train.parquet", columns=["inchikey14", "normalized_smiles"]).drop_duplicates("inchikey14")
tr = tr[~tr.inchikey14.isin(set(keys))]
tm = []
for s in tr.normalized_smiles:
    m = Chem.MolFromSmiles(s); tm.append(ExactMolWt(m) if m is not None else np.nan)
mass.append(np.array(tm))
P = np.sort(np.concatenate(mass)); P = P[np.isfinite(P)]
print("pool", len(P))
AD = {"[M+H]+": 1.007276, "[M-H]-": -1.007276, "[M+Na]+": 22.989218, "[M+NH4]+": 18.033823, "[M]+": -0.000549, "[M+K]+": 38.963158}
T = pd.read_parquet(f"{VC}/test.parquet")
T["nm"] = T.precursor_mz - T.adduct.map(AD)
print("unknown adducts", T.adduct[T.nm.isna()].unique())
A = pd.read_parquet("/kaggle/working/val_answers.parquet").set_index("molecule_id")
rows = []
for mid, g in T.groupby("molecule_id"):
    t = float(np.median(g.nm.dropna())) if g.nm.notna().any() else np.nan
    c = lambda p: int(np.searchsorted(P, t * (1 + p / 1e6), "right") - np.searchsorted(P, t * (1 - p / 1e6), "left")) if np.isfinite(t) else -1
    rows.append(dict(mid=mid, target=t, n10=c(10), n30=c(30), adducts=",".join(sorted(set(g.adduct))), val3_id=A.val3_id.get(mid), smiles=A.smiles.get(mid)))
R = pd.DataFrame(rows)
print(R.n10.describe())
print("EMPTY_POOL", R[(R.n30 <= 0)].to_dict("records"))
print("THIN_POOL", R[(R.n10 <= 2)].sort_values("n10").head(15).to_dict("records"))
