"""Enveda-180 library extension for the public pipeline.

  python hybrid/build_e180.py extract --slug casmi26-e180            # CPU kernel, internet on: builds e180_extra.parquet
  python hybrid/build_e180.py submit  --upstream base480.ipynb --slug casmi26-e180-sub [--dry out.ipynb]

`submit` runs the upstream notebook unchanged except that its competition folder is
/kaggle/working/augcomp = {test.parquet, sample_submission.csv: links to the real ones,
train.parquet: the real train plus the extra Enveda-180 spectra}. Same three path patches as
build_val.py, with the pinned inner notebook re-encoded and its SHA-256 updated.
"""
import argparse, ast, base64, copy, hashlib, json, os, sys
import requests
sys.path.insert(0, os.path.dirname(__file__))
import build_val
ROOT = os.path.join(os.path.dirname(__file__), "..")
API = "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel"
H = {"Authorization": "Bearer " + os.environ.get("KAGGLE_API_TOKEN", "proxy")}
AUG = "/kaggle/working/augcomp"
PRE = r'''# ── casmi26: competition folder = real test + (train ∪ Enveda-180 extra spectra) ──
import glob, os
import pyarrow.dataset as ds, pyarrow.parquet as pq
AUG = "__AUG__"
os.makedirs(AUG, exist_ok=True)
_real = os.path.dirname(glob.glob("/kaggle/input/**/enveda-CASMI26-molecule-id-mass-spectra/test.parquet", recursive=True)[0])
for _f in ["test.parquet", "sample_submission.csv"]:
    if not os.path.exists(f"{AUG}/{_f}"):
        os.symlink(f"{_real}/{_f}", f"{AUG}/{_f}")
_x = [p for p in glob.glob("/kaggle/input/**/e180_extra.parquet", recursive=True)]
assert _x, "e180_extra.parquet not attached"
if not os.path.exists(f"{AUG}/train.parquet"):
    _d = ds.dataset(f"{_real}/train.parquet")
    _e = pq.read_table(_x[0]).cast(_d.schema)
    _n = 0
    with pq.ParquetWriter(f"{AUG}/train.parquet", _d.schema) as _w:
        for _b in _d.to_batches(batch_size=65536):
            _w.write_batch(_b); _n += _b.num_rows
        _w.write_table(_e)
    print(f"AUGCOMP train {_n} + {_e.num_rows} Enveda-180 spectra ({len(set(_e.column('inchikey14').to_pylist()))} structures)")
'''
DATASETS = ["sanpier/casmi-spectrum-model-v1", "dmitriigluzdov/casmi26-fold-safe-fpnet",
            "dmitriigluzdov/casmi26-natural-product-knowledge-table", "dmitriigluzdov/casmi26-natural-product-panel-benchmark",
            "dmitriigluzdov/casmi26-pubchem-popularity-prior", "dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets",
            "sanpier/catalogue-experiment-saner", "metric/rdkit-2026-3-3-wheel"]
mk = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def save(slug, nb, gpu, internet, **src):
    r = requests.post(API, headers=H, timeout=300, json={
        "slug": f"xaxipirulazo/{slug}", "newTitle": slug, "text": json.dumps(nb), "language": "python",
        "kernelType": "notebook", "isPrivate": True, "enableGpu": gpu, "enableInternet": internet,
        "competitionDataSources": ["enveda-CASMI26-molecule-id-mass-spectra"], **src})
    print(r.status_code, r.text[:400])


def build_submit(upstream, src_kernel):
    nb = copy.deepcopy(json.load(open(upstream)))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    cell = next(c for c in code if "FROZEN_B64 = '" in "".join(c["source"]))
    src = "".join(cell["source"])
    vals = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("FROZEN_B64", "FROZEN_SHA256")}
    fr = base64.b64decode(vals["FROZEN_B64"])
    assert hashlib.sha256(fr).hexdigest() == vals["FROZEN_SHA256"]
    build_val.VC = AUG
    new = build_val.patch_frozen(fr)
    src = build_val.sub1(src, vals["FROZEN_B64"], base64.b64encode(new).decode(), "b64")
    cell["source"] = build_val.sub1(src, vals["FROZEN_SHA256"], hashlib.sha256(new).hexdigest(), "sha")
    c2 = next(c for c in code if "COMP = next((p for p in (INPUT / 'competitions/" in "".join(c["source"]))
    c2["source"] = build_val.sub1("".join(c2["source"]), "COMP = next((p for p in (INPUT / 'competitions/",
                                  f"COMP = next((p for p in (Path('{AUG}'), INPUT / 'competitions/", "outer comp")
    nb["cells"].insert(0, mk(PRE.replace("__AUG__", AUG)))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "submit"])
    ap.add_argument("--slug", required=True)
    ap.add_argument("--upstream", default="")
    ap.add_argument("--src", default="casmi26-e180", help="kernel whose output holds e180_extra.parquet")
    ap.add_argument("--dry", default="")
    a = ap.parse_args()
    if a.mode == "extract":
        nb = {"cells": [mk("!pip install -q rdkit 2>&1 | tail -1"), mk(open(os.path.join(ROOT, "hybrid", "e180_cell.py")).read())],
              "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        return save(a.slug, nb, False, True)
    nb = build_submit(a.upstream, a.src)
    if a.dry:
        json.dump(nb, open(a.dry, "w")); print("wrote", a.dry); return
    save(a.slug, nb, True, False, datasetDataSources=DATASETS,
         kernelDataSources=["bgunshui/casmi26-scorer-train-public", f"xaxipirulazo/{a.src}"])


if __name__ == "__main__":
    main()
