"""Build (and optionally push) the class-3 submission notebook: public 0.480 pipeline + hybrid/class3_sub_cell.py.

  python hybrid/build_c3.py --upstream base480.ipynb --slug casmi26-c3-sub [--dry out.ipynb]
"""
import argparse, base64, copy, io, json, os, tarfile
import requests
ROOT = os.path.join(os.path.dirname(__file__), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--upstream", required=True)
ap.add_argument("--slug", required=True)
ap.add_argument("--ins-pos", type=int, default=10)
ap.add_argument("--ins-k", type=int, default=5)
ap.add_argument("--budget", type=int, default=7200)
ap.add_argument("--ckpt", default="casmi26-train-wm7c3")
ap.add_argument("--dry", default="", help="write the notebook here instead of pushing")
a = ap.parse_args()
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    tar.add(os.path.join(ROOT, "src"), arcname="src", filter=lambda t: None if "__pycache__" in t.name else t)
cell = open(os.path.join(ROOT, "hybrid", "class3_sub_cell.py")).read() \
    .replace("__INS_POS__", str(a.ins_pos)).replace("__INS_K__", str(a.ins_k)).replace("__CKPT__", a.ckpt) \
    .replace("__DRY__", "False").replace("__BUDGET__", str(a.budget)) \
    .replace("__PAYLOAD__", base64.b64encode(buf.getvalue()).decode())
compile(cell, "class3_sub_cell", "exec")
nb = copy.deepcopy(json.load(open(a.upstream)))
nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell})
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
if a.dry:
    json.dump(nb, open(a.dry, "w")); print("wrote", a.dry, len(nb["cells"]), "cells"); raise SystemExit
r = requests.post("https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel", headers={"Authorization": "Bearer proxy"}, timeout=180, json={
    "slug": f"xaxipirulazo/{a.slug}", "newTitle": a.slug, "text": json.dumps(nb), "language": "python",
    "kernelType": "notebook", "isPrivate": True, "enableGpu": True, "enableInternet": False,
    "competitionDataSources": ["enveda-CASMI26-molecule-id-mass-spectra"],
    "datasetDataSources": ["sanpier/casmi-spectrum-model-v1", "dmitriigluzdov/casmi26-fold-safe-fpnet",
                           "dmitriigluzdov/casmi26-natural-product-knowledge-table", "dmitriigluzdov/casmi26-natural-product-panel-benchmark",
                           "dmitriigluzdov/casmi26-pubchem-popularity-prior", "dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets",
                           "sanpier/catalogue-experiment-saner", "metric/rdkit-2026-3-3-wheel", "xaxipirulazo/casmi26-speclib"],
    "kernelDataSources": ["bgunshui/casmi26-scorer-train-public", "xaxipirulazo/casmi26-train-v2", f"xaxipirulazo/{a.ckpt}"],
})
print(r.status_code, r.text[:400])
