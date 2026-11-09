"""Run one analysis cell as a private CPU Kaggle notebook over finished kernels' outputs.

  python hybrid/build_cpu_cell.py --slug casmi26-signals --cell hybrid/signal_ensemble_cell.py \
      --sub __SRC__=casmi26-eval-ext/ --src casmi26-eval-ext
"""
import argparse, json, os
import requests
ap = argparse.ArgumentParser()
ap.add_argument("--slug", required=True)
ap.add_argument("--cell", required=True, action="append", help="cells, in order")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--sub", action="append", default=[], help="KEY=VALUE text substitutions in the cell")
ap.add_argument("--src", action="append", default=[], help="kernel sources (slug under our account)")
ap.add_argument("--pip", default="rdkit lightgbm")
ap.add_argument("--data", default="", help="small local dir shipped as __DATA__ (private notebook only; never commit it)")
ap.add_argument("--dataset", action="append", default=[], help="dataset sources (owner/slug)")
a = ap.parse_args()
import base64, io, tarfile
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    tar.add(os.path.join(os.path.dirname(__file__), "..", "src"), arcname="src", filter=lambda t: None if "__pycache__" in t.name else t)
dbuf = io.BytesIO()
if a.data:
    with tarfile.open(fileobj=dbuf, mode="w:gz") as tar:
        tar.add(a.data, arcname="data")
cells = []
for path in a.cell:
    cell = open(path).read().replace("__PAYLOAD__", base64.b64encode(buf.getvalue()).decode()).replace("__DATA__", base64.b64encode(dbuf.getvalue()).decode())
    for kv in a.sub:
        k, v = kv.split("=", 1); cell = cell.replace(k, v)
    cells.append(cell)
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
nb = {"cells": [code(f"!pip install -q {a.pip} 2>&1 | tail -1")] + [code(c) for c in cells],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
r = requests.post("https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel", headers={"User-Agent": "casmi26-tools", "Authorization": "Bearer " + (os.environ.get("KAGGLE_API_TOKEN") or next((open(p).read().strip() for p in [os.path.expanduser("~/.kaggle/access_token")] if os.path.exists(p)), "proxy"))},
                  timeout=300, json={"slug": f"xaxipirulazo/{a.slug}", "newTitle": a.slug, "text": json.dumps(nb),
                  "language": "python", "kernelType": "notebook", "isPrivate": True, "enableGpu": a.gpu, "enableInternet": True,
                  "competitionDataSources": ["enveda-CASMI26-molecule-id-mass-spectra"],
                  "kernelDataSources": [f"xaxipirulazo/{s}" for s in a.src], "datasetDataSources": a.dataset})
print(r.status_code, r.text[:300])
