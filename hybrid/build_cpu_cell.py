"""Run one analysis cell as a private CPU Kaggle notebook over finished kernels' outputs.

  python hybrid/build_cpu_cell.py --slug casmi26-signals --cell hybrid/signal_ensemble_cell.py \
      --sub __SRC__=casmi26-eval-ext/ --src casmi26-eval-ext
"""
import argparse, json, os
import requests
ap = argparse.ArgumentParser()
ap.add_argument("--slug", required=True)
ap.add_argument("--cell", required=True)
ap.add_argument("--sub", action="append", default=[], help="KEY=VALUE text substitutions in the cell")
ap.add_argument("--src", action="append", default=[], help="kernel sources (slug under our account)")
ap.add_argument("--pip", default="rdkit lightgbm")
a = ap.parse_args()
cell = open(a.cell).read()
for kv in a.sub:
    k, v = kv.split("=", 1); cell = cell.replace(k, v)
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
nb = {"cells": [code(f"!pip install -q {a.pip} 2>&1 | tail -1"), code(cell)],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
r = requests.post("https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel", headers={"Authorization": "Bearer proxy"},
                  timeout=300, json={"slug": f"xaxipirulazo/{a.slug}", "newTitle": a.slug, "text": json.dumps(nb),
                  "language": "python", "kernelType": "notebook", "isPrivate": True, "enableGpu": False, "enableInternet": True,
                  "competitionDataSources": ["enveda-CASMI26-molecule-id-mass-spectra"],
                  "kernelDataSources": [f"xaxipirulazo/{s}" for s in a.src]})
print(r.status_code, r.text[:300])
