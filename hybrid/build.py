"""Build + push the hybrid notebook: upstream public 0.480 pipeline + our re-rank cell.

  python hybrid/build.py --upstream /path/base480.ipynb --slug casmi26-hybrid-w05 --w 0.5
"""
import argparse, base64, copy, io, json, os, sys, tarfile
import requests
ROOT = os.path.join(os.path.dirname(__file__), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--upstream", required=True)
ap.add_argument("--slug", required=True)
ap.add_argument("--w", type=float, default=0.5)
a = ap.parse_args()
nb = json.load(open(a.upstream))
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    tar.add(os.path.join(ROOT, "src"), arcname="src", filter=lambda t: None if "__pycache__" in t.name else t)
cell = open(os.path.join(ROOT, "hybrid", "rerank_cell.py")).read() \
    .replace("__OUR_W__", repr(a.w)).replace("__PAYLOAD__", base64.b64encode(buf.getvalue()).decode())
nb = copy.deepcopy(nb)
nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell})
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
r = requests.post("https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel", headers={"Authorization": "Bearer proxy"}, timeout=180, json={
    "slug": f"xaxipirulazo/{a.slug}", "newTitle": a.slug, "text": json.dumps(nb), "language": "python",
    "kernelType": "notebook", "isPrivate": True, "enableGpu": True, "enableInternet": False,
    "competitionDataSources": ["enveda-CASMI26-molecule-id-mass-spectra"],
    "datasetDataSources": ["sanpier/casmi-spectrum-model-v1", "dmitriigluzdov/casmi26-fold-safe-fpnet",
                           "dmitriigluzdov/casmi26-natural-product-knowledge-table", "dmitriigluzdov/casmi26-natural-product-panel-benchmark",
                           "dmitriigluzdov/casmi26-pubchem-popularity-prior", "dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets",
                           "sanpier/catalogue-experiment-saner", "metric/rdkit-2026-3-3-wheel"],
    "kernelDataSources": ["bgunshui/casmi26-scorer-train-public", "xaxipirulazo/casmi26-train-v2"],
})
print(r.status_code, r.text[:400])
