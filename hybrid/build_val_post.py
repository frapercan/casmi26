"""Re-run only the scoring + re-rank sweep of the validation harness (CPU, minutes) on the
outputs of a finished casmi26-hybrid-val run, attached as a kernel source.

  python hybrid/build_val_post.py --slug casmi26-hybrid-val-post
"""
import argparse, base64, io, json, os, tarfile
import requests
ROOT = os.path.join(os.path.dirname(__file__), "..")
COPY = '''# copy the finished harness outputs into the working dir, where the scoring cell expects them
import glob, os, shutil
_src = [p for p in glob.glob("/kaggle/input/**/val_answers.parquet", recursive=True) if "__SRC__" in p][0]
_R = os.path.dirname(_src)
for p in glob.glob(f"{_R}/*.csv") + glob.glob(f"{_R}/variants*/*.csv") + [_src, f"{_R}/valcomp/test.parquet"]:
    d = os.path.join("/kaggle/working", os.path.relpath(p, _R)); os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copyfile(p, d)
print("copied from", _R, len(glob.glob("/kaggle/working/**/*.csv", recursive=True)), "csv")
'''
ap = argparse.ArgumentParser()
ap.add_argument("--slug", default="casmi26-hybrid-val-post")
ap.add_argument("--src", default="casmi26-hybrid-val")
ap.add_argument("--gpu", action="store_true")
a = ap.parse_args()
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    tar.add(os.path.join(ROOT, "src"), arcname="src", filter=lambda t: None if "__pycache__" in t.name else t)
post = open(os.path.join(ROOT, "hybrid", "val_post_cell.py")).read().replace("__PAYLOAD__", base64.b64encode(buf.getvalue()).decode())
mk = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
nb = {"cells": [mk("!pip install -q rdkit duckdb 2>&1 | tail -1"), mk(COPY.replace("__SRC__", a.src)), mk(post)],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
r = requests.post("https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel", headers={"User-Agent": "casmi26-tools", "Authorization": "Bearer " + (os.environ.get("KAGGLE_API_TOKEN") or next((open(p).read().strip() for p in [os.path.expanduser("~/.kaggle/access_token")] if os.path.exists(p)), "proxy"))}, timeout=300, json={
    "slug": f"xaxipirulazo/{a.slug}", "newTitle": a.slug, "text": json.dumps(nb), "language": "python",
    "kernelType": "notebook", "isPrivate": True, "enableGpu": a.gpu, "enableInternet": True,
    "competitionDataSources": ["enveda-CASMI26-molecule-id-mass-spectra"],
    "kernelDataSources": [f"xaxipirulazo/{a.src}", "xaxipirulazo/casmi26-train-v2", "xaxipirulazo/casmi26-train-v3"],
})
print(r.status_code, r.text[:400])
