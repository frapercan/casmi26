"""Run a public competition notebook verbatim on our account (Pick A base, rerun controls).

The source kernel's code cells, data sources and Docker image are read from the API and reused unchanged; outputs are
cleared, a title cell naming the source (and the sha256 of its code cells) is prepended, internet is off and the
image is pinned, so the run reproduces the source as closely as Kaggle allows. Nothing in the code is patched.

  python hybrid/build_fork.py --src alexchilton/casmi26-lb-0-487-peaks-to-structures-scorer --slug casmi26-pick-a487
  python hybrid/build_fork.py --src OWNER/SLUG --dry nb/fork.ipynb        # write the notebook, push nothing
"""
import argparse, hashlib, json, os, sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import kaggle_api as K  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="owner/slug of the public kernel")
ap.add_argument("--slug", default="")
ap.add_argument("--version", type=int, default=0, help="source version (0 = current)")
ap.add_argument("--dry", default="")
a = ap.parse_args()
owner, kslug = a.src.split("/")
body = {"userName": owner, "kernelSlug": kslug} | ({"versionNumber": a.version} if a.version else {})
j = K.post("kernels.KernelsApiService/GetKernel", body).json()
m, nb = j["metadata"], json.loads(j["blob"]["source"])
code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
sha = hashlib.sha256(code.encode()).hexdigest()
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
title = (f"# Verbatim run of [{a.src}](https://www.kaggle.com/code/{a.src}) (version {m.get('currentVersionNumber')})\n\n"
         f"Every code cell, dataset, kernel input and the Docker image are the source notebook's, unchanged; only this cell "
         f"is added. Code-cell sha256 `{sha[:16]}`. All credit belongs to its authors and to the work they credit below.\n")
nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": title})
src = {k: m.get(k) or [] for k in ("datasetDataSources", "kernelDataSources", "competitionDataSources")}
print(f"source v{m.get('currentVersionNumber')} image {m.get('dockerImage')} shape {m.get('machineShape')} code sha {sha[:16]}")
print(json.dumps(src, indent=1))
if a.dry:
    json.dump(nb, open(a.dry, "w"), indent=1)
    raise SystemExit(f"wrote {a.dry}")
assert a.slug, "--slug"
r = requests.post(K.API + "kernels.KernelsApiService/SaveKernel", headers=K.H, timeout=300, json={
    "slug": f"{K.USER}/{a.slug}", "newTitle": a.slug, "text": json.dumps(nb), "language": "python", "kernelType": "notebook",
    "isPrivate": True, "enableGpu": True, "enableInternet": False, "dockerImage": m.get("dockerImage"),
    "dockerImagePinningType": "original", "machineShape": m.get("machineShape") or "NvidiaTeslaT4", **src})
print(r.status_code, r.text[:400])
