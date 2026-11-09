"""Private (for now) Kaggle notebook: honest evaluation of CASMI26 pipelines.

  python hybrid/build_honest_eval.py --slug casmi26-honest-eval [--public]

CPU only: it reads the outputs of finished harness runs (casmi26-eval-ext: public pipeline on
leak-free Enveda-180 queries, without the withdrawn scorer; casmi26-eval-ext-b: with it) and
reports per-arm paired CIs, the oracle ceiling, observable strata and a gated router.
"""
import argparse, json, os
import requests
ROOT = os.path.join(os.path.dirname(__file__), "..")
MD = [
"""# CASMI26 · An honest evaluation: strata, oracle ceiling and a gated router

**Why.** A validation set cut from `train.parquet` cannot rank the public pipelines: their models and
priors were trained on all of `train.parquet`, so every validation molecule has already been seen.
On our account that leakage showed up as a +0.019 gain on a train-based validation that became
−0.017 on the public board.

**What this notebook does differently.** The queries are spectra of structures that are **absent from
`train.parquet`**, taken from the host-approved Enveda-180 Zenodo release (record 21346580, CC-BY-4.0,
thread 745148): same instrument family as the test (timsTOF), adducts restricted to the ones the test
uses. No model or prior in the pipeline has seen these structures.

**Protocol** (borrowed from CAFA evaluation practice, see PROTEA / cafaeval-protea):
1. Every arm (pipeline variant, or a model re-ordering the pipeline's top-25) is scored per molecule with
   the official rule: stereo removed, tautomer canonicalised, InChIKey first block, MRR@25.
2. Differences against a base arm are **paired bootstrap** 95% CIs. A mean without an interval is not a result.
3. Strata: *observable* ones (mass, number of spectra, peak count, polarity, stepped collision energy,
   agreement between arms) can drive a router; the *true* class is for evaluation only.
4. **Oracle ceiling**: the MRR if the best arm were picked for every molecule. If it is close to the base,
   no amount of routing helps and a new expert is needed.
5. **Router**: shallow trees on observables only, cross-fitted on 10 disjoint molecule folds. It counts only
   if it passes a gate: positive in ≥ 8/10 folds **and** CI lower bound > 0.""",
"""## Run
Inputs are the outputs of two harness runs (the public 0.480 pipeline pointed at the leak-free query set),
attached as notebook sources. Everything below is CPU and takes a few minutes.""",
]
ap = argparse.ArgumentParser()
ap.add_argument("--slug", default="casmi26-honest-eval")
ap.add_argument("--roots", default=json.dumps({"no_scorer": "casmi26-eval-ext/", "with_scorer": "casmi26-eval-ext-b/"}))
ap.add_argument("--base", default="with_scorer:submission.csv")
ap.add_argument("--public", action="store_true")
a = ap.parse_args()
roots = json.loads(a.roots)
cell = open(os.path.join(ROOT, "hybrid", "honest_eval_cell.py")).read()
cell = cell.replace("__ROOTS__", repr(roots)).replace("__BASE__", a.base)
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
nb = {"cells": [md(MD[0]), md(MD[1]), code("!pip install -q rdkit 2>&1 | tail -1"), code(cell)],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
r = requests.post("https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel", headers={"User-Agent": "casmi26-tools", "Authorization": "Bearer " + (os.environ.get("KAGGLE_API_TOKEN") or next((open(p).read().strip() for p in [os.path.expanduser("~/.kaggle/access_token")] if os.path.exists(p)), "proxy"))},
                  timeout=300, json={
    "slug": f"xaxipirulazo/{a.slug}", "newTitle": a.slug, "text": json.dumps(nb), "language": "python",
    "kernelType": "notebook", "isPrivate": not a.public, "enableGpu": False, "enableInternet": True,
    "competitionDataSources": ["enveda-CASMI26-molecule-id-mass-spectra"],
    "kernelDataSources": [f"xaxipirulazo/{k.rstrip('/')}" for k in roots.values()]})
print(r.status_code, r.text[:400])
