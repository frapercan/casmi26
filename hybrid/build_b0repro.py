"""B0-repro: the public 0.480 notebook with our model in the slot of the withdrawn second scorer.

The 0.480 notebook ranks 2-25 by 0.5*z(public neural) + 0.5*z(second neural), the second model coming from a
kernel that no longer exists (without it the notebook silently falls back to the public model alone, ~0.479).
Here the second model is ours (code and training in this repository, bundle from hybrid/build_ours_bundle.py,
attached as a dataset), scored by hybrid/ours_neural_scores.py with the pipeline's own stage runner. Our model
also scores the outside candidates (the 480 run used its second model there). Every other cell is unchanged.

  python hybrid/build_b0repro.py --upstream nb/base480.ipynb --dry nb/b0repro.ipynb
  python hybrid/build_b0repro.py --upstream nb/base480.ipynb --slug casmi26-b0repro --bundle-ds xaxipirulazo/casmi26-ours-bundle
"""
import argparse, copy, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--upstream", required=True)
ap.add_argument("--slug", default="casmi26-b0repro")
ap.add_argument("--bundle-ds", default="xaxipirulazo/casmi26-ours-bundle")
ap.add_argument("--dry", default="", help="write the notebook here instead of pushing")
a = ap.parse_args()

CELL14 = r'''BUNDLE = materialise_sources()

# ---- second scorer: our spectrum-structure model, in the slot of the withdrawn one ----
# Reproducible: code and training in the casmi26 repository; bundle (code, checkpoint, PLM snapshot) attached as a dataset.
from pathlib import Path as _P
_cfg = sorted(_P('/kaggle/input').rglob('ours_config.json'))
if not _cfg:   # the dataset holds the bundle as one tar: unpack it into the working directory
    import tarfile as _tf
    _tars = sorted(_P('/kaggle/input').rglob('bundle_ours.tar'))
    if _tars:
        with _tf.open(_tars[0]) as _t:
            try:
                _t.extractall(ROOT / 'ours_extracted', filter='data')
            except TypeError:
                _t.extractall(ROOT / 'ours_extracted')
        _cfg = sorted((ROOT / 'ours_extracted').rglob('ours_config.json'))
OURS_DIR = _cfg[0].parent if _cfg else None
print('OUR SCORER BUNDLE:', OURS_DIR, flush=True)


def score_ours(evidence, output, label):
    run_stage([sys.executable, OURS_DIR / 'ours_neural_scores.py', OURS_DIR, 'ours.pt', evidence,
               ROOT / 'effective_query_spectra.parquet', output, 'cuda'], ROOT / f'{_P(output).stem}.log', label)


PIPELINE_DONE = run_public_pipeline()
if not PIPELINE_DONE and not IS_RERUN and COMMIT_MODE == 'fast':
    shutil.copyfile(COMP / 'sample_submission.csv', ROOT / 'submission.csv')
    raise SystemExit('fast commit: sample file written, the scoring rerun runs every stage')
if not PIPELINE_DONE:
    raise RuntimeError('A real baseline run is required; no placeholder-only submission')
NEURAL_OK = score_final_lists()          # the public scorer (BUNDLE = public kit)
shutil.copyfile(ROOT / 'neural_candidate_features.parquet', ROOT / 'neural_public.parquet')
OURS_OK = False
if NEURAL_OK and OURS_DIR is not None:
    try:
        score_ours(ROOT / 'frozen_candidate_evidence.parquet', ROOT / 'neural_ours.parquet', 'Our scorer · final lists')
        print(''.join(l for l in open(ROOT / 'neural_ours.log') if l.startswith('ours:')), flush=True)
        OURS_OK = True
    except Exception as exc:
        note('Our scorer failed.', f'Public neural model only. {type(exc).__name__}: {str(exc)[:300]}', RED)
if OURS_OK:
    import pandas as _pd
    _a = _pd.read_parquet(ROOT / 'neural_public.parquet').set_index(['molecule_id', 'smiles'])
    _b = _pd.read_parquet(ROOT / 'neural_ours.parquet').set_index(['molecule_id', 'smiles']).reindex(_a.index)
    def _z(_s):
        _g = _s.groupby(level=0)
        return (_s - _g.transform('mean')) / _g.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)
    _a['neural_score'] = 0.5 * _z(_a['neural_score'].fillna(0.0)) + 0.5 * _z(_b['neural_score'].fillna(0.0))
    _a['neural_cos'] = 0.5 * _a['neural_cos'] + 0.5 * _b['neural_cos'].fillna(0.0)
    _a['neural_fp'] = 0.5 * _a['neural_fp'] + 0.5 * _b['neural_fp'].fillna(0.0)
    _a['neural_std'] = 0.5 * _a['neural_std'] + 0.5 * _b['neural_std'].fillna(0.0)
    _a['neural_ok'] = _a['neural_ok'].astype(bool) & _b['neural_ok'].fillna(False).astype(bool)
    _rc = [_g['p'].rank().corr(_g['o'].rank()) for _, _g in
           _pd.DataFrame({'p': _pd.read_parquet(ROOT / 'neural_public.parquet').set_index(['molecule_id', 'smiles']).neural_score,
                          'o': _b.neural_score}).groupby(level=0) if len(_g) > 2]
    print(f'OURS-vs-PUBLIC rank correlation: median {float(np.nanmedian(_rc)):.3f} over {len(_rc)} molecules', flush=True)
    _a.reset_index().to_parquet(ROOT / 'neural_candidate_features.parquet', index=False)
    print('ENSEMBLE neural = 0.5*z(public) + 0.5*z(ours)', flush=True)
if not NEURAL_OK:
    raise RuntimeError('The 0.468 V2 neural checkpoint is required; refusing silent 0.465 fallback')


def score_outside_candidates_ours(needed):
    """Stage 2b: outside candidates collected by the pipeline, scored by our model (public model if ours failed)."""
    if not needed or (time.time() - NB_T0) / 3600 >= TIME_GUARD_HOURS:
        return False
    try:
        run_stage([sys.executable, CODE / 'outsiders.py', FROZEN, ROOT, ROOT / 'frozen_candidate_evidence.parquet',
                   ROOT / 'outsider_evidence.parquet', OUTSIDERS_PER_MOLECULE],
                  ROOT / 'outsiders.log', 'Outside candidates · collect')
        if not len(pd.read_parquet(ROOT / 'outsider_evidence.parquet')):
            return False
        if OURS_OK:
            score_ours(ROOT / 'outsider_evidence.parquet', ROOT / 'outsider_neural_features.parquet', 'Our scorer · outside candidates')
        else:
            run_stage([sys.executable, CODE / 'neural_scores.py', BUNDLE, 'final_model.pt',
                       ROOT / 'outsider_evidence.parquet', ROOT / 'effective_query_spectra.parquet',
                       ROOT / 'outsider_neural_features.parquet', 'cuda'],
                      ROOT / 'outsiders_neural.log', 'Neural scorer · outside candidates')
        return True
    except Exception as exc:
        note('Tail rescue disabled.', f'{type(exc).__name__}: {str(exc)[:300]}', YELLOW)
        return False


OUTSIDERS_OK = score_outside_candidates_ours(needed=True)
CONSENSUS_OK, REPORT, MOVES = build_consensus()
if not CONSENSUS_OK:
    raise RuntimeError('Unable to reproduce the V2 control')
SUBMITTED = choose_and_submit()
if SUBMITTED != VARIANT:
    raise RuntimeError(f'Expected {VARIANT}, got {SUBMITTED}')
shutil.copy2(ROOT/'submission.csv', ROOT/'submission_control_V2.csv')
print('Exact V2 control:', hashlib.sha256((ROOT/'submission_control_V2.csv').read_bytes()).hexdigest()[:16])
print('SECOND SCORER:', 'ours' if OURS_OK else 'none (public only)', flush=True)
'''

TITLE = """# CASMI26 · public 0.480 pipeline with a reproducible second scorer

The public 0.480 notebook (bgunshui's *480 neural ensemble*, run verbatim below) orders ranks 2-25 by
`0.5*z(public neural) + 0.5*z(second neural)`. Its second model came from a kernel that has since been deleted.
Here that slot holds our spectrum-to-structure model instead: a peak Transformer trained contrastively against
ChemBERTa + Morgan molecule embeddings, with a forward head, on format-clean candidates (training code and data
pipeline in our repository). It also scores the outside candidates. Every other cell is the 0.480 notebook, unchanged;
the credits further down apply to all of it.
"""

nb = copy.deepcopy(json.load(open(a.upstream)))
c14 = "".join(nb["cells"][14]["source"])
assert c14.startswith("BUNDLE = materialise_sources()") and "_swap_to_ours" in c14, "unexpected upstream cell 14"
compile(CELL14, "cell14", "exec")
nb["cells"][14]["source"] = CELL14
nb["cells"][0]["source"] = TITLE + "\n---\n\n" + "".join(nb["cells"][0]["source"])
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
if a.dry:
    json.dump(nb, open(a.dry, "w"), indent=1)
    print("wrote", a.dry, len(nb["cells"]), "cells")
    raise SystemExit
import requests  # noqa: E402
TOK = os.environ.get("KAGGLE_API_TOKEN") or next((open(p).read().strip() for p in [os.path.expanduser("~/.kaggle/access_token")] if os.path.exists(p)), "proxy")
H = {"User-Agent": "casmi26-tools", "Authorization": "Bearer " + TOK}
DS = ["sanpier/casmi-spectrum-model-v1", "dmitriigluzdov/casmi26-fold-safe-fpnet",
      "dmitriigluzdov/casmi26-natural-product-knowledge-table", "dmitriigluzdov/casmi26-natural-product-panel-benchmark",
      "dmitriigluzdov/casmi26-pubchem-popularity-prior", "dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets",
      "sanpier/catalogue-experiment-saner", "metric/rdkit-2026-3-3-wheel", a.bundle_ds]
r = requests.post("https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel", headers=H, timeout=180, json={
    "slug": f"xaxipirulazo/{a.slug}", "newTitle": a.slug, "text": json.dumps(nb), "language": "python",
    "kernelType": "notebook", "isPrivate": True, "enableGpu": True, "enableInternet": False,
    "competitionDataSources": ["enveda-CASMI26-molecule-id-mass-spectra"], "datasetDataSources": DS, "kernelDataSources": []})
print(r.status_code, r.text[:400])
