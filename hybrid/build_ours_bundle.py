"""Assemble the self-contained bundle that hybrid/ours_neural_scores.py runs from (no network needed):
src/casmi (our code), the checkpoint, the PLM snapshot (tokenizer + weights), plm_mean.npy, ours_config.json.

  python hybrid/build_ours_bundle.py --ckpt work/runs/w4base_s0/best.pt --plm-mean work/emb_v4/plm_mean.npy --out work/bundle_ours
"""
from __future__ import annotations

import argparse, json, os, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--plm-mean", required=True)
ap.add_argument("--out", default="work/bundle_ours")
ap.add_argument("--electron-shift-pos", type=float, default=0.000548)
ap.add_argument("--tar", default="", help="also pack the bundle into this uncompressed tar (one file to upload)")
ap.add_argument("--src", default="", help="directory holding the casmi package to bundle (default: this repo's src), "
                "e.g. an exploration's worktree whose checkpoint needs its code")
ap.add_argument("--inst-override", default="", help="instrument class every query is scored as (instrument-conditioned "
                "checkpoints only; default: each query's own instrument_type)")
a = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.abspath(a.src or os.path.join(root, "src"))
sys.path.insert(0, SRC)
if os.path.exists(a.out):
    shutil.rmtree(a.out)
os.makedirs(a.out)
shutil.copytree(os.path.join(SRC, "casmi"), os.path.join(a.out, "src", "casmi"),
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copy2(a.ckpt, os.path.join(a.out, "ours.pt"))
shutil.copy2(a.plm_mean, os.path.join(a.out, "plm_mean.npy"))
shutil.copy2(os.path.join(os.path.dirname(__file__), "ours_neural_scores.py"), os.path.join(a.out, "ours_neural_scores.py"))
# the PLM as the original hub files (config, tokenizer, safetensors weights), not re-saved by our newer transformers,
# so the older transformers of the Kaggle image reads them as it reads the hub snapshot
import glob  # noqa: E402
snaps = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots/*/"))
files = {}
for sd in snaps:
    for f in os.listdir(sd):
        if f != "pytorch_model.bin":
            files.setdefault(f, os.path.join(sd, f))
need = {"config.json", "model.safetensors", "tokenizer_config.json", "vocab.json", "merges.txt"}
assert need <= set(files), f"PLM snapshot incomplete: missing {need - set(files)}"
os.makedirs(os.path.join(a.out, "plm"))
for f, src in files.items():
    shutil.copy2(os.path.realpath(src), os.path.join(a.out, "plm", f))
from casmi.plm import PLMEmbedder  # noqa: E402
import numpy as np  # noqa: E402
ref = PLMEmbedder(device="cpu").embed(["CCO", "c1ccccc1O"])
got = PLMEmbedder(os.path.join(a.out, "plm"), device="cpu").embed(["CCO", "c1ccccc1O"])
assert np.allclose(ref, got, atol=1e-5), "bundled PLM differs from the hub model"
# reference embeddings (centred) so the scorer can detect a PLM that behaves differently in another environment
CHECK = ["CCO", "c1ccccc1O", "CC(=O)OC1=CC=CC=C1C(=O)O", "OC1C(O)C(O)C(CO)OC1O"]
pe = PLMEmbedder(os.path.join(a.out, "plm"), device="cpu", mean=np.load(a.plm_mean))
np.save(os.path.join(a.out, "plm_check.npy"), pe.embed(CHECK).astype(np.float32))
json.dump(CHECK, open(os.path.join(a.out, "plm_check.json"), "w"))
cfg = {"checkpoint": "ours.pt", "electron_shift_pos": a.electron_shift_pos, "source_ckpt": a.ckpt,
       "src": os.path.relpath(SRC, os.path.abspath(root))}
if a.inst_override:
    cfg["inst_override"] = a.inst_override
json.dump(cfg, open(os.path.join(a.out, "ours_config.json"), "w"), indent=1)
print("bundle", a.out, sorted(os.listdir(a.out)))
if a.tar:
    import tarfile
    with tarfile.open(a.tar, "w") as t:
        t.add(a.out, arcname="bundle_ours")
    print("packed", a.tar, os.path.getsize(a.tar) // 2**20, "MB")
