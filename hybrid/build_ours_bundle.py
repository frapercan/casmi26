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
a = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(root, "src"))
if os.path.exists(a.out):
    shutil.rmtree(a.out)
os.makedirs(a.out)
shutil.copytree(os.path.join(root, "src", "casmi"), os.path.join(a.out, "src", "casmi"),
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copy2(a.ckpt, os.path.join(a.out, "ours.pt"))
shutil.copy2(a.plm_mean, os.path.join(a.out, "plm_mean.npy"))
shutil.copy2(os.path.join(os.path.dirname(__file__), "ours_neural_scores.py"), os.path.join(a.out, "ours_neural_scores.py"))
from casmi.plm import PLMEmbedder  # noqa: E402
e = PLMEmbedder(device="cpu")
e.tok.save_pretrained(os.path.join(a.out, "plm"))
e.model.save_pretrained(os.path.join(a.out, "plm"))
json.dump({"checkpoint": "ours.pt", "electron_shift_pos": a.electron_shift_pos, "source_ckpt": a.ckpt},
          open(os.path.join(a.out, "ours_config.json"), "w"), indent=1)
print("bundle", a.out, sorted(os.listdir(a.out)))
