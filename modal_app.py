"""Run the casmi26 training stack on Modal (persistent volume, CPU prep, GPU sweeps).

One-time:
  pip install modal && modal setup        (or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment)
  Data: either upload what is already built locally (no Kaggle credentials needed on Modal)
    modal volume put casmi26-data <local data dir> data      (train.parquet, sample_submission.csv)
    modal volume put casmi26-data <local work dir>/<x> work/<x>   for lib, candidates.parquet, holdout.parquet, val
  or let `prepare` download it: create a Modal secret "kaggle" (KAGGLE_USERNAME/KAGGLE_KEY) and set
  CASMI_KAGGLE_SECRET=1 when running.

Then (each step is idempotent; outputs live in the "casmi26-data" volume):
  modal run modal_app.py::prepare            # CPU: download, library, candidates, validation set (~1-2 h)
  modal run modal_app.py::features           # GPU: ChemBERTa embeddings, SAE, JEPA, throughput bench
  modal run modal_app.py::sweep --exp exp/reg.json --parallel 4
  modal run modal_app.py::evaluate --run <dir under /data/work/runs>   # hidden-test-like validation + LB fit
  modal run modal_app.py::report
  modal volume get casmi26-data work/runs/<run>/best.pt .               # bring a model back for Kaggle

Cost control: prepare runs on CPU only; features on an L4; sweeps on the GPU set below.
Every function has a timeout, so a forgotten run cannot drain the credits.
"""
import os
import subprocess

import modal

GPU_SWEEP = "H100"   # B200 if available on the account; A100-80GB is the cheaper fallback
GPU_FEAT = "L4"
app = modal.App("casmi26")
vol = modal.Volume.from_name("casmi26-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unzip")
    .pip_install("torch==2.8.0", "numpy", "pandas", "pyarrow", "scipy", "numba", "rdkit==2026.3.3",
                 "duckdb", "transformers", "kaggle", "requests")
    .add_local_dir(".", "/root/casmi26", ignore=["data", "work", ".git", "**/__pycache__", "*.parquet", "*.npy", "*.pt"])
)
kaggle = [modal.Secret.from_name("kaggle")] if os.environ.get("CASMI_KAGGLE_SECRET") else []
R, W = "/root/casmi26", "/data/work"


def sh(cmd: str):
    print(">>>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=R)


@app.function(image=image, volumes={"/data": vol}, secrets=kaggle, cpu=8, memory=49152, timeout=4 * 3600)
def prepare():
    try:
        sh("bash scripts/setup_b200.sh /data prep")
    finally:
        vol.commit()


@app.function(image=image, volumes={"/data": vol}, gpu=GPU_FEAT, cpu=8, memory=65536, timeout=4 * 3600)
def features():
    try:
        sh("bash scripts/setup_b200.sh /data gpu")
    finally:
        vol.commit()


@app.function(image=image, volumes={"/data": vol}, gpu=GPU_SWEEP, cpu=16, memory=131072, timeout=6 * 3600)
def sweep(exp: str = "exp/reg.json", parallel: int = 4):
    try:
        sh(f"python3 tools/sweep.py {exp} --work {W} --val {W}/val --parallel {parallel}")
        sh(f"python3 tools/report.py {W}/results.jsonl")
    finally:
        vol.commit()


@app.function(image=image, volumes={"/data": vol}, gpu=GPU_FEAT, cpu=8, memory=65536, timeout=2 * 3600)
def evaluate(run: str = "run"):
    d = f"{W}/runs/{run}"
    try:
        sh(f"python3 scripts/eval_valset.py --work {W} --val {W}/val --ckpt {d}/best.pt --sae {W}/runs/sae.pt "
           f"--out {d}/valeval --config fused=configs/weights_v2.json --config lib=configs/weights_libonly.json "
           f"--config model=configs/weights_modelonly.json --lb fused=0.179 --lb lib=0.152 --lb model=0.122")
    finally:
        vol.commit()


@app.function(image=image, volumes={"/data": vol}, timeout=600)
def report():
    sh(f"python3 tools/report.py {W}/results.jsonl")
