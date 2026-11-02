"""Push / monitor private Kaggle GPU kernels that run this repo's pipeline.

The kernel script embeds src/ + scripts/ (+ the holdout split) as a base64
tarball, so no code dataset is needed. Auth: the session proxy injects the
Bearer token, any placeholder token works locally.

  python tools/kaggle_run.py push  --slug casmi26-train-v1 -- --epochs 20
  python tools/kaggle_run.py status --slug casmi26-train-v1
  python tools/kaggle_run.py log    --slug casmi26-train-v1
  python tools/kaggle_run.py output --slug casmi26-train-v1 --dest /home/user/work/kaggle
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shlex
import tarfile

import requests

API = "https://api.kaggle.com/v1/kernels.KernelsApiService/"
USER = os.environ.get("KAGGLE_USER", "xaxipirulazo")
COMP = "enveda-CASMI26-molecule-id-mass-spectra"
DATASETS = ["rishavae/casmi-2026-candidate-database", "xjoannax88/casmi-pubchemlite-candidates"]
ROOT = os.path.join(os.path.dirname(__file__), "..")

KERNEL_TEMPLATE = r'''
import base64, glob, io, os, shutil, subprocess, sys, tarfile, time
PAYLOAD = "{payload}"
TRAIN_ARGS = {train_args!r}
JEPA_ARGS = {jepa_args!r}
STAGES = {stages!r}
code = "/kaggle/working/casmi26"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))).extractall(code)
W = "/kaggle/working/work"
os.makedirs(W, exist_ok=True)

def run(cmd):
    print(">>>", cmd, flush=True); t = time.time()
    subprocess.run(cmd, shell=True, check=True)
    print(f"<<< {{time.time()-t:.0f}}s", flush=True)

def find(pattern):
    hits = sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True))
    assert hits, pattern
    return hits

run("pip install -q rdkit duckdb 2>&1 | tail -2; nvidia-smi -L")
py = f"{{sys.executable}} {{code}}/scripts"
if "reuse" in STAGES:  # link lib/emb/candidates from an attached previous kernel's output
    src = os.path.dirname(find("work/lib/peaks.npz")[0])[:-len("/lib")]
    os.makedirs(f"{{W}}/emb", exist_ok=True)
    os.symlink(f"{{src}}/lib", f"{{W}}/lib")
    os.symlink(f"{{src}}/candidates.parquet", f"{{W}}/candidates.parquet")
    for f in os.listdir(f"{{src}}/emb"):
        os.symlink(f"{{src}}/emb/{{f}}", f"{{W}}/emb/{{f}}")
    shutil.copy(f"{{code}}/splits/holdout.parquet", f"{{W}}/holdout.parquet")
if "prep" in STAGES:
    run(f"{{py}}/prep_library.py --train {{find('train.parquet')[0]}} --out {{W}}/lib")
    coco = find("coconut_pure_shard_*.parquet")[0]
    pub = os.path.dirname(find("pubchem_pure_shard_01.parquet")[0]) + "/*.parquet"
    pcl = find("pubchemlite_structures.parquet")[0]
    run(f"{{py}}/build_candidates.py --coconut {{coco}} --pubchem '{{pub}}' --pcl {{pcl}} --out {{W}}/candidates.parquet")
    shutil.copy(f"{{code}}/splits/holdout.parquet", f"{{W}}/holdout.parquet")
    run(f"{{py}}/embed_molecules.py --work {{W}} --out {{W}}/emb")
if "sae" in STAGES:
    run(f"{{py}}/train_sae.py --work {{W}} --out {{W}}/runs/sae.pt")
if "jepa" in STAGES:
    run(f"{{py}}/pretrain_jepa.py --work {{W}} --out {{W}}/runs/jepa " + JEPA_ARGS)
if "train" in STAGES:
    extra = (f" --sae {{W}}/runs/sae.pt" if "sae" in STAGES else "") + \
            (f" --init-encoder {{W}}/runs/jepa/encoder.pt" if "jepa" in STAGES else "")
    run(f"{{py}}/train.py --work {{W}} --out {{W}}/runs/run " + TRAIN_ARGS + extra)
if "cleanup" in STAGES:  # keep only the run outputs
    for d in ("lib", "emb", "candidates.parquet"):
        p = f"{{W}}/{{d}}"
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
print("DONE", flush=True)
'''


def call(method, body):
    r = requests.post(API + method, json=body, headers={"Authorization": "Bearer proxy"}, timeout=120)
    if r.status_code >= 400:
        raise SystemExit(f"{method} -> {r.status_code}: {r.text[:500]}")
    return r


def payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in ("src", "scripts", "splits"):
            tar.add(os.path.join(ROOT, sub), arcname=sub,
                    filter=lambda t: None if "__pycache__" in t.name else t)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status", "log", "output"])
    ap.add_argument("--slug", required=True)
    ap.add_argument("--stages", default="prep,train,cleanup")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--dest", default="/home/user/work/kaggle")
    ap.add_argument("--jepa-args", default="")
    ap.add_argument("--reuse-kernel", default="", help="owner/slug whose output has work/lib, emb")
    ap.add_argument("train_args", nargs="*")
    a = ap.parse_args()
    ref = f"{USER}/{a.slug}"

    if a.cmd == "push":
        text = KERNEL_TEMPLATE.format(payload=payload(), train_args=shlex.join(a.train_args),
                                      stages=a.stages.split(","), jepa_args=a.jepa_args)
        r = call("SaveKernel", {
            "slug": ref, "newTitle": a.slug, "text": text, "language": "python",
            "kernelType": "script", "isPrivate": True, "enableGpu": not a.no_gpu,
            "enableInternet": True, "competitionDataSources": [COMP],
            "datasetDataSources": DATASETS,
            "kernelDataSources": [a.reuse_kernel] if a.reuse_kernel else [],
        }).json()
        print(json.dumps({k: r.get(k) for k in ("ref", "url", "versionNumber", "error")}, indent=1))
    elif a.cmd == "status":
        print(call("GetKernelSessionStatus", {"userName": USER, "kernelSlug": a.slug}).json())
    elif a.cmd in ("log", "output"):
        r = call("ListKernelSessionOutput", {"userName": USER, "kernelSlug": a.slug}).json()
        if a.cmd == "log":
            for line in json.loads(r.get("log") or "[]"):
                print(line.get("data", ""), end="")
        else:
            os.makedirs(a.dest, exist_ok=True)
            for f in r.get("files", []):
                path = os.path.join(a.dest, f["fileName"])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as fh:
                    fh.write(requests.get(f["url"], timeout=600).content)
                print("saved", path)


if __name__ == "__main__":
    main()
