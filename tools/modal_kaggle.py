"""Drive Modal from a Kaggle CPU kernel (internet on, no GPU quota): Claude cloud sessions
cannot speak Modal's gRPC, but they can push Kaggle kernels and read their logs.

  python tools/modal_kaggle.py --slug casmi26-modal-1 --steps upload,prepare,features
  python tools/modal_kaggle.py --slug casmi26-modal-2 --steps "sweep:exp/reg.json:4,report"

Steps:
  upload            competition files + candidate datasets -> Modal volume casmi26-data (data/...)
  prepare|features|report        modal run modal_app.py::<step>
  sweep:<exp>[:<parallel>]       modal run modal_app.py::sweep
  evaluate:<run>                 modal run modal_app.py::evaluate
  sh:<command>                   any shell command in the repo dir (e.g. modal volume ls ...)

The Modal token is read from MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in this process's
environment, or from ~/.casmi_modal (two lines: id, secret), and written into the private
kernel's source. Rotate the token if the kernel is ever made public.
"""
import argparse, base64, io, json, os, sys, tarfile
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
API = "https://api.kaggle.com/v1/"
H = {"User-Agent": "casmi26-tools", "Authorization": "Bearer " + (os.environ.get("KAGGLE_API_TOKEN") or next((open(p).read().strip() for p in [os.path.expanduser("~/.kaggle/access_token")] if os.path.exists(p)), "proxy"))}

CELL = r'''
import base64, glob, io, os, subprocess, sys, tarfile, time
os.environ["MODAL_TOKEN_ID"] = __TID__
os.environ["MODAL_TOKEN_SECRET"] = __TSEC__
R = "/kaggle/working/casmi26"
tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__"))).extractall(R)
def sh(cmd, check=True):
    print(f"\n>>> {cmd}  [{time.strftime('%H:%M:%S')}]", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=R)
    print(f"<<< exit {p.returncode}  [{time.strftime('%H:%M:%S')}]", flush=True)
    if check and p.returncode:
        raise SystemExit(f"step failed: {cmd}")
sh("pip install -q modal 2>&1 | tail -1")
sh("modal volume create casmi26-data", check=False)
def first(pat):
    h = sorted(glob.glob(pat, recursive=True), key=len)
    return h[0] if h else None
for step in __STEPS__:
    kind, _, arg = step.partition(":")
    if kind == "upload":
        comp = os.path.dirname(first("/kaggle/input/**/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"))
        for f in ["train.parquet", "test.parquet", "sample_submission.csv"]:
            sh(f"modal volume put --force casmi26-data {comp}/{f} data/{f}")
        coco = first("/kaggle/input/**/coconut_pure_shard_*.parquet")
        pcl = first("/kaggle/input/**/pubchemlite_structures.parquet")
        pub = first("/kaggle/input/**/pubchem_pure_shard_01.parquet")
        ris = os.path.commonpath([os.path.dirname(coco), os.path.dirname(pub)])
        sh(f"du -sh {ris} {os.path.dirname(pcl)}", check=False)
        sh(f"modal volume put --force casmi26-data {ris} data/cands/rishav")
        sh(f"modal volume put --force casmi26-data {os.path.dirname(pcl)} data/cands/pcl")
        sh("modal volume ls casmi26-data data")
    elif kind == "sweep":
        exp, _, par = arg.partition(":")
        sh(f"modal run modal_app.py::sweep --exp {exp} --parallel {par or 4}")
    elif kind == "evaluate":
        sh(f"modal run modal_app.py::evaluate --run {arg}")
    elif kind == "sh":
        sh(arg, check=False)
    else:
        sh(f"modal run modal_app.py::{kind}")
print("MODAL BRIDGE DONE", flush=True)
'''


def token():
    tid, sec = os.environ.get("MODAL_TOKEN_ID"), os.environ.get("MODAL_TOKEN_SECRET")
    if not (tid and sec):
        p = os.path.expanduser("~/.casmi_modal")
        tid, sec = open(p).read().split()[:2]
    return tid, sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--steps", required=True, help="comma-separated, see module doc")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    tid, sec = token()
    buf = io.BytesIO()
    skip = ("/.git", "__pycache__", "/data", "/work")
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in ["src", "scripts", "tools", "exp", "configs", "modal_app.py", "requirements.txt"]:
            tar.add(os.path.join(ROOT, d), arcname=d, filter=lambda t: None if any(s in t.name for s in skip) else t)
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    src = (CELL.replace("__TID__", repr(tid)).replace("__TSEC__", repr(sec))
           .replace("__STEPS__", repr(steps)).replace("__PAYLOAD__", base64.b64encode(buf.getvalue()).decode()))
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    if a.dry:
        print(src[:1500].replace(sec, "***")); return
    r = requests.post(API + "kernels.KernelsApiService/SaveKernel", headers=H, timeout=300, json={
        "slug": f"xaxipirulazo/{a.slug}", "newTitle": a.slug, "text": json.dumps(nb), "language": "python",
        "kernelType": "notebook", "isPrivate": True, "enableGpu": False, "enableInternet": True,
        "competitionDataSources": ["enveda-CASMI26-molecule-id-mass-spectra"],
        "datasetDataSources": ["rishavae/casmi-2026-candidate-database", "xjoannax88/casmi-pubchemlite-candidates"],
    })
    print(r.status_code, r.text[:300])


if __name__ == "__main__":
    main()
