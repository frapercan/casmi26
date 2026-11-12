"""Train arms of scripts/train.py on one Kaggle GPU session: one arm per visible GPU at a time (T4 x2: two at once).

The kernel embeds src/ + scripts/ (+ --extra files, e.g. a frozen bias expert, shipped under extra/) as a base64
tarball and reads the w4 training store from a private dataset of flat files. It rebuilds work/w4/{lib,emb},
work/w4/holdout.parquet and work/cells/exclude*_ik.txt with symlinks and checks every file's size and sha256 against
the manifest taken at push time (the arms train on exactly the local store), then runs the arms in waves.
Outputs: runs/<tag>/ (checkpoint, log.json, train.log) and runs/results.jsonl; optimizer state files are dropped.

  python tools/kaggle_train.py stage                  # sha256 manifest of the store files -> work/kaggle_store.json
  python tools/kaggle_train.py upload                 # create / version the private dataset with those files
  python tools/kaggle_train.py push --slug casmi26-t4bench --common "--work work/w4 ..." \
      --arm "b0=--seed 0 --max-steps 1500" --arm "b1=--seed 1 --max-steps 1500" [--minutes 150] [--extra FILE]
  python tools/kaggle_train.py status|log|output --slug casmi26-t4bench [--dest DIR]
"""
from __future__ import annotations

import argparse, base64, hashlib, io, json, os, tarfile

import requests

API = "https://api.kaggle.com/v1/"
USER = os.environ.get("KAGGLE_USER", "xaxipirulazo")
DATASET = "casmi26-w4-store"
# the image of every run so far (P001, eval-c3, t4bench), pinned so all arms of an exploration share one platform
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST = os.path.join(ROOT, "work", "kaggle_store.json")
# file name -> (local source relative to the repo, place inside the kernel's work/)
STORE = {"spectra.parquet": ("work/w4/lib/spectra.parquet", "w4/lib"), "peaks.npz": ("work/w4/lib/peaks.npz", "w4/lib"),
         "molecules.parquet": ("work/w4/lib/molecules.parquet", "w4/lib"),
         "instrument.parquet": ("work/w4/lib/instrument.parquet", "w4/lib"),
         "mol_table.parquet": ("work/w4/emb/mol_table.parquet", "w4/emb"), "plm.npy": ("work/w4/emb/plm.npy", "w4/emb"),
         "fp.npy": ("work/w4/emb/fp.npy", "w4/emb"), "plm_mean.npy": ("work/w4/emb/plm_mean.npy", "w4/emb"),
         "holdout.parquet": ("work/w4/holdout.parquet", "w4"),
         "exclude_all_ik.txt": ("work/cells/exclude_all_ik.txt", "cells"), "exclude_ik.txt": ("work/cells/exclude_ik.txt", "cells")}
H = {"User-Agent": "casmi26-tools", "Authorization": "Bearer " + (os.environ.get("KAGGLE_API_TOKEN") or next(
    (open(p).read().strip() for p in [os.path.expanduser("~/.kaggle/access_token")] if os.path.exists(p)), "proxy"))}

KERNEL = r'''
import base64, hashlib, io, json, os, subprocess, sys, tarfile, time
PAYLOAD = "{payload}"
MANIFEST = {manifest!r}
COMMON = {common!r}
ARMS = {arms!r}
MINUTES = {minutes!r}
os.chdir("/kaggle/working")
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))).extractall("casmi26")
src = {{}}
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        if f in MANIFEST:
            src.setdefault(f, os.path.join(root, f))
missing = sorted(set(MANIFEST) - set(src))
assert not missing, f"store files missing: {{missing}}"
for f, (sub, sha, size) in MANIFEST.items():
    os.makedirs(f"work/{{sub}}", exist_ok=True)
    if not os.path.exists(f"work/{{sub}}/{{f}}"):
        os.symlink(src[f], f"work/{{sub}}/{{f}}")
    assert os.path.getsize(src[f]) == size, f"size mismatch {{f}}"
    h = hashlib.sha256()
    with open(src[f], "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    assert h.hexdigest() == sha, f"sha256 mismatch {{f}}"
print("store verified:", len(MANIFEST), "files", flush=True)
subprocess.run(["nvidia-smi", "-L"])
import torch
n = max(torch.cuda.device_count(), 1)
os.makedirs("runs", exist_ok=True)
queue, running, t0 = list(ARMS), {{}}, time.time()
while queue or running:
    for g in range(n):
        if g not in running and queue:
            tag, args = queue.pop(0)
            os.makedirs(f"runs/{{tag}}", exist_ok=True)
            cmd = (f"{{sys.executable}} casmi26/scripts/train.py {{COMMON}} {{args}} --out runs/{{tag}} --tag {{tag}} "
                   f"--results runs/results.jsonl" + (f" --minutes {{MINUTES}}" if MINUTES else ""))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
            p = subprocess.Popen(cmd, shell=True, env=env, stdout=open(f"runs/{{tag}}/train.log", "w"), stderr=subprocess.STDOUT)
            running[g] = (tag, p, time.time())
            print(time.strftime("%H:%M:%S"), "start", tag, "GPU", g, ":", cmd, flush=True)
    time.sleep(60)
    if int((time.time() - t0) / 60) % 5 == 0:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.replace("\n", " | ")
        ram = os.popen("free -g").read().splitlines()[1].split()
        print(time.strftime("%H:%M:%S"), "gpu", smi, "host RAM used/total GB", ram[2], "/", ram[1], flush=True)
    for g, (tag, p, ts) in list(running.items()):
        lines = [l for l in open(f"runs/{{tag}}/train.log").read().splitlines()
                 if l.startswith(("step", "EVAL", "LIFT", "FINAL", "Traceback")) or "Error" in l]
        print(time.strftime("%H:%M:%S"), tag, (lines[-1] if lines else "")[:200], flush=True)
        if p.poll() is not None:
            print(time.strftime("%H:%M:%S"), "done", tag, "rc", p.returncode, f"{{(time.time() - ts) / 60:.1f}} min", flush=True)
            if p.returncode:
                print(open(f"runs/{{tag}}/train.log").read()[-3000:], flush=True)
            del running[g]
for root, _, files in os.walk("runs"):
    for f in files:
        if f.startswith("state") and f.endswith(".pt"):
            os.remove(os.path.join(root, f))
print(open("runs/results.jsonl").read() if os.path.exists("runs/results.jsonl") else "no results.jsonl", flush=True)
print(f"total {{(time.time() - t0) / 60:.1f}} min", flush=True)
'''


def post(method, body, timeout=300):
    r = requests.post(API + method, json=body, headers=H, timeout=timeout)
    if r.status_code != 200:
        raise SystemExit(f"{method}: HTTP {r.status_code} {r.text[:300]}")
    return r.json()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def payload(extra):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in ("src", "scripts"):
            tar.add(os.path.join(ROOT, d), arcname=d, filter=lambda t: None if "__pycache__" in t.name else t)
        for f in extra:
            tar.add(f, arcname=f"extra/{os.path.basename(f)}")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "upload", "push", "status", "log", "output"])
    ap.add_argument("--slug", default="")
    ap.add_argument("--common", default="")
    ap.add_argument("--arm", action="append", default=[], help="tag=extra train.py args (one per arm)")
    ap.add_argument("--minutes", type=int, default=0, help="pass --minutes to every arm (needs the resumable train.py)")
    ap.add_argument("--extra", action="append", default=[], help="file shipped as extra/<name> inside the kernel")
    ap.add_argument("--dest", default=os.path.join(ROOT, "work", "kaggle_runs"))
    a = ap.parse_args()
    if a.cmd == "stage":
        m = {f: [sub, sha256(os.path.join(ROOT, p)), os.path.getsize(os.path.join(ROOT, p))] for f, (p, sub) in STORE.items()}
        json.dump(m, open(MANIFEST, "w"), indent=1)
        print(f"{len(m)} files, {sum(v[2] for v in m.values()) / 1e9:.2f} GB -> {MANIFEST}")
    elif a.cmd == "upload":
        m, toks = json.load(open(MANIFEST)), []
        for f, (p, _) in STORE.items():
            path = os.path.join(ROOT, p)
            assert sha256(path) == m[f][1], f"{f} changed since stage"
            j = post("blobs.BlobApiService/StartBlobUpload", {"type": "DATASET", "name": f, "contentLength": os.path.getsize(path),
                     "contentType": "application/octet-stream", "lastModifiedEpochSeconds": int(os.path.getmtime(path))})
            with open(path, "rb") as fh:
                u = requests.put(j["createUrl"], data=fh, headers={"Content-Type": "application/octet-stream"}, timeout=7200)
            print("put", f, u.status_code, flush=True)
            assert u.status_code in (200, 201), u.text[:300]
            toks.append({"token": j["token"]})
        r = requests.post(API + "datasets.DatasetApiService/CreateDataset", headers=H, timeout=300, json={
            "ownerSlug": USER, "slug": DATASET, "title": DATASET, "isPrivate": True, "licenseName": "unknown", "files": toks})
        print("create", r.status_code, r.text[:300])
        if r.status_code != 200 or r.json().get("error") or "already" in r.text.lower():
            r = requests.post(API + "datasets.DatasetApiService/CreateDatasetVersion", headers=H, timeout=300, json={
                "ownerSlug": USER, "datasetSlug": DATASET,
                "body": {"versionNotes": "w4 store", "files": toks, "deleteOldVersions": True}})
            print("version", r.status_code, r.text[:300])
    elif a.cmd == "push":
        assert a.slug and a.arm, "--slug and at least one --arm"
        m = json.load(open(MANIFEST))
        arms = [x.split("=", 1) for x in a.arm]
        assert all(len(x) == 2 and x[0] for x in arms) and len({t for t, _ in arms}) == len(arms), "arms are unique tag=args"
        text = KERNEL.format(payload=payload(a.extra), manifest=m, common=a.common, arms=arms, minutes=a.minutes or None)
        r = post("kernels.KernelsApiService/SaveKernel", {
            "slug": f"{USER}/{a.slug}", "newTitle": a.slug, "text": text, "language": "python", "kernelType": "script",
            "isPrivate": True, "enableGpu": True, "enableInternet": False, "dockerImagePinningType": "original",
            "dockerImage": IMAGE, "machineShape": "NvidiaTeslaT4",
            "datasetDataSources": [f"{USER}/{DATASET}"], "competitionDataSources": [], "kernelDataSources": []})
        print(json.dumps({k: r.get(k) for k in ("ref", "url", "versionNumber", "error")}, indent=1))
    elif a.cmd == "status":
        print(post("kernels.KernelsApiService/GetKernelSessionStatus", {"userName": USER, "kernelSlug": a.slug}))
    else:
        r = post("kernels.KernelsApiService/ListKernelSessionOutput", {"userName": USER, "kernelSlug": a.slug})
        if a.cmd == "log":
            for line in json.loads(r.get("log") or "[]"):
                print(line.get("data", ""), end="")
        else:
            for f in r.get("files", []):
                path = os.path.join(a.dest, a.slug, f["fileName"])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as fh:
                    fh.write(requests.get(f["url"], timeout=900).content)
                print("saved", path)


if __name__ == "__main__":
    main()
