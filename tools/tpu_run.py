"""Push a JAX/Flax training run to a Kaggle TPU VM (separate quota from GPU).

  python tools/tpu_run.py --slug casmi26-tpu-wm4 -- --epochs 30 --batch 2048 --d 384 --layers 6 --fwd --hard-k 32 --hard-pool 128 --bf16

The work dir (spectral library, molecule embeddings, holdout split, validation set) is assembled from the
outputs of earlier GPU kernels, in order (later ones win). Outputs: work/runs/jax/{best.msgpack,args.json,log.json}.
"""
import argparse, base64, io, json, os, shlex, tarfile
import requests

ROOT = os.path.join(os.path.dirname(__file__), "..")
TEMPLATE = r'''
import base64, glob, io, os, subprocess, sys, tarfile, time
code = "/kaggle/working/casmi26"
tarfile.open(fileobj=io.BytesIO(base64.b64decode("{payload}"))).extractall(code)
W = "/kaggle/working/work"
os.makedirs(W, exist_ok=True)
def link_tree(src, dst):
    for root, _, files in os.walk(src):
        d = os.path.join(dst, os.path.relpath(root, src)); os.makedirs(d, exist_ok=True)
        for f in files:
            t = os.path.join(d, f)
            if os.path.lexists(t): os.remove(t)
            os.symlink(os.path.join(root, f), t)
for slug in {reuse!r}:
    hit = [h for h in glob.glob("/kaggle/input/**/work/", recursive=True) if "/" + slug + "/" in h]
    assert hit, slug
    print("assemble", slug, "<-", hit[0], flush=True); link_tree(hit[0], W)
def run(cmd):
    print(">>>", cmd, flush=True); t = time.time()
    subprocess.run(cmd, shell=True, check=True)
    print(f"<<< {{time.time()-t:.0f}}s", flush=True)
run("pip install -q flax optax rdkit 2>&1 | tail -1; python -c 'import jax; print(jax.devices())'")
run(f"{{sys.executable}} {{code}}/scripts/train_jax.py --work {{W}} --out {{W}}/runs/jax " + {args!r})
for d in ("lib", "emb", "candidates.parquet", "val", "plm"):  # keep only the run
    p = f"{{W}}/{{d}}"
    if os.path.islink(p) or os.path.isfile(p): os.remove(p)
    elif os.path.isdir(p): subprocess.run(f"rm -rf {{p}}", shell=True)
print("DONE", flush=True)
'''

ap = argparse.ArgumentParser()
ap.add_argument("--slug", required=True)
ap.add_argument("--shape", default="TpuV5E8")
ap.add_argument("--reuse", default="casmi26-train-v1,casmi26-train-v2,casmi26-train-v3")
ap.add_argument("train_args", nargs="*")
a = ap.parse_args()
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for sub in ("src", "scripts", "splits"):
        tar.add(os.path.join(ROOT, sub), arcname=sub, filter=lambda t: None if "__pycache__" in t.name else t)
reuse = [r for r in a.reuse.split(",") if r]
text = TEMPLATE.format(payload=base64.b64encode(buf.getvalue()).decode(), reuse=reuse, args=shlex.join(a.train_args))
r = requests.post("https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel", headers={"User-Agent": "casmi26-tools", "Authorization": "Bearer " + (os.environ.get("KAGGLE_API_TOKEN") or next((open(p).read().strip() for p in [os.path.expanduser("~/.kaggle/access_token")] if os.path.exists(p)), "proxy"))},
                  timeout=300, json={"slug": f"xaxipirulazo/{a.slug}", "newTitle": a.slug, "text": text, "language": "python",
                  "kernelType": "script", "isPrivate": True, "enableGpu": False, "enableTpu": True, "machineShape": a.shape,
                  "enableInternet": True, "competitionDataSources": ["enveda-CASMI26-molecule-id-mass-spectra"],
                  "kernelDataSources": [f"xaxipirulazo/{s}" for s in reuse]})
print(r.status_code, r.text[:300])
