"""One LB probe of B0-repro, end to end, with checks before anything is submitted:
bundle (checkpoint) -> private dataset -> private kernel (its commit run is the smoke test) -> log checks -> submit.

Checks on the commit log, all required: our bundle found, PLM check max |diff| < 0.05, the 0.5/0.5 ensemble built,
median rank correlation of our scores with the public model's > 0.15 (0.40 on the eval-c3 run; ~0 would mean a
broken environment), 'SECOND SCORER: ours', 'SUBMISSION READY', no 'Our scorer failed'.

  python hybrid/submit_probe.py --ckpt work/runs/w4base_s0/best.pt --label w4base_s0 [--submit]
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import kaggle_api as K  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--plm-mean", default="work/emb_v4/plm_mean.npy")
ap.add_argument("--dataset", default="casmi26-ours-bundle")
ap.add_argument("--slug", default="casmi26-b0repro")
ap.add_argument("--submit", action="store_true")
a = ap.parse_args()
PY = sys.executable


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
    return r.stdout


def stamp(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


run([PY, "hybrid/build_ours_bundle.py", "--ckpt", a.ckpt, "--plm-mean", a.plm_mean, "--out", "work/bundle_ours", "--tar", "work/bundle_ours.tar"])
out = run([PY, "tools/kaggle_api.py", "upload", a.dataset, "work/bundle_ours.tar"])
assert re.search(r"(create|version) 200", out) and "put work/bundle_ours.tar 200" in out, "dataset upload failed"
for _ in range(30):   # wait for Kaggle to process the dataset version
    try:
        st = K.post("datasets.DatasetApiService/GetDatasetStatus", {"ownerSlug": K.USER, "datasetSlug": a.dataset}).json()
        stamp(f"dataset status {st}")
        if str(st.get("status", "")).lower() in ("ready", "complete", "1"):
            break
    except Exception as exc:  # unknown method or transient error: fall back to waiting
        stamp(f"dataset status unavailable ({exc}); waiting")
    time.sleep(60)
out = run([PY, "hybrid/build_b0repro.py", "--upstream", "nb/base480.ipynb", "--slug", a.slug, "--bundle-ds", f"{K.USER}/{a.dataset}"])
m = re.search(r'"versionNumber":\s*(\d+)', out)
err = re.search(r'"error":\s*"([^"]+)"', out)
assert m and not err, f"kernel push failed: {err.group(1) if err else out[-300:]}"
version = int(m.group(1))
stamp(f"kernel {a.slug} version {version} pushed; waiting for its commit run")
time.sleep(120)
while True:
    s = K.status(a.slug); stamp(f"kernel status {s}")
    if s in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
        break
    time.sleep(120)
log = K.log(a.slug, r"OUR SCORER|ours:|ENSEMBLE|OURS-vs-PUBLIC|SECOND SCORER|SUBMISSION READY|Our scorer failed|Error|Traceback")
print(log, flush=True)
plm = re.search(r"PLM check max \|diff\| ([0-9.]+)", log)
rc = re.search(r"rank correlation: median ([-0-9.na]+)", log)
checks = {
    "kernel complete": s == "COMPLETE",
    "bundle found": bool(re.search(r"OUR SCORER BUNDLE: /kaggle/", log)),
    "plm check": bool(plm) and float(plm.group(1)) < 0.05,
    "ensemble built": "ENSEMBLE neural = 0.5*z(public) + 0.5*z(ours)" in log,
    "rank correlation": bool(rc) and rc.group(1) not in ("nan",) and float(rc.group(1)) > 0.15,
    "second scorer ours": "SECOND SCORER: ours" in log,
    "submission ready": "SUBMISSION READY" in log,
    "no scorer failure": "Our scorer failed" not in log,
}
stamp(json.dumps(checks))
if not all(checks.values()):
    sys.exit("checks failed: not submitting")
if a.submit:
    desc = f"B0-repro: public 0.480 pipeline, our model ({a.label}) in the second-scorer slot (0.5/0.5, ranks 2-25)"
    run([PY, "tools/kaggle_api.py", "submit", a.slug, str(version), desc])
    stamp("submitted; scoring rerun on the hidden test takes a few hours")
    while True:   # the scoring rerun: report the newest submission until it is scored
        time.sleep(600)
        S = K.post("competitions.CompetitionApiService/ListSubmissions", {"competitionName": K.COMP, "pageSize": 3}).json().get("submissions", [])
        top = S[0] if S else {}
        stamp(f"{top.get('status')} public {top.get('publicScore')} | {str(top.get('description', ''))[:70]}")
        if any(x in str(top.get("status", "")).upper() for x in ("COMPLETE", "ERROR")):
            break
