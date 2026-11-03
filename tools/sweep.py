"""Run a grid of train.py experiments and log every result to one JSONL file.

An experiment file (exp/*.json) holds shared args and a grid:
  {"name": "reg", "base": {"epochs": 10, "gpu-data": true, "compile": true},
   "grid": {"drop": [0.1, 0.3], "wd": [0.01, 0.05]}}
Each grid point becomes one run tagged <name>/<k=v,...>. Runs already in the results
file (same tag) are skipped, so a sweep can be resumed after an interruption.

  python tools/sweep.py exp/reg.json --work /data/work --parallel 4
--parallel N runs N trainings at once on the same GPU (a small model leaves a B200
mostly idle; several runs share it well). Use tools/report.py to compare.
"""
import argparse, itertools, json, os, subprocess, sys, time

ROOT = os.path.join(os.path.dirname(__file__), "..")


def cli(args: dict) -> list[str]:
    out = []
    for k, v in args.items():
        if v is True:
            out.append(f"--{k}")
        elif v is False or v is None:
            continue
        else:
            out += [f"--{k}", str(v)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--work", default="/home/user/work")
    ap.add_argument("--runs", default="", help="output dir for checkpoints (default <work>/runs/sweeps)")
    ap.add_argument("--results", default="", help="default <work>/results.jsonl")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    e = json.load(open(a.exp))
    runs = a.runs or f"{a.work}/runs/sweeps"
    res = a.results or f"{a.work}/results.jsonl"
    done = set()
    if os.path.exists(res):
        done = {json.loads(l)["tag"] for l in open(res) if l.strip()}
    grid = e.get("grid", {})
    keys = list(grid)
    jobs = []
    for vals in itertools.product(*[grid[k] for k in keys]) if keys else [()]:
        pt = dict(zip(keys, vals))
        tag = e["name"] + ("/" + ",".join(f"{k}={v}" for k, v in pt.items()) if pt else "")
        if tag in done:
            print("skip (done)", tag); continue
        args = {**e.get("base", {}), **pt, "work": a.work, "tag": tag, "results": res,
                "out": f"{runs}/{tag.replace('/', '__').replace(',', '_').replace('=', '')}"}
        cmd = [sys.executable, f"{ROOT}/scripts/train.py"] + cli(args)
        jobs.append((tag, cmd, args["out"]))
    print(f"{len(jobs)} runs, {a.parallel} at a time -> {res}")
    if a.dry:
        for t, c, _ in jobs:
            print(" ".join(c))
        return
    live = []
    while jobs or live:
        while jobs and len(live) < a.parallel:
            tag, cmd, out = jobs.pop(0)
            os.makedirs(out, exist_ok=True)
            p = subprocess.Popen(cmd, stdout=open(f"{out}/train.log", "w"), stderr=subprocess.STDOUT)
            live.append((tag, p, time.time()))
            print("start", tag, flush=True)
        for x in list(live):
            if x[1].poll() is not None:
                print(f"end   {x[0]} rc={x[1].returncode} {time.time()-x[2]:.0f}s", flush=True)
                live.remove(x)
        time.sleep(5)


if __name__ == "__main__":
    main()
