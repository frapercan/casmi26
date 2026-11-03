"""Table of experiment results (results.jsonl from train.py --results), best first.

  python tools/report.py /data/work/results.jsonl [--filter reg/] [--key ens]
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--filter", default="")
ap.add_argument("--key", default="ens")
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.results) if l.strip()]
rows = [r for r in rows if a.filter in r["tag"]]
rows.sort(key=lambda r: -r["final"].get(a.key, 0))
heads = ["ens", "nce", "jepa", "fp", "sae"]
print(f"{'tag':55s} " + " ".join(f"{h:>6s}" for h in heads) + "   min  commit")
for r in rows:
    f = r["final"]
    print(f"{r['tag'][:55]:55s} " + " ".join(f"{f.get(h, float('nan')):6.3f}" for h in heads) +
          f" {r['time_s']/60:5.0f}  {r.get('commit', '')}")
