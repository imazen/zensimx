"""Paired summary of a ring_timing.sh JSONL: per-arm median of round medians, and the median per-round ratio
(second arm / first arm - 1) with a 95% bootstrap CI (5,000 resamples, seed 1).

Exploration record: benchmarks/byconstruction_2026-10-22.md.

  python3 timing_summary.py OUT.jsonl [--arms A,B]
"""

import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl")
    ap.add_argument("--arms", default="A,B", help="base,candidate arm labels as written in the JSONL")
    a = ap.parse_args()
    base, cand = a.arms.split(",")
    rows = [json.loads(line) for line in open(a.jsonl)]
    ms = {arm: {r["round"]: r["result"]["median_ms"] for r in rows if r["arm"] == arm} for arm in (base, cand)}
    if set(ms[base]) != set(ms[cand]) or not ms[base]:
        raise SystemExit("arms do not cover the same rounds")
    d = np.array([ms[cand][k] / ms[base][k] - 1 for k in sorted(ms[base])])
    rng = np.random.default_rng(1)
    boots = [np.median(rng.choice(d, len(d))) for _ in range(5000)]
    print(json.dumps({"rounds": len(d), f"{base}_median_ms": float(np.median(list(ms[base].values()))),
                      f"{cand}_median_ms": float(np.median(list(ms[cand].values()))),
                      "paired_delta_pct": float(np.median(d) * 100),
                      "ci95_pct": [float(np.percentile(boots, 2.5) * 100), float(np.percentile(boots, 97.5) * 100)]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
