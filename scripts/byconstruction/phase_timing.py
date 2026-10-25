"""Per-phase timing (ZENSIM_FOLD_TIMING) of byc_score_timing arms, interleaved across process starts.

Exploration record: benchmarks/byconstruction_2026-10-22.md. Whole-walk timing between two builds moves by several
percent from code placement alone (identical instructions, different addresses); the walk's phase timer separates
the phase a change touches from the phases it does not. Each start runs WALKS walks and dumps every WALKS/2; the
second (warm) dump is kept. Output: one JSON line per (start, arm) with every phase's ms/walk.

  phase_timing.py --bake B --size N --starts K --arm NAME=BIN:FORM [--arm ...] --out OUT.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import sys


def run(bin_path, form, bake, size, walks, cpus, threads):
    env = dict(os.environ, ZENSIM_FOLD_TIMING=str(walks // 2), ZENSIM_RING_FORM=form, RAYON_NUM_THREADS=str(threads))
    p = subprocess.run(["taskset", "-c", cpus, bin_path, "--bake", bake, "--size", str(size), "--iters", str(walks),
                        "--warmup", "0"], env=env, capture_output=True, text=True, check=True)
    dumps = p.stderr.split("=== ZENSIM FOLD TIMING ===")[1:]
    if len(dumps) != 2:
        raise ValueError(f"expected 2 dumps, got {len(dumps)}")
    warm = dumps[1]
    phases = {}
    for line in warm.splitlines():
        m = re.match(r"\s*(\S+(?:\(\S+\))?)\s+([0-9.]+)\s+[0-9.]+", line)
        if m and not line.lstrip().startswith(("phase", "walks", "accounted")):
            phases[m.group(1)] = float(m.group(2))
    walk = float(re.search(r"walk=([0-9.]+) ms/walk", warm).group(1))
    score = json.loads(p.stdout.strip().splitlines()[-1])["score"]
    return {"walk_ms": walk, "phases": phases, "score": score}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bake", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--walks", type=int, default=10)
    ap.add_argument("--cpus", default="2")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--arm", action="append", required=True, help="NAME=BIN:FORM")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = [(s.split("=", 1)[0], *s.split("=", 1)[1].rsplit(":", 1)) for s in a.arm]
    with open(a.out, "w") as f:
        for start in range(a.starts):
            order = arms if start % 2 == 0 else arms[::-1]
            for name, bin_path, form in order:
                r = run(bin_path, form, a.bake, a.size, a.walks, a.cpus, a.threads)
                f.write(json.dumps({"start": start, "arm": name, "form": form, "size": a.size, **r}) + "\n")
                f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
