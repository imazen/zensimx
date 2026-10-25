"""Exact comparison of two gates.py output directories for one candidate name (the rebuilt-tool control).

Exploration record: benchmarks/byconstruction_2026-10-22.md ("Revision 7 with raw ringing"). Reads no human label.

  python3 gates_compare.py --name mfull --a <stored gates dir> --b <rebuilt gates dir> [--steps nearid,verdict,steer,identity]

Every served number is compared by its bits; the verdict checks by state and measured value, and the dial predictions
by their text. Fields that name the run (paths, the steering repair cache) are excluded. Exits 1 on any difference.
"""

import argparse
import json
import sys
from pathlib import Path

# Steering fields that describe the run rather than its result.
STEER_RUN_FIELDS = {"repair_feature_cache"}


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def compare_rows(label, a, b, skip=()):
    diffs = []
    if len(a) != len(b):
        return [f"{label}: {len(a)} rows vs {len(b)}"]
    for i, (x, y) in enumerate(zip(a, b)):
        keys = (set(x) | set(y)) - set(skip)
        for k in sorted(keys):
            if x.get(k) != y.get(k):
                diffs.append(f"{label} row {i} ({x.get('key', x.get('reference', ''))}): {k} differs")
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--steps", default="nearid,verdict,steer,identity")
    a = ap.parse_args()
    steps, diffs, compared = a.steps.split(","), [], []
    if "nearid" in steps:
        rel = f"nearid-s0/candidate-{a.name}.jsonl"
        diffs += compare_rows("nearid", jsonl(a.a / rel), jsonl(a.b / rel))
        ga, gb = (json.loads((d / f"nearid-s0/candidate-{a.name}.GATES.json").read_text()) for d in (a.a, a.b))
        for k in ("N1", "N2", "N3"):
            if ga[k] != gb[k]:
                diffs.append(f"nearid {k} differs")
        compared.append(f"nearid {len(jsonl(a.a / rel))} rows")
    if "verdict" in steps:
        for grid in ("standard", "ladder"):
            d = f"verdict-s0-{a.name}/{grid}"
            ca, cb = ({c["id"]: (c["state"], c["measured"]) for c in json.loads((r / d / "gaddr.json").read_text())["checks"]}
                      for r in (a.a, a.b))
            if ca != cb:
                diffs += [f"verdict {grid} {k}: {ca.get(k)} vs {cb.get(k)}" for k in sorted(set(ca) | set(cb))
                          if ca.get(k) != cb.get(k)]
            pa, pb = ((r / d / "predictions.tsv").read_text() for r in (a.a, a.b))
            if pa != pb:
                diffs.append(f"verdict {grid} predictions differ")
            compared.append(f"verdict {grid} {len(ca)} checks, {pa.count(chr(10))} prediction lines")
    if "steer" in steps:
        ra, rb = (json.loads((r / f"steer135/{a.name}/result.json").read_text())["rows"] for r in (a.a, a.b))
        diffs += compare_rows("steer", ra, rb, STEER_RUN_FIELDS)
        compared.append(f"steer {len(ra)} cases")
    if "identity" in steps:
        ia, ib = (jsonl(r / "identity-s0.jsonl") for r in (a.a, a.b))
        diffs += compare_rows("identity", ia, ib)
        compared.append(f"identity {len(ia)} rows")
    print(json.dumps({"name": a.name, "a": str(a.a), "b": str(a.b), "compared": compared, "differences": len(diffs),
                      "first_differences": diffs[:20]}, indent=1))
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
