#!/usr/bin/env python3
"""Render a perf_grid.sh case-matrix grid.tsv as a markdown table: one row per case, and per bake x tier the two
arms' medians and the paired change with its 95% interval. Record: benchmarks/byconstruction_2026-10-22.md.

  perf_matrix.py GRID_TSV
"""

import csv
import sys
from collections import OrderedDict


def main() -> int:
    rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
    cols = list(OrderedDict.fromkeys((r["bake"], r["tier"]) for r in rows))
    cases = list(OrderedDict.fromkeys((r["case"], r["class"], r["size"]) for r in rows))
    cell = {(r["case"], r["bake"], r["tier"]): r for r in rows}
    head = ["case", "class", "size"] + [f"{b} {t}" for b, t in cols]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for case, cls, size in cases:
        out = [case, cls, size]
        for b, t in cols:
            r = cell.get((case, b, t))
            out.append(
                f"{float(r['A_ms']):.3f} → {float(r['B_ms']):.3f} ms, {r['delta']} ({r['ci95_lo']}..{r['ci95_hi']})"
                if r
                else "—"
            )
        print("| " + " | ".join(out) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
