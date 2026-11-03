"""E36 method review M1: summarize per-seed paired panels (`dataset_eval.py paired`, arm minus control, whole-reference
bootstrap) into one table per arm: for each EVAL MOS collection, the seed-mean change in SROCC and PLCC, and at how
many seeds the 95% interval lies wholly above or below zero. The row-weighted mean over collections is descriptive.

    m1_summary.py LABEL=DIR[,DIR...] [LABEL=DIR,...] ...    (one DIR per seed, each holding report/report.json)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(d):
    r = json.loads((Path(d) / "report" / "report.json").read_text())
    rows = {ds["id"]: sum(1 for _ in open(Path(d) / ds["path"])) - 1 for ds in
            json.loads((Path(d) / "suite.json").read_text())["datasets"]}
    out = {}
    for c in r["comparisons"]:
        if c["report"].get("state") != "measured":
            continue
        m = c["report"]["measurements"]
        out[c["spec"]["dataset"]] = dict(rows=rows[c["spec"]["dataset"]], refs=m["bootstrap"]["references"],
                                         **{k: m["delta"][i] for i, k in enumerate(m["delta_order"])})
    return out


def main(specs):
    lines = []
    for spec in specs:
        label, dirs = spec.split("=", 1)
        seeds = [load(d) for d in dirs.split(",")]
        colls = sorted(set.intersection(*(set(s) for s in seeds)))
        lines += [f"**{label}** ({len(seeds)} seed{'s' if len(seeds) > 1 else ''})", "",
                  "| collection | rows / refs | ΔSROCC (seed mean) | seeds above / below 0 | ΔPLCC (seed mean) | seeds above / below 0 |",
                  "|---|---|---:|---|---:|---|"]
        agg = defaultdict(float)
        for c in colls:
            row = []
            for k in ("srocc", "plcc"):
                est = np.mean([s[c][k]["estimate"] for s in seeds])
                up = sum(s[c][k]["lo"] > 0 for s in seeds)
                dn = sum(s[c][k]["hi"] < 0 for s in seeds)
                row += [f"{est:+.4f}", f"{up} / {dn}"]
                agg[k] += est * seeds[0][c]["rows"]
            lines.append(f"| {c} | {seeds[0][c]['rows']} / {seeds[0][c]['refs']} | " + " | ".join(row) + " |")
        n = sum(seeds[0][c]["rows"] for c in colls)
        lines += ["", f"Row-weighted mean over {len(colls)} collections: ΔSROCC {agg['srocc'] / n:+.4f}, "
                      f"ΔPLCC {agg['plcc'] / n:+.4f} (descriptive).", ""]
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
