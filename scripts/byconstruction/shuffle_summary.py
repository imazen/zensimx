"""Summarize shuffle_study.sh: per shuffle seed, the paired median ratio (second arm / first − 1) of each comparison,
then the mean and standard deviation over seeds. Exploration record: benchmarks/byconstruction_2026-10-22.md.

  python3 shuffle_summary.py OUT_DIR
"""

import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np


def paired(path: Path) -> tuple[float, float, float]:
    rows = [json.loads(line) for line in path.open()]
    ms = {arm: {r["round"]: r["result"]["median_ms"] for r in rows if r["arm"] == arm} for arm in ("A", "B")}
    d = np.array([ms["B"][k] / ms["A"][k] - 1 for k in sorted(ms["A"])])
    return float(np.median(d) * 100), float(np.median(list(ms["A"].values()))), float(np.median(list(ms["B"].values())))


def main() -> int:
    out = Path(sys.argv[1])
    by = {}
    for f in sorted(out.glob("shuf*_*.jsonl")):
        m = re.fullmatch(r"shuf(\d+)_(.+)\.jsonl", f.name)
        by.setdefault(m.group(2), {})[int(m.group(1))] = paired(f)
    for name, seeds in by.items():
        ds = [seeds[s][0] for s in sorted(seeds)]
        print(f"{name:32s} per seed " + " ".join(f"{d:+6.2f}%" for d in ds) +
              f" | mean {st.mean(ds):+6.2f}%  sd {st.stdev(ds) if len(ds) > 1 else 0:5.2f}  "
              f"A ms {st.mean(seeds[s][1] for s in seeds):8.3f}  B ms {st.mean(seeds[s][2] for s in seeds):8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
