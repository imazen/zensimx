#!/usr/bin/env python3
"""Score one TRAIN-side HDR dev fold: forward the fold model over the held-out HDR rows (hdr_dev/) and report the
within-reference Spearman per leg (groups of >=2 rows; a leg with fewer than two held-out families is report-only).
Registered selection statistic: HDR_P2P3_EXECUTION_2026-10-29.md. Reads only TRAIN-side held-out rows."""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))


def rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    xs = np.asarray(x)[order]
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--token", choices=["fx1", "fx2"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from v2_common import derived_declaration
    decl_path = derived_declaration(a.token)[0]
    bake = a.study / "full" / a.label / "full_s0/refit/production-f16.bin"
    dev = a.study / "root/hdr_dev"
    result = {}
    for table in sorted(dev.glob("*.parquet")):
        leg = table.stem
        pred = a.out.parent / f"pred_{leg}.tsv"
        subprocess.run([str(a.study / "bin/byc_predict"), "--bake", str(bake), "--decl", str(decl_path), "--corpus", str(table),
                        "--out", str(pred)], check=True)
        p = np.array([float(l.split("\t")[1]) for l in pred.read_text().splitlines() if l and not l.startswith("row")])
        t = pq.read_table(table, columns=["ref_basename", "human_score"])
        refs, y = t["ref_basename"].to_pylist(), np.array(t["human_score"].to_numpy())
        keys = json.loads((dev / f"{leg}.keys.json").read_text())
        families = {k["family"] for k in keys}
        groups = {}
        for i, r in enumerate(refs):
            groups.setdefault(r, []).append(i)
        rhos = [spearman(p[g], y[g]) for g in groups.values() if len(g) >= 2]
        rhos = [r for r in rhos if r is not None]
        result[leg] = dict(rows=len(refs), groups=len(groups), families=len(families), defined_groups=len(rhos),
                           within_ref_spearman_mean=float(np.mean(rhos)) if rhos else None,
                           selection_eligible=len(families) >= 2 and bool(rhos))
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
