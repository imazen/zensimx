#!/usr/bin/env python3
"""Lower-bound probe — a standing part of the metric eval stats procedure.

User directive (2026-07-14): "test completely different pairs to find the lower
score bounds as part of our eval stats procedure." Rank corpora (UPIQ/CID22/…)
live in the *positive* dial range; they never exercise where a metric bottoms
out. This probe scores a bake/profile on CATASTROPHIC pairs (the 2016-pair
corruption grid) to characterize the negative tail: does the metric reach the
low bound, and how far negative does valid-terrible content go?

Why it matters (measured 2026-07-14): B floors at ~1.5 even on the worst
corruption (0 negatives) — it does not model the negative region at all. BHdr
DOES (min −63.9, 20/2016 negatives): the negatives are BHdr's own honest HDR
sensitivity, which is exactly why a dial must NOT clamp them (metric.rs clamps
only at −100). A candidate that shows 0 negatives here has a broken/flattened
lower tail.

  usage: lower_bound_probe.py <bake.bin | profile:a|b|bhdr> [<more> ...]
"""
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

REPO = os.path.expanduser("~/work/zen/zensimx")
CORRUPTION = "/mnt/v/output/zensim/eval_panels_2026-05-29/corruption_grid_372col_2026-05-28.parquet"
RESCORE = f"{REPO}/target/release/rescore_parquet"


def feat_prefix(path):
    names = [f.name for f in pq.read_schema(path)]
    return "feat_" if "feat_0" in names else "f"


def score(spec, corpus):
    """spec = 'profile:a|b|bhdr' -> rescore_parquet; else a bake path via score_bake."""
    pfx = feat_prefix(corpus)
    if spec.startswith("profile:"):
        prof = spec.split(":", 1)[1]
        out = f"/tmp/lbp_{prof}.parquet"
        subprocess.run([RESCORE, "--input", corpus, "--output", out,
                        "--profile", prof, "--score-col", "s", "--feat-prefix", pfx],
                       check=True, capture_output=True)
        return np.array(pq.read_table(out, columns=["s"]).column("s"))
    # arbitrary bake → score_bake (needs feat_ prefix)
    sys.argv = ["xdi"]
    import importlib.util
    src = open(f"{REPO}/scripts/hdr/upiq_crossdomain_instrument.py").read()
    ns = {}
    exec(compile(src.split("# ---- SDR half")[0], "h", "exec"), ns)
    import pyarrow as pa
    t = pq.read_table(corpus)
    cols = {c: t[c] for c in t.schema.names}
    if pfx == "f":
        for i in range(372):
            if f"f{i}" in cols:
                cols[f"feat_{i}"] = cols.pop(f"f{i}")
    return ns["score_bake"](spec, pa.table(cols))


def main():
    specs = sys.argv[1:] or ["profile:b", "profile:bhdr"]
    print(f"=== lower-bound probe: corruption grid ({pq.read_metadata(CORRUPTION).num_rows} "
          f"catastrophic pairs) ===")
    print(f"{'candidate':32s}  {'min':>8s} {'p1':>7s} {'p5':>7s} {'median':>7s}  neg/N")
    for spec in specs:
        s = score(spec, CORRUPTION)
        n = len(s)
        label = spec if len(spec) <= 32 else "…" + spec[-31:]
        print(f"{label:32s}  {s.min():8.1f} {np.percentile(s,1):7.1f} {np.percentile(s,5):7.1f} "
              f"{np.median(s):7.1f}  {int((s<0).sum())}/{n}")
    print("\nread: a metric with 0 negatives here does NOT reach the low bound "
          "(broken/flattened tail); BHdr should reach strongly negative.")


if __name__ == "__main__":
    main()
