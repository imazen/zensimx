#!/usr/bin/env python3
"""Determinism gate for `train_corruption_head.py`'s baked corruption head.

## The bug this closes

`train_corruption_head.py`'s bake used to be a function of the AMBIENT
BLAS/OpenMP thread count: the identical recipe, identical data, identical
commit produced four different `corruption_head_d228.bin` files at 1/4/8/28
ambient threads (`benchmarks/corruption_head_theories_2026-09-06.md` §9). The
shipped 2026-09-05 `d228` head is the 28-thread one, so a `run-heavy --jobs 8`
re-run of the identical command did NOT reproduce it. See CLAUDE.md Known
Bugs and that doc's addendum for the full writeup.

## What this script proves

1. **Determinism** (the actual pass/fail gate): runs the owner script at
   however many `--threads` values you give it (default `1,4,8,28`, matching
   the values the bug was found at), each with that value forced into the
   AMBIENT environment (`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/...) — i.e.
   simulating a caller (a shell, `run-heavy --jobs N`, a fleet worker) that
   has already exported a thread-count opinion of its own — and asserts every
   run's baked `.bin` (and, if `--bake-extra-width` is in `--train-args`, its
   extra-width sibling) is byte-identical. This is the property the fix
   claims: the AMBIENT setting must not matter, because the owner now forces
   its own pin before numpy/sklearn import.

2. **Shipped reproduction** (informational, not a failure condition — a
   mismatch here is the DOCUMENTED, EXPECTED outcome and does not fail this
   gate): compares the new deterministic bake's sha256 against
   `--shipped-bake`. If they differ, scores `--gate-grid` (the `gb82_dog`
   held-out gate grid — 672 triples of `corruption` / `q10` / `q20` rows,
   372-wide, held out from training by construction) through BOTH the
   shipped bake and the new deterministic bake via `--predict-bin`
   (`predict_features_with_bake`, which forwards through the SAME baked
   model that ships — not the trainer's `CalibratedClassifierCV`-based
   `metrics.json`, which is a known-separate discrepancy; see CLAUDE.md
   "train_corruption_head.py reports a model it does not ship"), and prints
   the detection-rate / q10-FP / q20-FP delta at T=0.9 (score < 10, since the
   bake emits `score = 100*(1-P)`).

## Usage

    python3 scripts/v_next/corrhead_determinism_gate.py
    python3 scripts/v_next/corrhead_determinism_gate.py --threads 1,2,4 --keep

Each fit is small (~7 s, <3 GiB RSS on the canonical d228 recipe) — this is a
lightweight verification tool, not a heavy sweep; it does not need
`run-heavy`, though nothing stops you wrapping the whole invocation in one.

Exit 0 = every thread count produced a byte-identical bake (the fix holds).
Exit 1 = a divergence was found (the fix is absent, broken, or regressed).
Exit 2 = could not run at all (missing corpus/binaries/script).
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = "/mnt/v/output/zensim/corruption-head-2026-09-05"

DEFAULT_TRAIN_SCRIPT = os.path.join(HERE, "train_corruption_head.py")
DEFAULT_CORPUS = f"{ROOT}/im26_corruption_372_postC.parquet"
DEFAULT_NEGRICH = f"{ROOT}/negrich_372_postC.parquet"
DEFAULT_LADDER = ("/mnt/v/output/zensim/ladder-2026-09-05/instruments/"
                  "dial_grid_372col_ladder.parquet")
DEFAULT_GATE_GRID = f"{ROOT}/corruption_grid_372col_postC_2026-09-05.parquet"
DEFAULT_SHIPPED_BAKE = f"{ROOT}/d228/corruption_head_d228.bin"
DEFAULT_BAKE_BIN = os.path.expanduser(
    "~/work/zen/zenanalyze/target/release/zenpredict-bake")
DEFAULT_PREDICT_BIN = os.path.expanduser(
    "~/work/zen/zensimx/target/release/predict_features_with_bake")

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_one(train_script, out_dir, threads, corpus, negrich, ladder, feat_range, bake_bin):
    """Run the owner at an AMBIENT thread count and return the bake's sha256.

    `threads` is injected into the child's environment as if some external
    caller had already set it — the fix under test is that this must not
    change the result.
    """
    os.makedirs(out_dir, exist_ok=True)
    bake_out = os.path.join(out_dir, "corruption_head_d228.bin")
    cmd = [sys.executable, train_script,
          "--corpus", corpus, "--negrich", negrich,
          "--feat-range", feat_range, "--thresholds", "0.5,0.9,0.95",
          "--out", os.path.join(out_dir, "corruption_head_d228.json"),
          "--bake-out", bake_out,
          "--bake-extra-width", "944",
          "--bake-bin", bake_bin,
          "--split-out", os.path.join(out_dir, "split.tsv"),
          "--broad-honest", f"ladder:{ladder}:image_id"]
    env = dict(os.environ)
    for v in THREAD_ENV_VARS:
        env[v] = str(threads)
    log_path = os.path.join(out_dir, "train.log")
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"train_corruption_head.py failed at threads={threads} "
                         f"(rc={r.returncode}); see {log_path}")
    w944 = os.path.join(out_dir, "corruption_head_d228_w944.bin")
    return sha256_file(bake_out), (sha256_file(w944) if os.path.exists(w944) else None), bake_out


def score_bake(predict_bin, bake_path, X372):
    """Forward-pass X372 (n_rows x 372, row-major f32) through a ZNPR bake."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tf.write(struct.pack("<II", X372.shape[1], X372.shape[0]))
        tf.write(X372.astype(np.float32).tobytes(order="C"))
        feat_path = tf.name
    try:
        r = subprocess.run([predict_bin, "--bake", bake_path, "--bake-post", "raw",
                            "--features-file", feat_path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"predict_features_with_bake failed on {bake_path}: "
                             f"{r.stderr.strip()}")
        return np.array([float(x) for x in r.stdout.split()])
    finally:
        os.unlink(feat_path)


def gate_grid_comparison(gate_grid, shipped_bake, new_bake, predict_bin):
    """Detection / q10-FP / q20-FP at T=0.9 (score<10), shipped vs new, on the
    gb82_dog held-out gate grid (672 triples, held out from training by
    construction — no train/test split logic to reconstruct)."""
    t = pq.read_table(gate_grid)
    names = t.schema.names
    nfeat = 1 + max(int(c[1:]) for c in names if c.startswith("f") and c[1:].isdigit())
    X = np.column_stack([t.column(f"f{i}").to_numpy(zero_copy_only=False).astype(np.float64)
                         for i in range(nfeat)])
    entry = np.asarray(t.column("entry").to_pylist())
    is_corr = np.array([e.endswith("__corruption") for e in entry])
    is_q10 = np.array([e.endswith("__q10") for e in entry])
    is_q20 = np.array([e.endswith("__q20") for e in entry])
    assert (is_corr | is_q10 | is_q20).all(), "gate grid rows outside corruption/q10/q20"

    out = {}
    for label, bake in (("shipped", shipped_bake), ("new_deterministic", new_bake)):
        raw = score_bake(predict_bin, bake, X)
        fires = raw < 10.0  # score = 100*(1-P); P>0.9 <=> score<10
        out[label] = {
            "detection_at_T0.9": float(fires[is_corr].mean()),
            "fp_q10_at_T0.9": float(fires[is_q10].mean()),
            "fp_q20_at_T0.9": float(fires[is_q20].mean()),
            "n_corruption": int(is_corr.sum()), "n_q10": int(is_q10.sum()),
            "n_q20": int(is_q20.sum()),
        }
    out["delta_new_minus_shipped"] = {
        k: out["new_deterministic"][k] - out["shipped"][k]
        for k in ("detection_at_T0.9", "fp_q10_at_T0.9", "fp_q20_at_T0.9")
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-script", default=DEFAULT_TRAIN_SCRIPT)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--negrich", default=DEFAULT_NEGRICH)
    ap.add_argument("--ladder", default=DEFAULT_LADDER)
    ap.add_argument("--feat-range", default="0:228")
    ap.add_argument("--bake-bin", default=DEFAULT_BAKE_BIN)
    ap.add_argument("--predict-bin", default=DEFAULT_PREDICT_BIN)
    ap.add_argument("--gate-grid", default=DEFAULT_GATE_GRID)
    ap.add_argument("--shipped-bake", default=DEFAULT_SHIPPED_BAKE)
    ap.add_argument("--threads", default="1,4,8,28",
                    help="comma-separated ambient thread counts to test (default: "
                         "the exact values the bug was found + measured at)")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/tmp/corrhead_determinism_gate"))
    ap.add_argument("--keep", action="store_true", help="don't delete --out-dir on exit")
    ap.add_argument("--skip-gate-comparison", action="store_true",
                    help="skip step 2 (shipped-vs-new detection/FP delta); "
                         "just run the byte-identity check")
    a = ap.parse_args()

    for label, path in (("--train-script", a.train_script), ("--corpus", a.corpus),
                        ("--negrich", a.negrich), ("--ladder", a.ladder),
                        ("--bake-bin", a.bake_bin)):
        if not os.path.exists(path):
            print(f"SKIP (could not run): {label} not found: {path}", file=sys.stderr)
            return 2

    threads = [int(x) for x in a.threads.split(",")]
    if len(threads) < 2:
        print("SKIP (could not run): need at least 2 --threads values to test invariance",
             file=sys.stderr)
        return 2

    if os.path.exists(a.out_dir) and not a.keep:
        shutil.rmtree(a.out_dir)
    os.makedirs(a.out_dir, exist_ok=True)

    print(f"corrhead_determinism_gate: fitting at ambient threads {threads} "
         f"(feat-range {a.feat_range})")
    results = {}
    for n in threads:
        d = os.path.join(a.out_dir, f"t{n}")
        sha_bin, sha_w944, bake_path = run_one(a.train_script, d, n, a.corpus, a.negrich,
                                               a.ladder, a.feat_range, a.bake_bin)
        results[n] = (sha_bin, sha_w944, bake_path)
        print(f"  threads={n:>3}  d228.bin={sha_bin[:16]}…  w944.bin="
             f"{(sha_w944 or 'n/a')[:16]}…")

    shas_bin = {n: r[0] for n, r in results.items()}
    shas_w944 = {n: r[1] for n, r in results.items()}
    ok_bin = len(set(shas_bin.values())) == 1
    ok_w944 = len(set(v for v in shas_w944.values() if v is not None)) <= 1

    if ok_bin and ok_w944:
        print(f"PASS: byte-identical across all {len(threads)} thread counts "
             f"(d228.bin sha256 {next(iter(shas_bin.values()))})")
    else:
        print("FAIL: bake differs across thread counts — determinism is broken:")
        for n in threads:
            print(f"  threads={n}: d228.bin={shas_bin[n]}  w944.bin={shas_w944[n]}")

    if not a.skip_gate_comparison:
        if not (os.path.exists(a.gate_grid) and os.path.exists(a.shipped_bake)
               and os.path.exists(a.predict_bin)):
            print("(gate-grid comparison SKIPPED: --gate-grid/--shipped-bake/--predict-bin "
                 "not all present)")
        else:
            new_bake = results[threads[0]][2]
            new_sha = shas_bin[threads[0]]
            shipped_sha = sha256_file(a.shipped_bake)
            print(f"\nshipped d228.bin sha256:            {shipped_sha}")
            print(f"new deterministic d228.bin sha256:  {new_sha}")
            if new_sha == shipped_sha:
                print("shipped reproduction: MATCH (the new deterministic build "
                     "byte-reproduces the shipped artifact)")
            else:
                print("shipped reproduction: DIFFERS (documented/expected — a "
                     "single-thread-equivalent reduction order differs from the "
                     "historical unpinned 28-thread run). Scoring the gb82_dog "
                     "held-out gate grid through both, via the actual baked bytes:")
                cmp = gate_grid_comparison(a.gate_grid, a.shipped_bake, new_bake, a.predict_bin)
                print(json.dumps(cmp, indent=2))

    if not a.keep:
        shutil.rmtree(a.out_dir, ignore_errors=True)

    return 0 if (ok_bin and ok_w944) else 1


if __name__ == "__main__":
    raise SystemExit(main())
