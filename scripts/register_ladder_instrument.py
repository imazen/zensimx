#!/usr/bin/env python3
"""Derive `peer_ssim2`'s bars on a NEW dial grid and APPEND them to the floor registry.

The bars are DERIVED THROUGH THE OWNER (`bake_verdict --dial-peer-scores` +
`--gaddr-json`) and copied verbatim. This script computes no statistic of its own —
if it did, the registry would hold a number the grading path cannot reproduce.

APPEND-ONLY, and enforced: a row whose `(dial_grid_sha256, reference)` key already
exists is REFUSED, never rewritten. An unregistered grid grades NOT-MEASURABLE, which
is the property that stops a candidate dodging a bar by picking a friendlier
instrument — so silently replacing a row would be worse than adding none.

Usage:
  register_ladder_instrument.py --grid <parquet> --truth <ssim2 cells.tsv> \
      --label <text> [--registry <json>] [--bake <any 372 bake>] [--dry-run]
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, tempfile
from datetime import datetime, timezone

# Resolve everything relative to THIS FILE's checkout, never a hard-coded repo
# root: this script is run from jj sibling workspaces, and a hard-coded
# `~/work/zen/zensimx` made it append to the PRIMARY checkout's registry — which was
# sitting on an older commit, so the append landed on a stale copy of the file.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG_DEFAULT = os.path.join(REPO, "benchmarks/dial_addressability_floor_2026-09-04.json")


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def derive(bv: str, bake: str, grid: str, truth: str, out: str,
           regime: str = "372", features_root: str | None = None) -> dict:
    """peer mode: the scorer described is the REFERENCE metric, not `--bake`."""
    cmd = [bv, "--bake", bake, "--corpora", "cid22", "--dial-grid", grid]
    if features_root is not None:
        cmd += ["--features-root", features_root,
                "--corruption-grid", os.path.join(features_root, "not-requested-corruption.parquet"),
                "--perpair-metrics", os.path.join(features_root, "not-requested-perpair.parquet")]
    if regime != "372":
        cmd += ["--regime", regime]
    cmd += [
           "--gaddr-grid-truth", truth,
           "--dial-peer-scores", f"peer_ssim2={truth}",
           "--gaddr-json", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"bake_verdict failed rc={r.returncode}")
    d = json.load(open(out))
    # `--gaddr-json` IS the G-ADDR block (not nested under `dial`). Its `scorer`
    # says WHOSE numbers these are; registering a candidate's as the mentor's bar
    # would silently pin the gate to the thing being graded.
    sc = d.get("scorer") or {}
    if sc.get("kind") != "peer" or sc.get("label") != "peer_ssim2":
        raise SystemExit(f"--gaddr-json describes scorer={sc!r}, not peer_ssim2 — "
                         f"refusing to register a candidate's numbers as the bar")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--registry", default=REG_DEFAULT)
    ap.add_argument("--bake", default=os.path.join(
        REPO, "zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin"))
    ap.add_argument("--bv", default=os.path.join(REPO, "target/release/bake_verdict"))
    ap.add_argument("--regime", default="372", help="372 / 720 / 944 — the grid's width")
    ap.add_argument("--features-root", help="explicit admitted corpus root for the reference run")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    registered = datetime.now(timezone.utc).date().isoformat()

    gsha = sha256(a.grid)
    reg = json.load(open(a.registry))
    for row in reg.get("grids", []):
        if row["dial_grid_sha256"] == gsha:
            raise SystemExit(f"grid {gsha[:16]} is ALREADY registered as {row['label']!r} "
                             f"— this registry is append-only; refusing")
    for row in reg.get("grid_floor_representability", []):
        if row["dial_grid_sha256"] == gsha and row["reference"] == "peer_ssim2":
            raise SystemExit(f"(grid {gsha[:16]}, peer_ssim2) already has a floor row "
                             f"— refusing")

    work = tempfile.mkdtemp(prefix="ladderreg_")
    d = derive(a.bv, a.bake, a.grid, a.truth, os.path.join(work, "peer.json"), a.regime, a.features_root)
    meas = d["measured"]
    dial = meas["grid"]
    import pyarrow.parquet as pq
    n_rows = pq.read_metadata(a.grid).num_rows

    # `reference` is LOAD-BEARING on a grids row: the A1-A6 lookup keys on
    # (dial_grid_sha256, reference), and a row without it never resolves — the grid
    # reads "not in the G-ADDR floor registry" even though a row exists for its sha.
    grid_row = {"dial_grid_sha256": gsha, "reference": "peer_ssim2",
                "instrument_kind": "floor-dense-ladder",
                "label": a.label, "path": os.path.abspath(a.grid),
                "n_rows": n_rows,
                **{k: dial[k] for k in ("min", "max", "p5", "p95", "reach",
                                        "dynamic_range", "mono", "tied")},
                "registered": registered, "active": True}
    floor_row = {
        "dial_grid_sha256": gsha, "reference": "peer_ssim2",
        "label": f"{a.label} -- per-CODEC FLOOR REPRESENTABILITY",
        "path": os.path.abspath(a.grid), "bottom_k": 3,
        "floor_rule": d["floor_rule"],
        "floor_margin": d["floor_rule_params"]["margin"],
        "registered": registered, "active": True,
        "codecs": [{"codec": c["codec"], "n_ladders": c["n_ladders"],
                    "represented_frac": c["represented_frac"]}
                   for c in meas["codec_floor"]],
        "grid_truth_sha256": sha256(a.truth),
        "note": ("THE BAR for this instrument. `peer_ssim2`'s own floor representability, "
                 "derived through the owner (`bake_verdict --dial-peer-scores "
                 "peer_ssim2=<cells> --gaddr-json`), copied verbatim. The floor here is each "
                 "codec's lowest DISTINCT settings: saturated steps (identical encoded bytes) "
                 "are removed from the instrument, so jpeg's bottom three are three SETTINGS "
                 "rather than three samples of one. `avif-svt` and `avif-rav1e` are separate "
                 "ladders — different encoders, different quantizer mappings."),
    }
    print(json.dumps({"grid": grid_row, "floor": floor_row}, indent=1))
    if a.dry_run:
        print("\n--dry-run: registry NOT written"); return
    reg.setdefault("grids", []).append(grid_row)
    reg.setdefault("grid_floor_representability", []).append(floor_row)
    with open(a.registry, "w") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nappended 2 rows to {a.registry}")


if __name__ == "__main__":
    main()
