#!/usr/bin/env python3
"""Co-calibrate BHdr's output-calibration spline onto B's product scale.

WHY (bhdr_improvement_split_lineage §7.4 / §8.27–§8.29): the shipped BHdr dial
was fit on SDR-shell features forwarded through the HDR weights — the wrong
feature regime (doubly out-of-domain). This re-fits ONLY the spline on the raw
outputs BHdr actually produces on the **PU-linear (203-nit)** path, mapped to
**B's native dial on the same content**, so `BHdr(203-nit SDR) ≈ B(SDR)` — one
product scale across the SDR/HDR seam. WEIGHTS ARE UNCHANGED → rank-invariant →
every SROCC (incl. UPIQ 0.7536) is identical; only the dial VALUES move.

NEGATIVES ARE PRESERVED (user constraint, load-bearing): `fit_spline_knots`
places the bottom knot at the anchor's low-raw percentile and the runtime spline
extrapolates linearly (uncapped, clamped only at −100 by metric.rs) BELOW it. We
do NOT add a `(floor, 0)` clamp knot (that is exactly the withdrawn
`bottom-extend`), so content worse than the anchor floor still scores negative.

Self-validates: rebakes the SHIPPED spline (original anchor) → must equal
`7d7f2123` before emitting the co-cal bake, proving the rebake path is
byte-faithful to `lp.bake_candidate` (only the spline differs).

  usage: bhdr_dial_cocal.py --out <bake.bin> [--repro-only]
"""
import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path.home() / "work/zen/zensimx"
PROBE = Path("/mnt/v/output/zensim-multicodec-probe")
COCAL = Path("/mnt/v/output/zensim/reports/bhdr_cocal")
SHIPPED = REPO / "zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin"
SHIPPED_SHA = "7d7f212369f734aa9de072f84ac0e8b97b86deefa3c8bfe26de94de6b49b7ce4"
FIT = PROBE / "linear-probe/fits/hdrmix-lasso0.0003-shaped.npz"
ANCHOR = PROBE / "linear-probe/val/anchor.npz"

# import the linear-probe machinery (constants + shaping + spline + baker path)
_spec = importlib.util.spec_from_file_location(
    "lp", REPO / "scripts/v_next/linear_projections_2026-07-03.py")
lp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lp)
N_FEAT, BAKER, SPLINE_KEY = lp.N_FEAT, lp.BAKER, lp.SPLINE_KEY


def load_bhdr_fit():
    z = np.load(FIT)
    w = z["w"].astype(np.float64).copy()
    w16 = w.astype(np.float16).astype(np.float64)   # f16 pack, tau=0 (== bake_candidate)
    return w16, float(z["bias"]), z["mu"], z["sd"], str(z["space"])


def rebake_with_spline(cx, cy, w16, bias, mu, sd, out_path: Path) -> str:
    """Mirror lp.bake_candidate's payload EXACTLY; only the spline (cx,cy) varies."""
    payload = struct.pack("<I", len(cx)) + b"".join(
        struct.pack("<ff", x, y) for x, y in zip(cx, cy))
    transforms, tparams = lp.load_transforms()
    metadata = [
        {"key": "zentrain.feature_transforms", "type": "utf8",
         "text": "\n".join(transforms)},
        {"key": "zentrain.feature_transform_params", "type": "utf8",
         "text": "\n".join(",".join(f"{p}" for p in row) if row else "" for row in tparams)},
        {"key": SPLINE_KEY, "type": "bytes", "hex": payload.hex()},
    ]
    req = {
        "schema_hash": 0, "flags": 0, "compressed": True,
        "scaler_mean": [float(v) for v in mu.astype(np.float32)],
        "scaler_scale": [float(v) for v in sd.astype(np.float32)],
        "layers": [{
            "in_dim": N_FEAT, "out_dim": 1, "activation": "identity", "dtype": "f16",
            "weights": [float(v) for v in w16], "biases": [bias],
        }],
        "metadata": metadata,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(req, f)
        jp = Path(f.name)
    try:
        r = subprocess.run([str(BAKER), "bake", str(jp), str(out_path)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"zenpredict bake failed: {r.stderr[:400]}")
    finally:
        jp.unlink(missing_ok=True)
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def read_feats(path: Path):
    t = pq.read_table(path)
    X = np.stack([np.asarray(t[f"feat_{i}"], dtype=np.float64) for i in range(N_FEAT)], axis=1)
    q = np.array([int(float(x)) for x in t["q"].to_pylist()])
    return X, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(COCAL / "bhdr_cocal_203nit_bscale.bin"))
    ap.add_argument("--repro-only", action="store_true")
    a = ap.parse_args()
    w16, bias, mu, sd, space = load_bhdr_fit()
    assert space == "shaped", space
    transforms, tparams = lp.load_transforms()

    # ---- self-validation: reproduce the shipped spline from the original anchor
    az = np.load(ANCHOR)
    raw_orig = (az["shaped"].astype(np.float64) - mu) / sd @ w16 + bias
    cx0, cy0 = lp.fit_spline_knots(raw_orig, az["y"].astype(np.float64))
    tmp = Path(tempfile.mktemp(suffix=".bin"))
    sha0 = rebake_with_spline(cx0, cy0, w16, bias, mu, sd, tmp)
    ok = (sha0 == SHIPPED_SHA)
    print(f"[self-check] rebake original-anchor spline: sha {sha0[:12]} "
          f"{'== shipped ✅' if ok else '!= shipped ❌'} ({len(cx0)} knots [{cy0[0]:.2f},{cy0[-1]:.2f}])")
    tmp.unlink(missing_ok=True)
    if not ok:
        raise SystemExit("rebake path is NOT byte-faithful — aborting before co-cal")
    if a.repro_only:
        return

    # ---- co-calibration by Y-REMAP of the shipped spline.
    # We do NOT recompute raw in Python (shape_block diverges ~0.07 SROCC from the
    # Rust runtime on out-of-range content). Instead we KEEP the shipped spline's
    # raw knot positions cx0 (the runtime's own raw scale) and remap only its
    # Y-values through a 1-D monotone map f: shipped_dial → B_dial, fit on the
    # runtime's OWN outputs. Positive knots adopt B's product scale (seam fix);
    # the bottom knot is PINNED at 0 so the negative extrapolation is byte-for-byte
    # the shipped behaviour — valid negatives preserved by construction.
    bt = pq.read_table(COCAL / "upiq_sdr_u8shell_bdial.parquet",
                       columns=["q", "score_b"]).to_pydict()
    st = pq.read_table(COCAL / "upiq_sdr_pl203_shipped.parquet",
                       columns=["q", "score_bhdr"]).to_pydict()
    q_b = np.array([int(float(x)) for x in bt["q"]])
    q_s = np.array([int(float(x)) for x in st["q"]])
    common = np.intersect1d(q_b, q_s)
    ob = {q: i for i, q in enumerate(q_b)}
    os_ = {q: i for i, q in enumerate(q_s)}
    b_dial = np.array([bt["score_b"][ob[q]] for q in common], dtype=np.float64)
    ship = np.array([st["score_bhdr"][os_[q]] for q in common], dtype=np.float64)

    # f fit on the POSITIVE product range only (B never models negatives).
    pos = ship > 0
    fx, fy = lp.fit_spline_knots(ship[pos], b_dial[pos])  # monotone shipped→B map
    fx = np.asarray(fx); fy = np.asarray(fy)
    # remap each shipped knot; PIN the bottom knot at 0 → negatives extrapolate
    # exactly as the shipped bake (cy0[0] == 0).
    new_cy = np.interp(cy0, fx, fy)
    new_cy[0] = 0.0
    new_cy = np.maximum.accumulate(new_cy)  # enforce monotone non-decreasing
    out = Path(a.out)
    sha1 = rebake_with_spline(cx0, new_cy.tolist(), w16, bias, mu, sd, out)

    print(f"[co-cal] f (shipped→B) fit on {int(pos.sum())} positive cells; "
          f"shipped→B example: 20→{np.interp(20,fx,fy):.1f}, 50→{np.interp(50,fx,fy):.1f}, "
          f"80→{np.interp(80,fx,fy):.1f}")
    print(f"[co-cal] Y-remap: {len(cx0)} knots, cx UNCHANGED (runtime raw scale); "
          f"cy [{cy0[0]:.2f},{cy0[-1]:.2f}] → [{new_cy[0]:.2f},{new_cy[-1]:.2f}]  "
          f"(bottom knot pinned 0 → negatives preserved)")
    print(f"[co-cal] wrote {out} ({out.stat().st_size} B, sha {sha1[:12]})")


if __name__ == "__main__":
    main()
