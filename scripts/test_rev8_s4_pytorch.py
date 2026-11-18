#!/usr/bin/env python3
"""Parity and property tests comparing `rev8_s4_pytorch.py` against Rust `zensim::rev8`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev8_s4_pytorch as pt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE = REPO / "target/release/examples/rev8_mixf3_pilot"
SHAPES = [(1, 1), (17, 13), (64, 64), (256, 256)]
RECIPES = ["identity", "uniform_shift", "checker_gradient", "asymmetric_blobs"]


def synth_pair_np(w: int, h: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic uint8 mirror of `rev8_mixf3_pilot::synth_pair`."""
    a = np.empty((h, w, 3), dtype=np.uint8)
    b = np.empty((h, w, 3), dtype=np.uint8)
    state = (
        ((w * 73856093) & 0xFFFFFFFF)
        ^ ((h * 19349663) & 0xFFFFFFFF)
        ^ 0x9E3779B9
    ) & 0xFFFFFFFF
    for y in range(h):
        for x in range(w):
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= (state >> 17) & 0xFFFFFFFF
            state ^= (state << 5) & 0xFFFFFFFF
            n0 = (state >> 24) & 0xFF
            base = (
                min(255, ((x * 255) // max(w, 1)) + (n0 & 15)),
                min(255, ((y * 255) // max(h, 1)) + (((x * 7) & 0xFF) & 15)),
                (((x + y) * 19) + (n0 >> 3)) & 0xFF,
            )
            if kind == "identity":
                pa = pb = base
            elif kind == "uniform_shift":
                pa = base
                pb = (min(255, base[0] + 9), min(255, base[1] + 7), min(255, base[2] + 11))
            elif kind == "checker_gradient":
                cb = 220 if (x + y) % 2 == 0 else 36
                pa = (cb, base[1], (x * 13 + y * 5) & 0xFF)
                pb = (
                    (pa[0] // 12) * 12,
                    max(0, pa[1] - (6 if ((x // 8 + y // 8) % 2 == 0) else 0)),
                    min(255, pa[2] + 4),
                )
            elif kind == "asymmetric_blobs":
                pa = (
                    210 if ((x // 4 + y // 4) % 2 == 0) else 40,
                    (x * 11 + y * 17) & 0xFF,
                    (y * 29) & 0xFF,
                )
                in_blob = (x >= w // 4) and (x < (3 * w) // 4) and (y >= h // 4) and (y < (3 * h) // 4)
                if in_blob:
                    pb = (max(0, pa[0] - 35), min(255, pa[1] + 24), max(0, pa[2] - 28))
                else:
                    pb = ((pa[0] // 8) * 8, pa[1], pa[2])
            else:
                raise ValueError(f"unknown kind {kind}")
            a[y, x] = pa
            b[y, x] = pb
    return a, b


def run_oracle(oracle: Path, args: list[str]) -> dict:
    raw = subprocess.check_output([str(oracle), "oracle-json", *args], text=True)
    return json.loads(raw)


def check_case(
    label: str,
    oracle_data: dict,
    a_np: np.ndarray,
    b_np: np.ndarray,
    tol: float,
    is_identity: bool = False,
) -> dict[str, float]:
    ta = torch.from_numpy(np.ascontiguousarray(a_np))
    tb = torch.from_numpy(np.ascontiguousarray(b_np))
    summary = {}
    for nits, key in ((255.0, "nits255"), (500.0, "nits500")):
        ref = oracle_data[key]
        assert ref["swap_exact_bits"], f"{label} {key}: Rust swap not bit-exact"
        res_ab = pt.extract_rev8_s4(ta, tb, nits=nits)
        res_ba = pt.extract_rev8_s4(tb, ta, nits=nits)

        feat_ab = res_ab.features.numpy()
        feat_ba = res_ba.features.numpy()
        frag_ab = res_ab.scale_fragility.numpy()
        frag_ba = res_ba.scale_fragility.numpy()
        fx_ab = res_ab.fx2_vector().numpy()
        fx_ba = res_ba.fx2_vector().numpy()

        swap_feat_diff = float(np.max(np.abs(feat_ab - feat_ba)))
        swap_frag_diff = float(np.max(np.abs(frag_ab - frag_ba)))
        swap_fx_diff = float(np.max(np.abs(fx_ab - fx_ba)))
        assert swap_feat_diff == 0.0, f"{label} {key}: PyTorch swap feat diff {swap_feat_diff}"
        assert swap_frag_diff == 0.0, f"{label} {key}: PyTorch swap frag diff {swap_frag_diff}"
        assert swap_fx_diff == 0.0, f"{label} {key}: PyTorch swap fx2 diff {swap_fx_diff}"

        if is_identity:
            assert float(np.max(np.abs(feat_ab))) == 0.0, f"{label} {key}: nonzero identity features"
            assert float(np.max(np.abs(fx_ab))) == 0.0, f"{label} {key}: nonzero identity fx2"
            assert np.all((frag_ab > 0.0) & (frag_ab <= 1.0)), f"{label} {key}: frag out of (0,1]"

        rust_feat = np.asarray(ref["features"], dtype=np.float64)
        rust_frag = np.asarray(ref["scale_fragility"], dtype=np.float64)
        rust_fx = np.asarray(ref["fx2_480"], dtype=np.float64)

        df = float(np.max(np.abs(feat_ab - rust_feat)))
        dg = float(np.max(np.abs(frag_ab - rust_frag)))
        dx = float(np.max(np.abs(fx_ab - rust_fx)))
        assert df < tol, f"{label} {key}: features max_abs_diff={df:.3e} >= {tol}"
        assert dg < tol, f"{label} {key}: fragility max_abs_diff={dg:.3e} >= {tol}"
        assert dx < tol, f"{label} {key}: fx2 max_abs_diff={dx:.3e} >= {tol}"
        summary[f"{key}_max_feat"] = df
        summary[f"{key}_max_frag"] = dg
        summary[f"{key}_max_fx2"] = dx
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument(
        "--image-pair",
        nargs=2,
        action="append",
        default=[],
        metavar=("REF", "DIST"),
        help="Optional real image pair to include in parity verification",
    )
    args = ap.parse_args()
    if not args.oracle.is_file():
        raise SystemExit(
            f"Oracle binary not found at {args.oracle}. Build with:\n"
            "  cargo build --release --locked -p zensim-validate --features rev8-proto --example rev8_mixf3_pilot"
        )

    worst = {"255_feat": 0.0, "255_frag": 0.0, "255_fx2": 0.0, "500_feat": 0.0, "500_frag": 0.0, "500_fx2": 0.0}
    cases_run = 0

    for w, h in SHAPES:
        for kind in RECIPES:
            a_np, b_np = synth_pair_np(w, h, kind)
            ref = run_oracle(args.oracle, ["--synth", f"{w}x{h}:{kind}"])
            s = check_case(f"{w}x{h}:{kind}", ref, a_np, b_np, args.tol, is_identity=(kind == "identity"))
            cases_run += 1
            worst["255_feat"] = max(worst["255_feat"], s["nits255_max_feat"])
            worst["255_frag"] = max(worst["255_frag"], s["nits255_max_frag"])
            worst["255_fx2"] = max(worst["255_fx2"], s["nits255_max_fx2"])
            worst["500_feat"] = max(worst["500_feat"], s["nits500_max_feat"])
            worst["500_frag"] = max(worst["500_frag"], s["nits500_max_frag"])
            worst["500_fx2"] = max(worst["500_fx2"], s["nits500_max_fx2"])
            print(
                f"PASS {w:>3}x{h:<3} {kind:<16} | "
                f"255nit diff=(feat {s['nits255_max_feat']:.2e}, frag {s['nits255_max_frag']:.2e}) | "
                f"500nit diff=(feat {s['nits500_max_feat']:.2e}, frag {s['nits500_max_frag']:.2e})"
            )

    for rp, dp in args.image_pair:
        a_np = np.asarray(Image.open(rp).convert("RGB"), dtype=np.uint8)
        b_np = np.asarray(Image.open(dp).convert("RGB"), dtype=np.uint8)
        ref = run_oracle(args.oracle, ["--images", rp, dp])
        s = check_case(f"img:{Path(dp).name}", ref, a_np, b_np, args.tol)
        cases_run += 1
        worst["255_feat"] = max(worst["255_feat"], s["nits255_max_feat"])
        worst["255_frag"] = max(worst["255_frag"], s["nits255_max_frag"])
        worst["255_fx2"] = max(worst["255_fx2"], s["nits255_max_fx2"])
        worst["500_feat"] = max(worst["500_feat"], s["nits500_max_feat"])
        worst["500_frag"] = max(worst["500_frag"], s["nits500_max_frag"])
        worst["500_fx2"] = max(worst["500_fx2"], s["nits500_max_fx2"])
        print(
            f"PASS img {Path(dp).name:<24} | "
            f"255nit diff=(feat {s['nits255_max_feat']:.2e}, frag {s['nits255_max_frag']:.2e}) | "
            f"500nit diff=(feat {s['nits500_max_feat']:.2e}, frag {s['nits500_max_frag']:.2e})"
        )

    print(
        f"\nALL {cases_run} PARITY CASES PASSED (tol={args.tol:.1e}):\n"
        f"  worst 255-nit: feat={worst['255_feat']:.3e}, frag={worst['255_frag']:.3e}, fx2={worst['255_fx2']:.3e}\n"
        f"  worst 500-nit: feat={worst['500_feat']:.3e}, frag={worst['500_frag']:.3e}, fx2={worst['500_fx2']:.3e}"
    )


if __name__ == "__main__":
    main()
