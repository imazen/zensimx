#!/usr/bin/env python3
"""Comprehensible pure-PyTorch reference implementation of zensim Rev8 Alg S4.

Symmetric `Rev8 Alg S4` corresponds to the symmetric `Moments` feature family
(in contrast to the directed `AsyMoments` features in Rev7).

Computes all 240 symmetric per-(scale, channel) distortion features across
4 dyadic scales (1x, 1/2x, 1/4x, 1/8x) in positive XYB (255 or 500 nits),
plus the 12 symmetric scale-channel gradient fragility scalars (`fx2`):

    fx2_fragility[s, c] = 0.02 / (mean(0.5 * (g_a + g_b)) + 0.02) in (0, 1]

Notes on slot names matching `zensim::rev8::FEATURE_NAMES`:
  Slot  0  L1          mean(l)              local mean luminance difference
  Slot  1  L2          mean(l^2)            second moment of l (unrooted)
  Slot  2  L4          mean(l^4)            fourth moment of l (unrooted)
  Slot  3  D21         mean(structure)      local structure/contrast loss
  Slot  4  D42         mean(structure^2)    second moment of structure (unrooted)
  Slot  5  SD2         mean(damage^2)       second moment of combined damage
  Slot  6  SD4         mean(damage^4)       fourth moment of combined damage
  Slot  7  MAE         mean(|a - b|)        mean absolute pixel error
  Slot  8  MSE         mean((a - b)^2)      mean squared pixel error
  Slot  9  GMS         mean(dis(ga, gb))    gradient magnitude dissimilarity
  Slot 10  edge_width  mean(|w_a - w_b|)    normalized curvature/gradient ratio gap
  Slot 11  BK          mean(|bk_a - bk_b|)  8-pixel codec lattice boundary jump gap
  Slot 12  RG          mean(|rg_a - rg_b|)  slope-reversal ringing gap
  Slot 13  Md          mean(damage * Md)    weak contrast masking (1/sigma, th=0.03)
  Slot 14  Me          mean(damage * Me)    medium contrast masking (1/sigma^2, th=0.05)
  Slot 15  Mf          mean(damage * Mf)    strong contrast masking (1/sigma^4, th=0.10)
  Slot 16  ART         mean(delta * exp(-low/0.045))        smooth-region artifact proxy
  Slot 17  DET         mean((delta/high) * (1-exp(-low/0.08))) textured detail-change proxy
  Slot 18  fragility   mean(max(ra/(ga+ra+C2), rb/(gb+rb+C2)) * damage)  ringing fragility
  Slot 19  SD8         mean(damage^8)       eighth moment of combined damage (unrooted)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Positive XYB constants matching jpegli / zensim.
_OPSIN = (
    (0.30, 0.622, 0.078),
    (0.23, 0.692, 0.078),
    (0.24342269, 0.20476745, 1.0 - 0.24342269 - 0.20476745),
)
_K_B0 = 0.0037930734
_OFFSETS = (0.42, 0.01, 0.55)  # X, Y, B shift applied on XYB color conversion
_WINDOW_CENTERS = (0.42, 0.45, 0.55)  # Rebasing centers for numerically clean 11x11 variances
_C2 = 0.0009
_EPS = 1e-12
_C_FRAG = 0.02
_RADIUS = 5
_WIN = 2 * _RADIUS + 1

ALG_S4_FEATURE_NAMES = (
    "L1",
    "L2",
    "L4",
    "D21",
    "D42",
    "SD2",
    "SD4",
    "MAE",
    "MSE",
    "GMS",
    "edge_width",
    "BK",
    "RG",
    "Md",
    "Me",
    "Mf",
    "ART",
    "DET",
    "fragility",
    "SD8",
)


@dataclass(frozen=True)
class Rev8S4Result:
    """Extracted Rev8 Alg S4 features (shape [4, 3, 20]) and fx2 fragilities ([4, 3])."""

    features: torch.Tensor
    scale_fragility: torch.Tensor

    def fx2_vector(self) -> torch.Tensor:
        """Flatten to 480-element `fx2` input vector: [240 direct, 240 direct * frag]."""
        direct = self.features.reshape(-1)
        frag = self.scale_fragility.unsqueeze(-1).expand(4, 3, 20).reshape(-1)
        return torch.cat([direct, direct * frag], dim=0)


def _srgb8_lut(device: torch.device) -> torch.Tensor:
    u = torch.arange(256, dtype=torch.float64, device=device) / 255.0
    lin = torch.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)
    return lin.to(torch.float32)


def srgb8_to_positive_xyb(rgb8: torch.Tensor, nits: float = 255.0) -> torch.Tensor:
    """Convert uint8 sRGB image `(3, H, W)` or `(H, W, 3)` -> float32 positive XYB `(3, H, W)`."""
    if rgb8.ndim != 3:
        raise ValueError(f"expected 3D RGB image tensor, got {tuple(rgb8.shape)}")
    if rgb8.shape[-1] == 3 and rgb8.shape[0] != 3:
        rgb8 = rgb8.permute(2, 0, 1)
    if rgb8.shape[0] != 3:
        raise ValueError(f"expected 3 channels, got {tuple(rgb8.shape)}")

    lut = _srgb8_lut(rgb8.device)
    lin = lut[rgb8.to(torch.long)]  # (3, H, W)
    r, g, b = lin[0], lin[1], lin[2]

    mul = nits / 255.0
    bias_cbrt = _K_B0 ** (1.0 / 3.0)
    scale = (
        1.0
        if nits == 255.0
        else (((1.0 + _K_B0) ** (1.0 / 3.0) - bias_cbrt) / ((mul + _K_B0) ** (1.0 / 3.0) - bias_cbrt))
    )

    m = torch.tensor(_OPSIN, dtype=torch.float32, device=lin.device) * mul
    lms = []
    for i in range(3):
        mix = m[i, 0] * r + m[i, 1] * g + m[i, 2] * b + _K_B0
        lms.append(scale * (torch.sign(mix) * mix.abs().pow(1.0 / 3.0) - bias_cbrt))
    l, mm, s = lms

    y = 0.5 * (l + mm)
    x = 7.0 * (l - mm) + _OFFSETS[0]
    black = (r + g + b) == 0.0
    yc = torch.where(black, torch.full_like(y, _OFFSETS[1]), y + _OFFSETS[1])
    bc = torch.where(black, torch.full_like(s, _OFFSETS[2]), (s - y) + _OFFSETS[2])
    return torch.stack([x, yc, bc], dim=0)


def reflect101_coords(n: int, r: int, device: torch.device) -> torch.Tensor:
    """Periodic reflect-101 index map of length `n + 2*r` (safe even when `n == 1` or `n <= r`)."""
    if n <= 1:
        return torch.zeros(n + 2 * r, dtype=torch.long, device=device)
    period = 2 * (n - 1)
    idx = torch.arange(-r, n + r, device=device)
    phase = torch.remainder(idx, period)
    return torch.minimum(phase, period - phase)


def reflect101_pad_2d(x: torch.Tensor, r: int) -> torch.Tensor:
    """Pad `(*, H, W)` by `r` on both spatial axes using periodic reflect-101."""
    h, w = x.shape[-2], x.shape[-1]
    yi = reflect101_coords(h, r, x.device)
    xi = reflect101_coords(w, r, x.device)
    return x.index_select(-2, yi).index_select(-1, xi)


def dyadic_pyramid_4(xyb: torch.Tensor) -> list[torch.Tensor]:
    """Build 4 dyadic scales via 2x2 averaging; odd trailing edges replicate the last sample."""
    scales = [xyb]
    cur = xyb
    for _ in range(3):
        _, h, w = cur.shape
        if h % 2 == 1:
            cur = torch.cat([cur, cur[:, -1:, :]], dim=1)
        if w % 2 == 1:
            cur = torch.cat([cur, cur[:, :, -1:]], dim=2)
        cur = F.avg_pool2d(cur.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
        scales.append(cur)
    return scales


def local_moments_11x11(
    a: torch.Tensor, b: torch.Tensor, center: float
) -> tuple[torch.Tensor, ...]:
    """Compute sliding 11x11 local means and population variances for one `(H, W)` plane pair.

    Error `e = a - b` is formed *before* rebasing around `center`, moments are accumulated
    in float64 cleanly over the 11x11 window, and variances are clamped at zero so flat regions
    never produce negative variance NaNs.
    """
    e = (a - b).to(torch.float64)
    ac = (a - center).to(torch.float64)
    bc = (b - center).to(torch.float64)
    stack = torch.stack([ac, bc, e, ac * ac, bc * bc, e * e], dim=0)  # (6, H, W)
    padded = reflect101_pad_2d(stack, _RADIUS).unsqueeze(1)  # (6, 1, H+10, W+10)
    moments = F.avg_pool2d(padded, kernel_size=_WIN, stride=1).squeeze(1)
    ma, mb, me, ma2, mb2, me2 = moments.unbind(0)

    mean_a = (ma + center).to(torch.float32)
    mean_b = (mb + center).to(torch.float32)
    mean_e = me.to(torch.float32)
    va = (ma2 - ma * ma).clamp_min(0.0).to(torch.float32)
    vb = (mb2 - mb * mb).clamp_min(0.0).to(torch.float32)
    ve = (me2 - me * me).clamp_min(0.0).to(torch.float32)
    return mean_a, mean_b, mean_e, va, vb, ve


def stencil_terms(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract gradient magnitude `g`, curvature, 8-px block boundary jump `bk`, and ringing `rg`."""
    h, w = x.shape
    p = reflect101_pad_2d(x, 1)
    c = p[1:-1, 1:-1]
    dx = p[1:-1, 2:] - c
    dy = p[2:, 1:-1] - c
    bx = c - p[1:-1, :-2]
    by = c - p[:-2, 1:-1]

    g = torch.sqrt(dx * dx + dy * dy)
    curvature = (dx - bx).abs() + (dy - by).abs()

    xs = torch.arange(w, device=x.device)
    ys = torch.arange(h, device=x.device)
    mask_x = ((xs % 8 == 7) & (xs + 1 < w)).to(torch.float32).unsqueeze(0)
    mask_y = ((ys % 8 == 7) & (ys + 1 < h)).to(torch.float32).unsqueeze(1)
    block = mask_x * (dx.abs() - bx.abs()).clamp_min(0.0) + mask_y * (dy.abs() - by.abs()).clamp_min(0.0)

    ring = 0.5 * (
        (dx.abs() + bx.abs() - (dx + bx).abs()).clamp_min(0.0)
        + (dy.abs() + by.abs() - (dy + by).abs()).clamp_min(0.0)
    )
    return g, curvature, block, ring


def _scale_channel_terms(a: torch.Tensor, b: torch.Tensor, center: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the 20 Alg S4 feature means and mean symmetric gradient for one `(scale, channel)`."""
    ma, mb, me, va, vb, ve = local_moments_11x11(a, b, center)

    lum = ((me * me) / (ma * ma + mb * mb + _C2)).clamp_max(1.0)
    structure = (ve / (va + vb + _C2)).clamp_max(2.0)
    damage = (lum + (1.0 - lum) * structure).clamp_min(0.0)

    ga, ca, ba, ra = stencil_terms(a)
    gb, cb, bb, rb = stencil_terms(b)
    e = a - b

    gms = ((ga - gb) ** 2) / (ga * ga + gb * gb + _C2)
    edge = (ca / (ca + ga + _C2) - cb / (cb + gb + _C2)).abs()
    bk = (ba - bb).abs()
    rg = (ra - rb).abs()

    low = torch.sqrt(torch.minimum(va, vb))
    high = torch.sqrt(torch.maximum(va, vb))
    delta = high - low

    md = torch.where(low <= 0.03, torch.ones_like(low), (0.03 / low.clamp_min(0.03)).clamp_max(1.0))
    me_m = torch.where(low <= 0.05, torch.ones_like(low), (0.0025 / low.clamp_min(0.05) ** 2).clamp_max(1.0))
    mf = torch.where(low <= 0.10, torch.ones_like(low), (0.0001 / low.clamp_min(0.10) ** 4).clamp_max(1.0))

    art = delta * torch.exp((-low / 0.045).clamp_min(-32.0))
    det = (delta / high.clamp_min(_EPS)) * (1.0 - torch.exp((-low / 0.08).clamp_min(-32.0)))

    frag_a = ra / (ga + ra + _C2)
    frag_b = rb / (gb + rb + _C2)
    fragility = torch.maximum(frag_a, frag_b) * damage

    # Widen sums and higher-order powers to float64 before global spatial mean.
    l64 = lum.to(torch.float64)
    s64 = structure.to(torch.float64)
    d64 = damage.to(torch.float64)
    e64 = e.to(torch.float64)
    l2 = l64 * l64
    d2 = d64 * d64
    d4 = d2 * d2

    scalars = torch.stack(
        [
            l64.mean(),
            l2.mean(),
            (l2 * l2).mean(),
            s64.mean(),
            (s64 * s64).mean(),
            d2.mean(),
            d4.mean(),
            e64.abs().mean(),
            (e64 * e64).mean(),
            gms.to(torch.float64).mean(),
            edge.to(torch.float64).mean(),
            bk.to(torch.float64).mean(),
            rg.to(torch.float64).mean(),
            (damage * md).to(torch.float64).mean(),
            (damage * me_m).to(torch.float64).mean(),
            (damage * mf).to(torch.float64).mean(),
            art.to(torch.float64).mean(),
            det.to(torch.float64).mean(),
            fragility.to(torch.float64).mean(),
            (d4 * d4).mean(),
        ]
    )
    mean_sym_grad = (0.5 * (ga + gb)).to(torch.float64).mean()
    return scalars, mean_sym_grad


@torch.no_grad()
def extract_rev8_s4(rgb_a: torch.Tensor, rgb_b: torch.Tensor, nits: float = 255.0) -> Rev8S4Result:
    """Extract Rev8 Alg S4 features ([4, 3, 20]) and symmetric fx2 fragility ([4, 3])."""
    xyb_a = dyadic_pyramid_4(srgb8_to_positive_xyb(rgb_a, nits=nits))
    xyb_b = dyadic_pyramid_4(srgb8_to_positive_xyb(rgb_b, nits=nits))

    features = torch.zeros((4, 3, len(ALG_S4_FEATURE_NAMES)), dtype=torch.float64, device=rgb_a.device)
    scale_fragility = torch.zeros((4, 3), dtype=torch.float64, device=rgb_a.device)
    for s in range(4):
        for c in range(3):
            feats, mean_g = _scale_channel_terms(xyb_a[s][c], xyb_b[s][c], _WINDOW_CENTERS[c])
            features[s, c] = feats
            scale_fragility[s, c] = _C_FRAG / (mean_g + _C_FRAG)
    return Rev8S4Result(features=features, scale_fragility=scale_fragility)


def _load_rgb_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Rev8 Alg S4 features in PyTorch.")
    ap.add_argument("ref_image")
    ap.add_argument("dist_image")
    ap.add_argument("--nits", type=float, default=255.0, choices=(255.0, 500.0))
    args = ap.parse_args()
    res = extract_rev8_s4(_load_rgb_tensor(args.ref_image), _load_rgb_tensor(args.dist_image), nits=args.nits)
    payload = {
        "nits": args.nits,
        "feature_names": list(ALG_S4_FEATURE_NAMES),
        "features": res.features.tolist(),
        "scale_fragility": res.scale_fragility.tolist(),
        "fx2_480": res.fx2_vector().tolist(),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
