#!/usr/bin/env python3
"""E5A analysis driver — preregistered protocol, benchmarks/e5a-render_prereg_2026-09-23.md.

Joins the per-arm score tables on `key`, calibrates `testlin` on TRAIN
families ONLY (then freezes it), computes benign-pooled thresholds, detection
rates at the 1% and 0.1% nominal-FA points, per-family severity SROCC (via
`panel --batch`, canonical midrank ties), map localisation (lift + top-decile
coverage), and the paired origin-cluster bootstrap (B=2000, seed 20260923).

Inputs:
  --scores   e5a_scores.tsv   (e5a_render_score: maxabs..zensim_b, anchor, maps)
  --peer     peer.tsv         (peer_metric_pairs: ssim2, butter_*)
  --dssim    dssim.tsv        (zenmetrics batch --metric dssim)
  --tuner    tuner.parquet    (score_pairs_tuner: score_d, score_r915*)
  --maps     DIR              (u8/lin/gmsd/zensim_b f32 maps from e5a_render_score)
  --peer-maps DIR             (butteraugli f32 diffmaps from peer_metric_pairs)
  --panel    PATH             (panel binary; default zensim release target)
  --out      results.json     (machine-readable record)

Every item is keyed by `key`; the score TSVs all carry the pairs.tsv key
columns verbatim, so joins are exact.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_FAMS = [
    "gamma_downsample",
    "alpha_nopremul",
    "geometry_shift",
    "channel_chroma",
    "exif_orientation",
]
TEST_FAMS = [
    "alpha_premul_state",
    "gamma_apply",
    "wrong_kernel",
    "primaries_dropped",
    "bitdepth",
]
FAMILIES = TRAIN_FAMS + TEST_FAMS

# arm -> direction: "+" higher is worse, "-" lower is worse.
ARMS = {
    "maxabs": "+",
    "psnr": "-",
    "ssim2": "-",
    "butter_max": "+",
    "butter_p3": "+",
    "dssim": "+",
    "gmsd": "+",
    "zensim_b": "-",
    "zensim_d": "-",
    "r915_fast": "-",
    "r915_rich": "-",
    "testlin": "+",
}
GENERAL_ARMS = [a for a in ARMS if a != "testlin"]
TESTLIN_CANDS = ["t1_lin_max", "t2_lin_q999", "t3_lin_q99", "t4_enc_max", "t5_enc_q999"]

BOOT_B = 2000
BOOT_SEED = 20260923


def q_lin(sorted_vals: np.ndarray, q: float) -> float:
    """Empirical quantile, numpy-linear interpolation (matches scorer)."""
    if len(sorted_vals) == 0:
        return float("nan")
    return float(np.quantile(sorted_vals, q, method="linear"))


def threshold(benign: np.ndarray, q: float, direction: str) -> float:
    """Upper-tail quantile for +, lower-tail (1-q) for -."""
    b = np.sort(benign[~np.isnan(benign)])
    return q_lin(b, q if direction == "+" else 1.0 - q)


def flagged(scores: np.ndarray, thr: float, direction: str) -> np.ndarray:
    if direction == "+":
        return scores > thr
    return scores < thr


def load_tables(args) -> pd.DataFrame:
    sc = pd.read_csv(args.scores, sep="\t")
    keycols = ["key", "origin", "kind", "family", "variant", "severity", "index"]
    for extra in ["peer", "dssim"]:
        df = pd.read_csv(getattr(args, extra), sep="\t")
        add = [c for c in df.columns if c not in sc.columns]
        sc = sc.merge(df[["key"] + add], on="key", how="left", validate="one_to_one")  # joinsafety-ok: per-stimulus unique key, one-to-one validated
    if args.tuner:
        tp = pd.read_parquet(args.tuner)
        keep = ["key"] + [c for c in tp.columns if c.startswith("score_")]
        tp = tp[keep].rename(
            columns={
                "score_d": "zensim_d",
                "score_r915_fast": "r915_fast",
                "score_r915fast": "r915_fast",
                "score_r915_rich": "r915_rich",
                "score_r915rich": "r915_rich",
            }
        )
        sc = sc.merge(tp, on="key", how="left", validate="one_to_one")  # joinsafety-ok: per-stimulus unique key, one-to-one validated
    missing = [c for c in keycols if c not in sc.columns]
    if missing:
        sys.exit(f"scores table missing key columns: {missing}")
    for arm in list(ARMS) + TESTLIN_CANDS + ["anchor_lin_mean"]:
        if arm in sc.columns:
            sc[arm] = pd.to_numeric(sc[arm], errors="coerce")
    return sc


def panel_srocc(pairs: list[tuple[str, np.ndarray, np.ndarray]], panel_bin: str) -> dict:
    """Run `panel --batch --stats srocc` over (label, x, y) vector triples.

    Returns {label: {"srocc": float, "n": int}}. NaN pairs are dropped before
    batching; empty vectors get n=0 srocc=nan.
    """
    lines = []
    meta = []
    for label, x, y in pairs:
        ok = ~(np.isnan(x) | np.isnan(y))
        xv, yv = x[ok], y[ok]
        meta.append((label, int(ok.sum())))
        if len(xv) < 3:
            continue
        xs = ",".join(f"{v:.9g}" for v in xv)
        ys = ",".join(f"{v:.9g}" for v in yv)
        lines.append(f"L\t{xs}\t{ys}")
    out: dict[str, dict] = {}
    if lines:
        manifest = "\n".join(lines) + "\n"
        r = subprocess.run(
            [panel_bin, "--batch", "-", "--stats", "srocc"],
            input=manifest,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            sys.exit(f"panel --batch failed: {r.stderr[:2000]}")
        rows = [ln for ln in r.stdout.strip().split("\n") if ln.strip()]
        # header + one row per L line, in order (labels are not echoed back —
        # the L rows all carry the literal label "L", so rows map by position).
        hdr = rows[0].split("\t") if rows else []
        srocc_col = next(
            (i for i, c in enumerate(hdr) if c.lower() == "srocc_signed"),
            next((i for i, c in enumerate(hdr) if "srocc" in c.lower()), None),
        )
        li = 0
        for label, n in meta:
            if n < 3:
                out[label] = {"srocc": float("nan"), "n": n}
                continue
            vals = rows[li + 1].split("\t")
            li += 1
            s = float(vals[srocc_col]) if srocc_col is not None else float("nan")
            out[label] = {"srocc": s, "n": n}
    for label, n in meta:
        out.setdefault(label, {"srocc": float("nan"), "n": n})
    return out


def load_f32_map(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.fromfile(path, dtype="<f4")


def align_changed(changed: np.ndarray, mw: int, mh: int, w: int, h: int) -> np.ndarray | None:
    """Changed mask on the map grid: identity when same size, else 2x2
    max-pool (gmsd map is (w/2)x(h/2)). Returns None when unalignable."""
    ch = changed.reshape(h, w)
    if (mw, mh) == (w, h):
        return ch
    if mw * 2 == w and mh * 2 == h:
        return ch.reshape(mh, 2, mw, 2).max(axis=(1, 3)).astype(bool)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--dssim", required=True)
    ap.add_argument("--tuner")
    ap.add_argument("--maps", required=True)
    ap.add_argument("--peer-maps", required=True)
    ap.add_argument("--panel", default="/home/lilith/work/zen/zensimx/target/release/panel")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = load_tables(args)
    # Prereg: inert items (a/b byte-identical) are DROPPED and counted —
    # excluded from thresholds, detection denominators, and the bootstrap.
    n_corr_inert = int(((df.kind == "corruption") & (df.inert == 1)).sum())
    n_bn_inert = int(((df.kind == "benign") & (df.inert == 1)).sum())
    corrupt = df[(df.kind == "corruption") & (df.inert == 0)]
    benign = df[(df.kind == "benign") & (df.inert == 0)]
    n_benign = len(benign)
    print(
        f"items: {len(df)} total, {len(corrupt)} corruption "
        f"(+{n_corr_inert} inert dropped), {n_benign} benign "
        f"(+{n_bn_inert} inert dropped)"
    )

    # ---- thresholds on pooled benign -------------------------------------
    thr: dict[str, dict[str, float]] = {}
    for arm, d in ARMS.items():
        if arm == "testlin":
            continue
        col = arm if arm in df.columns else None
        if col is None:
            continue
        bv = benign[arm].to_numpy(float)
        thr[arm] = {
            "t99": threshold(bv, 0.99, d),
            "t999": threshold(bv, 0.999, d),
            "n_benign": int(np.isfinite(bv).sum()),
            "n_nan": int(np.isnan(bv).sum()),
        }
    for cand in TESTLIN_CANDS:
        bv = benign[cand].to_numpy(float)
        thr[cand] = {
            "t99": threshold(bv, 0.99, "+"),
            "t999": threshold(bv, 0.999, "+"),
            "n_benign": int(np.isfinite(bv).sum()),
            "n_nan": int(np.isnan(bv).sum()),
        }

    # ---- testlin selection on TRAIN families ONLY -------------------------
    tr0 = corrupt[corrupt.family.isin(TRAIN_FAMS)]
    sel = {}
    for cand in TESTLIN_CANDS:
        sv = tr0[cand].to_numpy(float)
        sel[cand] = float(np.nanmean(flagged(sv, thr[cand]["t99"], "+")))
    winner = max(TESTLIN_CANDS, key=lambda c: (sel[c], -TESTLIN_CANDS.index(c)))
    df["testlin"] = df[winner]
    thr["testlin"] = thr[winner]
    testlin_map = {"t1_lin_max": "lin", "t2_lin_q999": "lin", "t3_lin_q99": "lin",
                   "t4_enc_max": "u8", "t5_enc_q999": "u8"}[winner]
    print(f"testlin := {winner} (TRAIN rate {sel[winner]:.4f}); map={testlin_map}")
    # Re-slice now that `testlin` exists — earlier subframes lack the column.
    corrupt = df[(df.kind == "corruption") & (df.inert == 0)]
    benign = df[(df.kind == "benign") & (df.inert == 0)]
    tr = corrupt[corrupt.family.isin(TRAIN_FAMS)]
    te = corrupt[corrupt.family.isin(TEST_FAMS)]

    # ---- detection --------------------------------------------------------
    def rates(sub: pd.DataFrame, level: str) -> dict:
        out = {}
        for arm, d in ARMS.items():
            if arm not in df.columns or arm not in thr:
                continue
            sv = sub[arm].to_numpy(float)
            t = thr[arm][level]
            ok = np.isfinite(sv)
            fl = flagged(sv, t, d)  # NaN never flags
            out[arm] = {
                "rate": float(fl.mean()) if len(fl) else float("nan"),
                "n": int(ok.sum()),
                "n_nan": int((~ok).sum()),
            }
        return out

    det = {
        "pooled_train_t99": rates(tr, "t99"),
        "pooled_test_t99": rates(te, "t99"),
        "pooled_test_t999": rates(te, "t999"),
        "per_family_t99": {f: rates(corrupt[corrupt.family == f], "t99") for f in FAMILIES},
        "per_family_t999": {f: rates(corrupt[corrupt.family == f], "t999") for f in FAMILIES},
        "family_x_variant_t99": {
            f"{f}|{v}": rates(g, "t99")
            for (f, v), g in corrupt.groupby(["family", "variant"])
        },
        "benign_realized_fa": {
            lv: {
                arm: (
                    float(flagged(benign[arm].to_numpy(float), thr[arm][lv], d).mean())
                    if arm in df.columns and arm in thr
                    else float("nan")
                )
                for arm, d in ARMS.items()
            }
            for lv in ["t99", "t999"]
        },
    }

    # ---- severity ordering (panel --batch, srocc) --------------------------
    batch_pairs = []
    for fam in FAMILIES:
        g = corrupt[corrupt.family == fam]
        anchor = g["anchor_lin_mean"].to_numpy(float)
        for arm in ARMS:
            if arm not in df.columns:
                continue
            batch_pairs.append((f"{fam}|{arm}", g[arm].to_numpy(float), anchor))
    srocc = panel_srocc(batch_pairs, args.panel)

    # ---- localisation ------------------------------------------------------
    maps_dir = Path(args.maps)
    peer_maps = Path(args.peer_maps)
    map_src = {
        "maxabs": (maps_dir, "__u8.f32"),
        "testlin": (maps_dir, f"__{testlin_map}.f32"),
        "butter_max": (peer_maps, "__butter.f32"),
        "gmsd": (maps_dir, "__gmsd.f32"),
        "zensim_b": (maps_dir, "__zensim_b.f32"),
    }
    loc: dict[str, dict] = {}
    for fam in FAMILIES:
        g = corrupt[corrupt.family == fam]
        for arm, (mdir, suffix) in map_src.items():
            lifts, covs, degen, n = [], [], 0, 0
            for _, r in g.iterrows():
                u8m = load_f32_map(maps_dir / f"{r['key']}__u8.f32")
                mm = load_f32_map(mdir / f"{r['key']}{suffix}")
                if u8m is None or mm is None:
                    continue
                w, h = int(r["width"]), int(r["height"])
                n += 1
                changed = u8m > 0
                cf = changed.mean()
                if cf == 0 or cf >= 0.99:
                    degen += 1
                    continue
                if len(mm) == w * h:
                    mw, mh = w, h
                elif len(mm) * 4 == w * h:
                    mw, mh = w // 2, h // 2
                else:
                    continue
                chg = align_changed(changed, mw, mh, w, h)
                if chg is None:
                    continue
                m2 = mm.reshape(mh, mw)
                un = ~chg
                if un.sum() == 0 or np.isnan(m2[chg].mean()):
                    degen += 1
                    continue
                mean_un = m2[un].mean()
                lift = float(m2[chg].mean() / mean_un) if mean_un != 0 else float("nan")
                # coverage: fraction of changed map cells in the map's top decile
                t90 = np.quantile(m2.ravel(), 0.9, method="linear")
                cov = float((m2[chg] >= t90).mean())
                lifts.append(lift)
                covs.append(cov)
            finite_lifts = [x for x in lifts if np.isfinite(x)]
            loc[f"{fam}|{arm}"] = {
                "lift_mean": float(np.mean(finite_lifts)) if finite_lifts else float("nan"),
                "n_lift_defined": len(finite_lifts),
                "coverage_mean": float(np.mean(covs)) if covs else float("nan"),
                "n_items": n,
                "n_degenerate": degen,
            }

    # ---- origin-cluster bootstrap (paired across arms) ---------------------
    origins = sorted(df.origin.unique())
    oid = {o: i for i, o in enumerate(origins)}
    oidx_te = te.origin.map(oid).to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(origins), size=(BOOT_B, len(origins)))

    test_flags: dict[str, np.ndarray] = {}
    for arm, d in ARMS.items():
        if arm in df.columns and arm in thr:
            sv = te[arm].to_numpy(float)
            test_flags[arm] = np.where(np.isfinite(sv), flagged(sv, thr[arm]["t99"], d), False)

    boot_rates = {arm: np.empty(BOOT_B) for arm in test_flags}
    fam_boot = {
        fam: {arm: np.empty(BOOT_B) for arm in test_flags} for fam in TEST_FAMS
    }
    fam_mask = {fam: (te.family == fam).to_numpy() for fam in TEST_FAMS}
    for b in range(BOOT_B):
        sel_origins = draws[b]
        inb = np.isin(oidx_te, sel_origins)
        for arm, fl in test_flags.items():
            boot_rates[arm][b] = fl[inb].mean() if inb.any() else np.nan
            for fam in TEST_FAMS:
                fm = fam_mask[fam] & inb
                fam_boot[fam][arm][b] = fl[fm].mean() if fm.any() else np.nan

    def ci(v: np.ndarray) -> list[float]:
        v = v[np.isfinite(v)]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [float("nan")] * 2

    boot = {
        "B": BOOT_B,
        "seed": BOOT_SEED,
        "origins": origins,
        "pooled_test_t99": {
            arm: {
                "point": det["pooled_test_t99"].get(arm, {}).get("rate", float("nan")),
                "ci95": ci(v),
            }
            for arm, v in boot_rates.items()
        },
        "per_family_t99": {
            fam: {
                arm: {
                    "point": det["per_family_t99"][fam].get(arm, {}).get("rate", float("nan")),
                    "ci95": ci(v),
                }
                for arm, v in arms.items()
            }
            for fam, arms in fam_boot.items()
        },
    }

    # realized benign FA on resamples
    oidx_bn = benign.origin.map(oid).to_numpy()
    fa_boot = {arm: np.empty(BOOT_B) for arm in test_flags}
    bn_flags = {}
    for arm, d in ARMS.items():
        if arm in df.columns and arm in thr:
            sv = benign[arm].to_numpy(float)
            bn_flags[arm] = np.where(np.isfinite(sv), flagged(sv, thr[arm]["t99"], d), False)
    for b in range(BOOT_B):
        inb = np.isin(oidx_bn, draws[b])
        for arm, fl in bn_flags.items():
            fa_boot[arm][b] = fl[inb].mean() if inb.any() else np.nan
    boot["benign_fa_t99"] = {arm: {"ci95": ci(v)} for arm, v in fa_boot.items()}

    # ---- decision rule ------------------------------------------------------
    gen_rates = {
        a: det["pooled_test_t99"][a]["rate"]
        for a in GENERAL_ARMS
        if a in det["pooled_test_t99"]
    }
    best_general = max(gen_rates, key=lambda a: gen_rates[a])
    delta = det["pooled_test_t99"]["testlin"]["rate"] - gen_rates[best_general]
    d_boot = boot_rates["testlin"] - boot_rates[best_general]
    d_ci = ci(d_boot)
    if delta >= 0.10 and d_ci[0] > 0:
        verdict = "CONFIRMED"
    elif d_ci[1] < 0.10:
        verdict = "NO-SHIP"
    else:
        verdict = "UNRESOLVED"

    result = {
        "schema": "rev4-e5a-render-results-v1",
        "n_items": len(df),
        "n_corruption": len(corrupt),
        "n_benign": n_benign,
        "n_corruption_inert": n_corr_inert,
        "n_benign_inert": n_bn_inert,
        "testlin": {
            "winner": winner,
            "train_t99_rates": sel,
            "map": testlin_map,
        },
        "thresholds": thr,
        "detection": det,
        "severity_srocc": srocc,
        "localisation": loc,
        "bootstrap": boot,
        "decision": {
            "best_general": best_general,
            "best_general_rate": gen_rates[best_general],
            "testlin_rate": det["pooled_test_t99"]["testlin"]["rate"],
            "delta": delta,
            "delta_ci95": d_ci,
            "verdict": verdict,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    print(
        f"DECISION: {verdict}  (testlin {det['pooled_test_t99']['testlin']['rate']:.3f} "
        f"vs {best_general} {gen_rates[best_general]:.3f}, "
        f"Δ={delta:.3f} CI[{d_ci[0]:.3f},{d_ci[1]:.3f}])"
    )


if __name__ == "__main__":
    main()
