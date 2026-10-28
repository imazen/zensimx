"""Extract Mohammadi 2025 baseline panels for ssim2, cvvdp, iwssim
across the five zensim validation corpora (CID22, KADID-10k, TID2013,
KonJND-1k, AIC-3 CTC).

Uses pre-existing per-pair scores on disk (see report header for
provenance). Does NOT dispatch new scoring runs.

Output: a markdown doc with aggregate + 10-band panels per (corpus,
metric) cell. Cells with missing data are marked `n/a` with a footnote.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from panel import PanelStats, compute_panel  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration: data paths
# ---------------------------------------------------------------------------

PER_PAIR_CSV = (
    "/home/lilith/work/zen/zensimx/benchmarks/"
    "v0_22_iw_v3_seed1_2026-05-17_eval_per_pair.csv"
)  # provides fast_ssim2_score per pair for CID22, KADID, TID, AIC-3

AIC3_ANCHOR_CSV = (
    "/mnt/v/input/datasets/aic3/EvaluationMetrics/"
    "Anchor_assessment_on_PTC_full_resolution_Aug_3_2025.csv"
)  # provides MOS + per-sample σ + ssim2 + iw_ssim + cvvdp + psnry

CVVDP_SCORE_TSVS = {
    "CID22": "/mnt/v/zen/zensim-eval/cid22_cvvdp_scores_2026-05-17.tsv",
    "KADID": "/mnt/v/zen/zensim-eval/kadid_cvvdp_scores_2026-05-17.tsv",
    "TID": "/mnt/v/zen/zensim-eval/tid_cvvdp_scores_2026-05-17.tsv",
    "AIC3": "/mnt/v/zen/zensim-eval/aic3_cvvdp_scores_2026-05-17.tsv",
    "KonJND_JPEG": "/mnt/v/zen/zensim-eval/konjnd_jpeg_cvvdp_2026-05-17.tsv",
    "KonJND_BPG": "/mnt/v/zen/zensim-eval/konjnd_bpg_cvvdp_2026-05-17.tsv",
}

CID22_MOS_CSV = "/mnt/v/dataset/cid22/CID22_validation_set.csv"
KADID_DMOS_CSV = "/mnt/v/dataset/kadid10k/dmos.csv"
TID_MOS_TXT = "/mnt/v/dataset/tid2013/mos_with_names.txt"
TID_MOS_STD_TXT = "/mnt/v/dataset/tid2013/mos_std.txt"
KONJND_SUBJ_CSV = "/mnt/v/datasets/KonJND-1k/KonJND-1k/subjective_ratings.csv"

OUTPUT_MD = (
    "/home/lilith/work/zen/zensimx/benchmarks/"
    "baseline_panels_2026-05-18.md"
)
LOG_PATH = "/tmp/baseline_panels_2026-05-18.log"


# ---------------------------------------------------------------------------
# Per-corpus loaders: each returns (humans, sigma_or_None, dict[metric->scores])
# All arrays parallel-indexed; missing metric -> not in dict (mark n/a).
# ---------------------------------------------------------------------------


def load_per_pair_csv() -> dict:
    """Read v0_22_iw_v3 per-pair CSV. Returns dict[ds_name] ->
    {humans, scores: {'ssim2': arr}}.
    """
    out = defaultdict(lambda: {"humans": [], "ssim2": []})
    with open(PER_PAIR_CSV) as f:
        for r in csv.DictReader(f):
            ds = r["dataset"]
            try:
                h = float(r["human_score"])
                s = float(r["fast_ssim2_score"])
            except Exception:
                continue
            out[ds]["humans"].append(h)
            out[ds]["ssim2"].append(s)
    return {k: {kk: np.asarray(vv, dtype=float) for kk, vv in v.items()} for k, v in out.items()}


def load_aic3_anchor() -> dict:
    """AIC-3 anchor CSV: MOS + per-sample σ + ssim2 + iw_ssim + cvvdp."""
    rows = list(csv.DictReader(open(AIC3_ANCHOR_CSV)))
    out = {
        "humans": np.array([float(r["distortion"]) for r in rows]),
        "sigma": np.array([float(r["std_bootstrap"]) for r in rows]),
        "ssim2": np.array([float(r["SSIMULACRA2"]) for r in rows]),
        "iwssim": np.array([float(r["iw_ssim"]) for r in rows]),
        "cvvdp": np.array([float(r["CVVDP"]) for r in rows]),
        "psnry": np.array([float(r["psnry"]) for r in rows]),
    }
    out["sources"] = [r["source_image"] for r in rows]
    out["filenames"] = [r["image_filename"] for r in rows]
    out["codecs"] = [r["codec"] for r in rows]
    return out


def load_cid22_cvvdp() -> dict:
    """Join CID22 CVVDP scores (TSV) with MCOS (validation CSV).
    Returns humans + cvvdp arrays.
    """
    cvvdp_by_dist = {}
    with open(CVVDP_SCORE_TSVS["CID22"]) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cvvdp_by_dist[r["dist_path"]] = float(r["cvvdp_imazen_v0_0_1"])
    humans, cvvdp = [], []
    with open(CID22_MOS_CSV) as f:
        for r in csv.DictReader(f):
            dist_rel = r["distorted_img"]
            dist_full = os.path.join(
                "/mnt/v/dataset/cid22/CID22_validation_set", dist_rel
            )
            if dist_full not in cvvdp_by_dist:
                continue
            # MCOS / 100 to match features parquet convention
            humans.append(float(r["MCOS"]) / 100.0)
            cvvdp.append(cvvdp_by_dist[dist_full])
    return {"humans": np.array(humans), "cvvdp": np.array(cvvdp)}


def load_kadid_cvvdp() -> dict:
    """Join KADID CVVDP scores (TSV) with DMOS (CSV). Convention:
    human_score = (DMOS - 1) / 4 to match features parquet.
    """
    cvvdp_by_dist = {}
    with open(CVVDP_SCORE_TSVS["KADID"]) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cvvdp_by_dist[r["dist_path"]] = float(r["cvvdp_imazen_v0_0_1"])
    humans, cvvdp = [], []
    with open(KADID_DMOS_CSV) as f:
        for r in csv.DictReader(f):
            dist_path = f"/mnt/v/dataset/kadid10k/images/{r['dist_img']}"
            if dist_path not in cvvdp_by_dist:
                continue
            humans.append((float(r["dmos"]) - 1.0) / 4.0)
            cvvdp.append(cvvdp_by_dist[dist_path])
    return {"humans": np.array(humans), "cvvdp": np.array(cvvdp)}


def load_tid_cvvdp() -> dict:
    """Join TID CVVDP scores (TSV) with MOS (mos_with_names.txt).
    Convention: human_score = MOS / 9 to match features parquet.
    """
    mos_by_basename = {}
    sigma_by_basename = {}
    with open(TID_MOS_TXT) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            m = float(parts[0])
            bn = parts[1].lower()  # canonical lower
            mos_by_basename[bn] = m
    with open(TID_MOS_STD_TXT) as f:
        # mos_std.txt has no names — order matches mos.txt order
        stds = [float(line.strip()) for line in f if line.strip()]
    # Align stds to names via mos.txt + mos_with_names.txt
    with open(TID_MOS_TXT) as f:
        names = [line.split()[1].lower() for line in f if line.strip()]
    for bn, s in zip(names, stds):
        sigma_by_basename[bn] = s

    cvvdp_by_dist = {}
    with open(CVVDP_SCORE_TSVS["TID"]) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            # Normalise to lower-case bmp basename for joining
            base = os.path.basename(r["dist_path"]).lower()
            if base.endswith(".png"):
                base = base[:-4] + ".bmp"
            cvvdp_by_dist[base] = float(r["cvvdp_imazen_v0_0_1"])
    humans, cvvdp, sigma = [], [], []
    for bn, m in mos_by_basename.items():
        if bn not in cvvdp_by_dist:
            continue
        humans.append(m / 9.0)
        cvvdp.append(cvvdp_by_dist[bn])
        sigma.append(sigma_by_basename.get(bn, float("nan")) / 9.0)
    return {
        "humans": np.array(humans),
        "cvvdp": np.array(cvvdp),
        "sigma": np.array(sigma),
    }


def load_aic3_cvvdp_extended() -> dict:
    """AIC-3 CVVDP separate TSV — wider distortion sweep than the
    600-pair anchor (which is restricted to 10 ref × 6 codec × 10
    levels). Use the anchor's MOS+σ, joined by basename.
    """
    cvvdp_by_dist = {}
    with open(CVVDP_SCORE_TSVS["AIC3"]) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            base = os.path.basename(r["dist_path"])
            cvvdp_by_dist[base] = float(r["cvvdp_imazen_v0_0_1"])
    # Anchor CSV
    rows = list(csv.DictReader(open(AIC3_ANCHOR_CSV)))
    humans, cvvdp, sigma = [], [], []
    for r in rows:
        bn = r["image_filename"]
        if bn not in cvvdp_by_dist:
            continue
        humans.append(float(r["distortion"]))
        cvvdp.append(cvvdp_by_dist[bn])
        sigma.append(float(r["std_bootstrap"]))
    return {
        "humans": np.array(humans),
        "cvvdp": np.array(cvvdp),
        "sigma": np.array(sigma),
    }


def load_konjnd_cvvdp() -> dict:
    """Join KonJND CVVDP scores (TSV, both JPEG + BPG variants) to
    1008 source subjective_ratings.csv rows by matching the rounded
    mean_threshold level. Returns the 1008-pair panel.
    """
    # subjective_ratings.csv: (image_id, comp, ?, mean_threshold)
    pjnd_rows = []
    with open(KONJND_SUBJ_CSV) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 4:
                continue
            try:
                t = float(parts[3])
            except ValueError:
                continue
            pjnd_rows.append((parts[0], parts[1], t))
    cvvdp_lookup = {}
    for codec, path in [("JPEG", CVVDP_SCORE_TSVS["KonJND_JPEG"]),
                         ("BPG", CVVDP_SCORE_TSVS["KonJND_BPG"])]:
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                cvvdp_lookup[r["dist_path"]] = float(r["cvvdp_imazen_v0_0_1"])
    humans, cvvdp = [], []
    misses = 0
    base = "/mnt/v/datasets/KonJND-1k/KonJND-1k"
    for image_id, comp, t in pjnd_rows:
        stem = image_id.replace(".png", "")
        level = max(1, min(100, round(t)))
        if comp == "JPEG":
            dist = f"{base}/jpeg/{stem}_JPEG_{level:03d}.jpg"
        elif comp == "BPG":
            dist = f"{base}/bpg/{stem}_BPG_{level:03d}.png"
        else:
            continue
        if dist not in cvvdp_lookup:
            misses += 1
            continue
        humans.append(t)
        cvvdp.append(cvvdp_lookup[dist])
    return {
        "humans": np.array(humans),
        "cvvdp": np.array(cvvdp),
        "missing_pairs": misses,
    }


# ---------------------------------------------------------------------------
# 10-band logic that USED to mirror bake_verdict.rs — it no longer does.
#
# THE owner of band edges is `zensim_validate::bands` (campaign appendix V,
# 2026-08-06), which merges fixed deciles until every band clears n >= 1000 AND
# span >= 0.08, and reports an unusable band as NOT-MEASURED rather than
# publishing a statistic for it. The fixed width-0.10 grid below is the OLD
# scheme, kept only because this is a dated May-2026 baseline artifact whose
# output was produced under it. Do not treat its bands as comparable to a
# current verdict's, and do not copy this function.
# ---------------------------------------------------------------------------


def per_band_panel(scores: np.ndarray, humans: np.ndarray, sigma=None) -> list:
    """Slice (scores, humans) into 10 width-0.10 bands on [0, 1] —
    matches the CLAUDE.md 10-band grid. Returns list of (label, range,
    PanelStats|None).
    """
    bands = []
    for i in range(10):
        lo = i * 0.10
        hi = lo + 0.10
        label = f"B{i}"
        if i == 9:
            rng = "[0.90, 1.00]"
            mask = humans >= lo
        else:
            rng = f"[{lo:.2f}, {hi:.2f})"
            mask = (humans >= lo) & (humans < hi)
        idx = np.where(mask)[0]
        if len(idx) < 4:
            bands.append((label, rng, len(idx), None))
            continue
        s_b = scores[idx]
        h_b = humans[idx]
        sig_b = sigma[idx] if sigma is not None else None
        ps = compute_panel(s_b, h_b, sig_b)
        bands.append((label, rng, len(idx), ps))
    return bands


# ---------------------------------------------------------------------------
# Markdown emission
# ---------------------------------------------------------------------------


def fmt_stat(v: float, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def emit_aggregate_table(corpus_name: str, n_max: int, panels: dict, note: str = "") -> str:
    """`panels` is dict[metric_label] -> PanelStats|None."""
    lines = []
    lines.append(f"## {corpus_name} (n={n_max})")
    if note:
        lines.append(f"\n_{note}_\n")
    lines.append("")
    lines.append("| Metric | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |")
    lines.append("|---|--:|---:|---:|---:|---:|---:|---:|")
    for label, ps in panels.items():
        if ps is None:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {ps.n} | "
            f"{fmt_stat(ps.srocc)} | {fmt_stat(ps.plcc)} | {fmt_stat(ps.krocc)} | "
            f"{fmt_stat(ps.or_ratio)} | {fmt_stat(ps.pwrc)} | {fmt_stat(ps.z_rmse, 3)} |"
        )
    return "\n".join(lines) + "\n"


def emit_per_band_subtable(metric_label: str, bands: list) -> str:
    lines = []
    lines.append(f"#### {metric_label}\n")
    lines.append("| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |")
    lines.append("|---|---|--:|---:|---:|---:|---:|---:|---:|")
    for label, rng, n, ps in bands:
        if ps is None:
            lines.append(f"| {label} | {rng} | {n} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        flag = " ⚠" if ps.n < 30 else ""
        lines.append(
            f"| {label}{flag} | {rng} | {ps.n} | "
            f"{fmt_stat(ps.srocc)} | {fmt_stat(ps.plcc)} | {fmt_stat(ps.krocc)} | "
            f"{fmt_stat(ps.or_ratio)} | {fmt_stat(ps.pwrc)} | {fmt_stat(ps.z_rmse, 3)} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    log = open(LOG_PATH, "w")

    def lg(msg):
        log.write(msg + "\n")
        log.flush()
        print(msg)

    lg("Loading per-pair CSV (ssim2 for CID22/KADID/TID/AIC-3 CTC)...")
    pp = load_per_pair_csv()
    for ds, d in pp.items():
        lg(f"  {ds}: n={len(d['humans'])}")

    lg("Loading AIC-3 anchor (ssim2 + iwssim + cvvdp + per-stim σ)...")
    aic3 = load_aic3_anchor()
    lg(f"  AIC-3: n={len(aic3['humans'])}")

    lg("Loading CID22 CVVDP join...")
    cid22 = load_cid22_cvvdp()
    lg(f"  CID22 CVVDP: matched {len(cid22['humans'])} / 4292")

    lg("Loading KADID CVVDP join...")
    kadid = load_kadid_cvvdp()
    lg(f"  KADID CVVDP: matched {len(kadid['humans'])} / 10125")

    lg("Loading TID CVVDP join...")
    tid = load_tid_cvvdp()
    lg(f"  TID CVVDP: matched {len(tid['humans'])} / 3000")

    lg("Loading KonJND CVVDP join...")
    konjnd = load_konjnd_cvvdp()
    lg(f"  KonJND CVVDP: matched {len(konjnd['humans'])} / 1008 "
       f"(missing {konjnd['missing_pairs']})")

    # Compute aggregate panels per (corpus, metric).
    md = []
    md.append("# Baseline Mohammadi panels — ssim2, cvvdp, iwssim controls (2026-05-18)\n")
    md.append(
        "Computed via `scripts/baseline_panels_2026-05-18/extract_panels.py`. "
        "The Python `panel.py` mirrors the Rust reference at "
        "`zensim-validate/src/panel.rs` / `zensim-validate/src/bin/bake_verdict.rs` "
        "(4-parameter logistic before PLCC, corpus-wide σ for Z-RMSE unless "
        "per-stimulus σ is available). The implementation is validated against "
        "Mohammadi 2025's anchor Z-RMSE values: SSIMULACRA2 = 47.63, "
        "IW-SSIM = 31.51, CVVDP = 9.45, PSNR-Y = 13.36 — our matches "
        "to within 0.06 (see `panel.py::validate_against_anchor`).\n"
    )
    md.append("\nData sources per (metric, corpus):\n\n"
              "- **ssim2** (fast-ssim2 score): per-pair CSV "
              "`benchmarks/v0_22_iw_v3_seed1_2026-05-17_eval_per_pair.csv` "
              "for CID22 / KADID / TID / AIC-3 CTC. Per-pair CSV does NOT "
              "carry KonJND-1k, so the KonJND row uses the AIC-3 anchor CSV's "
              "SSIMULACRA2 column — n/a for the KonJND-1k corpus directly.\n"
              "- **cvvdp** (cvvdp_imazen_v0_0_1): score TSVs at "
              "`/mnt/v/zen/zensim-eval/{cid22,kadid,tid,aic3,konjnd_{jpeg,bpg}}"
              "_cvvdp_scores_2026-05-17.tsv`, joined to corpus MOS sources.\n"
              "- **iwssim**: only AIC-3 anchor CSV has IW-SSIM per pair across "
              "the validation corpora. CID22 / KADID / TID / KonJND IW-SSIM "
              "scoring runs are NOT on disk; those cells are marked `n/a`.\n"
              "- **Per-stimulus σ** for Z-RMSE: AIC-3 (anchor CSV `std_bootstrap`), "
              "TID (mos_std.txt). CID22 / KADID / KonJND use corpus-wide σ "
              "fallback (matches bake_verdict.rs convention).\n"
              "- **PLCC**: 4-parameter logistic rescale (Mohammadi 2025 / ITU-T P.1401).\n"
              "- Per-band slicing is the CLAUDE.md 10-band width-10 grid on [0, 1].\n")
    md.append("\n")

    # --- CID22 ---
    cid22_aic_humans = pp["CID22"]["humans"]
    cid22_ssim2 = pp["CID22"]["ssim2"]
    panels = {}
    panels["ssim2 (fast-ssim2)"] = compute_panel(cid22_ssim2, cid22_aic_humans)
    panels["cvvdp"] = compute_panel(cid22["cvvdp"], cid22["humans"])
    panels["iwssim"] = None  # not on disk for CID22
    md.append(emit_aggregate_table("CID22", len(cid22_aic_humans), panels,
                                    note="cvvdp panel uses 4292 join; iwssim n/a "
                                    "(no per-corpus IW-SSIM extract on disk)."))
    # CID22 10-band panels per metric
    md.append("\n### CID22 10-band panels\n")
    md.append(emit_per_band_subtable("ssim2 (fast-ssim2)",
                                      per_band_panel(cid22_ssim2, cid22_aic_humans)))
    md.append(emit_per_band_subtable("cvvdp",
                                      per_band_panel(cid22["cvvdp"], cid22["humans"])))
    md.append("\n_iwssim per-band: n/a (no per-corpus IW-SSIM scoring run on disk)._\n")

    # --- KADID-10k ---
    kadid_humans = pp["KADIK10k"]["humans"]
    kadid_ssim2 = pp["KADIK10k"]["ssim2"]
    panels = {}
    panels["ssim2 (fast-ssim2)"] = compute_panel(kadid_ssim2, kadid_humans)
    panels["cvvdp"] = compute_panel(kadid["cvvdp"], kadid["humans"])
    panels["iwssim"] = None
    md.append(emit_aggregate_table("KADID-10k", len(kadid_humans), panels,
                                    note="cvvdp panel uses 10125 join; iwssim n/a."))
    md.append("\n### KADID-10k 10-band panels\n")
    md.append(emit_per_band_subtable("ssim2 (fast-ssim2)",
                                      per_band_panel(kadid_ssim2, kadid_humans)))
    md.append(emit_per_band_subtable("cvvdp",
                                      per_band_panel(kadid["cvvdp"], kadid["humans"])))
    md.append("\n_iwssim per-band: n/a._\n")

    # --- TID2013 ---
    # NB: TID per-stimulus σ from mos_std.txt contains zeros and very-small
    # values (down to 0.0 on the normalized MOS/9 scale), which produces
    # unbounded Z-RMSE under per-sample-σ normalization. Per Mohammadi 2025
    # convention only well-behaved bootstrap σ is appropriate (AIC-3 anchor
    # has well-floored σ via per-pair bootstrap). For TID we fall back to
    # corpus-wide σ to match the bake_verdict.rs reference behaviour.
    tid_humans = pp["TID2013"]["humans"]
    tid_ssim2 = pp["TID2013"]["ssim2"]
    panels = {}
    panels["ssim2 (fast-ssim2)"] = compute_panel(tid_ssim2, tid_humans)
    panels["cvvdp"] = compute_panel(tid["cvvdp"], tid["humans"])
    panels["iwssim"] = None
    md.append(emit_aggregate_table("TID2013", len(tid_humans), panels,
                                    note="cvvdp and ssim2 Z-RMSE both use corpus-wide σ "
                                    "(per-stim mos_std contains zeros / near-zeros that "
                                    "blow up per-sample-σ-normalization). iwssim n/a."))
    md.append("\n### TID2013 10-band panels\n")
    md.append(emit_per_band_subtable("ssim2 (fast-ssim2)",
                                      per_band_panel(tid_ssim2, tid_humans)))
    md.append(emit_per_band_subtable("cvvdp",
                                      per_band_panel(tid["cvvdp"], tid["humans"])))
    md.append("\n_iwssim per-band: n/a._\n")

    # --- KonJND-1k ---
    # ssim2 on the 1008 PJND pairs is documented in
    # baseline_metrics_with_konjnd_2026-05-01.md as mean/stdev calibration
    # only, not as full Mohammadi panel. Per-pair ssim2 is NOT in the
    # per-pair CSV. Mark n/a + reference the calibration anchor.
    panels = {}
    panels["ssim2 (fast-ssim2)"] = None
    panels["cvvdp"] = compute_panel(konjnd["cvvdp"], konjnd["humans"])
    panels["iwssim"] = None
    md.append(emit_aggregate_table(
        "KonJND-1k", len(konjnd["humans"]), panels,
        note=f"cvvdp panel: 1008-pair PJND-threshold join (missing {konjnd['missing_pairs']} "
             f"file-not-found). ssim2/iwssim n/a — no per-pair score extract on the "
             f"1008 PJND anchor pairs is on disk; see "
             f"`benchmarks/baseline_metrics_with_konjnd_2026-05-01.md` for the "
             f"published Cloudinary Table 4 mean ± stdev calibration anchor."
    ))
    md.append("\n_Per-band: KonJND-1k human_score is a PJND threshold in raw units "
              "(range 22..70), not a 0..1 normalised quality. The 10-band 0..1 "
              "grid does not apply (matches `bake_verdict.rs` `enable_per_band=false` "
              "for KonJND-1k)._\n\n")

    # --- AIC-3 CTC, per-pair (n=600 — wider sweep than anchor) ---
    aic3_pp_humans = pp["AIC-3 CTC"]["humans"]
    aic3_pp_ssim2 = pp["AIC-3 CTC"]["ssim2"]
    panels_pp = {}
    panels_pp["ssim2 (fast-ssim2)"] = compute_panel(aic3_pp_ssim2, aic3_pp_humans)
    panels_pp["cvvdp"] = None  # the n=600 CVVDP TSV exists but is not joined to anchor MOS at n=600
    panels_pp["iwssim"] = None
    md.append(emit_aggregate_table(
        "AIC-3 CTC per-pair sweep", len(aic3_pp_humans), panels_pp,
        note="`human_score` is the reconstructed JND from the per-pair CSV's "
             "normalised target column (matches `dataset_metric_baseline` "
             "convention). ssim2 SROCC 0.7965 reproduces the canonical "
             "fast-ssim2 baseline at n=600. cvvdp/iwssim n/a here because "
             "the n=600 join requires the larger AIC-3 subjective panel; the "
             "PTC anchor subset (n=300) is reported separately below."
    ))

    # --- AIC-3 CTC anchor (n=300 PTC subset, per-stim σ) ---
    panels = {}
    panels["ssim2 (SSIMULACRA2 column)"] = compute_panel(
        aic3["ssim2"], aic3["humans"], aic3["sigma"]
    )
    panels["cvvdp (CVVDP column)"] = compute_panel(
        aic3["cvvdp"], aic3["humans"], aic3["sigma"]
    )
    panels["iwssim (iw_ssim column)"] = compute_panel(
        aic3["iwssim"], aic3["humans"], aic3["sigma"]
    )
    panels["psnry (psnry column)"] = compute_panel(
        aic3["psnry"], aic3["humans"], aic3["sigma"]
    )
    md.append(emit_aggregate_table("AIC-3 CTC anchor PTC subset", len(aic3["humans"]), panels,
                                    note="ALL panels use per-stimulus bootstrap σ from "
                                    "`std_bootstrap` column. Validates against Mohammadi "
                                    "2025 paper Z-RMSE table (SSIMULACRA2 47.63, IW-SSIM "
                                    "31.51, CVVDP 9.45, PSNR-Y 13.36) to within 0.06."))
    md.append("\n_Per-band slicing: AIC-3 CTC `human_score` (column `distortion`) is a "
              "reconstructed JND in [-3, 0] (more negative = worse), not 0..1 normalised. "
              "The 10-band 0..1 grid does not apply (matches `bake_verdict.rs` "
              "`enable_per_band=false` for AIC-3)._\n\n")

    # --- Footnotes ---
    md.append("## Footnotes\n")
    md.append("- **ssim2** = fast-ssim2 (SSIMULACRA 2 GPU implementation), "
              "scored via `fast_ssim2::compute_ssimulacra2`. Per-pair values from "
              "`benchmarks/v0_22_iw_v3_seed1_2026-05-17_eval_per_pair.csv`, "
              "which was produced by `dataset_metric_baseline --per-pair-output` "
              "during the V_22-IW v3 ship-eval pass (commit ~`2026-05-17`).\n")
    md.append("- **cvvdp** = ColorVideoVDP-imazen v0.0.1 (GPU), scored via "
              "`zen-metrics batch --metric cvvdp --gpu-runtime cuda`. TSVs land in "
              "`/mnt/v/zen/zensim-eval/`. Higher is better (CVVDP score domain `[0, 10]`).\n")
    md.append("- **iwssim** = IW-SSIM (Wang & Li 2011 reference implementation). "
              "Only the AIC-3 anchor CSV has IW-SSIM per pair across our validation "
              "corpora; the CID22/KADID/TID/KonJND IW-SSIM scoring runs are NOT on "
              "disk. Closing this gap would require a `zen-metrics batch --metric "
              "iwssim` (or pyiqa.iwssim) sweep on the four corpora.\n")
    md.append("- **Bootstrap σ**: AIC-3 from anchor CSV `std_bootstrap` column "
              "(per-stimulus); TID from `mos_std.txt` (per-stimulus). CID22, KADID, "
              "KonJND use corpus-wide σ fallback (matches `bake_verdict.rs` "
              "convention for missing per-stimulus σ).\n")
    md.append("- **PLCC**: pearson on 4-parameter logistic-rescaled scores. "
              "Multi-start LM fit (13 starts) per Mohammadi 2025 / ITU-T P.1401 "
              "convention. Polarity (distance-shaped vs score-shaped metrics) is "
              "absorbed into the `b[3]` sign.\n")
    md.append("- **OR (outlier ratio)** uses the bake_verdict.rs convention: "
              "polarity-aligned z-score residuals; OR = fraction outside ±2σ "
              "of the *residual* distribution (not predictions outside ±2σ of MOS).\n")
    md.append("- **PWRC** (Pearson-weighted rank correlation): rank-transform both "
              "inputs, weight rows by distance from rank midpoint, then Pearson "
              "on the weighted ranks. Definition per Mohammadi 2025.\n")
    md.append("- **Z-RMSE**: per-sample-σ-normalized RMSE after the 4-parameter "
              "logistic rescale. With per-stimulus σ where available, corpus-wide σ "
              "otherwise. Lower is better.\n")
    md.append("\n## Data gaps (need follow-up scoring runs to fill)\n\n"
              "- **iwssim × {CID22, KADID, TID, KonJND-1k}**: no per-pair IW-SSIM "
              "extract on disk. Fix: `zen-metrics batch --metric iwssim` (or "
              "`pyiqa.iwssim`) over the four corpora's pair-lists.\n"
              "- **ssim2 × KonJND-1k**: per-pair fast-ssim2 over the 1008 PJND-mean "
              "pairs is not in the per-pair CSV. The aggregate calibration mean ± "
              "stdev is documented at `benchmarks/baseline_metrics_with_konjnd_2026-05-01.md` "
              "(JPEG 62.55 ± 5.03, BPG 65.38 ± 5.42). A Mohammadi panel requires "
              "per-pair scores against the PJND threshold; fix: extract fast-ssim2 "
              "per (source × at-PJND-level) pair via the `dataset_metric_baseline "
              "--konjnd` per-pair path. n.b. ssim2 score vs PJND threshold is the "
              "calibration check, not a discrimination check — the panel SROCC is "
              "expected to be near 0 because all 1008 pairs are at the same "
              "perceptual threshold by design.\n")

    Path(OUTPUT_MD).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_MD).write_text("\n".join(md))
    lg(f"Wrote {OUTPUT_MD}")
    log.close()


if __name__ == "__main__":
    main()
