#!/usr/bin/env python3
"""Summer-gauntlet INTERACTIVE dashboard builder — the ``--fulleval-dir`` mode of
``bandwise_dashboard.py`` (user 2026-07-26: "interactively compare all the summer's best bakes,
with correlation scatterplots for every reference — MOS, JND, ssim2, butteraugli, cvvdp").

It reads the per-bake ``*.fulleval.json`` files (schema + fixtures: ``make_stub_fulleval.py``)
and emits ONE self-contained, offline HTML page with:

  * a toggle bar (checkbox per bake — hide/show it across EVERY chart; stable color per bake),
  * a SORTABLE scoreboard table (click any header to sort by CID22 / KonJND / dial-mono / M3 /
    corruption-detection / composite / …),
  * the correlation SCATTER MATRIX — for the selected reference, one clean scatter per
    (bake x corpus) with an OLS fit line + canonical SROCC/PLCC annotated, faceted so bakes sit
    side by side per corpus,
  * a cross-corpus SROCC heatmap and a CID22-vs-{nonphoto,KonJND} operating-point trade map,
  * the JXL loop-targeting panel (2026-08-01): 2-shot/3-shot within-±2 scoreboard columns
    (emit-best, bakes mapped via ``LOOP_BAKE_MAP``) + a section table of every loop model
    (emit-last detail, outer arms, ssim2), fed verbatim by the jxl-encoder sweep summary
    JSON (``--loop-targeting``; counts/medians are READ, never re-derived here).

ENSEMBLE rows (2026-08-04): a fulleval JSON carrying ``model.kind == "ensemble"`` (stamped by
``scripts/promote_fulleval.py --members``) renders an ``ens×k`` marker everywhere the bake is
named, and its Model-details card leads with a warning that the architecture/repro shown is the
ANCHOR member. An ensemble is an evaluation FUNCTION, not a shippable artifact — its rank/dial/
corruption numbers come from the identical verdict invocation as every single-bake row and are
directly comparable, but ``m3_coherence``/``m3a_coherence`` are **null** because the coherence
instrument loads one ZNPR. Null renders as an em-dash (NOT MEASURED) and is excluded from column
shading and min/max — it is never displayed or shaded as a measured zero.

NO external requests: all CSS/JS/data are inlined (no CDN, no web fonts) so the file opens
offline. NO hand-rolled statistics: every SROCC/PLCC comes from the canonical ``panel`` (via the
fulleval JSON's precomputed ``scatter`` block, or computed at build through
``scripts/lib/zen_stats.panel`` when a JSON omits it). Only OLS fit-line endpoints are computed
here (numpy polyfit — a display aid, like bake_report.py), never an IQA stat.

Colors follow the dataviz skill's validated categorical palette (see the palette validator run in
the commit); identity is never color-alone — every series is labeled and the table view exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# dataviz validated categorical palette (8 hues, light/dark), + chart ink. Validated by
# scripts/validate_palette.js (light: all CVD/normal gates pass, 3 slots need the relief rule =
# labels+table, both provided; dark: all gates pass incl. contrast).
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
# ONE source for the page's CSS custom properties AND the ECharts option colors
# (DATA.chartThemes): the charts must ink from the same tokens as the page in both
# modes, and generating both from this dict makes drift impossible. Values are the
# dataviz-validated ones the CSS carried before the ECharts migration.
THEME_VARS = {
    "light": {
        "surface-1": "#fcfcfb", "plane": "#f9f9f7", "text-primary": "#0b0b0b",
        "text-secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
        "axis": "#c3c2b7", "border": "rgba(11,11,11,.10)",
        "good": "#0ca30c", "warn": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
        "seq-lo": "#cde2fb", "seq-hi": "#104281",
    },
    "dark": {
        "surface-1": "#1a1a19", "plane": "#0d0d0d", "text-primary": "#fff",
        "text-secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "border": "rgba(255,255,255,.10)",
        "good": "#0ca30c", "warn": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
        "seq-lo": "#0d366b", "seq-hi": "#cde2fb",
    },
}

# ---- vendored Apache ECharts (semantic zoom for the heavyweight panels) --------------
# The bundle is >30 KB so it is NEVER in git: scripts/v_next/vendor/echarts.pointer.md
# records version + block-storage path + sha256, and the build verifies + inlines it
# (the page allows no external requests). env ZEN_ECHARTS_JS overrides the file path.
ECHARTS_POINTER = Path(__file__).resolve().parent / "vendor" / "echarts.pointer.md"


def _load_echarts():
    """Return (bundle_js, version) for the sha256-verified vendored ECharts bundle.
    LOUD failure with download instructions when the bundle is missing or mismatched —
    a silently chartless page would be worse than no page."""
    meta = {}
    try:
        for ln in ECHARTS_POINTER.read_text().splitlines():
            m = re.match(r"^-\s+(\w+):\s+(.+?)\s*$", ln)
            if m:
                meta[m.group(1)] = m.group(2)
    except FileNotFoundError:
        raise SystemExit(f"ECharts pointer missing: {ECHARTS_POINTER} — restore it from git "
                         "(it names the vendored bundle + sha256)")
    path = Path(os.environ.get("ZEN_ECHARTS_JS") or meta.get("file", ""))
    want = (meta.get("sha256") or "").lower()
    if not path.is_file():
        raise SystemExit(
            f"vendored ECharts bundle NOT FOUND at {path}\n"
            f"Download it once (full instructions in {ECHARTS_POINTER}):\n"
            f"  mkdir -p {path.parent}\n"
            f"  curl -sL -o {path} {meta.get('upstream', '<upstream url in the pointer>')}\n"
            f"  sha256sum {path}   # must print {want}"
        )
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise SystemExit(
            f"vendored ECharts sha256 MISMATCH at {path}\n  pointer: {want}\n  actual:  {got}\n"
            f"Re-download per {ECHARTS_POINTER}, or update the pointer deliberately (version bump)."
        )
    js = data.decode("utf-8")
    if "</script" in js.lower():
        raise SystemExit("vendored ECharts contains '</script' — cannot be inlined safely")
    return js, meta.get("version", "?")
REFERENCES = ["mos", "jnd", "ssim2", "butter", "cvvdp"]
REF_LABELS = {"mos": "MOS (human)", "jnd": "JND (human)", "ssim2": "SSIMULACRA2",
              "butter": "butteraugli (↑=better)", "cvvdp": "ColorVideoVDP"}
# scoreboard columns beyond CID22: (key, header, higher_is_better, fmt)
CORP_ORDER = ["cid22", "nonphoto", "konjnd", "aic3", "aic4", "live", "csiq", "kadid", "tid"]
SCATTER_MAX = 500  # subsample dense per_pair for embedding — keeps the offline file responsive
MODEL_TRANSFORMS_EMBED = 48  # Model-details shows at most 48 transform chips (+ "+N more");
                             # embedding more per bake (944 on the A-arm lasso cells) is payload
                             # the page cannot display — the fulleval JSON on disk keeps them all

# ---- JXL loop-targeting (2/3-shot) summary — produced by the jxl-encoder repo's exact
# 2/3-shot sweep (raw cells: benchmarks/zensim_loop_23shot_2026-08-01.tsv there; doc:
# benchmarks/zensim_loop_23shot_2026-08-01.md). The dashboard READS the machine summary
# JSON — counts/medians are never re-derived here (no-duplication rule; the jxl analyze
# script is the owner). Loop-model keys -> gauntlet bake names (fulleval `name`): models
# not in this map (the outer arms + ssim2, which are not bakes) render only in the
# section's own table; bakes without loop data render an em-dash.
DEFAULT_LOOP_TARGETING = (
    # 2026-08-07: the BEATS-BUTTER panel (jxl-encoder
    # benchmarks/zensim_loop_beatbutter_2026-08-07.md) — supersedes the
    # 2026-08-05 file by carrying every entry (analyze-owner regenerated,
    # priors byte-reproduced) plus W10L9_h3ctrl2, the ADOPTED-DEFAULTS
    # frontier arm (candidate + h3-mag own-map + CTRL exp 1.0 / clamp 2.0 +
    # binned attribution): k2 18/27 med 1.19, k3 24/27 med 0.54 — beats the
    # butter comparator (outer_zensimA 12/27 j2, 14/27 j3) at both budgets.
    "/home/lilith/work/zen/jxl-encoder/benchmarks/zensim_loop_23shot_summary_2026-08-26.json"
)
# ---- 372 ERA SUFFIX (2026-08-30) ------------------------------------------------------
# A board name ending in this suffix is the SAME BAKE as its unsuffixed sibling, read on
# the dated current-extractor 372 root (/mnt/v/zen/zensim-training/2026-08-30-full-features-372,
# _MANIFEST.json build_commit ea16c7ee) instead of the 2026-05-15 root whose masked/IW block
# (f156-371) was a function of RAYON_NUM_THREADS. The stored-era row is NEVER overwritten —
# it stays, flagged by benchmarks/eval_annotations.json. Details + the era table:
# benchmarks/eval372_current_root_2026-08-30.md, benchmarks/board_era_rows_2026-08-30.md.
ERA372_CUR_SUFFIX = "@cur372"


def era_base_name(name: str) -> str:
    """The unsuffixed stem of a board name — i.e. the bake's identity, era stripped.
    Any rule that judges the MODEL (not the ruler it was read on) must scope on this,
    or an era twin gets judged differently from the row it is paired with."""
    return name[:-len(ERA372_CUR_SUFFIX)] if name.endswith(ERA372_CUR_SUFFIX) else name

LOOP_BAKE_MAP = {
    # loop-model key (summary JSON `models` key, = the sweep TSV run prefix)
    #   ->  bake `name` on the gauntlet board (fulleval JSON `name`).
    # Order matters: the FIRST model mapping to a bake is that bake's scoreboard
    # primary — the adopted frontier arm leads for the sota944 candidate.
    "W10L9_h3ctrl2": "W10L9_s4003_packed",
    "v47A_base": "v47_strict_QAT_native",
    "v47A_h3g20c135": "v47_strict_QAT_native",
    "B_base": "b_sdr_linear_cid80_inclwinsor_dense_dial",
    "bvls_base": "v02_bvls_NO_shaping",
    "outer_zensimA": "v47_strict_QAT_native",
    # sota944 wave-11 candidate (2026-08-05 panel): 944-class PRUNED bake,
    # folded-class loop route — k3 emit-best 15/27 (ties B, best inner median).
    "W10L9_base": "W10L9_s4003_packed",
    # blend2L_base's bake (mlp_2L_diverse_H128) has no fulleval JSON on the board —
    # its row shows the bake filename from the summary JSON; map it when one lands.
}


# ---- BOARD CURATION (registered rule 2026-08-04, dashboard-overhaul session) ----------
# The full sota944 campaign grid (~160 cells) lives on the board so that every number
# cited in any report is findable here. Two-tier presentation keeps that from drowning
# a fresh reader:
#   * CURATED (this list) = the era flagships + every arm-candidate / named leader from
#     benchmarks/sota944_campaign_2026-08-03.md + the six wave-5 ensembles + the wave-6
#     arm-G candidate. These are DEFAULT-VISIBLE and keep embedded per-pair scatter.
#   * Everything else = grid-interior: default-hidden (one family-toggle away), every
#     scalar stat present (scoreboard/heatmap/Mohammadi/bands/dial/gates), no embedded
#     scatter points (per_pair is skipped at build even when the JSON carries it; the
#     full data stays in the source verdict — `source_verdict` in each fulleval JSON).
# This list is THE owner of curation: scripts/promote_sota944_board.py imports it to
# decide --strip-per-pair at promotion time.
CURATED_BOARD = [
    # era flagships (pre-944 eras)
    "winner_dial_Ebothg_hfgain_winsor_dial", "Ebothg_scr0_5_dial",
    "ADD156_safesyn_only_raw_lasso", "b_sdr_linear_cid80_inclwinsor_dense_dial",
    "b_sdr_linear_cid80_inclwinsor_dense_dial" + ERA372_CUR_SUFFIX,
    "v47_strict_QAT_native", "coherent924_selected", "bhdr_linear_shaped_cvvdpmix",
    "v02_bvls_NO_shaping", "v02_bvls_NO_shaping" + ERA372_CUR_SUFFIX,
    # 372 ERA PAIRS (2026-08-30, benchmarks/board_era_rows_2026-08-30.md): the four
    # decision-relevant cells of the 41 ordering flips are default-visible as PAIRS —
    # shipped B (4th -> 1st on CID22), the 2-layer blend (new composite leader; its
    # published "+0.004 over B" is an era artifact), cl_tfm (1st -> LAST on KonJND and
    # AIC-3, composite -0.049) and the BVLS no-shaping arm (current-era KonJND leader).
    # cl_tfm's STORED half joins curation here so the pair reads together. The other
    # seven @cur372 rows are grid-interior behind the "@cur372" family toggle.
    "cl_tfm_corruption_LQ_MLP_s13",
    "cl_tfm_corruption_LQ_MLP_s13" + ERA372_CUR_SUFFIX,
    "mlp_2L_diverse_H128" + ERA372_CUR_SUFFIX,
    # the 944 era-bridge (EM4 evaluated on the 944 root = the bar source, 0.8923796503)
    "sota944_EM4_s42_on944root",
    # HDR-944 wave candidates (2026-08-27; D2 freeze pending — user asked for
    # default visibility)
    "HDR944_L1T1_s4005", "HDR944_L1T2_s4004",
    # classical reference metrics as PEER rows (user request 2026-08-28;
    # built from stored refmetrics per-pair tables — build_peer_fullevals.py)
    "peer_ssim2", "peer_butteraugli", "peer_cvvdp", "peer_iwssim",
    # HDR-944 retrain wave (2026-08-28): the selected winner + the
    # HF-discrimination runner-up
    "HDR944_L1T1_s4005_hfpack", "HDR944R_t2_s4003_hfpack",
    # SDR purity-retrain winner (2026-08-28 wave; freeze pending)
    "W10L9P_s4005_packed",
    # balance-campaign flagships (2026-08-28/29; standing rule: curated covers
    # LIVE candidates — the frozen pair + W11 stars + incumbent + lodestar finals)
    "W10L9PH_s4004_packed", "PH_s4004_e060", "W10L9_s4003_packed",
    "w11_s4014_e050", "w11_s4014_final",
    "LSTAR_s4021_packed", "LSTAR_s4022_packed", "LSTAR2_s4033_packed",
    # campaign arm candidates + named leaders (benchmarks/sota944_campaign_2026-08-03.md)
    "sota944_winner_A_bvls_X_AM5",       # arm A candidate = campaign winner (§SELECTION)
    "sota944_B_blend_lam1e-3_a0.7_w",    # arm B candidate
    "sota944_C_em944_s31",               # arm C candidate; closest-to-bar single bake
    "C_co1a_s1307",                      # amendment-3 arm-1 candidate
    "C_co2a_s1307",                      # arm-2 candidate + coherence-wave winner-by-rule
    "C_co3b_s1303",                      # arm-3 candidate + raw M3a leader (0.8470)
    "C_co3a_s1301",                      # wave-3 raw CID22 leader (0.89067)
    "C_co3a_s1307",                      # best bar coverage (4/5 rows)
    "C_co3a_s1319",                      # wave-4 arm-D CID22 leader (0.88851)
    "C_co3a_s1321",                      # wave-4 arm-D candidate
    "C_co4_s1303",                       # wave-4 arm-E candidate
    "C_co4_s1307",                       # arm-E CID22 leader + KonJND leader (0.4725)
    "sota944_nt223",                     # near-top arm selected (amendment 2)
    # the six wave-5 seed-ensembles (amendment 5)
    "sota944_ens_E1_k2", "sota944_ens_E1_k3", "sota944_ens_E1_k5", "sota944_ens_E1_k8",
    "sota944_ens_E2_diverse5", "sota944_ens_E3_all51",
    # wave-6 arm-G candidate (highest composite in the campaign; CID22 −0.00051 vs bar)
    "sota944_ens_GE2_trio",
    # wave-11 battery-selected candidate (k=8-confirmed recipe; entered the jxl
    # 2/3-shot loop panel 2026-08-05 — the first new-era model with loop columns)
    "W10L9_s4003_packed",
    # balance campaign (2026-08-28): the first fully-eligible two-zone candidate
    # (dominates incumbent on BOTH zones, G-OUT v2 PASS) + the frozen-lens picks
    "W10L9PH_s4004_packed", "W10L9PH_s4003_packed", "BAL_E1_s4010_s4006",
    # H-TRAJ checkpoint alternative on the live decision (M3a 0.833, 7/8 floors)
    "PH_s4004_e060",
    # W-LIN round-7b candidate-the-registered-rule-names (benchmarks/
    # wlin_round7_rawframe_2026-08-30.md §"The candidate the registered rule names"):
    # 3,583 B, 944-input, 5/5 bars + G-RANGE PASS. Curated 2026-08-31 as the ADD156
    # comparator the sparse-linear lane is actually judged against. It carries 7 of the
    # 14 board corpora (its verdict ran a partial corpus list) — the other seven render
    # as NOT-MEASURED em-dashes, never zeros.
    "Q7b_pools_g0.2_a0.2_b0.97",
    # HYBRID lane (benchmarks/hybrid_candidate_2026-09-01.md). Both earn
    # curation by the registered rule (arm-candidate / named leader, and the
    # first is an ensemble):
    #  * HYA_w084 = 0.84*W10L9PH_s4004_packed + 0.16*Q7b — the ONLY arm in the
    #    exam that passes W1 (no held-out human axis worse than peer_ssim2 by
    #    more than delta), which NEITHER parent does. Under the amended W4
    #    (<=1.25x the 156 walk) it is the lane's TEACHER / quality ceiling, not
    #    a passing candidate: it costs 2.6x the bar.
    #  * SADD_BIGLEG = the 156-class additive student that distillation was
    #    aimed at. Ties ADD156 on CID22/CSIQ and beats it on KonJND (0.5432,
    #    above peer_ssim2's 0.5272) at 31 coefficients / 4,117 B. It gains NO
    #    exam clause: on the regime-matched 372 grid both it and ADD156 fail W3
    #    (0.98541 vs 0.98495 pooled monotonicity, 2% of q>=85 ladders ending
    #    backwards either way).
    "HYA_w084", "SADD_BIGLEG",
]
# "Sprint bests" (user request 2026-08-28): ONE selected leader per sprint/era,
# newest last. The ensembles sprint's best is resolved at build time (highest
# composite among ensemble rows) rather than hardcoded.
SPRINT_BEST = [
    ("v0x QAT era", "v47_strict_QAT_native"),
    ("linear BVLS", "v02_bvls_NO_shaping"),
    ("shipped-B linear", "b_sdr_linear_cid80_inclwinsor_dense_dial"),
    ("Ebothg era", "winner_dial_Ebothg_hfgain_winsor_dial"),
    ("additive era", "ADD156_safesyn_only_raw_lasso"),
    ("924 coherent", "coherent924_selected"),
    ("944 era-bridge", "sota944_EM4_s42_on944root"),
    ("wave-7 kon leg", "sota944_B_konhead_w"),
    ("wave-10", "W10L9_s4001"),
    ("KFG sprint", "KFG75_s4101"),
    ("nt sprint", "sota944_nt223"),
    ("HDR-372 era", "bhdr_linear_shaped_cvvdpmix"),
    ("HDR-944 wave", "HDR944_L1T1_s4005_hfpack"),
    ("balance campaign", "W10L9PH_s4004_packed"),
]

CURATED = set(CURATED_BOARD)


def family_of(name: str) -> str:
    """Control-bar family grouping (group toggles). Input = the board name."""
    if name.endswith(ERA372_CUR_SUFFIX):
        # Era-suffixed rows read the SAME bake on the 2026-08-30 current-extractor 372
        # root; checked FIRST so a pair's two halves never land in different families
        # (every stem here also matches a prefix rule below).
        return "@cur372 (current extractor)"
    if name.startswith("peer_"):
        return "peers"
    n = name[len("sota944_"):] if name.startswith("sota944_") else name
    if n.startswith(("ens_", "W5_", "W6_")):
        return "ensembles"
    if n.startswith(("winner_A_", "A_")):
        return "arm A"
    if n.startswith(("B_", "B2_")):
        return "arm B"
    if n.startswith("C_em944"):
        return "arm C seeds"
    if n.startswith("C_ensk"):
        return "distilled"      # wave-6 arm F: single-bake students of the W5 ensembles
    if n.startswith(("C_co1", "C_co2", "C_co3", "C_co4")):
        return "coherence/W4"
    if n.startswith(("C_nt944", "nt")):
        return "near-top"
    if n.startswith("H_"):
        return "arm H (konjnd leg)"   # wave-7 cells + their packaged forms
    if n.startswith("EM4_"):
        return "era bridge"
    if n.startswith(("W10L9PH", "W10L9PB", "BAL_", "PH_s4004")):
        return "balance campaign"
    if n.startswith("W10L9P"):
        return "purity retrain"
    if n.startswith(("W10L9", "W10L", "KFG", "w11_")):
        return "wave 10/11"
    if n.startswith(("LSTAR", "lstar")):
        return "lodestar"
    if n.startswith(("HDR944", "bhdr")):
        return "HDR"
    # W-LIN lane (round 7/7b linear blends + the copperline wlin4 cells). Without this
    # branch every one of these stems fell through to "pre-944 era", which is both wrong
    # (they are 944-input) and invisible — the standing check on every new stem pattern.
    if n.startswith(("Q7b_", "T7b_", "H7b_", "wlin", "copperline_wlin")):
        return "W-LIN"
    return "pre-944 era"


DEFAULT_HFNL_AXIS = str(
    Path(__file__).resolve().parent.parent.parent / "benchmarks" / "hfnl_axis_2026-08-05.json")


def load_hfnl_axis(path=None):
    """Read the committed appendix-O HF-NL axis study JSON (per-model per-ref
    histograms + means/CIs, reference/ceiling rows, the registered axis LSD,
    split-half reliability). Values are READ verbatim, never re-derived here —
    the owner is the appendix-O battery (`panel --batch` over the per-pair-refs
    dumps; benchmarks/hfnl_axis_report_2026-08-05.md). Missing file -> the
    HF-NL panel is omitted with a loud note (loop-targeting pattern)."""
    p = Path(path or DEFAULT_HFNL_AXIS)
    if not p.exists():
        print(f"NOTE: hfnl axis JSON not found at {p} — HF-NL panel omitted", file=sys.stderr)
        return None
    return json.loads(p.read_text())


def load_loop_targeting(path=None):
    """Read the Part-A machine summary JSON (jxl-encoder sweep). Returns the embed dict
    {meta, models, bakeMap, modelBake} or None (missing file -> section omitted, loud note).
    Counts/medians are READ verbatim, never re-derived here."""
    p = Path(path or DEFAULT_LOOP_TARGETING)
    if not p.exists():
        print(f"NOTE: loop-targeting summary not found at {p} — JXL loop panel omitted",
              file=sys.stderr)
        return None
    o = json.loads(p.read_text())
    models = o.get("models") or {}
    bake_map = {}     # bake name -> PRIMARY loop-model key (first map hit wins = baseline arm)
    model_bake = {}   # loop-model key -> bake name (for the section table's bake column)
    for mk, bake in LOOP_BAKE_MAP.items():
        if mk in models:
            model_bake[mk] = bake
            if bake not in bake_map:
                bake_map[bake] = mk
    return {"meta": {k: o.get(k) for k in ("date", "matrix", "notes", "source") if k in o},
            "models": models, "bakeMap": bake_map, "modelBake": model_bake}


def regime_of(o):
    """Displayed regime = the model's TRUE input width (372/720/924/944-class), derived
    from n_inputs read out of the ZNPR itself. The stored `regime` flag string is
    cosmetic on the sota944 campaign verdicts (every one of the 166 board JSONs says
    "720" while n_inputs spans 156/372/504/720/924/944) — so the flag is only the
    fallback when no width is recorded. For an ensemble the width shown is the ANCHOR
    member's (the model block describes the anchor; the scoreboard caption says so)."""
    m = o.get("model") or {}
    n = m.get("n_inputs") or o.get("n_inputs")
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = None
    return str(n) if n else str(o.get("regime", "?"))


def _fit_line(x, y):
    """OLS endpoints [x0,y0,x1,y1] for the display trend (a fit aid, not a stat)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    a = np.polyfit(x, y, 1)
    x0, x1 = float(np.min(x)), float(np.max(x))
    return [round(x0, 4), round(float(a[0] * x0 + a[1]), 4),
            round(x1, 4), round(float(a[0] * x1 + a[1]), 4)]


def _panel_srocc_plcc(pred, ref):
    """Canonical SROCC/PLCC via the Rust panel shim (fallback when a JSON omits `scatter`)."""
    from lib.zen_stats import panel
    p = panel(list(map(float, pred)), list(map(float, ref)))
    return {"srocc": round(abs(p["srocc"]), 4), "plcc": round(p["plcc"], 4), "n": int(p["n"])}



# Corpus label ORIENTATION — read from THE registry, never hardcoded here.
#
# The owner is `EXPECTED_ORIENTATION` in
# scripts/canonical_corpus/check_target_orientation.py (campaign Appendix I,
# 2026-08-04): three eval corpora — aic4 (188/188 board fullevals negative),
# sdr25 (171/171), konjnd (187/188) — carry DISTORTION-oriented JND-family
# labels (`q_jnd` distance / PJND threshold), so their signed SROCC is negative
# BY CONVENTION and |SROCC| is the correct magnitude reading. Everything
# quality-oriented keeps its sign: a negative there is a genuine ranking
# INVERSION (the Appendix F failure mode) and must not earn credit anywhere.
# A POSITIVE signed SROCC on a distortion-oriented corpus is the same defect
# mirrored — an orientation MISMATCH against the declared convention.
#
# We AST-parse the owner instead of importing it (its module imports pyarrow +
# reads corpus paths at import time) and instead of copying the set (which is
# how display code drifts from the registry). Parse failure fails the BUILD:
# a board that silently mis-renders orientation is worse than no board.
def _load_expected_orientation():
    import ast
    p = Path(__file__).resolve().parents[2] / "scripts" / "canonical_corpus" / \
        "check_target_orientation.py"
    tree = ast.parse(p.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "EXPECTED_ORIENTATION"
                        for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            names = {"QUALITY": "quality", "DISTORTION": "distortion"}
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Name) and v.id in names:
                    out[k.value] = names[v.id]
            if out:
                return out
    raise SystemExit(f"EXPECTED_ORIENTATION not parseable from {p} — the board must not "
                     "guess corpus orientation; fix the registry (or this parser).")


EXPECTED_ORIENTATION = _load_expected_orientation()
SIGN_ABS_CORPORA = {c for c, o in EXPECTED_ORIENTATION.items() if o == "distortion"}


def _signed(rank, corpus):
    """Signed SROCC for `corpus` out of a fulleval `rank` block; |SROCC| for the
    JND↓ corpora in SIGN_ABS_CORPORA (distortion-oriented per EXPECTED_ORIENTATION —
    negative signed SROCC is their declared convention, not an inversion).

    Prefers `srocc_signed` (emitted by `bake_verdict` since 2026-07) and falls back to
    `srocc` for older JSONs, which is the only sign information those carry."""
    r = rank.get(corpus) or {}
    v = r.get("srocc_signed")
    if v is None:
        v = r.get("srocc")
    if v is None:
        return None
    return abs(v) if corpus in SIGN_ABS_CORPORA else v


def _composite(rank):
    """FALLBACK ONLY (pre-2026-07-26 JSONs). The canonical composite is the Rust
    `product_composite`, emitted as `composite` in the fulleval JSON; `load_fulleval`
    READS that and only calls this when the field is absent, so there is one source
    of truth (stats review Rec-7). Goal-aware ranking scalar (reuses blend_lib.composite
    — the owner — when importable, else a transparent documented fallback). rank:
    {corpus: {srocc,...}} with srocc already
    polarity-corrected (abs for JND corpora), per the fulleval schema."""
    def g(c):
        v = _signed(rank, c)
        return float(v) if v is not None and np.isfinite(v) else 0.0
    try:
        import blend_lib as B
        res = {}
        for c in B.VAL_CORPORA:
            v = g(c)
            res[c] = {"srocc": v, "srocc_abs": v}
        score, reject = B.composite(res)
        return round(float(score), 4), bool(reject)
    except Exception:
        # documented fallback (same weights as blend_lib.composite)
        score = g("cid22") + 0.30 * g("nonphoto") + 0.20 * g("konjnd") + 0.10 * g("aic3") + 0.05 * g("aic4")
        reject = (g("cid22") < 0.84) or (g("nonphoto") < 0.80)
        return round(score, 4), bool(reject)


def load_annotations_registry():
    """The committed invalidation/annotation registry (board-integrity pass
    2026-08-04): benchmarks/eval_annotations.json. Returns (entries, meta) where
    meta = {id: {kind, reason}} for tooltip embedding. Missing file -> ([], {})
    with a loud note (the board then renders no badges — never fabricated ones)."""
    p = Path(__file__).resolve().parents[2] / "benchmarks" / "eval_annotations.json"
    if not p.exists():
        print(f"NOTE: annotations registry not found at {p} — no ⚠ badges", file=sys.stderr)
        return [], {}
    reg = json.loads(p.read_text())
    entries = reg.get("entries", [])
    # `fields` rides along so the page can badge the SPECIFIC scoreboard column an
    # entry covers (COL_FIELD + annForCol below), instead of needing a hand-written
    # JS rule per entry id.
    meta = {e["id"]: {"kind": e.get("kind", ""), "reason": e.get("reason", ""),
                      "fields": list(e.get("fields") or [])}
            for e in entries if "id" in e}
    return entries, meta


def _ann_field_present(o, dotpath):
    cur = o
    for part in dotpath.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return cur is not None


def _ann_covers(entry_field, target_field):
    """Segment-boundary prefix match — the exact rule freeze_check's `ann_covers` uses
    (freeze_check.rs): `rank.hfnlproxy` covers `rank.hfnlproxy.per_ref_mean`;
    `rank.hfnl` covers neither. The JS side already carries this as `annCovers`."""
    return (target_field == entry_field
            or (target_field.startswith(entry_field)
                and target_field[len(entry_field):len(entry_field) + 1] == "."))


def _ann_matches(o, entry):
    """Mirror of freeze_check's scope predicates (missing/present/names/all).

    `{"manual": ...}` is the explicit documentation-only form (ADD156 ship
    audit D10): no machine predicate, so it badges NO cell — but it is still
    carried in `meta`/`DATA.annRegistry` so the finding stays visible. Before
    D10 an unrecognised scope fell through to `bool(scope.get("all"))` and was
    silently inert, indistinguishable from one that legitimately did not apply;
    freeze_check now REJECTS any other shape at load."""
    scope = entry.get("scope") or {}
    if "manual" in scope:
        return False
    if "missing" in scope:
        return not _ann_field_present(o, scope["missing"])
    if "present" in scope:
        return _ann_field_present(o, scope["present"])
    if "names" in scope:
        # MIRROR the Rust owner exactly (freeze_check.rs:596 `bake_name`): `name`, then
        # `bake`, then "?". Before 2026-09-04 this side read only `name`, so a fulleval
        # carrying `bake` but no `name` matched in freeze_check and NOT on the board —
        # a silent divergence between an owner and its documented mirror.
        nm = o.get("name") or o.get("bake") or "?"
        return nm in (scope["names"] or [])
    return bool(scope.get("all"))


# ============================ FAIRNESS LAYER (2026-09-04) =============================
# User request: "generate an updated gauntlet with all of the things we discovered
# corrected, and the data/models filtered to those we can verify we did fairly."
# Record: benchmarks/fair_gauntlet_2026-09-04.md. Nothing here re-derives a statistic —
# every number rendered is READ from a fulleval JSON, the annotations registry, the
# G-ADDR floor registry, or the ssim2-exam scorecard transcription.

# ---- SEED GROUPS -------------------------------------------------------------------
# A board cell is one TRAINING RUN. Several cells can be draws of the SAME recipe at
# different seeds, and a headline quoted from the best of them is a best-of-k statistic
# reported as if it were the recipe's value. MEASURED on this board's own control arm
# (benchmarks/fastclass_distill_wave_2026-09-04.md §7.1): A4b's published KonJND 0.4327
# and composite 0.8664 are the best of 3 seeds; the same recipe's mean is 0.3561 /
# 0.8572 and its KonJND seed spread is 0.133.
#
# THE GROUPING RULE (this function is the board's owner of it; `freeze_check --select`
# is adding a `--seed-group` key concurrently and MUST adopt the identical rule — see
# the doc's "pending unification" note):
#   1. Only single-model rows participate. An ensemble is an evaluation FUNCTION over
#      members, not a training replicate of anything.
#   2. Key = the embedded `zentrain.repro.argv` with `--seed` and the output-path flags
#      (and their values) removed. No argv -> UNGROUPABLE (and it also fails criterion
#      (a), since no argv means no embedded repro).
#   3. Rows inside a key are collapsed by SEED IDENTITY (`seed_identity`): two cells with
#      the SAME recipe
#      and the SAME seed are one training run promoted twice (MEASURED: 33 such pairs on
#      this board, e.g. A4b_s4004 / FC_C0_s4004 — identical argv modulo seed, identical
#      seed 4004, CID22 identical to 16 digits, different bake sha). So k = the number
#      of DISTINCT seeds, never the number of board rows.
# VALIDATION: this rule reproduces §7.1's numbers exactly on its own arm — k=3,
# KonJND best 0.4327 / mean 0.3561 / spread 0.1329, composite best 0.8664 / mean 0.8572.
#
# MIRRORED IN RUST (2026-09-04): `freeze_check --select --seed-group` implements the same
# three clauses — `norm_argv_for_seed_group` / `seed_group_key` / `seed_identity` in
# zensim-validate/src/bin/freeze_check.rs, sha1 and all, so a `--select` group id and a
# board `seed_group` id are the SAME STRING. `scripts/verify_seed_group_parity.py` is the
# gate: it runs both owners over the board's fulleval dir and fails on any disagreement in
# normalized argv, key, seed identity, or the resulting k>=2 partitions. ONE deliberate
# divergence, and only one: `build_seed_groups` returns k>=2 groups because the board
# renders k=1 rows through another path, while `--select` must rank every candidate it was
# handed and so keeps ungroupable/single-seed cells as labelled singletons.
#
# HOW A GROUP IS PRESENTED (USER CORRECTION, 2026-09-04, verbatim): *"data-subsets are
# not equal though — one might be more representative and diverse, while another sucks,
# objectively."* A seed drives pair SAMPLING, so a seed's subset can be objectively
# better- or worse-covered, and part of the spread is subset coverage rather than model
# variance. Consequences the board must honour, and does:
#   * the group is shown as MEAN + SPREAD (min-max) + k, with the PER-SEED values
#     reachable (hover, and every member row stays on the board),
#   * the mean ranks the default view — it is the honest estimator against best-of-k —
#     but it is NEVER labelled definitive,
#   * the badge says so, in `SEED_COVERAGE_NOTE`,
#   * when a bake carries `zentrain.sample_coverage` the board renders it beside the
#     seed. It LANDED 2026-09-04 (the trainer embeds it; `bake_verdict` surfaces it at
#     `repro.sample_coverage`, which is exactly where this file reads it), so bakes
#     trained from that commit on carry it and every earlier board row does not —
#     absent renders NOT MEASURED, never a zero.
SEED_COVERAGE_NOTE = ("seed spread partly reflects subset coverage; coverage is measured "
                      "per bake in `zentrain.sample_coverage` (bakes trained from "
                      "2026-09-04 on) and NOT MEASURED on earlier rows")
# `--init-seed` / `--sample-seed` joined the set 2026-09-04, when the ownerfix lane split
# the trainer's two RNG streams: a split run and an unsplit run of the same recipe must
# still land in one group. Both owners changed in the same commit.
SEED_GROUP_DROP_FLAGS = {"--seed", "--init-seed", "--sample-seed",
                         "--out", "--output", "-o", "--bake-out", "--manifest"}


def _norm_argv_for_seed_group(argv):
    out, i = [], 0
    argv = [str(x) for x in argv]
    while i < len(argv):
        tok = argv[i]
        if tok in SEED_GROUP_DROP_FLAGS:
            i += 2 if i + 1 < len(argv) and not argv[i + 1].startswith("--") else 1
            continue
        out.append(tok)
        i += 1
    return out


def seed_group_key(o):
    """Recipe identity of a fulleval object, seed and output path removed.
    None when the row cannot be grouped (ensemble, or no embedded repro argv)."""
    if (o.get("model") or {}).get("kind") == "ensemble":
        return None
    argv = (o.get("repro") or {}).get("argv") or []
    if not argv:
        return None
    return hashlib.sha1("\x00".join(_norm_argv_for_seed_group(argv)).encode()).hexdigest()[:12]


def seed_identity(o):
    """The seed a group collapses duplicates by (clause 3).

    The `(init, sample)` PAIR once the trainer's streams are split
    (`--init-seed` / `--sample-seed`, 2026-09-04), the single `seed` on a legacy
    bake, None when the repro records no seed at all. Mirrored in Rust as
    `freeze_check::seed_identity`; gated by scripts/verify_seed_group_parity.py."""
    r = o.get("repro") or {}
    i, p = r.get("init_seed"), r.get("sample_seed")
    if i is not None and p is not None:
        return f"{i}/{p}"
    s = r.get("seed")
    return None if s is None else str(s)


def build_seed_groups(objs):
    """[fulleval dicts] -> {group_id: {"k": distinct seeds, "members": [names],
    "reps": [names, one per distinct seed], "dup_seed": [[names sharing a seed], ...]}}.
    Only groups with k >= 2 are returned; every other row is UNREPLICATED or UNGROUPABLE."""
    by_key = {}
    for o in objs:
        key = seed_group_key(o)
        if not key:
            continue
        seed = seed_identity(o)
        if seed is None:
            continue
        by_key.setdefault(key, {}).setdefault(seed, []).append(o.get("name"))
    groups = {}
    for key, by_seed in by_key.items():
        if len(by_seed) < 2:
            continue
        members = sorted(n for ns in by_seed.values() for n in ns)
        groups[key] = {
            "k": len(by_seed),
            "members": members,
            # one representative per DISTINCT seed — the unit a k-seed statistic counts
            "reps": sorted(sorted(ns)[0] for ns in by_seed.values()),
            "seeds": {str(s): sorted(ns) for s, ns in sorted(by_seed.items(), key=lambda kv: str(kv[0]))},
            "dup_seed": sorted([sorted(ns) for ns in by_seed.values() if len(ns) > 1]),
        }
    return groups


def seed_group_stats(group, values):
    """{name: value} over a group's representatives -> mean / min / max / spread / n /
    the per-seed list. Returns None when fewer than 2 representatives carry the value.

    The mean is the honest estimator AGAINST best-of-k. It is NOT "the true score": a
    seed selects a data subset and subsets are not equally representative, so part of
    the spread is coverage, not model variance (SEED_COVERAGE_NOTE). Nothing here is a
    statistic in the IQA sense — mean/min/max of already-computed SROCCs, never a
    re-derived correlation."""
    per = [(n, values.get(n)) for n in group["reps"]]
    per = [(n, v) for n, v in per if isinstance(v, (int, float)) and np.isfinite(v)]
    if len(per) < 2:
        return None
    vs = [v for _, v in per]
    return {"mean": round(float(np.mean(vs)), 5), "min": round(min(vs), 5),
            "max": round(max(vs), 5), "spread": round(max(vs) - min(vs), 5),
            "n": len(vs), "per": [[n, round(float(v), 5)] for n, v in per]}


# ---- CIRCULARITY (ssim2 exam §2.1) --------------------------------------------------
# "`nonphoto`, `imazen26` and `hfnlproxy` are ssim2-anchored axes. Their targets are
# ssim2 scores. A model's number there is *agreement with ssim2*, never a win over it,
# and `peer_ssim2`'s 1.0 there is a definition, not a measurement."
# hf_nearlossless joins them by MEASUREMENT (CLAUDE.md, 2026-09-01): its `human_score`
# IS `ssim2_gpu / 100`, exactly, in float equality, on 1200/1200 rows.
CIRCULAR_AXES = ["nonphoto", "imazen26", "hfnlproxy", "hf_nearlossless"]
# train==val by construction; `rank.<c>.train_eq_val` is True on every board cell that
# carries them (429/429 measured). Integrity guards only, never ranking signal
# (registry `kadid-tid-train-eq-val`).
TRAIN_EQ_VAL_AXES = ["kadid", "tid"]
# The held-out human axes the exam's W1/W2 clauses are evaluated on, after the
# circularity exclusion and the train==val exclusion.
HELD_OUT_HUMAN_AXES = ["cid22", "konjnd", "aic3", "aic4", "csiq", "live"]

# ---- registry ids that make a row NOT verifiably fair -------------------------------
# (b) era: the only registered ERA defect that INVALIDATES a board row's rank/composite.
# The sibling entries (`board372-row-read-on-ext720-root-2026-08-30`,
# `eval372-basic-only-bakes-era-independent-2026-08-30`) are MEASURED corrections saying
# the era flag does NOT apply to those rows, so they are notes, not failures.
ERA_INVALIDATING_IDS = {"eval372-stored-root-thread-dependent-2026-08-30"}
# (f) ensemble/teacher units: appended by this pass; see the registry entry.
ENS_UNITS_IDS = {"distill-teacher-raw-units-pre-58baf010-2026-09-04"}
# (e) entries whose `fields` declaration has been CORRECTED by a later registry entry.
# Honoured explicitly, never silently: `balanced-composite-bandtail-abs` declares
# fields:["composite"], but its reason + evidence are about `freeze_check`'s
# balanced_composite, while a FULLEVAL's `composite` is product_composite — which
# carries no band term and cannot be touched by that defect. The correction is itself a
# committed registry entry (…-field-scope-corrected-2026-09-04); this set is the code
# that reads it. Without it, 164 rows were marked unfair for a defect that does not
# reach the number they are ranked on.
E_FIELD_SCOPE_SUPERSEDED = {
    "balanced-composite-bandtail-abs": "balanced-composite-bandtail-field-scope-corrected-2026-09-04",
}
# Bakes with a committed, byte-verified reproduce script satisfy criterion (a) even
# when the fulleval predates mandatory embedded repro (scripts/reproduce_*.sh).
REPRO_SCRIPT_VERIFIED = {
    "b_sdr_linear_cid80_inclwinsor_dense_dial": "scripts/reproduce_b.sh",
    "bhdr_linear_shaped_cvvdpmix": "scripts/reproduce_bhdr.sh",
    "v47_strict_QAT_native": "scripts/reproduce_v47.sh",
}

FAIR_CRITERIA = [
    ("a_repro", "embedded zentrain.repro (or a committed byte-verified reproduce script)"),
    ("b_era", "no registered ERA defect invalidates its ruler"),
    ("c_no_train_eq_val", "its ranking composite uses no train==val corpus"),
    ("d_seed", "seed group aggregated (k>=2), or badged UNREPLICATED"),
    ("e_no_invalidated", "no unresolved `invalidated` registry entry applies"),
    ("f_ens_units", "if ensemble/teacher-derived, built post `--score-units` (58baf010)"),
    ("g_split", "its eval read the canonical held-out val group"),
]


def fairness_of(o, ann_entries, seed_groups, name_to_group):
    """One board row -> {"tier", "fails": [criterion ids], "notes": [...], "k", ...}.

    Tiers:
      VERIFIED-FAIR  every criterion holds and the row is a replicated seed group
      FAIR-NOTED     every criterion holds, but k == 1 (UNREPLICATED) or the row is
                     UNGROUPABLE-but-repro'd; still fair, just noted
      LEGACY         at least one of (a)(b)(c)(e)(f)(g) fails -> default-hidden, badged
    Criterion (d) never FAILS by itself: an unreplicated cell is fair as long as it is
    badged as one. That is the brief's rule and it is what keeps k=1 rows honest rather
    than invisible."""
    name = o.get("name") or ""
    matched = [e for e in ann_entries if "id" in e and _ann_matches(o, e)]
    matched_ids = {e["id"] for e in matched}
    invalid_ids = {e["id"] for e in matched if e.get("kind") == "invalidated"}
    fails, notes = [], []

    # (a) reproducible
    has_repro = bool((o.get("repro") or {}).get("argv"))
    script = REPRO_SCRIPT_VERIFIED.get(era_base_name(name))
    if not has_repro and not script:
        fails.append("a_repro")
    elif not has_repro:
        notes.append("repro via " + script)

    # (b) era
    if invalid_ids & ERA_INVALIDATING_IDS:
        fails.append("b_era")
    if o.get("features_root"):
        notes.append("ruler recorded (features_root)")

    # (c) train==val: the product composite's own weights (CID22 / imazen26 / nonphoto /
    # KonJND / AIC-3 / AIC-4) carry no KADID and no TID, so this holds structurally for
    # every cell whose composite is the Rust product_composite. A cell whose composite
    # came from gauntlet's legacy fallback is flagged instead of trusted.
    if o.get("composite") is None and not o.get("peer"):
        notes.append("composite absent (legacy fallback would be used)")
    for c in TRAIN_EQ_VAL_AXES:
        if (o.get("rank", {}).get(c) or {}).get("train_eq_val") is False:
            fails.append("c_no_train_eq_val")   # a cell claiming KADID/TID are held out
            break

    # (d) replication
    gid = name_to_group.get(name)
    k = seed_groups[gid]["k"] if gid else (1 if (o.get("repro") or {}).get("seed") is not None else None)

    # (e) any unresolved invalidation THAT REACHES THE RANKING VIEW.
    # Blunt "any invalidated entry matches" is wrong and was measured wrong: two
    # `invalidated` entries carry {"all": true} over `gates` / `class`
    # (f8-b9-abs-bar-superseded, add156-audit-d3-unselectable-falsified-2026-08-31),
    # so every one of the 433 rows failed and the fair set was empty. Those are
    # caveats on OTHER columns and are already badged there by `annForCol`.
    # The rule instead uses the registry's OWN coverage semantics: an entry fails the
    # row iff one of its `fields` covers a field the RANKING view reads — `composite`,
    # or `rank.<axis>.srocc_signed` for an axis that survives the circularity and
    # train==val exclusions. So `rank.kadid.*` never fails a row (KADID is an integrity
    # guard, never ranking signal) and `rank.cid22.bands` never fails one either
    # (a band is not the ranking scalar), while `composite` and `rank` do.
    ranked_fields = ["composite"] + [f"rank.{c}.srocc_signed" for c in HELD_OUT_HUMAN_AXES]
    for e in matched:
        if e.get("kind") != "invalidated":
            continue
        if e["id"] in E_FIELD_SCOPE_SUPERSEDED:
            notes.append(e["id"] + " -> field scope corrected by "
                         + E_FIELD_SCOPE_SUPERSEDED[e["id"]])
            continue
        for ef in (e.get("fields") or []):
            if any(_ann_covers(ef, rf) for rf in ranked_fields):
                fails.append("e_no_invalidated")
                notes.append("invalidated: " + e["id"])
                break

    # (f) ensemble / teacher units
    if matched_ids & ENS_UNITS_IDS:
        fails.append("f_ens_units")

    # (g) split. `bake_verdict` reads the registered val parquets for every corpus it
    # reports, so the machine-checkable form here is: the row carries the gold holdout
    # (CID22) and reports it as held out.
    cid = o.get("rank", {}).get("cid22") or {}
    if not cid or cid.get("srocc_signed") is None and cid.get("srocc") is None:
        fails.append("g_split")
    elif cid.get("train_eq_val") is True:
        fails.append("g_split")

    fails = sorted(set(fails))
    if fails:
        tier = "LEGACY"
    elif gid:
        tier = "VERIFIED-FAIR"
    else:
        tier = "FAIR-NOTED"
    return {"tier": tier, "fails": fails, "notes": notes, "k": k, "group": gid,
            "ann": sorted(matched_ids)}


def load_ssim2_exam(path=None):
    """The W1-W7 exam scorecard, TRANSCRIBED verbatim from
    benchmarks/ssim2_replacement_bar_2026-08-31.md §3.0 into a committed JSON so the
    board can render it without re-deriving anything. Missing file -> no exam panel
    (the loud-note pattern used by loop-targeting and the HF-NL axis)."""
    p = Path(path) if path else (Path(__file__).resolve().parents[2] / "benchmarks"
                                 / "ssim2_exam_scorecard_2026-08-31.json")
    if not p.exists():
        print(f"NOTE: ssim2 exam scorecard not found at {p} — exam columns omitted", file=sys.stderr)
        return None
    return json.loads(p.read_text())


def load_gaddr_registry(path=None):
    """The G-ADDR dial-addressability floor registry, READ as it stands at regen time
    (owner: zensim-validate/src/dial_addressability.rs). Renders bars + tiers; a bake's
    own values come from its fulleval `dial.addressability` when present, and from the
    `dial` block's already-stored p5/p95/reach/dynamic_range/mono/tied for the six axes
    those cover. Every other axis is NOT MEASURED, never a zero."""
    p = Path(path) if path else (Path(__file__).resolve().parents[2] / "benchmarks"
                                 / "dial_addressability_floor_2026-09-04.json")
    if not p.exists():
        print(f"NOTE: G-ADDR registry not found at {p} — addressability columns omitted", file=sys.stderr)
        return None
    reg = json.loads(p.read_text())
    sch = reg.get("_schema") or {}
    refsets = sch.get("reference_sets") or {}
    # WHICH PIN SET IS THE BAR. Read from the registry, never assumed. As of 2026-09-04
    # the ACTIVE reference is `peer_ssim2` — USER DECISION, verbatim in the registry:
    # "I don't think we should pin to B, ssim2 seems a better mentor." A row carrying
    # `reference: "peer_ssim2"` is a BAR; a row without the field is a pre-2026-09-04
    # `shipped_b` row, which the registry says is "printed, never a bar" and labels
    # BIASED (A1/A3/A6 sit ABOVE the reference metric's own values on the same grid).
    # Both are rendered — the bar set decides PASS/FAIL, the incumbent set is context —
    # so a reader can always tell "worse than the mentor" from "worse than what shipped".
    ACTIVE_REF = "peer_ssim2"

    def _rows(key):
        act = [x for x in reg.get(key, []) if x.get("active")]
        bar = next((x for x in act if x.get("reference") == ACTIVE_REF), None)
        # the canonical instrument for the incumbent set = same sha as the bar row when
        # there is one (so the two sets describe the SAME instrument), else the first.
        sha = (bar or {}).get("dial_grid_sha256") or (bar or {}).get("probe_sha256")
        inc = next((x for x in act
                    if not x.get("reference")
                    and (sha is None
                         or x.get("dial_grid_sha256") == sha or x.get("probe_sha256") == sha)),
                   None) or next((x for x in act if not x.get("reference")), None)
        return bar, inc

    gbar, ginc = _rows("grids")
    nbar, ninc = _rows("negtail_probes")
    ibar, iinc = _rows("identity_probes")
    if gbar is None:
        print("NOTE: G-ADDR registry has no active `reference: peer_ssim2` grid row — "
              "falling back to the incumbent pins and SAYING SO on the page.", file=sys.stderr)
    return {"fixed_bars": reg.get("fixed_bars") or {},
            "reference_bake": reg.get("reference_bake") or {},
            "activeRef": ACTIVE_REF if gbar else "shipped_b",
            "refsets": refsets,
            "grid": gbar or ginc, "gridIncumbent": ginc if gbar else None,
            "negtail": nbar or ninc, "negtailIncumbent": ninc if nbar else None,
            "identity": ibar or iinc, "identityIncumbent": iinc if ibar else None}


# The six G-ADDR axes computable from what EVERY board cell already stores, and the
# registry field each bar is read from. The other nine (A1/A2 pooled min+max, A7-A9
# negative-tail, C3-C6 probes) need instruments that were never run for these cells:
# they render NOT MEASURED with that reason, never a zero.
GADDR_AXES = [
    ("A3", "p95", "p95", "ge", "robust ceiling"),
    ("A4", "p5", "p5", "le", "robust floor"),
    ("A5", "reach", "reach", "ge", "reach = max - min"),
    ("A6", "dynamic_range", "dynamic_range", "ge", "dynamic range = p95 - p5"),
    ("C1", "mono_pct", "mono", "ge", "monotonicity"),
    ("C2", "tied_pct", "tied", "le", "flat/clamp dead-zone"),
]
GADDR_NOT_MEASURED = ("A1/A2 need the pooled dial min+max, A7-A9 the negative-tail probe "
                      "and C3-C6 the negative-tail + identity probes; none of those were "
                      "run for this cell's verdict (the gate landed 2026-09-04, after it).")
# ======================================================================================


# Ladder-inversion zone matrix (bake_verdict `dial.zones`, 2026-08-31). Embedded
# COMPACTLY: the verdict block averages 9.4 KB and the board carries 322 of them
# (3.0 MB against a page already at ~20 MB), so the page takes the columns the
# failure panel reads and rounds them, ~1 KB/bake. Nothing is re-derived — every
# number below is copied out of the verdict.
ZONE_ORDER = ["q<50", "q50-85", "q>=85"]
# The columns the failure panel reads, in row order. Counts are integers copied
# verbatim; the three rates and the two magnitudes are the verdict's OWN stored
# values (rounded for the wire), never recomputed on the page.
ZONE_INT_COLS = ["n_pairs", "inv_material", "flat", "codec_sat", "n_ladders",
                 "ladders_with_inv", "ladders_ends_backwards"]
ZONE_RATE_COLS = [("inv_rate", 5), ("frac_ladders_with_inv", 4),
                  ("frac_ladders_ends_backwards", 4),
                  ("inv_mag_med", 2), ("inv_mag_max", 2)]
ZONE_ROW_KEYS = ZONE_INT_COLS + [c for c, _ in ZONE_RATE_COLS]


def compact_zones(dial):
    """`dial.zones` -> {"grid":…, "rows": {"split|key|zone": [ZONE_ROW_KEYS…]}}
    or None when the cell was not measured (never a zero — an unmeasured cell
    and a cell measured at zero are different facts)."""
    z = (dial or {}).get("zones")
    if not isinstance(z, dict) or not z.get("cells"):
        return None
    rows = {}
    for c in z["cells"]:
        k = f'{c.get("split")}|{c.get("key")}|{c.get("zone")}'
        row = [c.get(col, 0) for col in ZONE_INT_COLS]
        for col, nd in ZONE_RATE_COLS:
            v = c.get(col)
            row.append(None if v is None else round(float(v), nd))
        rows[k] = row
    grid = str(z.get("grid") or "")
    # the named worst ladders — the only form of "which content" a reader can go
    # and look at. Top 6 (the verdict keeps 12); each is [image, codec, class,
    # zone, endpoint delta, worst backwards step].
    worst = [[w.get("image_id"), w.get("codec"), w.get("class"), w.get("zone"),
              round(float(w.get("end_delta") or 0.0), 2),
              round(float(w.get("worst_step") or 0.0), 2)]
             for w in (z.get("worst_ladders") or [])[:6]]
    return {"grid": grid.rsplit("/", 1)[-1], "rows": rows, "worst": worst,
            "classes": z.get("class_images") or {}, "scheme": z.get("scheme")}


def load_fulleval(fulleval_dir, best_per_day=None):
    """Read every *.fulleval.json; order by best_per_day date when available. Returns the list of
    bake dicts prepared for embedding (subsampled scatter points + fit lines + composite)."""
    fulleval_dir = Path(fulleval_dir)
    files = sorted(fulleval_dir.glob("*.fulleval.json"))
    if not files:
        raise SystemExit(f"no *.fulleval.json in {fulleval_dir} — run make_stub_fulleval.py to stub them")
    order = {}
    bpd = Path(best_per_day) if best_per_day else fulleval_dir.parent / "best_per_day.json"
    if bpd.exists():
        try:
            for i, r in enumerate(json.loads(bpd.read_text())):
                order[r.get("name")] = (r.get("date", ""), i)
        except Exception:
            pass
    raw = [json.loads(f.read_text()) for f in files]
    raw.sort(key=lambda o: order.get(o.get("name"), (o.get("date", ""), 99)))
    # Curated-first presentation order (stable): flagships/candidates lead the chip list
    # and the scatter columns; grid-interior cells follow alphabetically.
    raw.sort(key=lambda o: ((0, CURATED_BOARD.index(o.get("name")), "")
                            if o.get("name") in CURATED
                            else (1, len(CURATED_BOARD), str(o.get("name", "")).lower())))

    ann_entries, _ann_meta = load_annotations_registry()
    # FAIRNESS LAYER (2026-09-04): seed groups + per-row tier, computed ONCE over the
    # whole set (a group is a property of the population, not of a row).
    seed_groups = build_seed_groups(raw)
    name_to_group = {n: gid for gid, g in seed_groups.items() for n in g["members"]}
    # group-level stats over the DISTINCT-seed representatives, for the axes the
    # scoreboard ranks on. Values are READ from each member's own fulleval.
    _vals = {}
    for _axis, _get in (("composite", lambda o: o.get("composite")),
                        ("cid22", lambda o: _signed(o.get("rank", {}), "cid22")),
                        ("konjnd", lambda o: _signed(o.get("rank", {}), "konjnd"))):
        _vals[_axis] = {o.get("name"): _get(o) for o in raw}
    for _gid, _g in seed_groups.items():
        _g["stats"] = {a: seed_group_stats(_g, _vals[a]) for a in _vals}
    gaddr = load_gaddr_registry()
    # Why a cell has no ladder-inversion split (measure_dial_zones.py records one
    # reason per board cell). NOT MEASURED must always come with its reason.
    zone_skip = {}
    _zl = Path("/mnt/v/output/zensim/failure-profiles-2026-08-31/dial_zones_measure_log.json")
    if _zl.exists():
        try:
            for r in json.loads(_zl.read_text()).get("records", []):
                if not r.get("measured") and r.get("name"):
                    zone_skip[r["name"]] = r.get("reason")
        except Exception:
            pass
    rng = np.random.RandomState(0)
    bakes = []
    for ci, o in enumerate(raw):
        name = o.get("name", f"bake{ci}")
        curated = name in CURATED
        rank = o.get("rank", {})
        matched_ann = [e["id"] for e in ann_entries if "id" in e and _ann_matches(o, e)]
        fair = fairness_of(o, ann_entries, seed_groups, name_to_group)
        # Prefer the Rust-emitted `composite` (product_composite is the single
        # source — stats review Rec-7); the dashboard READS it rather than
        # re-deriving a divergent one. `_composite` stays only as the fallback
        # for pre-2026-07-26 JSONs that predate the field. The reject gate is a
        # dashboard concern (CID22<0.84 or nonphoto<0.80), computed either way.
        emitted = o.get("composite")
        if emitted is not None:
            comp = round(float(emitted), 4)
            # SIGNED (2026-08-04, APPENDIX F): `abs()` here let an ANTI-CORRELATED
            # bake clear the reject gate on the strength of its inversion. CID22 and
            # nonphoto are quality-oriented, so a negative is a backwards ranker and
            # must reject. (konjnd is the only corpus whose sign is structurally
            # negative; it is not part of this gate.)
            cid = _signed(rank, "cid22")
            nph = _signed(rank, "nonphoto")
            reject = (cid is None or cid < 0.84) or (nph is not None and nph < 0.80)
        elif o.get("peer") or (o.get("model") or {}).get("kind") == "reference-metric":
            # A PEER row has no product_composite and never will — `bake_verdict` (the
            # formula's owner) does not run on a reference metric. Falling through to
            # `_composite` gave it the UNNORMALISED legacy sum (max ~1.65) while every
            # bake carries the normalised Rust value (max 1.0), so the four peers sat at
            # the top of the board's DEFAULT SORT with 1.11-1.42 against the best bake's
            # 0.872 — a scale artefact reading as "ssim2 beats every model". Publish NOT
            # MEASURED instead: the scoreboard renders an em-dash and sorts nulls last.
            comp = None
            cid, nph = _signed(rank, "cid22"), _signed(rank, "nonphoto")
            reject = (cid is None or cid < 0.84) or (nph is not None and nph < 0.80)
        else:
            comp, reject = _composite(rank)
        scatter_out = {}
        # Registered board-size rule (2026-08-04): scatter embeds for the CURATED set
        # only. Grid-interior cells keep every scalar stat; their per-pair data stays in
        # the source verdict (never deleted) — the charts degrade gracefully without it.
        # Registered board-size rule (2026-08-04) EXTENDED for the fairness pass
        # (2026-09-04): per-pair scatter embeds for the curated set MINUS the LEGACY
        # tier. A legacy row keeps every scalar stat and every badge; only its point
        # cloud is stripped, and its per-pair data stays in the source verdict (never
        # deleted) exactly as `--strip-per-pair` already does for grid-interior cells.
        # This is what buys the fair view its size budget without dropping any DATA.
        pp = o.get("per_pair", {}) if (curated and fair["tier"] != "LEGACY") else {}
        sc_json = o.get("scatter", {})
        for corp, cols in pp.items():
            pred = cols.get("pred")
            if not pred:
                continue
            pred = np.asarray(pred, float)
            n = len(pred)
            idx = np.arange(n) if n <= SCATTER_MAX else rng.permutation(n)[:SCATTER_MAX]
            cell = {}
            for ref in REFERENCES:
                if ref not in cols:
                    continue
                rv = np.asarray(cols[ref], float)
                pts = [[round(float(pred[i]), 4), round(float(rv[i]), 4)] for i in idx
                       if np.isfinite(pred[i]) and np.isfinite(rv[i])]
                stats = sc_json.get(corp, {}).get(ref)
                if not stats:                      # JSON omitted it -> canonical panel at build
                    stats = _panel_srocc_plcc(pred, rv)
                # Geometric plot diagnostics (user directive 2026-08-28): computed in
                # shape-normalized space (pred mapped BY RANK onto the reference's
                # quantiles), residual r = qq - ref in ref units:
                #   out4      — fraction outside the ±4·MAD envelope (G-OUT severe band)
                #   maxd/p99d — max / p99 |r| as a fraction of the ref span p1..p99
                #   cov/clump — fraction of 20 ref-span bins holding ≥0.5% of points /
                #               largest single-bin share (density structure)
                #   clampLo/Hi— mass sitting AT the prediction's exact min/max value
                #               (dial floor/ceiling saturation, the incumbent-5.4 class)
                geo = None
                okm = np.isfinite(pred) & np.isfinite(rv)
                if okm.sum() >= 50:
                    pv, rr = pred[okm], rv[okm]
                    order = np.argsort(pv, kind="stable")
                    qq = np.empty(len(pv)); qq[order] = np.sort(rr)
                    r = qq - rr
                    mad = float(np.median(np.abs(r - np.median(r))) * 1.4826) or 1e-9
                    span = float(np.percentile(rr, 99) - np.percentile(rr, 1)) or 1e-9
                    hist, _ = np.histogram(rr, bins=20)
                    geo = {"out4": round(float((np.abs(r) > 4 * mad).mean()), 4),
                           "maxd": round(float(np.max(np.abs(r)) / span), 3),
                           "p99d": round(float(np.percentile(np.abs(r), 99) / span), 3),
                           "cov": round(float((hist >= max(1, len(rr) // 200)).mean()), 2),
                           "clump": round(float(hist.max() / len(rr)), 2),
                           "clampLo": round(float((pv == pv.min()).mean()), 4),
                           "clampHi": round(float((pv == pv.max()).mean()), 4),
                           "mad": round(mad, 3)}
                cell[ref] = {"pts": pts, "fit": _fit_line(pred, rv), "geo": geo,
                             "srocc": stats.get("srocc"), "plcc": stats.get("plcc"),
                             "n": stats.get("n", len(pts))}
            if cell:
                scatter_out[corp] = cell
        model = o.get("model") or {}
        ft = model.get("feature_transforms")
        if isinstance(ft, list) and len(ft) > MODEL_TRANSFORMS_EMBED:
            model = dict(model)
            model["n_feature_transforms"] = len(ft)   # true count for the card
            model["feature_transforms"] = ft[:MODEL_TRANSFORMS_EMBED]
        # board-integrity pass (2026-08-04): registry annotations, dominance
        # marks and the static block-usage fingerprint ride into the embed.
        # (matched_ann / fair are computed above — the scatter policy reads the tier)
        # `zentrain.sample_coverage` — landed 2026-09-04. The trainer embeds it and
        # `bake_verdict` surfaces it here, so a bake trained from that commit on
        # renders coverage beside the seed and every earlier row reads NOT MEASURED.
        # Absent is NOT MEASURED, never a zero.
        fair["coverage"] = (o.get("repro") or {}).get("sample_coverage")
        # G-ADDR: the six axes every cell already stores (dial p5/p95/reach/DR/mono/tied)
        # against the registry's bars, PLUS the emitted `dial.addressability` block when
        # a verdict carries one (the gate landed 2026-09-04; no board cell has it yet).
        addr = (o.get("dial") or {}).get("addressability")
        gcells, gpass, gfail = [], 0, 0
        gmeta = None
        if isinstance(addr, dict) and isinstance(addr.get("checks"), list):
            # The cell CARRIES the owner's own G-ADDR verdict (grafted by
            # `promote_fulleval.py --graft-gaddr`, sha-gated + same-grid gated).
            # Render THAT — all 15 rows, both tiers — instead of the six the board
            # can re-derive from stored dial scalars. Nothing is recomputed here:
            # every state/bar/measured value is copied out of the block.
            _cmp = {"\u2265": "ge", "\u2264": "le", ">": "gt", "<": "lt"}
            for c in addr["checks"]:
                st = c.get("state")
                mv, bar, inc = c.get("measured"), c.get("bar"), c.get("incumbent")
                gpass += (st == "pass")
                gfail += (st == "fail")
                gcells.append([c.get("id"), c.get("what"),
                               None if mv is None else round(float(mv), 5),
                               None if bar is None else round(float(bar), 5),
                               _cmp.get(c.get("cmp"), c.get("cmp")),
                               True if st == "pass" else (False if st == "fail" else None),
                               None if inc is None else round(float(inc), 5),
                               c.get("tier"), st, (c.get("note") or "")[:220]])
            cfail = [c.get("id") for c in addr["checks"]
                     if c.get("tier") == "contract" and c.get("state") == "fail"]
            cnm = [c.get("id") for c in addr["checks"]
                   if c.get("tier") == "contract" and c.get("state") == "not_measured"]
            gmeta = {"src": "emitted", "total": len(addr["checks"]),
                     "headline": addr.get("headline"), "contract": addr.get("contract"),
                     "regression": addr.get("regression"),
                     "shippable": bool(addr.get("shippable")),
                     "nm": addr.get("n_not_measured"),
                     "ref": addr.get("reference") or addr.get("active_reference"),
                     "grid": addr.get("grid_label"),
                     "cfail": cfail, "cnm": cnm}
        elif gaddr and gaddr.get("grid"):
            _grid = gaddr["grid"]
            _fx = gaddr["fixed_bars"]
            for aid, dial_key, bar_key, cmp_, what in GADDR_AXES:
                mv = (o.get("dial") or {}).get(dial_key)
                bar = _grid.get(bar_key) if aid.startswith("A") else _fx.get(
                    {"mono": "mono_min", "tied": "tied_max"}[bar_key])
                _tier = "regression" if aid.startswith("A") else "contract"
                if mv is None or bar is None:
                    gcells.append([aid, what, None, bar, cmp_, None, None,
                                   _tier, "not_measured", ""])
                    continue
                ok = (mv >= bar) if cmp_ == "ge" else (mv <= bar)
                gpass += ok
                gfail += (not ok)
                inc = None
                if aid.startswith("A") and gaddr.get("gridIncumbent"):
                    iv = gaddr["gridIncumbent"].get(bar_key)
                    inc = None if iv is None else round(float(iv), 5)
                gcells.append([aid, what, round(float(mv), 5), round(float(bar), 5), cmp_,
                               bool(ok), inc, _tier, "pass" if ok else "fail", ""])
        blocks = None
        bp = o.get("block_profile")
        if isinstance(bp, dict):
            fams = bp.get("families") or {}
            blocks = {"uses156": bool(bp.get("uses_f156_371")),
                      "fams": {k: [v.get("used"), v.get("cols")]
                               for k, v in fams.items() if isinstance(v, dict)}}
        # Training provenance (user ask 2026-08-27): date the bake was produced +
        # a parsed recipe so theory/training DIFFS read straight off the board.
        # Source of truth = the embedded zentrain.repro (timestamp_epoch,
        # target_column, legs, arch); bakes predating mandatory-repro fall back
        # to the bake FILE mtime, labeled src:"file" so nobody mistakes it for
        # a training timestamp.
        repro_o = o.get("repro") or {}
        train_date = None
        if isinstance(repro_o, dict) and repro_o.get("timestamp_epoch"):
            import datetime as _dt
            train_date = {"d": _dt.datetime.fromtimestamp(
                int(repro_o["timestamp_epoch"]), _dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "src": "repro"}
        else:
            bp = o.get("bake")
            if bp and Path(bp).exists():
                import datetime as _dt
                train_date = {"d": _dt.datetime.fromtimestamp(
                    Path(bp).stat().st_mtime, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "src": "file"}
        recipe = None
        if isinstance(repro_o, dict) and repro_o:
            legs = []
            for inp in (repro_o.get("inputs") or []):
                if isinstance(inp, dict):
                    legs.append({"name": inp.get("name", "?"),
                                 "nf": inp.get("n_features"),
                                 "rows": inp.get("rows") or inp.get("n_rows")})
            argv = repro_o.get("argv") or []
            extras = []
            interesting = {"--coarse-decay", "--coarse-l2-mult", "--qat-fine-tune-epochs",
                           "--dataset-weights", "--feature-set", "--auto-transforms",
                           "--target-scale", "--keep-features", "--also", "--val-policy",
                           "--out-dtype", "--allow-narrow-features"}
            i = 0
            while i < len(argv):
                tok = str(argv[i])
                if tok in interesting:
                    val = str(argv[i + 1]) if i + 1 < len(argv) and not str(argv[i + 1]).startswith("--") else ""
                    extras.append((tok.lstrip("-") + (" " + val if val else "")).strip())
                    i += 2 if val else 1
                else:
                    i += 1
            recipe = {"target": repro_o.get("target_column"),
                      "hidden": repro_o.get("n_hidden_layers"),
                      "width": repro_o.get("max_features"),
                      "epochs": repro_o.get("epochs"),
                      "pairs": repro_o.get("pairs_per_epoch"),
                      "seed": repro_o.get("seed"),
                      "best_val": repro_o.get("best_val"),
                      "legs": legs, "extras": extras}
        bakes.append({
            "name": name, "regime": regime_of(o), "regime_flag": o.get("regime", "?"),
            "train_date": train_date, "recipe": recipe,
            "curated": curated, "family": family_of(name),
            "date": o.get("date", ""), "colorIndex": ci,
            # dial WITHOUT `zones` — the raw zone block is ~9.4 KB/cell (3.0 MB
            # across the board); `compact_zones` is the ~1 KB projection the
            # failure panel actually reads.
            "rank": rank, "dial": {k: v for k, v in (o.get("dial") or {}).items() if k != "zones"},
            "m3": o.get("m3_coherence"),
            "zones": compact_zones(o.get("dial")),
            "zoneSkip": zone_skip.get(name),
            "m3a": o.get("m3a_coherence"),
            "corruption": o.get("corruption", {}), "composite": comp, "reject": reject,
            "m3_dropped_mass": o.get("m3_dropped_mass_pct"),
            "gates": o.get("gates") or {},
            "model": model,
            "repro": o.get("repro"),
            "annotations": matched_ann,
            "fair": fair,
            "gaddr": dict({"cells": gcells, "pass": gpass, "fail": gfail,
                           "emitted": bool(gmeta)}, **(gmeta or {"src": "derived",
                                                                 "total": 6})) if gcells else None,
            "dominated_by": o.get("dominated_by") or [],
            "blocks": blocks,
            "scatter": scatter_out, "is_stub": bool(o.get("_stub")),
        })
    # ride the population-level artefacts on the first row so build_html can lift them
    # without changing load_fulleval's return type (its callers are the dashboard's
    # --fulleval-dir mode and the fairness TSV emitter).
    if bakes:
        bakes[0]["_seed_groups"] = seed_groups
        bakes[0]["_gaddr"] = gaddr
    return bakes


def write_fairness_tsv(bakes, path):
    """The committed audit surface for the filter: one row per board cell ->
    tier -> failing criteria -> k -> group stats -> matched registry ids.
    Written from the SAME `fairness_of` decisions the board renders, so the page and
    the file can never disagree."""
    seed_groups = {}
    for b in bakes:
        if "_seed_groups" in b:
            seed_groups = b["_seed_groups"]
    n2g = {n: g for g, v in seed_groups.items() for n in v["members"]}
    cols = ["name", "tier", "fails", "k", "seed_group", "composite",
            "composite_k_mean", "composite_k_spread", "composite_k_min",
            "composite_k_max", "cid22_signed", "cid22_k_mean", "konjnd_signed",
            "konjnd_k_mean", "sample_coverage", "curated", "regime", "gaddr_pass",
            "gaddr_fail", "annotations", "notes"]
    lines = ["\t".join(cols)]
    for b in sorted(bakes, key=lambda x: str(x.get("name"))):
        f = b.get("fair") or {}
        gid = n2g.get(b.get("name"))
        st = (seed_groups.get(gid) or {}).get("stats") or {}
        c, ci, ko = st.get("composite"), st.get("cid22"), st.get("konjnd")
        g = b.get("gaddr") or {}
        row = [
            b.get("name", ""), f.get("tier", "?"), ";".join(f.get("fails") or []) or "-",
            "" if f.get("k") is None else str(f["k"]), gid or "-",
            "" if b.get("composite") is None else f"{b['composite']:.6f}",
            "" if not c else f"{c['mean']:.6f}", "" if not c else f"{c['spread']:.6f}",
            "" if not c else f"{c['min']:.6f}", "" if not c else f"{c['max']:.6f}",
            "" if _signed(b.get("rank", {}), "cid22") is None else f"{_signed(b.get('rank', {}), 'cid22'):.6f}",
            "" if not ci else f"{ci['mean']:.6f}",
            "" if _signed(b.get("rank", {}), "konjnd") is None else f"{_signed(b.get('rank', {}), 'konjnd'):.6f}",
            "" if not ko else f"{ko['mean']:.6f}",
            "NOT MEASURED" if f.get("coverage") is None else json.dumps(f["coverage"]),
            "1" if b.get("curated") else "0", str(b.get("regime", "")),
            "" if not g else str(g.get("pass", "")), "" if not g else str(g.get("fail", "")),
            ";".join(b.get("annotations") or []) or "-",
            " | ".join(f.get("notes") or []) or "-",
        ]
        lines.append("\t".join(x.replace("\t", " ") for x in row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    return path, len(lines) - 1


# ------------------------------------------------------------------ HTML assembly ------------
def _css_vars(mode):
    """One CSS custom-property run generated from THEME_VARS (same dict the charts ink from)."""
    return " ".join(f"--{k}:{v};" for k, v in THEME_VARS[mode].items())


_CSS = """
:root{color-scheme:light dark}
.viz-root{
  /*__LIGHT_VARS__*/
  color:var(--text-primary); background:var(--plane);
  font:13px system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
  /*__DARK_VARS__*/
}}
:root[data-theme="dark"] .viz-root{
  /*__DARK_VARS__*/
}
.viz-root{margin:0;padding:1.1rem 1.3rem 4rem}
h1{font-size:1.32rem;margin:.1rem 0 .2rem}
h2{font-size:1.03rem;margin:1.5rem 0 .5rem;border-top:1px solid var(--axis);padding-top:.55rem}
.sub{color:var(--text-secondary);max-width:70rem;line-height:1.45}
a{color:var(--seq-hi)}
code{background:var(--surface-1);border:1px solid var(--border);padding:.05rem .3rem;border-radius:3px;font-size:11px}
.bar{position:sticky;top:0;z-index:5;background:var(--plane);border-bottom:1px solid var(--border);
     padding:.5rem 0 .55rem;margin-bottom:.4rem;display:flex;flex-wrap:wrap;gap:.55rem;align-items:center}
.chip{display:inline-flex;align-items:center;gap:.4rem;padding:.24rem .55rem;border:1px solid var(--border);
      border-radius:1rem;cursor:pointer;user-select:none;background:var(--surface-1);font-size:12px;white-space:nowrap}
.chip input{margin:0;cursor:pointer}
.chip.off{opacity:.4}
.gchip{display:inline-flex;align-items:center;gap:.25rem;padding:.2rem .5rem;border:1px dashed var(--border);
       border-radius:.4rem;cursor:pointer;user-select:none;background:var(--surface-1);font-size:11.5px;white-space:nowrap}
.gchip.off{opacity:.45}
.gchip:hover{border-color:var(--muted)}
.allbakes{flex:1 1 100%}
.allbakes summary{cursor:pointer}
.chipwrap{display:flex;flex-wrap:wrap;gap:.4rem;max-height:180px;overflow-y:auto;padding:.35rem 0}
.echart{flex:0 0 auto}
.sw{width:11px;height:11px;border-radius:50%;flex:0 0 auto;border:1px solid var(--border)}
.btn{padding:.24rem .6rem;border:1px solid var(--border);border-radius:.35rem;background:var(--surface-1);
     color:var(--text-primary);cursor:pointer;font-size:12px}
.btn:hover{border-color:var(--muted)}
.btn.active{background:var(--seq-hi);color:#fff;border-color:var(--seq-hi)}
.tabs{display:flex;gap:.3rem;flex-wrap:wrap}
table{border-collapse:collapse;margin:.4rem 0;font-size:11.5px;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--border);padding:3px 7px;text-align:right;white-space:nowrap}
th{cursor:pointer;background:var(--surface-1);position:relative;text-align:right}
th:hover{color:var(--seq-hi)}
th.sorted::after{content:" \\2193";font-size:9px}
th.sorted.asc::after{content:" \\2191"}
td.lbl,th.lbl{text-align:left;font-weight:600}
tr.reject td{opacity:.55}
.grid{display:flex;flex-wrap:wrap;gap:.5rem;align-items:flex-start}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:6px;padding:.35rem .4rem}
.cap{font-size:10.5px;color:var(--muted)}
.scrow{display:flex;flex-wrap:wrap;gap:.5rem;align-items:flex-start;margin:.35rem 0 .6rem}
.corpttl{font-weight:600;margin:.7rem 0 .1rem;font-size:12.5px}
.corpttl .badge{font-weight:600;font-size:9.5px;padding:.05rem .35rem;border-radius:.25rem;color:#fff;margin-left:.4rem}
svg{display:block;max-width:100%;height:auto}
.tt{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--muted);
    border-radius:4px;padding:.25rem .45rem;font-size:11px;z-index:20;opacity:0;transition:opacity .08s;
    box-shadow:0 2px 8px rgba(0,0,0,.18);white-space:nowrap}
.stub{color:var(--serious);font-weight:600}
"""


def build_html(bakes, out_path, title="zensim summer gauntlet", loop_targeting=None,
               hfnl_axis=None, fair_only=False):
    ech_js, ech_ver = _load_echarts()
    _, ann_meta = load_annotations_registry()
    # population-level artefacts lifted off row 0 (see load_fulleval)
    seed_groups = (bakes[0].pop("_seed_groups", None) if bakes else None) or {}
    gaddr_reg = (bakes[0].pop("_gaddr", None) if bakes else None)
    exam = load_ssim2_exam()
    # FAIR-ONLY board (2026-09-04): the LEGACY tier is dropped from the FILE, not
    # hidden in it — that is the only way the fair view fits the registered 12 MB cap.
    # Nothing is deleted: every legacy row keeps its fulleval JSON on disk and appears
    # in full on the companion all-rows board and in the committed fairness TSV.
    n_legacy_dropped = 0
    if fair_only:
        n_legacy_dropped = sum(1 for b in bakes if (b.get("fair") or {}).get("tier") == "LEGACY")
        bakes = [b for b in bakes if (b.get("fair") or {}).get("tier") != "LEGACY"]
    # FULL dominance exclusion (user directive 2026-08-28): dominated bakes are
    # EXCLUDED from the board entirely (files + fullevals retained on disk;
    # marks from promote_fulleval --mark-dominated, rule strict-pareto-2026-08-04).
    n_dom = sum(1 for b in bakes if b.get("dominated_by"))
    bakes = [b for b in bakes if not b.get("dominated_by")]
    # Knob-end failure (user directive 2026-08-28): a model that cannot reach
    # or span the top of the dial (G-GRAN semantics: HF-zone q>=88 per codec,
    # top p50 >= incumbent-reach - 1 and span >= 8) is EXCLUDED from the
    # DEFAULT compare set (still toggleable, unlike dominance).
    _REACH = {"avif": 96.2, "jpeg": 94.4, "jxl": 96.6, "webp": 91.9}
    for b in bakes:
        # Scope: the SDR dial grid judges SDR dial models only — peers are
        # reference metrics (not dials) and HDR-family bakes are judged on
        # the HDR route panel, not this grid.
        # Scope on the ERA-STRIPPED stem: an @cur372 twin is the same MODEL as its
        # sibling, so the peers/HDR exemption must apply identically to both (its
        # `dial` block is bit-equal — both eras read the same --dial-grid file).
        nm = era_base_name(b.get("name") or "")
        fam = family_of(nm)
        if nm.startswith("peer_") or fam in ("HDR", "peers"):
            b["knob_end_fail"] = []
            continue
        fails = []
        curves = ((b.get("dial") or {}).get("curves") or {})
        for c, pts in curves.items():
            if c not in _REACH:
                continue
            hf = sorted([p for p in pts if p[0] >= 88])
            if len(hf) < 3:
                continue
            p50 = [p[2] for p in hf]
            if (p50[-1] - p50[0]) < 8 or p50[-1] < _REACH[c] - 1:
                fails.append(c)
        b["knob_end_fail"] = fails
    # codename registry (user directive 2026-08-28: memorable word-chain names).
    _np = Path(__file__).resolve().parents[2] / "benchmarks" / "candidate_names.json"
    _nm = {}
    if _np.exists():
        for cn, e in json.loads(_np.read_text()).get("names", {}).items():
            if e.get("board_name"):
                _nm[e["board_name"]] = cn
    for b in bakes:
        b["codename"] = _nm.get(b.get("name") or "")
    # gates + stars overlay (user directive 2026-08-28): committed, append-only,
    # measured-verdicts-only registry (absent = NOT MEASURED, never failed).
    _gp = Path(__file__).resolve().parents[2] / "benchmarks" / "board_gates_2026-08-28.json"
    _gov = json.loads(_gp.read_text()).get("bakes", {}) if _gp.exists() else {}
    for b in bakes:
        ov = _gov.get(b.get("name") or "", {})
        b["gatecheck"] = {k: v for k, v in ov.items() if k != "star"}
        b["star"] = ov.get("star")
    for b in bakes:
        b["_n_dominated_excluded"] = n_dom  # caption reads it off any row
    # loop-utility PROXY (owner: scripts/v_next/loop_proxy.py; committed JSON —
    # READ, never re-derived here). Name map: proxy run names -> board names.
    proxy = None
    _pp = Path(__file__).resolve().parents[2] / "benchmarks" / "loop_proxy_2026-08-28.json"
    if _pp.exists():
        _pj = json.loads(_pp.read_text())
        _pmap = {"A_PH_s4004": "W10L9PH_s4004_packed", "B_e060": "PH_s4004_e060",
                 "incumbent": "W10L9_s4003_packed", "GL2": "R1_GL2_s2503_packed"}
        proxy = {_pmap.get(k, k): v for k, v in _pj.items()}
    # discussion sets (user directive 2026-08-28): dropdown filters to the latest
    # discussion's models UNION incumbents UNION iqa peers.
    _dp = Path(__file__).resolve().parents[2] / "benchmarks" / "board_discussion_sets.json"
    _ds, _inc = [], []
    if _dp.exists():
        _dj = json.loads(_dp.read_text())
        _ds = sorted(_dj.get("sets", []), key=lambda x: x.get("date", ""), reverse=True)
        _inc = _dj.get("incumbents", [])
    _cp = Path(__file__).resolve().parents[2] / "benchmarks" / "loop_eval_coverage.json"
    _cov = json.loads(_cp.read_text()).get("rows", []) if _cp.exists() else []
    data = {"bakes": bakes, "loopCoverage": _cov,
            "zoneRowKeys": ZONE_ROW_KEYS,
        "discussionSets": _ds, "incumbents": _inc,
            "loopProxy": proxy, "nDominatedExcluded": n_dom,
            "palette": PALETTE, "references": REFERENCES,
            "refLabels": REF_LABELS, "corpOrder": CORP_ORDER,
            "chartThemes": THEME_VARS, "echartsVersion": ech_ver,
            "annRegistry": ann_meta,
            "orientation": EXPECTED_ORIENTATION,
            # ---- fairness layer (2026-09-04) ----
            "seedGroups": seed_groups,
            "seedCoverageNote": SEED_COVERAGE_NOTE,
            "fairCriteria": FAIR_CRITERIA,
            "fairOnly": bool(fair_only),
            "nLegacyDropped": n_legacy_dropped,
            "circularAxes": CIRCULAR_AXES,
            "trainEqValAxes": TRAIN_EQ_VAL_AXES,
            "heldOutHumanAxes": HELD_OUT_HUMAN_AXES,
            "exam": exam,
            "gaddrRegistry": gaddr_reg,
            "gaddrNotMeasured": GADDR_NOT_MEASURED,
            "loopTargeting": loop_targeting,
            "hfnlAxis": hfnl_axis}
    present = {b.get("name") for b in bakes}
    sprint = [{"s": lbl, "n": n} for lbl, n in SPRINT_BEST if n in present]
    ens = [b for b in bakes if (b.get("model") or {}).get("kind") == "ensemble"
           and isinstance(b.get("composite"), (int, float))]
    if ens:
        best_e = max(ens, key=lambda b: b["composite"])
        sprint.append({"s": "ensembles", "n": best_e["name"]})
    data["sprintBest"] = sprint
    any_stub = any(b.get("is_stub") for b in bakes)
    stub_note = ("<span class='stub'>STUB DATA</span> — synthesized fixtures "
                 "(<code>make_stub_fulleval.py</code>); drop the eval agent's real "
                 "<code>*.fulleval.json</code> in and re-run. " if any_stub else "")
    n_cur = sum(1 for b in bakes if b.get("curated"))
    _tiers = {}
    for b in bakes:
        _tiers[(b.get("fair") or {}).get("tier", "?")] = _tiers.get(
            (b.get("fair") or {}).get("tier", "?"), 0) + 1
    fair_note = (
        "<b>FAIRNESS FILTER (2026-09-04).</b> The default view is <b>VERIFIED-FAIR</b>: "
        + str(_tiers.get("VERIFIED-FAIR", 0)) + " rows that carry an embedded repro, an "
        "uninvalidated ruler, no train==val corpus in their ranking composite, a "
        "replicated seed group, no unresolved <i>invalidated</i> registry entry, no "
        "pre-<code>--score-units</code> teacher, and a held-out CID22 read. "
        + str(_tiers.get("FAIR-NOTED", 0)) + " rows are <b>FAIR-NOTED</b> (fair, but "
        "UNREPLICATED at k=1 or ungroupable) and "
        + str(_tiers.get("LEGACY", 0)) + " are <b>LEGACY</b> "
        + ("(dropped from THIS file to fit the 12 MB cap — they are complete on the "
           "companion all-rows board and in the committed fairness TSV; nothing was "
           "deleted). " if fair_only else "(default-hidden behind the <i>legacy / "
           "unverified</i> toggle, with the failing criterion badged on the row). ")
        + "Per-row tiers + failing criteria: <code>benchmarks/fair_gauntlet_2026-09-04.md</code>. "
        + ("<b>" + str(n_legacy_dropped) + " LEGACY rows are not in this file.</b> " if n_legacy_dropped else "")
    )
    coverage = (
        "This board carries <b>every promoted evaluation cell</b> (" + str(len(bakes)) + " bakes, "
        "including the full sota944 campaign grid) — any number cited in a report is findable "
        "here. The <b>default view shows the " + str(n_cur) + "-bake curated set</b> (era "
        "flagships + campaign arm candidates/leaders + ensembles); use the family toggles or "
        "<i>all</i> to reach every grid cell. " if n_cur else "")
    head = (
        "<h1>" + title + "</h1>"
        "<p class='sub'>" + stub_note + fair_note + coverage +
        "Toggle bakes below to compare them across every chart; click any table header to sort. "
        "The scatter matrix shows <b>predicted vs each reference</b> (MOS, JND, SSIMULACRA2, "
        "butteraugli, ColorVideoVDP) per corpus, with an OLS fit and canonical SROCC/PLCC. "
        "Charts are Apache ECharts " + ech_ver + " (inlined — still fully offline) with "
        "<b>semantic zoom</b>: wheel/drag and the sliders rescale the AXES and re-plot the data "
        "while marks, line widths and labels stay constant size (crowded labels de-overlap as "
        "you zoom in); double-click a chart to reset. "
        "All data, styles and scripts are inlined — this page opens offline. "
        "By <code>scripts/v_next/bandwise_dashboard.py --fulleval-dir</code>.</p>"
    )
    payload = json.dumps(data, separators=(",", ":"))
    css = (_CSS.replace("/*__LIGHT_VARS__*/", _css_vars("light"))
               .replace("/*__DARK_VARS__*/", _css_vars("dark")))
    html = (
        "<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>" + title + "</title>"
        "<style>" + css + "</style>"
        "<div class='viz-root'>" + head +
        "<div id='bar' class='bar'></div>"
        "<div id='panels'></div>"
        "<div id='tt' class='tt'></div>"
        "</div>"
        # vendored ECharts rides its OWN script tag ahead of the app script (the gates
        # extract + node --check every block; the render harness runs only the app one).
        "<script id='vendor-echarts'>\n" + ech_js + "\n</script>"
        "<script>\nconst DATA=" + payload + ";\n" + _JS + "\n</script>"
    )
    Path(out_path).write_text(html)
    return out_path, len(html)


# ------------------------------------------------------------------ client JS ----------------
_JS = r"""
'use strict';
const $=(s,r=document)=>r.querySelector(s);
const el=(t,a={},kids=[])=>{const e=document.createElementNS(t.startsWith('svg:')?'http://www.w3.org/2000/svg':'http://www.w3.org/1999/xhtml',t.replace('svg:',''));
  for(const k in a){if(k==='text')e.textContent=a[k];else if(k==='html')e.innerHTML=a[k];else e.setAttribute(k,a[k]);}
  (Array.isArray(kids)?kids:[kids]).forEach(c=>c&&e.appendChild(c));return e;};
const S=(t,a={},k=[])=>el('svg:'+t,a,k);

// Default-visible = the CURATED headline set (era flagships + campaign arm candidates/
// leaders + ensembles — flagged at build from gauntlet.py CURATED_BOARD). The full grid
// stays one toggle away; payloads without curated flags fall back to all-visible.
// Board-integrity pass (2026-08-04): DOMINATED cells (strict same-class Pareto,
// promote_fulleval.py --mark-dominated) are additionally default-OFF + dimmed; the
// annotations registry (benchmarks/eval_annotations.json) feeds the ⚠ badges.
const DOM=b=>!!(b.dominated_by&&b.dominated_by.length);
const ANN=(b,id)=>!!(b.annotations&&b.annotations.indexOf(id)>=0);
const annReason=id=>(DATA.annRegistry&&DATA.annRegistry[id]&&DATA.annRegistry[id].reason)||id;
const annKind=id=>(DATA.annRegistry&&DATA.annRegistry[id]&&DATA.annRegistry[id].kind)||'annotated';
// Scoreboard column -> the fulleval dot-path it displays. This is what lets a registry
// entry badge the SPECIFIC number it caveats: an entry whose `fields` cover a column's
// path renders ⚠ on that cell, so a new entry needs NO new JS (2026-08-06 — added with
// r1-gl2-cid22-k1-unreplicated, which had to reach the CID22 cell; before this the only
// generic surface was the chip-picker tooltip, easy to miss on the number itself).
const COL_FIELD={composite:'composite',cid22:'rank.cid22',nonphoto:'rank.nonphoto',
  konjnd:'rank.konjnd',aic3:'rank.aic3',live:'rank.live',csiq:'rank.csiq',
  hfnl:'rank.hfnlproxy.per_ref_mean',dial_mono:'dial.mono_pct',dial_tied:'dial.tied_pct',
  m3a:'m3a_coherence',m3:'m3_coherence',m3_mass:'m3_dropped_mass_pct',
  cid22_ci:'rank.cid22.srocc_ci',cid22_bwd:'rank.cid22.frac_negative'};
// Segment-boundary prefix match — the exact rule freeze_check's ann_covers uses:
// `rank.hfnlproxy` covers `rank.hfnlproxy.per_ref_mean`; `rank.hfnl` covers neither.
const annCovers=(entryField,colField)=>colField===entryField
  ||(colField.indexOf(entryField)===0&&colField.charAt(entryField.length)==='.');
function annForCol(b,colKey){
  const f=COL_FIELD[colKey];if(!f||!b.annotations||!b.annotations.length)return [];
  return b.annotations.filter(id=>{
    const m=DATA.annRegistry&&DATA.annRegistry[id];
    return m&&(m.fields||[]).some(ef=>annCovers(ef,f));});
}
// Corpus label ORIENTATION (2026-08-05) — from the EXPECTED_ORIENTATION registry in
// scripts/canonical_corpus/check_target_orientation.py (AST-read at build time; campaign
// Appendix I). A corpus declared "distortion" carries a JND-family label (q_jnd distance /
// PJND threshold): its signed SROCC is negative BY CONVENTION, |SROCC| is the magnitude
// reading, and a POSITIVE signed value there is an orientation MISMATCH — the defect and
// the convention are exact mirrors of the quality-oriented case. Fallback (payloads built
// before the registry rode along): konjnd only, the pre-registry behavior.
const ORIENT=DATA.orientation||{konjnd:"distortion"};
const JND_CORPORA=new Set(Object.keys(ORIENT).filter(c=>ORIENT[c]==="distortion"));
const JND_TIP=c=>c+": distortion-oriented JND-family label (EXPECTED_ORIENTATION registry, "
  +"campaign Appendix I) — signed SROCC is negative BY CONVENTION and is displayed as |SROCC|. "
  +"A positive signed value here would be an orientation MISMATCH, not a win.";
// sdr25 ⊂ aic4 (Appendix I structural finding; registry id sdr25-subset-of-aic4): the 50
// sdr25 stimuli are all contained in aic4's 300 — the two are not independent corpora.
const corpMark=c=>c+(JND_CORPORA.has(c)?" JND↓":"")+(c==="sdr25"?" ⊂aic4":"");
const corpTitle=(node,c)=>{const t=[];
  if(JND_CORPORA.has(c))t.push(JND_TIP(c));
  if(c==="sdr25")t.push("⊂ aic4: "+annReason("sdr25-subset-of-aic4"));
  if(t.length)node.setAttribute("title",t.join(" | "));
  return node;};
// ---- FAIRNESS LAYER (2026-09-04) ---------------------------------------------------
// Every value below is READ from the payload (which read it from a fulleval, the
// annotations registry, the G-ADDR floor registry or the ssim2-exam transcription).
// Nothing on this page computes a statistic.
const TIER=b=>((b.fair&&b.fair.tier)||'?');
const FAIRSET=DATA.bakes.filter(b=>TIER(b)!=='LEGACY').map(b=>b.name);
const VFAIRSET=DATA.bakes.filter(b=>TIER(b)==='VERIFIED-FAIR').map(b=>b.name);
const CRIT=Object.fromEntries((DATA.fairCriteria||[]).map(c=>[c[0],c[1]]));
const tierGlyph=b=>TIER(b)==='VERIFIED-FAIR'?'✔':(TIER(b)==='FAIR-NOTED'?'◐':'⚑');
const tierTitle=b=>{const f=b.fair||{};const L=['tier: '+TIER(b)];
  if(f.fails&&f.fails.length)L.push('FAILS: '+f.fails.map(k=>k+' — '+(CRIT[k]||'')).join(' | '));
  if(f.k===1)L.push('UNREPLICATED (k=1): this recipe has no seed sibling on the board, so its value is one draw and its spread is unknown.');
  if(f.k==null)L.push('UNGROUPABLE: no embedded repro argv, so no seed group can be formed.');
  if(f.notes&&f.notes.length)L.push('notes: '+f.notes.join(' | '));
  return L.join('\n');};
// Seed groups. Presentation is fixed by the user correction of 2026-09-04: mean +
// spread + k, per-seed values reachable, mean NEVER labelled definitive.
const SG=DATA.seedGroups||{};
const SGOF={};Object.keys(SG).forEach(g=>(SG[g].members||[]).forEach(n=>{SGOF[n]=g;}));
const sgOf=b=>SG[SGOF[b.name]]||null;
const sgStat=(b,axis)=>{const g=sgOf(b);return g&&g.stats?g.stats[axis]:null;};
const SEEDNOTE=DATA.seedCoverageNote||'';
const sgTitle=(b,axis)=>{const g=sgOf(b),st=sgStat(b,axis);if(!st)return '';
  return 'SEED GROUP k='+g.k+' on '+axis+'\nmean '+f3(st.mean)+'  spread '+f3(st.spread)
    +'  [min '+f3(st.min)+' .. max '+f3(st.max)+']'
    +'\nper seed: '+st.per.map(x=>x[0]+' '+f3(x[1])).join('   ')
    +'\nThe mean is the honest estimator against best-of-k, NOT "the true score": '+SEEDNOTE
    +(b.fair&&b.fair.coverage?('\nsample_coverage: '+JSON.stringify(b.fair.coverage)):'\nsample_coverage: NOT MEASURED');};
// Circularity + train==val, from the exam's own registered lists.
const CIRC=new Set(DATA.circularAxes||[]);
const TEV=new Set(DATA.trainEqValAxes||[]);
const HELDOUT=DATA.heldOutHumanAxes||[];
const EXAM=DATA.exam||null;
const examOf=n=>{if(!EXAM)return null;return (EXAM.candidates||[]).find(c=>c.board_name===n)||null;};
const PEERSSIM2=DATA.bakes.find(b=>b.name==='peer_ssim2')||null;
// G-ADDR — the registry's bars against what each cell already stores. Absent axes are
// NOT MEASURED with the reason, never a zero.
const GADDR=DATA.gaddrRegistry||null;
const gaddrTitle=b=>{if(!b.gaddr)return 'G-ADDR: NOT MEASURED (no registry / no dial block)';
  const L=['G-ADDR dial addressability — bars from benchmarks/dial_addressability_floor_2026-09-04.json'];
  const ar=(GADDR&&GADDR.activeRef)||'?';
  L.push('REGRESSION bars = the ACTIVE reference "'+ar+'" on the same instrument'
    +(ar==='peer_ssim2'?' — the reference METRIC, re-pinned by user decision 2026-09-04 ("ssim2 seems a better mentor"). A candidate must address AT LEAST the range ssim2 addresses.':''));
  if(GADDR&&GADDR.gridIncumbent)L.push('the shipped-B column is CONTEXT, never a bar — the registry labels it BIASED (A1/A3/A6 sit ABOVE the reference metric\u2019s own values on this grid).');
  const EM=b.gaddr.src==='emitted';
  if(EM){L.push('this cell carries the OWNER’s own G-ADDR verdict (bake_verdict --gaddr-json, grafted '
    +'sha-gated + same-grid gated) — all '+b.gaddr.total+' rows below are copied from it, none re-derived.');
    L.push('VERDICT: '+(b.gaddr.headline||'?'));
    L.push('  regression: '+(b.gaddr.regression||'?')+'   contract: '+(b.gaddr.contract||'?'));
    if(b.gaddr.grid)L.push('  instrument: '+b.gaddr.grid);}
  let tier='';
  (b.gaddr.cells||[]).forEach(c=>{
    const t=c[7]||(c[0].charAt(0)==='A'?'regression':'contract');
    if(t!==tier){tier=t;L.push(' '+tier.toUpperCase()+' tier');}
    L.push('  '+c[0]+' '+c[1]+': '+(c[2]==null?'NOT MEASURED':f3(c[2]))
      +(c[3]==null?'':('  bar('+(t==='contract'?'absolute':ar)+') '
        +(c[4]==='ge'?'≥':c[4]==='le'?'≤':c[4]==='gt'?'>':'<')+' '+f3(c[3])))
      +(c[6]==null?'':('   [shipped B '+f3(c[6])+']'))
      +'  '+(c[5]==null?'— NOT MEASURED':(c[5]?'PASS':'FAIL'))
      +((c[5]==null&&c[9])?('  ('+c[9]+')'):''));});
  if(EM){if((b.gaddr.cfail||[]).length)L.push('NOT SHIPPABLE — CONTRACT rows FAILED: '+b.gaddr.cfail.join(', ')
      +'  (user rule 2026-09-04: any model that limits dial range cannot ship)');
    if((b.gaddr.cnm||[]).length)L.push('contract rows NOT MEASURED (no in-era probe): '+b.gaddr.cnm.join(', ')
      +'  — an unmeasured row is never a pass, so this cell is not certified shippable either.');}
  else L.push('NOT MEASURED here: '+(DATA.gaddrNotMeasured||''));
  return L.join('\n');};
// NOT-SHIPPABLE badge: drawn ONLY off a measured CONTRACT-tier FAIL in the owner's
// own emitted verdict. An INCOMPLETE contract (a row nobody could measure) never
// draws it — unmeasured is not a fail.
const notShip=b=>!!(b.gaddr&&b.gaddr.src==='emitted'&&(b.gaddr.cfail||[]).length);
const shipBadge=b=>notShip(b)?el('span',{style:'font-size:9px;font-weight:700;letter-spacing:.03em;'
  +'padding:0 4px;margin-left:5px;border-radius:7px;vertical-align:1px;white-space:nowrap;'
  +'background:color-mix(in srgb, var(--critical) 30%, var(--surface-1));border:1px solid var(--critical)',
  title:'NOT SHIPPABLE — the G-ADDR CONTRACT tier FAILS on '+(b.gaddr.cfail||[]).join(', ')
    +'. '+(b.gaddr.headline||'')+'  USER RULE 2026-09-04: "floor and ceiling dial addressability is '
    +'crucial — any model that limits dial range cannot ship."',text:'NOT SHIPPABLE'}):null;
const KNOBFAIL=b=>!!(b.knob_end_fail&&b.knob_end_fail.length);
const CURATED_ALL=DATA.bakes.filter(b=>b.curated&&!DOM(b)).map(b=>b.name);
// default compare set excludes knob-end failers (dial cannot reach/span the
// top zone — G-GRAN semantics); they stay toggleable + in 'curated+knobfail'.
const CURATED=DATA.bakes.filter(b=>b.curated&&!DOM(b)&&!KNOBFAIL(b)).map(b=>b.name);
// DEFAULT VIEW = VERIFIED-FAIR (user request 2026-09-04). Curated ∩ fair when that
// is non-empty, else the whole fair set — never an empty board.
const _DEFVIS=(()=>{const f=new Set(FAIRSET);
  const c=CURATED.filter(n=>f.has(n));
  return c.length?c:(FAIRSET.length?FAIRSET:DATA.bakes.map(b=>b.name));})();
const state={shapeNorm:true,visible:new Set(_DEFVIS),
  ref:null, sortKey:'composite', sortDir:-1, mcorp:null, chipsOpen:false, gateFilter:new Set()};
function effTheme(){const dt=document.documentElement.getAttribute('data-theme');
  return dt||((window.matchMedia&&matchMedia('(prefers-color-scheme:dark)').matches)?'dark':'light');}
const pal=()=>DATA.palette[effTheme()==='dark'?'dark':'light'];
const color=b=>pal()[b.colorIndex%8];
const cssv=n=>getComputedStyle($('.viz-root')).getPropertyValue(n).trim()||'#888';
const visBakes=()=>DATA.bakes.filter(b=>state.visible.has(b.name));

// pick default reference = first one that any bake carries
function initRef(){const have=new Set();DATA.bakes.forEach(b=>Object.values(b.scatter).forEach(c=>Object.keys(c).forEach(r=>have.add(r))));
  state.ref=DATA.references.find(r=>have.has(r))||'mos';}

// ---- number helpers
const f3=v=>v==null||!isFinite(v)?'—':(+v).toFixed(3);
const f2=v=>v==null||!isFinite(v)?'—':(+v).toFixed(2);
const pct=v=>v==null||!isFinite(v)?'—':(v*100).toFixed(1)+'%';

// ---- ENSEMBLE marker. An equal-weight ensemble of k bakes is an evaluation
// FUNCTION, not a shippable artifact: there is no single ZNPR, so M3/M3a are
// not computable (they render as an em-dash = NOT MEASURED, never a low score)
// and the model-details card describes the ANCHOR member only. Flag set by
// scripts/promote_fulleval.py --members (model.kind / model.members).
const isEns=b=>!!(b.model&&b.model.kind==='ensemble');
const ensK=b=>(b.model&&b.model.members)||null;
const ensTag=b=>isEns(b)?' ens×'+ensK(b):'';
const ensBadge=b=>isEns(b)?el('span',{style:'font-size:9px;font-weight:700;letter-spacing:.03em;'
  +'padding:0 4px;margin-left:5px;border-radius:7px;vertical-align:1px;white-space:nowrap;'
  +'background:color-mix(in srgb, var(--warn) 34%, var(--surface-1));border:1px solid var(--border)',
  title:'equal-weight ensemble of '+ensK(b)+' bakes — an evaluation function, not a single '
    +'shippable bake; M3/M3a not computable',text:'ens×'+ensK(b)}):null;
// DOMINATED marker (board-integrity pass 2026-08-04): strictly beaten by a same-class
// sibling on every measured floor axis + composite. Cells stay on the board (never
// deleted) but render dimmed, default-off, behind the 'dominated' chip.
const domBadge=b=>DOM(b)?el('span',{style:'font-size:9px;font-weight:700;letter-spacing:.03em;'
  +'padding:0 4px;margin-left:5px;border-radius:7px;vertical-align:1px;white-space:nowrap;'
  +'opacity:.75;background:color-mix(in srgb, var(--muted) 22%, var(--surface-1));border:1px solid var(--border)',
  title:'DOMINATED (strict-pareto-2026-08-04) by: '+b.dominated_by.join(', ')
    +' — beaten on every measured floor axis + composite within its class; kept for the record, default-off',
  text:'dom'}):null;
// swatch + name (+ ens/dom badges) cell content, shared by every table that names a bake
// ---- gates (overlay benchmarks/board_gates_2026-08-28.json + build-computed knob-end).
// Verdicts: 'pass' | 'fail' | null (NOT MEASURED — never treated as fail).
const GATE_DEFS=[
  ['gout','G','G-OUT v2 outlier gate (owner outlier_gate.py)'],
  ['elig','E','frozen two-zone eligibility battery (HDR bakes: HDR-lane freeze battery)'],
  ['dialv2','D','G-GRAN v2 peer-anchored dial gate (REGISTERED W12 candidate, not yet frozen)'],
  ['knob','K','knob-end check (G-GRAN v1 semantics: HF-zone reach/span; computed at build)']];
function gateV(b,g){
  if(g==='knob')return (b.knob_end_fail===undefined)?null:(b.knob_end_fail.length?'fail':'pass');
  const e=(b.gatecheck||{})[g];return e?e.v:null;}
function gateWhy(b,g){
  if(g==='knob')return b.knob_end_fail&&b.knob_end_fail.length?('fails: '+b.knob_end_fail.join(', ')):'';
  const e=(b.gatecheck||{})[g];return e?((e.why?e.why+' ':'')+(e.src?'['+e.src+']':'')):'';}
function gateGlyphs(b){return GATE_DEFS.map(([g])=>{const v=gateV(b,g);
  return v==='pass'?'✓':(v==='fail'?'✗':'·');}).join('');}
function gateTitle(b){return GATE_DEFS.map(([g,l,d])=>{const v=gateV(b,g);
  return l+' = '+d+': '+(v?v.toUpperCase():'not measured')+(gateWhy(b,g)?' — '+gateWhy(b,g):'');}).join('\n');}
function gateExcluded(b){let x=false;state.gateFilter.forEach(g=>{if(gateV(b,g)==='fail')x=true;});return x;}
function nameInto(node,b,suffix){
  node.append(el('span',{class:'sw',style:'display:inline-block;margin-right:5px;background:'+color(b)}));
  node.append(document.createTextNode(b.name+(suffix||'')));
  if(b.codename)node.append(el('span',{text:' \u201c'+b.codename+'\u201d',
    style:'margin-left:4px;color:var(--seq-hi);font-weight:600;font-size:11px',
    title:'codename (benchmarks/candidate_names.json)'}));
  if(b.star)node.append(el('span',{text:' \u{1F31F}',style:'margin-left:2px;cursor:help',
    title:'\u{1F31F} '+b.star}));
  const bd=ensBadge(b);if(bd)node.append(bd);
  const dd=domBadge(b);if(dd)node.append(dd);
  const nb=shipBadge(b);if(nb)node.append(nb);
  return node;
}

// ---- tooltip
const tt=$('#tt');
function showTip(html,ev){tt.innerHTML=html;tt.style.opacity=1;
  let x=ev.clientX+12,y=ev.clientY+12;if(x>innerWidth-160)x=ev.clientX-tt.offsetWidth-12;
  tt.style.left=x+'px';tt.style.top=y+'px';}
function hideTip(){tt.style.opacity=0;}

// ---- ECharts mount layer (2026-08-04). The five heavyweight panels (scatter matrix,
// dial curves, 10-band bars, cross-corpus heatmap, trade maps) render through Apache
// ECharts (vendored + inlined, canvas renderer) for SEMANTIC zoom: dataZoom rescales
// the AXES and re-plots into the new domain while symbol sizes, stroke widths and label
// fonts stay constant — the predecessor viewBox zoom scaled the whole picture, so
// overlapping labels stayed overlapping at every zoom level (2026-08-04 user report).
// Shim/canvas-less safety: every mount ALWAYS builds its option (pure data — stashed on
// host._chartOption for the render harness) and only calls echarts.init when a real
// canvas 2d context exists; charts lazy-init on first viewport intersection so a
// ~160-cell board doesn't pay for offscreen canvases. Double-click = reset (restore).
const CANVAS_OK=(()=>{try{const c=document.createElement&&document.createElement('canvas');
  return !!(c&&typeof c.getContext==='function'&&c.getContext('2d'));}catch(e){return false;}})();
const HAS_ECH=typeof echarts!=='undefined'&&echarts&&typeof echarts.init==='function';
let CHARTS=[],IOS=[];
function disposeCharts(){CHARTS.forEach(c=>{try{c.dispose();}catch(e){}});CHARTS=[];
  IOS.forEach(o=>{try{o.disconnect();}catch(e){}});IOS=[];}
function TH(){return DATA.chartThemes[effTheme()==='dark'?'dark':'light'];}
function axStyle(name){const t=TH();return{
  type:'value',scale:true,name:name||'',nameLocation:'middle',nameGap:24,
  nameTextStyle:{color:t['text-secondary'],fontSize:10},
  axisLine:{lineStyle:{color:t.axis}},axisTick:{lineStyle:{color:t.axis}},
  axisLabel:{color:t.muted,fontSize:9},
  splitLine:{show:true,lineStyle:{color:t.grid,width:0.6}}};}
function ttStyle(){const t=TH();return{backgroundColor:t['surface-1'],borderColor:t.muted,
  borderWidth:1,padding:[4,7],confine:true,
  textStyle:{color:t['text-primary'],fontSize:11},
  extraCssText:'box-shadow:0 2px 8px rgba(0,0,0,.18)'};}
function dzSlider(extra){const t=TH();return Object.assign({type:'slider',height:12,bottom:4,
  borderColor:t.axis,fillerColor:'rgba(128,128,128,.18)',handleSize:'90%',
  dataBackground:{lineStyle:{color:t.axis,width:.5},areaStyle:{color:t.grid,opacity:.4}},
  selectedDataBackground:{lineStyle:{color:t.muted,width:.5},areaStyle:{color:t.grid}},
  moveHandleSize:0,brushSelect:false,textStyle:{color:t.muted,fontSize:8}},extra||{});}
function mountChart(kind,w,h,option){
  const host=el('div',{class:'echart','data-kind':kind,
    style:'width:'+w+'px;height:'+h+'px;max-width:100%;background:var(--surface-1);'
      +'border:1px solid var(--border);border-radius:6px'});
  host._chartOption=option;                       // pure data — the render harness checks it
  if(CANVAS_OK&&HAS_ECH){
    const init=()=>{if(host._chart)return;
      const c=echarts.init(host,null,{renderer:'canvas',width:w,height:h});
      c.setOption(option);
      if(typeof host.addEventListener==='function')
        host.addEventListener('dblclick',()=>{try{c.dispatchAction({type:'restore'});}catch(e){}});
      host._chart=c;CHARTS.push(c);};
    if(typeof IntersectionObserver==='function'){
      const io=new IntersectionObserver(es=>{es.forEach(x=>{if(x.isIntersecting){init();io.disconnect();}});},
        {rootMargin:'250px'});
      io.observe(host);IOS.push(io);
    }else init();
  }
  return host;
}

// ---- CONTROL BAR: preset buttons + FAMILY toggles + collapsible per-bake chips +
// reference tabs + theme. With ~160 grid cells on the board, the sticky bar leads with
// the curated preset and family groups; individual chips live in a collapsible picker
// (scoreboard rows also toggle visibility on click).
function renderBar(){
  const bar=$('#bar');bar.innerHTML='';
  const mk=(t,fn,title)=>{const x=el('button',{class:'btn',text:t});if(title)x.setAttribute('title',title);x.onclick=fn;return x;};
  bar.append(
    mk('VERIFIED-FAIR',()=>{state.visible=new Set(VFAIRSET);rerender();renderBar();},
       'only rows that pass EVERY fairness criterion AND are a replicated seed group (k>=2). '+VFAIRSET.length+' rows.'),
    mk('fair (incl. unreplicated)',()=>{state.visible=new Set(FAIRSET);rerender();renderBar();},
       'VERIFIED-FAIR + FAIR-NOTED (fair, but k=1 UNREPLICATED or ungroupable). '+FAIRSET.length+' rows.'),
    mk('legacy / unverified',()=>{state.visible=new Set(DATA.bakes.filter(b=>TIER(b)==='LEGACY').map(b=>b.name));rerender();renderBar();},
       'ONLY the rows that fail a fairness criterion — each badged with which one. Nothing is deleted; these rows keep every stat.'),
    mk('curated',()=>{state.visible=new Set(CURATED.length?CURATED:DATA.bakes.map(b=>b.name));rerender();renderBar();},
    mk('curated+knobfail',()=>{state.visible=new Set(CURATED_ALL);rerender();renderBar();},
      'curated including knob-end failers (dial cannot reach/span the top zone)'),
      'the default set: era flagships + campaign arm candidates/leaders + ensembles'),
    mk('sprint bests',()=>{state.visible=new Set((DATA.sprintBest||[]).map(x=>x.n));rerender();renderBar();},
      'one selected leader per sprint/era: '+((DATA.sprintBest||[]).map(x=>x.s+' \u2192 '+x.n).join(' \u00b7 ')||'none resolved')),
    mk('all',()=>{DATA.bakes.forEach(b=>state.visible.add(b.name));rerender();renderBar();}),
    mk('none',()=>{state.visible.clear();rerender();renderBar();}),
    mk('top 6',()=>{const cmp=(a,b)=>{const x=a.composite,y=b.composite;
      if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return -1;return y-x;};
      const s=[...DATA.bakes].sort(cmp).slice(0,6).map(b=>b.name);
      state.visible=new Set(s);rerender();renderBar();}));
  // gate pre-filter (user directive 2026-08-28): exclude gate-FAILING rows from the
  // scoreboard list itself (and drop them from the visible chart set). Not-measured
  // gates never exclude. 'usable' = the frozen-gate trio G+E+K (dial-v2 stays opt-in
  // while it is a registered-not-adopted W12 candidate).
  // discussion-set dropdown (user directive 2026-08-28): pick a board
  // generation's discussion set -> visible = set UNION incumbents UNION peers.
  const ds=DATA.discussionSets||[];
  if(ds.length){
    const sel=el('select',{class:'btn',title:'filter to a discussion set + incumbents + iqa peers (benchmarks/board_discussion_sets.json, latest first)'});
    sel.append(el('option',{text:'discussion set\u2026',value:''}));
    ds.forEach((d,i)=>sel.append(el('option',{text:d.label,value:String(i)})));
    sel.onchange=()=>{if(sel.value==='')return;const d=ds[+sel.value];
      const peers=DATA.bakes.filter(b=>b.name.startsWith('peer_')).map(b=>b.name);
      const want=new Set([...(d.bakes||[]),...(DATA.incumbents||[]),...peers]);
      state.visible=new Set(DATA.bakes.filter(b=>want.has(b.name)).map(b=>b.name));
      rerender();renderBar();};
    bar.append(sel);
  }
  bar.append(el('span',{text:'gate filter:',style:'margin-left:.6rem;color:var(--text-secondary);font-size:11px'}));
  const applyGF=()=>{if(state.gateFilter.size)DATA.bakes.forEach(b=>{if(gateExcluded(b))state.visible.delete(b.name);});
    rerender();renderBar();};
  bar.append(mk('usable',()=>{state.gateFilter=new Set(['gout','elig','knob']);applyGF();},
    'exclude rows with a MEASURED fail on G-OUT, eligibility, or knob-end (dial-v2 opt-in via its chip)'));
  GATE_DEFS.forEach(([g,l,d])=>{
    const on=state.gateFilter.has(g);
    const ch=el('span',{class:'gchip'+(on?'':' off'),text:l,
      title:(on?'ON — excluding measured fails of: ':'off — click to exclude measured fails of: ')+d});
    ch.onclick=()=>{on?state.gateFilter.delete(g):state.gateFilter.add(g);applyGF();};
    bar.append(ch);});
  if(state.gateFilter.size)bar.append(mk('clear gates',()=>{state.gateFilter.clear();rerender();renderBar();},
    'remove the gate filter (rows return to the list; visibility unchanged)'));
  // family toggles: click = show the whole family (or hide it when fully shown)
  const fams=[];DATA.bakes.forEach(b=>{if(b.family&&fams.indexOf(b.family)<0)fams.push(b.family);});
  fams.forEach(f=>{
    const members=DATA.bakes.filter(b=>b.family===f);
    const on=members.filter(b=>state.visible.has(b.name)).length;
    const full=on===members.length;
    const chip=el('span',{class:'gchip'+(on?'':' off'),
      title:(full?'hide':'show')+' all '+members.length+' “'+f+'” bakes'});
    chip.append(el('b',{text:f}),el('span',{class:'cap',text:' '+on+'/'+members.length}));
    chip.onclick=()=>{members.forEach(b=>full?state.visible.delete(b.name):state.visible.add(b.name));
      rerender();renderBar();};
    bar.appendChild(chip);
  });
  // ---- board-integrity chips (2026-08-04): dominated + block-usage filters.
  const domCells=DATA.bakes.filter(DOM);
  if(domCells.length){
    const on=domCells.filter(b=>state.visible.has(b.name)).length,full=on===domCells.length;
    const chip=el('span',{class:'gchip'+(on?'':' off'),
      title:(full?'hide':'show')+' the '+domCells.length+' DOMINATED cells — strictly beaten by a '
        +'same-class sibling on every measured floor axis + composite (strict-pareto-2026-08-04; '
        +'files kept, marks via promote_fulleval.py --mark-dominated). Default-off so trimmed '
        +'cells never sit on stolen wins.'});
    chip.append(el('b',{text:'dominated'}),el('span',{class:'cap',text:' '+on+'/'+domCells.length}));
    chip.onclick=()=>{domCells.forEach(b=>full?state.visible.delete(b.name):state.visible.add(b.name));
      rerender();renderBar();};
    bar.appendChild(chip);
  }
  const blkCells=DATA.bakes.filter(b=>b.blocks&&b.blocks.uses156);
  if(blkCells.length){
    const on=blkCells.filter(b=>state.visible.has(b.name)).length,full=on===blkCells.length;
    const chip=el('span',{class:'gchip'+(on?'':' off'),
      title:(full?'hide':'show')+' the '+blkCells.length+' bakes with STRUCTURAL use of f156-371 — '
        +'the block ZEROED by the folded 924/944 regimes (slots preserved per the append-only '
        +'discipline, not removed). Mostly separates eras: 944-class bakes are structurally zero '
        +'there by construction; 372-wide era bakes carry real IW-pool weight; ADD156 bakes do not '
        +'even carry the slots. Static fingerprint from bake bytes (bake_block_profile) — the '
        +'corpus-based contribution measure is the sibling instrument.'});
    chip.append(el('b',{text:'uses f156-371'}),el('span',{class:'cap',text:' '+on+'/'+blkCells.length}));
    chip.onclick=()=>{blkCells.forEach(b=>full?state.visible.delete(b.name):state.visible.add(b.name));
      rerender();renderBar();};
    bar.appendChild(chip);
  }
  // per-bake chips (collapsible picker; open-state survives re-renders)
  const det=el('details',{class:'allbakes'});
  if(state.chipsOpen)det.setAttribute('open','');
  det.addEventListener('toggle',()=>{state.chipsOpen=!!det.open;});
  det.append(el('summary',{class:'cap',
    text:'pick individual bakes — '+state.visible.size+' of '+DATA.bakes.length+' visible (scoreboard rows toggle too)'}));
  const wrap=el('div',{class:'chipwrap'});
  DATA.bakes.forEach(b=>{
    const on=state.visible.has(b.name);
    const chip=el('label',{class:'chip'+(on?'':' off'),
      title:b.regime+' inputs'
        +(b.regime_flag&&b.regime_flag!==b.regime?' (recorded flag: '+b.regime_flag+')':'')
        +(b.family?' · '+b.family:'')+(b.curated?' · curated':'')
        +(b.is_stub?' (stub)':'')+(isEns(b)?' · ensemble of '+ensK(b)+' bakes':'')
        +(DOM(b)?' · DOMINATED by '+b.dominated_by.join(', '):'')
        +(b.annotations&&b.annotations.length?' · ⚠ '+b.annotations.join(', '):'')});
    const cb=el('input',{type:'checkbox'});cb.checked=on;
    cb.onchange=()=>{on?state.visible.delete(b.name):state.visible.add(b.name);rerender();renderBar();};
    chip.append(cb, el('span',{class:'sw',style:'background:'+color(b)}),
      el('span',{text:b.name}), el('span',{class:'cap',text:b.regime+ensTag(b)}));
    wrap.appendChild(chip);
  });
  det.append(wrap);
  bar.appendChild(det);
  // reference tabs
  const tabs=el('span',{class:'tabs'});
  tabs.append(el('span',{class:'cap',text:'reference:',style:'align-self:center'}));
  DATA.references.forEach(r=>{
    const has=DATA.bakes.some(b=>Object.values(b.scatter).some(c=>r in c));
    if(!has)return;
    const x=el('button',{class:'btn'+(state.ref===r?' active':''),text:DATA.refLabels[r]||r});
    x.onclick=()=>{state.ref=r;renderBar();renderScatter();};
    tabs.appendChild(x);
  });
  bar.appendChild(tabs);
  const th=el('button',{class:'btn',text:'◐ theme'});
  th.onclick=()=>{const cur=document.documentElement.getAttribute('data-theme');
    document.documentElement.setAttribute('data-theme',cur==='dark'?'light':(cur==='light'?'dark':'light'));
    renderBar();rerender();};
  bar.appendChild(th);
}

// ---- generic table sorting (2026-08-04). The scoreboard sorts through state + a full
// re-render (mountTable); every OTHER stat table (Mohammadi, per-band, gates, loop) gets
// THIS: click a header, rows re-order in place by that column's displayed value —
// numeric when the column parses numeric (first click = best/descending), string
// otherwise (first click = ascending); an em-dash (NOT MEASURED) always sinks to the
// bottom. Values are read from the rendered cells, so nothing is recomputed. Shim-safe:
// rows re-attach via innerHTML='' + appendChild (a bare appendChild move would duplicate
// entries in the render-harness DOM shim), and only Array.from/getAttribute are used.
const deepText=n=>n==null?'':(n.nodeType===3?String(n.textContent||''):
  (n.textContent?String(n.textContent):Array.from(n.childNodes||[]).map(deepText).join('')));
function cellNum(t){const m=String(t).replace(/[,%±()]/g,' ').match(/-?\d+(?:\.\d+)?(?:[eE]-?\d+)?/);
  return m?parseFloat(m[0]):null;}
function makeSortable(tbl){
  const thead=Array.from(tbl.children||[])[0],tb=Array.from(tbl.children||[])[1];
  const hrow=thead&&Array.from(thead.children||[])[0];
  if(!hrow||!tb)return tbl;
  const ths=Array.from(hrow.children||[]);
  ths.forEach((th,ci)=>{
    th.onclick=()=>{
      const rows=Array.from(tb.children||[]);
      const vals=rows.map(r=>{const td=Array.from(r.children||[])[ci];const t=deepText(td).trim();
        return {r,t,n:(t===''||t==='—')?null:cellNum(t)};});
      const seen=vals.filter(v=>v.t!==''&&v.t!=='—');
      const numeric=seen.length>0&&seen.every(v=>v.n!=null);
      const dir=(tbl._sk===ci)?-(tbl._sd||1):(numeric?-1:1);
      tbl._sk=ci;tbl._sd=dir;
      vals.sort((a,b)=>{
        if(numeric){const an=a.n==null,bn=b.n==null;
          if(an&&bn)return 0;if(an)return 1;if(bn)return -1;return dir*(a.n-b.n);}
        return dir*String(a.t).localeCompare(String(b.t));});
      tb.innerHTML='';vals.forEach(v=>tb.appendChild(v.r));
      ths.forEach((h,k)=>{const c=String((h.getAttribute&&h.getAttribute('class'))||'')
        .replace(/\s*\b(sorted|asc)\b/g,'');
        h.setAttribute('class',c+(k===ci?' sorted'+(dir>0?' asc':''):''));});
    };
  });
  return tbl;
}

// ---- SCOREBOARD TABLE (sortable). columns = derived metrics per bake.
const COLS=[
  ['name','bake',true,b=>b.name],
  // FAIRNESS (2026-09-04). `fair` = the tier glyph; `k` = seed-group size; `cmean` =
  // the group's MEAN composite with its spread — the honest estimator against
  // best-of-k, never labelled definitive (per-seed values on hover).
  ['fair','fair',true,b=>tierGlyph(b)],
  ['k','k',false,b=>{const f=b.fair||{};return f.k==null?null:f.k;}],
  ['cmean','composite k-mean',false,b=>{const st=sgStat(b,'composite');return st?st.mean:null;}],
  ['cspread','k-spread',false,b=>{const st=sgStat(b,'composite');return st?st.spread:null;}],
  ['regime','regime',true,b=>b.regime],
  ['trained','trained',true,b=>b.train_date?b.train_date.d+(b.train_date.src==='file'?'*':''):null],
  ['gates','gates',true,b=>gateGlyphs(b)],
  ['composite','composite',false,b=>b.composite],
  ['cid22','CID22',false,b=>rs(b,'cid22')],
  ['nonphoto','nonphoto',false,b=>rs(b,'nonphoto')],
  ['konjnd','KonJND',false,b=>rs(b,'konjnd')],
  ['aic3','AIC-3',false,b=>rs(b,'aic3')],
  ['aic4','AIC-4',false,b=>rs(b,'aic4')],
  ['live','LIVE',false,b=>rs(b,'live')],
  ['csiq','CSIQ',false,b=>rs(b,'csiq')],
  ['hfnl','HF-NL/ref',false,b=>b.rank.hfnlproxy&&b.rank.hfnlproxy.per_ref_mean!=null?b.rank.hfnlproxy.per_ref_mean:null],
  ['dial_mono','dial-mono',false,b=>b.dial.mono_pct],
  ['dial_tied','tied',false,b=>b.dial.tied_pct],
  ['m3a','M3a-attr',false,b=>b.m3a],
  ['m3','M3-coh',false,b=>b.m3],
  ['m3_mass','M3 drop%',false,b=>b.m3_dropped_mass],
  ['corr','corr-passq20',false,b=>b.corruption&&b.corruption.pass_q20!=null?b.corruption.pass_q20:null],
  ['cid22_ci','CID22 95%CI±',false,b=>{const r=b.rank.cid22;return r&&r.srocc_ci?(r.srocc_ci[1]-r.srocc_ci[0])/2:null;}],
  ['cid22_bwd','CID22 %bwd',false,b=>{const r=b.rank.cid22;return r&&r.frac_negative!=null?r.frac_negative:null;}],
  // G-ADDR (2026-09-04): pass/fail count over the six axes every cell already stores,
  // against the committed floor registry. The other nine axes are NOT MEASURED for
  // these cells (the gate landed after their verdicts) and are never counted as zeros.
  ['gaddr','G-ADDR p/f',false,b=>b.gaddr?b.gaddr.pass:null],
  // ssim2 EXAM (transcribed verdicts, benchmarks/ssim2_exam_scorecard_2026-08-31.json)
  ['w1','W1 no-reg',true,b=>{const e=examOf(b.name);return e?e.W1.v:null;}],
  ['w2','W2 real-win',true,b=>{const e=examOf(b.name);return e?e.W2.v:null;}],
  ['w3','W3 ladder',true,b=>{const e=examOf(b.name);return e?e.W3.v:null;}],
  ['w4','W4 speed',true,b=>{const e=examOf(b.name);return e?e.W4.v:null;}],
  ['w5','W5 HDR',true,b=>{const e=examOf(b.name);return e?e.W5.v:null;}],
  ['w6','W6 non-circ',true,b=>{const e=examOf(b.name);return e?e.W6.v:null;}],
  ['w7','W7 reachable',true,b=>{const e=examOf(b.name);return e?e.W7.v:null;}],
];
// Scoreboard columns that are NOT ranking signal, and why. Rendered dimmed with the
// reason on hover so they stay visible as integrity guards / sanity rows without ever
// being read as a win (exam §2.1 + registry kadid-tid-train-eq-val +
// hfnl-ssim2-self-target-circular-2026-09-01).
const NONRANK={
  kadid:'train==val (100% pair overlap) — integrity guard, never ranking signal',
  tid:'train==val, and retired to train-only by user ruling 2026-08-29 — historical guard',
  nonphoto:'ssim2-ANCHORED: its target IS an ssim2 score, so this is AGREEMENT with ssim2, never a win over it',
  imazen26:'ssim2-ANCHORED: agreement with ssim2, not a win over it',
  hfnl:'hfnlproxy is ssim2-ANCHORED; hf_nearlossless human_score IS ssim2_gpu/100 exactly on 1200/1200 rows'};
// SIGNED SROCC accessor (2026-08-04; registry-driven 2026-08-05). JND↓ corpora
// (JND_CORPORA, from the EXPECTED_ORIENTATION registry — aic4/konjnd/sdr25) carry
// distortion-oriented labels: their signed SROCC is negative BY CONVENTION, so the
// magnitude reading is |SROCC|. Every quality-oriented corpus keeps its sign — a
// negative there is a genuine ranking INVERSION and must never render as a high
// score. This became load-bearing when the ext-lineage KADID eval target was found
// stored inverted — 110 of 188 board bakes were anti-correlated with KADID's real
// human MOS while the board drew all 188 as positive magnitudes
// (benchmarks/sota944_campaign_2026-08-03.md REGISTERED APPENDIX F).
const SIGN_ABS_CORPORA=JND_CORPORA;
const rawSigned=r=>r?(r.srocc_signed!=null?r.srocc_signed:(r.srocc!=null?r.srocc:null)):null;
const sgn=(c,r)=>{if(!r)return null;
  if(SIGN_ABS_CORPORA.has(c)){const v=rawSigned(r);return v!=null?Math.abs(v):null;}
  return rawSigned(r);};
const rs=(b,c)=>sgn(c,b.rank[c]);
// ---- JXL loop-targeting join (2/3-shot). LT is the jxl-encoder sweep summary (READ, not
// re-derived). Scoreboard shows the mapped bake's emit-best cells; full detail (emit-last,
// outer arms, ssim2) lives in the JXL loop targeting section.
const LT=DATA.loopTargeting||null;
const ltN=()=>(LT&&LT.meta&&LT.meta.matrix&&LT.meta.matrix.n_cells)||27;
const ltCell=(b,mode)=>{if(!LT)return null;const mk=LT.bakeMap[b.name];if(!mk)return null;
  const m=LT.models[mk];return (m&&m.cells&&m.cells[mode])||null;};
const PX=DATA.loopProxy||null;
const pxc=(b,codec,kk,key)=>{if(!PX)return null;const r=PX[b.name];if(!r||!r[codec])return null;
  const c=r[codec].cells&&r[codec].cells[kk];return c?c[key]:null;};
if(PX){COLS.push(
  ['proxy_jxl_s','jxl-k3 ssim2⌁ (scalar)',false,b=>pxc(b,'jxl','k3','ssim2_fwd_med')],
  ['proxy_jxl_n','jxl-k3 nat⌁ (scalar)',false,b=>pxc(b,'jxl','k3','native_med')],
  ['proxy_avif_s','avif-k3 ssim2⌁ (scalar)',false,b=>pxc(b,'avif','k3','ssim2_fwd_med')],
  ['proxy_avif_n','avif-k3 nat⌁ (scalar)',false,b=>pxc(b,'avif','k3','native_med')]);}
if(LT){COLS.push(
  ['loop2','2shot ±2',false,b=>{const c=ltCell(b,'k2_emit_best');return c?c.within2:null;}],
  ['loop3','3shot ±2',false,b=>{const c=ltCell(b,'k3_emit_best');return c?c.within2:null;}],
  ['loop3err','3shot med|err|',false,b=>{const c=ltCell(b,'k3_emit_best');return c!=null&&c.med_abs_err!=null?c.med_abs_err:null;}]);}
function fmtCell(key,v,b){
  if(key==='name'||key==='regime'||key==='fair')return v;
  if(key==='k')return v==null?'—':(v===1?'1 ⚠':String(v));
  if(key==='cspread')return v==null?'—':f3(v);
  if(key&&key.charAt(0)==='w'&&key.length===2)return v==null?'—':v;
  if(key==='gaddr')return v==null?'—':v+'/'+((b&&b.gaddr&&b.gaddr.total)||6);
  if(key==='trained')return v==null?'—':v;
  if(key==='gates')return v;
  if(key==='cid22_bwd')return v==null?'—':pct(v);
  if(key==='dial_tied')return pct(v);
  if(key==='corr'||key==='dial_mono')return pct(v);
  if(key==='m3_mass')return v==null?'—':v.toFixed(1)+'%';
  if(key==='cid22_ci')return v==null?'—':'±'+v.toFixed(3);
  if(key==='loop2'||key==='loop3')return v==null?'—':v+'/'+ltN();
  if(key==='loop3err')return v==null?'—':(+v).toFixed(2);
  if(key&&key.startsWith('proxy_'))return v==null?'—':(+v).toFixed(2);
  return f3(v);
}
function renderTable(){
  const wrap=el('div',{});
  const domNote=DATA.nDominatedExcluded?('<b>'+DATA.nDominatedExcluded+' dominated bakes FULLY EXCLUDED</b> '
    +'(strict-pareto-2026-08-04; files + fullevals retained — registry/dominance TSV). '):'';
  const pxNote=DATA.loopProxy?('⌁ SCALAR-STEERING proxy (seeded-secant on the stored dial grid, census-validated; DIFFMAP allocation value is structurally invisible to any on-ladder simulation — measured separately by the h3own paired-encode A/B): '
    +'the <b>ssim2-judged</b> columns are the FAIR cross-bake reading; native columns are per-bake diagnostics '
    +'only — bakes with compressed dial spans are flattered natively (measured: GL2). ') : '';
  const cap=el('div',{class:'cap',html:domNote+pxNote+'Sortable scoreboard — click a header. SROCC is polarity-corrected '
    +'(|SROCC| for JND corpora). <b>regime</b> = the model’s TRUE input width, derived from the ZNPR’s '
    +'<code>n_inputs</code> (372/720/924/944-class) — NOT the recorded flag string, which reads “720” '
    +'cosmetically on the campaign verdicts (the flag shows in the bake-picker tooltip when it differs; for an '
    +'ens×k row the width is the anchor member’s). <b>composite</b> = the Rust <code>product_composite</code> (CID22·1.0 + '
    +'imazen26·0.5 + nonphoto·0.3 + KonJND·0.2 + AIC·0.15; KADID/TID excluded, train==val), READ from the JSON '
    +'not re-derived. <b>CID22 95%CI±</b> = bootstrap half-width; bakes with overlapping CIs are a statistical '
    +'TIE, not an ordering. <b>CID22 %bwd</b> = share of reference ladders ranked BACKWARDS (no pooled stat sees '
    +'it). <b>M3a-attr</b> = the DEPLOYABLE attribution-density steering map vs \u0394S (exact integrands + SAT, task #67 \u2014 the map codecs query); <b>M3-coh</b> = the legacy signal fold, kept for the before/after story (the 128px fold inversion the attribution map cures). <b>M3 drop%</b> = f156-371 mass the FOLD cannot spatialize — read a low M3 against it (high drop% '
    +'= M3 structurally capped, not incoherent). An <b>em-dash in any cell means NOT MEASURED</b> — never a '
    +'measured zero. Greyed row = reject-gate (CID22&lt;0.84 or nonphoto&lt;0.80). '
    +'<b>ens×k</b> = an equal-weight ENSEMBLE of k bakes, scored through the identical verdict invocation '
    +'as every single-bake row: rank/dial/corruption numbers are directly comparable, but an ensemble is an '
    +'<b>evaluation function, not a shippable artifact</b> — there is no single ZNPR, so <b>M3a/M3 are not '
    +'computable for it</b> (the coherence instrument loads one bake) and its Model-details card describes '
    +'the ANCHOR member only. Distillation to a single bake is pending. '
    +'Rows list EVERY promoted cell (dimmed = hidden from charts; click a row to toggle it). '
    +'Hidden-by-default grid cells carry the same scalar stats as curated ones — only embedded '
    +'scatter data is curated-set-only (see the scatter section). '
    +'<b>⚠ = registry annotation</b> (benchmarks/eval_annotations.json — the badge sits on '
    +'the exact number the entry caveats; hover for the reason): '
    +'dial-mono on spline-less bakes is RAW-UNIT (flattered ~3-6 pts vs real dial units); '
    +'a k=1 CID22 draw whose seed sibling lands ~0.09 lower is flagged UNREPLICATED (context, '
    +'never a candidate claim — R.R0); '
    +'<b>HF-NL/ref</b> = hfnlproxy per-reference mean signed SROCC (quality-oriented; per-ref, '
    +'never pooled — hover the header; Δ under the ~0.04 axis LSD is noise; 80 pre-pin cells were '
    +'sign-flipped and are REPAIRED per appendix O — see the HF-NL axis panel) — “— (absent)” on cells that predate '
    +'the instrument is <b>absent-not-failed</b> (not measured ≠ measured fail); KADID/TID stay '
    +'train==val integrity guards everywhere. <b>dom</b>-tagged rows are DOMINATED (strictly '
    +'beaten by a same-class sibling on every measured floor axis + composite) — kept on the '
    +'board, dimmed + default-off behind the “dominated” chip; nothing is deleted.'
    +(LT?' <b>2shot/3shot ±2</b> = JXL loop targeting: cells (of '+ltN()+') where the DECODED-judged score lands '
    +'within ±2 of target in the bake’s own units at encode budget k=2/3, emit-best (emit-last + outer arms: '
    +'see the JXL loop targeting section).':'')});
  const tbl=el('table',{});
  const thead=el('tr',{});
  COLS.forEach(c=>{const jnd=JND_CORPORA.has(c[0]);
    const th=el('th',{class:(c[0]==='name'||c[0]==='regime'?'lbl':'')
      +(state.sortKey===c[0]?' sorted'+(state.sortDir>0?' asc':''):''),
      text:c[1]+(jnd?' JND↓':'')});
    if(jnd)th.setAttribute('title',JND_TIP(c[0]));
    if(c[0]==='gates')th.setAttribute('title','Gate glyphs, in order G E D K: '
      +GATE_DEFS.map(([g,l,d])=>l+' = '+d).join('; ')
      +'. \u2713 pass, \u2717 fail, \u00b7 NOT MEASURED (absent \u2260 failed). '
      +'Use the gate-filter chips in the top bar to EXCLUDE failing rows from this list.');
    if(c[0]==='hfnl')th.setAttribute('title','hfnlproxy per-REFERENCE mean signed SROCC '
      +'(quality-oriented, pin 730a386e): + = orders each near-lossless ladder like ssim2, '
      +'- = inverted. NOT the pooled SROCC (range-restricted). Differences under the axis LSD '
      +(HA&&HA._meta&&HA._meta.axis_lsd?'~'+(+HA._meta.axis_lsd.median).toFixed(3)+' (p90 '+(+HA._meta.axis_lsd.p90).toFixed(3)+') ':'~0.04 ')
      +'are ref-sampling noise. See the HF-NL axis panel below.');
    // mountTable, NOT renderTable: renderTable RETURNS a detached wrapper — calling it
    // from the click handler built the sorted table and threw it away, so the visible
    // scoreboard never re-sorted (2026-08-04 user report; bug present since 62404415).
    th.onclick=()=>{if(state.sortKey===c[0])state.sortDir*=-1;else{state.sortKey=c[0];state.sortDir=c[2]?1:-1;}mountTable();};
    thead.appendChild(th);});
  tbl.appendChild(el('thead',{},thead));
  // column min/max for shading (numeric cols only), across ALL bakes
  const ranges={};
  COLS.forEach(c=>{if(c[0]==='name'||c[0]==='regime')return;
    const vs=DATA.bakes.map(c[3]).filter(v=>v!=null&&isFinite(v));
    ranges[c[0]]=vs.length?[Math.min(...vs),Math.max(...vs)]:[0,1];});
  const col=COLS.find(c=>c[0]===state.sortKey)||COLS[2];
  const pool=state.gateFilter.size?DATA.bakes.filter(b=>!gateExcluded(b)):DATA.bakes;
  const nGateHidden=DATA.bakes.length-pool.length;
  const rows=[...pool].sort((a,b)=>{let x=col[3](a),y=col[3](b);
    if(x==null&&y==null)return 0; if(x==null)return 1; if(y==null)return -1; // nulls last either direction
    if(typeof x==='string')return state.sortDir*String(x).localeCompare(String(y));
    x=x==null?-1e9:x;y=y==null?-1e9:y;return state.sortDir*(x-y);});
  const tb=el('tbody',{});
  rows.forEach(b=>{
    const tr=el('tr',{class:b.reject?'reject':''});
    if(!state.visible.has(b.name))tr.style.opacity=.45;
    if(DOM(b))tr.style.opacity=Math.min(tr.style.opacity||1,.35);
    COLS.forEach(c=>{
      const v=c[3](b);
      const td=el('td',{class:(c[0]==='name'||c[0]==='regime')?'lbl':'',text:fmtCell(c[0],v,b)});
      if(c[0]==='name'){td.textContent='';nameInto(td,b,b.is_stub?' ✳':'');}
      if(c[0]==='gates'){td.setAttribute('title',gateTitle(b));td.style.cursor='help';
        td.style.fontFamily='ui-monospace,monospace';td.style.letterSpacing='1px';}
      // ---- fairness layer cells (2026-09-04) ----
      if(c[0]==='fair'){td.setAttribute('title',tierTitle(b));td.style.cursor='help';
        td.style.color=TIER(b)==='VERIFIED-FAIR'?'var(--good)':(TIER(b)==='FAIR-NOTED'?'var(--warn)':'var(--critical)');}
      if(c[0]==='k'||c[0]==='cmean'||c[0]==='cspread'){
        const t=sgTitle(b,'composite');
        td.setAttribute('title',t||((b.fair&&b.fair.k===1)
          ?'UNREPLICATED (k=1) — one draw, spread unknown. '+SEEDNOTE
          :'UNGROUPABLE — no embedded repro argv, so no seed group can be formed.'));
        td.style.cursor='help';}
      if(c[0]==='gaddr'){td.setAttribute('title',gaddrTitle(b));td.style.cursor='help';
        if(b.gaddr&&b.gaddr.fail)td.style.color='var(--critical)';}
      if(c[0].charAt(0)==='w'&&c[0].length===2){
        const e=examOf(b.name);
        if(e){const cl=e[c[0].toUpperCase()];
          td.setAttribute('title','ssim2 EXAM '+c[0].toUpperCase()+' — '+
            ((EXAM.clauses||[]).find(x=>x.id===c[0].toUpperCase())||{}).question+
            '\nverdict: '+cl.v+(cl.note?('  ('+cl.note+')'):'')+
            '\nreference row: peer_ssim2. Source: '+EXAM._schema.source);
          td.style.cursor='help';
          td.style.color=cl.v==='PASS'?'var(--good)':(cl.v==='FAIL'?'var(--critical)':'var(--muted)');
        }else{td.setAttribute('title','this bake is not one of the exam\u2019s six scored candidates — NOT MEASURED, never a fail.');}
      }
      // NON-RANKING columns: dimmed, with the reason. Visible as guards, never a win.
      if(NONRANK[c[0]]){td.style.opacity=.55;td.style.fontStyle='italic';
        td.setAttribute('title',(td.getAttribute('title')||'')+'\nNOT RANKING SIGNAL — '+NONRANK[c[0]]);
        td.style.cursor='help';}
      // ⚠ registry badges (benchmarks/eval_annotations.json). GENERIC (2026-08-06):
      // any matched entry whose `fields` cover this column's dot-path badges this cell,
      // so adding a registry entry is sufficient — no per-id JS. Only on a rendered
      // value; the null/absent case is the dedicated block below (it rewrites the text).
      if(v!=null){
        const ids=annForCol(b,c[0]);
        if(ids.length){
          td.append(el('span',{style:'margin-left:3px;cursor:help',
            title:ids.map(id=>'⚠ '+annKind(id)+' ('+id+'): '+annReason(id)).join('\n\n'),
            text:'⚠'}));
        }
      }
      if(c[0]==='hfnl'&&v==null&&ANN(b,'hfnl-absent-not-failed')){
        td.textContent='— (absent)';
        td.setAttribute('title','⚠ absent-not-failed (hfnl-absent-not-failed): '
          +annReason('hfnl-absent-not-failed'));
        td.style.cursor='help';
      }
      if(c[0]!=='name'&&c[0]!=='regime'&&c[0]!=='fair'&&!NONRANK[c[0]]
         &&!(c[0].charAt(0)==='w'&&c[0].length===2)&&v!=null&&isFinite(v)){
        const[lo,hi]=ranges[c[0]];let t=hi===lo?.5:(v-lo)/(hi-lo);
        // invert shading where lower is better (tied dead-zone, CI width,
        // backwards-ref share, dropped-mass — all "smaller is better")
        if(c[0]==='dial_tied'||c[0]==='cid22_ci'||c[0]==='cid22_bwd'||c[0]==='m3_mass'||c[0]==='loop3err')t=1-t;
        td.style.background='color-mix(in srgb, var(--seq-hi) '+Math.round(t*62)+'%, var(--surface-1))';
        if(t>.6)td.style.color='#fff';
      }
      tr.appendChild(td);
    });
    tr.onclick=()=>{state.visible.has(b.name)?state.visible.delete(b.name):state.visible.add(b.name);rerender();renderBar();};
    tr.style.cursor='pointer';
    tb.appendChild(tr);
  });
  tbl.appendChild(tb);
  const gf=state.gateFilter.size?el('div',{class:'cap',html:'<b>GATE FILTER ON ['
    +[...state.gateFilter].map(g=>(GATE_DEFS.find(x=>x[0]===g)||[g,g])[1]).join('+')
    +']: '+nGateHidden+' gate-FAILING rows excluded from this list.</b> '
    +'Only measured FAILs are excluded — a not-measured gate (\u00b7) never hides a row.'}):null;
  // Gates legend (user directive 2026-08-29: "the gauntlet gates are entirely
  // obscure and not explained in the report"). A visible, plain-language
  // explainer for BOTH gate systems + the ruler caveats; collapsed by default.
  const lg=el('details',{class:'cap'});
  lg.append(el('summary',{html:'<b>How to read the gates + rulers (click to expand)</b>'}));
  lg.append(el('div',{html:
    '<p><b>The gates column (✓✗·, order G E D K)</b> — hard PASS/FAIL verdicts from the '
    +'committed registry (<span class="mono">benchmarks/board_gates_2026-08-28.json</span>) + build-time checks. '
    +'✓ pass · ✗ fail · · = NOT MEASURED (never treated as fail). Hover a glyph for the specific failing clause + source.</p>'
    +'<ul>'
    +'<li><b>G = G-OUT v2</b> (owner <span class="mono">outlier_gate.py</span>): per-corpus outlier discipline of the '
    +'model’s raw predictions — clause R (outlier ratio), S (p99 chart-z: the 99th-percentile studentized miss vs a '
    +'peer-calibrated bar), B (max-z blowup), D (declared dial range vs emitted floor). Bars are PEER-derived '
    +'(calibrated on the flagship population), never model-derived.</li>'
    +'<li><b>E = eligibility</b>: the frozen two-zone battery — paired bootstrap CIs on cid22 AND validate-hfnl vs both '
    +'the incumbent and the candidate-of-record (CI must not show a significant loss), + LF-band floors + LF monotonicity. '
    +'HDR bakes take the HDR-lane freeze battery instead.</li>'
    +'<li><b>D = G-GRAN v2</b> (REGISTERED, not yet frozen — opt-in filter): peer-anchored two-sided dial calibration at '
    +'three truth anchors, knob-quantum-aware gap checks, secant attainability.</li>'
    +'<li><b>K = knob-end</b> (G-GRAN v1 semantics, computed at build): can the dial REACH each codec’s top '
    +'(bar = incumbent reach − 1: avif 96.2 / jpeg 94.4 / jxl 96.6 / webp 91.9), with HF-zone span ≥ 8 and '
    +'q-curve monotonicity ≥ 0.93.</li>'
    +'</ul>'
    +'<p><b>The "Gate scorecard" table below</b> is a DIFFERENT system: CODEC_TARGET_GOALS soft-gates '
    +'(continuous 0–1 scores, weighted into a shippability scalar) — diagnostic shading, not pass/fail law.</p>'
    +'<p><b>Ruler caveats (read before comparing rows):</b> '
    +'kadid rows are <b>train==eval for every current model</b> (integrity guards, not skill); '
    +'tid is <b>RETIRED TO TRAIN-ONLY</b> (user ruling 2026-08-29) — do not rank on it; '
    +'konjnd board rows for 372-class bakes historically scored the full 1,008-ref file while 944 bakes scored the '
    +'JPEG-504 — same-pair kon reads live in the campaign doc’s single-ruler table; '
    +'imazen26/nonphoto/hfnlproxy are family slices whose CUT changed 2026-08-28 (validate-family) — all 944 rows '
    +'now read the same generation, but 372-class bakes read a different (test-family) cut of the same corpus family: '
    +'direction is comparable, decimals are not. The registered single-ruler comparison (identical pairs, '
    +'rank-identical targets, paired CIs) is in <span class="mono">benchmarks/balance_campaign_2026-08-28.md</span>.</p>'}));
  wrap.append(el('h2',{text:'Scoreboard'}),cap,lg);if(gf)wrap.append(gf);wrap.append(tbl);
  return wrap;
}

// ---- scatter-matrix cell (ECharts): pred (x) vs reference (y), OLS fit as a line
// series, dataZoom inside (both axes) + x slider, constant symbol size at any zoom.
function qqMap(pts){
  // shape normalization: map each prediction, BY RANK, onto the reference's
  // own quantiles — removes output shaping (splines / range compression) so
  // cells are visually comparable across models. SROCC is rank-invariant.
  const ys=pts.map(p=>p[1]).sort((a,b)=>a-b);
  const order=pts.map((p,i)=>[p[0],i]).sort((a,b)=>a[0]-b[0]);
  const out=new Array(pts.length);
  order.forEach((oi,rank)=>{const i=oi[1];out[i]=[ys[Math.min(rank,ys.length-1)],pts[i][1],pts[i][0]];});
  return out;
}
function scatterOpt(b,corp,ref,cell){
  const t=TH();const c=color(b);
  const refLab=DATA.refLabels[ref]||ref;
  const norm=state.shapeNorm!==false;
  const pts=norm?qqMap(cell.pts):cell.pts;
  const series=[{type:'scatter',name:b.name,data:pts,symbolSize:6,
    itemStyle:{color:c,opacity:.55},emphasis:{itemStyle:{opacity:1}},z:2}];
  if(norm){
    const ys=cell.pts.map(p=>p[1]);const lo=Math.min.apply(null,ys),hi=Math.max.apply(null,ys);
    series.push({type:'line',silent:true,symbol:'none',z:3,
      data:[[lo,lo],[hi,hi]],lineStyle:{color:t['text-secondary'],width:1.5,opacity:.7,type:'dashed'}});
    if(cell.geo&&cell.geo.mad){const m=4*cell.geo.mad;
      [[lo,lo+m,hi,hi+m],[lo,lo-m,hi,hi-m]].forEach(l=>series.push({type:'line',silent:true,
        symbol:'none',z:1,data:[[l[0],l[1]],[l[2],l[3]]],
        lineStyle:{color:t['text-secondary'],width:1,opacity:.35,type:'dotted'}}));}
  }else if(cell.fit)series.push({type:'line',silent:true,symbol:'none',z:3,
    data:[[cell.fit[0],cell.fit[1]],[cell.fit[2],cell.fit[3]]],
    lineStyle:{color:c,width:2,opacity:.9}});
  // hfnl visual scaling: the near-lossless band fills the plot instead of
  // hiding in a corner (tight data-quantile limits both axes).
  let axX=axStyle(),axY=axStyle();
  if(corp==='hfnlproxy'){
    const q=(arr,f)=>{const a=[...arr].sort((x,y)=>x-y);return a[Math.floor(f*(a.length-1))];};
    const xs=pts.map(p=>p[0]),ys2=pts.map(p=>p[1]);
    const pad=(lo,hi)=>{const m=(hi-lo)*0.04||0.5;return[lo-m,hi+m];};
    const xr=pad(q(xs,0.01),q(xs,0.99)),yr=pad(q(ys2,0.01),q(ys2,0.99));
    axX=Object.assign(axStyle(),{min:xr[0],max:xr[1]});
    axY=Object.assign(axStyle(),{min:yr[0],max:yr[1]});
  }
  return{animation:false,
    title:{text:(b.name.length>30?b.name.slice(0,29)+'…':b.name)+ensTag(b),
      subtext:'ρ '+f3(cell.srocc)+'   r '+f3(cell.plcc)+'   n='+cell.n
        +(cell.geo?('\nout '+(cell.geo.out4*100).toFixed(1)+'%·maxd '+(cell.geo.maxd*100).toFixed(0)
          +'%sp·cov '+(cell.geo.cov*100).toFixed(0)+'%·clump '+(cell.geo.clump*100).toFixed(0)
          +'%·clamp '+(cell.geo.clampLo*100).toFixed(1)+'/'+(cell.geo.clampHi*100).toFixed(1)+'%'):''),
      top:2,left:8,itemGap:1,
      textStyle:{color:t['text-primary'],fontSize:10.5,fontWeight:600},
      subtextStyle:{color:t['text-secondary'],fontSize:9.5}},
    tooltip:Object.assign(ttStyle(),{trigger:'item',formatter:p=>
      p.seriesType==='scatter'
        ?('<b>'+b.name+'</b><br>'+(state.shapeNorm!==false&&p.value.length>2
            ?('pred(raw) <b>'+f3(p.value[2])+'</b> → ref-scale <b>'+f3(p.value[0])+'</b>')
            :('pred <b>'+f3(p.value[0])+'</b>'))+'<br>'+refLab+' <b>'+f3(p.value[1])+'</b>')
        :('OLS fit')}),
    grid:{left:42,right:10,top:38,bottom:42},
    xAxis:axX,yAxis:axY,
    dataZoom:[{type:'inside',xAxisIndex:0,filterMode:'none'},
              {type:'inside',yAxisIndex:0,filterMode:'none'},
              dzSlider({xAxisIndex:0,filterMode:'none'})],
    series};
}

function renderScatter(){
  const host=$('#scatter');if(!host)return;host.innerHTML='';
  const ref=state.ref;const bs=visBakes();
  host.append(el('h2',{text:'Correlation scatter matrix — predicted vs '+(DATA.refLabels[ref]||ref)}));
  const nSc=bs.filter(b=>Object.keys(b.scatter).length).length;
  const tgl=el('button',{class:'btn',text:state.shapeNorm!==false?'shape-normalized ✓ (click for raw units)':'raw units (click to shape-normalize)'});
  tgl.onclick=()=>{state.shapeNorm=!(state.shapeNorm!==false);renderScatter();};
  host.append(tgl);
  host.append(el('div',{class:'cap',html:(state.shapeNorm!==false
    ?'<b>Shape-normalized</b>: each prediction is mapped BY RANK onto the reference’s own quantiles — output shaping (splines, range compression) is removed, cells are visually comparable across models, and the dashed diagonal is the ideal. ρ is rank-invariant (unchanged). hfnl cells use tight band limits so the near-lossless range fills the plot. '
    :'<b>Raw units</b>: predictions in each model’s own dial units — shaping differences (spline shape, range compression) dominate the visual; use for calibration reading only. ')
    +'One clean scatter per (bake × corpus) for the selected reference; '
    +'bakes sit side by side per corpus so you can compare fits. ρ = canonical SROCC, r = PLCC. '
    +'Switch reference in the bar above; toggle bakes to add/remove columns. '
    +'<b>Scatter data is embedded for the curated headline set only</b> (registered size rule — '
    +'the offline file stays openable); '+nSc+' of '+bs.length+' visible bakes carry it here. '
    +'Grid-interior cells keep every scalar stat in the sections above, and their full '
    +'per-pair data lives in the source verdict recorded in each fulleval JSON '
    +'(<code>source_verdict</code> / <code>per_pair_stripped</code>).'}));
  if(!bs.length){host.append(el('p',{class:'sub',text:'no bakes selected.'}));return;}
  // corpora that carry this reference for any visible bake
  const corps=DATA.corpOrder.filter(c=>bs.some(b=>b.scatter[c]&&b.scatter[c][ref]));
  const extra=[...new Set(bs.flatMap(b=>Object.keys(b.scatter)))].filter(c=>!DATA.corpOrder.includes(c)&&bs.some(b=>b.scatter[c]&&b.scatter[c][ref]));
  [...corps,...extra].forEach(corp=>{
    host.append(corpTitle(el('div',{class:'corpttl',text:corpMark(corp)}),corp));
    const row=el('div',{class:'scrow'});
    bs.forEach(b=>{const cell=b.scatter[corp]&&b.scatter[corp][ref];
      if(cell&&cell.pts.length)row.appendChild(mountChart('scatter',238,252,scatterOpt(b,corp,ref,cell)));});
    if(!row.children.length)row.appendChild(el('div',{class:'cap',text:'(no visible bake has '+ref+' here)'}));
    host.appendChild(row);
  });
}

// ---- cross-corpus SROCC heatmap (ECharts heatmap + visualMap; visible bakes only)
function heatOpt(bs,corps,TVSET){
  const t=TH();
  const names=bs.map(b=>b.name+ensTag(b));
  const data=[];
  bs.forEach((b,i)=>corps.forEach((c,j)=>{const r=b.rank[c];const sv=sgn(c,r);
    if(sv!=null&&isFinite(sv)){
      const v=+(+sv).toFixed(4);
      const raw=rawSigned(r);const jnd=JND_CORPORA.has(c);
      // Defect flag is ORIENTATION-AWARE: a quality corpus is inverted when
      // signed < 0; a JND↓ corpus is inverted (orientation MISMATCH) when
      // signed > 0 — negative there is its declared convention, not a defect.
      const bad=raw!=null&&(jnd?raw>0:raw<0);
      data.push({value:[j,i,v],_n:r.n,_raw:raw,_jnd:jnd,_bad:bad,
        label:{color:Math.abs(v)>.72?'#fff':t['text-secondary']}});
    }}));
  return{animation:false,
    tooltip:Object.assign(ttStyle(),{formatter:p=>{const d=p.data||{};const c=corps[p.value[0]];
      let s='<b>'+names[p.value[1]]+'</b> × '+c+'<br>'
        +(d._jnd?'|SROCC| <b>'+f3(p.value[2])+'</b> (signed '+f3(d._raw)+' — JND↓ convention)'
                :'signed SROCC <b>'+f3(p.value[2])+'</b>');
      if(d._bad)s+=' <b>⛔ '+(d._jnd?'ORIENTATION MISMATCH (positive on a JND↓ corpus)':'INVERTED')+'</b>';
      if(c==='sdr25')s+='<br>⊂ aic4 — the 50 sdr25 stimuli are a subset of aic4 (not independent)';
      return s+(d._n!=null?' · n='+d._n:'');}}),
    grid:{left:250,right:26,top:56,bottom:48},
    xAxis:{type:'category',position:'top',data:corps.map(c=>corpMark(c)+(TVSET.has(c)?' ⚠':'')),
      axisLine:{show:false},axisTick:{show:false},
      axisLabel:{fontSize:9.5,rotate:32,color:v=>String(v).includes('⚠')?t.warn:t['text-secondary']}},
    yAxis:{type:'category',inverse:true,
      data:names.map(n=>n.length>38?n.slice(0,37)+'…':n),
      axisLine:{show:false},axisTick:{show:false},
      axisLabel:{fontSize:10,color:t['text-primary']}},
    // SIGNED scale (2026-08-04): spans [-1,1] so an ANTI-CORRELATED bake reads as the
    // cold/serious end instead of drawing a hot cell off |SROCC|. Diverging at 0.
    visualMap:{min:-1,max:1,calculable:true,orient:'horizontal',left:'center',bottom:4,
      precision:2,itemHeight:110,itemWidth:11,
      inRange:{color:[t.serious||'#b4413f',t['seq-lo'],t['seq-hi']]},
      textStyle:{color:t.muted,fontSize:9}},
    dataZoom:[{type:'inside',yAxisIndex:0,filterMode:'filter',
               zoomOnMouseWheel:'shift',moveOnMouseWheel:true},
              {type:'inside',xAxisIndex:0,filterMode:'filter',zoomOnMouseWheel:'shift'}],
    series:[{type:'heatmap',data,
      label:{show:true,fontSize:9,formatter:p=>f3(p.value[2]).replace('0.','.')},
      itemStyle:{borderColor:t.plane,borderWidth:1.5,borderRadius:3},
      emphasis:{itemStyle:{shadowBlur:6,shadowColor:'rgba(0,0,0,.35)'}}}]};
}
function renderHeat(){
  const host=$('#heat');if(!host)return;host.innerHTML='';
  const bs=visBakes();if(!bs.length){return;}
  const corps=DATA.corpOrder.filter(c=>bs.some(b=>b.rank[c]));
  // train==val corpora (KADID/TID) read from the Rust-emitted flag — mark them so
  // their SROCC is not read as held-out skill (stats review Rec-6).
  const TVSET=new Set();
  DATA.bakes.forEach(b=>Object.entries(b.rank||{}).forEach(([c,r])=>{if(r&&r.train_eq_val)TVSET.add(c);}));
  const W=250+corps.length*64+40,Ht=Math.max(220,120+bs.length*22+40);
  host.append(el('h2',{text:'Cross-corpus SROCC'}),
    el('div',{class:'cap',html:'Bake × corpus, SROCC (<b>|SROCC| for JND↓ corpora</b> — aic4/konjnd/sdr25 carry '
      +'distortion-oriented JND-family labels per the EXPECTED_ORIENTATION registry, so negative signed SROCC '
      +'is their declared CONVENTION, shown as magnitude; a positive signed value there would be an '
      +'orientation mismatch and is ⛔-flagged in the tooltip, exactly mirroring a negative on a '
      +'quality-oriented corpus). <b>sdr25 ⊂ aic4</b>: all 50 sdr25 stimuli are contained in aic4&#39;s 300 '
      +'(Appendix I) — read them as one instrument, not two corroborating corpora. Sequential blue: darker = '
      +'higher (drag the visualMap handles to range-filter cells). <b>⚠</b> (amber header) = KADID/TID, '
      +'train==val — SROCC rewards memorization, not held-out generalization. Shift+wheel zooms rows/columns; '
      +'wheel scrolls rows when many bakes are visible; double-click resets.'}));
  const wrap=el('div',{style:'overflow-x:auto'});
  wrap.append(mountChart('heat',W,Ht,heatOpt(bs,corps,TVSET)));
  host.append(wrap);
}

// ---- operating-point trade map (ECharts scatter, labeled points, dataZoom). Labels
// auto-hide on overlap and REAPPEAR as you zoom in (labelLayout re-runs per zoom — the
// semantic-zoom fix for label pile-ups; every point still names itself in the tooltip).
function tradeOpt(xc,yc,xl,yl,pts){
  const t=TH();
  return{animation:false,
    tooltip:Object.assign(ttStyle(),{formatter:p=>'<b>'+p.data.name+'</b><br>'
      +xl+' <b>'+f3(p.value[0])+'</b><br>'+yl+' <b>'+f3(p.value[1])+'</b>'}),
    grid:{left:52,right:16,top:14,bottom:46},
    xAxis:Object.assign(axStyle(xl),{nameGap:26}),
    yAxis:Object.assign(axStyle(yl),{nameGap:36}),
    dataZoom:[{type:'inside',xAxisIndex:0,filterMode:'none'},
              {type:'inside',yAxisIndex:0,filterMode:'none'}],
    series:[{type:'scatter',symbolSize:10,
      data:pts.map(p=>({value:[p.x,p.y],name:p.b.name+ensTag(p.b),
        itemStyle:{color:color(p.b),borderColor:t['surface-1'],borderWidth:1.2}})),
      label:{show:true,position:'right',distance:5,fontSize:9.5,color:t['text-primary'],
        formatter:p=>p.data.name},
      labelLayout:{hideOverlap:true},
      emphasis:{label:{fontWeight:700},itemStyle:{shadowBlur:5,shadowColor:'rgba(0,0,0,.3)'}}}]};
}
function renderTrade(){
  const host=$('#trade');if(!host)return;host.innerHTML='';
  const bs=visBakes();if(!bs.length)return;
  host.append(el('h2',{text:'Operating-point trade map'}),
    el('div',{class:'cap',text:'Upper-right = better on both. Points are directly labeled (identity is never '
      +'color-alone); overlapping labels hide at 1x and re-appear as you zoom (wheel/drag; double-click resets).'}));
  const grid=el('div',{class:'grid'});
  [['cid22','nonphoto','CID22 SROCC','non-photo SROCC'],
   ['cid22','konjnd','CID22 SROCC','KonJND |SROCC|']].forEach(([xc,yc,xl,yl])=>{
    const pts=bs.map(b=>({b,x:rs(b,xc),y:rs(b,yc)})).filter(p=>p.x!=null&&p.y!=null&&isFinite(p.x)&&isFinite(p.y));
    if(pts.length)grid.append(mountChart('trade',390,300,tradeOpt(xc,yc,xl,yl,pts)));
  });
  if(grid.children.length)host.appendChild(grid);
}

// ---- FULL MOHAMMADI PANEL (all six stats per corpus, per visible bake)
function renderMPanel(){
  const host=$('#mpanel');if(!host)return;host.innerHTML='';
  const bs=visBakes();if(!bs.length)return;
  const corps=DATA.corpOrder.filter(c=>DATA.bakes.some(b=>b.rank[c]));
  if(!state.mcorp||!corps.includes(state.mcorp))state.mcorp=corps[0];
  const TV=new Set();
  DATA.bakes.forEach(b=>Object.entries(b.rank||{}).forEach(([c,r])=>{if(r&&r.train_eq_val)TV.add(c);}));
  host.append(el('h2',{text:'Full Mohammadi panel'}));
  host.append(el('div',{class:'cap',html:'All six stats (Mohammadi 2025): SROCC/KROCC on raw ranks; '
    +'PLCC, OR, PWRC, Z-RMSE on the 4-param-logistic-rescaled prediction. <b>OR + Z-RMSE: lower is '
    +'better</b>; OR is a catastrophe gate, not a ranker. <b>SROCC</b> is signed, with the bootstrap 95% '
    +'CI half-width: on a quality-oriented corpus a negative = globally inverted bake; on a <b>JND↓</b> '
    +'corpus (aic4/konjnd/sdr25 — distortion-oriented labels per the EXPECTED_ORIENTATION registry) '
    +'negative is the declared CONVENTION and a POSITIVE would be the defect — row shading follows the '
    +'orientation, not the bare sign. <b>per-ref / %bwd</b> = within-image mean SROCC '
    +'and share of reference ladders ranked backwards (— when the corpus carries no ref identity). '
    +'⚠ = train==val (KADID/TID: memorization, not held-out skill). Click a header to sort.'}));
  const sel=el('div',{class:'bar',style:'margin:6px 0 10px'});
  corps.forEach(c=>{
    const b=corpTitle(el('button',{class:'btn',text:corpMark(c)+(TV.has(c)?' ⚠':'')}),c);
    if(state.mcorp===c)b.style.cssText='font-weight:700;outline:2px solid var(--seq-hi)';
    b.onclick=()=>{state.mcorp=c;renderMPanel();};
    sel.append(b);
  });
  host.append(sel);
  const c=state.mcorp;
  const tbl=el('table',{});
  const thead=el('tr',{});
  ['bake','n','SROCC ±CI','PLCC','KROCC','OR','PWRC','Z-RMSE','per-ref','%bwd'].forEach((h,i)=>
    thead.append(el('th',{class:i===0?'lbl':'',text:h})));
  tbl.append(el('thead',{},thead));
  const tb=el('tbody',{});
  const rows=bs.filter(b=>b.rank[c]).sort((a,b)=>(b.rank[c].srocc||0)-(a.rank[c].srocc||0));
  const cJnd=JND_CORPORA.has(c);
  rows.forEach(b=>{
    const r=b.rank[c];
    const sroccs=rawSigned(r);
    const ciw=r.srocc_ci?(r.srocc_ci[1]-r.srocc_ci[0])/2:null;
    const tr=el('tr',{});
    tr.append(nameInto(el('td',{class:'lbl'}),b));
    const cells=[
      r.n!=null?String(r.n):'—',
      (sroccs!=null?(sroccs>=0?'+':'')+sroccs.toFixed(4):'—')+(ciw!=null?' ±'+ciw.toFixed(3):'')
        +(cJnd&&sroccs!=null&&sroccs<0?' (JND↓)':''),
      f3(r.plcc), f3(r.krocc), r.or!=null?r.or.toFixed(4):'—', f3(r.pwrc),
      r.z_rmse!=null?r.z_rmse.toFixed(3):'—',
      r.per_ref_mean!=null?(r.per_ref_mean>=0?'+':'')+r.per_ref_mean.toFixed(4):'—',
      r.frac_negative!=null?pct(r.frac_negative):'—'];
    cells.forEach(v=>tr.append(el('td',{text:v})));
    // Orientation-aware defect shading: bare sign on a quality corpus,
    // MIRRORED sign on a JND↓ corpus (negative there is the convention).
    if(sroccs!=null&&(cJnd?sroccs>0:sroccs<0))
      tr.style.background='color-mix(in srgb, var(--serious) 18%, transparent)';
    tb.append(tr);
  });
  tbl.append(tb);
  makeSortable(tbl);
  const wrap=el('div',{style:'overflow-x:auto'});wrap.append(tbl);host.append(wrap);
  // Per-band SROCC grouped bars for the selected corpus (when the corpus is banded).
  // ONLY cells cut on the current band scheme are shown: a legacy fixed-decile cell's
  // B9 is a different quantity (on CID22, 43 pairs spanning 0.019 MOS, published as an
  // ABSOLUTE value), so mixing the two in one table would compare two measurements.
  const bandedAll=rows.filter(b=>b.rank[c]&&b.rank[c].bands);
  const banded=bandedAll.filter(b=>b.rank[c].band_scheme);
  const bandLegacy=bandedAll.length-banded.length;
  if(banded.length){
    const schemeName=banded[0].rank[c].band_scheme.name;
    const bmin=banded[0].rank[c].band_scheme.n_min,bspan=banded[0].rank[c].band_scheme.span_min;
    host.append(el('h3',{text:'Per-band SROCC — '+c}));
    host.append(el('div',{class:'cap',html:'SIGNED per-band SROCC across the quality range '
      +'(low quality → high). Scheme <code>'+schemeName+'</code>: fixed deciles accumulated into '
      +'the finest partition whose every band holds n&nbsp;≥&nbsp;'+bmin+' pairs spanning '
      +'≥&nbsp;'+bspan+' of target. A band that cannot clear both floors is <b>NOT-MEASURED</b> — '
      +'it publishes no statistic and draws nothing; that is "not measured", never zero. '
      +'Values are SIGNED, so an inverted band draws BELOW the axis instead of being hidden by '
      +'|·| (which used to make a more deeply inverted band score HIGHER). Band SROCC is still '
      +'range-restricted, so compare bakes, not bands. Wheel/slider zooms the band axis.'
      +(bandLegacy?' <b>'+bandLegacy+'</b> visible bake'+(bandLegacy===1?'':'s')
        +' cut on the pre-2026-08-06 fixed deciles '+(bandLegacy===1?'is':'are')
        +' EXCLUDED here (re-verdict or recut to include).':'')}));
    const bands=banded[0].rank[c].bands.map(x=>x.band);
    // Every shown cell must carry identical edges, or the columns lie.
    const bandMismatch=banded.filter(b=>b.rank[c].bands.map(x=>x.band).join(',')!==bands.join(','));
    if(bandMismatch.length){
      host.append(el('div',{class:'cap',html:'<b>⚠ '+bandMismatch.length+'</b> bake(s) carry '
        +'different band edges and are excluded from the bars/table below: '
        +bandMismatch.map(b=>b.name).join(', ')}));
      bandMismatch.forEach(b=>{const i=banded.indexOf(b);if(i>=0)banded.splice(i,1);});
    }
    const t=TH();
    const bseries=banded.map(b=>({type:'bar',name:b.name,
      barGap:'20%',barCategoryGap:'25%',
      data:b.rank[c].bands.map(row=>({
        value:row&&row.srocc_signed!=null?row.srocc_signed:null,
        _n:row?row.n:null,_plcc:row?row.plcc:null,_pwrc:row?row.pwrc:null,
        _span:row?row.span:null,_nm:row?row.not_measured_reason:null,
        _z:row&&row.z_rmse!=null?row.z_rmse:null,
        itemStyle:{color:color(b),opacity:0.95}}))}));
    const bandOption={animation:false,
      tooltip:Object.assign(ttStyle(),{trigger:'item',formatter:p=>{
        const d=p.data||{};
        const head='<b>'+p.seriesName+'</b> '+p.name+' n='+(d._n!=null?d._n:'?')
          +(d._span!=null?' span='+(+d._span).toFixed(3):'');
        if(d._nm)return head+'<br><b>NOT MEASURED</b> — '+d._nm;
        return head
          +'<br>SROCC <b>'+f3(d.value)+'</b> · PLCC '+f3(d._plcc)+' · PWRC '+f3(d._pwrc)
          +' · Z-RMSE '+(d._z!=null?(+d._z).toFixed(2):'—');}}),
      grid:{left:46,right:10,top:12,bottom:44},
      xAxis:{type:'category',data:bands,
        axisLine:{lineStyle:{color:t.axis}},axisTick:{alignWithLabel:true,lineStyle:{color:t.axis}},
        axisLabel:{color:t['text-secondary'],fontSize:9.5}},
      yAxis:Object.assign(axStyle(),{scale:false,min:-1,max:1}),
      dataZoom:[{type:'inside',xAxisIndex:0},dzSlider({xAxisIndex:0})],
      series:bseries};
    const bw=el('div',{style:'overflow-x:auto'});
    bw.append(mountChart('band',Math.max(680,Math.min(1250,bands.length*(banded.length*8+22)+90)),240,bandOption));
    host.append(bw);
    // ---- the NUMBERS behind those bars: cross-bake per-band SROCC table.
    // Columns = bands populated somewhere in this corpus (n>0); on CID22 that drops the
    // structurally-empty B0/B1. Values come straight from rank.<corpus>.bands[] — nothing
    // is recomputed here (the fulleval JSON, i.e. zenstats, owns every statistic).
    const bandN=i=>Math.max(...banded.map(b=>{const r=b.rank[c].bands[i];return r&&r.n!=null?r.n:0;}));
    const bandSpan=i=>{const r=banded[0].rank[c].bands[i];return r&&r.span!=null?r.span:null;};
    const bandNM=i=>{const r=banded[0].rank[c].bands[i];return r?r.not_measured_reason:null;};
    // SIGNED — the whole point of the re-cut. |.| hid inversions and rewarded them.
    const bandS=(b,i)=>{const r=b.rank[c].bands[i];return r&&r.srocc_signed!=null?r.srocc_signed:null;};
    const cols=bands.map((_,i)=>i).filter(i=>bandN(i)>0);
    const scored=cols.filter(i=>banded.some(b=>bandS(b,i)!=null));
    if(cols.length&&scored.length){
      // band-profile summary: who leads the top band vs the bottom band (the finding —
      // a bake can own the near-lossless end and trail at the low end, or vice versa).
      const lead=i=>{let best=null;banded.forEach(b=>{const v=bandS(b,i);
        if(v!=null&&(best===null||v>best.v))best={nm:b.name,v};});return best;};
      const span=i=>{const vs=banded.map(b=>bandS(b,i)).filter(v=>v!=null);
        return [Math.min.apply(null,vs),Math.max.apply(null,vs)];};
      const loI=scored[0],hiI=scored[scored.length-1];
      const loL=lead(loI),hiL=lead(hiI),loS=span(loI),hiS=span(hiI);
      const sum=el('div',{class:'cap',style:'margin:8px 0 3px'});
      const put=(t,b)=>sum.append(b?el('b',{text:t}):document.createTextNode(t));
      put('Band profile ('+banded.length+' bake'+(banded.length===1?'':'s')+' shown) — ');
      put('top band '+bands[hiI]+' (n='+bandN(hiI)+')',1);
      put(' spans '+f3(hiS[0])+' → '+f3(hiS[1])+', led by ');
      put(hiL.nm+' '+f3(hiL.v),1);
      put('.  ');
      put('bottom band '+bands[loI]+' (n='+bandN(loI)+')',1);
      put(' spans '+f3(loS[0])+' → '+f3(loS[1])+', led by ');
      put(loL.nm+' '+f3(loL.v),1);
      put('. ');
      put(hiL.nm===loL.nm
        ? 'Same bake leads both ends.'
        : 'Different leaders at the two ends — this is a band PROFILE, not one ranking.');
      host.append(sum);
      const bt=el('table',{});
      const bh=el('tr',{});
      bh.append(el('th',{class:'lbl',text:'bake'}));
      cols.forEach(i=>{const t=el('th',{text:bands[i]});
        const sp=bandSpan(i);
        t.append(el('div',{style:'font-weight:400;font-size:9px;color:var(--muted)',
          text:'n='+bandN(i)+(sp!=null?' · span '+(+sp).toFixed(3):'')}));
        if(bandNM(i))t.append(el('div',{style:'font-weight:400;font-size:9px;color:var(--muted)',
          text:'NOT MEASURED'}));
        bh.append(t);});
      bt.append(el('thead',{},bh));
      const bb=el('tbody',{});
      banded.forEach(b=>{
        const tr=el('tr',{});
        tr.append(nameInto(el('td',{class:'lbl'}),b));
        cols.forEach(i=>{
          const r=b.rank[c].bands[i],v=bandS(b,i);
          const nm=r?r.not_measured_reason:null;
          const td=el('td',{text:v==null?'—':(v>=0?'+':'')+v.toFixed(3)});
          if(v==null){td.style.color='var(--muted)';
            td.title=nm?('NOT MEASURED — '+nm):'not measured';}
          else if(v<0)td.style.color='var(--danger, #c0392b)';
          tr.append(td);
        });
        bb.append(tr);
      });
      bt.append(bb);
      makeSortable(bt);
      const btw=el('div',{style:'overflow-x:auto'});btw.append(bt);host.append(btw);
      host.append(el('div',{class:'cap',style:'margin-top:3px',html:
        'Read DOWN a column (which bake wins that band), never ACROSS one: band SROCC is '
        +'range-restricted, so every value runs low by construction and bands are not comparable '
        +'to each other. Values are SIGNED — a red negative is a band ordered BACKWARDS, which the '
        +'previous absolute-valued column could not show (it ranked models by how inverted their '
        +'top band was). An em-dash is NOT MEASURED, with the reason on hover: that band could not '
        +'clear the count/span floors, so it publishes no statistic. It is not a zero.'}));
    }
  }
  // Calibration curve (binned pred → mean target) for MOS corpora from per_pair.
  const mosRows=rows.filter(b=>b.scatter[c]&&b.scatter[c].mos&&b.scatter[c].mos.pts.length>30);
  if(mosRows.length){
    host.append(el('h3',{text:'Calibration — '+c}));
    host.append(el('div',{class:'cap',text:'Binned mean MOS per predicted-score bin (15 bins). A straight rising line = well-calibrated dial; plateaus = dead zones; non-monotone = mis-calibration.'}));
    const W=430,H=250,mL=40,mB=30,mT=10,mR=10;
    const svg=S('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,style:'background:var(--surface-1);border:1px solid var(--border);border-radius:6px'});
    let xmin=1e9,xmax=-1e9;mosRows.forEach(b=>b.scatter[c].mos.pts.forEach(p=>{if(p[0]<xmin)xmin=p[0];if(p[0]>xmax)xmax=p[0];}));
    const SX=v=>mL+(v-xmin)/(xmax-xmin||1)*(W-mL-mR),SY=v=>mT+(1-v)*(H-mT-mB);
    [0,0.5,1].forEach(g=>{svg.append(S('line',{x1:mL,y1:SY(g),x2:W-mR,y2:SY(g),stroke:cssv('--grid'),'stroke-width':.5}));
      svg.append(S('text',{x:mL-4,y:SY(g)+3,'text-anchor':'end','font-size':8.5,fill:cssv('--muted'),text:g.toFixed(1)}));});
    svg.append(S('text',{x:(mL+W-mR)/2,y:H-6,'text-anchor':'middle','font-size':9.5,fill:cssv('--text-secondary'),text:'predicted score'}));
    mosRows.forEach(b=>{
      const pts=b.scatter[c].mos.pts,NB=15,acc=Array.from({length:NB},()=>[0,0]);
      pts.forEach(([x,y])=>{let i=Math.min(NB-1,Math.floor((x-xmin)/(xmax-xmin||1)*NB));acc[i][0]+=y;acc[i][1]++;});
      const line=acc.map((a,i)=>a[1]>=3?[SX(xmin+(i+0.5)/NB*(xmax-xmin)),SY(a[0]/a[1])]:null).filter(Boolean);
      if(line.length>1)svg.append(S('polyline',{points:line.map(p=>p.join(',')).join(' '),fill:'none',stroke:color(b),'stroke-width':1.8,opacity:.9}));
    });
    host.append(svg);
  }
}

// ---- PER-CODEC DIAL CURVES (median dial score vs q per codec, per visible bake)
function renderDial(){
  const host=$('#dialsec');if(!host)return;host.innerHTML='';
  const bs=visBakes().filter(b=>b.dial&&b.dial.curves&&Object.keys(b.dial.curves).length);
  if(!bs.length)return;
  host.append(el('h2',{text:'Per-codec dial curves'}));
  host.append(el('div',{class:'cap',html:'Median dial score vs grid quality per codec family (across each family\u2019s '
    +'image ladders on the densified grid; jxl x-axis = butteraugli-distance mapped to q-equiv). A good dial rises '
    +'monotonically and spans low→high. Hover for each bake’s <b>p25 / p50 / p75</b> at that q plus the '
    +'per-codec mono%/tied% — a family can be broken while the pooled headline stays green. Wheel/slider zooms '
    +'the axes (marks stay constant size); double-click resets.'}));
  const codecs=[...new Set(bs.flatMap(b=>Object.keys(b.dial.curves)))].sort();
  const grid=el('div',{style:'display:flex;flex-wrap:wrap;gap:10px'});
  codecs.forEach(cd=>{
    const t=TH();
    const meta={};    // bake name -> per-codec mono/tied/ladders for the tooltip
    const dseries=[];
    bs.forEach(b=>{
      const cv=b.dial.curves[cd];if(!cv||cv.length<2)return;
      const pc=(b.dial.per_codec||[]).find(x=>x.codec===cd);
      if(pc)meta[b.name]='mono '+pct(pc.mono)+' · tied '+pct(pc.tied)+' · '+pc.n_curves+' ladders';
      dseries.push({type:'line',name:b.name,showSymbol:false,symbol:'circle',symbolSize:6,
        lineStyle:{width:2,color:color(b),opacity:.9},itemStyle:{color:color(b)},
        emphasis:{focus:'series'},
        data:cv.map(p=>({value:[p[0],p[2]],p25:p[1],p75:p[3]}))});
    });
    if(!dseries.length)return;
    const dialOption={animation:false,
      title:{text:cd,top:4,left:10,textStyle:{color:t['text-primary'],fontSize:11,fontWeight:700}},
      tooltip:Object.assign(ttStyle(),{trigger:'axis',
        axisPointer:{type:'cross',label:{backgroundColor:t['surface-1'],color:t['text-primary'],fontSize:9},
          crossStyle:{color:t.muted},lineStyle:{color:t.muted}},
        formatter:ps=>{if(!ps||!ps.length)return'';
          let s='<b>'+cd+'</b> q='+f2(ps[0].axisValue);
          ps.forEach(p=>{const d=p.data||{};
            s+='<br>'+(p.marker||'')+p.seriesName+' p50 <b>'+f2(d.value?d.value[1]:null)+'</b>'
              +' <span style="opacity:.75">[p25 '+f2(d.p25)+' · p75 '+f2(d.p75)+']</span>'
              +(meta[p.seriesName]?'<br><span style="opacity:.6;font-size:10px">'+meta[p.seriesName]+'</span>':'');});
          return s;}}),
      grid:{left:44,right:12,top:30,bottom:44},
      xAxis:Object.assign(axStyle(),{name:''}),
      yAxis:Object.assign(axStyle(),{scale:false,min:0,max:100}),
      dataZoom:[{type:'inside',xAxisIndex:0,filterMode:'none'},
                {type:'inside',yAxisIndex:0,filterMode:'none'},
                dzSlider({xAxisIndex:0,filterMode:'none'})],
      series:dseries};
    grid.append(mountChart('dial',360,285,dialOption));
  });
  host.append(grid);
}

// ---- GATE SCORECARD (CODEC_TARGET_GOALS soft-gates per bake)
function renderGates(){
  const host=$('#gates');if(!host)return;host.innerHTML='';
  const bs=visBakes().filter(b=>b.gates&&Object.keys(b.gates).length);
  if(!bs.length)return;
  host.append(el('h2',{text:'Gate scorecard'}));
  host.append(el('div',{class:'cap',html:'CODEC_TARGET_GOALS soft-gates (1.00 = full pass). <b>weighted</b> = the '
    +'shippability gate (G1·3 + G8·2.5 + G5·1.5 + G9·1 + G-IM26·1 + G-NP·1 + G7·0.5 + G-OR·0.5) — a DIFFERENT '
    +'question from the ranking composite. G-OR is the catastrophe floor (worst-corpus outlier ratio). '
    +'Click a header to sort.'}));
  const KEYS=[['g1_dynamic_range','G1 range'],['g5_hf_rank','G5 HF'],['g7_cid22','G7 CID22'],['g8_zrmse','G8 Z-RMSE'],
    ['g9_ds_auc','G9 DS-AUC'],['g_np_nonphoto','G-NP'],['g_im26_realcodec','G-IM26'],['g_or_catastrophe','G-OR'],['weighted_goal','weighted']];
  const tbl=el('table',{});
  const thead=el('tr',{});thead.append(el('th',{class:'lbl',text:'bake'}));
  KEYS.forEach(([,h])=>thead.append(el('th',{text:h})));
  tbl.append(el('thead',{},thead));
  const tb=el('tbody',{});
  bs.sort((a,b)=>(b.gates.weighted_goal||0)-(a.gates.weighted_goal||0)).forEach(b=>{
    const tr=el('tr',{});
    tr.append(nameInto(el('td',{class:'lbl'}),b));
    KEYS.forEach(([k])=>{
      const v=b.gates[k];
      const td=el('td',{text:v!=null?v.toFixed(2):'—'});
      if(v!=null){const t=Math.max(0,Math.min(1,v));
        td.style.background='color-mix(in srgb, var(--seq-hi) '+Math.round(t*55)+'%, var(--surface-1))';
        if(t>.65)td.style.color='#fff';}
      tr.append(td);
    });
    tb.append(tr);
  });
  tbl.append(tb);
  makeSortable(tbl);
  const wrap=el('div',{style:'overflow-x:auto'});wrap.append(tbl);host.append(wrap);
}

// ---- JXL LOOP TARGETING (2-shot / 3-shot) — fed by the jxl-encoder exact-sweep summary
// JSON (READ verbatim; the jxl-encoder analyze script is the stats owner). Shows EVERY
// loop model, including the outer arms + ssim2 which are not bakes on this board.
function renderRecipes(){
  // Training-recipe table (2026-08-27, user ask): the theory of a bake = its
  // TARGET + data LEGS; capacity = hidden/width/epochs x pairs. Reading down a
  // column IS the training diff between any two visible bakes. Dates come from
  // the embedded zentrain.repro timestamp; '*' marks a bake-file-mtime fallback.
  const host=$('#recipes');if(!host)return;host.innerHTML='';
  const bs=DATA.bakes.filter(b=>state.visible.has(b.name));
  if(!bs.length)return;
  host.append(el('h2',{text:'Training recipes — dates, theory, capacity'}));
  host.append(el('div',{class:'cap',html:'<b>theory</b> = target column + data legs (name:width, k-rows); '
    +'<b>capacity</b> = hidden layers, feature width, epochs\u00d7pairs. Differences between bakes read '
    +'straight down each column. trained\u00a0<b>*</b> = bake-file date (pre-repro bake, no embedded '
    +'training timestamp). em-dash = no embedded repro. Click a header to sort.'}));
  const tbl=el('table',{});
  const thead=el('tr',{});
  ['bake','trained','target','legs','hidden','width','epochs\u00d7pairs','seed','best_val','extras'].forEach(h=>thead.append(el('th',{text:h,class:h==='bake'?'lbl':''})));
  tbl.append(el('thead',{},thead));
  const tb=el('tbody',{});
  bs.slice().sort((a,b)=>((b.train_date&&b.train_date.d)||'').localeCompare((a.train_date&&a.train_date.d)||'')).forEach(b=>{
    const r=b.recipe;const tr=el('tr',{});
    tr.append(nameInto(el('td',{class:'lbl'}),b));
    tr.append(el('td',{text:b.train_date?b.train_date.d+(b.train_date.src==='file'?'*':''):'\u2014'}));
    if(!r){for(let i=0;i<8;i++)tr.append(el('td',{text:'\u2014'}));tb.append(tr);return;}
    tr.append(el('td',{text:r.target||'\u2014'}));
    tr.append(el('td',{text:(r.legs&&r.legs.length)?r.legs.map(l=>l.name+(l.nf?(':'+l.nf):'')+(l.rows?('('+Math.round(l.rows/1000)+'k)'):'')).join(' + '):'\u2014'}));
    tr.append(el('td',{text:r.hidden!=null?String(r.hidden):'\u2014'}));
    tr.append(el('td',{text:r.width!=null?String(r.width):'\u2014'}));
    tr.append(el('td',{text:(r.epochs!=null&&r.pairs!=null)?(r.epochs+'\u00d7'+Math.round(r.pairs/1000)+'k'):'\u2014'}));
    tr.append(el('td',{text:r.seed!=null?String(r.seed):'\u2014'}));
    tr.append(el('td',{text:r.best_val!=null?Number(r.best_val).toFixed(4):'\u2014'}));
    tr.append(el('td',{text:(r.extras&&r.extras.length)?r.extras.join('; '):'\u2014'}));
    tb.append(tr);
  });
  tbl.append(tb);
  makeSortable(tbl);
  const wrap=el('div',{style:'overflow-x:auto'});wrap.append(tbl);host.append(wrap);
}

function renderCoverage(){
  const host=$('#loopcov');if(!host)return;host.innerHTML='';
  const rows=DATA.loopCoverage||[];if(!rows.length)return;
  host.append(el('h2',{text:'Codec-loop eval coverage'}));
  host.append(el('div',{class:'cap',html:'Which backend × steering-mode combinations HAVE an eval '
    +'(user directive 2026-08-29). Statuses: <b>measured</b> (committed record), <b>running</b> '
    +'(launched, lands on the next regen), <b>not-built</b> (mechanism absent — owner work), '
    +'<b>ruled-premature</b> (registered ruling). Numbers are READ from the owning records: '
    +'<code>benchmarks/loop_eval_coverage.json</code> (append-only).'}));
  const t=el('table',{});
  t.append(el('tr',{},[el('th',{text:'backend'}),el('th',{text:'mode'}),el('th',{text:'status'}),el('th',{text:'detail'})]));
  rows.forEach(r=>{
    const col={measured:'var(--seq-hi)',running:'var(--text-secondary)'}[r.status]||'#c0392b';
    t.append(el('tr',{},[el('td',{class:'lbl',text:r.backend}),el('td',{text:r.mode}),
      el('td',{html:'<b style="color:'+col+'">'+r.status+'</b>',title:r.src||''}),
      el('td',{text:r.detail})]));
  });
  t.style.fontSize='12px';host.append(t);
}
function renderLoop(){
  const host=$('#looptgt');if(!host)return;host.innerHTML='';
  if(!LT||!LT.models||!Object.keys(LT.models).length)return;
  const meta=LT.meta||{};const mx=meta.matrix||{};const N=ltN();
  host.append(el('h2',{text:'JXL loop targeting — 2-shot / 3-shot'}));
  host.append(el('div',{class:'cap',html:'Which model, driving the jxl-encoder zensim loop, HITS a requested target '
    +'in k encodes? '+(mx.refs||9)+' refs × targets {'+((mx.targets||[70,80,88]).join(', '))+'} = '+N+' cells; a cell '
    +'scores when the DECODED-judged result lands within ±'+(mx.within_tol!=null?mx.within_tol:2)+' of target '
    +'<b>in the arm’s OWN metric units</b> — rows are NOT unit-comparable across metrics. <b>k2/k3</b> = inner-loop '
    +'budget of 2/3 encodes; <b>emit-best</b> = best-scoring iterate kept (primary, what the scoreboard columns '
    +'show); <b>emit-last</b> = final iterate. Outer arms (<b>j2/j3</b>, marked °) re-encode outside the inner loop, '
    +'judged at outer_iter ≤ 2/3, and sit in the k2/k3 emit-last columns (an outer iterate IS its last emit). '
    +'Hover a cell for median |err|, median bytes and provenance (fresh run vs derived from a committed TSV). '
    +'Source: <code>'+(meta.source||'jxl-encoder benchmarks/zensim_loop_23shot_summary_2026-08-26.json')+'</code>.'}));
  const MODES=[['k2_emit_best','k2 emit-best'],['k3_emit_best','k3 emit-best'],['k2_emit_last','k2 emit-last'],['k3_emit_last','k3 emit-last']];
  const OUTER={k2_emit_last:'j2',k3_emit_last:'j3'};
  const cellOf=(m,mode)=>{const cs=m.cells||{};if(cs[mode])return cs[mode];
    if(m.kind==='outer'&&OUTER[mode]&&cs[OUTER[mode]])return cs[OUTER[mode]];return null;};
  const tbl=el('table',{});
  const h1=el('tr',{});
  h1.append(el('th',{class:'lbl',text:'loop model'}),el('th',{class:'lbl',text:'bake row'}));
  MODES.forEach(([,lab])=>{h1.append(el('th',{text:lab+' ±2'}),el('th',{text:'med|err|'}));});
  h1.append(el('th',{text:'med bytes (k3 best)'}));
  tbl.append(el('thead',{},h1));
  const tb=el('tbody',{});
  Object.entries(LT.models).forEach(([mk,m])=>{
    const tr=el('tr',{});
    const bakeName=(LT.modelBake&&LT.modelBake[mk])||null;
    const bk=bakeName?DATA.bakes.find(x=>x.name===bakeName):null;
    const nameTd=el('td',{class:'lbl'});
    if(bk)nameTd.append(el('span',{class:'sw',style:'display:inline-block;margin-right:5px;background:'+color(bk)}));
    nameTd.append(document.createTextNode(mk));
    tr.append(nameTd);
    tr.append(el('td',{class:'lbl',text:bakeName?bakeName:(m.bake?m.bake+' (bake not on board)':'(not a bake)')}));
    MODES.forEach(([mode])=>{
      const c=cellOf(m,mode);
      const outer=!!(m.kind==='outer'&&!(m.cells||{})[mode]&&c);
      const tdA=el('td',{text:c&&c.within2!=null?(c.within2+'/'+(c.n_cells||N)+(outer?'°':'')):'—'});
      if(c&&c.within2!=null){const t=Math.max(0,Math.min(1,c.within2/(c.n_cells||N)));
        tdA.style.background='color-mix(in srgb, var(--seq-hi) '+Math.round(t*62)+'%, var(--surface-1))';
        if(t>.6)tdA.style.color='#fff';}
      if(c){const tip='<b>'+mk+'</b> '+mode+(outer?' (outer '+OUTER[mode]+')':'')
        +'<br>within ±2: <b>'+c.within2+'/'+(c.n_cells||N)+'</b>'
        +'<br>med|err| '+(c.med_abs_err!=null?(+c.med_abs_err).toFixed(2):'—')
        +' · med bytes '+(c.med_bytes!=null?Math.round(c.med_bytes/1024)+' KB':'—')
        +(c.provenance?'<br>'+c.provenance:'');
        tdA.addEventListener('mousemove',ev=>showTip(tip,ev));tdA.addEventListener('mouseleave',hideTip);}
      tr.append(tdA);
      tr.append(el('td',{text:c&&c.med_abs_err!=null?(+c.med_abs_err).toFixed(2):'—'}));
    });
    const c3=cellOf(m,'k3_emit_best')||cellOf(m,'k3_emit_last');
    tr.append(el('td',{text:c3&&c3.med_bytes!=null?Math.round(c3.med_bytes/1024)+' KB':'—'}));
    tb.append(tr);
  });
  tbl.append(tb);
  makeSortable(tbl);
  const wrap=el('div',{style:'overflow-x:auto'});wrap.append(tbl);host.append(wrap);
  if(meta.notes)host.append(el('div',{class:'cap',text:'notes: '+meta.notes}));}

// ---- HF-NL AXIS PANEL (appendix O, 2026-08-05). HA is the committed axis-study JSON:
// per-model per-reference SROCC histograms + means/CIs, the reference/ceiling context
// rows, and the registered axis LSD. Every value is READ verbatim from the JSON —
// nothing is recomputed here (owner: the appendix-O battery over panel --batch).
const HA=DATA.hfnlAxis||null;
function renderHfnl(){
  const host=$('#hfnlsec');if(!host)return;host.innerHTML='';
  if(!HA||!HA.models||!HA.models.length)return;
  const meta=HA._meta||{};const lsd=meta.axis_lsd||{};const edges=meta.hist_edges||[];
  host.append(el('h2',{text:'HF-NL axis — per-reference SROCC distributions'}));
  host.append(el('div',{class:'cap',html:'Each row is one model: the shaded strip is its distribution of '
    +'per-reference signed SROCC over the '+(HA.models[0].n_groups||757)+' hfnlproxy references (bar height ∝ '
    +'√count per 0.05 bin), the tick + whisker its mean and 95% bootstrap CI, the small open circle its mean on '
    +'the historical <b>118-ref non-avif subset</b> (the pre-2026-08-05 sidecar coverage). The dashed reference '
    +'lines are <b>full-corpus</b> per-ref means (avif reference-metric coverage was backfilled 2026-08-05 — '
    +'compare them to the ticks directly; the old subset values are retained in the JSON as subset_mean). '
    +'<b>Axis LSD '+(lsd.median!=null?(+lsd.median).toFixed(3):'—')+'</b> (p90 '
    +(lsd.p90!=null?(+lsd.p90).toFixed(3):'—')+', drawn bottom-right): mean differences under it are '
    +'ref-sampling noise. Split-half model-ranking reliability SROCC '
    +(meta.split_half&&meta.split_half.srocc_mean!=null?(+meta.split_half.srocc_mean).toFixed(3):'—')
    +' (Spearman–Brown '+(meta.split_half&&meta.split_half.srocc_sb!=null?(+meta.split_half.srocc_sb).toFixed(3):'—')
    +') — the axis ordering is reliable. Ensembles carry corrected scoreboard means but no distribution here '
    +'(the instrument loads one ZNPR). 80 pre-pin board cells were sign-flipped and are REPAIRED '
    +'(registry id hfnl-preauto-orientation-flip-REPAIRED). Study: '
    +'<code>benchmarks/hfnl_axis_report_2026-08-05.md</code>.'}));
  const ms=HA.models.slice().sort((a,b)=>b.per_ref_mean-a.per_ref_mean);
  const vis=new Set(visBakes().map(b=>b.name));
  const W=980,rowH=24,padL=252,padR=26,padT=40,padB=46;
  const H=padT+ms.length*rowH+padB;
  const x=v=>padL+(Math.max(-1,Math.min(1,v))+1)/2*(W-padL-padR);
  const svg=el('svg:svg',{viewBox:'0 0 '+W+' '+H,style:'width:100%;max-width:'+W+'px;height:auto;display:block'});
  // grid + axis labels
  [-1,-0.5,0,0.5,1].forEach(v=>{
    svg.append(el('svg:line',{x1:x(v),x2:x(v),y1:padT-6,y2:H-padB+4,
      stroke:'var(--grid, #8884)','stroke-width':v===0?1.4:0.6,'stroke-dasharray':v===0?'':'3 3'}));
    svg.append(el('svg:text',{x:x(v),y:H-padB+18,'text-anchor':'middle',fill:'var(--muted)','font-size':11,
      text:(v>0?'+':'')+v}));});
  svg.append(el('svg:text',{x:(padL+W-padR)/2,y:H-padB+34,'text-anchor':'middle',fill:'var(--muted)',
    'font-size':11,text:'per-reference signed SROCC vs the ssim2 target (quality-oriented; + = orders ladders like ssim2)'}));
  // reference/ceiling context lines (SUBSET values)
  (HA.reference_rows||[]).forEach((r,i)=>{
    const dash=r.kind==='ceiling'?'':'5 4';
    svg.append(el('svg:line',{x1:x(r.mean),x2:x(r.mean),y1:padT-6,y2:H-padB+4,
      stroke:'var(--accent, #b8860b)','stroke-width':1,'stroke-dasharray':dash,opacity:0.75}));
    svg.append(el('svg:text',{x:x(r.mean),y:padT-10-(i%2)*11,'text-anchor':'middle',
      fill:'var(--accent, #b8860b)','font-size':10,text:r.display+' '+(+r.mean).toFixed(2)}));});
  // rows
  ms.forEach((m,i)=>{
    const cy=padT+i*rowH+rowH/2;
    const bk=DATA.bakes.find(b=>b.name===m.name);
    const col=bk?color(bk):'var(--muted)';
    const dim=bk&&!vis.has(m.name);
    const g=el('svg:g',{opacity:dim?0.35:1});
    if(i%2)g.append(el('svg:rect',{x:padL,y:cy-rowH/2,width:W-padL-padR,height:rowH,
      fill:'var(--surface-1, #8881)',opacity:0.5}));
    const mx=Math.max.apply(null,m.hist.concat([1]));
    m.hist.forEach((c,bi)=>{if(!c||bi+1>=edges.length)return;
      const x0=x(edges[bi]),x1=x(edges[bi+1]);
      const h=Math.max(2,Math.sqrt(c/mx)*(rowH-8));
      g.append(el('svg:rect',{x:x0,y:cy-h/2,width:Math.max(1,x1-x0-0.6),height:h,fill:col,opacity:0.5}));});
    if(m.ci)g.append(el('svg:line',{x1:x(m.ci[0]),x2:x(m.ci[1]),y1:cy,y2:cy,stroke:col,'stroke-width':2}));
    g.append(el('svg:line',{x1:x(m.per_ref_mean),x2:x(m.per_ref_mean),y1:cy-rowH/2+3,y2:cy+rowH/2-3,
      stroke:col,'stroke-width':2.5}));
    if(m.subset_mean!=null)g.append(el('svg:circle',{cx:x(m.subset_mean),cy:cy,r:3.2,fill:'none',
      stroke:col,'stroke-width':1.4}));
    g.append(el('svg:text',{x:padL-8,y:cy+4,'text-anchor':'end',fill:'var(--ink, currentColor)','font-size':11.5,
      text:(m.display||m.name)+'  '+(m.per_ref_mean>=0?'+':'')+(+m.per_ref_mean).toFixed(3)}));
    const hit=el('svg:rect',{x:0,y:cy-rowH/2,width:W,height:rowH,fill:'transparent'});
    const tip='<b>'+m.name+'</b>'+(m.family?' · '+m.family:'')
      +'<br>per-ref mean <b>'+(m.per_ref_mean>=0?'+':'')+(+m.per_ref_mean).toFixed(4)+'</b>'
      +(m.ci?' [CI '+(+m.ci[0]).toFixed(3)+', '+(+m.ci[1]).toFixed(3)+']':'')
      +'<br>'+m.n_groups+' refs · '+Math.round((m.frac_negative||0)*100)+'% refs backwards'
      +(m.subset_mean!=null?'<br>subset (non-avif, vs reference lines): '+(m.subset_mean>=0?'+':'')+(+m.subset_mean).toFixed(3):'')
      +(m.wide_band_mean!=null?'<br>wide-band refs only: '+(m.wide_band_mean>=0?'+':'')+(+m.wide_band_mean).toFixed(3):'');
    hit.addEventListener('mousemove',ev=>showTip(tip,ev));hit.addEventListener('mouseleave',hideTip);
    g.append(hit);
    svg.append(g);
  });
  // LSD scale bar (bottom right)
  if(lsd.median!=null){
    const y0=H-10;const x1=W-padR,x0=x1-(lsd.median/2)*(W-padL-padR);
    svg.append(el('svg:line',{x1:x0,x2:x1,y1:y0,y2:y0,stroke:'var(--ink, currentColor)','stroke-width':2}));
    [x0,x1].forEach(xx=>svg.append(el('svg:line',{x1:xx,x2:xx,y1:y0-4,y2:y0+4,
      stroke:'var(--ink, currentColor)','stroke-width':1.2})));
    svg.append(el('svg:text',{x:x0-6,y:y0+4,'text-anchor':'end',fill:'var(--muted)','font-size':10,
      text:'axis LSD '+(+lsd.median).toFixed(3)+' (p90 '+(+lsd.p90).toFixed(3)+')'}));}
  const wrap=el('div',{style:'overflow-x:auto'});wrap.append(svg);host.append(wrap);
}

// ---- MODEL DETAILS (architecture + in/out modifiers per bake, from the ZNPR itself)
// n_feature_transforms = the TRUE transform count (the embed is capped at 48 chips —
// MODEL_TRANSFORMS_EMBED in the builder; the fulleval JSON on disk keeps the full list).
const nft=m=>(m.n_feature_transforms!=null?m.n_feature_transforms:(m.feature_transforms||[]).length);
function renderModels(){
  const host=$('#models');if(!host)return;host.innerHTML='';
  const bs=visBakes().filter(b=>b.model&&b.model.layers);
  if(!bs.length)return;
  host.append(el('h2',{text:'Model details'}));
  host.append(el('div',{class:'cap',html:'Read from each bake\u2019s ZNPR (structured <code>zenpredict inspect</code>): '
    +'architecture, weight dtype, INPUT modifiers (per-feature transforms + winsor guard count + scaler), and the '
    +'OUTPUT modifier — the dial calibration spline (plotted raw→dial; the top cap at 100 and any negative-tail '
    +'extension are visible in the knots). Hover a transform chip for its params.'}));
  const grid=el('div',{style:'display:flex;flex-wrap:wrap;gap:12px;align-items:stretch'});
  bs.forEach(b=>{
    const m=b.model;
    const card=el('div',{style:'border:1px solid var(--border);border-radius:8px;padding:10px 12px;'
      +'background:var(--surface-1);min-width:300px;max-width:360px;flex:1 1 300px'});
    const hd=el('div',{style:'display:flex;align-items:center;gap:6px;margin-bottom:6px;flex-wrap:wrap'});
    hd.append(el('span',{class:'sw',style:'display:inline-block;background:'+color(b)}),
      el('b',{text:b.name}));
    const hbd=ensBadge(b);if(hbd)hd.append(hbd);
    card.append(hd);
    // An ensemble has no single ZNPR: everything below (arch, size, transforms,
    // repro, spline) is the ANCHOR member. Say so before the numbers, not after.
    if(isEns(b)){
      const mem=(m.member_names||[]);
      const note=el('div',{style:'font-size:10px;line-height:1.4;margin:-2px 0 7px;padding:5px 7px;'
        +'border-radius:5px;background:color-mix(in srgb, var(--warn) 16%, var(--surface-1));'
        +'border:1px solid var(--border)'});
      note.append(el('b',{text:'Equal-weight ensemble of '+ensK(b)+' bakes.'}),
        document.createTextNode(' Not a shippable artifact — the fields below describe the '
          +'ANCHOR member '+(m.anchor||'?')+' only, and M3/M3a are NOT COMPUTABLE for an ensemble '
          +'(the coherence instrument loads one ZNPR). Distillation to a single bake is pending.'));
      if(mem.length){
        const det=el('details',{style:'margin-top:4px'});
        det.append(el('summary',{style:'font-size:9.5px;cursor:pointer;opacity:.75',
          text:'members ('+mem.length+')'}),
          el('div',{style:'font-size:9px;word-break:break-all;opacity:.85',text:mem.join(', ')}));
        note.append(det);
      }
      card.append(note);
    }
    // Full dim chain with hidden sizes: "720 → 128 (LeakyRelu) → 1", trainer-log style.
    // Identity on the last layer is the plain linear output head — omit the label.
    const L=m.layers||[];
    let arch=L.length?String(L[0].in):'—';
    let nparams=0;
    L.forEach((l,i)=>{
      const act=(l.activation==='Identity'&&i===L.length-1)?'':' ('+l.activation+(l.dtype!=='f32'?' '+l.dtype:'')+')';
      arch+=' → '+l.out+act;
      nparams+=l.in*l.out+l.out;
    });
    const kb=m.file_bytes?(m.file_bytes/1024).toFixed(1)+' KB':'—';
    const lines=[
      ['arch', arch+(nparams?'  ·  '+(nparams>=1000?(nparams/1000).toFixed(1)+'k':nparams)+' params':'')],
      ['size / ZNPR', kb+' · v'+(m.znpr_version||'?')],
      ['inputs', m.n_inputs+' feats · scaler '+(m.scaler&&m.scaler.present?('z-norm ('+m.scaler.n+')'):'none')],
      ['in-mods', (m.feature_transforms&&m.feature_transforms.length?nft(m)+' transforms':'none')
        +' · '+(m.n_feature_bounds||0)+' winsor bounds'],
      ['heads', ['per_sample_alpha','hybrid','minmax'].filter(k=>m.heads&&m.heads[k]).join(', ')
        +((m.heads&&m.heads.tanh_pin_scale!=null)?' tanh-pin '+m.heads.tanh_pin_scale:'')||'none'],
      ['out-mods', (m.output_spline?('spline '+m.output_spline.n_knots+' knots'):'no spline')
        +(m.n_output_specs?(' · '+m.n_output_specs+' output_specs'):'')
        +(m.n_discrete_sets?(' · '+m.n_discrete_sets+' discrete'):'')],
    ];
    // block usage (static fingerprint from bake bytes — bake_block_profile;
    // f156-371 were ZEROED by the folded 924/944 regimes, slots preserved per
    // the append-only discipline, not removed). used/total encoder columns
    // with nonzero norm per family; ensembles: the ANCHOR member's bake.
    if(b.blocks&&b.blocks.fams){
      const ord=['f0_155','f156_371','f372_719','f720_943'];
      const parts=ord.filter(k=>b.blocks.fams[k]).map(k=>{
        const uc=b.blocks.fams[k];
        return k.replace('_','-')+' '+uc[0]+'/'+uc[1]+(k==='f156_371'&&uc[0]===0?' (zeroed)':'');
      });
      lines.push(['blocks', parts.join(' · ')+(isEns(b)?'  (anchor member)':'')
        +(b.blocks.uses156?'  · USES f156-371':'')]);
    }
    const tb=el('table',{style:'font-size:11px;width:100%'});
    lines.forEach(([k,v])=>{const tr=el('tr',{});
      tr.append(el('td',{class:'lbl',style:'opacity:.65;padding-right:8px;white-space:nowrap',text:k}),
        el('td',{text:v}));tb.append(tr);});
    card.append(tb);
    // transform chips (hover = kind + params)
    if(m.feature_transforms&&m.feature_transforms.length){
      const chips=el('div',{style:'display:flex;flex-wrap:wrap;gap:3px;margin-top:6px'});
      m.feature_transforms.slice(0,48).forEach(t=>{
        const ch=el('span',{style:'font-size:9.5px;padding:1px 5px;border-radius:8px;'
          +'background:color-mix(in srgb, var(--seq-hi) 18%, var(--surface-1));border:1px solid var(--border)',
          text:'f'+t.idx});
        ch.addEventListener('mousemove',ev=>showTip('<b>f'+t.idx+'</b> '+t.kind+'<br>params ['+(t.params||[]).map(p=>(+p).toPrecision(4)).join(', ')+']',ev));
        ch.addEventListener('mouseleave',hideTip);
        chips.append(ch);
      });
      if(nft(m)>48)chips.append(el('span',{style:'font-size:9.5px;opacity:.6',text:'+'+(nft(m)-48)+' more'}));
      card.append(chips);
    }
    // Reproduction provenance: source badge + seed/commit + input-parquet
    // chips (hover = path + sha256 prefix + rows) + argv in a collapsible.
    {
      const r=b.repro;
      const rep=el('div',{style:'margin-top:7px;border-top:1px dashed var(--border);padding-top:6px'});
      const badge=(txt,tone)=>el('span',{style:'font-size:9px;font-weight:700;letter-spacing:.04em;'
        +'padding:1px 6px;border-radius:8px;margin-right:6px;background:'+tone,text:txt});
      if(!r){
        rep.append(badge('NO REPRO','color-mix(in srgb, var(--serious) 30%, var(--surface-1))'),
          el('span',{style:'font-size:10px;opacity:.7',text:'no embedded zentrain.repro, no .spec.json — irreproducible without archaeology'}));
      }else{
        const emb=r.source==='embedded';
        rep.append(badge(emb?'REPRO: EMBEDDED':'REPRO: SIDECAR',
          emb?'color-mix(in srgb, var(--good) 25%, var(--surface-1))':'color-mix(in srgb, var(--warn) 25%, var(--surface-1))'));
        const bits=[];
        if(r.seed!=null)bits.push('seed '+r.seed);
        if(r.epochs!=null)bits.push(r.epochs+' ep');
        if(r.trainer_head_at_train)bits.push('@'+r.trainer_head_at_train);
        if(r.timestamp_epoch)bits.push(new Date(r.timestamp_epoch*1000).toISOString().slice(0,10));
        rep.append(el('span',{style:'font-size:10px;opacity:.85',text:bits.join(' · ')}));
        const ins=r.inputs||[];
        if(ins.length){
          const chips=el('div',{style:'display:flex;flex-wrap:wrap;gap:3px;margin-top:4px'});
          ins.forEach(inp=>{
            const ch=el('span',{style:'font-size:9.5px;padding:1px 6px;border-radius:8px;'
              +'background:color-mix(in srgb, var(--good) 12%, var(--surface-1));border:1px solid var(--border)',
              text:inp.name+(inp.rows?' ('+(inp.rows>=1000?Math.round(inp.rows/1000)+'k':inp.rows)+')':'')});
            ch.addEventListener('mousemove',ev=>showTip('<b>'+inp.name+'</b><br>'+(inp.path||'?')
              +'<br>sha256 '+String(inp.sha256||'?').slice(0,16)+'… · '+(inp.rows||'?')+' rows',ev));
            ch.addEventListener('mouseleave',hideTip);
            chips.append(ch);
          });
          rep.append(chips);
        }
        if(r.argv&&r.argv.length){
          const det=el('details',{style:'margin-top:4px'});
          det.append(el('summary',{style:'font-size:9.5px;cursor:pointer;opacity:.7',text:'reproduction command (argv)'}),
            el('pre',{style:'font-size:9px;white-space:pre-wrap;word-break:break-all;max-height:120px;'
              +'overflow-y:auto;background:var(--plane);padding:5px;border-radius:4px',text:r.argv.join(' ')}));
          rep.append(det);
        }
      }
      card.append(rep);
    }
    // spline mini-plot: raw pred (x) -> dial score (y)
    if(m.output_spline&&m.output_spline.xs&&m.output_spline.xs.length>1){
      const xs=m.output_spline.xs,ys=m.output_spline.ys;
      const W=300,H=110,mL=30,mB=18,mT=6,mR=6;
      const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(0,...ys),y1=Math.max(100,...ys);
      const SX=v=>mL+(v-x0)/(x1-x0||1)*(W-mL-mR),SY=v=>mT+(y1-v)/(y1-y0||1)*(H-mT-mB);
      const svg=S('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,style:'margin-top:6px;background:var(--plane);border-radius:5px;max-width:100%'});
      [0,50,100].forEach(g=>{if(g>=y0&&g<=y1){svg.append(S('line',{x1:mL,y1:SY(g),x2:W-mR,y2:SY(g),stroke:cssv('--grid'),'stroke-width':.5}));
        svg.append(S('text',{x:mL-3,y:SY(g)+3,'text-anchor':'end','font-size':7.5,fill:cssv('--muted'),text:String(g)}));}});
      svg.append(S('polyline',{points:xs.map((x,i)=>SX(x)+','+SY(ys[i])).join(' '),fill:'none',stroke:color(b),'stroke-width':1.6}));
      xs.forEach((x,i)=>svg.append(S('circle',{cx:SX(x),cy:SY(ys[i]),r:1.6,fill:color(b)})));
      svg.append(S('text',{x:(mL+W-mR)/2,y:H-4,'text-anchor':'middle','font-size':7.5,fill:cssv('--muted'),text:'output spline: raw → dial'}));
      card.append(svg);
    }
    grid.append(card);
  });
  host.append(grid);
}

// ================= FAILURE PROFILE ==========================================
// "Where does this model hurt." Every row below is a FORMATTING of a number the
// verdict already carries — SROCC signs, per-reference frac_negative, the stored
// zone counts/rates, dial mono/tied/p5/p95, corruption pass rates, M3/M3a. The
// only logic here is (a) thresholding a stored number into a severity and (b)
// naming the production situation that number predicts. No statistic is
// computed on the page; a value the verdict does not carry renders as NOT
// MEASURED with its reason, never as a blank and never as a zero.
const ZK=DATA.zoneRowKeys||['n_pairs','inv_material','flat','codec_sat','n_ladders',
  'ladders_with_inv','ladders_ends_backwards','inv_rate','frac_ladders_with_inv',
  'frac_ladders_ends_backwards','inv_mag_med','inv_mag_max'];
const ZI={};ZK.forEach((k,i)=>{ZI[k]=i;});
const zrow=(b,split,key,zone)=>{const z=b.zones;if(!z||!z.rows)return null;
  return z.rows[split+'|'+key+'|'+zone]||null;};
const zv=(row,k)=>row?row[ZI[k]]:null;
const ZONES=['q<50','q50-85','q>=85'];
const ZONE_MEANING={'q<50':'aggressive web compression',
  'q50-85':'ordinary web quality','q>=85':'high-fidelity / near-lossless (q90+ pipelines)'};
// What each corpus PREDICTS about production. Used to turn "nonphoto 0.845" into
// a sentence about the content a user would actually ship.
const SITUATION={
  cid22:'human MOS on real codec output — the closest stand-in for a user looking at an encode',
  nonphoto:'non-photographic content (UI, illustration, screenshots, text) judged by ssim2',
  imazen26:'the real-codec web corpus judged by ssim2',
  hfnlproxy:'high-fidelity / near-lossless encodes — where a q90+ web pipeline operates',
  hf_nearlossless:'high-fidelity / near-lossless encodes',
  konjnd:'the just-noticeable-difference threshold — the visually-lossless boundary',
  kadid:'synthetic distortion families (blur, noise, blocking, colour)',
  tid:'synthetic distortion families (TID2013)',
  csiq:'classic photographic distortion sets',
  live:'classic photographic distortion sets (LIVE)',
  aic3:'AIC-3 near-threshold JND triplets — fine discrimination just above visibility',
  aic4:'AIC-4 near-threshold JND triplets',
  kadis:'700k synthetic distortions scored by reference metrics',
  pipal:'GAN / restoration outputs',
  sdr25:'the JPEG-AI SDR high-quality zone'};
const situ=c=>SITUATION[c]||(c+' pairs');
const SEV={blocker:0,serious:1,watch:2};
const sevTone={blocker:'var(--serious)',serious:'var(--warn)',watch:'var(--muted)'};
const pc=v=>v==null||!isFinite(v)?'—':(v*100).toFixed(v*100<10?1:0)+'%';

// One finding: what breaks / how big / where you meet it / evidence.
function F(sev,rank,what,size,where,ev){return {sev:sev,rank:rank,what:what,size:size,where:where,ev:ev};}

function failures(b){
  const bad=[],good=[],nm=[];
  const R=b.rank||{},D=b.dial||{};
  // ---- 1. a corpus ranked backwards outright ------------------------------
  Object.keys(R).forEach(c=>{
    const r=R[c];if(!r)return;const v=rs(b,c);
    if(v==null)return;
    if(v<0){
      bad.push(F('blocker',0,'Ranks '+c+' BACKWARDS',
        'pooled SROCC '+f3(v)+' over n='+(r.n||'?')+' pairs',
        situ(c),'rank.'+c+'.srocc_signed'));
    }
  });
  // ---- 2. whole reference ladders ranked backwards -------------------------
  // frac_negative is the share of REFERENCES whose within-image ranking is
  // inverted. It is the statistic a codec loop meets: the loop only ever
  // compares encodes of ONE image.
  Object.keys(R).forEach(c=>{
    const r=R[c];if(!r||r.frac_negative==null||r.per_ref_n==null)return;
    if(r.frac_negative<=0.02)return;
    // aic4/sdr25 are inverted for essentially EVERY board cell — a corpus
    // property, not a model finding (registry aic4-corpus-wide-per-ref-inversion).
    // Report it, but never as this model's defect.
    const corpusWide=(c==='aic4'||c==='sdr25');
    // A POSITIVE pooled SROCC sitting on a NEGATIVE per-reference mean is the
    // most misleading shape the board can show: the model separates images and
    // orders encodes of one image backwards, and only the pooled number is on
    // the scoreboard. Always a blocker.
    const flipped=(r.per_ref_mean!=null&&r.per_ref_mean<0&&!corpusWide);
    const sev=corpusWide?'watch':(flipped?'blocker':(r.frac_negative>=0.15?'serious':'watch'));
    bad.push(F(sev,corpusWide?11:(flipped?0:1),
      (flipped?'INVERTED per image on '+c+' while its pooled score looks healthy'
             :'Ranks whole reference ladders backwards on '+c),
      pc(r.frac_negative)+' of '+r.per_ref_n+' references (within-image mean SROCC '
        +f3(r.per_ref_mean)+' vs pooled '+f3(rs(b,c))+')',
      corpusWide?('a CORPUS-WIDE inversion, not evidence about this model: measured board-wide, '
        +(c==='aic4'?'median 60% of aic4 references are backwards across 373 cells'
                    :'median 20% of sdr25 references are backwards across 356 cells (sdr25 is a '
                     +'subset of aic4)'))
        :(flipped?('a per-image tuning loop on '+situ(c)+' — and the scoreboard’s pooled '
          +f3(rs(b,c))+' does not show it')
        :('a per-image tuning loop on '+situ(c))),
      'rank.'+c+'.frac_negative / per_ref_mean'));
  });
  // ---- 3+4. ladder inversions by codec x quality zone ----------------------
  if(b.zones&&b.zones.rows){
    const codecs=new Set();Object.keys(b.zones.rows).forEach(k=>{const p=k.split('|');
      if(p[0]==='codec')codecs.add(p[1]);});
    ZONES.forEach(z=>{
      Array.from(codecs).sort().forEach(cd=>{
        const row=zrow(b,'codec',cd,z);if(!row)return;
        const back=zv(row,'ladders_ends_backwards'),nl=zv(row,'n_ladders');
        if(back>0){
          bad.push(F('blocker',2,'Whole '+cd+' ladders run BACKWARDS at '+z,
            back+' of '+nl+' ladders ('+pc(zv(row,'frac_ladders_ends_backwards'))
              +'): the best-quality rung scores materially below the worst',
            'a codec loop tuning '+cd+' in the '+ZONE_MEANING[z]+' band walks the wrong way',
            'dial.zones codec|'+cd+'|'+z));
        }
        const ir=zv(row,'inv_rate');
        if(ir!=null&&ir>=0.02){
          bad.push(F(ir>=0.05?'serious':'watch',3,
            'Orders adjacent '+cd+' quality steps backwards at '+z,
            zv(row,'inv_material')+' of '+zv(row,'n_pairs')+' steps ('+pc(ir)
              +'), worst reversal '+f2(zv(row,'inv_mag_max'))+' dial pts; '
              +zv(row,'ladders_with_inv')+' of '+nl+' ladders affected',
            'raising '+cd+' quality inside the '+ZONE_MEANING[z]
              +' band can LOWER the reported score',
            'dial.zones codec|'+cd+'|'+z));
        }
      });
      // content class
      ['text_lineart','nonphoto','photo'].forEach(cl=>{
        const row=zrow(b,'class',cl,z);if(!row)return;
        const ir=zv(row,'inv_rate');const nimg=(b.zones.classes||{})[cl];
        if(ir!=null&&ir>=0.03){
          bad.push(F(ir>=0.06?'serious':'watch',4,
            'Orders '+cl.replace('_','/')+' quality steps backwards at '+z,
            zv(row,'inv_material')+' of '+zv(row,'n_pairs')+' steps ('+pc(ir)
              +') across '+(nimg!=null?nimg:'?')+' reference images, worst '
              +f2(zv(row,'inv_mag_max'))+' dial pts',
            (cl==='text_lineart'?'text / line-art content (scanned documents, sheet music, '
              +'high-contrast lettering)':cl==='nonphoto'?'rendered or synthetic content '
              +'(illustration, CGI, display captures)':'ordinary photographic content')
              +' in the '+ZONE_MEANING[z]+' band',
            'dial.zones class|'+cl+'|'+z));
        }
      });
    });
  }else{
    nm.push({what:'Ladder inversions by codec / quality zone / content class',
      why:(b.zoneSkip||'not measured for this cell')});
  }
  // ---- 5. dial dead zone ---------------------------------------------------
  if(D.tied_pct!=null&&D.tied_pct>0){
    bad.push(F(D.tied_pct>0.05?'serious':'watch',5,'Dial dead zone: distinct encodes score identically',
      pc(D.tied_pct)+' of adjacent quality steps move the score by <1e-9',
      'a loop cannot tell two different encodes apart there — it stops converging',
      'dial.tied_pct'));
  }
  // ---- 6. unreachable dial ends -------------------------------------------
  if(D.p5!=null&&D.p5>25){
    bad.push(F('serious',6,'Dial cannot reach low quality',
      'p5 of the whole codec grid is '+f2(D.p5)+' (gate G1 wants <=25)',
      'a user asking for a target below '+f2(D.p5)+' can never be served — the loop '
        +'saturates and returns the wrong bytes','dial.p5'));
  }
  if(D.p95!=null&&D.p95<85){
    bad.push(F('serious',6,'Dial cannot reach high quality',
      'p95 is '+f2(D.p95)+' (gate G1 wants >=85)',
      'near-lossless targets are unreachable','dial.p95'));
  }
  // ---- 7. corruption ordering ---------------------------------------------
  if(b.corruption&&b.corruption.pass_q20!=null){
    const p=b.corruption.pass_q20;
    if(p<0.6){
      // per_family ships in TWO shapes: bake cells carry a list of
      // {family,pass_rate,n}; the peer rows carry an object keyed by family
      // whose values hold pass_q10/pass_q20. Normalise before reading, and
      // read pass_q20 for the object form so both shapes name the SAME rate.
      const pfr=b.corruption.per_family;
      const fams=(Array.isArray(pfr)?pfr.filter(f=>f.pass_rate===0).map(f=>f.family)
        :(pfr&&typeof pfr==='object'?Object.keys(pfr).filter(k=>pfr[k]&&pfr[k].pass_q20===0):[]));
      bad.push(F(p<0.3?'serious':'watch',7,'Does not rank corrupted images below a bad encode',
        'passes '+pc(p)+' of '+(b.corruption.n_triples||'?')+' corruption triples'
          +(fams.length?'; '+fams.length+' families at 0% ('+fams.slice(0,4).join(', ')
            +(fams.length>4?', …':'')+')':''),
        'a truncated / mis-decoded / occluded image can be reported as good quality '
          +'and shipped','corruption.pass_q20 + per_family'));
    }
  }else{nm.push({what:'Corruption ordering',why:'no corruption block in this cell'});}
  // ---- 8. steering coherence ----------------------------------------------
  if(b.m3a!=null){
    if(b.m3a<0.85)bad.push(F(b.m3a<0.6?'serious':'watch',8,
      'Its attribution map disagrees with its own score',
      'M3a block coherence '+f3(b.m3a)+' (gate >=0.85)',
      'per-block steering points at the wrong blocks — a codec loop spends bits '
        +'where the metric will not reward them','m3a_coherence'));
  }else if(b.m3!=null&&b.m3<0.70){
    bad.push(F('watch',8,'Its deployable diffmap disagrees with its score',
      'M3 '+f3(b.m3)+' (M3a NOT MEASURED for this cell)',
      'the map a codec can use today does not track the scalar it is steering to',
      'm3_coherence'));
  }
  if(b.m3a==null&&b.m3==null)nm.push({what:'Steering coherence (M3 / M3a)',
    why:isEns(b)?'ensemble — the coherence instrument loads one ZNPR':'not measured for this cell'});
  // ---- 9. band tails -------------------------------------------------------
  Object.keys(R).forEach(c=>{
    const r=R[c];if(!r||!r.bands)return;
    r.bands.forEach(bd=>{
      if(bd.srocc_signed==null){
        nm.push({what:'Per-band ranking on '+c+' band '+bd.band,
          why:bd.not_measured_reason||'band not usable'});return;}
      if(bd.srocc_signed<0.09){
        bad.push(F(bd.srocc_signed<0?'serious':'watch',9,
          'Weak or inverted ranking inside one quality band of '+c,
          'band '+bd.band+' SROCC '+f3(bd.srocc_signed)+' (n='+bd.n+', MOS span '
            +f2(bd.span)+')',
          'choosing between two encodes that are already close in quality — '
            +'exactly the decision a near-lossless pipeline makes','rank.'+c+'.bands[]'));
      }
    });
  });
  // ---- 10. memorised corpora ----------------------------------------------
  const teq=Object.keys(R).filter(c=>R[c]&&R[c].train_eq_val);
  if(teq.length)bad.push(F('watch',10,'Some corpus numbers reward memorisation',
    teq.join(', ')+': train and validation pairs overlap',
    'those SROCCs are integrity guards, not evidence of generalisation','rank.*.train_eq_val'));
  // ---- 11. registry annotations -------------------------------------------
  (b.annotations||[]).forEach(id=>{
    bad.push(F('watch',11,'Registry annotation: '+id,
      annKind(id)==='invalidated'?'INVALIDATED — do not cite the affected number'
        :(annKind(id)==='absent-not-failed'?'the field is absent because the instrument was not '
          +'run here — absence is not a measured fail':'the number is real but must be read '
          +'with the stated caveat'),
      annReason(id),'eval_annotations.json '+id));
  });
  // ---- 12. reproducibility + root exposure --------------------------------
  if(!b.repro)bad.push(F('watch',12,'No reproduction provenance',
    'no embedded zentrain.repro and no .spec.json',
    'the model cannot be rebuilt or re-derived without archaeology','repro'));
  if(b.blocks&&b.blocks.uses156&&String(b.regime||'').indexOf('944')>=0){
    bad.push(F('serious',12,'Reads a feature block the folded roots zero',
      'the encoder has live weight in f156-371, which folded 944-class roots feed as '
        +'structural zeros',
      'scored at the wrong regime this model returns a plausible number from zeroed '
        +'inputs (the ebothg_m504 class)','block_profile.uses_f156_371'));
  }
  if(b.zones&&/dial_grid_372col_2026-05-29[.]parquet/.test(b.zones.grid||'')){
    bad.push(F('serious',12,'This cell dial was measured on the UN-quarantined 2026-05-29 grid',
      'that grid carries both documented defects — the 9 w11-corrupt masked/IW ladders and '
        +'the 33 JXL cells at butteraugli distance 0.025 encoded before jxl-encoder eeb52735, '
        +'where 66.7% of the 372 features GROW by 5-8 orders of magnitude from the q99.8 rung '
        +'to the q99.9 rung',
      'every ladder row above is INFLATED by the grid, not by the model: the ~19-point drop at '
        +'that rung is correct scoring of a broken encode. Read the @cur372 sibling cell, which '
        +'is cut on the quarantined_v2 grid',
      'dial.zones.grid + eval_annotations dial-cells-on-unquarantined-2026-05-29-grid'));
  }
  // ---- the honest inverse --------------------------------------------------
  Object.keys(R).forEach(c=>{
    const r=R[c],v=rs(b,c);if(!r||v==null)return;
    if(v>=0.85&&(r.frac_negative==null||r.frac_negative<=0.02)&&!r.train_eq_val)
      good.push('Ranks '+c+' well and consistently per image — SROCC '+f3(v)
        +(r.per_ref_n!=null?', '+pc(r.frac_negative)+' of '+r.per_ref_n
          +' references backwards':'')+' ('+situ(c)+')');
  });
  if(b.zones&&b.zones.rows){
    const codecs=new Set();Object.keys(b.zones.rows).forEach(k=>{const p=k.split('|');
      if(p[0]==='codec')codecs.add(p[1]);});
    Array.from(codecs).sort().forEach(cd=>{
      let tot=0,inv=0,back=0,ok=true;
      ZONES.forEach(z=>{const r=zrow(b,'codec',cd,z);if(!r){ok=false;return;}
        tot+=zv(r,'n_pairs');inv+=zv(r,'inv_material');back+=zv(r,'ladders_ends_backwards');});
      if(ok&&inv===0&&back===0&&tot>0)
        good.push('Never orders a '+cd+' quality step backwards — 0 of '+tot
          +' adjacent steps across all three quality zones');
    });
  }
  if(D.mono_pct!=null&&D.mono_pct>=0.99&&D.tied_pct===0)
    good.push('Dial is monotone with no dead zone — '+pc(D.mono_pct)
      +' of steps forward, 0% tied');
  if(b.m3a!=null&&b.m3a>=0.85)
    good.push('Attribution map agrees with the score (M3a '+f3(b.m3a)
      +') — safe to steer a codec per block');
  // NOT MEASURED that applies to every board cell
  nm.push({what:'Out-of-distribution / extrapolation exposure (G-RANGE: share of rows '
    +'predicting outside the spline knots)',
    why:'owned by bake_dial_refit gate, which is not run per board cell — see '
      +'benchmarks/add156_ship_audit_2026-08-31.md for the one model it has been run on'});
  nm.push({what:'Bytes saved at equal judged quality (G-RD) and target-hitting (G-TARGET)',
    why:'owned by the codec probe matrix; only mapped bakes carry the JXL loop panel'});
  bad.sort((x,y)=>(SEV[x.sev]-SEV[y.sev])||(x.rank-y.rank)||x.what.localeCompare(y.what));
  return {bad:bad,good:good,nm:nm};
}

function renderFailures(){
  const host=$('#failures');if(!host)return;host.innerHTML='';
  host.append(el('h2',{text:'Failure profile — what breaks, how big, where you meet it'}));
  host.append(el('div',{class:'cap',html:'Ranked by product impact, per model. Every row is a '
    +'number the verdict already carries, restated as the production situation it predicts — '
    +'nothing here is recomputed. <b>blocker</b> = a codec loop is led the wrong way; '
    +'<b>serious</b> = a gate the shipping design depends on; <b>watch</b> = qualified or '
    +'thin evidence. Ladder rows come from <code>dial.zones</code> '
    +'(<code>bake_verdict</code>, scheme <code>ladder-inversion-2026-08-31</code>); '
    +'reference rows from <code>rank.*.frac_negative</code>, the share of whole reference '
    +'ladders the model orders backwards. A cell with no measurement says NOT MEASURED and '
    +'why — never a blank, never a zero.'}));
  const bs=visBakes();
  if(!bs.length){host.append(el('div',{class:'cap',text:'no bakes visible'}));return;}
  // ---- side-by-side comparison table (the choosing-between-models view) ----
  const HEAD=['bake','blockers','serious','ladders backwards q>=85','ladder-inv q>=85',
    'ladder-inv q<50','refs bwd (HF-NL)','refs bwd (CID22)','dial dead-zone','corruption q20',
    'steer M3a','worst single reversal'];
  const tbl=el('table',{});
  const hr=el('tr',{});HEAD.forEach(h=>hr.append(el('th',{text:h})));
  const th=el('thead',{});th.append(hr);tbl.append(th);
  const tb=el('tbody',{});
  bs.forEach(b=>{
    const f=failures(b);
    const zAll=z=>zrow(b,'all','all',z);
    const hi=zAll('q>=85'),lo=zAll('q<50');
    const hfr=b.rank.hfnlproxy||b.rank.hf_nearlossless;
    const worst=['q<50','q50-85','q>=85'].map(z=>zv(zAll(z),'inv_mag_max'))
      .filter(v=>v!=null).reduce((a,v)=>Math.max(a,v),0);
    const tr=el('tr',{});
    const nameTd=el('td',{});nameInto(nameTd,b,'');
    tr.append(nameTd);
    const cells=[
      String(f.bad.filter(x=>x.sev==='blocker').length),
      String(f.bad.filter(x=>x.sev==='serious').length),
      hi?zv(hi,'ladders_ends_backwards')+' / '+zv(hi,'n_ladders'):'— NOT MEASURED',
      hi?pc(zv(hi,'inv_rate')):'— NOT MEASURED',
      lo?pc(zv(lo,'inv_rate')):'— NOT MEASURED',
      hfr&&hfr.frac_negative!=null?pc(hfr.frac_negative):'— NOT MEASURED',
      b.rank.cid22&&b.rank.cid22.frac_negative!=null?pc(b.rank.cid22.frac_negative):'—',
      b.dial&&b.dial.tied_pct!=null?pc(b.dial.tied_pct):'—',
      b.corruption&&b.corruption.pass_q20!=null?pc(b.corruption.pass_q20):'— NOT MEASURED',
      b.m3a!=null?f3(b.m3a):'— NOT MEASURED',
      hi||lo?f2(worst)+' pts':'— NOT MEASURED'];
    cells.forEach(c=>tr.append(el('td',{text:String(c)})));
    tb.append(tr);
  });
  tbl.append(tb);
  makeSortable(tbl);
  host.append(tbl);
  host.append(el('div',{class:'cap',html:'<b>ladders backwards q&gt;=85</b> = (image, codec) '
    +'ladders whose best-quality rung scores materially BELOW its worst-quality rung inside '
    +'the near-lossless band. <b>ladder-inv</b> = share of adjacent quality steps ordered '
    +'backwards by more than 0.5 dial points (the same events the G3 gate counts). '
    +'<b>refs bwd</b> = share of reference images whose within-image ranking is inverted. '
    +'<b>corruption q20</b> is a pass rate (higher is better); the dial alone is not the '
    +'corruption owner in the shipping design — a companion head is.'}));
  // ---- per-model cards -----------------------------------------------------
  const grid=el('div',{style:'display:flex;flex-wrap:wrap;gap:10px;margin-top:12px'});
  bs.forEach(b=>{
    const f=failures(b);
    const card=el('div',{style:'border:1px solid var(--border);border-radius:8px;padding:10px 12px;'
      +'background:var(--surface-1);min-width:340px;max-width:520px;flex:1 1 380px'});
    const hd=el('div',{style:'display:flex;align-items:center;gap:6px;margin-bottom:6px;flex-wrap:wrap'});
    hd.append(el('span',{class:'sw',style:'display:inline-block;background:'+color(b)}),el('b',{text:b.name}));
    const eb=ensBadge(b);if(eb)hd.append(eb);
    hd.append(el('span',{style:'font-size:9.5px;opacity:.7',
      text:(b.zones?('ladders: '+b.zones.grid):'ladders: NOT MEASURED')}));
    card.append(hd);
    if(!f.bad.length)card.append(el('div',{style:'font-size:11px;opacity:.8',
      text:'No failure rows cross the reporting thresholds on the evidence this cell carries. '
        +'Read the NOT MEASURED list below before reading that as clean.'}));
    f.bad.forEach(x=>{
      const row=el('div',{style:'margin:5px 0;padding:5px 7px;border-radius:5px;border:1px solid var(--border);'
        +'background:color-mix(in srgb, '+sevTone[x.sev]+' 12%, var(--surface-1))'});
      const t=el('div',{style:'font-size:11px'});
      t.append(el('span',{style:'font-size:8.5px;font-weight:700;letter-spacing:.05em;padding:1px 5px;'
        +'border-radius:7px;margin-right:6px;background:color-mix(in srgb, '+sevTone[x.sev]
        +' 35%, var(--surface-1))',text:x.sev.toUpperCase()}),el('b',{text:x.what}));
      row.append(t);
      row.append(el('div',{style:'font-size:10.5px;margin-top:2px',text:'how big: '+x.size}));
      row.append(el('div',{style:'font-size:10.5px;opacity:.9',text:'where you meet it: '+x.where}));
      row.append(el('div',{style:'font-size:9px;opacity:.6;margin-top:2px',text:'evidence: '+x.ev}));
      card.append(row);
    });
    if(b.zones&&b.zones.worst&&b.zones.worst.length){
      const d=el('details',{style:'margin-top:6px'});
      d.append(el('summary',{style:'font-size:10px;cursor:pointer;opacity:.85',
        text:'worst individual ladders, by reference image ('+b.zones.worst.length+')'}));
      const t=el('table',{style:'font-size:10px;margin-top:4px'});
      const hr=el('tr',{});['reference image','codec','content','zone','end delta','worst step']
        .forEach(h=>hr.append(el('th',{text:h})));
      const th2=el('thead',{});th2.append(hr);t.append(th2);
      const tb2=el('tbody',{});
      b.zones.worst.forEach(w=>{const tr=el('tr',{});
        [w[0],w[1],String(w[2]||'').replace('_','/'),w[3],
         (w[4]>0?'+':'')+f2(w[4])+' pts',f2(w[5])+' pts']
          .forEach(c=>tr.append(el('td',{text:String(c)})));tb2.append(tr);});
      t.append(tb2);makeSortable(t);d.append(t);
      d.append(el('div',{class:'cap',style:'margin-top:3px',
        text:'end delta = score at the zone best-quality rung minus the worst-quality rung '
          +'(negative = the whole ladder runs backwards); worst step = the deepest single '
          +'backwards move inside the zone. Sources are the dial grid reference PNGs — '
          +'/mnt/v/input/zensim/sources/<reference image>.png.'}));
      card.append(d);
    }
    if(f.good.length){
      const d=el('details',{style:'margin-top:6px'});
      d.append(el('summary',{style:'font-size:10px;cursor:pointer;opacity:.8',
        text:'reliably good at ('+f.good.length+')'}));
      const ul=el('ul',{style:'font-size:10.5px;margin:4px 0 0 16px'});
      f.good.forEach(g=>ul.append(el('li',{text:g})));
      d.append(ul);card.append(d);
    }
    if(f.nm.length){
      const d=el('details',{style:'margin-top:4px'});
      d.append(el('summary',{style:'font-size:10px;cursor:pointer;opacity:.8',
        text:'NOT MEASURED ('+f.nm.length+')'}));
      const ul=el('ul',{style:'font-size:10.5px;margin:4px 0 0 16px'});
      f.nm.forEach(g=>ul.append(el('li',{text:g.what+' — '+g.why})));
      d.append(ul);card.append(d);
    }
    grid.append(card);
  });
  host.append(grid);
}

// ---- layout + orchestration
function layout(){
  const p=$('#panels');p.innerHTML='';
  p.append(el('div',{id:'table'}),el('div',{id:'fairsec'}),el('div',{id:'beatssec'}),el('div',{id:'failures'}),el('div',{id:'heat'}),el('div',{id:'mpanel'}),el('div',{id:'dialsec'}),el('div',{id:'looptgt'}),el('div',{id:'loopcov'}),el('div',{id:'hfnlsec'}),el('div',{id:'gates'}),el('div',{id:'recipes'}),el('div',{id:'models'}),el('div',{id:'trade'}),el('div',{id:'scatter'}));
}

// ================= FAIRNESS AUDIT PANEL (2026-09-04) ==================================
// Row -> tier -> failing criteria, so the filter is auditable on the page and not only
// in the committed TSV. Computes nothing: it prints what `gauntlet.fairness_of` decided
// from the fulleval + the annotations registry.
function renderFair(){
  const d=$('#fairsec');if(!d)return;d.innerHTML='';
  const bs=DATA.bakes.slice().sort((a,b)=>{
    const o={'VERIFIED-FAIR':0,'FAIR-NOTED':1,'LEGACY':2};
    return (o[TIER(a)]-o[TIER(b)])||String(a.name).localeCompare(String(b.name));});
  const n={'VERIFIED-FAIR':0,'FAIR-NOTED':0,'LEGACY':0};bs.forEach(b=>{n[TIER(b)]=(n[TIER(b)]||0)+1;});
  d.append(el('h2',{text:'Fairness audit — which rows we can verify we did fairly'}));
  d.append(el('div',{class:'cap',html:
    'A row is <b>VERIFIED-FAIR</b> when every criterion below holds AND it is a replicated seed group '
    +'(k&ge;2). <b>FAIR-NOTED</b> = every criterion holds but the row is <b>UNREPLICATED</b> (k=1) or '
    +'ungroupable — still fair, just noted. <b>LEGACY</b> = at least one criterion fails; the row keeps '
    +'every stat and every badge and nothing is deleted, it is simply not default-visible. '
    +'Counts: <b>'+n['VERIFIED-FAIR']+'</b> verified-fair · <b>'+n['FAIR-NOTED']+'</b> fair-noted · <b>'
    +n['LEGACY']+'</b> legacy'+(DATA.fairOnly?(' ('+DATA.nLegacyDropped+' legacy rows are NOT IN THIS FILE — see the companion all-rows board)'):'')+'. '
    +'Criteria: '+(DATA.fairCriteria||[]).map(c=>'<code>'+c[0]+'</code> '+c[1]).join(' · ')+'.'}));
  const t=el('table',{class:'tbl'});
  t.append(el('thead',{html:'<tr><th>bake</th><th>tier</th><th>k</th><th>failing criteria</th><th>notes</th></tr>'}));
  const tb=el('tbody',{});
  bs.forEach(b=>{const f=b.fair||{};
    const tr=el('tr',{});
    const nd=el('td',{class:'lbl'});nameInto(nd,b,'');tr.append(nd);
    const td=el('td',{text:TIER(b)});
    td.style.color=TIER(b)==='VERIFIED-FAIR'?'var(--good)':(TIER(b)==='FAIR-NOTED'?'var(--warn)':'var(--critical)');
    tr.append(td);
    const kd=el('td',{text:f.k==null?'— (ungroupable)':(f.k===1?'1 (UNREPLICATED)':String(f.k))});
    kd.setAttribute('title',sgTitle(b,'composite')||SEEDNOTE);kd.style.cursor='help';tr.append(kd);
    const fd=el('td',{text:(f.fails&&f.fails.length)?f.fails.join(', '):'—'});
    if(f.fails&&f.fails.length)fd.setAttribute('title',f.fails.map(k=>k+': '+(CRIT[k]||'')).join('\n'));
    tr.append(fd);
    tr.append(el('td',{class:'lbl',text:(f.notes&&f.notes.length)?f.notes.join(' | '):'—'}));
    tb.append(tr);});
  t.append(tb);d.append(t);makeSortable(t);
}

// ============ BEATS-SSIM2 EVIDENCE PANEL (2026-09-04) =================================
// The product composite carries ssim2-ANCHORED axes at weight 0.50 (imazen26) + 0.30
// (nonphoto) of 2.15 — 37% of it — which is how a peer row can top a board that is
// supposed to be ranking models against that peer. This panel answers the "beats ssim2"
// question on the NON-CIRCULAR held-out human axes only, with peer_ssim2 as the
// reference ROW. No new composite is invented: the per-axis deltas are differences of
// numbers each row already carries, and the W1-W7 verdicts are TRANSCRIBED from the
// exam, never re-derived here.
function renderBeats(){
  const d=$('#beatssec');if(!d)return;d.innerHTML='';
  if(!PEERSSIM2){d.append(el('div',{class:'cap',html:'<b>NOT MEASURED</b> — no <code>peer_ssim2</code> row on this board, so no beats-ssim2 evidence view can be drawn.'}));return;}
  const dCorp=(EXAM&&EXAM.thresholds)?EXAM.thresholds:{};
  const dlt=c=>c==='cid22'?(dCorp.delta_cid22_pooled||0.010):(dCorp.delta_corpus||0.010);
  d.append(el('h2',{text:'Beats-ssim2 evidence — non-circular axes only'}));
  d.append(el('div',{class:'cap',html:
    'Reference row = <b>peer_ssim2</b>. Axes shown are the exam’s <b>genuinely held-out human corpora</b>: '
    +HELDOUT.join(', ')+'. <b>EXCLUDED by construction:</b> '+(DATA.circularAxes||[]).join(', ')
    +' (ssim2-ANCHORED — their targets ARE ssim2 scores, so a number there is agreement with ssim2, never a win '
    +'over it; <code>hf_nearlossless</code>’s <code>human_score</code> IS <code>ssim2_gpu/100</code>, exactly, on '
    +'1200/1200 rows) and '+(DATA.trainEqValAxes||[]).join(', ')+' (train==val integrity guards). '
    +'A cell is a <b>LOSS</b> when it is worse than peer_ssim2 by more than δ ('+f3(dlt('cid22'))
    +' pooled CID22, '+f3(dlt('csiq'))+' elsewhere — derived, §2.4). δ is a reference-clustered '
    +'bootstrap half-width; <b>the board’s own srocc_ci understates it by ~2×</b> (4,292 CID22 pairs are 49 '
    +'clusters, not 4,292 draws). W1–W7 are TRANSCRIBED verdicts — present only for the exam’s six scored '
    +'candidates; every other row reads NOT MEASURED, never a fail. '+(EXAM?('<b>'+EXAM.headline+'</b>'):'')}));
  const t=el('table',{class:'tbl'});
  const hd=['bake','tier','n LOSS >δ'].concat(HELDOUT).concat(['W1','W2','W3','W4','W5','W6','W7']);
  t.append(el('thead',{html:'<tr>'+hd.map(h=>'<th>'+h+'</th>').join('')+'</tr>'}));
  const tb=el('tbody',{});
  const ref={};HELDOUT.forEach(c=>{ref[c]=rs(PEERSSIM2,c);});
  const rows=DATA.bakes.filter(b=>state.visible.has(b.name)||b.name==='peer_ssim2');
  rows.forEach(b=>{
    const tr=el('tr',{});
    const nd=el('td',{class:'lbl'});nameInto(nd,b,b.name==='peer_ssim2'?' ◀ reference':'');tr.append(nd);
    tr.append(el('td',{text:b.name==='peer_ssim2'?'reference':tierGlyph(b)}));
    let nloss=0,nmeas=0;
    const cells=HELDOUT.map(c=>{const v=rs(b,c),r=ref[c];
      if(v==null||r==null)return [c,null,null];
      nmeas++;const dv=v-r;if(b.name!=='peer_ssim2'&&dv< -dlt(c))nloss++;return [c,v,dv];});
    const ld=el('td',{text:b.name==='peer_ssim2'?'—':(nmeas?(nloss+'/'+nmeas):'—')});
    if(nloss)ld.style.color='var(--critical)';
    ld.setAttribute('title',nmeas?('counted over the '+nmeas+' held-out axes BOTH rows carry. '
      +'This is W1’s own rule applied per row; it is a COUNT of already-measured differences, not a new statistic.')
      :'NOT MEASURED — this row shares no held-out axis with peer_ssim2.');
    ld.style.cursor='help';tr.append(ld);
    cells.forEach(([c,v,dv])=>{
      const td=el('td',{text:v==null?'—':f3(v)});
      if(dv!=null&&b.name!=='peer_ssim2'){
        td.append(el('span',{style:'opacity:.75;font-size:.85em',text:' ('+(dv>=0?'+':'')+f3(dv)+')'}));
        if(dv< -dlt(c))td.style.color='var(--critical)';
        else if(dv>dlt(c))td.style.color='var(--good)';
        td.setAttribute('title',c+': '+f3(v)+' vs peer_ssim2 '+f3(ref[c])+'  Δ '+(dv>=0?'+':'')+f3(dv)
          +'\nδ = '+f3(dlt(c))+' — '+(dv< -dlt(c)?'LOSS beyond δ':(dv>dlt(c)?'nominal gain beyond δ (a STRICT win additionally needs the paired CI to exclude zero — W2)':'inside δ: a TIE')));
        td.style.cursor='help';}
      tr.append(td);});
    const e=examOf(b.name);
    ['W1','W2','W3','W4','W5','W6','W7'].forEach(w=>{
      const td=el('td',{text:e?e[w].v:'—'});
      if(e){td.style.color=e[w].v==='PASS'?'var(--good)':(e[w].v==='FAIL'?'var(--critical)':'var(--muted)');
        td.setAttribute('title',e[w].note||e[w].v);td.style.cursor='help';}
      else td.setAttribute('title','NOT one of the exam’s six scored candidates — NOT MEASURED, never a fail.');
      tr.append(td);});
    tb.append(tr);});
  t.append(tb);d.append(t);makeSortable(t);
  if(EXAM&&EXAM.amendments&&EXAM.amendments.length){
    d.append(el('div',{class:'cap',html:'<b>Amendments carried by the exam record:</b><br>'
      +EXAM.amendments.map(a=>'• '+a).join('<br>')}));}
}

// renderTable() returns a wrapper without an id; mountTable tags it and swaps it in.
function mountTable(){const w=renderTable();w.id='table';const cur=$('#table');cur?cur.replaceWith(w):$('#panels').prepend(w);}
function rerender(){disposeCharts();state.renderedTheme=effTheme();
  mountTable();renderFair();renderBeats();renderFailures();renderHeat();renderMPanel();renderDial();renderLoop();renderCoverage();renderHfnl();renderGates();renderRecipes();renderModels();renderTrade();renderScatter();}

initRef();layout();renderBar();rerender();
// Theme reactivity: charts (and everything else) rebuild with the other theme's option
// variant on (a) an OS prefers-color-scheme flip when no explicit data-theme is set, and
// (b) the artifact viewer stamping data-theme on <html> — watched via MutationObserver
// (typeof-guarded: the DOM-shim harness has none). state.renderedTheme dedupes the manual
// theme button, which sets the attribute and re-renders synchronously itself.
if(window.matchMedia)matchMedia('(prefers-color-scheme:dark)').addEventListener('change',()=>{if(!document.documentElement.getAttribute('data-theme'))rerender();});
if(typeof MutationObserver==='function'&&document.documentElement){
  try{new MutationObserver(()=>{if(effTheme()!==state.renderedTheme){renderBar();rerender();}})
    .observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});}catch(e){}
}
"""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fulleval-dir", default="/mnt/v/output/zensim/reports/fulleval")
    ap.add_argument("--best-per-day", default=None)
    ap.add_argument("--loop-targeting", default=DEFAULT_LOOP_TARGETING,
                    help="jxl-encoder 2/3-shot loop-targeting summary JSON (section omitted if absent)")
    ap.add_argument("--hfnl-axis", default=DEFAULT_HFNL_AXIS,
                    help="appendix-O HF-NL axis study JSON (panel omitted if absent)")
    ap.add_argument("--out", default="/mnt/v/output/zensim/reports/summer_gauntlet.html")
    ap.add_argument("--fair-only", action="store_true",
                    help="emit the VERIFIED-FAIR board: LEGACY rows are dropped from the "
                         "FILE (not merely hidden) so the fair view fits the registered "
                         "12 MB cap. They stay on disk, on the all-rows board, and in "
                         "the fairness TSV — nothing is deleted.")
    ap.add_argument("--fairness-tsv", default=None,
                    help="also write the per-row fairness audit TSV here")
    a = ap.parse_args()
    bakes = load_fulleval(a.fulleval_dir, a.best_per_day)
    if a.fairness_tsv:
        tp, n = write_fairness_tsv(bakes, a.fairness_tsv)
        print(f"wrote {tp}  ({n} rows, fairness audit)")
    out, size = build_html(bakes, a.out, loop_targeting=load_loop_targeting(a.loop_targeting),
                           hfnl_axis=load_hfnl_axis(a.hfnl_axis), fair_only=a.fair_only)
    print(f"wrote {out}  ({size // 1024} KB)  {len(bakes)} bakes"
          + ("  [FAIR-ONLY]" if a.fair_only else ""))
