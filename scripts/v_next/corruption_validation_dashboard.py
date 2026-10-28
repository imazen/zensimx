#!/usr/bin/env python3
"""Corruption-validation dashboard — how zensim validates STRUCTURAL corruption gating,
visualized. Companion to bandwise_dashboard.py (which covers rank SROCC on the canonical
corpora); this covers the CORRUPTION GATE + the NEGATIVE-dial tail, a different axis.

The gate (methodology): the gb82_dog corruption corpus is one source image put through
~44 structural-break FAMILIES × 6 regions (whole..sq8) × 3 severities, each paired with
two HONEST low-quality JPEG anchors (q10, q20). A metric "catches" a corruption when it
scores the structurally-broken variant WORSE than the honest-but-low-quality q20 anchor.
Gate pass@q20 = fraction of recipes where damage(corruption) > damage(q20). This is the
thing a plain quality metric fails: a torn/garbled decode often has GOOD local SSIM, so
ssim2/cvvdp rank it ABOVE honest q20 (the inversion) — butteraugli's max-norm catches it,
and so does a zensim bake trained on the corruption corpus (cl_tfm).

Bakes are scored on the held-out corruption_gate.parquet features (predict_features_with_bake
--bake-post raw); reference metrics (ssim2/butteraugli/cvvdp/dssim) come from the May-28
multimetric TSV, joined per recipe. The negative-dial section scores the kadis_negrich tail
(human_score down to -14) which ships ssim2_gpu for a direct rank comparison.

  usage: python3 scripts/v_next/corruption_validation_dashboard.py [--out x.html]
"""
import argparse
import io
import struct
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

REPO = Path.home() / "work/zen/zensimx"
SCORER = str(REPO / "target/release/predict_features_with_bake")
CORRLQ = Path("/mnt/v/output/zensim/corr-lq")
TSV = Path("/mnt/v/output/zensim/corruption_gate_results/corruption_multimetric_2026-05-28.tsv")
OUTDIR = Path("/mnt/v/output/zensim/dashboards")

# series to compare: (label, kind, ref) — bakes scored from features, metrics joined from TSV
BAKES = [
    ("cl_tfm-s13", CORRLQ / "cl_tfm.bin"),
    ("cl_tfm-s31", CORRLQ / "cl_tfm_s31.bin"),
    ("B(shipped)", REPO / "zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin"),
]
# reference-metric columns in the TSV and their orientation (True = higher is WORSE quality)
METRICS = [
    ("ssim2", "ssim2_gpu", False),
    ("butter-max", "butter_max_gpu", True),
    ("butter-p3", "butter_pnorm3_gpu", True),
    ("cvvdp", "cvvdp", False),
    ("dssim", "dssim_gpu", True),
]
REGIONS = ["whole", "frac2", "frac4", "sq64", "sq16", "sq8"]
REGION_SET = set(REGIONS)

plt.rcParams.update({"font.size": 7.5, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.facecolor": "white", "svg.fonttype": "none"})


def svg_of(fig):
    b = io.StringIO()
    fig.savefig(b, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = b.getvalue()
    return s[s.find("<svg"):]


def score_bake_raw(bake, feats):
    """feats: (n, 372) float32 -> raw model output per row."""
    n = feats.shape[0]
    buf = struct.pack("<II", 372, n) + feats.astype(np.float32).tobytes()
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(buf)
        fp = f.name
    out = subprocess.run([SCORER, "--bake", str(bake), "--bake-post", "raw", "--features-file", fp],
                         capture_output=True, text=True)
    Path(fp).unlink(missing_ok=True)
    return np.array([float(x) for x in out.stdout.split()])


def parse_recipe(rb):
    """gb82_dog__family__region__sev -> (family, region, sev). family may hold single _."""
    p = rb.split("__")
    rest = p[1:]
    region = next((x for x in rest if x in REGION_SET), "whole")
    sev = next((x for x in rest if x.startswith("op")), rest[-1])
    fam = "_".join(x for x in rest if x not in REGION_SET and not x.startswith("op"))
    return fam, region, sev


def level_of(h):
    return "corruption" if h < 0.05 else ("q20" if h > 0.5 else "q10")


def collect():
    """-> recipes[ref_basename] = {family,region,sev, series: {level: damage_value}}.
    damage_value: higher = more damage, so gate pass = damage(corruption) > damage(q20)."""
    t = pq.read_table(str(CORRLQ / "corruption_gate.parquet"))
    n = t.num_rows
    feats = np.stack([np.asarray(t[f"f{i}"], dtype=np.float32) for i in range(372)], axis=1)
    rb = [str(x) for x in t["ref_basename"].to_pylist()]
    hs = np.asarray(t["human_score"], dtype=float)

    rec = {}
    for i, r in enumerate(rb):
        if r not in rec:
            fam, region, sev = parse_recipe(r)
            rec[r] = {"family": fam, "region": region, "sev": sev, "series": defaultdict(dict)}

    # bakes: raw score, orient so higher = more damage => damage = -score (bakes: high=good)
    for lab, path in BAKES:
        sc = score_bake_raw(path, feats)
        for i, r in enumerate(rb):
            rec[r]["series"][lab][level_of(hs[i])] = -sc[i]

    # metrics from TSV, joined on (name, kind)
    tsv = {}
    lines = TSV.read_text().splitlines()
    hdr = lines[0].split("\t")
    ci = {c: k for k, c in enumerate(hdr)}
    for ln in lines[1:]:
        c = ln.split("\t")
        tsv[(c[ci["name"]], c[ci["kind"]])] = c
    for lab, col, high_bad in METRICS:
        for r in rec:
            for lvl in ("corruption", "q10", "q20"):
                row = tsv.get((r, lvl))
                if row and row[ci[col]] not in ("", "nan"):
                    v = float(row[ci[col]])
                    rec[r]["series"][lab][lvl] = v if high_bad else -v
    return rec


def gate_rates(rec):
    """series -> {'q20':pass_frac, 'q10':pass_frac, per_family:{fam:frac}, per_region:{reg:frac}}."""
    series = [lab for lab, _ in BAKES] + [m[0] for m in METRICS]
    out = {}
    for s in series:
        p20 = p10 = tot = 0
        byfam = defaultdict(lambda: [0, 0])
        byreg = defaultdict(lambda: [0, 0])
        for r, d in rec.items():
            sd = d["series"].get(s, {})
            if not all(k in sd for k in ("corruption", "q20", "q10")):
                continue
            tot += 1
            ok20 = sd["corruption"] > sd["q20"]
            ok10 = sd["corruption"] > sd["q10"]
            p20 += ok20
            p10 += ok10
            byfam[d["family"]][0] += ok20
            byfam[d["family"]][1] += 1
            byreg[d["region"]][0] += ok20
            byreg[d["region"]][1] += 1
        out[s] = {
            "q20": p20 / max(tot, 1), "q10": p10 / max(tot, 1), "n": tot,
            "per_family": {f: a / b for f, (a, b) in byfam.items()},
            "per_region": {rg: a / b for rg, (a, b) in byreg.items()},
        }
    return out


def fig_gate_bar(rates):
    series = list(rates.keys())
    order = sorted(series, key=lambda s: rates[s]["q20"], reverse=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(order))
    bake_labels = {lab for lab, _ in BAKES}
    colors = ["#c0392b" if s in bake_labels else "#2980b9" for s in order]
    ax.barh(y, [rates[s]["q20"] * 100 for s in order], color=colors, alpha=0.85)
    ax.barh(y, [rates[s]["q10"] * 100 for s in order], color="none", edgecolor="#333",
            hatch="////", linewidth=0.4)
    for i, s in enumerate(order):
        ax.text(rates[s]["q20"] * 100 + 1, i, f'{rates[s]["q20"]*100:.0f}%', va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("gate pass @ q20 (solid) · @ q10 (hatched) — higher = catches more corruption")
    ax.set_xlim(0, 105)
    ax.set_title("Corruption gate: corruption ranked BELOW honest q20  (red=zensim bake, blue=reference metric)")
    return svg_of(fig)


def fig_region_line(rates):
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for s in rates:
        ys = [rates[s]["per_region"].get(rg, np.nan) * 100 for rg in REGIONS]
        bake = s in {lab for lab, _ in BAKES}
        ax.plot(REGIONS, ys, marker="o", ms=3, lw=1.8 if bake else 1.0,
                label=s, alpha=0.9 if bake else 0.7)
    ax.set_ylabel("gate pass @ q20 (%)")
    ax.set_xlabel("corruption region — smaller = subtler/harder")
    ax.set_title("Difficulty gradient by region size")
    ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(-3, 105)
    return svg_of(fig)


def fig_family_heatmap(rates):
    fams = sorted({f for s in rates for f in rates[s]["per_family"]},
                  key=lambda f: np.mean([rates[s]["per_family"].get(f, 0) for s in rates]))
    series = list(rates.keys())
    M = np.array([[rates[s]["per_family"].get(f, np.nan) * 100 for f in fams] for s in series])
    fig, ax = plt.subplots(figsize=(max(9, len(fams) * 0.22), 2.6))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels(series, fontsize=7)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, rotation=90, fontsize=5.2)
    ax.set_title("Per-family gate pass @ q20 (%) — sorted hardest→easiest left→right")
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    return svg_of(fig)


def fig_separation(rec, series):
    """Per series: scatter damage(q20) vs damage(corruption) per recipe. Above diagonal = pass."""
    figs = []
    for s in series:
        xs, ys = [], []
        for r, d in rec.items():
            sd = d["series"].get(s, {})
            if all(k in sd for k in ("corruption", "q20")):
                xs.append(sd["q20"])
                ys.append(sd["corruption"])
        xs, ys = np.array(xs), np.array(ys)
        # rank-percentile so different metric scales compare
        def pct(a):
            return np.argsort(np.argsort(a)) / max(len(a) - 1, 1) * 100
        px, py = pct(xs), pct(ys)
        passf = (ys > xs).mean() * 100
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.scatter(px, py, s=4, alpha=0.4,
                   c=["#27ae60" if a > b else "#c0392b" for a, b in zip(ys, xs)])
        ax.plot([0, 100], [0, 100], "k--", lw=0.6)
        ax.set_xlabel("honest q20 damage %ile")
        ax.set_ylabel("corruption damage %ile")
        ax.set_title(f"{s}\npass {passf:.0f}% (green=caught)", fontsize=6.5)
        figs.append(svg_of(fig))
    return "".join(figs)


def negrich_section():
    """kadis_negrich tail: cl_tfm vs B vs ssim2 rank-track the negative tail (human_score to -14)."""
    t = pq.read_table(str(CORRLQ / "kadis_negrich_gate.parquet"))
    n = t.num_rows
    feats = np.stack([np.asarray(t[f"f{i}"], dtype=np.float32) for i in range(372)], axis=1)
    h = np.asarray(t["human_score"], dtype=float)
    ss = np.asarray(t["ssim2_gpu"], dtype=float)

    def srocc(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        ra, rb = np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m]))
        return np.corrcoef(ra, rb)[0, 1]

    series = {}
    for lab, path in [("cl_tfm-s13", CORRLQ / "cl_tfm.bin"), ("B(shipped)", REPO / "zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin")]:
        series[lab] = score_bake_raw(path, feats)
    series["ssim2"] = ss

    sroccs = {lab: srocc(v, h) for lab, v in series.items()}

    # SROCC bar
    fig1, ax = plt.subplots(figsize=(3.4, 2.6))
    labs = list(sroccs)
    ax.bar(labs, [sroccs[l] for l in labs],
           color=["#c0392b", "#c0392b", "#2980b9"][:len(labs)], alpha=0.85)
    for i, l in enumerate(labs):
        ax.text(i, sroccs[l] + 0.01, f"{sroccs[l]:.3f}", ha="center", fontsize=7)
    ax.set_ylabel("SROCC vs human_score")
    ax.set_ylim(0, 1)
    ax.set_title("Negative-tail rank tracking (kadis_negrich, n=%d)" % n)
    bar = svg_of(fig1)

    # scatter (subsample)
    idx = np.random.default_rng(0).choice(n, size=min(2500, n), replace=False)
    fig2, axs = plt.subplots(1, len(series), figsize=(2.5 * len(series), 2.6), sharex=True)
    for ax, (lab, v) in zip(np.atleast_1d(axs), series.items()):
        def pct(a):
            return np.argsort(np.argsort(a)) / max(len(a) - 1, 1) * 100
        ax.scatter(h[idx], pct(v)[idx], s=3, alpha=0.25, c="#333")
        ax.set_title(f"{lab}  (SROCC {sroccs[lab]:.3f})", fontsize=6.5)
        ax.set_xlabel("human_score (→ -14 severe)")
    np.atleast_1d(axs)[0].set_ylabel("pred %ile")
    sca = svg_of(fig2)
    return bar, sca, sroccs, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTDIR / "corruption_validation_2026-07-17.html"))
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("scoring bakes on corruption gate + joining reference metrics ...")
    rec = collect()
    rates = gate_rates(rec)
    print("gate pass @q20:", {s: round(rates[s]["q20"] * 100, 1) for s in rates})

    bake_series = [lab for lab, _ in BAKES]
    body = [
        "<h1>zensim corruption-gate validation — how we catch structural corruption</h1>",
        "<p class='sub'>Companion to the <a href='cl_tfm_dashboard_2026-07-17.html'>bandwise rank dashboard</a>. "
        "This is a different axis: a <b>gate</b>, not SROCC. One source image (gb82_dog) is put through ~44 "
        "structural-break families × 6 regions × 3 severities; each is paired with two HONEST low-quality JPEG "
        "anchors (q10, q20). A metric <b>catches</b> a corruption when it scores the broken variant WORSE than "
        "the honest q20 anchor — <code>damage(corruption) &gt; damage(q20)</code>. Bakes scored on held-out "
        "<code>corruption_gate.parquet</code> (raw output); reference metrics from the May-28 multimetric TSV. "
        "By <code>scripts/v_next/corruption_validation_dashboard.py</code>.</p>",
        "<h2>Why a plain quality metric fails here</h2>",
        "<p class='sub'>A torn / block-garbled / channel-swapped decode often has LOCALLY good SSIM, so ssim2 / "
        "cvvdp / dssim frequently rank it ABOVE an honest but low-quality JPEG (the <i>inversion</i>: median "
        "ssim2 corruption−q20 gap is −28, wrong sign). butteraugli's max-norm is the reference win (72%). A "
        "zensim bake trained on the corruption corpus (cl_tfm) folds that catch into the metric itself via the "
        "psa-α pool head's max / p-norm terms.</p>",
        "<h2>Headline — gate pass rate</h2>",
        fig_gate_bar(rates),
        "<div class='row'>",
        "<div>" + fig_region_line(rates) + "</div>",
        "</div>",
        "<h2>Which corruptions each catches</h2>",
        fig_family_heatmap(rates),
        "<h2>The gate, per recipe (green = caught)</h2>",
        "<p class='sub'>Each point is one corruption recipe: honest-q20 damage percentile (x) vs corruption "
        "damage percentile (y). Above the diagonal = the corruption is ranked more-damaged than honest q20 = "
        "caught. A metric that clusters BELOW the diagonal is inverting (ranking corruption as better than "
        "honest low-q).</p>",
        "<div class='row'>" + fig_separation(rec, bake_series + [m[0] for m in METRICS]) + "</div>",
    ]

    try:
        bar, sca, sroccs, nn = negrich_section()
        body += [
            "<h2>Negative-dial validation — the severe tail</h2>",
            "<p class='sub'>The <code>kadis_negrich</code> holdout runs human_score down to −14 (far below any "
            "honest codec output). cl_tfm rank-tracks that tail; ssim2 (bounded [0,100]) and B track it less. "
            "The dial spline then maps this rank onto negative user-facing scores (measured dial p5 ≈ −41). "
            f"n={nn}.</p>",
            "<div class='row'><div>" + bar + "</div></div>",
            sca,
        ]
    except Exception as e:
        body.append(f"<p class='sub'>negative-dial section skipped: {e}</p>")

    # numeric table
    rows = ["<h2>Gate pass table</h2>",
            "<table><tr><th>series</th><th>type</th><th>pass@q20</th><th>pass@q10</th><th>n</th></tr>"]
    for s in sorted(rates, key=lambda s: rates[s]["q20"], reverse=True):
        typ = "zensim bake" if s in bake_series else "reference metric"
        rows.append(f"<tr><td class='lbl'>{s}</td><td>{typ}</td>"
                    f"<td>{rates[s]['q20']*100:.1f}%</td><td>{rates[s]['q10']*100:.1f}%</td>"
                    f"<td>{rates[s]['n']}</td></tr>")
    rows.append("</table>")
    body += rows

    html = ("<style>body{font:13px system-ui,sans-serif;margin:1.3rem;background:#fff;color:#111}"
            "h1{font-size:1.3rem}h2{font-size:1.05rem;margin-top:1.5rem;border-top:1px solid #ccc;padding-top:.5rem}"
            ".sub{color:#555;max-width:74rem}.row{display:flex;flex-wrap:wrap;gap:6px;align-items:flex-start}"
            "svg{max-width:100%;height:auto;border:1px solid #eee}"
            "table{border-collapse:collapse;font-size:11px}th,td{border:1px solid #ddd;padding:2px 6px;text-align:center}"
            "td.lbl{text-align:left;font-weight:600;background:#f4f4f4}code{background:#f0f0f0;padding:.1rem .3rem}"
            "a{color:#06c}</style>" + "".join(body))
    Path(a.out).write_text(html)
    print(f"wrote {a.out}  ({len(html)//1024} KB)")
    print("  view: " + a.out.replace("/mnt/v/output/", "http://localhost:3300/"))


if __name__ == "__main__":
    main()
