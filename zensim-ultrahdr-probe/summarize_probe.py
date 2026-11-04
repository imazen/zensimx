#!/usr/bin/env python3
"""Summarise the Ultra HDR same-base probe TSVs: per q, SDR score vs HDR score (gain map applied to both bases) for
zensim F, SSIMULACRA2 (native HDR) and butteraugli (linear path), at each display boost; plus rank agreement between
the SDR and HDR readings across (image, q) and the per-image gap."""
import csv, sys
from collections import defaultdict
import numpy as np


def rank(a):
    a = np.asarray(a, float); o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); s = a[o]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


def load(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    for r in rows:
        for k, v in list(r.items()):
            if k != "file":
                try: r[k] = float(v)
                except ValueError: pass
    return rows


def main():
    out = []
    for path in sys.argv[1:]:
        rows = load(path)
        files = sorted({r["file"] for r in rows})
        boosts = sorted({round(r["boost"], 3) for r in rows})
        out.append(f"## {path}: {len(files)} images, {len(rows)} rows, boosts {boosts}")
        # full-boost values vary per image; label rows as '4x' or 'full'
        def tag(r):
            if abs(r["boost"] - 1.0) < 1e-6:
                return "sdr-as-hdr-203" if abs(r.get("white_nits", 203.0) - 203.0) < 1e-3 else "sdr-as-hdr-100"
            return "4x" if abs(r["boost"] - 4.0) < 1e-6 else "full"
        labels = {"sdr-as-hdr-203": "the SDR base fed through the HDR paths, no gain-map boost, SDR white 203 cd/m²",
                  "sdr-as-hdr-100": "the SDR base fed through the HDR paths, no gain-map boost, SDR white 100 cd/m²",
                  "4x": "gain map at a 4x display boost, 812 cd/m² peak", "full": "gain map at its full boost, per image"}
        for t in ("sdr-as-hdr-203", "sdr-as-hdr-100", "4x", "full"):
            sel = [r for r in rows if tag(r) == t]
            if not sel: continue
            out.append(f"\n### {t}: {labels[t]}")
            out.append("| q | base kB | zensim SDR | zensim HDR | Δ | ssim2 SDR | ssim2 HDR | Δ | butter3 SDR | butter3 HDR | butter max SDR | butter max HDR |")
            out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for q in sorted({int(r["q"]) for r in sel}):
                g = [r for r in sel if int(r["q"]) == q]
                m = lambda k: float(np.mean([r[k] for r in g]))
                out.append(f"| {q} | {m('base_q_bytes')/1024:.0f} | {m('zensim_sdr'):.2f} | {m('zensim_hdr'):.2f} | {m('zensim_hdr')-m('zensim_sdr'):+.2f} | "
                           f"{m('ssim2_sdr'):.2f} | {m('ssim2_hdr'):.2f} | {m('ssim2_hdr')-m('ssim2_sdr'):+.2f} | {m('butter3_sdr'):.3f} | {m('butter3_hdr'):.3f} | {m('buttermax_sdr'):.3f} | {m('buttermax_hdr'):.3f} |")
            out.append("")
            for metric in ("zensim", "ssim2", "butter3", "buttermax"):
                x = [r[f"{metric}_sdr"] for r in sel]; y = [r[f"{metric}_hdr"] for r in sel]
                within = []
                for f in files:
                    g = [r for r in sel if r["file"] == f]
                    if len(g) >= 3: within.append(spearman([r[f"{metric}_sdr"] for r in g], [r[f"{metric}_hdr"] for r in g]))
                gap = np.array(y) - np.array(x)
                out.append(f"- {metric}: SDR vs HDR rank agreement across all rows {spearman(x, y):.3f}, within-image mean {np.nanmean(within):.3f}; "
                           f"HDR minus SDR median {np.median(gap):+.3f}, 10th/90th pct {np.percentile(gap, 10):+.3f} / {np.percentile(gap, 90):+.3f}")
            # per-image gap at q=75 for zensim
            out.append("")
            out.append("| image | gain-map max boost | zensim SDR q75 | zensim HDR q75 | Δ | ssim2 SDR q75 | ssim2 HDR q75 | Δ |")
            out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for f in files:
                g = [r for r in sel if r["file"] == f and int(r["q"]) == 75]
                if not g: continue
                r = g[0]
                out.append(f"| {f.split('/')[-1][:48]} | {r['full_boost']:.2f} | {r['zensim_sdr']:.2f} | {r['zensim_hdr']:.2f} | {r['zensim_hdr']-r['zensim_sdr']:+.2f} | {r['ssim2_sdr']:.2f} | {r['ssim2_hdr']:.2f} | {r['ssim2_hdr']-r['ssim2_sdr']:+.2f} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
