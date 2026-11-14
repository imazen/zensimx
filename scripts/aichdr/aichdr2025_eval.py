#!/usr/bin/env python3
"""AIC-HDR2025 frozen-metric read: zensim A against the dataset's 29 published metric columns.

Data: github.com/jpeg-aic/AIC-HDR2025 (commit 33ab64e5, released 2026-11-14), cloned to
/mnt/v/datasets/aic-hdr2025. Target = reconstructed JND (distortion-oriented: 0 = identical).
zensim scores come from tools/aichdr_score (the PUBLISHED zensim crate, compute_hdr, PQ code
values, BT.2020 primaries). Peer scores are the authors' own `data/metric_scores.csv`.

All statistics come from the canonical `panel` binary (zensim_validate::panel, Mohammadi 2025
panel: SROCC/PLCC/KROCC/OR/PWRC/Z-RMSE with the 4-parameter logistic). This script only joins
tables, drives `panel`, draws the bootstrap indices (fixed seed) and plots.

Usage: aichdr2025_eval.py --panel <panel-bin> --zensim <zensim_a_scores.tsv> --out <dir>
"""
import argparse, csv, json, math, os, subprocess, sys
import numpy as np

DATA = "/mnt/v/datasets/aic-hdr2025/data"


def run_panel(panel, rows, extra=()):
    path = os.path.join(ARGS.out, "_panel_in.tsv")
    with open(path, "w") as f:
        f.write("predicted\ttarget\tsigma\tband\n")
        for p, t, s, b in rows:
            f.write(f"{p!r}\t{t!r}\t{s!r}\t{b}\n")
    cmd = [panel, "--input", path, "--json", *extra]
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


def run_scatter(panel, pred, target):
    path = os.path.join(ARGS.out, "_scatter_in.tsv")
    with open(path, "w") as f:
        f.write("predicted\ttarget\n")
        for p, t in zip(pred, target):
            f.write(f"{p!r}\t{t!r}\n")
    out = subprocess.run([panel, "--input", path, "--json", "--scatter"], check=True,
                         capture_output=True, text=True).stdout
    return json.loads(out)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--zensim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261114)
    ARGS = ap.parse_args()
    jnd = {r["image"]: r for r in csv.DictReader(open(f"{DATA}/jnd_scores.csv"))}
    ms = list(csv.DictReader(open(f"{DATA}/metric_scores.csv")))
    images = [r["image"] for r in ms]
    assert len(images) == 100 and len(set(images)) == 100
    peer_cols = [c for c in ms[0].keys()
                 if c not in ("image", "source_id", "source", "codec", "level", "bpp", "jnd")]
    metrics = {}
    for prof_peak in ("1000", "10000"):
        z = {r["image"]: float(r["score"]) for r in csv.DictReader(open(ARGS.zensim), delimiter="\t")
             if r["peak_nits"] == prof_peak}
        metrics[f"zensim A (PQ, {prof_peak} nit peak)"] = [z[i] for i in images]
    for c in peer_cols:
        metrics[c] = [float(r[c]) for r in ms]
    target = [float(jnd[i]["jnd"]) for i in images]
    # The jnd column in metric_scores must equal jnd_scores.
    assert all(abs(float(r["jnd"]) - t) < 1e-9 for r, t in zip(ms, target))
    sigma = [(float(jnd[i]["jnd_ci95_high"]) - float(jnd[i]["jnd_ci95_low"])) / (2 * 1.96)
             for i in images]
    codec = [jnd[i]["codec"] for i in images]
    source = [jnd[i]["source"] for i in images]

    results = {}
    for name, pred in metrics.items():
        by_codec = run_panel(ARGS.panel, list(zip(pred, target, sigma, codec)))
        by_source = run_panel(ARGS.panel, list(zip(pred, target, sigma, source)))
        signed = run_panel(ARGS.panel, list(zip(pred, target, sigma, ["all"] * 100)),
                           ("--signed-quality",))
        scat = run_scatter(ARGS.panel, pred, target)
        g = {x["label"]: x for x in by_codec["groups"]}
        s = {x["label"]: x for x in by_source["groups"]}
        results[name] = {
            "all": g["ALL"],
            "per_codec": {k[5:]: v for k, v in g.items() if k.startswith("band=")},
            "per_source": {k[5:]: v for k, v in s.items() if k.startswith("band=")},
            "signed": signed,
            "scatter": {"geo": scat.get("geo"), "raw": scat.get("raw"), "status": scat.get("status")},
        }

    # Paired image bootstrap of SROCC(zensim A) - SROCC(peer), one panel --batch process.
    rng = np.random.default_rng(ARGS.seed)
    idx = [rng.integers(0, 100, 100) for _ in range(ARGS.resamples)]
    names = list(metrics)
    lines = [f"#def T\t{','.join(repr(v) for v in target)}"]
    for k, n in enumerate(names):
        lines.append(f"#def M{k}\t{','.join(repr(v) for v in metrics[n])}")
    for b, ix in enumerate(idx):
        ixs = ",".join(map(str, ix))
        for k in range(len(names)):
            lines.append(f"b{b}_m{k}\t@M{k}:@T\t{ixs}")
    out = subprocess.run([ARGS.panel, "--batch", "-", "--stats", "srocc"], input="\n".join(lines) + "\n",
                         check=True, capture_output=True, text=True).stdout.splitlines()
    hdr = out[0].split("\t")
    boot = np.full((ARGS.resamples, len(names)), np.nan)
    for line in out[1:]:
        f = dict(zip(hdr, line.split("\t")))
        b, k = f["label"][1:].split("_m")
        boot[int(b), int(k)] = float(f["srocc"])
    ref = names.index("zensim A (PQ, 1000 nit peak)")
    for k, n in enumerate(names):
        d = boot[:, ref] - boot[:, k]
        results[n]["boot"] = {
            "srocc_ci95": [float(np.percentile(boot[:, k], 2.5)), float(np.percentile(boot[:, k], 97.5))],
            "delta_vs_zensimA_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_zensimA_better": float(np.mean(d > 0)),
        }

    with open(os.path.join(ARGS.out, "aichdr2025_panel.json"), "w") as f:
        json.dump({"dataset_commit": "33ab64e5", "n": 100, "seed": ARGS.seed,
                   "resamples": ARGS.resamples, "metrics": results}, f, indent=1)
    write_md(results)
    plot(metrics, target, codec, source, results)


def write_md(results):
    order = sorted(results, key=lambda n: -results[n]["all"]["srocc"])
    L = ["| # | metric | SROCC [95% CI] | PLCC | KROCC | OR | PWRC | Z-RMSE | mean within-source SROCC | ΔSROCC A−metric [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, n in enumerate(order, 1):
        a, b = results[n]["all"], results[n]["boot"]
        ws = np.mean([v["srocc"] for v in results[n]["per_source"].values()])
        d = b["delta_vs_zensimA_ci95"]
        L.append(f"| {i} | {n} | {a['srocc']:.3f} [{b['srocc_ci95'][0]:.3f}, {b['srocc_ci95'][1]:.3f}] | "
                 f"{a['plcc']:.3f} | {a['krocc']:.3f} | {a['or']:.2f} | {a['pwrc']:.3f} | {a['z_rmse']:.3f} | "
                 f"{ws:.3f} | {d[0]:+.3f} … {d[1]:+.3f} |")
    L += ["", "Per-codec SROCC", "", "| metric | AVIF | JPEG-AI | JPEG-XL | JPEG-XT |", "|---|---|---|---|---|"]
    for n in order:
        pc = results[n]["per_codec"]
        L.append(f"| {n} | " + " | ".join(f"{pc[c]['srocc']:.3f}" for c in ("AVIF", "JPEG-AI", "JPEG-XL", "JPEG-XT")) + " |")
    open(os.path.join(ARGS.out, "aichdr2025_panel.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def plot(metrics, target, codec, source, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["zensim A (PQ, 1000 nit peak)", "SSIMULACRA 2", "Butteraugli-pnorm", "Butteraugli-max",
            "CVVDP", "HDR-VDP-3 (detect)", "PU2-MS-SSIM", "MS-SSIM_after TM", "IW-SSIM after TM",
            "VMAF after TM", "PSNR-HVS (PQ)", "LGFM"]
    show = [s for s in show if s in metrics]
    order = sorted(metrics, key=lambda n: -results[n]["all"]["srocc"])
    colors = {"AVIF": "#1f77b4", "JPEG-AI": "#d62728", "JPEG-XL": "#2ca02c", "JPEG-XT": "#9467bd"}
    marks = dict(zip(sorted(set(source)), "osD^v"))
    for fname, names in (("scatter_key.png", show), ("scatter_all.png", order)):
        cols = 4
        rows = math.ceil(len(names) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
        for ax, n in zip(axes.flat, names):
            for c, s, x, y in zip(codec, source, metrics[n], target):
                ax.scatter(x, y, s=18, c=colors[c], marker=marks[s], alpha=0.85, linewidths=0)
            a = results[n]["all"]
            ax.set_title(f"{n}\nSROCC {a['srocc']:.3f}  PLCC {a['plcc']:.3f}", fontsize=9)
            ax.set_ylabel("JND", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
        for ax in list(axes.flat)[len(names):]:
            ax.axis("off")
        handles = [plt.Line2D([], [], ls="", marker="o", color=v, label=k) for k, v in colors.items()]
        handles += [plt.Line2D([], [], ls="", marker=v, color="gray", label=k) for k, v in marks.items()]
        fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=8)
        fig.suptitle("AIC-HDR2025: metric vs reconstructed JND (n=100; 5 sources × 4 codecs × 5 levels)", fontsize=11)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        fig.savefig(os.path.join(ARGS.out, fname), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
