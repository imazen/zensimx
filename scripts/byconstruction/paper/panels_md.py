"""Markdown for the white paper's Full panels tab from paper_data.py's panels.json: one table per tier, a row per
dataset, subset and model, all six statistics plus the logistic-fit error tails.

  python3 panels_md.py DATA_DIR > panels.md"""
import json
import sys
from pathlib import Path

panels = json.loads((Path(sys.argv[1]) / "panels.json").read_text())
ORDER = ["General model", "Codec model", "Mlic (start)", "SSIMULACRA2", "Butteraugli 3-norm", "CVVDP"]
DS = {"AIC2026 public_train": "AIC-2026 public_train", "KADID": "KADID-10k"}


def f(v, n=4):
    return "" if v is None else f"{v:.{n}f}"


out = []
for tier, lead in (("TRAIN population", "Selection data. AIC-2026 public_train is out of fold for the models (each crop scored by a fold model that never saw its source) and zero-shot for the peers; KADID-10k, TID2013, KonFiG and CID22-A are out of sample for the models (seed-0 E21 cells that never trained on the set). “Codecs” keeps the compression types only (KADID JPEG and JPEG 2000; TID2013 JPEG, JPEG 2000 and lossy compression of noisy images; KonFiG JPEG 2000; every CID22-A row)."),
                   ("EVAL (frozen, read once)", "The frozen EVAL read: both packed seed-0 models, Mlic's seed 0 and the peers on every EVAL row of nine MOS collections.")):
    rows = [p for p in panels if p["tier"] == tier]
    out.append(f"## {tier}\n\n{lead}\n")
    out.append("| Dataset | Subset | Model | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | fit p99 | out4 |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    datasets = list(dict.fromkeys(p["dataset"] for p in rows))
    for ds in datasets:
        for sub in ("all", "codec"):
            if sub == "codec" and ds in ("AIC2026 public_train", "CID22-A"):
                continue  # every row is a codec there: the codec rows repeat the full set
            sel = [p for p in rows if p["dataset"] == ds and p["subset"] == sub]
            for m in ORDER:
                for p in [p for p in sel if p["model"] == m]:
                    out.append(f"| {DS.get(ds, ds)} | {'every type' if sub == 'all' else 'codecs'} | {m} | {p['n']:,} | "
                               f"{f(abs(p['srocc']) if p['srocc'] is not None else None)} | {f(p['plcc'])} | {f(p['krocc'])} | "
                               f"{f(p['or'], 3)} | {f(p['pwrc'])} | {f(p['z_rmse'], 3)} | {f(p['fit_p99'], 3)} | {f(p['out4'], 4)} |")
    out.append("")
out.append("SROCC is shown as a magnitude (every score oriented so higher is better). PLCC and Z-RMSE are after the panel's four-parameter logistic fit; OR is the outlier ratio; PWRC is the perceptually weighted rank correlation; fit p99 is the 99th percentile of the absolute error around that logistic fit, as a share of the label's 1st-to-99th-percentile range; out4 is the share of points outside zenstats' four-robust-scale envelope on the rank-mapped residual, left empty where zenstats does not measure geometry (IQA-ODI's 48 rows).")
print("\n".join(out))
