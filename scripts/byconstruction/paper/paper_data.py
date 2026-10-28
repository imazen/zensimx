"""White-paper tables and figure rows from existing outputs (no new statistics, no label read beyond what the inputs
already hold): full six-statistic panels (TRAIN-population curated stats with peers, EVAL scatter stats) and binned
scatter rows (2-D histograms over every point, so density and tails stay visible).

  python3 paper_data.py CURATE_DIR EVAL_DIR OUT_DIR"""
import json
import sys
from pathlib import Path

import numpy as np

cur, ev, out = (Path(p) for p in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
stats = json.loads((cur / "curate_stats.json").read_text())
KEYS = {"xref256": "General model", "aiconly": "Codec model", "mlic": "Mlic (start)", "ssim2": "SSIMULACRA2",
        "butter3": "Butteraugli 3-norm", "butter": "Butteraugli 3-norm", "cvvdp": "CVVDP"}
SIX = ("srocc", "plcc", "krocc", "or", "pwrc", "z_rmse")
GEO = ("out4", "p99d", "maxd", "clump", "clampLo", "clampHi", "fit_p99", "fit_max")
panels = []
for s in stats:
    if s.get("table") != "cscatter" or s["key"] not in KEYS:
        continue
    panels.append(dict(tier="TRAIN population", dataset=s["dataset"], subset=s["subset"], model=KEYS[s["key"]],
                       sample=s["sample"], n=s["n"], **{k: s.get(k) for k in SIX + GEO}))
evs = json.loads((ev / "eval_scatter_stats.json").read_text())
NAMES = {"cid-iq": "CID:IQ", "cid-oa": "CID:OA", "colourlab-rating-variability-2022": "Colourlab", "iqa-odi": "IQA-ODI",
         "irccyn-ivc": "IRCCyN/IVC", "live-md": "LIVE-MD", "mdid": "MDID", "nits-iqa": "NITS-IQA", "vcl-fer": "VCL@FER"}
for s in evs:
    if s["key"] not in KEYS:
        continue
    panels.append(dict(tier="EVAL (frozen, read once)", dataset=NAMES[s["collection"]], subset="all", model=KEYS[s["key"]],
                       sample="held out", n=s["n"], **{k: s.get(k) for k in SIX + GEO}))
(out / "panels.json").write_text(json.dumps(panels, indent=0))


def bins(x, y, nb=28):
    """Counts on an nb x nb grid over the 0.5-99.5 percentile box of each axis (clipped points land on the edge)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    xl, xh = np.percentile(x, [0.5, 99.5])
    yl, yh = np.percentile(y, [0.5, 99.5])
    xi = np.clip(((x - xl) / (xh - xl + 1e-12) * nb).astype(int), 0, nb - 1)
    yi = np.clip(((y - yl) / (yh - yl + 1e-12) * nb).astype(int), 0, nb - 1)
    grid = np.zeros((nb, nb), int)
    np.add.at(grid, (xi, yi), 1)
    cells = [dict(i=int(i), j=int(j), c=int(grid[i, j])) for i, j in zip(*np.nonzero(grid))]
    return dict(n=int(ok.sum()), xl=float(xl), xh=float(xh), yl=float(yl), yh=float(yh), nb=nb, cells=cells)


figs = {}
aic = json.loads((cur / "curate_aic.json").read_text())
aicp = json.loads((cur / "curate_aic_peers.json").read_text())
e21 = json.loads((cur / "curate_e21.json").read_text())
e21p = json.loads((cur / "curate_e21_peers.json").read_text())
# AIC views sit on the JND scale (higher = worse): plot quality = -jnd against each score mapped by mean/SD ("a_" view).
for k in ("xref256", "aiconly", "mlic"):
    figs[("AIC-2026 public_train", KEYS[k])] = bins([-r[f"a_{k}"] for r in aic if f"a_{k}" in r], [-r["jnd"] for r in aic if f"a_{k}" in r])
for k in ("ssim2", "butter3"):
    figs[("AIC-2026 public_train", KEYS[k])] = bins([-r[f"a_{k}"] for r in aicp], [-r["jnd"] for r in aicp])
DS = {"kadid": "KADID-10k", "tid2013": "TID2013", "konfig": "KonFiG", "cid22_a25": "CID22-A"}
for ds, name in DS.items():
    rows = [r for r in e21 if r["ds"] == ds]
    prow = [r for r in e21p if r["ds"] == ds]
    for k in ("xref256", "aiconly", "mlic"):
        sel = [r for r in rows if f"a_{k}" in r]
        figs[(name, KEYS[k])] = bins([r[f"a_{k}"] for r in sel], [r["label"] for r in sel], nb=20)
    for k in ("ssim2", "butter3"):
        figs[(name, KEYS[k])] = bins([r[f"a_{k}"] for r in prow], [r["label"] for r in prow], nb=20)
evr = json.loads((ev / "eval_scatter.json").read_text())
for c, name in NAMES.items():
    rows = [r for r in evr if r["ds"] == c]
    for k in ("xref256", "aiconly", "mlic", "ssim2", "butter"):
        sel = [r for r in rows if r.get(f"a_{k}") is not None]
        figs[(name, KEYS[k])] = bins([r[f"a_{k}"] for r in sel], [r["label"] for r in sel], nb=16)
(out / "scatter_bins.json").write_text(json.dumps([dict(dataset=d, model=m, **v) for (d, m), v in figs.items()]))
print(len(panels), "panel rows;", len(figs), "binned scatters")
