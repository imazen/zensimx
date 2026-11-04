"""E36 distortion-stratified EVAL readouts (record: R2 amendment 1, 2026-11-03; non-codec rows added for R10
amendment 5's follow-up, 2026-11-03).

The EVAL MOS collections mix many distortion types. `--rows compression` (the default) keeps the compression rows,
the product's use: CID:IQ JPEG and JPEG 2000, Colourlab JPEG and JPEG 2000, IQA-ODI (JPEG at every projection),
IRCCyN-IVC JPEG, JPEG 2000, LAR and the two JPEG colour variants, NITS-IQA D7 JPEG, D8 JPEG 2000 and D9 JPEG-XT, and
VCL-FER JPEG and JPEG 2000 (417 rows). `--rows noncodec` keeps the rows with no codec in them: CID:IQ blur, Poisson
noise and the two gamut mappings, Colourlab's nineteen other KADID-style distortions and its rated references, CID:OA's
optical aberrations and rated references, NITS-IQA D1-D6 (Gaussian blur, Gaussian noise, uniform noise, contrast
change, pixelate mosaic, motion blur), VCL-FER white noise and blur, IRCCyN-IVC blur, LIVE-MD blur+noise. Multiply distorted rows with a codec in them (LIVE-MD blur+JPEG, MDID) and IQA-ODI (all JPEG) belong to
neither. `--by-distortion` adds one line per distortion type with at least 20 rows. Distortion names come from each
collection's library `labels.tsv` (the `distortion` column only; the labels used are the EVAL rows' already-ledgered
targets).

Per model and collection: SROCC with a whole-reference 95% interval (canonical `panel rank-interval`), within-reference
SROCC, and the median absolute residual around the panel's 4PL fit over the collection's selected rows, divided by
their target's 1-99% span (the experiment 5 readout's error). Summary: the row-weighted mean SROCC and the mean error.

    ceval.py EVAL_OUT OUT.json NAME=MODEL[,NAME=MODEL...] [--rows compression|noncodec] [--by-distortion]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dataset_eval as de  # noqa: E402
from aic_panel import logistic_fit  # noqa: E402

COMPRESSION = {
    "cid-iq": {"JPEG2000_Compression", "JPEG_Compression"},
    "colourlab-rating-variability-2022": {"9 JPEG2000", "10 JPEG"},
    "iqa-odi": {"ERP", "cmp", "cpp", "isp", "ohp"},
    "irccyn-ivc": {"JPEG", "J2000", "LAR", "JPEG_lumi", "JPEG_lumi+chr"},
    "nits-iqa": {"D7", "D8", "D9"},
    "vcl-fer": {"J2K", "JPG"},
}
NONCODEC = {
    "cid-iq": {"DeltaE_Gamut_Mapping", "Gaussian_Blur", "Poisson_Noise", "SGCK_Gamut_Mapping"},
    "colourlab-rating-variability-2022": None,  # every type but the two codecs (filled below)
    "cid-oa": None,  # every row
    "irccyn-ivc": {"Blur"},
    "live-md": {"blur+noise"},
    "nits-iqa": {"D1", "D2", "D3", "D4", "D5", "D6"},
    "vcl-fer": {"AWGN", "BLU"},
}
# Colourlab's EVAL rows are nine per type, too few alone; `--by-distortion` also pools them by KADID family.
FAMILIES = {
    "colourlab-rating-variability-2022": {
        "blur": ["1 gaussian blur", "2 lens blur", "3 motion blur"],
        "colour": ["5 color shift", "6 color quantization", "7 color saturation 1 (decrease)",
                   "8 color saturation 2 (increase)"],
        "noise": ["11 white noise", "14 multiplicative noise", "15 denoise"],
        "brightness/contrast": ["16 brighten", "17 darken", "18 mean shift", "25 contrast increase", "27 contrast decrease"],
        "sharpen/quantize": ["24 high sharpen", "22 quantization"],
        "geometric": ["19 jitter", "26 lens distortion"],
    },
}


def selected_rows(out, mode):
    """Rows of the collections in the chosen set, each with its `distortion` name attached."""
    rows = [r for r in de.read_eval_rows(out, "e36-compression-stratified") if r["kind"] == "mos"]
    reads = json.loads((out / "LABEL_READS.json").read_text())["label_values_read_for"]
    spec = COMPRESSION if mode == "compression" else NONCODEC
    keep = []
    for c, kinds in spec.items():
        want = {r["row"] for r in rows if r["collection"] == c}
        dist = {}
        with open(reads[c]["table"], newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            col = next(reader).index("distortion")
            for i, rec in enumerate(reader):
                if i in want:
                    dist[i] = rec[col]
        for r in rows:
            if r["collection"] != c:
                continue
            name = dist[r["row"]]
            if kinds is None:
                ok = name not in COMPRESSION.get(c, set())
            else:
                ok = name in kinds
            if ok:
                keep.append(dict(r, distortion=name))
    return keep


def stats(rr, s):
    ci, n = de.mos_interval(rr, s)
    y = np.array([r["target"] for r in rr])
    x = np.array([s[r["a"]] for r in rr])
    span = float(np.quantile(y, 0.99) - np.quantile(y, 0.01))
    err = float(np.median(np.abs(np.asarray(logistic_fit(x, y)) - y)) / span)
    return dict(srocc=ci["signed"]["estimate"], lo=ci["signed"]["lo"], hi=ci["signed"]["hi"],
                within=de.within(rr, s), err=err, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("dest")
    ap.add_argument("spec")
    ap.add_argument("--rows", choices=("compression", "noncodec"), default="compression")
    ap.add_argument("--by-distortion", action="store_true")
    ap.add_argument("--min-rows", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    models = dict(s.split("=", 1) for s in a.spec.split(","))
    rows = selected_rows(out, a.rows)
    cols = [c for c in (COMPRESSION if a.rows == "compression" else NONCODEC) if any(r["collection"] == c for r in rows)]
    scores = de.load_scores(out)
    res = defaultdict(dict)
    for name, m in models.items():
        for c in cols:
            res[name][c] = stats([r for r in rows if r["collection"] == c], scores[m])
        w = {c: res[name][c]["n"] for c in cols}
        res[name]["summary"] = dict(srocc_rows=sum(res[name][c]["srocc"] * w[c] for c in w) / sum(w.values()),
                                    err_mean=float(np.mean([res[name][c]["err"] for c in cols])),
                                    rows=sum(w.values()))
    print(f"{a.rows} rows: {len(rows)} in {len(cols)} collections")
    print("| model | row-weighted SROCC | mean error | " + " | ".join(cols) + " |")
    print("|---|---:|---:|" + "---|" * len(cols))
    for name in models:
        s = res[name]["summary"]
        print(f"| {name} | {s['srocc_rows']:.4f} | {100 * s['err_mean']:.2f}% | "
              + " | ".join(f"{res[name][c]['srocc']:.3f}" for c in cols) + " |")
    if a.by_distortion:
        types = sorted({(r["collection"], r["distortion"]) for r in rows})
        bd = {}
        print(f"\n| collection | distortion | n | " + " | ".join(models) + " |")
        print("|---|---|---:|" + "---:|" * len(models))
        for c, dname in types:
            rr = [r for r in rows if r["collection"] == c and r["distortion"] == dname]
            if len(rr) < a.min_rows:
                continue
            y = np.array([r["target"] for r in rr])
            span = float(np.quantile(y, 0.99) - np.quantile(y, 0.01))
            cells = []
            for name, m in models.items():
                x = np.array([scores[m][r["a"]] for r in rr])
                sr = float(abs(spearmanr(x, y).statistic))
                err = float(np.median(np.abs(np.asarray(logistic_fit(x, y)) - y)) / span) if span > 0 else float("nan")
                bd[f"{c}/{dname}/{name}"] = dict(srocc=sr, err=err, n=len(rr))
                cells.append(f"{sr:.3f}")
            print(f"| {c} | {dname} | {len(rr)} | " + " | ".join(cells) + " |")
        for c, fams in FAMILIES.items():
            if not any(r["collection"] == c for r in rows):
                continue
            print(f"\n| {c} family | n | " + " | ".join(models) + " |")
            print("|---|---:|" + "---:|" * len(models))
            for fam, types in fams.items():
                rr = [r for r in rows if r["collection"] == c and r["distortion"] in types]
                y = np.array([r["target"] for r in rr])
                cells = []
                for name, m in models.items():
                    x = np.array([scores[m][r["a"]] for r in rr])
                    sr = float(abs(spearmanr(x, y).statistic))
                    bd[f"{c}/family:{fam}/{name}"] = dict(srocc=sr, n=len(rr))
                    cells.append(f"{sr:.3f}")
                print(f"| {fam} | {len(rr)} | " + " | ".join(cells) + " |")
        res["by_distortion"] = bd
    Path(a.dest).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
