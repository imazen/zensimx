"""E36 compression-stratified EVAL readout (record: R2 amendment 1, 2026-11-03).

The EVAL MOS collections mix many distortion types; this readout keeps their compression rows only, the product's
use: CID:IQ JPEG and JPEG 2000, Colourlab JPEG and JPEG 2000, IQA-ODI (JPEG at every projection), IRCCyN-IVC JPEG,
JPEG 2000, LAR and the two JPEG colour variants, NITS-IQA D7 JPEG, D8 JPEG 2000 and D9 JPEG-XT, and VCL-FER JPEG and
JPEG 2000 (417 rows). Multiply distorted rows (LIVE-MD blur+JPEG, MDID) and CID:OA's optical aberrations are left out.
Distortion names come from each collection's library `labels.tsv` (the `distortion` column only; the labels used
are the EVAL rows' already-ledgered targets).

Per model and collection: SROCC with a whole-reference 95% interval (canonical `panel rank-interval`), within-reference
SROCC, and the median absolute residual around the panel's 4PL fit over the collection's compression rows, divided by
their target's 1-99% span (the experiment 5 readout's error). Summary: the row-weighted mean SROCC and the mean error.

    ceval.py EVAL_OUT OUT.json NAME=MODEL[,NAME=MODEL...]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def compression_rows(out):
    rows = [r for r in de.read_eval_rows(out, "e36-compression-stratified") if r["kind"] == "mos"]
    reads = json.loads((out / "LABEL_READS.json").read_text())["label_values_read_for"]
    keep = []
    for c, kinds in COMPRESSION.items():
        want = {r["row"] for r in rows if r["collection"] == c}
        dist = {}
        with open(reads[c]["table"], newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            col = next(reader).index("distortion")
            for i, rec in enumerate(reader):
                if i in want:
                    dist[i] = rec[col]
        keep += [r for r in rows if r["collection"] == c and dist[r["row"]] in kinds]
    return keep


def main(out, dest, spec):
    out = Path(out)
    models = dict(s.split("=", 1) for s in spec.split(","))
    rows = compression_rows(out)
    scores = de.load_scores(out)
    res = defaultdict(dict)
    for name, m in models.items():
        for c in COMPRESSION:
            rr = [r for r in rows if r["collection"] == c]
            ci, n = de.mos_interval(rr, scores[m])
            y = np.array([r["target"] for r in rr])
            x = np.array([scores[m][r["a"]] for r in rr])
            span = float(np.quantile(y, 0.99) - np.quantile(y, 0.01))
            err = float(np.median(np.abs(np.asarray(logistic_fit(x, y)) - y)) / span)
            res[name][c] = dict(srocc=ci["signed"]["estimate"], lo=ci["signed"]["lo"], hi=ci["signed"]["hi"],
                                within=de.within(rr, scores[m]), err=err, n=n)
        w = {c: res[name][c]["n"] for c in COMPRESSION}
        res[name]["summary"] = dict(srocc_rows=sum(res[name][c]["srocc"] * w[c] for c in w) / sum(w.values()),
                                    err_mean=float(np.mean([res[name][c]["err"] for c in COMPRESSION])),
                                    rows=sum(w.values()))
    Path(dest).write_text(json.dumps(res, indent=1) + "\n")
    print("| model | row-weighted SROCC | mean error | " + " | ".join(COMPRESSION) + " |")
    print("|---|---:|---:|" + "---|" * len(COMPRESSION))
    for name in models:
        s = res[name]["summary"]
        print(f"| {name} | {s['srocc_rows']:.4f} | {100 * s['err_mean']:.2f}% | "
              + " | ".join(f"{res[name][c]['srocc']:.3f}" for c in COMPRESSION) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
