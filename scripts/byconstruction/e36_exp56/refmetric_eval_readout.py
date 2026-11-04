"""EVAL readout for every score in an assessment dir (the candidates' pred_*.tsv and the judge_*.tsv files that
refmetric_judges.py writes), per MOS collection: |SROCC|, the readout's error (median absolute residual around the
4PL fit, as a share of the target's 1-99% span) and coverage. A metric that lacks scores on some rows of a collection
(the AIC-4 Haar configs on IQA-ODI's 8K images) is reported on the collections it covers fully and marked.

    refmetric_eval_readout.py EVAL_DIR OUT.json [NAME=KEY ...]      # default: every key load_scores returns
"""
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

COMMON = ("iqa-odi", "nits-iqa", "vcl-fer")


def main():
    out = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    rows = de.read_eval_rows(out, "e36-R10-amendment-5-reference-metrics")
    scores = de.load_scores(out)
    names = dict(a.split("=", 1) for a in sys.argv[3:]) or {k: k for k in scores}
    mos = [r for r in rows if r["kind"] == "mos"]
    cols = sorted({r["collection"] for r in mos})
    per = defaultdict(dict)
    for c in cols:
        rr = [r for r in mos if r["collection"] == c]
        y = np.array([r["target"] for r in rr], np.float64)
        span = float(np.quantile(y, 0.99) - np.quantile(y, 0.01))
        for name, key in names.items():
            s = scores[key]
            have = [r for r in rr if r["a"] in s]
            if len(have) < len(rr):
                per[name][c] = dict(coverage=len(have) / len(rr), n=len(rr))
                continue
            x = np.array([s[r["a"]] for r in rr], np.float64)
            res = np.abs(np.asarray(logistic_fit(x, y)) - y)
            per[name][c] = dict(coverage=1.0, n=len(rr), srocc=float(abs(spearmanr(x, y).statistic)),
                                error=float(np.median(res) / span))
    summary = {}
    for name, v in per.items():
        full = [c for c in cols if v[c]["coverage"] == 1.0]
        summary[name] = dict(collections=full, missing=[c for c in cols if c not in full],
                             all_error=float(np.mean([v[c]["error"] for c in full])),
                             all_srocc=float(np.mean([v[c]["srocc"] for c in full])),
                             common_error=float(np.mean([v[c]["error"] for c in COMMON if c in full])),
                             common_srocc=float(np.mean([v[c]["srocc"] for c in COMMON if c in full])),
                             common_complete=all(c in full for c in COMMON))
    dest.write_text(json.dumps(dict(per=per, summary=summary), indent=1) + "\n")
    print("| score | collections | mean error | mean SROCC | common three: error / SROCC | " + " | ".join(cols) + " |")
    print("|---|---:|---:|---:|---|" + "---:|" * len(cols))
    for name in sorted(summary, key=lambda n: summary[n]["all_error"]):
        s, v = summary[name], per[name]
        cells = [f"{v[c]['srocc']:.3f}" if v[c]["coverage"] == 1.0 else f"({v[c]['coverage']:.0%})" for c in cols]
        common = f"{s['common_error'] * 100:.2f}% / {s['common_srocc']:.4f}" + ("" if s["common_complete"] else " (partial)")
        print(f"| {name} | {len(s['collections'])} of {len(cols)} | {s['all_error'] * 100:.2f}% | {s['all_srocc']:.4f} | "
              f"{common} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
