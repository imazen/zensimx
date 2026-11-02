"""Experiment 5 readout (zensimx benchmarks/e36_subset_test_2026-11-01.md): per arm, error against human ratings on
the EVAL MOS collections, computed as human_error.py does (median and p90 absolute residual around the collection's
4-parameter logistic fit, over the target's 1-99% span), plus SROCC.

Registered outcome: mean over the EVAL MOS collections the arm did NOT train on. Also reported: the mean over the
three collections no arm trains on (iqa-odi, nits-iqa, vcl-fer), the one set every arm shares.

    readout.py EVAL_OUT ARM=MODEL[,ARM=MODEL...] OUT.json   (MODEL = the score name in EVAL_OUT; one per seed: ARM@s)
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path.home() / 'work/zen/zensimx/scripts/byconstruction'))
sys.path.insert(0, str(Path.home() / 'work/zen/zensimx/scripts'))
import dataset_eval as de
from aic_panel import logistic_fit

TRAINED = {  # arm source -> EVAL MOS collection holding that dataset's eval split
    'mdid': 'mdid', 'live-md': 'live-md', 'cid-iq': 'cid-iq', 'cid-oa': 'cid-oa', 'irccyn-ivc': 'irccyn-ivc',
    'colourlab': 'colourlab-rating-variability-2022'}
ALL_DS = set(TRAINED)
COMMON = ['iqa-odi', 'nits-iqa', 'vcl-fer']


def trained_on(arm):
    base = arm.rsplit('@', 1)[0]
    if base in ('control', 'allsub', 'all-equal'):
        return set(TRAINED.values())
    if base == 'none':
        return set()
    parts = base.split('+')
    if not set(parts) <= set(TRAINED) | {'kadid', 'tid2013', 'konfig', 'cid22_a25', 'pipal'}:
        return set(TRAINED.values())  # not an experiment 5 arm: the full recipe
    return {TRAINED[s] for s in parts if s in TRAINED}


out, spec, dest = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
models = dict(a.split('=', 1) for a in spec.split(','))
rows = de.read_eval_rows(out, 'e36-experiment5')
scores = de.load_scores(out)
per = defaultdict(dict)
for c in sorted({r['collection'] for r in rows if r['kind'] == 'mos'}):
    rr = [r for r in rows if r['collection'] == c]
    y = np.array([r['target'] for r in rr], np.float64)
    span = float(np.quantile(y, 0.99) - np.quantile(y, 0.01))
    for arm, m in models.items():
        x = np.array([scores[m][r['a']] for r in rr], np.float64)
        res = np.abs(np.asarray(logistic_fit(x, y)) - y)
        per[arm][c] = dict(med=float(np.median(res) / span), p90=float(np.quantile(res, 0.9) / span),
                           srocc=float(abs(spearmanr(x, y).statistic)), n=len(y))
cols = sorted(next(iter(per.values())))
summary = {}
for arm, v in per.items():
    held = [c for c in cols if c not in trained_on(arm)]
    summary[arm] = dict(heldout=held, heldout_med=float(np.mean([v[c]['med'] for c in held])),
                        heldout_srocc=float(np.mean([v[c]['srocc'] for c in held])),
                        common_med=float(np.mean([v[c]['med'] for c in COMMON])),
                        common_srocc=float(np.mean([v[c]['srocc'] for c in COMMON])),
                        all_med=float(np.mean([v[c]['med'] for c in cols])))
json.dump(dict(per=per, summary=summary, models=models), open(dest, 'w'), indent=1)
print('| arm | held-out MOS sets | held-out median | common-3 median | common-3 SROCC | all-9 median |')
print('|---|---:|---:|---:|---:|---:|')
for arm, s in sorted(summary.items(), key=lambda kv: kv[1]['heldout_med']):
    print(f"| {arm} | {len(s['heldout'])} | {100*s['heldout_med']:.2f}% | {100*s['common_med']:.2f}% | "
          f"{s['common_srocc']:.4f} | {100*s['all_med']:.2f}% |")
