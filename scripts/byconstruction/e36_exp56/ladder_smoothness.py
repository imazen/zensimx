"""Dial smoothness on E36's held-out ladders, label-free: how often a step up the encoder's quality setting scores
worse, and how much of a ladder's score movement goes the wrong way, for zensim models and the two reference metrics.

A ladder is one reference view x codec x mode on the baseline panel, from the 40 selection sources only (no arm trains
on them), in quality-setting order. A step whose two encodes decode to identical pixels is skipped (encoders repeat
their output at some settings). Every metric is read on the same steps: the views every listed model scored (E36's
scored views are those whose reference is below 1 MP). Scores are oriented so that higher is better (butteraugli's
3-norm is negated).

Per step: reversal = the higher setting scores strictly worse; tie = exactly equal. Per ladder: wrong-way share = the sum
of downward moves over the sum of absolute moves (0 for a monotone ladder; scale-free, so metrics on different scales
compare). Reported overall, per codec and per content class: reversal and tie shares pooled over steps, the wrong-way
share's mean over ladders, and the share of ladders with no reversal; and the reversal share of step pairs 1, 2, 4, 8
and 16 steps apart. A model given at several seeds is reported as the mean of its per-seed statistics.

    ladder_smoothness.py OUT.json NAME=TABLE:COLUMN[+COLUMN...] ...
TABLE is a score directory under E36's features root (`scores`, `scores-R3`, ...); TABLE `cells` reads E36's own
`butteraugli_pnorm3` (negated) or `cvvdp_standard_fhd`."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path('/mnt/v/zensim/e36-features/2026-11-01')
CELLS = Path.home() / 'tmp/e36feat/full/cells.parquet'
SIGN = {'butteraugli_pnorm3': -1.0, 'cvvdp_standard_fhd': 1.0}


def load(specs):
    c = pq.read_table(CELLS, columns=['codec', 'content', 'q', 'split', 'mode_key', 'panel', 'reference_sha256',
                                      'distorted_sha256', 'distorted_pixels_sha256', *SIGN]).to_pandas()
    c = c[(c.split == 'selection') & (c.panel == 'baseline')].drop(columns=['split', 'panel'])
    cols = {}
    for name, spec in specs:
        table, columns = spec.split(':', 1)
        cols[name] = []
        for col in columns.split('+'):
            key = f'{name}#{col}'
            if table == 'cells':
                c[key] = SIGN[col] * c[col]
            else:
                s = pq.read_table(ROOT / table / 'scores.parquet', columns=['encoded_sha256', col]).to_pandas()
                s = s.drop_duplicates('encoded_sha256').set_index('encoded_sha256')[col]
                c[key] = c.distorted_sha256.map(s)  # the score tables' `encoded_sha256` is the cells' distorted_sha256
            cols[name].append(key)
    keys = [k for v in cols.values() for k in v]
    scored = c[keys].notna().all(axis=1)
    print(f'{len(c)} selection baseline cells, {int(scored.sum())} scored by every model', flush=True)
    if not scored.any():
        raise SystemExit('no cell is scored by every model')
    c = c[scored]
    lad = ['reference_sha256', 'codec', 'mode_key']
    # A few size-32 views were generated twice in different bundles: same setting, same encoded bytes. Keep one copy.
    c = c.drop_duplicates(lad + ['q', 'distorted_sha256'])
    dup = c.duplicated(lad + ['q'], keep=False)
    if dup.any():
        raise SystemExit(f'{int(dup.sum())} cells give one quality setting two different encodes inside a ladder')
    c = c.sort_values(lad + ['q']).reset_index(drop=True)
    first = (c[lad] != c[lad].shift()).any(axis=1)
    same_pixels = c.distorted_pixels_sha256 == c.distorted_pixels_sha256.shift()
    step = ~first & ~same_pixels
    c['ladder'] = first.cumsum()
    return c, cols, step


def stats(c, step, key):
    d = c[key].diff()
    d[~step] = np.nan
    g = pd.DataFrame({'ladder': c.ladder, 'codec': c.codec, 'content': c.content, 'rev': d < 0, 'tie': d == 0,
                      'down': (-d).clip(lower=0), 'moved': d.abs()})[step]
    per = g.groupby('ladder').agg(codec=('codec', 'first'), content=('content', 'first'), steps=('rev', 'size'),
                                  rev=('rev', 'sum'), tie=('tie', 'sum'), down=('down', 'sum'), moved=('moved', 'sum'))
    per['wrong'] = np.where(per.moved > 0, per.down / per.moved.where(per.moved > 0, 1.0), 0.0)

    # Steps L apart along the same ladder (identical-pixel repeats removed first): the share scored strictly worse.
    k = c[key][step | (c.ladder != c.ladder.shift())].reset_index(drop=True)
    ladder = c.ladder[step | (c.ladder != c.ladder.shift())].reset_index(drop=True)
    codec = c.codec[step | (c.ladder != c.ladder.shift())].reset_index(drop=True)
    lags = {}
    for lag in (1, 2, 4, 8, 16):
        same = ladder == ladder.shift(-lag)
        dd = (k.shift(-lag) - k)[same]
        lags[f'lag{lag}'] = float((dd < 0).mean())
        for v, m in dd.groupby(codec[same]):
            lags[f'lag{lag} codec={v}'] = float((m < 0).mean())

    def summary(p):
        return dict(ladders=int(len(p)), steps=int(p.steps.sum()), reversal=float(p.rev.sum() / p.steps.sum()),
                    tie=float(p.tie.sum() / p.steps.sum()), wrong_way=float(p.wrong.mean()),
                    clean_ladders=float((p.rev == 0).mean()))
    out = {'all': summary(per), 'lags': lags}
    for by in ('codec', 'content'):
        for v, p in per.groupby(by):
            out[f'{by}={v}'] = summary(p)
    return out


def main(dest, specs):
    c, cols, step = load(specs)
    res = {}
    for name, keys in cols.items():
        per_seed = [stats(c, step, k) for k in keys]
        res[name] = {s: {m: float(np.mean([p[s][m] for p in per_seed])) for m in per_seed[0][s]} for s in per_seed[0]}
        print(f"{name}: reversal share at lags 1/2/4/8/16: "
              + ' / '.join(f"{100 * res[name]['lags'][f'lag{lag}']:.2f}%" for lag in (1, 2, 4, 8, 16)), flush=True)
        res[name]['seeds'] = len(keys)
    Path(dest).write_text(json.dumps(res, indent=1) + '\n')
    strata = list(res[next(iter(res))].keys())
    strata.remove('seeds')
    strata.remove('lags')
    print('| model | ' + ' | '.join(strata) + ' |')
    print('|---|' + '---:|' * len(strata))
    for name in res:
        print(f'| {name} | ' + ' | '.join(f"{100 * res[name][s]['reversal']:.2f}% / {100 * res[name][s]['wrong_way']:.2f}%"
                                         for s in strata) + ' |')
    print('cells: reversal share of steps / mean wrong-way share of ladders')


if __name__ == '__main__':
    main(sys.argv[1], [a.split('=', 1) for a in sys.argv[2:]])
