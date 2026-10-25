"""Per distortion type (or codec): mean (pooled prediction rank - pooled human rank), out of sample, for each arm and
for E34's base-c, seeds 0-2 averaged. Negative = the model rates that type worse than humans do. Each set's `spread` is
the SD of its per-type biases (equal weight per type): 0 when one score means one quality for every type, so on
CID22-A (codecs) it is the codec-uniformity statistic. Type names: KADID-10k and TID2013 type order (label = type - 1);
KonFiG and CID22-A by index.

  type_bias.py [ARM ...] [--out JSON]   (cell directory names under ~/tmp/byc/cells; default mono-thr)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

panel = json.load(open('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json'))
cells = Path.home() / 'tmp/byc/cells'
KN = {1: 'gblur', 2: 'lensblur', 3: 'motionblur', 4: 'colordiffuse', 5: 'colorshift', 6: 'colorquant', 7: 'sat1',
      8: 'sat2', 9: 'jp2k', 10: 'jpeg', 11: 'whitenoise', 12: 'colornoise', 13: 'impulse', 14: 'multnoise',
      15: 'denoise', 16: 'brighten', 17: 'darken', 18: 'meanshift', 19: 'jitter', 20: 'noneccpatch', 21: 'pixelate',
      22: 'quantize', 23: 'colorblock', 24: 'sharpen', 25: 'contrast'}
TN = {1: 'awgn', 2: 'awgn-color', 3: 'spatial-noise', 4: 'masked-noise', 5: 'hf-noise', 6: 'impulse', 7: 'quant',
      8: 'gblur', 9: 'denoise', 10: 'jpeg', 11: 'jp2k', 12: 'jpeg-tx', 13: 'jp2k-tx', 14: 'pattern-noise',
      15: 'blockwise', 16: 'meanshift', 17: 'contrast', 18: 'saturation', 19: 'mult-noise', 20: 'comfort-noise',
      21: 'lossy-noisy', 22: 'cquant-dither', 23: 'chrom-aberr', 24: 'sparse-sampling'}


def preds(arm: str, d: dict, s: int) -> np.ndarray:
    if arm in d['predictions']:
        return np.array(d['predictions'][arm][s])
    return pd.read_csv(cells / arm / f"without_{d['name']}_s{s}" / f"pred_{d['name']}.tsv", sep='\t').pred.to_numpy()


def bias(arm: str, d: dict) -> dict:
    t = np.array(d['target'], dtype=float) * float(d['orientation'])
    cod = np.array(d['codec']) + 1
    n = len(t)
    rt = rankdata(t) / n
    names = {'kadid': KN, 'tid2013': TN}.get(d['name'], {})
    per = {}
    for typ in sorted(set(cod.tolist())):
        m = cod == typ
        per[names.get(typ, str(typ - 1))] = float(np.mean(
            [np.mean(rankdata(preds(arm, d, s))[m] / n - rt[m]) for s in (0, 1, 2)]))
    v = np.array(list(per.values()))
    return {'per_type': per, 'spread': float(np.std(v)), 'max_abs': float(np.max(np.abs(v)))}


def main() -> int:
    argv = sys.argv[1:]
    out_path = None
    if '--out' in argv:
        i = argv.index('--out')
        out_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    arms = argv or ['mono-thr']
    out = {}
    for d in panel['datasets']:
        out[d['name']] = {arm: bias(arm, d) for arm in ['base-c', *arms]}
        row = '  '.join(f"{arm} {out[d['name']][arm]['spread']:.4f}" for arm in ['base-c', *arms])
        print(f"== {d['name']}: spread of per-type rank bias (SD across types)  {row}")
        if d['name'] in ('kadid', 'tid2013'):
            base = out[d['name']]['base-c']['per_type']
            for arm in arms:
                a = out[d['name']][arm]['per_type']
                worst = sorted(a, key=lambda k: a[k] - base[k])[:5]
                print(f"   {arm}: most under-rated against base-c: "
                      + ', '.join(f"{k} {a[k]:+.3f} (base-c {base[k]:+.3f})" for k in worst))
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
