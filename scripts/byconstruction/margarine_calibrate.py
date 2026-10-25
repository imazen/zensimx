"""Margarine's map against butteraugli's block distances on the loop calibration encodes, by loop_calibrate's method
(bulk: every block with butteraugli >= 0.05; tail: each pair's largest tile), at its native placement and fitted.
Exploration record: benchmarks/byconstruction_2026-10-22.md ("Margarine as the steering map").

  python3 margarine_calibrate.py PAIRS_DIR [MARGARINE_CLI | margarine-score]   (margarine-score runs its teacher)
"""
import csv, json, subprocess, sys, tempfile, struct
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from margarine_map_server import write_bmp
MARG = sys.argv[2] if len(sys.argv) > 2 else '/home/lilith/work/zen/zenmetrics/crates/margarine/target/release/margarine'
D = Path(sys.argv[1])
EXP = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
rows = [r for r in csv.DictReader(open(D / 'pairs.tsv'), delimiter='\t') if 'validation' not in r['image']]
maps = []
with tempfile.TemporaryDirectory(dir=str(Path.home() / 'tmp')) as td:
    td = Path(td)
    for r in rows:
        w, h = int(r['width']), int(r['height'])
        ref = np.frombuffer((D / f"{r['index']}.ref.rgba").read_bytes(), np.uint8).reshape(h, w, 4)
        dist = np.frombuffer((D / f"{r['index']}.dist.rgba").read_bytes(), np.uint8).reshape(h, w, 4)
        write_bmp(td / 'r.bmp', ref); write_bmp(td / 'd.bmp', dist)
        m = td / 'm.f32'; m.unlink(missing_ok=True)
        cmd = ([MARG, 'teacher', str(td / 'r.bmp'), str(td / 'd.bmp'), str(m)] if MARG.endswith('margarine-score')
               else [MARG, '--diffmap', str(m), str(td / 'r.bmp'), str(td / 'd.bmp')])
        subprocess.run(cmd, check=True, capture_output=True, env={'RAYON_NUM_THREADS': '4', 'PATH': '/usr/bin:/bin'})
        v = np.fromfile(m, '<f4').astype(np.float64).reshape(h, w)
        b = np.fromfile(D / f"{r['index']}.butter.f32", '<f4').astype(np.float64)
        maps.append((r['set'], float(r['distance']), v, b, w, h))
def tiles(v, w, h, e):
    bx, by = (w + 7) // 8, (h + 7) // 8
    p = np.zeros((by * 8, bx * 8)); n = np.zeros_like(p)
    p[:h, :w] = np.maximum(v, 0) ** e; n[:h, :w] = 1
    s = (p ** 16).reshape(by, 8, bx, 8).sum((1, 3)); c = n.reshape(by, 8, bx, 8).sum((1, 3))
    return (1.2 * (s / c) ** (1 / 16)).ravel()
def stats(xs, ys):
    xs, ys = np.log(np.asarray(xs)), np.log(np.asarray(ys))
    return float(np.sqrt(np.mean((xs - ys) ** 2))), float(np.corrcoef(xs, ys)[0, 1]), int(len(xs))
out = {}
for mode in ('bulk', 'tail'):
    best = None
    for e in EXP:
        fit_x, fit_y, hold_x, hold_y = [], [], [], []
        for s, dd, v, b, w, h in maps:
            t = tiles(v, w, h, e)
            if mode == 'bulk':
                k = (t > 0) & (b >= 0.05); xs, ys = t[k], b[k]
            else:
                xs, ys = [t.max()], [b.max()]
                if not (xs[0] > 0 and ys[0] > 0): continue
            (fit_x if s == 'fit' else hold_x).extend(xs); (fit_y if s == 'fit' else hold_y).extend(ys)
        scale = float(np.exp(np.mean(np.log(fit_y) - np.log(fit_x))))
        f = stats(scale * np.asarray(fit_x), fit_y); hh = stats(scale * np.asarray(hold_x), hold_y)
        if best is None or f[0] < best['fit_rmse']:
            best = {'exponent': e, 'scale': scale, 'fit_rmse': f[0], 'fit_r': f[1], 'fit_n': f[2], 'hold_rmse': hh[0], 'hold_r': hh[1], 'hold_n': hh[2]}
        if e == 1.0:
            nat_f = stats(fit_x, fit_y); nat_h = stats(hold_x, hold_y)
            bias_f = float(np.mean(np.log(fit_x) - np.log(fit_y))); bias_h = float(np.mean(np.log(hold_x) - np.log(hold_y)))
            out[f'{mode}_native'] = {'fit_rmse': nat_f[0], 'fit_r': nat_f[1], 'hold_rmse': nat_h[0], 'hold_r': nat_h[1],
                                     'log_bias_fit': bias_f, 'log_bias_hold': bias_h}
    out[f'{mode}_fitted'] = best
print(json.dumps(out, indent=1))
