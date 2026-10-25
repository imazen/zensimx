"""Formula revision 7 candidates: score every pair of `pairs_scoring.tsv` with the default scorer and the
stable-forms scorer for each bake, and report the score differences. Exploration record:
benchmarks/byconstruction_2026-10-22.md ("Formula revision 7"). The 2026-10-24 run built the second scorer with
`--cfg zensim_stable_forms`, since removed: revision 7 now selects the forms, so the two binaries today are one build
scoring a revision-6 and a revision-7 stamp of the same bake (`bake_stamp_revision`).

  python3 rev7_score_compare.py DIR NAME=BAKE [NAME=BAKE ...]

DIR holds pairs_scoring.tsv (label, ref_path, dist_path), score_pair_with_bake.default and .stable; writes
DIR/score_compare_2026-10-24.json.
"""
import concurrent.futures as cf, json, os, subprocess, sys
import numpy as np
R = sys.argv[1]
bakes = dict(a.split('=', 1) for a in sys.argv[2:])
env = {k: v for k, v in os.environ.items() if k not in ('ZENSIM_RING_FORM', 'ZENSIM_FORMULA_REV')}
pairs = [l.split('\t') for l in open(f'{R}/pairs_scoring.tsv').read().splitlines()[1:]]
def score(binary, bake, r, d):
    out = subprocess.run([binary, '--bake', bake, '--ref', r, '--dist', d], capture_output=True, text=True, check=True, env=env)
    return float(out.stdout.strip().splitlines()[-1])
res = {}
for name, bake in bakes.items():
    def one(p):
        l, r, d = p
        return l, score(f'{R}/score_pair_with_bake.default', bake, r, d), score(f'{R}/score_pair_with_bake.stable', bake, r, d)
    with cf.ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(one, pairs))
    diffs = np.array([s - c for _, c, s in rows])
    res[name] = {'rows': rows, 'max_abs': float(np.abs(diffs).max()), 'p99_abs': float(np.quantile(np.abs(diffs), 0.99)),
                 'median_abs': float(np.median(np.abs(diffs)))}
    print(name, f"max |d| {res[name]['max_abs']:.3e}  p99 {res[name]['p99_abs']:.3e}  median {res[name]['median_abs']:.3e}")
    for l, c, s in sorted(rows, key=lambda r: -abs(r[2] - r[1]))[:6]:
        print(f"   {l:40s} default {c:.6f} stable {s:.6f} diff {s - c:+.3e}")
json.dump(res, open(f'{R}/score_compare_2026-10-24.json', 'w'), indent=1)
