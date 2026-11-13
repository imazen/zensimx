"""Publish completed study summaries and cached outer-test scatter points only."""
import argparse
import json
from pathlib import Path
import shutil
import statistics

import numpy as np
from scipy.stats import rankdata

from shaping_completion import ROLES, encoded, sha
from shaping_bootstrap import complete_output


def build(study, dispatch, output, a_data):
    if output.exists():
        raise ValueError('preserve existing final report')
    core = json.loads((dispatch/'final-recovery-20261113/CORE_COMPLETION_AUDIT.json').read_text())
    if core['required'] != 1461 or core['successful'] != 1461 or core['missing']:
        raise ValueError('complete registered core required')
    boot = dispatch/'bootstrap-final'
    status = json.loads((boot/'STATUS.json').read_text())
    if status['stage'] != 'complete' or len(status['archived']) != 25 or len(status['completed']) != 25:
        raise ValueError('all 25 completed and archived uncertainty analyses required')
    summaries = {}
    for name in status['completed']:
        if not complete_output(boot/name):
            raise ValueError('incomplete uncertainty result: '+name)
        summaries[name] = json.loads((boot/name/'SUMMARY.json').read_text())
    registry = json.loads((study.parent/'bootstrap-matched-reduced-20261108/MATCHED_REGISTRATION.json').read_text())
    comparisons = [dict(spec=r['spec'], summary=summaries[Path(r['file']).stem]['scopes']['mean_outer'])
                   for r in registry['comparisons']]
    cost = dispatch/'cost-final-corrected-20261113'
    cost_status = json.loads((cost/'STATUS.json').read_text())
    if cost_status['stage'] != 'complete' or cost_status['timed_models'] != 48 or cost_status['fresh_memory_models'] != 48:
        raise ValueError('all 48 cost and memory measurements required')
    sizes = []
    prepared = cost/'prepared'
    mirror = Path(json.loads((prepared/'MIRROR_VERIFIED.json').read_text())['destination'])
    for outer in range(3):
        fold = prepared/f'outer-{outer}'
        workload = json.loads((fold/'WORKLOAD.json').read_text())
        timing = json.loads((cost/'vector'/f'outer-{outer}-REPORT.json').read_text())
        for arm in workload['arms']:
            model = fold/arm['model']
            if not model.exists():
                model = mirror/f'outer-{outer}'/arm['model']
            if sha(model) != arm['sha256']:
                raise ValueError('timed model identity differs')
            sizes.append(dict(role=arm['id'].rsplit('-', 1)[0], bytes=model.stat().st_size,
                ns=timing['arms'][arm['id']]['forty_row_batch_per_row_arithmetic_mean_ns']['median']))
    costs = {role: dict(models=len(rows), median_bytes=statistics.median(r['bytes'] for r in rows),
                       median_batch_mean_row_ns=statistics.median(r['ns'] for r in rows))
             for role in ROLES for rows in [[r for r in sizes if r['role'] == role]]}
    # Reuse the original admission owner and its pinned cached predictions.
    # No inference, selection, calibration fitting or final-holdout access.
    from shaping_outer_evidence import collect
    from shaped_linear_study import OLD
    plan, partitions, rows, pins, _ = collect(study)
    if len(rows) != 48:
        raise ValueError('all 48 admitted outer models required')
    populations = np.load(OLD/'data/pop.npy', allow_pickle=False)
    targets = np.load(OLD/'data/y.npy', allow_pickle=False)
    points = {}
    for row in rows:
        cell = row['cell']; outer = int(cell['partition'].split('-')[1])
        test = np.asarray(partitions[cell['partition']]['test'], dtype=np.int64)
        lookup = {i: j for j, i in enumerate(row['indices'])}
        pred = np.asarray([row['predictions'][lookup[int(i)]] for i in test], dtype=float)
        for pop in sorted(set(populations[test])):
            chosen = populations[test] == pop
            x, y = pred[chosen], targets[test[chosen]]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('finite cached scatter points required')
            n = len(x)
            qx, qy = (rankdata(x)-.5)/n, (rankdata(y)-.5)/n
            seed = 'deterministic' if cell['seed'] is None else str(cell['seed'])
            key = '|'.join((row['role'], seed, str(pop)))
            points.setdefault(key, []).extend([[round(float(b), 7), round(float(a), 7), outer]
                                              for a, b in zip(qx, qy)])
    output.mkdir(parents=True)
    payload = dict(core={k: v for k, v in core.items() if k != 'rows'}, roles=list(ROLES),
        primary=summaries['primary']['scopes']['mean_outer'], comparisons=comparisons, costs=costs,
        convergence=summaries['primary']['convergence_qualifications'],
        a_data=json.loads(a_data.read_text()), scatter_points=points,
        population_reports=summaries['primary']['outer_population_reports'],
        final_holdout_read=False, serving_model_promoted=False,
        scatter_scope='Cached outer-test rows only; separate within-fold percentile ranks; all three paired seeds selectable.',
        uncertainty='2000 stratified whole-source bootstrap replicates; conditional on fitted models and observed sources; no training-seed uncertainty.')
    (output/'results.json').write_text(json.dumps(payload, allow_nan=False, separators=(',', ':'))+'\n')
    (output/'INPUTS.json').write_bytes(encoded(dict(plan_sha256=sha(study/'PLAN.json'),
        bootstrap_summary_sha256={name: sha(boot/name/'SUMMARY.json') for name in summaries},
        cached_outer_input_sha256=pins, a_data_sha256=sha(a_data), driver_sha256=sha(__file__))))
    shutil.copyfile(Path(__file__).with_name('shaping_final_report.html'), output/'index.html')
    print(json.dumps(dict(output=str(output), fits=1461, uncertainty_analyses=25,
                          timed_models=48, scatter_series=len(points), final_holdout_read=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'dispatch', 'output', 'a-data'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    build(args.study, args.dispatch, args.output, args.a_data)
