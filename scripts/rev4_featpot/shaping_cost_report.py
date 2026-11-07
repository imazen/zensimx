"""Replay complete native cost evidence; report paired-round timing uncertainty."""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
from shaping_completion import encoded, sha


def samples(directory, workload_path):
    pins = {}
    def read(path):
        pins[str(path)] = sha(path)
        return json.loads(path.read_text())
    workload = read(workload_path)
    identity = read(directory/'IDENTITY.json')
    result = read(directory/'RESULT.json')
    if identity['workload_sha256'] != pins[str(workload_path)] or identity['admission_sha256'] != workload['admission_sha256']:
        raise ValueError('workload identity differs')
    vector = workload['schema'] == 'shaping-cost-v1'
    if not vector and workload['schema'] != 'shaping-extract-cost-v1':
        raise ValueError('unknown cost workload')
    if vector:
        names = [arm['id'] for arm in workload['arms']]
        if len(names) != 16 or len(set(names)) != 16 or identity['rows_per_call'] != 40:
            raise ValueError('all sixteen vector arms and forty rows required')
        if identity.get('preflight_exact') is not True or result.get('preflight_exact') is not True:
            raise ValueError('vector preflight not exact')
        for arm in workload['arms']:
            path = workload_path.parent/arm['model']
            pins[str(path)] = sha(path)
            if pins[str(path)] != arm['sha256']:
                raise ValueError('model changed since timing')
    else:
        names = ['pair-0', 'pair-1', 'pair-2']
        if identity.get('preflight_strided_exact') is not True:
            raise ValueError('extraction preflight not exact')
        for key in ('size', 'threads', 'raw_ids', 'pixel_sha256'):
            if identity[key] != workload[key]:
                raise ValueError('extraction identity differs')
    if (identity['mode'] != 'bench' or result.get('timing_measured') is not True
            or result.get('complete') is not True or result.get('postflight_exact') is not True
            or result.get('unreliable') is not False):
        raise ValueError('complete reliable timing with exact postflight required')
    batch_count = result['batches']
    if type(batch_count) is not int or not 1 <= batch_count <= 64:
        raise ValueError('invalid timing batch count')
    expected_files = [directory/f'zenbench-{i:03}.json' for i in range(batch_count)]
    if set(directory.glob('zenbench-*.json')) != set(expected_files):
        raise ValueError('timing batch coverage differs')
    flags, durations = [], []
    for path in expected_files:
        batch = read(path)
        if batch.get('unreliable') is not False or len(batch['comparisons']) != 1:
            raise ValueError('unreliable or mixed timing comparison')
        comparison = batch['comparisons'][0]
        expected_group = 'shaping_native_40_rows' if vector else 'shaping_native_extraction'
        if comparison['group_name'] != expected_group:
            raise ValueError('timing group differs')
        order = [arm['name'] for arm in comparison['benchmarks']]
        if len(order) != len(names) or set(order) != set(names):
            raise ValueError('timing arm coverage differs')
        remap = [order.index(name) for name in names]
        rounds = comparison['samples']
        if comparison['completed_rounds'] != len(rounds) or not rounds:
            raise ValueError('timing round coverage differs')
        for row in rounds:
            count = row['iterations']
            if type(count) is not int or count <= 0 or sorted(row['execution_order']) != list(range(len(names))):
                raise ValueError('incomplete paired round')
            for field in ('elapsed_ns', 'compensated_ns'):
                values = row[field]
                if len(values) != len(names) or any(type(v) is not int or v <= 0 for v in values):
                    raise ValueError('invalid raw timing duration')
            if row['gate_clean'] is not None and type(row['gate_clean']) is not bool:
                raise ValueError('invalid timing gate')
            flags.append(row['gate_clean'])
            durations.append([row['compensated_ns'][i]/count for i in remap])
    retained = [i for i, flag in enumerate(flags) if flag is True][:32]
    if len(flags) > 64 or len(retained) != 32 or flags != result['gate_clean'] or retained != result['retained_round_indices']:
        raise ValueError('first thirty-two clean paired rounds required')
    values = np.asarray(durations, dtype=float)[retained]
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError('invalid normalized timing')
    return names, values, vector, identity, pins, retained


def summarize(names, values, vector):
    # Shared round draws retain pairing across every model. These are timing
    # repeats, never training seeds or independent source-image draws.
    draws = np.random.default_rng(20261107).integers(0, 32, size=(10000, 32))
    def estimate(series):
        distribution = np.median(series[draws], axis=1)
        return dict(median=float(np.median(series)),
                    percentile_95_interval=np.quantile(distribution, [.025, .975]).tolist())
    arms = {}
    for i, name in enumerate(names):
        arms[name] = dict(call_ns=estimate(values[:, i]))
        if vector:
            arms[name]['forty_row_batch_per_row_arithmetic_mean_ns'] = estimate(values[:, i]/40)
    ratios = []
    if vector:
        for i, j in itertools.combinations(range(len(names)), 2):
            ratios.append(dict(numerator=names[j], denominator=names[i],
                               paired_round_time_ratio=estimate(values[:, j]/values[:, i])))
    return dict(arms=arms, paired_ratios=ratios,
                uncertainty=dict(method='paired round percentile bootstrap of median',
                                 draws=10000, seed=20261107, confidence=.95,
                                 scope='conditional timing repeatability; assumes stationary independent rounds; no model-seed or source uncertainty; intervals not multiplicity adjusted'))


def report(directory, workload, output):
    if output.exists():
        raise ValueError('preserve existing cost report')
    names, values, vector, identity, pins, retained = samples(directory, workload)
    result = summarize(names, values, vector)
    for path, pin in pins.items():
        if sha(path) != pin:
            raise ValueError('timing evidence changed during reporting')
    result.update(identity=identity, evidence_sha256=pins, retained_round_indices=retained,
                  normalized_compensated_call_ns=values.tolist(),
                  scope='vector scoring' if vector else 'image extraction; three pairs reported separately',
                  end_to_end_latency_measured=False)
    output.write_bytes(encoded(result))
    return dict(output=str(output), sha256=sha(output), arms=len(names), paired_rounds=32)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('directory', 'workload', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(report(a.directory, a.workload, a.output)))
