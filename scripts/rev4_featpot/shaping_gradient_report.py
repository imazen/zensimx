"""Retain the first 32 clean paired rounds from at most two identical-binary batches."""
import argparse
import json
from pathlib import Path
import numpy as np
from shaping_completion import encoded, sha


def report(root, binary, output):
    if output.exists():
        raise ValueError('preserve existing gradient timing report')
    build = json.loads((root / 'inputs/BUILD.json').read_text())
    if sha(binary) != build['binary_sha256']:
        raise ValueError('microprobe binary changed')
    names = ['two_passes', 'paired_pass']; flags = []; values = []; evidence = {}
    batches = sorted(root.glob('timing-v[12]'))
    if not 1 <= len(batches) <= 2:
        raise ValueError('one or two original timing batches required')
    for batch in batches:
        terminal = json.loads((batch / 'RESULT.json').read_text())
        raw = json.loads((batch / 'zenbench.json').read_text())
        if (terminal['features'] != 984 or terminal['hidden'] != 128
                or terminal['preflight_exact'] is not True or terminal['training_changed'] is not False
                or terminal['unreliable'] or raw['unreliable'] or len(raw['comparisons']) != 1):
            raise ValueError('gradient batch scope or reliability differs')
        comparison = raw['comparisons'][0]
        if (comparison['group_name'] != '984_features_128_hidden_two_gradients'
                or comparison['completed_rounds'] != 32 or terminal['rounds'] != 32
                or len(comparison['samples']) != 32):
            raise ValueError('complete raw paired batch required')
        order = [b['name'] for b in comparison['benchmarks']]
        if len(order) != 2 or set(order) != set(names):
            raise ValueError('both gradient arms required')
        indices = [order.index(name) for name in names]
        for sample in comparison['samples']:
            count = sample['iterations']
            if (type(count) is not int or count <= 0 or sorted(sample['execution_order']) != [0, 1]
                    or len(sample['elapsed_ns']) != 2 or len(sample['compensated_ns']) != 2
                    or any(type(v) is not int or v <= 0 for v in sample['compensated_ns'])):
                raise ValueError('invalid paired raw round')
            flags.append(sample['gate_clean'])
            values.append([sample['compensated_ns'][i] / count for i in indices])
        for name in ('RESULT.json', 'zenbench.json'):
            evidence[str(batch / name)] = sha(batch / name)
    retained = [i for i, flag in enumerate(flags) if flag is True][:32]
    if len(flags) > 64 or len(retained) != 32:
        raise ValueError('32 clean paired rounds within the 64-round limit required')
    values = np.asarray(values, dtype=float)[retained]
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError('invalid normalized paired durations')
    draws = np.random.default_rng(20261107).integers(0, 32, size=(10000, 32))
    def estimate(series):
        distribution = np.median(series[draws], axis=1)
        return dict(median=float(np.median(series)), percentile_95_interval=np.quantile(distribution, [.025, .975]).tolist())
    result = dict(complete=True, arms={name: estimate(values[:, i]) for i, name in enumerate(names)},
                  units='nanoseconds per 984-feature dot-product bank',
                  paired_round_time_ratio=estimate(values[:, 1] / values[:, 0]),
                  retained_round_indices=retained, gate_clean=flags, raw_rounds=len(flags),
                  normalized_compensated_call_ns=values.tolist(), evidence_sha256=evidence,
                  binary_sha256=build['binary_sha256'], source_sha256=build['source_sha256'],
                  scope='synthetic dot-product kernel only; total training speed unmeasured',
                  training_changed=False, batch_results_not_reclassified=True,
                  uncertainty=dict(method='paired round percentile bootstrap of median', draws=10000, seed=20261107,
                                   scope='conditional timing repeatability across two processes; no training/source uncertainty'))
    output.write_bytes(encoded(result));print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'binary', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args();report(a.root, a.binary, a.output)
