"""Describe retained R14 head pairs without changing the frozen R12 instrument.

The instrument caps zenbench's adaptive sampler at 200 rounds. Historical R12
runs stopped at 30; its historical summary retains its exact-30 requirement.
R14 accepts the instrument's completed adaptive population, preserving every
sample and using identical gate-clean indices for the two measured heads.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


BINARY_SHA = 'aa409c8e689323f7ed0f8ca4f07fa34565d673efa18e26806f98572677af3a33'
CONTROL_SHA = '185f43f7e5e1327c34ad18f0b73b96b872663726a0fac6ff524430d0d8430088'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing(root):
    root = Path(root)
    directory = root / 'heads'
    load = lambda p: json.loads(p.read_text())
    manifest = load(directory / 'MANIFEST.json')
    execution = load(directory / 'EXECUTION.json')
    complete = load(directory / 'COMPLETE.json')
    host = load(directory / 'HOST.json')
    frozen = {f"{r['arm']}_s{r['seed']}": r['packed_sha256']
              for r in load(root / 'FROZEN_FITS.json')['models']}
    if (execution['rc'] != 0 or execution['status'] != 'complete'
            or host['host'] != 'i270'
            or set(complete['models']) != set(frozen)
            or set(manifest['models']) != set(frozen)
            or manifest['binary_sha256'] != BINARY_SHA
            or manifest['A0_sha256'] != CONTROL_SHA
            or sha(directory / 'prune_forward_bench') != BINARY_SHA
            or sha(directory / 'A0.bin') != CONTROL_SHA):
        raise ValueError('complete frozen i270 head cohort required')
    result = {}
    for name, row in manifest['models'].items():
        if sha(directory / row['file']) != row['sha256'] or row['sha256'] != frozen[name]:
            raise ValueError(f'{name}: frozen model identity differs')
        path = directory / f'timing-{name}.json'
        raw = load(path)
        comparison, = raw['comparisons']
        samples = comparison['samples']
        rounds = comparison['completed_rounds']
        if (raw['unreliable'] or comparison['group_name'] != 'predict_transformed'
                or [v['name'] for v in comparison['benchmarks']] != ['A-MLP', 'candidate']
                or not 30 <= rounds <= 200 or rounds != len(samples)
                or rounds % 10):
            raise ValueError(f'{name}: incomplete or unreliable adaptive timing pair')
        for sample in samples:
            if (not isinstance(sample['gate_clean'], bool) or sample['iterations'] <= 0
                    or sorted(sample['execution_order']) != [0, 1]
                    or len(sample['compensated_ns']) != 2
                    or any(not math.isfinite(v) or v < 0 for v in sample['compensated_ns'])):
                raise ValueError(f'{name}: invalid paired sample')
        indices = [i for i, sample in enumerate(samples) if sample['gate_clean']]
        if not indices:
            raise ValueError(f'{name}: no gate-clean paired rounds')

        def values(chosen, index):
            return [samples[i]['compensated_ns'][index] / samples[i]['iterations'] / 256
                    for i in chosen]

        a, b = values(indices, 0), values(indices, 1)
        result[name] = dict(baseline_ns=mean(a), candidate_ns=mean(b),
            baseline_median_ns=median(a), candidate_median_ns=median(b),
            clean=len(indices), total=rounds, clean_paired_indices=indices,
            all_round_baseline_ns=mean(values(range(rounds), 0)),
            all_round_candidate_ns=mean(values(range(rounds), 1)),
            round_cap_reached=rounds == 200,
            precision_convergence='not established by a run ending at the cap' if rounds == 200 else
                'adaptive early stop; retained native output owns convergence',
            model_sha256=row['sha256'], baseline_sha256=CONTROL_SHA,
            raw_json_sha256=sha(path))
    payload = dict(protocol=dict(instrument_sha256=BINARY_SHA,
        zenbench_commit='82ff70a3f0c212cf88f53d6e9c3e02de13f32eec',
        sampler='native auto_rounds; first convergence check 30, then every 10; max_rounds 200',
        summary='same gate-clean indices; descriptive means; no value-based trimming or truncation',
        rows=256, units='ns per image pair', includes='input transforms, scaling and raw network forward',
        excludes='feature extraction, derived product formation and served dial',
        scope='one thread, ZENSIM_MAX_TIER=v3, default process affinity; head-only synthetic positive rows'),
        host=host, pairs=result, source_sha256=sha(__file__),
        manifest_sha256=sha(directory / 'MANIFEST.json'),
        execution_sha256=sha(directory / 'EXECUTION.json'))
    path = directory / 'SUMMARY.json'
    text = json.dumps(payload, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('preserve existing summary; explicitly version a changed analysis')
    path.write_text(text)
    for name, row in result.items():
        print(f"{name}: A={row['baseline_ns']:.3f}, candidate={row['candidate_ns']:.3f} ns/pair; "
              f"clean={row['clean']}/{row['total']}; round_cap_reached={row['round_cap_reached']}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/var/tmp/byc-transforms-20261104'))
    timing(parser.parse_args().root)


if __name__ == '__main__':
    main()
