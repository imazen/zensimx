"""Summarize R12 native head inspections and paired zenbench samples.

Gate-clean rounds use the same indices for both measured heads. No value-based
trimming or model selection is performed. The complete native JSON stays beside
the summaries. The folded L2 formula is explanatory algebra, not an alternative
floating-point scorer.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def timing(root):
    result = {}
    for name in json.loads((root / 'MODELS.json').read_text()):
        raw = json.loads((root / f'heads/timing-{name}.json').read_text())
        if raw['unreliable']:
            raise ValueError(f'{name}: unreliable timing run')
        comparison, = raw['comparisons']
        samples = comparison['samples']
        clean = [s for s in samples if s['gate_clean']]
        if not clean or len(samples) != 30:
            raise ValueError(f'{name}: incomplete timing pair')
        def values(rows, index):
            return [s['compensated_ns'][index] / s['iterations'] / 256 for s in rows]
        a, b = values(clean, 0), values(clean, 1)
        result[name] = dict(baseline_ns=mean(a), candidate_ns=mean(b),
                            baseline_median_ns=median(a), candidate_median_ns=median(b),
                            clean=len(clean), total=len(samples),
                            all_round_baseline_ns=mean(values(samples, 0)),
                            all_round_candidate_ns=mean(values(samples, 1)))
    (root / 'heads/SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n')


def sparse_formula(root, destination):
    family = json.loads((root / 'FAMILIES.json').read_text())
    with open(family['audit']) as f:
        names = {int(r['feature']): r['name'] for r in csv.DictReader(f, delimiter='\t') if int(r['index']) < 492}
    keep = set(family['keep'])
    weights = []
    products = None
    direct = None
    for seed in range(3):
        d = json.loads((root / f'heads/L2_s{seed}.json').read_text())
        first, last = d['layers']
        if (first['in_dim'], first['out_dim'], last['out_dim']) != (984, 1, 1):
            raise ValueError('L2 is not the registered H1 head')
        assert set(d['scaler_mean']) == {0.}
        assert first['biases'] == [0.] and last['biases'] == [100.]
        assert last['weights'][0] <= 0 and min(first['weights']) >= 0
        declaration = [line.split() for line in d['derived_inputs'].splitlines()[1:]]
        current_direct = [int(v[1]) for v in declaration if v[0] == 'in']
        current_products = [(int(v[1]), int(v[2])) for v in declaration if v[0] == 'product']
        assert len(current_direct) == len(current_products) == 492
        assert [v[0] for v in current_products] == current_direct
        if direct is not None:
            assert direct == current_direct and products == current_products
        direct, products = current_direct, current_products
        folded = [-last['weights'][0] * w / s for w, s in zip(first['weights'], d['scaler_scale'])]
        assert all(folded[i] == folded[i + 492] == 0 for i, feature in enumerate(direct) if feature not in keep)
        weights.append(folded)
    with destination.open('w') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['feature_id', 'feature_name', 'fragility_id',
                         'a_seed0', 'b_seed0', 'a_seed1', 'b_seed1', 'a_seed2', 'b_seed2'])
        for i, feature in enumerate(direct):
            if feature in keep:
                row = [f'f{feature}', names[feature], f'f{products[i][1]}']
                for seed in range(3):
                    row.extend(format(weights[seed][j], '.17g') for j in (i, i + 492))
                writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--formula', type=Path)
    a = ap.parse_args()
    timing(a.root)
    if a.formula:
        sparse_formula(a.root, a.formula)


if __name__ == '__main__':
    main()
