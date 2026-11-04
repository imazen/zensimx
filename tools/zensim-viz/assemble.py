#!/usr/bin/env python3
"""Publish only completed native bundles into a local static manifest."""
import argparse
import datetime
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = read(args.config)
    root = Path(config['root'])
    manifest = {k: config[k] for k in ['banks', 'note']}
    manifest['pairs'] = []
    for pair in config['pairs']:
        directory = root / pair['directory']
        if not (directory / 'complete.json').is_file():
            continue
        meta = read(directory / 'meta.json')
        if meta['pooled_bit_identity'] is not True:
            raise RuntimeError(f'{directory}: pooled parity missing')
        manifest['pairs'].append({'id': pair['id'], 'label': pair['label'], 'meta': f"data/{pair['directory']}/meta.json"})
    manifest['bakes'] = []
    for bake in config['bakes']:
        result = {'id': bake['id'], 'label': bake['label'], 'banks': {}}
        for bank_id, sources in bake['banks'].items():
            audit_dir = root / sources['audit']
            if not (audit_dir / 'complete.json').is_file():
                continue
            audit = read(audit_dir / 'audit.json')
            target = {'audit': f"data/{sources['audit']}/audit.json", 'weights': f"data/{sources['audit']}/weights.json", 'lenses': {}, 'family_lenses': {}}
            lens_dir = root / sources['lens'] if sources.get('lens') else None
            if lens_dir and (lens_dir / 'complete.json').is_file():
                index = read(lens_dir / 'index.json')
                if index['completed'] != index['required'] or index['bake_sha256'] != audit['bake_sha256'] or index['bank_sha256'] != audit['bank_sha256']:
                    raise RuntimeError(f'{lens_dir}: incomplete or mismatched lens')
                for key in ['lenses', 'family_lenses']:
                    target[key] = {k: f"data/{sources['lens']}/{v}" for k, v in index[key].items()}
                target['summary'] = f"data/{sources['lens']}/{index['summary']}"
                target['monotonicity_flag_count'] = index['monotonicity_flag_count']
                target['monotonicity_flags'] = []
                for curve_name in [*index['lenses'].values(), *index['family_lenses'].values()]:
                    curve = read(lens_dir / curve_name)['pdp']
                    for row, point, delta in curve['monotonicity_violations']:
                        feature = curve['feature']
                        target['monotonicity_flags'].append(dict(kind='monotonicity', severity='critical', location=f"{curve['family'] or ('f' + str(feature))}, bank row {curve['row_indices'][row]}, grid {point}", feature=feature, layer=None, value=curve['ice'][row][point], statistic=delta, threshold=0, reason='Observed served score rises when damage increases; inspect the native curve.'))
            target['comparisons'] = {}
            for other, dirname in sources.get('comparisons', {}).items():
                if (root / dirname / 'complete.json').is_file():
                    target['comparisons'][other] = f'data/{dirname}/compare-f0.json'
            result['banks'][bank_id] = target
        if result['banks']:
            manifest['bakes'].append(result)
    path = root / 'manifest.json'
    if path.exists():
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path.rename(root / f'manifest-{stamp}.bak.json')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Published {len(manifest['pairs'])} verified pairs and {len(manifest['bakes'])} bakes")


if __name__ == '__main__':
    main()
