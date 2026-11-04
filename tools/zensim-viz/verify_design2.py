#!/usr/bin/env python3
"""Compare completed Design 2 artifacts with protected review-1 bundles.

This compares identities and serialized values; native exporters and tests own
all metric arithmetic. Refuse a partial bundle or an altered existing value.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--prefix', default='design2-')
    args = parser.parse_args()
    result = {'pairs': [], 'weights': {}, 'reused_native_evidence': {}}
    for number in range(1, 7):
        name = f'atlas-pair-{number:02}'
        old, new = [args.root / (p + name) for p in ['review1-', args.prefix]]
        assert (new / 'complete.json').is_file(), f'{new}: incomplete'
        a, b = read(old / 'meta.json'), read(new / 'meta.json')
        assert b['pooled_bit_identity'] is True
        assert canonical(read(old / 'control.json')['bits']) == canonical(read(new / 'control.json')['bits'])
        assert canonical(read(new / 'control.json')['bits']) == canonical(read(new / 'instrumented.json')['bits'])
        assert len(a['features']) == len(b['features']) == 984
        for first, second in zip(a['features'], b['features']):
            assert canonical(first) == canonical({k: second[k] for k in first}), f"{name}: changed feature {first['key']}"
        files = {f['map_file'] for f in a['features'] if f.get('map_file')}
        for filename in files:
            assert sha(old / filename) == sha(new / filename), f'changed map {name}/{filename}'
        terms = sorted({f['terms_file'] for f in b['features'] if f.get('terms_file')})
        assert len(terms) == 12
        samples = 0
        for filename in terms:
            header = read(new / filename)
            assert header['formula_replay_bit_identity'] is True
            count = header['width'] * header['height']
            assert header['formula_replay_bits_checked'] == count
            binary = (new / filename).parent / header['data_file']
            assert binary.stat().st_size == count * len(header['planes']) * 4
            samples += count
        result['pairs'].append(dict(pair=name, pooled_bits=b['pooled_bits_checked'], layout_bits=b['layout_bits_checked'], inputs=984, unchanged_map_files=len(files), basic_term_files=len(terms), native_formula_replay_samples=samples))
    for bake in ['F0', 'F1', 'F2', 'E']:
        old, new = args.root / f'review1-audit-v2-{bake}', args.root / f'{args.prefix}weights-{bake}'
        assert (new / 'complete.json').is_file()
        a, b = read(old / 'weights.json'), read(new / 'weights.json')
        assert canonical(a) == canonical({k:b[k] for k in a}), f'changed original weights {bake}'
        assert len(b['input_l1']) == len(b['inputs'])
        assert read(new / 'identity.json')['bake_sha256'] == read(old / 'audit.json')['bake_sha256']
        result['weights'][bake] = {'inputs':len(b['inputs']), 'original_fields_unchanged':True, 'weights_sha256':sha(new / 'weights.json')}
    manifest=read(args.root / 'manifest.json')
    for bake in manifest['bakes']:
        for bank, data in bake['banks'].items():
            # The bank diagnostics and all measured flag records are reused.
            for field in ['audit', 'summary']:
                path=args.root / data[field].removeprefix('data/')
                result['reused_native_evidence'][f"{bake['id']}/{bank}/{field}"]=sha(path)
    result['reference']=read(args.root/'reference.json')['counts']
    with args.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
