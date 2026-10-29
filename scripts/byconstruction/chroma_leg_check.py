#!/usr/bin/env python3
"""Read back the complete SDR teacher transport without fitting or label selection."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chroma_pilot import save, sha


def check(root, bank, out):
    assert not out.exists()
    frozen = json.loads((root/'wide/frozen.json').read_text())
    packet_path = root/'datasets/receipt.json'
    assert sha(packet_path) == frozen['dataset_legs_sha256']
    packet = json.loads(packet_path.read_text())
    for relative, digest in frozen['auxiliary_files'].items():
        assert sha(root/relative) == digest, relative
    for relative, digest in frozen['wide_receipts'].items():
        assert sha(root/'wide'/relative/'receipt.json') == digest, relative
    record = next(g for g in packet['groups'] if g['name'] == 'chroma_teacher')
    assert Path(record['path']) == root/'datasets/chroma.parquet'
    assert sha(record['path']) == record['table_sha256']
    assert sha(record['path']+'.manifest.json') == record['manifest_sha256']
    assert sha(record['keys_path']) == record['keys_sha256']
    keys = json.loads(Path(record['keys_path']).read_text())
    declared = json.loads(Path(record['path']+'.manifest.json').read_text())
    assert declared['keys_path'] == record['keys_path']
    assert declared['keys_sha256'] == record['keys_sha256']
    manifest = json.loads((bank/'_MANIFEST.json').read_text())
    assert manifest['qualified_for_leg'] and manifest['band_ac_included']
    assert sha(bank/'features.parquet') == manifest['features_parquet_sha256']
    assert sha(bank/'keys.parquet') == manifest['keys_sha256']
    diagnostic_path = Path(packet['chroma_teacher']['diagnostics_path'])
    assert diagnostic_path == root/'diagnostics/chroma.parquet'
    tables = {True: pq.ParquetFile(record['path']), False: pq.ParquetFile(diagnostic_path)}
    groups = {True: 0, False: 0}
    rows = {True: 0, False: 0}
    families = {True: Counter(), False: Counter()}
    feature_file = pq.ParquetFile(bank/'features.parquet')
    key_file = pq.ParquetFile(bank/'keys.parquet')
    for fb, kb in zip(feature_file.iter_batches(batch_size=1000), key_file.iter_batches(batch_size=1000), strict=True):
        feature = pa.Table.from_batches([fb])
        records = kb.to_pylist()
        assert feature['pair_key'].to_pylist() == [r['pair_key'] for r in records]
        for training in [True, False]:
            indices = [i for i, r in enumerate(records) if (r['mlic_role'] in ['fit', 'new-train']) == training]
            if not indices:
                continue
            expected = feature.take(pa.array(indices))
            actual = tables[training].read_row_group(groups[training])
            assert actual.num_rows == len(indices)
            for i in range(1825):
                assert actual[f'f{i}'].type == pa.float32()
                want = expected[f'f{i}'].to_numpy().astype(np.float32)
                assert np.array_equal(actual[f'f{i}'].to_numpy().view(np.uint32), want.view(np.uint32)), i
            for i in range(1825, declared['training_width']):
                assert np.isnan(actual[f'f{i}'].to_numpy()).all()
            assert actual['human_score'].type == pa.float64()
            selected = [records[i] for i in indices]
            targets = np.array([r['teacher_ssim2'] for r in selected])
            assert np.array_equal(actual['human_score'].to_numpy().view(np.uint64), targets.view(np.uint64))
            refs = ['chroma:'+r['view_id'] for r in selected]
            assert actual['ref_basename'].to_pylist() == refs
            if training:
                for offset, (r, ref) in enumerate(zip(selected, refs, strict=True)):
                    k = keys[rows[True]+offset]
                    assert k['observation'] == r['pair_key'] and k['reference'] == k['source'] == ref
                    assert k['mlic_role'] == r['mlic_role'] and k['codec_family'] == r['family']
                    assert k['ladder'] == r['view_id']+'|'+r['family'] and k['rung'] == r['rung']
                    assert k['row'] == rows[True]+offset and k['role'] == 'fit'
            families[training].update(r['family'] for r in selected)
            groups[training] += 1
            rows[training] += len(indices)
    assert all(groups[t] == tables[t].num_row_groups and rows[t] == tables[t].metadata.num_rows for t in tables)
    assert rows[True] == len(keys) == record['rows']
    assert rows[False] == packet['chroma_teacher']['diagnostic_rows']
    assert sum(rows.values()) == manifest['rows']
    assert set(families[True]) == set(families[False]) == {'ycc444', 'ycc420', 'xyb-capped', 'xyb-extended', 'band-ac'}
    save(out, dict(passed=True, root=str(root), frozen_sha256=sha(root/'wide/frozen.json'),
                   feature_bank_manifest_sha256=sha(bank/'_MANIFEST.json'), fitting_rows=rows[True],
                   diagnostic_rows=rows[False], fitting_family_rows=dict(families[True]),
                   diagnostic_family_rows=dict(families[False]), selected_features=len(packet['feature_ids']),
                   transport='all1825 f32 columns exact; tailsNaN; every f64 target bit exact; ordered source/role/family/rung exact',
                   training_launched=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('bank', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    check(a.root, a.bank, a.out)
