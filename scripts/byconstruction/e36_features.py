#!/usr/bin/env python3
"""Revision-7 features of the E36 ladders, their scores, inversions, and the --tv-pool of the subset test.

Registration: benchmarks/e36_subset_test_2026-11-01.md. Features come from zenfleet run outputs of zenmetrics
`evif_features.py` over E36's decoded views (`e36_pack_index.py`); the checks are gauntlet_features.py's (extractor
audit: reference and distorted pixel hashes, canonical f32 feature digest), plus E36's own recorded pixel hashes.

    harvest LEDGER_DIR CAS INDEX_DIR OUT     bank: features.parquet + keys.parquet (row per reference x decoded view)
    score BANK OUT --predict BIN --bake NAME=PATH ...      (gauntlet_features.py score; view_id = reference sha)
    inversions SCORES INDEX_DIR PHYSICAL_INDEX OUT --model NAME ...     step 0 tables
    pool BANK SCORES INDEX_DIR PHYSICAL_INDEX OUT --model F3 --derived-inputs JSON   the registered pool

A decoded view can belong to two references (tiny views of different sources that encode to the same pixels), so a
bank row is keyed on (reference_sha256, png_sha256), never the PNG alone.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import collections
import hashlib
import json
import multiprocessing
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
import gauntlet_features as gf  # noqa: E402

SEED = 20261101


def sha(b):
    return hashlib.sha256(b).hexdigest()


_SOURCES = {}
_E36 = {}


def harvest_job(archive, row):
    knobs = json.loads(row['knob_tuple_json'])
    read = []

    def get(digest):
        read.append(digest)
        return gf.read_object(archive, digest)
    result = json.loads(get(row['output_sha']))
    assert result['schema'] == 'evif-rev7-features-v1' and result['feature_set_id'] == gf.FEATURE_ID
    assert result['extractor_sha256'] == gf.EXTRACTOR_SHA and result['build_commit'] == gf.BUILD
    assert result['feature_width'] == 1825 and result['measured_feature_ids'] == gf.IDS
    assert result['image_path'] == row['image_path'] and result['knob_tuple_json'] == row['knob_tuple_json']
    ref = knobs['source_sha256']
    source = _SOURCES[ref]
    assert source['image_path'] == row['image_path'] and source['pixels_sha256'] == result['reference_pixels_sha256']
    cells = {c['evif_sha256']: c for c in result['cells']}
    assert len(cells) == len(result['cells']) and set(cells) == set(knobs['evif_shas'])
    blobs = {name: get(digest) for name, digest in result['sidecars'].items()}
    assert sha(blobs['features.csv']) == result['features_csv_sha256']
    with tempfile.NamedTemporaryFile(dir=Path.home() / 'tmp', suffix='.csv') as f:
        f.write(blobs['features.csv'])
        f.flush()
        feature_ids = sorted({c['row_id'] for c in cells.values()})
        assert feature_ids == list(range(len(feature_ids)))
        raw, columns = gf.read_feature_csv(Path(f.name), feature_ids, 1825)
    audits = [json.loads(line) for line in blobs['features.audit.jsonl'].splitlines()]
    audit_by_png = {Path(a['distorted']).stem: a for a in audits}
    raw_f32 = np.column_stack([raw[f'f{i}'].to_numpy() for i in range(1825)]).astype('<f4')
    keys, take, digests = [], [], {}
    for digest in sorted(cells):
        cell = cells[digest]
        audit = audit_by_png[cell['png_sha256']]
        assert not audit['model_inputs'] and audit['reference'] == row['image_path']
        assert audit['reference_pixels_sha256'] == result['reference_pixels_sha256']
        assert audit['distorted_pixels_sha256'] == cell['distorted_pixels_sha256']
        recorded = _E36[(ref, digest)]
        # E36's own records: the extractor saw exactly the pixels E36 encoded, decoded and scored.
        assert recorded['distorted_pixels_sha256'] == cell['distorted_pixels_sha256']
        assert recorded['reference_pixels_sha256'] == result['reference_pixels_sha256']
        rid = cell['row_id']
        if rid not in digests:
            digests[rid] = sha(raw_f32[rid].tobytes())
        assert audit['canonical_features_f32_le_sha256'] == digests[rid]
        keys.append(dict(view_id=ref, encoded_sha256=digest, reference_sha256=ref, png_sha256=digest,
                         distorted_pixels_sha256=cell['distorted_pixels_sha256'],
                         reference_pixels_sha256=result['reference_pixels_sha256'], image_path=row['image_path'],
                         width=source['width'], height=source['height'],
                         canonical_features_f32_le_sha256=digests[rid], feature_primary_sha256=row['output_sha']))
        take.append(rid)
    return pa.table(columns).take(pa.array(take)), keys, read


def harvest(ledger_dir, archive, index, out, workers):
    out.mkdir(parents=True)
    for line in (index / 'selection.jsonl').open():
        sel = json.loads(line)
        _SOURCES[sel['source']['source_sha256']] = sel['source']
    for r in pq.read_table(index / 'cells.parquet', columns=['reference_sha256', 'distorted_sha256',
                                                             'distorted_pixels_sha256', 'reference_pixels_sha256']).to_pylist():
        _E36.setdefault((r['reference_sha256'], r['distorted_sha256']), r)
    rows = pq.read_table(sorted(Path(ledger_dir).glob('*.parquet'))).to_pylist()
    done = {}
    for r in rows:
        if r['status'] == 'done':
            done.setdefault(r['job_id'], r)
    ordered = sorted(done.values(), key=lambda r: r['job_id'])
    writer = key_writer = None
    buffers, keys, count, seen = [], [], 0, set()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'))
    try:
        for n, (features, job_keys, _) in enumerate(pool.map(harvest_job, [archive] * len(ordered), ordered, chunksize=4)):
            for k in job_keys:
                key = (k['reference_sha256'], k['png_sha256'])
                assert key not in seen, key
                seen.add(key)
                k['row_id'] = count
                count += 1
            features = features.add_column(0, 'row_id', pa.array([k['row_id'] for k in job_keys], type=pa.int64()))
            buffers.append(features)
            keys.extend(job_keys)
            if len(keys) >= 20000 or n + 1 == len(ordered):
                bank, key_table = pa.concat_tables(buffers), pa.Table.from_pylist(keys)
                if writer is None:
                    writer = pq.ParquetWriter(out / 'features.parquet', bank.schema, compression='zstd')
                    key_writer = pq.ParquetWriter(out / 'keys.parquet', key_table.schema, compression='zstd')
                writer.write_table(bank)
                key_writer.write_table(key_table)
                buffers, keys = [], []
    finally:
        pool.shutdown()
        if writer is not None:
            writer.close()
            key_writer.close()
    manifest = dict(schema='e36-rev7-bank-v1', rows=count, jobs=len(ordered),
                    duplicate_done_rows=sum(1 for r in rows if r['status'] == 'done') - len(done),
                    feature_set_id=gf.FEATURE_ID, extractor_sha256=gf.EXTRACTOR_SHA, build_commit=gf.BUILD,
                    index_manifest_sha256=sha((index / 'MANIFEST.json').read_bytes()),
                    features_sha256=sha((out / 'features.parquet').read_bytes()),
                    keys_sha256=sha((out / 'keys.parquet').read_bytes()))
    (out / '_MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest))


def consensus_pairs(index, physical):
    """E36's registered consensus pairs, one row per pair, with both endpoints' (reference, png) keys and strata."""
    cells = pq.read_table(index / 'cells.parquet', columns=['reference_sha256', 'encoded_sha256', 'distorted_sha256',
                                                            'content', 'codec', 'size', 'q', 'split']).to_pandas()
    enc_to_png = dict(zip(zip(cells.reference_sha256, cells.encoded_sha256), cells.distorted_sha256))
    db = sqlite3.connect(f'file:{physical}?mode=ro', uri=True)
    feats = dict(((i), (r, e)) for i, r, e in db.execute('select id, ref, enc from features'))
    out = []
    for a, b, cls, band, codec, content, size, split, qlo, qhi in db.execute(
            'select a, b, class, band, codec, content, size, split, qlo, qhi from pairs'):
        ra, ea = feats[a]
        rb, eb = feats[b]
        assert ra == rb
        out.append((ra, enc_to_png.get((ra, ea)), enc_to_png.get((rb, eb)), codec, content, size, split, qlo, qhi))
    return out


def size_band(s):
    return '<=64' if s <= 64 else ('80-256' if s <= 256 else '320-1024' if s <= 1024 else '>1024')


def q_band(q):
    return '0-20' if q <= 20 else ('22-60' if q <= 60 else '62-100')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    h = sub.add_parser('harvest')
    for x in ('ledger_dir', 'cas', 'index', 'out'):
        h.add_argument(x, type=Path)
    h.add_argument('--workers', type=int, default=16)
    s = sub.add_parser('score')
    s.add_argument('bank', type=Path)
    s.add_argument('out', type=Path)
    s.add_argument('--predict', type=Path, required=True)
    s.add_argument('--bake', action='append', required=True)
    a = ap.parse_args()
    if a.cmd == 'harvest':
        harvest(a.ledger_dir, a.cas, a.index, a.out, a.workers)
    elif a.cmd == 'score':
        gf.score(a.bank, a.out, a.predict, [tuple(b.split('=', 1)) for b in a.bake])


if __name__ == '__main__':
    main()
