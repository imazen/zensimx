#!/usr/bin/env python3
"""Revision-7 feature sidecar for the chroma-trade gauntlet (evif-chroma-trade-archive-2026-10-30).

The gauntlet ledgers keep scores only. Its `.evif` samples (every sample except DE
non-finds, which were kept as bitstreams) carry the decoded pixels the judges saw, so the
frozen stage-C extractor (`evif-rev7-features-v1`, extractor ed495741, feature set
basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af) runs on them unchanged; zenmetrics
`evif_features.py` reads them out of the run's CAS tar packs (`evif_pack_index.py`).

    harvest LEDGER_DIR CAS INDEX_DIR OUT
    score BANK OUT --predict BIN --bake NAME=PATH ...
    augment SCORES ASSEMBLED OUT --key NAME=LEDGER_KEY ...

LEDGER_DIR holds the zenfleet ledger parquet chunks of the feature run, CAS the run's
objects, INDEX_DIR the pack index (selection.jsonl, MANIFEST.json). OUT gets
features.parquet (f0..f1824, float64, NaN for unrequested slots, as in the stage-C bank),
keys.parquet (one row per `.evif` sample, keyed on `encoded_sha256`, the ledger's bitstream
hash) and _MANIFEST.json. Every feature row is checked against the extractor's audit
(reference and distorted pixel hashes, canonical f32 feature digest), as stage C did.

`score` runs bakes over the bank (`predict_features_with_bake --f64-wire --production`,
as chroma_ladder_gate.py does) into scores.parquet keyed on encoded_sha256. `augment`
copies each gauntlet job's ledger.jsonl into OUT/<job>/ with the named scores added
under new ledger keys, for evif `tools/trade_gate.py --models ...`; rows without
features (DE non-finds) are copied unchanged. When a key already exists in the ledger
(a bake the gauntlet ran inline), augment refuses unless the values are bit-identical,
which checks the bank against the judges row by row.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
from rev5_bank import read_feature_csv  # noqa: E402

EXTRACTOR_SHA = 'ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8'
BUILD = 'b77883be18db5933a9db4b9e2670a2ba88a9729b'
FEATURE_ID = 'basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af'
IDS = list(range(228)) + list(range(372, 720))


def sha(data):
    return hashlib.sha256(data).hexdigest()


_OWNER = {}
_SOURCES = {}


def read_object(archive, digest):
    data = (archive / digest[:2] / digest[2:4] / digest).read_bytes()
    if sha(data) != digest:
        raise ValueError(f'CAS object {digest} differs')
    return data


def harvest_job(archive, row):
    """Verify one feature job's outputs; return (feature columns, keys without row ids, objects read)."""
    knobs = json.loads(row['knob_tuple_json'])
    read = []

    def get(digest):
        read.append(digest)
        return read_object(archive, digest)
    result = json.loads(get(row['output_sha']))
    assert result['schema'] == 'evif-rev7-features-v1' and result['feature_set_id'] == FEATURE_ID
    assert result['extractor_sha256'] == EXTRACTOR_SHA and result['build_commit'] == BUILD
    assert result['feature_width'] == 1825 and result['measured_feature_ids'] == IDS
    assert result['image_path'] == row['image_path'] and result['knob_tuple_json'] == row['knob_tuple_json']
    cells = {c['evif_sha256']: c for c in result['cells']}
    assert len(cells) == len(result['cells']) and set(cells) == set(knobs['evif_shas'])
    views = {_OWNER[d][0] for d in cells}
    assert len(views) == 1
    source = _SOURCES[views.pop()]
    assert source['image_path'] == row['image_path'] and source['pixels_sha256'] == result['reference_pixels_sha256']
    blobs = {name: get(digest) for name, digest in result['sidecars'].items()}
    assert sha(blobs['features.csv']) == result['features_csv_sha256']
    with tempfile.NamedTemporaryFile(dir=Path.home() / 'tmp', suffix='.csv') as f:
        f.write(blobs['features.csv'])
        f.flush()
        feature_ids = sorted({c['row_id'] for c in cells.values()})
        assert feature_ids == list(range(len(feature_ids)))
        raw, columns = read_feature_csv(Path(f.name), feature_ids, 1825)
    audits = [json.loads(line) for line in blobs['features.audit.jsonl'].splitlines()]
    audit_by_png = {Path(a['distorted']).stem: a for a in audits}
    assert len(audits) == len(audit_by_png) == len(feature_ids)
    raw_f32 = np.column_stack([raw[f'f{i}'].to_numpy() for i in range(1825)]).astype('<f4')
    digest_by_row = {}
    keys, take = [], []
    for digest in sorted(cells):
        cell = cells[digest]
        audit = audit_by_png[cell['png_sha256']]
        assert not audit['model_inputs'] and audit['reference'] == row['image_path']
        assert audit['reference_pixels_sha256'] == result['reference_pixels_sha256']
        assert audit['distorted_pixels_sha256'] == cell['distorted_pixels_sha256']
        rid = cell['row_id']
        if rid not in digest_by_row:
            digest_by_row[rid] = sha(raw_f32[rid].tobytes())
        assert audit['canonical_features_f32_le_sha256'] == digest_by_row[rid]
        view, trade_job = _OWNER[digest]
        keys.append(dict(encoded_sha256=cell['jpeg_sha256'], evif_sha256=digest, png_sha256=cell['png_sha256'],
                         distorted_pixels_sha256=cell['distorted_pixels_sha256'],
                         reference_pixels_sha256=result['reference_pixels_sha256'],
                         view_id=view, trade_job=trade_job, image_path=row['image_path'],
                         width=source['width'], height=source['height'],
                         canonical_features_f32_le_sha256=digest_by_row[rid], feature_primary_sha256=row['output_sha']))
        take.append(rid)
    return pa.table(columns).take(pa.array(take)), keys, read


def harvest(ledger_dir, archive, index, out, workers):
    out.mkdir(parents=True)
    for line in (index / 'selection.jsonl').open():
        sel = json.loads(line)
        _SOURCES[sel['view']] = sel['source']
        for digest in sel['evif_shas']:
            _OWNER.setdefault(digest, (sel['view'], sel['job']))
    rows = pq.read_table(sorted(Path(ledger_dir).glob('*.parquet'))).to_pylist()
    done = {}
    for r in rows:
        if r['status'] == 'done':
            done.setdefault(r['job_id'], r)
    duplicates = sum(1 for r in rows if r['status'] == 'done') - len(done)
    ordered = sorted(done.values(), key=lambda r: r['job_id'])
    verified = set()
    writer = key_writer = None
    buffers, keys, count, seen = [], [], 0, set()
    # Workers inherit the selection maps by fork; results come back in job order.
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'))
    try:
        for index_n, (features, job_keys, read) in enumerate(pool.map(harvest_job, [archive] * len(ordered), ordered, chunksize=4)):
            verified.update(read)
            numbered = []
            for k in job_keys:
                assert k['evif_sha256'] not in seen
                seen.add(k['evif_sha256'])
                numbered.append(dict(row_id=count, **k))
                count += 1
            job_keys = numbered
            features = features.add_column(0, 'row_id', pa.array([k['row_id'] for k in job_keys], type=pa.int64()))
            buffers.append(features)
            keys.extend(job_keys)
            if len(keys) >= 5000 or index_n + 1 == len(ordered):
                bank = pa.concat_tables(buffers)
                key_table = pa.Table.from_pylist(keys)
                if writer is None:
                    writer = pq.ParquetWriter(out / 'features.parquet', bank.schema, compression='zstd', compression_level=3)
                    key_writer = pq.ParquetWriter(out / 'keys.parquet', key_table.schema, compression='zstd')
                writer.write_table(bank, row_group_size=5000)
                key_writer.write_table(key_table, row_group_size=5000)
                buffers, keys = [], []
                (out / 'PROGRESS.json').write_text(json.dumps(dict(jobs=index_n + 1, required=len(done), rows=count)) + '\n')
    finally:
        pool.shutdown(cancel_futures=True)
        if writer:
            writer.close()
            key_writer.close()
    selected = len(_OWNER)
    manifest = dict(schema='rev5-featbank-v1', set='chroma-trade-gauntlet', rows=count, selected_containers=selected,
                    missing_containers=selected - count, feature_jobs=len(done), duplicate_executions=duplicates,
                    feature_width=1825, dtype='float64', absent_slots='NaN', requested_slot_ranges=[[0, 228], [372, 720]],
                    feature_set_id=FEATURE_ID, build_commit=BUILD, binary_sha256=EXTRACTOR_SHA,
                    index_manifest_sha256=sha((index / 'MANIFEST.json').read_bytes()),
                    features_parquet_sha256=sha((out / 'features.parquet').read_bytes()),
                    keys_sha256=sha((out / 'keys.parquet').read_bytes()), unique_objects_verified=len(verified),
                    scope='every .evif sample of the 2,755 gauntlet trade jobs; DE non-finds were stored as bitstreams only and are not here',
                    join='keys.encoded_sha256 = gauntlet ledger encoded_sha256')
    (out / '_MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest))


def score(bank, out, predict, bakes):
    out.mkdir(parents=True)
    keys = pq.read_table(bank / 'keys.parquet', columns=['encoded_sha256']).column(0)
    features = pq.ParquetFile(bank / 'features.parquet')
    names = [f'f{i}' for i in range(1825)]
    columns = {name: [] for name, _ in bakes}
    for batch in features.iter_batches(batch_size=20000, columns=names):
        matrix = np.column_stack([batch.column(i).to_numpy(zero_copy_only=False) for i in range(1825)]).astype('<f8')
        with tempfile.NamedTemporaryFile(dir=Path.home() / 'tmp', suffix='.bin') as f:
            f.write(struct.pack('<II', matrix.shape[1], matrix.shape[0]) + matrix.tobytes())
            f.flush()
            for name, path in bakes:
                values = subprocess.run([str(predict), '--bake', str(path), '--features-file', f.name, '--f64-wire',
                                         '--production'], capture_output=True, text=True, check=True).stdout.split()
                if len(values) != len(matrix):
                    raise ValueError(f'{name}: {len(values)} scores for {len(matrix)} rows')
                columns[name].extend(float(v) for v in values)
    table = pa.table({'encoded_sha256': keys, **{n: pa.array(v, type=pa.float64()) for n, v in columns.items()}})
    pq.write_table(table, out / 'scores.parquet', compression='zstd')
    manifest = dict(rows=table.num_rows, bank_manifest_sha256=sha((bank / '_MANIFEST.json').read_bytes()),
                    predict_sha256=sha(Path(predict).read_bytes()),
                    bakes={n: dict(path=str(p), sha256=sha(Path(p).read_bytes())) for n, p in bakes},
                    scores_sha256=sha((out / 'scores.parquet').read_bytes()))
    (out / 'SCORED.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest))


def augment(scores, assembled, out, keys):
    table = pq.read_table(scores / 'scores.parquet').to_pydict()
    by_encode = {}
    for i, digest in enumerate(table['encoded_sha256']):
        row = {key: table[name][i] for name, key in keys}
        if by_encode.setdefault(digest, row) != row:
            raise ValueError(f'{digest}: one bitstream, two feature rows with different scores')
    out.mkdir(parents=True)
    counts = dict(rows=0, scored=0, unscored=0, checked_existing=0)
    for job in sorted(p for p in assembled.iterdir() if p.is_dir()):
        (out / job.name).mkdir()
        with (job / 'ledger.jsonl').open() as src, (out / job.name / 'ledger.jsonl').open('w') as dst:
            for line in src:
                r = json.loads(line)
                counts['rows'] += 1
                extra = by_encode.get(r.get('encoded_sha256')) if 'scores' in r else None
                if extra is None:
                    counts['unscored'] += 'scores' in r
                    dst.write(line)
                    continue
                for key, value in extra.items():
                    if key in r['scores']:
                        if struct.pack('<d', r['scores'][key]) != struct.pack('<d', value):
                            raise ValueError(f'{job.name}: {key} {r["scores"][key]!r} != bank {value!r}')
                        counts['checked_existing'] += 1
                    r['scores'][key] = value
                counts['scored'] += 1
                dst.write(json.dumps(r) + '\n')
    (out / 'AUGMENTED.json').write_text(json.dumps(dict(counts, scores_sha256=sha((scores / 'scores.parquet').read_bytes()),
                                                        keys=dict(keys)), indent=1) + '\n')
    print(json.dumps(counts))


def pairs(values):
    return [tuple(v.split('=', 1)) for v in values]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    h = sub.add_parser('harvest')
    for name in ['ledger_dir', 'cas', 'index', 'out']:
        h.add_argument(name, type=Path)
    h.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('score')
    s.add_argument('bank', type=Path)
    s.add_argument('out', type=Path)
    s.add_argument('--predict', type=Path, required=True)
    s.add_argument('--bake', action='append', required=True, help='NAME=PATH')
    g = sub.add_parser('augment')
    g.add_argument('scores', type=Path)
    g.add_argument('assembled', type=Path)
    g.add_argument('out', type=Path)
    g.add_argument('--key', action='append', required=True, help='NAME=LEDGER_KEY (score column -> ledger score key)')
    a = ap.parse_args()
    if a.command == 'harvest':
        harvest(a.ledger_dir, a.cas, a.index, a.out, a.workers)
    elif a.command == 'score':
        score(a.bank, a.out, a.predict, pairs(a.bake))
    else:
        augment(a.scores, a.assembled, a.out, pairs(a.key))


if __name__ == '__main__':
    main()
