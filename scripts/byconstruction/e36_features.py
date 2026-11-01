#!/usr/bin/env python3
"""Revision-7 features of the E36 ladders, their scores, inversions, and the --tv-pool of the subset test.

Registration: benchmarks/e36_subset_test_2026-11-01.md. Features come from zenfleet run outputs of zenmetrics
`evif_features.py` over E36's decoded views (`e36_pack_index.py`); the checks are gauntlet_features.py's (extractor
audit: reference and distorted pixel hashes, canonical f32 feature digest), plus E36's own recorded pixel hashes.

    harvest LEDGER_DIR CAS INDEX_DIR OUT     bank: features.parquet + keys.parquet (row per reference x decoded view)
    score BANK OUT --predict BIN --bake NAME=PATH ...      (gauntlet_features.py score; view_id = reference sha)
    inversions SCORES INDEX_DIR POOLS OUT --model NAME ...     step 0 tables (POOLS = E36 pools-full-v1)
    pool BANK SCORES INDEX_DIR POOLS OUT --model F3 --derived-inputs JSON   the registered pool

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
        # buffersize bounds the finished-but-unwritten jobs; unbounded, they outran the writer to a 24 GB OOM
        for n, (features, job_keys, _) in enumerate(pool.map(harvest_job, [archive] * len(ordered), ordered,
                                                               buffersize=4 * workers)):
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
                # f32 bank: every scoring and training path casts each raw feature to f32 before use
                # (prep_bake_input_f32, derived fill_f32, the minmax path, the trainer), so E/ECodec/F3/F3codec
                # score the smoke bank bit-identically from f32 and f64. Byte-split + zstd: 1.9 KB/row against
                # 3.8 KB/row for f64 byte-split and 14.4 KB/row as worker CSV (4,508-row E36 sample, 2026-11-01).
                bank = bank.cast(pa.schema([pa.field(f.name, pa.float32()) if pa.types.is_floating(f.type) else f
                                            for f in bank.schema]))
                if writer is None:
                    floats = [f.name for f in bank.schema if pa.types.is_floating(f.type)]
                    writer = pq.ParquetWriter(out / 'features.parquet', bank.schema, compression='zstd',
                                              compression_level=19, use_dictionary=False,
                                              use_byte_stream_split=floats)
                    key_writer = pq.ParquetWriter(out / 'keys.parquet', key_table.schema, compression='zstd')
                writer.write_table(bank)
                key_writer.write_table(key_table)
                buffers, keys = [], []
    finally:
        pool.shutdown()
        if writer is not None:
            writer.close()
            key_writer.close()
    manifest = dict(schema='e36-rev7-bank-v1', feature_dtype='float32', rows=count, jobs=len(ordered),
                    duplicate_done_rows=sum(1 for r in rows if r['status'] == 'done') - len(done),
                    feature_set_id=gf.FEATURE_ID, extractor_sha256=gf.EXTRACTOR_SHA, build_commit=gf.BUILD,
                    index_manifest_sha256=sha((index / 'MANIFEST.json').read_bytes()),
                    features_sha256=sha((out / 'features.parquet').read_bytes()),
                    keys_sha256=sha((out / 'keys.parquet').read_bytes()))
    (out / '_MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest))


def consensus_pairs(index, pools):
    """E36's registered consensus pairs (pools-full-v1 edge proofs, fit and selection) as a DataFrame keyed on
    (reference_sha256, png) of the worse and better endpoints, with codec, content, size, quality and split."""
    import pandas as pd
    cells = pq.read_table(index / 'cells.parquet', columns=['reference_sha256', 'encoded_sha256', 'distorted_sha256',
                                                            'content']).to_pandas()
    png = dict(zip(zip(cells.reference_sha256, cells.encoded_sha256), cells.distorted_sha256))
    content = dict(zip(cells.reference_sha256, cells.content))
    frames = []
    for split in ('fit', 'selection'):
        e = pq.read_table(pools / split / 'edge-proof.parquet',
                          columns=['reference_sha256', 'codec', 'size', 'mode_key', 'q_low', 'q_high',
                                   'worse_encoded_sha256', 'better_encoded_sha256']).to_pandas()
        e['split'] = split
        frames.append(e)
    e = pd.concat(frames, ignore_index=True)
    e['worse_png'] = [png.get(k) for k in zip(e.reference_sha256, e.worse_encoded_sha256)]
    e['better_png'] = [png.get(k) for k in zip(e.reference_sha256, e.better_encoded_sha256)]
    e['content'] = e.reference_sha256.map(content)
    e['size_band'] = e['size'].map(size_band)
    e['q_band'] = e.q_low.map(q_band)
    return e


def attach_scores(pairs, scores, models):
    t = pq.read_table(scores / 'scores.parquet').to_pandas().set_index(['view_id', 'encoded_sha256'])
    for side in ('worse', 'better'):
        idx = list(zip(pairs.reference_sha256, pairs[side + '_png']))
        sub = t.reindex(idx)
        for m in models:
            pairs[f'{m}_{side}'] = sub[m].to_numpy()
    pairs['scored'] = np.all([np.isfinite(pairs[f'{m}_{s}']) for m in models for s in ('worse', 'better')], axis=0)
    for m in models:
        pairs[f'{m}_inv'] = pairs[f'{m}_better'] <= pairs[f'{m}_worse']
    return pairs


def inversions(scores, index, pools, out, models):
    out.mkdir(parents=True)
    p = attach_scores(consensus_pairs(index, pools), scores, models)
    report = dict(pairs=len(p), scored=int(p.scored.sum()), unscored_by_reason=dict(
        reference_area_ge_1mp_or_missing=int((~p.scored).sum())), tables={})
    s = p[p.scored]
    for split in ('fit', 'selection'):
        sp = s[s.split == split]
        block = dict(all=dict(pairs=len(sp), **{m: float(sp[f'{m}_inv'].mean()) for m in models}))
        for col in ('codec', 'content', 'size_band', 'q_band'):
            for k, g in sp.groupby(col):
                block[f'{col}={k}'] = dict(pairs=len(g), **{m: float(g[f'{m}_inv'].mean()) for m in models})
        report['tables'][split] = block
        report.setdefault('inverted_counts', {})[split] = {m: int(sp[f'{m}_inv'].sum()) for m in models}
    (out / 'INVERSIONS.json').write_text(json.dumps(report, indent=1) + '\n')
    p.to_parquet(out / 'pairs_scored.parquet', compression='zstd')
    print(json.dumps(dict(pairs=report['pairs'], scored=report['scored'], inverted=report['inverted_counts'],
                          all={k: v['all'] for k, v in report['tables'].items()})))


def pool(bank, scores, index, pools, out, model, derived):
    """The registered pool: F3's inverted fit pairs plus an equal random sample of its correct ones."""
    out.mkdir(parents=True)
    p = attach_scores(consensus_pairs(index, pools), scores, [model])
    fit = p[(p.split == 'fit') & p.scored].reset_index(drop=True)
    inv = fit[fit[f'{model}_inv']]
    ok = fit[~fit[f'{model}_inv']]
    rng = np.random.default_rng(SEED)
    anchor = ok.iloc[np.sort(rng.choice(len(ok), size=min(len(inv), len(ok)), replace=False))]
    chosen = pd_concat([inv, anchor])
    chosen['cls'] = (2 * chosen.content.isin(['screen', 'lineart']).astype(int) + (chosen.q_low <= 40).astype(int)).astype('uint8')
    ids = read_ids(derived)
    keys = pq.read_table(bank / 'keys.parquet', columns=['row_id', 'reference_sha256', 'png_sha256']).to_pandas()
    row_of = dict(zip(zip(keys.reference_sha256, keys.png_sha256), keys.row_id))
    needed = sorted({row_of[(r, w)] for r, w in zip(chosen.reference_sha256, chosen.worse_png)}
                    | {row_of[(r, b)] for r, b in zip(chosen.reference_sha256, chosen.better_png)})
    local = {b: i for i, b in enumerate(needed)}
    f = pq.ParquetFile(bank / 'features.parquet')
    want = set(needed)
    rows = {}
    for batch in f.iter_batches(batch_size=50000, columns=['row_id'] + [f'f{i}' for i in ids]):
        rid = batch.column(0).to_numpy()
        mask = np.isin(rid, list(want))
        if not mask.any():
            continue
        mat = np.column_stack([batch.column(j + 1).to_numpy(zero_copy_only=False) for j in range(len(ids))])[mask]
        for r, v in zip(rid[mask], mat):
            rows[int(r)] = v.astype('<f4')
    assert len(rows) == len(needed)
    feats = pa.table({'row_idx': pa.array(range(len(needed)), type=pa.uint64()),
                      'features': pa.array([rows[b].tolist() for b in needed], type=pa.list_(pa.float32()))})
    pairs = pa.table({'worse_idx': pa.array([local[row_of[(r, w)]] for r, w in zip(chosen.reference_sha256, chosen.worse_png)], type=pa.uint64()),
                      'better_idx': pa.array([local[row_of[(r, b)]] for r, b in zip(chosen.reference_sha256, chosen.better_png)], type=pa.uint64()),
                      'class': pa.array(chosen.cls.to_numpy(), type=pa.uint8())})
    pq.write_table(feats, out / 'features.parquet', compression='zstd')
    pq.write_table(pairs, out / 'pairs.parquet', compression='zstd')
    manifest = dict(schema='tv-pool-v1',
                    features=dict(path=str((out / 'features.parquet').resolve()), sha256=sha((out / 'features.parquet').read_bytes())),
                    pairs=dict(path=str((out / 'pairs.parquet').resolve()), sha256=sha((out / 'pairs.parquet').read_bytes())),
                    feature_ids=ids, class_weights=[1.0, 2.0, 2.0, 4.0], feature_rows=len(needed), pair_rows=len(chosen),
                    provenance=dict(registration='zensimx benchmarks/e36_subset_test_2026-11-01.md', model=model,
                                    inverted=len(inv), anchor=len(anchor), seed=SEED,
                                    bank_manifest_sha256=sha((bank / '_MANIFEST.json').read_bytes()),
                                    scores_manifest_sha256=sha((scores / 'SCORED.json').read_bytes()),
                                    derived_inputs_sha256=sha(Path(derived).read_bytes()),
                                    class_rule='2*[screen or lineart] + [q_low <= 40]',
                                    classes={int(k): int(v) for k, v in chosen.cls.value_counts().items()}))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest['provenance']))


def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def read_ids(derived):
    """The declaration's read set, as the trainer's Fx1Declaration::load_ids computes it."""
    d = json.loads(Path(derived).read_text())
    assert d['schema'] == 'zensim-fx1-v1'
    return sorted(set(d['direct']) | {p[1] for p in d['products']})


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
    i = sub.add_parser('inversions')
    for x in ('scores', 'index', 'pools', 'out'):
        i.add_argument(x, type=Path)
    i.add_argument('--model', action='append', required=True)
    q = sub.add_parser('pool')
    for x in ('bank', 'scores', 'index', 'pools', 'out'):
        q.add_argument(x, type=Path)
    q.add_argument('--model', required=True)
    q.add_argument('--derived-inputs', type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == 'inversions':
        inversions(a.scores, a.index, a.pools, a.out, a.model)
    elif a.cmd == 'pool':
        pool(a.bank, a.scores, a.index, a.pools, a.out, a.model, a.derived_inputs)
    elif a.cmd == 'harvest':
        harvest(a.ledger_dir, a.cas, a.index, a.out, a.workers)
    elif a.cmd == 'score':
        gf.score(a.bank, a.out, a.predict, [tuple(b.split('=', 1)) for b in a.bake])


if __name__ == '__main__':
    main()
