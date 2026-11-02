#!/usr/bin/env python3
"""A `map-pool-v1` for supervising a bake's attribution map with butteraugli (E36 experiment 3, step 3).

Joins `integrand_dump` output (per view: per-feature block integrands, butteraugli block sums) to the views' E36
bank feature rows. Per view the pool holds the read-set feature row (f32, the `--derived-inputs` read order), the
integrand matrix (rows = the read-set positions of the dumped ids, columns = blocks, f32) and two block targets:
the butteraugli block sums and their within-view rank, scaled to zero mean and unit variance (the Spearman
surrogate the trainer's map term fits).

    map_pool.py PAIRS DUMP_DIR DUMP_JSONL BANK OUT --derived-inputs JSON

PAIRS is the sample's pairs.tsv (`key`, `distorted_pixels_sha256`, `reference_pixels_sha256`). Views whose dump
names an id outside the read set, or that have no bank row, are refused rather than dropped.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e36_features import read_ids  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', type=Path)
    ap.add_argument('dump', type=Path)
    ap.add_argument('dump_jsonl', type=Path)
    ap.add_argument('bank', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--derived-inputs', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True)
    ids = read_ids(a.derived_inputs)
    pos = {f: i for i, f in enumerate(ids)}
    pairs = {r['key']: r for r in (dict(zip(h, l.rstrip('\n').split('\t'))) for h in
                                    [open(a.pairs).readline().rstrip('\n').split('\t')]
                                    for l in list(open(a.pairs))[1:])}
    dumps = [json.loads(l) for l in open(a.dump_jsonl)]
    keys = pq.read_table(a.bank / 'keys.parquet',
                         columns=['distorted_pixels_sha256', 'reference_pixels_sha256', 'row_id']).to_pandas()
    keys = keys.drop_duplicates(['distorted_pixels_sha256', 'reference_pixels_sha256'])
    row_of = {(d, r): int(i) for d, r, i in zip(keys.distorted_pixels_sha256, keys.reference_pixels_sha256, keys.row_id)}
    want = {}
    for d in dumps:
        p = pairs[d['key']]
        rid = row_of.get((p['distorted_pixels_sha256'], p['reference_pixels_sha256']))
        if rid is None:
            raise ValueError(f"{d['key']}: no bank row")
        bad = [i for i in d['ids'] if i not in pos]
        if bad:
            raise ValueError(f"{d['key']}: dumped ids outside the read set: {bad[:5]}")
        want[rid] = d['key']
    feats = {}
    f = pq.ParquetFile(a.bank / 'features.parquet')
    base = 0
    rows = np.array(sorted(want))
    for batch in f.iter_batches(batch_size=50000, columns=[f'f{i}' for i in ids]):
        n = batch.num_rows
        local = rows[(rows >= base) & (rows < base + n)] - base
        if len(local):
            mat = np.column_stack([batch.column(j).to_numpy(zero_copy_only=False) for j in range(len(ids))])[local]
            for i, v in zip(local, mat):
                feats[base + int(i)] = v.astype('<f4')
        base += n
    by_key = {k: r for r, k in want.items()}
    integ = open(a.out / 'integrands.f32', 'wb')
    targ = open(a.out / 'targets.f32', 'wb')
    meta, side, feat_rows, off_i, off_t = [], [], [], 0, 0
    for d in dumps:
        k = d['key']; bx, by = d['blocks']; nb = bx * by; n = len(d['ids'])
        I = np.fromfile(a.dump / f'{k}.integrands.f32', '<f4')
        if I.size != n * nb:
            raise ValueError(f'{k}: integrand size {I.size} != {n}x{nb}')
        bsum = np.fromfile(a.dump / f'{k}.butter_sum.f32', '<f4').astype(np.float64)
        if bsum.size != nb:
            raise ValueError(f'{k}: target size')
        r = rankdata(bsum)
        z = (r - r.mean()) / max(r.std(), 1e-12)
        I = I.reshape(n, nb)
        keep = np.flatnonzero(np.any(I != 0, axis=1))  # all-zero rows (reference-only ids) carry no map
        integ.write(I[keep].astype('<f4').tobytes())
        targ.write(np.stack([bsum, z]).astype('<f4').tobytes())
        x = feats[by_key[k]]
        if not np.isfinite(x).all() or not np.isfinite(I).all():
            raise ValueError(f'{k}: non-finite feature or integrand')
        feat_rows.append(x)
        meta.append(dict(key=k, blocks=nb, positions=[pos[d['ids'][r]] for r in keep],
                         integrand_offset=off_i, target_offset=off_t))
        side.append(dict(key=k, bank_row=by_key[k], blocks_x=bx, blocks_y=by, block=d['block'],
                         zero_rows=int(n - len(keep)), butteraugli=d['butteraugli'], score=d['score']))
        off_i += len(keep) * nb; off_t += 2 * nb
    integ.close(); targ.close()
    pq.write_table(pa.table({'row_idx': pa.array(np.arange(len(meta)), type=pa.uint64()),
                             'features': pa.array([v.tolist() for v in feat_rows], type=pa.list_(pa.float32()))}),
                   a.out / 'features.parquet', compression='zstd')
    (a.out / 'views.json').write_text(json.dumps(meta) + '\n')
    (a.out / 'views_meta.json').write_text(json.dumps(side) + '\n')
    bound = lambda n: dict(path=str((a.out / n).resolve()), sha256=sha(a.out / n))
    manifest = dict(schema='map-pool-v1', features=bound('features.parquet'), views=bound('views.json'),
                    integrands=bound('integrands.f32'), targets=bound('targets.f32'), feature_ids=ids,
                    view_count=len(meta),
                    provenance=dict(targets='per view: butteraugli block sums, then their within-view rank z-scores',
                                    views_meta=bound('views_meta.json'),registration='zensimx benchmarks/e36_subset_test_2026-11-01.md (experiment 3)',
                                    pairs=str(a.pairs.resolve()), pairs_sha256=sha(a.pairs),
                                    dump_jsonl_sha256=sha(a.dump_jsonl), bank=str(a.bank.resolve()),
                                    bank_manifest_sha256=sha(a.bank / '_MANIFEST.json'),
                                    derived_inputs_sha256=sha(a.derived_inputs)))
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(dict(views=len(meta), integrand_floats=off_i, target_floats=off_t)))


if __name__ == '__main__':
    main()
