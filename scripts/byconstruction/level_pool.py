#!/usr/bin/env python3
"""A `level-pool-v1` manifest for `zensim_mlp_train --level-pool`: TRAIN chroma ladder rows of the merged chroma bank.

Rows are drawn without replacement (numpy default_rng(SEED)) from the bank's `fit` and `new-train` roles only, never
`dev` or `calibration` (G-CHROMA-L reads those). Each row carries the --derived-inputs read set as f32, as tv-pool
rows do; targets are not stored (the trainer computes them from --level-ref).

    level_pool.py BANK OUT --derived-inputs JSON [--rows 32768]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e36_features import read_ids  # noqa: E402

SEED = 20261102
ROLES = ('fit', 'new-train')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bank', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--derived-inputs', type=Path, required=True)
    ap.add_argument('--rows', type=int, default=32768)
    a = ap.parse_args()
    a.out.mkdir(parents=True)
    keys = pq.read_table(a.bank / 'keys.parquet', columns=['mlic_role', 'family']).to_pandas()
    eligible = np.flatnonzero(keys.mlic_role.isin(ROLES).to_numpy())
    take = np.sort(np.random.default_rng(SEED).choice(eligible, size=a.rows, replace=False))
    ids = read_ids(a.derived_inputs)
    f = pq.ParquetFile(a.bank / 'features.parquet')
    if f.metadata.num_rows != len(keys):
        raise ValueError('bank features and keys differ in length')
    want, rows, base = set(take.tolist()), {}, 0
    for batch in f.iter_batches(batch_size=50000, columns=[f'f{i}' for i in ids]):
        n = batch.num_rows
        local = [i - base for i in take[(take >= base) & (take < base + n)]]
        if local:
            mat = np.column_stack([batch.column(j).to_numpy(zero_copy_only=False) for j in range(len(ids))])[local]
            for i, v in zip(local, mat):
                rows[base + i] = v.astype('<f4')
        base += n
    assert sorted(rows) == take.tolist() and all(np.isfinite(v).all() for v in rows.values())
    feats = pa.table({'row_idx': pa.array(range(len(take)), type=pa.uint64()),
                      'features': pa.array([rows[i].tolist() for i in take], type=pa.list_(pa.float32()))})
    pq.write_table(feats, a.out / 'features.parquet', compression='zstd')
    fam = keys.family.to_numpy()[take]
    manifest = dict(schema='level-pool-v1',
                    features=dict(path=str((a.out / 'features.parquet').resolve()), sha256=sha(a.out / 'features.parquet')),
                    feature_ids=ids, feature_rows=len(take),
                    provenance=dict(registration='zensimx benchmarks/e36_subset_test_2026-11-01.md (amendment 2)',
                                    bank=str(a.bank.resolve()), bank_manifest_sha256=sha(a.bank / '_MANIFEST.json'),
                                    roles=list(ROLES), seed=SEED, bank_rows=[int(i) for i in take],
                                    families={k: int(v) for k, v in zip(*np.unique(fam, return_counts=True))},
                                    derived_inputs_sha256=sha(a.derived_inputs)))
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: v for k, v in manifest['provenance'].items() if k != 'bank_rows'}))


if __name__ == '__main__':
    main()
