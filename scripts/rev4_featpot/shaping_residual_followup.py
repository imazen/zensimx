"""Paired error geometry from cached, already exposed outer predictions.

This is hypothesis discovery, never a new independent confirmation. No labels
from a final holdout are read and no model is trained or selected here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

from shaping_completion import encoded, sha


def ranks(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('finite nonempty paired population required')
    return (rankdata(values) - .5) / len(values)


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def geometry(target, product, tree):
    """Compare model ranks with target ranks; scales are population specific."""
    qy, qp, qt = ranks(target), ranks(product), ranks(tree)
    ep, et = qp - qy, qt - qy
    ap, at = np.abs(ep), np.abs(et)
    # A descriptive oracle ceiling and fixed equal-rank blend, not serving models.
    pbad, tbad = ap >= np.quantile(ap, .9), at >= np.quantile(at, .9)
    return dict(rows=len(qy), product_srocc=correlation(qy, qp), tree_srocc=correlation(qy, qt),
        equal_rank_blend_srocc=correlation(qy, ranks((qp+qt)*.5)),
        signed_rank_error_correlation=correlation(ep, et),
        product_rank_mae=float(ap.mean()), tree_rank_mae=float(at.mean()),
        product_p99_rank_error=float(np.quantile(ap, .99)),
        tree_p99_rank_error=float(np.quantile(at, .99)),
        oracle_rank_mae=float(np.minimum(ap, at).mean()),
        tree_better_fraction=float((at < ap).mean()),
        product_top_decile_rows=int(pbad.sum()), tree_top_decile_rows=int(tbad.sum()),
        shared_top_decile_rows=int((pbad & tbad).sum()),
        tree_better_on_product_top_decile_fraction=float((at[pbad] < ap[pbad]).mean()),
        product_better_on_tree_top_decile_fraction=float((ap[tbad] < at[tbad]).mean())), (ep, et, qp, qt, qy)


def build(study, output):
    from shaping_outer_evidence import collect
    from shaped_linear_study import OLD
    if output.exists():
        raise ValueError('preserve existing follow-up output')
    plan, partitions, rows, pins, _ = collect(study)
    data = {name: np.load(OLD/'data'/f'{name}.npy', allow_pickle=False)
            for name in ('pop', 'y', 'refs')}
    for name in data:
        pins[str(OLD/'data'/f'{name}.npy')] = sha(OLD/'data'/f'{name}.npy')
    paired = {}; samples = []; summaries = []
    for row in rows:
        if row['role'] not in ('best_product', 'extra_trees'):
            continue
        key = row['cell']['partition'], row['cell']['seed']
        test = np.asarray(partitions[key[0]]['test'], dtype=np.int64)
        lookup = {i: j for j, i in enumerate(row['indices'])}
        pred = np.asarray([row['predictions'][lookup[int(i)]] for i in test])
        paired.setdefault(key, {})[row['role']] = (test, pred)
    if len(paired) != 9 or any(set(p) != {'best_product', 'extra_trees'} for p in paired.values()):
        raise ValueError('all nine paired outer conditions required')
    for (partition, seed), pair in sorted(paired.items()):
        test, prod = pair['best_product']; ti, tree = pair['extra_trees']
        if not np.array_equal(test, ti):
            raise ValueError('paired membership differs')
        for pop in sorted(set(data['pop'][test])):
            mask = data['pop'][test] == pop; ids = test[mask]
            stats, (ep, et, qp, qt, qy) = geometry(data['y'][ids], prod[mask], tree[mask])
            summaries.append(dict(partition=partition, seed=seed, population=str(pop), **stats))
            samples.extend(dict(index=int(i), partition=partition, seed=seed, population=str(pop),
                source=str(data['refs'][i]), product_rank_error=float(p), tree_rank_error=float(t),
                product_rank=float(pr), tree_rank=float(tr), target_rank=float(yr))
                for i, p, t, pr, tr, yr in zip(ids, ep, et, qp, qt, qy))
    means = {}
    for pop in sorted({r['population'] for r in summaries}):
        local = [r for r in summaries if r['population'] == pop]
        means[pop] = {k: float(np.mean([r[k] for r in local]))
                      for k in local[0] if k not in ('partition', 'seed', 'population')
                      and all(r[k] is not None for r in local)}
    output.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(samples), output/'paired-rank-errors.parquet', compression='zstd')
    result = dict(schema='shaping-residual-discovery-v1', stage='complete',
        scope='Previously exposed source-disjoint outer predictions; development hypothesis discovery only.',
        synthetic_population='cid22_ssim2; reported separately from all human populations',
        final_holdout_read=False, fits=0, paired_conditions=9, panels=summaries,
        equal_fold_seed_means=means, uncertainty='Descriptive only; no new significance or selection claim.',
        fixed_blend_scope='Within-population ranks, unavailable as a standalone deployable score.',
        residual_table_sha256=sha(output/'paired-rank-errors.parquet'))
    (output/'SUMMARY.json').write_bytes(encoded(result))
    (output/'INPUTS.json').write_bytes(encoded(dict(files=pins, driver_sha256=sha(__file__))))
    print(json.dumps(dict(stage='complete', output=str(output), means=means)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.study, args.output)
