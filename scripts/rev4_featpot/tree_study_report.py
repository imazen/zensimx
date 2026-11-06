#!/usr/bin/env python3
"""Exact-key peer/source joins and canonical panels for the registered tree study."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tree_study import load_data, panels, sha, write, estimator, weights_and_targets, native_gate, SEEDS


def unique_index(frame):
    result = frame.set_index('pair_key')
    if not result.index.is_unique:
        raise ValueError('duplicate pair key')
    return result


def metadata(base_path, run, output):
    base = json.loads(base_path.read_text())
    prepared = json.loads((Path(base['output']) / 'PREPARED.json').read_text())
    plan = json.loads((run / 'PLAN.json').read_text())
    if sha(base_path) != plan['baseline_plan_sha256']:
        raise ValueError('plan mismatch')
    data = load_data(run / 'data')
    frames, pins, bank_cache, peer_cache = [], {}, {}, {}
    bank_root = Path('/mnt/v/output/zensim/byconstruction-2026-10-22/rev7raw/bank')
    peer_root = Path('/var/tmp/rev4-featbank-r4/peer_gmsd')
    for g in prepared['groups']:
        path = Path(g['path']).with_suffix('.keys.parquet')
        if sha(path) != g['keys_sha256']:
            raise ValueError('input key drift')
        keys = pq.read_table(path).to_pandas()
        keys['position'] = np.arange(len(keys))
        chunks = []
        for member in keys.member_set.unique():
            if member not in bank_cache:
                bp = bank_root / member / 'keys.parquet'
                if str(bp) not in base['inputs'] or sha(bp) != base['inputs'][str(bp)]['sha256']:
                    raise ValueError('unregistered bank keys: ' + str(bp))
                pins[str(bp)] = sha(bp)
                bank_cache[member] = unique_index(pq.read_table(bp).to_pandas())
                pp = peer_root / (member + '.parquet')
                pins[str(pp)] = sha(pp)
                peer = pq.read_table(pp).to_pandas()
                if peer.groupby('pair_key')[['gmsd', 'gmsm', 'ref_sha_new', 'dist_sha_new', 'digest_match']].nunique().max().max() > 1:
                    raise ValueError('ambiguous duplicate peer scores')
                peer_cache[member] = unique_index(peer.drop_duplicates('pair_key'))
            selected = keys[keys.member_set == member]
            b = bank_cache[member].loc[selected.pair_key].reset_index()
            p = peer_cache[member].loc[selected.pair_key].reset_index()
            if not np.array_equal(b.ref_pixels_sha256, p.ref_sha_new) or not np.array_equal(b.dist_pixels_sha256, p.dist_sha_new) or not (p.digest_match == 1).all():
                raise ValueError('peer pixel digest mismatch')
            b['position'] = selected.position.to_numpy()
            b['gmsd_quality'] = -p.gmsd.to_numpy()
            b['gmsm'] = p.gmsm.to_numpy()
            chunks.append(b)
        frame = pd.concat(chunks).sort_values('position')
        if frame.pair_key.tolist() != keys.pair_key.tolist():
            raise ValueError('metadata reordered rows')
        frames.append(frame)
    joined = pd.concat(frames, ignore_index=True)
    joined['population'], joined['reference'], joined['development'], joined['fold'] = data['pop'], data['refs'], data['dev'], data['fold']
    # Exact decoded reference identity, independent of filenames and dataset names.
    for digest, group in joined.groupby('ref_pixels_sha256'):
        if group.development.nunique() > 1 or (group[~group.development].fold.nunique() > 1):
            raise ValueError('decoded source crosses split/folds: ' + digest)
    peer_results = {}
    e21 = Path('/mnt/v/output/zensim/e21-peers-20261028')
    for name, file, col, sign in [('ssim2', 'judge_ssim2.tsv', 'ssim2', 1),
                                  ('butter3', 'judge_butteraugli.tsv', 'butteraugli_pnorm3', -1)]:
        path = e21 / file
        pins[str(path)] = sha(path)
        peer = pd.read_csv(path, sep='\t', usecols=['pair_key', 'ref_path', 'dist_path', col])
        if peer.groupby('pair_key')[col].nunique().max() != 1:
            raise ValueError('ambiguous E21 peer scores')
        peer = unique_index(peer.drop_duplicates('pair_key'))
        human = joined.population != 'cid22_ssim2'
        missing = sorted(set(joined.loc[human, 'pair_key']) - set(peer.index))
        if missing:
            raise ValueError(f'{name}: missing {len(missing)} admitted human pairs')
        rows = peer.loc[joined.loc[human, 'pair_key']]
        # Keys bind the original pair; mismatched path spellings need explicit review.
        if rows.ref_path.tolist() != joined.loc[human, 'ref_path'].tolist() or rows.dist_path.tolist() != joined.loc[human, 'dist_path'].tolist():
            raise ValueError('peer key/path inconsistency')
        joined[name] = np.nan
        joined.loc[human, name] = sign * rows[col].to_numpy()
        peer_results[name] = dict(human_rows=int(human.sum()), missing=0)
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pandas(joined, preserve_index=False), output / 'metadata.parquet', compression='zstd')
    summaries = {}
    for name in ('ssim2', 'butter3', 'gmsd_quality', 'gmsm'):
        summaries[name] = {}
        for split, mask in [('fit', ~data['dev']), ('development', data['dev'])]:
            ids = np.flatnonzero(mask & (data['pop'] != 'cid22_ssim2'))
            summaries[name][split] = panels(output / name / split, joined[name].to_numpy()[ids], ids, data, plan['panel_bin'])
    write(output / 'PEERS.json', dict(inputs=pins, metrics=summaries, coverage=peer_results,
        decoded_reference_split_audit=True, source_hashes=joined.ref_pixels_sha256.nunique(),
        metadata_sha256=sha(output / 'metadata.parquet'), peer_labels_used_for_fit=False))
    print(json.dumps(summaries, indent=2))


def peer_cell(run, metadata_path, output, config, fold, seed):
    data = load_data(run / 'data')
    original = json.loads((run / 'PLAN.json').read_text())
    meta = pq.read_table(metadata_path, columns=['ssim2', 'butter3', 'gmsd_quality', 'gmsm'])
    peers = np.column_stack([meta[n].to_numpy() for n in meta.column_names]).astype(np.float32)
    x = peers if config['features'] == 'peers' else np.column_stack([data['raw'], peers])
    human = data['pop'] != 'cid22_ssim2'
    fit = human & ~data['dev']
    if fold >= 0:
        fit &= data['fold'] != fold
    test = human & ((~data['dev'] & (data['fold'] == fold)) if fold >= 0 else data['dev'])
    fit_ids, test_ids = np.flatnonzero(fit), np.flatnonzero(test)
    if not np.isfinite(x[human]).all():
        raise ValueError('missing peer input on admitted human row')
    name = f"{config['id']}-fold{fold}-seed{seed}"
    dest = output / 'cells' / name
    dest.mkdir(parents=True, exist_ok=False)
    w, target, maps = weights_and_targets(data['y'][fit_ids], data['pop'][fit_ids], data['refs'][fit_ids], 'cdf', 0.)
    model = estimator(config, seed, x.shape[1])
    write(dest / 'FIT.json', dict(config=config, fold=fold, seed=seed, parameters=model.get_params(),
        maps=maps, fit_rows=len(fit_ids), test_rows=len(test_ids), plan_sha256=sha(output / 'PLAN.json')))
    model.fit(x[fit_ids], target, sample_weight=w)
    with (dest / 'model.pickle').open('wb') as f:
        pickle.dump(model, f, protocol=5)
    predicted = model.predict(x[test_ids])
    result = dict(name=name, config=config, fold=fold, seed=seed,
        metrics=panels(dest / 'held', predicted, test_ids, data, original['panel_bin']))
    if fold < 0:
        ids = np.concatenate([fit_ids, test_ids])
        native, receipt = native_gate(model, x[ids], dest / 'raw', original['parity_bin'], config)
        delta = float(np.max(np.abs(native - model.predict(x[ids]))))
        if delta > 1e-12:
            raise ValueError('peer forest export rounding bound')
        result.update(native=receipt, maximum_sklearn_export_rounding_difference=delta,
            in_sample=panels(dest / 'in_sample', native[:len(fit_ids)], fit_ids, data, original['panel_bin']),
            native_development=panels(dest / 'native_development', native[len(fit_ids):], test_ids, data, original['panel_bin']))
    write(dest / 'RESULT.json', result)
    return result


def peer_search(run, metadata_path, output):
    output.mkdir(parents=True, exist_ok=False)
    configs = []
    for family in ('rf', 'extra', 'hgb'):
        for features in ('peers', 'features_and_peers'):
            for complexity in (0, 1):
                params = (dict(max_leaf_nodes=(15, 63)[complexity], min_samples_leaf=20) if family == 'hgb'
                          else dict(min_samples_leaf=(2, 16)[complexity], max_features=1. if features == 'peers' else .33))
                configs.append(dict(id=f'{family}-{features}-{complexity}', family=family,
                    features=features, params=params, mapping='cdf', teacher_mass=0.))
    write(output / 'PLAN.json', dict(schema='tree-peer-assisted-v1', configs=configs, folds=3,
        seeds=SEEDS, source_plan_sha256=sha(run / 'PLAN.json'), metadata_sha256=sha(metadata_path),
        script_sha256=sha(__file__), tree_owner_sha256=sha(Path(__file__).with_name('tree_study.py')),
        synthetic_labels_used=False, peer_inputs=['ssim2', 'butter3', 'gmsd_quality', 'gmsm'],
        selection='same equal-population pooled/within composite; best per family and input set',
        development_for_selection=False, required=54))
    results, failures = [], []
    def batch(tasks, stage):
        with ProcessPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(peer_cell, run, metadata_path, output, c, f, s): (c['id'], f, s) for c, f, s in tasks}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as error:
                    failures.append(dict(cell=futures[future], error=repr(error)))
                write(output / 'PROGRESS.json', dict(stage=stage, complete=len(results), failures=failures,
                    pid=os.getpid(), checked_utc=datetime.now(timezone.utc).isoformat()))
                print(json.dumps(dict(stage=stage, complete=len(results), failures=failures)), flush=True)
        if failures:
            raise RuntimeError('peer-assisted failures preserved in progress record')
    batch([(c, f, SEEDS[0]) for c in configs for f in range(3)], 'peer_source_folds')
    scores = {}
    for c in configs:
        cells = [r for r in results if r['config']['id'] == c['id']]
        values = [(m['pooled'] + m['within']) / 2 for r in cells for m in r['metrics'].values()]
        if len(cells) != 3 or len(values) != 12 or not np.isfinite(values).all():
            raise ValueError('incomplete peer selection composite')
        scores[c['id']] = float(np.mean(values))
    winners = [max([c for c in configs if c['family'] == family and c['features'] == feature], key=lambda c: scores[c['id']])
               for family in ('rf', 'extra', 'hgb') for feature in ('peers', 'features_and_peers')]
    write(output / 'SELECTION.json', dict(scores=scores, winners=winners, uses_development=False))
    batch([(c, -1, s) for c in winners for s in SEEDS], 'peer_frozen_finalists')
    write(output / 'SUMMARY.json', dict(required=54, completed=len(results), winners=winners,
        selection=scores, results=results, failures=failures))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-plan', type=Path)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--peer-search', type=Path, help='run the registered peer-assisted search with this joined metadata')
    args = p.parse_args()
    if args.peer_search:
        peer_search(args.run, args.peer_search, args.output)
    else:
        if args.baseline_plan is None:
            p.error('--baseline-plan is required for a metadata join')
        metadata(args.baseline_plan, args.run, args.output)


if __name__ == '__main__':
    main()
