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
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tree_study import load_data, panels, sha, write, estimator, weights_and_targets, native_gate, SEEDS, canonical_ref, cell


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


def summarize(root, output, neural_summaries):
    """Reduce completed studies without fitting or selecting another model."""
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((root / 'search/PLAN.json').read_text())
    data = load_data(root / 'search/data')
    metadata_ = pq.read_table(root / 'peers/metadata.parquet').to_pandas()
    populations = ['kadid', 'tid2013', 'konfig', 'cid22_a25']
    ids = {p: np.flatnonzero(data['dev'] & (data['pop'] == p)) for p in populations}
    metrics, vectors, pins, selections, diagnostics = {}, {}, {}, {}, {}
    for study in ('search', 'peer-search', 'constraint-controls'):
        path = root / study / 'SUMMARY.json'
        summary = json.loads(path.read_text())
        if summary['completed'] != summary['required'] or summary['failures']:
            raise ValueError('incomplete study: ' + study)
        pins[str(path)] = sha(path)
        selections[study] = summary['selection']
        for winner in summary['winners']:
            cells = sorted([r for r in summary['results'] if r['config']['id'] == winner['id'] and r['fold'] == -1], key=lambda r: r['seed'])
            if len(cells) != 3:
                raise ValueError('missing final seed')
            name = (winner['family'] if study == 'search' else winner['family'] + '_constraint_control'
                    if study == 'constraint-controls' else winner['family'] + '_' + winner['features'])
            metrics[name] = {p: {k: dict(mean=float(np.mean([r['native_development'][p][k] for r in cells])),
                    minimum=float(min(r['native_development'][p][k] for r in cells)),
                    maximum=float(max(r['native_development'][p][k] for r in cells))) for k in ('pooled', 'within')} for p in populations}
            diagnostics[name] = dict(config=winner, native_rows=[r['native']['rows'] for r in cells],
                max_export_rounding=max(r['maximum_sklearn_export_rounding_difference'] for r in cells),
                probes=[r.get('feature_probes') for r in cells])
            first = root / study / 'cells' / cells[0]['name'] / 'native_development'
            vectors[name] = {}
            for p in populations:
                table_path = first / (p + '.parquet')
                t = pq.read_table(table_path)
                if not np.array_equal(t['target'].to_numpy(), data['y'][ids[p]]) or t['band'].to_pylist() != data['refs'][ids[p]].tolist():
                    raise ValueError('tree panel rows do not match admitted development')
                vectors[name][p] = t['predicted'].to_numpy()
                pins[str(table_path)] = sha(table_path)
    peer = json.loads((root / 'peers/PEERS.json').read_text())
    for name in ('ssim2', 'butter3', 'gmsd_quality', 'gmsm'):
        metrics[name] = {p: {k: dict(mean=v, minimum=v, maximum=v) for k, v in peer['metrics'][name]['development'][p].items()} for p in populations}
        vectors[name] = {p: metadata_[name].to_numpy()[ids[p]] for p in populations}
    # Reuse completed native neural assessments only after checking their bytes and row identity.
    for path in neural_summaries:
        doc = json.loads(path.read_text())
        if doc['missing_cells']:
            raise ValueError('incomplete neural comparator')
        pins[str(path)] = sha(path)
        for arm in sorted({c['arm'] for c in doc['cells']}):
            cells = sorted([c for c in doc['cells'] if c['arm'] == arm], key=lambda c: c['seed'])
            if len(cells) != 3:
                raise ValueError('neural seed count')
            name = 'mlp_' + arm
            scores = []
            for index, cell_ in enumerate(cells):
                result_path = Path(cell_['result_path'])
                if sha(result_path) != cell_['result_sha256']:
                    raise ValueError('neural receipt changed')
                record = json.loads(result_path.read_text())
                admitted_hashes = {pin for file, pin in original['inputs'].items()
                                   if file.endswith('.parquet') and not file.endswith('.keys.parquet')}
                if any(r['table_sha256'] not in admitted_hashes for r in record['selection']['strict_table_admission']):
                    raise ValueError('neural comparator used different input tables')
                bake = Path(record['bake'])
                if sha(bake) != cell_['bake_sha256']:
                    raise ValueError('neural bake changed')
                pins[str(result_path)], pins[str(bake)] = sha(result_path), sha(bake)
                values = {}
                for p in populations:
                    table_path = Path(record['assessment_directory']) / ('human_' + p + '_development.panel.parquet')
                    t = pq.read_table(table_path)
                    if not np.array_equal(t['target'].to_numpy(), data['y'][ids[p]]) or [canonical_ref(r) for r in t['band'].to_pylist()] != data['refs'][ids[p]].tolist():
                        raise ValueError('neural assessment rows differ')
                    pins[str(table_path)] = sha(table_path)
                    values[p] = dict(pooled=record['panels']['human_' + p + '_development']['signed-quality']['srocc_signed'],
                        within=record['panels']['human_' + p + '_development']['per-group']['per_group']['mean'])
                    if index == 0:
                        vectors.setdefault(name, {})[p] = t['predicted'].to_numpy()
                scores.append(values)
            metrics[name] = {p: {k: dict(mean=float(np.mean([v[p][k] for v in scores])),
                minimum=float(min(v[p][k] for v in scores)), maximum=float(max(v[p][k] for v in scores))) for k in ('pooled', 'within')} for p in populations}
    write(output / 'COMPARISON.json', dict(metrics=metrics, diagnostics=diagnostics, selections=selections,
        sources=pins, development_rows={p: len(ids[p]) for p in populations},
        development_sources={p: len(set(data['refs'][ids[p]])) for p in populations},
        seed_statistic='mean and range across three fitted seeds, not a confidence interval',
        paired_analysis_seed='first predeclared seed in each model family; not a seed average'))
    np.savez_compressed(output / 'comparison_vectors.npz', **{n + '__' + p: x for n, ps in vectors.items() for p, x in ps.items()})
    # Canonical group diagnostics by actual distortion labels, independent of dataset names.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ['ZEN_PANEL_BIN'] = original['panel_bin']
    from lib.zen_stats import panel_batch, panel_batch_indexed
    jobs = []
    for name, ps in vectors.items():
        for pop, pred in ps.items():
            labels = metadata_.iloc[ids[pop]].codec.to_numpy()
            for label in sorted(set(labels)):
                keep = labels == label
                jobs.append((name + '__' + pop + '__' + label, pred[keep], data['y'][ids[pop]][keep]))
    rows = panel_batch(jobs, stats='srocc')
    if any(r['n_dropped'] for r in rows):
        raise ValueError('distortion panel dropped rows')
    pq.write_table(pa.Table.from_pylist(rows), output / 'distortion_panels.parquet', compression='zstd')
    # Paired source bootstrap: native signed SROCC; never resample variants independently.
    chosen = ['rf', 'extra', 'hgb', 'rf_features_and_peers', 'extra_features_and_peers',
              'hgb_features_and_peers', 'mlp_shared_relu_ssim2', 'mlp_four_human', 'ssim2']
    intervals = {}
    for pop in populations:
        references = data['refs'][ids[pop]]
        groups = [np.flatnonzero(references == r) for r in sorted(set(references))]
        if len(groups) < 2:
            intervals[pop] = dict(status='not estimable from one development reference', references=1)
            continue
        bases = {'human': data['y'][ids[pop]]}
        bases.update({name: vectors[name][pop] for name in chosen})
        jobs = [(name + '_point', name, 'human', None) for name in chosen]
        rng = np.random.default_rng(20261106)
        for b in range(2000):
            indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            jobs.extend((name + '_' + str(b), name, 'human', indices) for name in chosen)
        sampled = panel_batch_indexed(bases, jobs, stats='srocc')
        if any(r['n_dropped'] for r in sampled):
            raise ValueError('bootstrap dropped rows')
        pq.write_table(pa.Table.from_pylist(sampled), output / (pop + '.bootstrap.parquet'), compression='zstd')
        by = {r['label']: r['srocc_signed'] for r in sampled}
        if not np.isfinite(list(by.values())).all():
            raise ValueError('nonfinite source bootstrap')
        comparisons = {}
        for baseline in ('mlp_shared_relu_ssim2', 'mlp_four_human', 'ssim2'):
            for name in chosen[:6]:
                delta = np.array([by[name + '_' + str(b)] - by[baseline + '_' + str(b)] for b in range(2000)])
                comparisons[name + '_minus_' + baseline] = dict(point=by[name + '_point'] - by[baseline + '_point'],
                    low=float(np.quantile(delta, .025)), high=float(np.quantile(delta, .975)))
        intervals[pop] = dict(references=len(groups), draws=2000, seed=20261106,
            status='conditional on already-exposed development sources; no multiplicity correction', comparisons=comparisons)
    write(output / 'SOURCE_BOOTSTRAP.json', intervals)
    lines = ['# Tree study comparison', '', 'Development SROCC, mean across three seeds; peers are fixed.', '',
             '| Model | KADID | TID2013 | KonFiG F | CID22-A |', '|---|---:|---:|---:|---:|']
    for name, m in metrics.items():
        lines.append('| ' + name + ' | ' + ' | '.join(f"{m[p]['pooled']['mean']:.6f}" for p in populations) + ' |')
    (output / 'TABLE.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def closure(root, base_path):
    """Save exact consumed files and source before the existing mirror owner runs."""
    base = json.loads(base_path.read_text())
    original = json.loads((root / 'search/PLAN.json').read_text())
    peer = json.loads((root / 'peers/PEERS.json').read_text())
    comparison = json.loads((root / 'analysis/COMPARISON.json').read_text())
    dest = root / 'input-closure'
    dest.mkdir(exist_ok=False)
    inputs = {**original['inputs'], **peer['inputs'], **comparison['sources']}
    inputs[str(base_path)] = sha(base_path)
    prepared = Path(base['output']) / 'PREPARED.json'
    inputs[str(prepared)] = original['prepared_sha256']
    for key, pin in original['binary_sha256'].items():
        inputs[original[key]] = pin
    role = Path(base['output']) / 'root/human_role_decision.json'
    inputs[str(role)] = sha(role)
    index = []
    for file, pin in sorted(inputs.items()):
        path = Path(file)
        if sha(path) != pin:
            raise ValueError('closure input drift: ' + file)
        blob = dest / pin
        if not blob.exists():
            shutil.copyfile(path, blob)
        if sha(blob) != pin:
            raise ValueError('closure copy mismatch')
        index.append(dict(source=file, sha256=pin, bytes=blob.stat().st_size, blob=blob.name))
    write(dest / 'INDEX.json', index)
    sources = root / 'source'
    sources.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    records = [('7b57f9cbc259', 'scripts/rev4_featpot/tree_study.py'),
               ('8c39499e5916', 'scripts/rev4_featpot/tree_study_report.py')]
    for commit, file in records:
        target = sources / (commit + '-' + Path(file).name)
        target.write_bytes(subprocess.check_output(['git', 'show', commit + ':' + file], cwd=repo))
    for file in ['scripts/rev4_featpot/activation_curriculum.py', 'scripts/rev4_featpot/tree_study_report.py',
                 'scripts/v_next/train_corruption_head.py', 'scripts/lib/zen_stats.py',
                 'scripts/byconstruction/dataset_mirror.py', 'scripts/tests/test_tree_study.py',
                 'scripts/rev4_featpot/color_noise_fx2_declaration.json', 'docs/TREE_REGRESSION_STUDY.md']:
        shutil.copyfile(repo / file, sources / Path(file).name)
    if sha(sources / '7b57f9cbc259-tree_study.py') != original['script_sha256']:
        raise ValueError('search source identity mismatch')
    peer_plan = json.loads((root / 'peer-search/PLAN.json').read_text())
    if sha(sources / '8c39499e5916-tree_study_report.py') != peer_plan['script_sha256']:
        raise ValueError('peer-search source identity mismatch')
    import importlib.metadata
    write(sources / 'PACKAGES.json', {name: importlib.metadata.version(name)
          for name in ['scikit-learn', 'numpy', 'scipy', 'pyarrow', 'pandas', 'joblib', 'threadpoolctl']})
    completed = 0
    for study in ('search', 'peer-search', 'constraint-controls'):
        receipt = json.loads((root / study / 'SUMMARY.json').read_text())
        if receipt['completed'] != receipt['required'] or receipt['failures']:
            raise ValueError('unclosed scientific fits')
        completed += receipt['completed']
    write(root / 'CLOSURE.json', dict(input_files=len(index), input_index_sha256=sha(dest / 'INDEX.json'),
        source_files={p.name: sha(p) for p in sorted(sources.iterdir())}, scientific_fits=completed,
        artifact_mirror_owner='scripts/byconstruction/dataset_mirror.py', complete=True))


def constraint_controls(run, output):
    """Fixed full-fit unconstrained twins; reuse the exact registered cell owner."""
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(run / 'data', output / 'data')
    source = json.loads((run / 'PLAN.json').read_text())
    configs = [dict(family='rf', features='direct', mapping='cdf', teacher_mass=1.,
                    params=dict(min_samples_leaf=8, max_features=.33), id='rf_control'),
               dict(family='hgb', features='direct', mapping='cdf', teacher_mass=1.,
                    params=dict(max_leaf_nodes=31, min_samples_leaf=20), id='hgb_control')]
    write(output / 'PLAN.json', {**source, 'schema': 'tree-matched-constraint-controls-v1',
        'configurations': configs, 'source_plan_sha256': sha(run / 'PLAN.json'),
        'controller_sha256': sha(__file__), 'selection': 'fixed matched twins, no tuning'})
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=6) as pool:
        tasks = {pool.submit(cell, output, c, -1, s): (c['id'], s) for c in configs for s in SEEDS}
        for future in as_completed(tasks):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(dict(cell=tasks[future], error=repr(error)))
            write(output / 'PROGRESS.json', dict(complete=len(results), required=6, failures=failures,
                pid=os.getpid(), checked_utc=datetime.now(timezone.utc).isoformat()))
            print(json.dumps(dict(complete=len(results), required=6, failures=failures)), flush=True)
    write(output / 'SUMMARY.json', dict(completed=len(results), required=6, failures=failures,
        results=results, winners=configs, selection={}))
    if failures:
        raise RuntimeError('matched constraint controls failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-plan', type=Path)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--peer-search', type=Path, help='run the registered peer-assisted search with this joined metadata')
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--closure', action='store_true')
    p.add_argument('--constraint-controls', action='store_true')
    p.add_argument('--neural-summary', type=Path, action='append', default=[])
    args = p.parse_args()
    if args.constraint_controls:
        constraint_controls(args.run, args.output)
    elif args.closure:
        if args.baseline_plan is None:
            p.error('--baseline-plan is required for closure')
        closure(args.run, args.baseline_plan)
    elif args.summarize:
        summarize(args.run, args.output, args.neural_summary)
    elif args.peer_search:
        peer_search(args.run, args.peer_search, args.output)
    else:
        if args.baseline_plan is None:
            p.error('--baseline-plan is required for a metadata join')
        metadata(args.baseline_plan, args.run, args.output)


if __name__ == '__main__':
    main()
