#!/usr/bin/env python3
"""Source-grouped tree experiments over pinned cached features; see study protocol.

Training uses sklearn; statistics use Rust panel. Final opaque-vector exports
use the existing ZCTH writer and native parity gate, never pixel-profile admission.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from types import SimpleNamespace

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from activation_curriculum import sha, write

HERE = Path(__file__).resolve().parents[2]
BASE_PIN = '80ec04c1cba1ef851192e63c0a03687a9da019ae483713c978c1dc0daa7ec915'
SEEDS = (20261106, 20261107, 20261108)
RAW_IDS = list(range(228)) + list(range(372, 720))


def canonical_ref(ref):
    return re.sub(r'_part[ab]$', '', ref.lower())


def fold_ids(pop, refs):
    result = np.full(len(pop), -1, dtype=np.int8)
    for name in np.unique(pop):
        mask = pop == name
        names = sorted(set(refs[mask]), key=lambda s: sha_text('tree-fold-v1|' + name + '|' + s))
        if len(names) < 3:
            raise ValueError('three source folds require at least three references: ' + name)
        lookup = {name: i % 3 for i, name in enumerate(names)}
        result[mask] = [lookup[r] for r in refs[mask]]
    return result


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def weights_and_targets(y, pop, refs, mapping, teacher_mass):
    w = np.empty(len(y), dtype=np.float64)
    target = np.empty(len(y), dtype=np.float64)
    maps = {}
    for name in np.unique(pop):
        mask = pop == name
        counts = Counter(refs[mask])
        local = np.array([1 / (len(counts) * counts[r]) for r in refs[mask]])
        yy = y[mask]
        values, inverse = np.unique(yy, return_inverse=True)
        mass = np.bincount(inverse, weights=local)
        if mapping == 'cdf':
            transformed = (np.cumsum(mass) - mass / 2) / mass.sum()
        elif mapping == 'range':
            if values[-1] == values[0]:
                raise ValueError('constant training target')
            transformed = (values - values[0]) / (values[-1] - values[0])
        else:
            raise ValueError(mapping)
        target[mask] = transformed[inverse]
        w[mask] = local * (teacher_mass if name == 'cid22_ssim2' else 1.)
        maps[name] = dict(values=values.tolist(), mapped=transformed.tolist(), references=len(counts))
    w *= len(w) / w.sum()
    return w, target, maps


def candidates():
    configs = []
    def add(family, features='raw', mapping='cdf', teacher=1., **params):
        c = dict(family=family, features=features, mapping=mapping, teacher_mass=teacher, params=params)
        c['id'] = f'{family}-{len(configs):02d}'
        configs.append(c)
    for feature in ('direct', 'raw', 'products'):
        for fraction in ('sqrt', .33):
            for leaf in (2, 16):
                add('rf', feature, min_samples_leaf=leaf, max_features=fraction)
        for leaf in (2, 16):
            add('extra', feature, min_samples_leaf=leaf, max_features=.33)
        for leaves in (15, 63):
            add('hgb', feature, max_leaf_nodes=leaves, min_samples_leaf=20)
    # Matched preprocessing and teacher-dose ablations for each family.
    for family in ('rf', 'extra', 'hgb'):
        for mapping, teacher in [('range', 1.), ('cdf', 0.), ('cdf', .25), ('range', 0.)]:
            params = dict(max_leaf_nodes=31, min_samples_leaf=20) if family == 'hgb' else dict(min_samples_leaf=8, max_features=.33)
            add(family, mapping=mapping, teacher=teacher, **params)
    add('rf_mono', 'direct', min_samples_leaf=8, max_features=.33)
    add('hgb_mono', 'direct', max_leaf_nodes=31, min_samples_leaf=20)
    return configs


def estimator(config, seed, width):
    family = config['family']
    mono = [-1] * width if family.endswith('_mono') else None
    if family.startswith('hgb'):
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=.05,
            l2_regularization=1., early_stopping=False, categorical_features=None,
            random_state=seed, monotonic_cst=mono, **config['params'])
    cls = ExtraTreesRegressor if family == 'extra' else RandomForestRegressor
    return cls(n_estimators=128, max_leaf_nodes=2048, n_jobs=1,
               random_state=seed, monotonic_cst=mono, **config['params'])


def panels(out, predictions, indices, data, panel_bin):
    out.mkdir(parents=True, exist_ok=False)
    result = {}
    for pop in sorted(set(data['pop'][indices])):
        local = data['pop'][indices] == pop
        rows = indices[local]
        path = out / (pop + '.parquet')
        pq.write_table(pa.table(dict(predicted=predictions[local], target=data['y'][rows],
                                    band=data['refs'][rows])), path, compression='zstd')
        stats = {}
        for mode in ('signed-quality', 'per-group', 'scatter'):
            cmd = [str(panel_bin), '--input', str(path), '--json', '--' + mode]
            p = subprocess.run(cmd, capture_output=True, text=True)
            (out / (pop + '.' + mode + '.log')).write_text(p.stdout + '\n' + p.stderr)
            p.check_returncode()
            stats[mode] = json.loads(p.stdout)
        if stats['signed-quality']['n'] != len(rows) or stats['signed-quality']['n_dropped'] != 0 or stats['scatter']['status'] != 'MEASURED':
            raise ValueError('incomplete canonical panel: ' + pop)
        result[pop] = stats
    write(out / 'PANELS.json', result)
    return {p: dict(pooled=v['signed-quality']['srocc_signed'],
                    within=v['per-group']['per_group']['mean']) for p, v in result.items()}


def export_view(model, x, anchored=False):
    """Adapt nodes, retaining sklearn's independent compiled tree traversal."""
    if hasattr(model, '_predictors'):
        view = copy.deepcopy(model)
        if anchored:
            zero = np.zeros((1, x.shape[1]), dtype=np.float64)
            bits, mapping = model._bin_mapper.make_known_categories_bitsets()
            for group in view._predictors:
                tree = group[0]
                origin = tree.predict(zero, bits, mapping, 1)[0]
                tree.nodes['value'][tree.nodes['is_leaf'] != 0] -= origin
            view._baseline_prediction[:] = 1.
        return view, view.predict(x)
    dtype = [('num_threshold', '<f8'), ('left', '<u4'), ('right', '<u4'),
             ('feature_idx', '<u4'), ('is_leaf', 'u1'), ('missing_go_to_left', 'u1'),
             ('value', '<f8'), ('is_categorical', 'u1')]
    predictors = []
    result = np.full(len(x), 1. if anchored else 0.)
    for estimator_ in model.estimators_:
        t = estimator_.tree_
        n = np.zeros(t.node_count, dtype=dtype)
        leaf = t.children_left < 0
        n['is_leaf'] = leaf
        n['num_threshold'] = t.threshold
        n['left'] = np.maximum(0, t.children_left)
        n['right'] = np.maximum(0, t.children_right)
        n['feature_idx'] = np.maximum(0, t.feature)
        n['missing_go_to_left'] = t.missing_go_to_left
        values = t.value[:, 0, 0].copy()
        if anchored:
            values -= values[estimator_.apply(np.zeros((1, x.shape[1]), dtype=np.float32))[0]]
        n['value'] = values / len(model.estimators_)
        result += n['value'][estimator_.apply(x)]
        predictors.append([SimpleNamespace(nodes=n)])
    return SimpleNamespace(_predictors=predictors,
                           _baseline_prediction=np.array([[1. if anchored else 0.]])), result


def native_gate(model, x, dest, parity_bin, provenance, anchored=False):
    exporter = HERE / 'scripts/v_next/train_corruption_head.py'
    spec = importlib.util.spec_from_file_location('tree_export_owner', exporter)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    view, reference = export_view(model, x, anchored)
    path = dest.with_suffix('.opaque-vector.zcth')
    owner.emit_zcth(str(path), x.shape[1], list(range(x.shape[1])), np.zeros(x.shape[1]),
                    np.ones(x.shape[1]), float(np.finfo(np.float32).max), view, None, .99,
                    {**provenance, 'pixel_servable': False, 'format_scope': 'opaque vector arithmetic only',
                     'actual_feature_formula_revision': 7, 'anchored': anchored}, input_precision='f32')
    npz = dest.with_suffix('.parity.npz')
    np.savez_compressed(npz, test_X=x.astype(np.float64), test_raw=reference,
                        test_p=1 / (1 + np.exp(-reference)))
    cmd = [str(parity_bin), '--head', str(path), '--parity', str(npz), '--set', 'test']
    p = subprocess.run(cmd, capture_output=True, text=True)
    dest.with_suffix('.parity.log').write_text(p.stdout + '\n' + p.stderr)
    p.check_returncode()
    if 'corrhead_parity: PASS' not in p.stdout:
        raise ValueError('missing native gate success')
    return reference, dict(model_sha256=sha(path), parity_sha256=sha(npz),
                           log_sha256=sha(dest.with_suffix('.parity.log')), rows=len(x), raw_ulp=0)


def load_data(output):
    return {k: np.load(output / (k + '.npy'), mmap_mode='r')
            for k in ('direct', 'raw', 'products', 'y', 'pop', 'refs', 'dev', 'fold')}


def cell(output, config, fold, seed):
    data = load_data(output / 'data')
    plan = json.loads((output / 'PLAN.json').read_text())
    name = f"{config['id']}-fold{fold}-seed{seed}"
    dest = output / 'cells' / name
    dest.mkdir(parents=True, exist_ok=False)
    fit = ~data['dev']
    if fold >= 0:
        fit &= data['fold'] != fold
    if config['teacher_mass'] == 0:
        fit &= data['pop'] != 'cid22_ssim2'
    test = (~data['dev'] & (data['fold'] == fold)) if fold >= 0 else data['dev']
    fit_ids, test_ids = np.flatnonzero(fit), np.flatnonzero(test)
    x = data[config['features']]
    weights, targets, maps = weights_and_targets(data['y'][fit_ids], data['pop'][fit_ids],
        data['refs'][fit_ids], config['mapping'], config['teacher_mass'])
    model = estimator(config, seed, x.shape[1])
    write(dest / 'FIT.json', dict(config=config, seed=seed, fold=fold, fit_rows=len(fit_ids),
        test_rows=len(test_ids), maps=maps, parameters=model.get_params(), plan_sha256=sha(output / 'PLAN.json')))
    model.fit(x[fit_ids], targets, sample_weight=weights)
    with (dest / 'model.pickle').open('wb') as f:
        pickle.dump(model, f, protocol=5)
    pred = model.predict(x[test_ids])
    stats = panels(dest / 'held', pred, test_ids, data, plan['panel_bin'])
    result = dict(name=name, config=config, fold=fold, seed=seed, metrics=stats,
                  model_sha256=sha(dest / 'model.pickle'))
    if fold < 0:
        # All final-fit predictions, not a sampled parity subset.
        all_ids = np.concatenate([fit_ids, test_ids])
        rng = np.random.default_rng(20261106)
        top = np.max(x[fit_ids], axis=0)
        probes = rng.uniform(0, 1, (512, x.shape[1])).astype(np.float32) * top
        larger = probes.copy()
        columns = rng.integers(x.shape[1], size=len(probes))
        larger[np.arange(len(probes)), columns] += top[columns] * .2
        ray = np.median(x[fit_ids], axis=0)[None, :] * np.array([0, 1e-8, 1e-6, 1e-4, .01, .1, 1], dtype=np.float32)[:, None]
        check = np.concatenate([x[all_ids], probes, larger, ray]).astype(np.float32)
        native, receipt = native_gate(model, check, dest / 'raw', plan['parity_bin'], config)
        anchored, anchored_receipt = native_gate(model, check, dest / 'anchored', plan['parity_bin'], config, True)
        delta = float(np.max(np.abs(native - model.predict(check))))
        if delta > 1e-12:
            raise ValueError('forest adaptation differs beyond declared rounding bound')
        offset = len(all_ids)
        increases = anchored[offset + 512:offset + 1024] - anchored[offset:offset + 512]
        violations = int(np.sum(increases > 1e-12))
        if config['family'].endswith('_mono') and violations:
            raise ValueError('monotone tree coordinate violation')
        if anchored[-7] != 1.:
            raise ValueError('anchored identity is not exact')
        result.update(native=receipt, anchored_native=anchored_receipt,
            maximum_sklearn_export_rounding_difference=delta,
            feature_probes=dict(coordinate_pairs=512, monotonicity_violations=violations,
                anchored_identity=float(anchored[-7]), near_zero_ray=anchored[-7:].tolist()),
            in_sample=panels(dest / 'in_sample', native[:len(fit_ids)], fit_ids, data, plan['panel_bin']))
        # Explicitly retain the native-verified final vectors, separate from search output.
        result['native_development'] = panels(dest / 'native_development', native[len(fit_ids):offset], test_ids, data, plan['panel_bin'])
    write(dest / 'RESULT.json', result)
    return result


def prepare(base_path, output, parity_bin):
    if sha(base_path) != BASE_PIN:
        raise ValueError('unregistered preparation plan')
    base = json.loads(base_path.read_text())
    prepared_path = Path(base['output']) / 'PREPARED.json'
    prepared = json.loads(prepared_path.read_text())
    if prepared['plan_sha256'] != BASE_PIN or len(prepared['groups']) != 9:
        raise ValueError('preparation identity mismatch')
    declaration_path = HERE / 'scripts/rev4_featpot/color_noise_fx2_declaration.json'
    declaration = json.loads(declaration_path.read_text())
    direct_ids = declaration['direct']
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data').mkdir()
    arrays = {k: [] for k in ('raw', 'y', 'pop', 'refs', 'dev')}
    pins = {}
    for g in prepared['groups']:
        path = Path(g['path'])
        for p, key in [(path, 'sha256'), (Path(str(path) + '.manifest.json'), 'manifest_sha256'),
                       (path.with_suffix('.keys.parquet'), 'keys_sha256')]:
            if sha(p) != g[key]:
                raise ValueError('prepared input drift: ' + str(p))
            pins[str(p)] = g[key]
        manifest = json.loads(Path(str(path) + '.manifest.json').read_text())
        if manifest['formula_revision'] != 7:
            raise ValueError('feature arithmetic mismatch')
        t = pq.read_table(path, columns=['human_score', 'ref_basename'] + [f'f{i}' for i in RAW_IDS])
        raw = np.column_stack([t[f'f{i}'].to_numpy() for i in RAW_IDS]).astype(np.float32)
        if not np.isfinite(raw).all() or not np.isfinite(t['human_score'].to_numpy()).all():
            raise ValueError('nonfinite admitted features or labels')
        arrays['raw'].append(raw)
        arrays['y'].append(t['human_score'].to_numpy())
        arrays['pop'].append(np.full(len(t), g['source']))
        arrays['refs'].append(np.array([canonical_ref(r) for r in t['ref_basename'].to_pylist()]))
        arrays['dev'].append(np.full(len(t), g['split'] == 'dev'))
    data = {k: np.concatenate(v) for k, v in arrays.items()}
    lookup = {f: i for i, f in enumerate(RAW_IDS)}
    data['direct'] = data['raw'][:, [lookup[i] for i in direct_ids]]
    if np.min(data['direct']) < 0:
        raise ValueError('monotone input domain has negative distortion features')
    products = np.column_stack([data['raw'][:, lookup[a]] * data['raw'][:, lookup[b]] for a, b in declaration['products']])
    data['products'] = np.column_stack([data['direct'], products])
    if not np.isfinite(data['products']).all():
        raise ValueError('derived input overflow')
    fit = ~data['dev']
    if set(data['refs'][fit]) & set(data['refs'][~fit]):
        raise ValueError('fit/development reference overlap')
    data['fold'] = np.full(len(fit), -1, dtype=np.int8)
    data['fold'][fit] = fold_ids(data['pop'][fit], data['refs'][fit])
    for k, values in data.items():
        np.save(output / 'data' / (k + '.npy'), values)
    membership = sorted(set(zip(data['pop'].tolist(), data['refs'].tolist(), data['dev'].tolist(), data['fold'].tolist())))
    write(output / 'MEMBERSHIP.json', membership)
    plan = dict(schema='tree-study-v2-quality-search', configurations=candidates(), folds=3,
        final_seeds=SEEDS, selection='equal-human-population mean of (pooled + within-reference SROCC)/2 over source folds',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=sha(__file__), prepared_sha256=sha(prepared_path), baseline_plan_sha256=BASE_PIN,
        feature_declaration_sha256=sha(declaration_path), feature_ids=dict(raw=RAW_IDS, direct=direct_ids, products=declaration['products']),
        inputs=pins, data_sha256={p.name: sha(p) for p in (output / 'data').iterdir()},
        versions=dict(python=sys.version, numpy=np.__version__, sklearn=sklearn.__version__, pyarrow=pa.__version__),
        panel_bin=str(Path(base['binary_dir']) / 'panel'), parity_bin=str(parity_bin.resolve()),
        protected_populations_read=False, pixel_qualified=False)
    plan['binary_sha256'] = {k: sha(plan[k]) for k in ('panel_bin', 'parity_bin')}
    write(output / 'PLAN.json', plan)


def execute(output, workers):
    plan = json.loads((output / 'PLAN.json').read_text())
    results, failures = [], []
    def batch(tasks, stage):
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(cell, output, c, fold, seed): (c['id'], fold, seed) for c, fold, seed in tasks}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as error:
                    failures.append(dict(cell=futures[future], error=repr(error)))
                write(output / 'PROGRESS.json', dict(stage=stage, complete=len(results),
                    failures=failures, checked_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))
                print(json.dumps(dict(stage=stage, complete=len(results), failures=failures)), flush=True)
        if failures:
            write(output / 'FAILURES.json', failures)
            raise RuntimeError('preserved failed cells; no silent omission')
    batch([(c, f, SEEDS[0]) for c in plan['configurations'] for f in range(3)], 'source_fold_search')
    scores = {}
    for c in plan['configurations']:
        cells = [r for r in results if r['config']['id'] == c['id']]
        values = [(m['pooled'] + m['within']) / 2 for r in cells for p, m in r['metrics'].items() if p != 'cid22_ssim2']
        if len(cells) != 3 or len(values) != 12 or not np.isfinite(values).all():
            raise ValueError('incomplete selection composite')
        scores[c['id']] = float(np.mean(values))
    winners = []
    for family in ('rf', 'extra', 'hgb', 'rf_mono', 'hgb_mono'):
        eligible = [c for c in plan['configurations'] if c['family'] == family]
        winners.append(max(eligible, key=lambda c: (scores[c['id']], c['id'])))
    write(output / 'SELECTION.json', dict(scores=scores, winners=winners, uses_development=False))
    batch([(c, -1, s) for c in winners for s in SEEDS], 'frozen_finalists')
    write(output / 'SUMMARY.json', dict(required=3 * len(plan['configurations']) + 15,
        completed=len(results), selection=scores, winners=winners, results=results, failures=failures))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-plan', type=Path, required=True)
    p.add_argument('--parity-bin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('worker budget must be 1..4')
    prepare(args.baseline_plan, args.output, args.parity_bin)
    execute(args.output, args.workers)


if __name__ == '__main__':
    main()
