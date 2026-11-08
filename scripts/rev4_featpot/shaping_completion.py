#!/usr/bin/env python3
"""Register the completion study without fitting models or opening final holdouts."""
from __future__ import annotations

# Qualified first-layer direct-max representation; later activations remain unchanged.
MAX_PAIR_FAMILIES=('relu','softplus')


def max_pair_families(study):
    from shaping_stable_pairs import families
    return (*MAX_PAIR_FAMILIES,*families(study))

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
NEURAL = ('relu', 'softplus', 'silu_monotone', 'silu_diagnostic',
          'product_joint_identity', 'product_pretrained_frozen', 'product_pretrained_joint',
          'separate_joint_identity', 'separate_pretrained_frozen', 'separate_pretrained_joint')
FAMILIES = (*NEURAL, 'shaped_additive', 'extra_trees')
ROLES = ('relu', 'extra_trees', 'shaped_additive', 'best_analytic', 'best_product', 'best_separate')
SEEDS = ((1101, 101), (1107, 200000101), (1129, 400000101))
SETTINGS = ((1e-3, 0.), (1e-3, 1e-5), (3e-4, 0.), (3e-4, 1e-5))
RESCUES = ('init_half', 'init_double', 'constant_lr', 'consolidate_lr')
# Exact declaration used by the frozen primary build and worker baseline replays.
DECLARATION_SHA256 = 'a3802ddeeede6ec5c9e3d03efdd6cc5af683ab4ea2b0d1143c8e9183c5bd46e5'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def seeds(family):
    return (None,) if family == 'shaped_additive' else range(3)


def matrix():
    cells = []
    def add(stage, family, partition, setting='selected', seed=None, **extra):
        cell = dict(stage=stage, family=family, partition=partition, setting=setting,
                    seed=seed, **extra)
        cell['id'] = hashlib.sha256(encoded(cell)).hexdigest()[:20]
        cells.append(cell)
    for outer in range(3):
        for family in FAMILIES:
            for seed in seeds(family):
                add('outer', family, f'outer-{outer}', seed=seed)
                for inner in range(2):
                    for setting in range(4):
                        add('inner', family, f'inner-{outer}-{inner}', setting, seed)
        for seed in range(3):
            add('fresh_control_outer', 'relu', f'outer-{outer}', seed=seed)
            for inner in range(2):
                for setting in range(4):
                    add('fresh_control_inner', 'relu', f'inner-{outer}-{inner}', setting, seed)
        for family in NEURAL:
            for inner in range(2):
                for seed in range(3):
                    for rescue in RESCUES:
                        add('rescue', family, f'inner-{outer}-{inner}', 'base_selected', seed,
                            rescue=rescue, condition=f'any_neural_diagnostic_trigger_outer_{outer}')
        for role in ROLES:
            for seed in seeds(role):
                for fraction in (.25, .5, .75):
                    for order in (20261107, 20261108):
                        add('source_curve', role, f'outer-{outer}', seed=seed,
                            fraction=fraction, order_seed=order)
                    add('row_curve', role, f'outer-{outer}', seed=seed,
                        fraction=fraction, order_seed=20261109)
                for mode in ('population', 'conditional'):
                    for permutation in (20261107, 20261108):
                        add('shuffle', role, f'outer-{outer}', seed=seed,
                            permutation=mode, permutation_seed=permutation)
                add('human_only', role, f'outer-{outer}', seed=seed)
    for role in ROLES:
        for seed in seeds(role):
            add('full_fit', role, 'full-fit', seed=seed)
            for fold in range(3):
                for setting in range(4):
                    add('full_fit_selection', role, f'outer-{fold}', setting, seed)
    add('resource_pilot', 'relu', 'inner-0-0', 0, 0,
        epochs=1, scientific_selection=False)
    if len({c['id'] for c in cells}) != len(cells):
        raise ValueError('cell id collision')
    return cells


def validate_partition(component, train, test, development):
    tr, te, dev = set(map(int, train)), set(map(int, test)), set(map(int, development))
    if not tr or not te or tr & te or (tr | te) & dev:
        raise ValueError('invalid row partition or development exposure')
    if {str(component[i]) for i in tr} & {str(component[i]) for i in te}:
        raise ValueError('source component crosses partition')
    if ({str(component[i]) for i in tr | te} & {str(component[i]) for i in dev}):
        raise ValueError('source component crosses development boundary')


def register(args):
    import numpy as np
    old_path = args.original / 'PLAN.json'
    old = json.loads(old_path.read_text())
    if args.output.exists():
        raise ValueError('registration output exists; preserve it rather than overwrite')
    pins = {**old['inputs'], **old['data_sha256'], str(old_path): sha(old_path)}
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError(f'frozen input drift: {path}')
    data = args.original / 'data'
    component = np.load(data/'component.npy', allow_pickle=False)
    fold = np.load(data/'fold.npy', allow_pickle=False)
    inner = np.load(data/'inner.npy', allow_pickle=False)
    fit = np.flatnonzero(fold >= 0)
    dev = np.flatnonzero(fold < 0)
    if len(component) != old['rows'] or len(set(component[fit])) != old['fit_components']:
        raise ValueError('component inventory differs')
    partitions = {}
    for outer in range(3):
        tr = fit[fold[fit] != outer]; te = fit[fold[fit] == outer]
        validate_partition(component, tr, te, dev)
        partitions[f'outer-{outer}'] = dict(train=tr.tolist(), test=te.tolist())
        for k in range(2):
            itr = tr[inner[outer, tr] != k]; ite = tr[inner[outer, tr] == k]
            if set(inner[outer,tr]) != {0,1}:
                raise ValueError('invalid inner membership')
            validate_partition(component, itr, ite, dev)
            if set(map(int, itr)) | set(map(int, ite)) != set(map(int,tr)):
                raise ValueError('inner partition differs from outer complement')
            partitions[f'inner-{outer}-{k}'] = dict(train=itr.tolist(), test=ite.tolist())
    partitions['full-fit'] = dict(train=fit.tolist(), test=[], descriptive_development=dev.tolist())
    cells = matrix()
    protocol = REPO/'docs/SHAPING_COMPLETION_PROTOCOL.md'
    partition_bytes, matrix_bytes = encoded(partitions), encoded(cells)
    plan = dict(schema='shaping-completion-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                original_plan=str(old_path), original_plan_sha256=sha(old_path),
                protocol=str(protocol), protocol_sha256=sha(protocol), driver_sha256=sha(__file__),
                source_revision=subprocess.check_output(['jj','log','--no-graph','-r','@-','-T','commit_id'],cwd=REPO,text=True).strip(),
                input_pins=pins, partition_sha256=hashlib.sha256(partition_bytes).hexdigest(),
                matrix_sha256=hashlib.sha256(matrix_bytes).hexdigest(),
                seed_pairs=SEEDS, neural_settings=SETTINGS, families=FAMILIES,
                required_by_stage=dict(sorted(Counter(c['stage'] for c in cells).items())),
                conditional_rescue_cells=sum('condition' in c for c in cells),
                convergence_extensions='triggered and registered before affected refits; original cells retained',
                scientific_fits_started=0, binary_admission='required before any fit',
                minimum_ssd_free_bytes=30*1024**3)
    args.output.mkdir(parents=True)
    (args.output/'PARTITIONS.json').write_bytes(partition_bytes)
    (args.output/'MATRIX.json').write_bytes(matrix_bytes)
    (args.output/'PLAN.json').write_bytes(encoded(plan))
    print(json.dumps({'plan_sha256':sha(args.output/'PLAN.json'), 'counts':plan['required_by_stage'],
                      'scientific_fits_started':0}, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    register(p.parse_args())


if __name__ == '__main__':
    main()
