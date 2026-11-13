#!/usr/bin/env python3
"""Execute the two registered A-recipe follow-up arms with bounded snapshots.

Each invocation owns one fresh arm/seed directory. Original recipe files and
released inputs stay read-only; no final holdout or pixel extraction is used.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import math
import statistics
import struct


def prepare_templates(identity, banks, extra, output):
    if output.exists():
        raise ValueError('preserve existing template registration')
    configs = [json.loads(p.read_text()) for p in banks]
    template = json.loads(identity.read_text())
    if len(configs) != 3 or any(len(c['terms']) != len(template['terms']) for c in configs):
        raise ValueError('three schema-matched base-phase banks required')
    frozen = json.loads(identity.read_text()); frozen['trainable'] = False
    for i, term in enumerate(template['terms']):
        for c in configs:
            if any(c['terms'][i][k] != term[k] for k in ('column','damage','reference','separate')) or any(
                [struct.pack('<f',v) for v in c['terms'][i][k]] != [struct.pack('<f',v) for v in term[k]]
                for k in ('knots','gate')):
                raise ValueError('bank identity, anchor, knots or gates differ')
        slopes = [statistics.median(c['terms'][i]['slopes'][k] for c in configs) for k in range(3)]
        if not all(math.isfinite(v) and v >= 0 for v in slopes):
            raise ValueError('bank slope domain differs')
        frozen['terms'][i]['slopes'] = slopes
    output.mkdir(parents=True)
    write(output/'slow_joint.json', template)
    write(output/'common_frozen.json', frozen)
    write(output/'CONTROLS.json', dict(schema='shaping-stability-v1',lr_scale=0.1,identity_l2=0.001))
    import shutil
    shutil.copy2(extra, output/'EXTRA_READONLY_INPUTS.json')
    for item in json.loads(extra.read_text()):
        source=extra.parent/item['file']; destination=output/item['file']
        if sha(source) != item['sha256']:
            raise ValueError('A provenance input drift')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    write(output/'BANK_PROVENANCE.json',dict(identity_sha256=sha(identity),
        base_banks={str(p):sha(p) for p in banks},reuse='Three completed base-phase A banks; no new bank fitting.',
        training_exposure_per_bank=dict(epochs=120,pairs_per_epoch=50000),
        heldout_labels_used=False,rule='coordinatewise median; identical field identities, gates and knots'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(args):
    recipe_path = args.release_root / 'recipes' / f'MixF3-s{args.seed}.json'
    recipe = json.loads(recipe_path.read_text())
    if 'public_test' in json.dumps(recipe):
        raise ValueError('final holdout is outside this comparison')
    if args.work.exists():
        raise ValueError('preserve every existing fit attempt')
    args.work.mkdir(parents=True)
    if os.statvfs(args.work).f_bavail * os.statvfs(args.work).f_frsize < 38 * 1024**3:
        raise ValueError('30 GiB reserve plus two 4 GiB fit reservations unavailable')
    mounts = list(csv.DictReader((args.release_root / 'release/2026-11-04-mix/MOUNTS.tsv').open(), delimiter='\t'))
    status = dict(schema='zensim-a-shaping-comparison-run-v1', seed=args.seed, arm=args.arm,
                  recipe_sha256=sha(recipe_path), runner_sha256=sha(__file__), pid=os.getpid(),
                  trainer_sha256=sha(args.binaries / 'zensim_mlp_train'), steps=[],
                  final_holdouts_read=False, stage='preparing')
    write(args.work / 'RUN.json', status)
    fresh = {}
    for index, step in enumerate(recipe['steps'][:2]):
        directory = args.work / f'{index}-{step["name"]}'
        directory.mkdir()
        writes = directory / 'writes'
        scratch = directory / 'scratch'
        writes.mkdir()
        scratch.mkdir()
        # The physical scratch paths are disk backed. Historical origins are
        # aliases inside this process's mount namespace, never host writes.
        command = ['bwrap', '--dev-bind', '/', '/']
        # Hide historical mutable trees before installing released read-only
        # aliases. Rebind this run's disk-backed directory explicitly.
        for hidden in ('/var/tmp', '/mnt', str(Path.home() / 'tmp')):
            command += ['--tmpfs', hidden]
        command += ['--bind', str(args.work), str(args.work), '--bind', str(args.snapshots), str(args.snapshots)]
        for item in json.loads((args.shape_template.parent / 'EXTRA_READONLY_INPUTS.json').read_text()):
            source = args.shape_template.parent / item['file']
            if sha(source) != item['sha256']:
                raise ValueError('extra frozen provenance input differs')
            command += ['--ro-bind', str(source), item['origin']]
        for mount in sorted(mounts, key=lambda m: len(m['origin'])):
            source = args.release_root / mount['release_prefix']
            if source.exists():
                command += ['--ro-bind', str(source), mount['origin']]
        command += ['--ro-bind', str(args.binaries), str(args.work / 'native-bin')]
        for origin in step.get('reads_from', {}):
            command += ['--ro-bind', str(fresh[origin]), origin]
        command += ['--bind', str(writes), step['writes'], '--bind', str(scratch), step['tmpdir']]
        fresh[step['writes']] = writes
        environment = dict(step.get('env', {}))
        environment.update(REV4_V2_BIN_DIR=str(args.work / 'native-bin'), TMPDIR=step['tmpdir'],
                           RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1',
                           PATH=str(Path(sys.executable).parent) + ':' + os.environ.get('PATH', ''),
                           ZENSIM_TRAIN_SNAPSHOT_ROOT=str(args.snapshots))
        environment['PYTHONPATH'] = environment.get('PYTHONPATH', '') + ':' + os.environ['ZENSIM_FOLLOWUP_PYTHON_SITE']
        if args.arm in ('slow_joint', 'common_frozen'):
            if index == 0:
                shape = args.shape_template
            else:
                previous = fresh[recipe['steps'][0]['writes']] / 'refit/last.bin'
                inspection = subprocess.run([str(args.inspector), str(previous)],
                                            capture_output=True, check=True, text=True)
                (directory / 'initial-model-inspection.json').write_text(inspection.stdout)
                metadata = json.loads(inspection.stdout)['metadata']
                entry, = [v for v in metadata if v['key'] == 'zentrain.input_shaping']
                shape = directory / 'continuation-shapes.json'
                write(shape, json.loads(entry['value_text']))
            visible_shape = directory / 'shapes-input.json'
            command += ['--ro-bind', str(shape), str(visible_shape)]
            environment.update(ZENSIM_INPUT_SHAPING_MANIFEST=str(visible_shape), ZENSIM_INPUT_SHAPING_SHA256=sha(shape))
        else:
            # A service must not inherit another experiment's manifest.
            command += ['--unsetenv', 'ZENSIM_INPUT_SHAPING_MANIFEST',
                        '--unsetenv', 'ZENSIM_INPUT_SHAPING_SHA256']
        if args.arm == 'slow_joint':
            visible_controls = directory / 'shape-optimizer.json'
            command += ['--ro-bind', str(args.controls), str(visible_controls)]
            environment.update(ZENSIM_SHAPE_OPTIMIZER_MANIFEST=str(visible_controls),
                               ZENSIM_SHAPE_OPTIMIZER_SHA256=sha(args.controls))
        else:
            command += ['--unsetenv', 'ZENSIM_SHAPE_OPTIMIZER_MANIFEST', '--unsetenv', 'ZENSIM_SHAPE_OPTIMIZER_SHA256']
        for key, value in environment.items():
            command += ['--setenv', key, value]
        command += ['--chdir', str(directory), '--die-with-parent']
        argv = [sys.executable if v == '{python}' else v for v in step['argv']]
        # These options only run post-fit packing. Training preparation, group
        # weights, sampler, architecture, epochs and fixed checkpoint stay exact.
        argv = [v for v in argv if v not in ('--pack-production', '--e33-output-stage')]
        command += argv
        write(directory / 'COMMAND.json', command)
        status.update(stage=step['name'])
        write(args.work / 'RUN.json', status)
        with (directory / 'step.log').open('x') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            except OSError as error:
                status.update(stage='failed_preserved', launch_error=str(error))
                write(args.work / 'RUN.json', status)
                raise
        final = writes / 'refit/last.bin'
        result = dict(name=step['name'], exit_code=code, final_bake_sha256=sha(final) if final.exists() else None)
        status['steps'].append(result)
        if code or not final.exists():
            status.update(stage='failed_preserved')
            write(args.work / 'RUN.json', status)
            raise RuntimeError(f'{step["name"]} failed; preserve its log and partial outputs')
        write(args.work / 'RUN.json', status)
    status.update(stage='trained')
    write(args.work / 'RUN.json', status)
    print(json.dumps(status), flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='prepare-templates':
        p=argparse.ArgumentParser();p.add_argument('action')
        p.add_argument('--identity',type=Path,required=True);p.add_argument('--bank',type=Path,action='append',required=True)
        p.add_argument('--extra',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        a=p.parse_args();prepare_templates(a.identity,a.bank,a.extra,a.output);return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release-root', 'work', 'binaries', 'shape-template', 'inspector', 'snapshots', 'controls'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=range(3), required=True)
    parser.add_argument('--arm', choices=('slow_joint', 'common_frozen'), required=True)
    args = parser.parse_args()
    for name in ('release_root', 'work', 'binaries', 'shape_template', 'inspector', 'snapshots', 'controls'):
        setattr(args, name, getattr(args, name).resolve())
    run(args)


if __name__ == '__main__':
    main()
