"""Prepare and execute the finite November 13 follow-up; never replay a fit."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from shaping_completion import encoded, sha


def write(path, value):
    path.write_bytes(encoded(value))


def predictions(study, archive, cell, pins):
    import numpy as np
    cid = cell['id']
    candidates = [study/'reexport_max_pairs'/cid/'attempt-001/final',
        archive/f'supplemental-panels-{cid}'/'attempt-001/final',
        study/'runs'/cid/'attempt-001/evaluation',
        archive/f'inner-{cid}'/'attempt-001/evaluation',
        archive/'tree-inner-archive/runs'/cid/'attempt-001/evaluation']
    for directory in candidates:
        path = directory/'predictions.npz'
        if not path.exists():
            continue
        report = json.loads((directory/'REPORT.json').read_text())
        marker = json.loads((directory/'COMPLETE.json').read_text())
        if report['cell'] != cell or marker['report_sha256'] != sha(directory/'REPORT.json'):
            raise ValueError('cached prediction report identity differs')
        if report['plan_sha256'] != sha(study/'PLAN.json'):
            raise ValueError('cached prediction bytes or plan differ')
        if 'predictions_sha256' in report:
            if report['predictions_sha256'] != sha(path):
                raise ValueError('cached prediction bytes differ')
        else:
            # Early tree reports predate the NPZ field. Their verified archive
            # binds the table, report and marker together; do not invent a pin.
            from shaping_artifact_retention import archive_inventory
            root = next((p for p in directory.parents if (p/'SHA256SUMS').exists()), None)
            if root is None:
                raise ValueError('legacy table needs its verified archive')
            _, inventory = archive_inventory(root)
            for p in (path, directory/'REPORT.json', directory/'COMPLETE.json'):
                if inventory.get(p.relative_to(root).as_posix()) != sha(p):
                    raise ValueError('legacy archived prediction identity differs')
            for p in (root/'SHA256SUMS',root/'MIRROR_VERIFIED.json'):
                pins[str(p)] = sha(p)
        for p in (path, directory/'REPORT.json', directory/'COMPLETE.json'):
            pins[str(p)] = sha(p)
        with np.load(path, allow_pickle=False) as a:
            indices, values = a['original_indices'], a['native']
        if indices.shape != values.shape or not np.isfinite(values).all():
            raise ValueError('prediction shape or finiteness differs')
        return {int(i): float(v) for i, v in zip(indices, values)}
    raise ValueError('no retained prediction table: '+cid)


def prepare_human(study, archive, output, binary, source):
    import numpy as np
    import pyarrow.parquet as pq
    from scipy.stats import rankdata
    from shaping_execution import load_registration
    from shaping_outer_admission import resolve
    from shaped_linear_study import OLD, PREP
    if output.exists():
        raise ValueError('preserve existing preparation')
    _, cells, partitions = load_registration(study)
    admission, decision = resolve(study, 0)
    family = admission['finalist_roles']['best_product']
    if family != 'product_joint_identity':
        raise ValueError('registered product finalist differs')
    settings = {f: admission['schedules'][f]['setting'] for f in (family, 'extra_trees')}
    train = set(partitions['outer-0']['train'])
    arrays = {n: np.load(OLD/'data'/f'{n}.npy', allow_pickle=False) for n in ('y','pop','refs')}
    pins = {str(p): sha(p) for p in (study/'PLAN.json', study/'PARTITIONS.json', decision, PREP)}
    pins.update({str(OLD/'data'/f'{n}.npy'): sha(OLD/'data'/f'{n}.npy') for n in arrays})
    flags = {}; proof = []
    for fold in range(2):
        partition = f'inner-0-{fold}'; member = partitions[partition]; ids = np.array(member['test'])
        if set(member['train']) & set(ids) or set(member['train']) | set(ids) != train:
            raise ValueError('inner folds do not partition outer training complement')
        # The original partition authority owns cross-dataset source components.
        components = np.load(Path(json.loads((study/'PLAN.json').read_text())['original_plan']).parent/'data/component.npy', allow_pickle=False)
        if set(components[member['train']]) & set(components[ids]):
            raise ValueError('cross-fitted source component overlap')
        cache = {}
        for f in (family, 'extra_trees'):
            cache[f] = []
            for seed in range(3):
                selected = [c for c in cells if c['stage'] == 'inner' and c['partition'] == partition
                    and c['family'] == f and c['setting'] == settings[f] and c['seed'] == seed]
                if len(selected) != 1:
                    raise ValueError('one matched cached condition required')
                values = predictions(study, archive, selected[0], pins)
                cache[f].append(np.array([values[int(i)] for i in ids]))
        for pop in sorted(set(arrays['pop'][ids])):
            m = arrays['pop'][ids] == pop; local = ids[m]; n = len(local)
            qy = (rankdata(arrays['y'][local])-.5)/n
            model_ranks = {f: (rankdata(np.mean([(rankdata(v[m])-.5)/n for v in cache[f]], axis=0))-.5)/n
                           for f in cache}
            ep = np.abs(model_ranks[family]-qy); et = np.abs(model_ranks['extra_trees']-qy)
            selected = (ep >= np.quantile(ep, .9)) & (et < ep) if pop != 'cid22_ssim2' else np.zeros(n, bool)
            for i, flag in zip(local, selected):
                if int(i) in flags:
                    raise ValueError('duplicate cross-fitted row')
                flags[int(i)] = bool(flag)
            proof.append(dict(partition=partition, population=str(pop), rows=n,
                              hard_rows=int(selected.sum()), source_components_disjoint=True))
    if set(flags) != train:
        raise ValueError('cross-fitted training coverage incomplete')
    weights = {i: 1.+float(flags[i]) for i in train}
    for pop in set(arrays['pop'][list(train)]):
        for ref in set(arrays['refs'][[i for i in train if arrays['pop'][i] == pop]]):
            ids = [i for i in train if arrays['pop'][i] == pop and arrays['refs'][i] == ref]
            mean = np.mean([weights[i] for i in ids])
            for i in ids:
                weights[i] /= mean
    output.mkdir(parents=True)
    groups_path = study/'comparators/outer-0/GROUPS.json'
    groups = json.loads(groups_path.read_text()); manifest = []; offset = 0
    prepared = json.loads(PREP.read_text())
    by_name = {g[0]: g for g in groups['groups']}
    for g in prepared['groups']:
        ids = [i for i in range(offset, offset+g['rows']) if i in train]; offset += g['rows']
        if not ids:
            continue
        path = Path(by_name[g['name']][1])
        scores = pq.read_table(path, columns=['human_score'])['human_score'].to_numpy().astype('<f8')
        if not np.array_equal(scores, arrays['y'][ids]):
            raise ValueError('prepared score order differs')
        manifest.append(dict(name=g['name'], scores_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
                             weights=[float(weights[i]) for i in ids]))
    if [g['name'] for g in manifest] != [g[0] for g in groups['groups']]:
        raise ValueError('native group order differs')
    for p, pin in groups['file_sha256'].items():
        if sha(p) != pin:
            raise ValueError('training closure drift')
        pins[p] = pin
    weights_path = output/'HARD_CASE_WEIGHTS.json'
    write(weights_path, dict(schema='human-hard-case-rank-weights-v1', groups=manifest))
    write(output/'CROSSFIT_PROOF.json', dict(scope='outer-0 training complement only', panels=proof,
        input_pins=pins, final_holdout_read=False, outer_test_used_for_weights=False))
    declaration = source/'scripts/rev4_featpot/color_noise_fx2_declaration.json'
    jobs = []
    for seed in range(3):
        cell, = [c for c in cells if c['stage'] == 'outer' and c['partition'] == 'outer-0'
            and c['family'] == family and c['seed'] == seed]
        baseline = study/'runs'/cell['id']/'attempt-001/fit'
        old = json.loads((baseline/'RUN.json').read_text())
        command = old['command']; command = command[command.index('--')+1:]
        for arm in ('hard_case', 'negative_abs_init', 'negative_softplus'):
            dest = output.parent/'runs'/f'human-{arm}-s{seed}'
            config = output/f'{arm}-s{seed}'; config.mkdir()
            shutil.copy2(baseline/'SHAPING.json', config/'SHAPING.json')
            optimizer = json.loads((baseline/'OPTIMIZER.json').read_text())
            if arm != 'hard_case':
                optimizer['output_update'] = arm
            write(config/'OPTIMIZER.json', optimizer)
            keep = config/'keep.txt'
            shutil.copy2(command[command.index('--keep-features')+1], keep)
            argv = list(command); argv[0] = str(binary)
            for flag, value in (('--out', dest/'best.bin'), ('--dump-checkpoints-dir', dest/'checkpoints'),
                                ('--keep-features', keep), ('--derived-inputs', declaration)):
                argv[argv.index(flag)+1] = str(value)
            env = dict(ZENSIM_INPUT_SHAPING_MANIFEST=str(config/'SHAPING.json'),
                ZENSIM_INPUT_SHAPING_SHA256=sha(config/'SHAPING.json'),
                ZENSIM_PLAIN_OPTIMIZER_MANIFEST=str(config/'OPTIMIZER.json'),
                ZENSIM_PLAIN_OPTIMIZER_SHA256=sha(config/'OPTIMIZER.json'), ZENSIM_OPTIMIZER_DIAGNOSTICS='1',
                ZENSIM_TRAIN_SNAPSHOT_ROOT=str(output.parent/'human-snapshots'), ZENSIM_FOLLOWUP_TRAINER_SHA256=sha(binary))
            if arm == 'hard_case':
                env.update(ZENSIM_HARD_CASE_MANIFEST=str(weights_path), ZENSIM_HARD_CASE_SHA256=sha(weights_path))
            job = dict(id=dest.name, arm=arm, seed=seed, output=str(dest), argv=argv, env=env,
                binary_sha256=sha(binary), baseline_cell=cell, baseline_result_sha256=sha(baseline/'RESULT.json'),
                file_pins={**groups['file_sha256'], str(declaration):sha(declaration),
                    **{str(p):sha(p) for p in config.iterdir()}, str(weights_path):sha(weights_path)},
                epochs=240, pairs_per_epoch=50000, final_holdout_read=False)
            request = config/'JOB.json';write(request, job);jobs.append(str(request))
    write(output/'JOBS.json', jobs)
    print(json.dumps(dict(stage='prepared', jobs=jobs, crossfit=proof)), flush=True)


def run_human(request):
    job = json.loads(request.read_text()); dest = Path(job['output'])
    if dest.exists():
        raise ValueError('existing fit is preserved; use an explicitly audited full-state resume')
    if sha(job['argv'][0]) != job['binary_sha256']:
        raise ValueError('trainer binary drift')
    for path, pin in job['file_pins'].items():
        if sha(path) != pin:
            raise ValueError('job input drift: '+path)
    if shutil.disk_usage(dest.parent.parent).free < 34*1024**3:
        raise ValueError('30 GiB floor plus two fit reservations unavailable')
    dest.mkdir(parents=True); (dest/'checkpoints').mkdir()
    env = {k:v for k,v in os.environ.items() if not k.startswith('ZENSIM_')}
    env.update(job['env']); env.update(TMPDIR=str(Path.home()/'tmp'), RAYON_NUM_THREADS='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    heavy = Path.home()/'work/claudehints/scripts/run-heavy'
    argv = [str(heavy), '--mem', '2G', '--jobs', '1', '--', *job['argv']]
    record = dict(job=job, request_sha256=sha(request), pid=os.getpid(), host=__import__('socket').gethostname(),
        stage='training', started_utc=datetime.now(timezone.utc).isoformat())
    with (dest/'train.log').open('x') as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['wrapper_pid'] = proc.pid;write(dest/'RUN.json', record);code=proc.wait()
    final = dest/'checkpoints/ckpt_epoch239.bin'
    record.update(stage='trained' if code == 0 and final.exists() else 'failed_preserved',
        exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat(),
        final_bake_sha256=sha(final) if final.exists() else None)
    write(dest/'RESULT.json', record)
    if record['stage'] != 'trained':
        raise RuntimeError('fit failed; preserve its attempt and checkpoints')
    print(json.dumps(dict(id=job['id'], stage='trained', bake_sha256=record['final_bake_sha256'])), flush=True)


def qualify(study, binary, output):
    """Short real-input old/new arithmetic control, separate from research fits."""
    if output.exists():
        raise ValueError('preserve previous qualification')
    output.mkdir(parents=True)
    baseline=study/'runs/ec63ca02f61e2062c868/attempt-001/fit'
    command=json.loads((baseline/'RUN.json').read_text())['command']
    command=command[command.index('--')+1:]
    env={k:v for k,v in os.environ.items() if not k.startswith('ZENSIM_')}
    env.update(TMPDIR=str(Path.home()/'tmp'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1',
        ZENSIM_INPUT_SHAPING_MANIFEST=str(baseline/'SHAPING.json'),ZENSIM_INPUT_SHAPING_SHA256=sha(baseline/'SHAPING.json'))
    heavy=Path.home()/'work/claudehints/scripts/run-heavy'
    for depth in ('plain','shared'):
        for version in ('old','new'):
            dest=output/f'{depth}-{version}';dest.mkdir();(dest/'checkpoints').mkdir()
            argv=list(command)
            if depth=='plain':
                for flag in ('--n-hidden-layers','--shared-hidden-activation'):
                    i=argv.index(flag);del argv[i:i+2]
            if version=='new':argv[0]=str(binary)
            for flag,value in (('--epochs','2'),('--pairs-per-epoch','64'),('--out',str(dest/'best.bin')),
                               ('--dump-checkpoints-dir',str(dest/'checkpoints'))):
                argv[argv.index(flag)+1]=value
            write(dest/'COMMAND.json',argv)
            with (dest/'train.log').open('x') as log:
                subprocess.run([str(heavy),'--mem','2G','--jobs','1','--',*argv],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for epoch in range(2):
            hashes=[]
            for version in ('old','new'):
                dest=output/f'{depth}-{version}';original=dest/'checkpoints'/f'ckpt_epoch{epoch:03}.bin'
                stripped=dest/f'stripped-{epoch}.bin'
                subprocess.run([str(binary.parent/'bake_dial_refit'),'strip','--in',str(original),'--out',str(stripped),
                    '--key','zentrain.repro'],stdout=subprocess.DEVNULL,check=True)
                hashes.append(sha(stripped))
            if hashes[0]!=hashes[1]:
                raise ValueError(f'changed default model bytes: {depth}/{epoch}')
    write(output/'QUALIFIED.json',dict(stage='pass',check='Both fixed checkpoints match old native bytes after repro stripping; plain and shared-depth, real admitted inputs.',
        new_binary_sha256=sha(binary),old_binary_sha256=sha(command[0]),fits_replayed=0,
        qualification_draws_per_case=128,scientific_gate_unchanged=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare-human')
    for name in ('study','archive','output','binary','source'):
        prep.add_argument('--'+name,type=Path,required=True)
    run=sub.add_parser('run-human');run.add_argument('--request',type=Path,required=True)
    check=sub.add_parser('qualify')
    for name in ('study','binary','output'):check.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.action == 'prepare-human':
        prepare_human(a.study,a.archive,a.output,a.binary,a.source)
    elif a.action=='qualify':
        qualify(a.study,a.binary,a.output)
    else:
        run_human(a.request)
