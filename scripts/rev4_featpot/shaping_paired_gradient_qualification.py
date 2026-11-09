"""Qualify only the paired-gradient runtime's two affected joint-shaping classes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from shaping_completion import encoded,sha
from shaping_control_compatibility import command_for
from shaping_neutral_trajectory import compare_checkpoints

CASES={'product_joint_identity':'caa39cae7cd461126a97',
       'separate_joint_identity':'1d9acf972b7722929b41'}


def register(study, original_registration, binaries, output):
    from shaping_checkpoint_retention import archive_inventory
    original=json.loads(original_registration.read_text())
    if (original['schema']!='corrected-runtime-control-compatibility-v1'
            or original['plan_sha256']!=sha(study/'PLAN.json')
            or original['epochs']!=240 or original['pairs_per_epoch']!=50000
            or original['scientific_selection'] or original['outer_results_read']):
        raise ValueError('original prespecified compatibility cases required')
    build=json.loads((binaries/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(binaries/name)!=pin:raise ValueError('candidate binary identity differs')
    if output.exists():raise ValueError('preserve previous candidate registration')
    cases={}
    for family,cid in CASES.items():
        case=original['cases'][family]
        if case['cell']!=dict(id=cid,family=family,partition='inner-0-0',stage='inner',setting=0,seed=0):
            raise ValueError('prespecified joint-shaping case differs')
        for name,pin in case['source_file_sha256'].items():
            if sha(name)!=pin:raise ValueError('original compatibility source differs')
        directory=Path(case['run']).parent
        mirror,pins=archive_inventory(directory)
        references={f'ckpt_epoch{e:03}.bin':pins[f'attempt-001/checkpoints/ckpt_epoch{e:03}.bin'] for e in range(240)}
        if references['ckpt_epoch239.bin']!=case['original_final_sha256']:
            raise ValueError('archived control final identity differs')
        cases[family]=dict(case,reference_sha256=references,archive=str(mirror),
                          archive_manifest_sha256=sha(directory/'SHA256SUMS'))
    output.mkdir(parents=True)
    copied=0
    for family,case in cases.items():
        dest=output/'references'/family;dest.mkdir(parents=True)
        for name,pin in case['reference_sha256'].items():
            source=Path(case['archive'])/'attempt-001/checkpoints'/name
            copied+=source.stat().st_size
            if copied>256*1024**2:raise ValueError('joint reference closure exceeds 256 MiB')
            shutil.copyfile(source,dest/name)
            if sha(dest/name)!=pin:raise ValueError('archived reference checkpoint differs')
    record=dict(schema='paired-gradient-native-trajectory-v1',cases=cases,
        original_registration_sha256=sha(original_registration),plan_sha256=sha(study/'PLAN.json'),
        build_sha256=sha(binaries/'BUILD.json'),binary_dir=str(binaries),
        epochs=240,pairs_per_epoch=50000,reference_bytes=copied,
        scientific_selection=False,outer_results_read=False,other_families_qualified=False,
        driver_sha256=sha(__file__),created_utc=datetime.now(timezone.utc).isoformat())
    (output/'REGISTRATION.json').write_bytes(encoded(record));return record


def run(study,registration,family,output):
    from shaping_reduced_scope import admit_compute
    from shaping_rescue_dispatch import reserve_worker,bind_worker,release_worker
    from shaping_snapshot_retention import capacity,wait_and_bind
    from shaping_storage import reservation
    admit_compute(study);record=json.loads(registration.read_text());binaries=Path(record['binary_dir'])
    if (record['schema']!='paired-gradient-native-trajectory-v1' or record['driver_sha256']!=sha(__file__)
            or record['plan_sha256']!=sha(study/'PLAN.json') or record['build_sha256']!=sha(binaries/'BUILD.json')
            or record['scientific_selection'] or record['outer_results_read'] or set(record['cases'])!=set(CASES)):
        raise ValueError('registered candidate qualification required')
    case=record['cases'][family];key=case['cell']['id'];source=Path(case['run'])
    if output.exists():raise ValueError('preserve previous qualification attempt; never replay implicitly')
    reference=registration.parent/'references'/family
    for name,pin in case['source_file_sha256'].items():
        if sha(name)!=pin:raise ValueError('original control evidence differs')
    for name,pin in case['reference_sha256'].items():
        if sha(reference/name)!=pin:raise ValueError('original full trajectory differs')
    for name,pin in json.loads((binaries/'BUILD.json').read_text())['binary_sha256'].items():
        if sha(binaries/name)!=pin:raise ValueError('candidate binary differs')
    while True:
        if not reserve_worker(study,key)['can_start']:
            time.sleep(10);continue
        try:ready=capacity(study,key)['can_start']
        except BaseException:release_worker(study,key);raise
        if ready:break
        release_worker(study,key);time.sleep(10)
    try:
        with reservation(study,output,30*1024**3):
            output.mkdir(parents=True);(output/'checkpoints').mkdir()
            original=json.loads((source/'RUN.json').read_text())
            command=command_for(dict(original,_source=str(source)),binaries,output)
            env=os.environ.copy()
            for name in list(env):
                if name.startswith(('ZENSIM_MONOTONE_RESEARCH_','ZENSIM_INPUT_SHAPING_',
                                    'ZENSIM_PLAIN_OPTIMIZER_','ZENSIM_TRAIN_RESUME','ZENSIM_TRAIN_SNAPSHOT')):
                    env.pop(name,None)
            env.update(TMPDIR=str(Path.home()/'tmp'),OMP_NUM_THREADS='1',RAYON_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                       ZENSIM_TRAIN_SNAPSHOT_ROOT=str(study.parent/'shared-training-snapshots'),
                       ZENSIM_INPUT_SHAPING_MANIFEST=str(source/'SHAPING.json'),
                       ZENSIM_INPUT_SHAPING_SHA256=sha(source/'SHAPING.json'))
            result=dict(cell=case['cell'],source_run=str(source),registration_sha256=sha(registration),
                build_sha256=record['build_sha256'],command=command,execution_host=socket.gethostname(),
                scientific_selection=False,outer_results_read=False,epochs=240,pairs_per_epoch=50000,stage='running')
            (output/'RUN.json').write_bytes(encoded(result))
            with (output/'train.log').open('xb') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                bind_worker(study,key,child.pid)
                code=wait_and_bind(child,study.parent,output)
            result.update(exit_code=code,stage='trained' if not code else 'failed')
            (output/'TRAINING_RESULT.json').write_bytes(encoded(result))
            if code:raise RuntimeError('native qualification failed; preserve all states')
            comparison=compare_checkpoints(reference,output/'checkpoints',binaries/'bake_dial_refit',
                output/'trajectory',240,('zentrain.repro',))
            result.update(stage='qualified' if comparison['all_equal'] else 'trajectory_mismatch',
                          trajectory=comparison,all_240_checkpoints_equal=comparison['all_equal'])
            (output/'RESULT.json').write_bytes(encoded(result))
            if not comparison['all_equal']:raise ValueError('joint trajectory differs; candidate use blocked')
            return result
    finally:release_worker(study,key)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','run'))
    for name in ('study','original-registration','binary-dir','registration','output'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--family',choices=CASES);a=p.parse_args()
    result=register(a.study,a.original_registration,a.binary_dir,a.output) if a.mode=='register' else run(a.study,a.registration,a.family,a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','trajectory')}),flush=True)
