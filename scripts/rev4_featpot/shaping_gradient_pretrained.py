"""Check the paired-gradient trainer against two frozen corrected 120-epoch fits."""
import argparse,json,os,shutil,socket,subprocess,time
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_rescue_execution import portable_command

CASES={'product_pretrained_joint':'5e2cce20a78639f2cca8',
       'separate_pretrained_joint':'a868b99338b668c63467'}


def command_for(identity,source,binaries,output):
    command=portable_command(identity['command'],source)
    for flag,expected in (('--epochs','120'),('--pairs-per-epoch','50000')):
        if command.count(flag)!=1 or command[command.index(flag)+1]!=expected:
            raise ValueError('complete corrected pretrained exposure required')
    native=[i for i,v in enumerate(command) if Path(v).name=='zensim_mlp_train']
    if len(native)!=1:raise ValueError('one native trainer required')
    command[native[0]]=str(binaries/'zensim_mlp_train')
    for flag,value in (('--mem','2G'),('--out',output/'best.bin'),('--dump-checkpoints-dir',output/'checkpoints')):
        if command.count(flag)!=1:raise ValueError('one original command field required: '+flag)
        command[command.index(flag)+1]=str(value)
    return command


def register(study,binaries,output):
    from shaping_execution import load_registration
    from shaping_corrected_view import admit_fit,BUILD_SHA256
    from shaping_artifact_retention import archive_inventory
    if output.exists():raise ValueError('preserve previous pretrained qualification')
    _,cells,_=load_registration(study);cases={}
    for family,key in CASES.items():
        cell=dict(stage='inner',family=family,partition='inner-0-0',setting=0,seed=0,id=key)
        if cell not in cells:raise ValueError('prespecified corrected cell differs')
        source=study/'runs'/key/'attempt-001';identity=json.loads((source/'RUN.json').read_text());result=json.loads((source/'RESULT.json').read_text())
        if (identity['cell']!=cell or result['cell']!=cell or identity['build_sha256']!=BUILD_SHA256
            or result['build_sha256']!=BUILD_SHA256 or result['stage']!='trained' or result['exit_code']!=0
            or identity['plan_sha256']!=sha(study/'PLAN.json') or result['plan_sha256']!=sha(study/'PLAN.json')
            or identity['epochs']!=120 or identity['pairs_per_epoch']!=50000
            or result['epochs']!=120 or result['pairs_per_epoch']!=50000):
            raise ValueError('completed corrected 120-epoch source required')
        admit_fit(study,cell,BUILD_SHA256,120)
        files={str(source/name):sha(source/name) for name in ('RUN.json','RESULT.json','SHAPING.json','SHAPING.provenance.json',
            'keep.txt','COLLECTION_DERIVED_INPUTS.json','WORKER_COLLECTION.json')}
        groups=study/'comparators'/cell['partition']/'GROUPS.json';g=json.loads(groups.read_text())
        if any(w<=0 or v!=0 for _,_,w,v,_ in g['groups']):raise ValueError('TRAIN-only source groups required')
        files.update(g['file_sha256']);files[str(groups)]=sha(groups)
        mirror,pins=archive_inventory(source.parent)
        refs={f'ckpt_epoch{e:03}.bin':pins[f'attempt-001/checkpoints/ckpt_epoch{e:03}.bin'] for e in range(120)}
        if refs['ckpt_epoch119.bin']!=result['bake_sha256']:raise ValueError('archived corrected final model differs')
        command_for(identity,source,binaries,output/family)
        cases[family]=dict(cell=cell,source=str(source),source_file_sha256=files,
            reference_sha256=refs,archive=str(mirror),archive_manifest_sha256=sha(source.parent/'SHA256SUMS'))
    build=json.loads((binaries/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(binaries/name)!=pin:raise ValueError('paired-gradient binary differs')
    output.mkdir(parents=True);copied=0
    for family,case in cases.items():
        dest=output/'references'/family;dest.mkdir(parents=True)
        for name,pin in case['reference_sha256'].items():
            source=Path(case['archive'])/'attempt-001/checkpoints'/name;copied+=source.stat().st_size
            if copied>256*1024**2:raise ValueError('pretrained reference closure exceeds 256 MiB')
            shutil.copyfile(source,dest/name)
            if sha(dest/name)!=pin:raise ValueError('corrected reference checkpoint differs')
    record=dict(schema='paired-gradient-corrected-pretrained-v1',cases=cases,plan_sha256=sha(study/'PLAN.json'),
        build_sha256=sha(binaries/'BUILD.json'),binary_dir=str(binaries),epochs=120,pairs_per_epoch=50000,
        reference_bytes=copied,driver_sha256=sha(__file__),scientific_fit_count=0,engineering_native_fits=2,outer_results_read=False,
        imported_original_shaping_banks=True,pretraining_replayed=False,scientific_deployment_authorized=False)
    (output/'REGISTRATION.json').write_bytes(encoded(record));return record


def run(study,registration,family,output):
    from shaping_reduced_scope import admit_compute
    from shaping_rescue_dispatch import reserve_worker,bind_worker,release_worker
    from shaping_snapshot_retention import capacity,wait_and_bind
    from shaping_storage import reservation
    from shaping_neutral_trajectory import compare_checkpoints
    admit_compute(study);record=json.loads(registration.read_text());binaries=Path(record['binary_dir'])
    if (record['schema']!='paired-gradient-corrected-pretrained-v1' or record['driver_sha256']!=sha(__file__)
        or record['plan_sha256']!=sha(study/'PLAN.json') or record['build_sha256']!=sha(binaries/'BUILD.json')
        or set(record['cases'])!=set(CASES) or record['epochs']!=120 or record['pairs_per_epoch']!=50000
        or record['scientific_fit_count']!=0 or record['outer_results_read'] is not False
        or record['scientific_deployment_authorized'] is not False):raise ValueError('frozen pretrained qualification required')
    if output.exists():raise ValueError('preserve existing qualification; no implicit replay')
    case=record['cases'][family];source=Path(case['source']);key=case['cell']['id'];refs=registration.parent/'references'/family
    expected=dict(stage='inner',family=family,partition='inner-0-0',setting=0,seed=0,id=CASES[family])
    if (case['cell']!=expected or set(case['reference_sha256'])!={f'ckpt_epoch{e:03}.bin' for e in range(120)}):
        raise ValueError('all 120 prespecified corrected reference checkpoints required')
    for name,pin in {**case['source_file_sha256'],**{str(refs/n):p for n,p in case['reference_sha256'].items()}}.items():
        if sha(name)!=pin:raise ValueError('pretrained qualification source drift')
    for name,pin in json.loads((binaries/'BUILD.json').read_text())['binary_sha256'].items():
        if sha(binaries/name)!=pin:raise ValueError('candidate binary drift')
    while True:
        if not reserve_worker(study,key)['can_start']:time.sleep(10);continue
        try:ready=capacity(study,key)['can_start']
        except BaseException:release_worker(study,key);raise
        if ready:break
        release_worker(study,key);time.sleep(10)
    try:
        with reservation(study,output,30*1024**3):
            output.mkdir(parents=True);(output/'checkpoints').mkdir()
            command=command_for(json.loads((source/'RUN.json').read_text()),source,binaries,output)
            env={k:v for k,v in os.environ.items() if not k.startswith(('ZENSIM_MONOTONE_RESEARCH_',
                'ZENSIM_INPUT_SHAPING_','ZENSIM_PLAIN_OPTIMIZER_','ZENSIM_TRAIN_RESUME','ZENSIM_TRAIN_SNAPSHOT'))}
            env.update(TMPDIR=str(Path.home()/'tmp'),OMP_NUM_THREADS='1',RAYON_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                ZENSIM_TRAIN_SNAPSHOT_ROOT=str(study.parent/'shared-training-snapshots'),
                ZENSIM_INPUT_SHAPING_MANIFEST=str(source/'SHAPING.json'),ZENSIM_INPUT_SHAPING_SHA256=sha(source/'SHAPING.json'))
            identity=dict(cell=case['cell'],source_run=str(source),registration_sha256=sha(registration),
                build_sha256=record['build_sha256'],command=command,execution_host=socket.gethostname(),
                scientific_selection=False,outer_results_read=False,epochs=120,pairs_per_epoch=50000,stage='running')
            (output/'RUN.json').write_bytes(encoded(identity))
            with (output/'train.log').open('xb') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                bind_worker(study,key,child.pid);code=wait_and_bind(child,study.parent,output)
            (output/'RESULT.json').write_bytes(encoded(dict(identity,stage='trained' if code==0 else 'failed',exit_code=code)))
            if code:raise RuntimeError('candidate pretrained run failed; preserve all state')
            compared=compare_checkpoints(refs,output/'checkpoints',binaries/'bake_dial_refit',output/'trajectory',120,('zentrain.repro',))
            proof=dict(stage='qualified' if compared['all_equal'] else 'trajectory_mismatch',trajectory=compared,
                all_120_checkpoints_equal=compared['all_equal'],registration_sha256=sha(registration),
                build_sha256=record['build_sha256'],source_file_sha256=case['source_file_sha256'],scientific_deployment_authorized=False)
            (output/'QUALIFICATION.json').write_bytes(encoded(proof))
            if not compared['all_equal']:raise ValueError('corrected pretrained trajectory differs; deployment blocked')
            return proof
    finally:release_worker(study,key)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','run'))
    for name in ('study','binary-dir','registration','output'):p.add_argument('--'+name,type=Path)
    p.add_argument('--family',choices=CASES);a=p.parse_args()
    value=register(a.study,a.binary_dir,a.output) if a.mode=='register' else run(a.study,a.registration,a.family,a.output)
    print(json.dumps({k:v for k,v in value.items() if k not in ('cases','trajectory','source_file_sha256')}),flush=True)
