"""Qualify corrected-runtime reuse of completed controls, without model selection."""
import argparse,json,os,socket,subprocess,time
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_neutral_trajectory import compare_checkpoints

FAMILIES=('relu','softplus','silu_monotone','silu_diagnostic','product_joint_identity','separate_joint_identity')
BUILD_SHA256='56a05880e05dcfa78d28207a88df6f605888814e47707e216e5941d743ac0cff'
PROPOSAL_SHA256='96d5ccbe3fa1104523a6aeb5c36b4d0632edbc74db8975e782a548fdbfe99c66'


def register(study,approval,binaries,output):
    from shaping_execution import load_registration
    approved=json.loads(approval.read_text())
    if (approved['answer']!='Approve four settings: 1,461 fits' or approved['selected_option']!='four_settings'
        or approved['proposal_sha256']!=PROPOSAL_SHA256 or approved['corrected_fits']!=288
        or approved['total_core_fits']!=1461 or not approved['qualification_required_before_control_reuse']
        or sha(binaries/'BUILD.json')!=BUILD_SHA256):raise ValueError('approved corrected-runtime scope required')
    _,cells,_=load_registration(study);cases={}
    for family in FAMILIES:
        found=[c for c in cells if c['stage']=='inner' and c['family']==family and
               (c['id']=='e3c79ecb9ba271091aa1' if family=='silu_monotone' else
                c['partition']=='inner-0-0' and c['setting']==0 and c['seed']==0)]
        if len(found)!=1:raise ValueError('one prespecified control per class required')
        cell=found[0];run=study/'runs'/cell['id']/'attempt-001'
        result=json.loads((run/'RESULT.json').read_text());identity=json.loads((run/'RUN.json').read_text())
        if (identity['cell']!=cell or result['cell']!=cell or result['stage']!='trained' or result['exit_code']!=0
            or identity['epochs']!=240 or identity['pairs_per_epoch']!=50000):raise ValueError('completed full control required')
        files={str(run/name):sha(run/name) for name in ('RUN.json','RESULT.json','keep.txt')}
        for name in ('HEAD.json','SHAPING.json','WORKER_COLLECTION.json','COLLECTION_DERIVED_INPUTS.json'):
            if (run/name).exists():files[str(run/name)]=sha(run/name)
        groups=study/'comparators'/cell['partition']/'GROUPS.json';g=json.loads(groups.read_text())
        if any(weight<=0 or validation!=0 for _,_,weight,validation,_ in g['groups']):raise ValueError('TRAIN-only compatibility groups required')
        files.update(g['file_sha256']);files[str(groups)]=sha(groups)
        cases[family]=dict(cell=cell,run=str(run),source_file_sha256=files,
                           original_build_sha256=identity['build_sha256'],original_final_sha256=result['bake_sha256'])
    record=dict(schema='corrected-runtime-control-compatibility-v1',approval_path=str(approval),approval_sha256=sha(approval),
        plan_sha256=sha(study/'PLAN.json'),build_sha256=BUILD_SHA256,cases=cases,
        epochs=240,pairs_per_epoch=50000,scientific_selection=False,outer_results_read=False,
        driver_sha256=sha(__file__),
        required_comparison='all 240 finalized checkpoints; remove only provenance and research metadata')
    output.mkdir(parents=True,exist_ok=False);(output/'REGISTRATION.json').write_bytes(encoded(record));return record


def command_for(original,binaries,output):
    from shaping_rescue_execution import portable_command
    command=portable_command(original['command'],Path(original['_source']))
    if command.count('--epochs')!=1 or command[command.index('--epochs')+1]!='240':raise ValueError('full control exposure required')
    if command.count('--pairs-per-epoch')!=1 or command[command.index('--pairs-per-epoch')+1]!='50000':raise ValueError('control draw budget differs')
    native=[i for i,v in enumerate(command) if Path(v).name=='zensim_mlp_train']
    if len(native)!=1:raise ValueError('one native trainer required')
    command[native[0]]=str(binaries/'zensim_mlp_train')
    for flag,value in [('--mem','2G'),('--out',output/'best.bin'),('--dump-checkpoints-dir',output/'checkpoints')]:
        if command.count(flag)!=1:raise ValueError('missing control command field: '+flag)
        command[command.index(flag)+1]=str(value)
    return command


def run(args):
    from shaping_reduced_scope import admit_compute
    from shaping_rescue_dispatch import reserve_worker,bind_worker,release_worker
    from shaping_storage import reservation
    admit_compute(args.study);record=json.loads(args.registration.read_text())
    if (record['schema']!='corrected-runtime-control-compatibility-v1' or record['outer_results_read']
        or record['plan_sha256']!=sha(args.study/'PLAN.json') or record['build_sha256']!=sha(args.binary_dir/'BUILD.json')
        or record['build_sha256']!=BUILD_SHA256 or record['driver_sha256']!=sha(__file__)
        or sha(Path(record['approval_path']))!=record['approval_sha256']):
        raise ValueError('control compatibility registration differs')
    case=record['cases'][args.family];source=Path(case['run']);key=case['cell']['id']
    if args.output.exists():raise ValueError('preserve existing compatibility attempt; never replay')
    for path,pin in case['source_file_sha256'].items():
        if sha(path)!=pin:raise ValueError('control source evidence drift')
    original=json.loads((source/'RUN.json').read_text())
    if sha(source/'checkpoints/ckpt_epoch239.bin')!=case['original_final_sha256']:raise ValueError('control final model drift')
    expected={f'ckpt_epoch{e:03}.bin' for e in range(240)}
    if {p.name for p in (source/'checkpoints').glob('ckpt_epoch*.bin')}!=expected:
        raise ValueError('restore every original checkpoint before compatibility fitting')
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('compatibility binary drift')
    while not reserve_worker(args.study,key)['can_start']:time.sleep(10)
    try:
        minimum=json.loads((args.study/'PLAN.json').read_text())['minimum_ssd_free_bytes']
        with reservation(args.study,args.output,minimum):
            args.output.mkdir(parents=True);(args.output/'checkpoints').mkdir()
            command=command_for(dict(original,_source=str(source)),args.binary_dir,args.output)
            env=os.environ.copy()
            for name in list(env):
                if name.startswith(('ZENSIM_MONOTONE_RESEARCH_','ZENSIM_INPUT_SHAPING_','ZENSIM_PLAIN_OPTIMIZER_','ZENSIM_TRAIN_RESUME')):env.pop(name,None)
            for name,prefix in [('HEAD.json','ZENSIM_MONOTONE_RESEARCH'),('SHAPING.json','ZENSIM_INPUT_SHAPING')]:
                path=source/name
                if path.exists():env[prefix+'_MANIFEST']=str(path);env[prefix+'_SHA256']=sha(path)
            env.update(TMPDIR=str(Path.home()/'tmp'),OMP_NUM_THREADS='1',RAYON_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                       ZENSIM_TRAIN_SNAPSHOT_ROOT=str(args.study.parent/'shared-training-snapshots'))
            result=dict(cell=case['cell'],source_run=str(source),source_result_sha256=sha(source/'RESULT.json'),
                registration_sha256=sha(args.registration),build_sha256=BUILD_SHA256,command=command,execution_host=socket.gethostname(),
                scientific_selection=False,outer_results_read=False,epochs=240,pairs_per_epoch=50000,stage='running')
            with (args.output/'train.log').open('x') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env);bind_worker(args.study,key,process.pid)
                result['pid']=process.pid;(args.output/'RUN.json').write_bytes(encoded(result));code=process.wait()
            result.update(exit_code=code,stage='failed' if code else 'trained')
            (args.output/'TRAINING_RESULT.json').write_bytes(encoded(result))
            if code:raise RuntimeError('compatibility control failed; preserve all evidence')
            keys=('zentrain.repro','zentrain.monotone_research') if args.family in ('softplus','silu_monotone','silu_diagnostic') else ('zentrain.repro',)
            comparison=compare_checkpoints(source/'checkpoints',args.output/'checkpoints',args.binary_dir/'bake_dial_refit',args.output/'trajectory',240,keys)
            result.update(stage='qualified' if comparison['all_equal'] else 'trajectory_mismatch',trajectory=comparison,
                          driver_sha256=sha(__file__),all_240_checkpoints_equal=comparison['all_equal'])
            (args.output/'RESULT.json').write_bytes(encoded(result))
            if not comparison['all_equal']:raise ValueError('compatibility trajectory differs; control reuse blocked')
            return result
    finally:release_worker(args.study,key)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','run'))
    for name in ('study','approval','binary-dir','output','registration'):p.add_argument('--'+name,type=Path)
    p.add_argument('--family',choices=FAMILIES);a=p.parse_args()
    result=register(a.study,a.approval,a.binary_dir,a.output) if a.mode=='register' else run(a)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','trajectory')}),flush=True)
