"""Two short TRAIN-only probes of the corrected shaped-threshold coordinates."""
from __future__ import annotations
import argparse,json,os,re,socket,subprocess,time
from pathlib import Path
from shaping_completion import DECLARATION_SHA256,encoded,sha

CELLS=('2760a078c4eed657096f','e70a807c32a3d34b505f')
TRAINER='58fa6317f2fc192a5e840f449f79b4fddc205d7a92d2965ea83ad5201f46a0fd'


def command_for(original,binary,destination):
    if (original['cell']['id'] not in CELLS or original['cell']['partition']!='inner-0-0'
        or original['cell']['family']!='separate_pretrained_frozen' or original['cell']['setting']!=3):
        raise ValueError('only the two declared TRAIN diagnostic conditions')
    command=list(original['command']);command[command.index('--')+1]=str(binary)
    for flag,value in [('--epochs','2'),('--out',destination/'best.bin'),
                       ('--dump-checkpoints-dir',destination/'checkpoints')]:
        if command.count(flag)!=1:raise ValueError('ambiguous frozen command')
        command[command.index(flag)+1]=str(value)
    if command[command.index('--pairs-per-epoch')+1]!='50000':raise ValueError('pair budget drift')
    return command


def register(study,binaries,output):
    if sha(binaries/'zensim_mlp_train')!=TRAINER:raise ValueError('qualified corrected trainer required')
    groups=study/'comparators/inner-0-0/GROUPS.json';group_record=json.loads(groups.read_text())
    if any(g[2]<=0 or g[3]!=0 for g in group_record['groups']):raise ValueError('TRAIN groups only')
    runs={}
    for cell in CELLS:
        old=study/'runs'/cell/'attempt-001';original=json.loads((old/'RUN.json').read_text())
        if json.loads((old/'RESULT.json').read_text())['stage']!='trained':raise ValueError('completed baseline required')
        command_for(original,binaries/'zensim_mlp_train',output/cell)
        specifications=[original['command'][i+1] for i,v in enumerate(original['command']) if v=='--group']
        if specifications!=[':'.join(map(str,g)) for g in group_record['groups']]:raise ValueError('original TRAIN groups differ')
        runs[cell]=dict(original_sha256={name:sha(old/name) for name in ('RUN.json','RESULT.json','SHAPING.json','keep.txt','train.log')},
            checkpoint_sha256={name:sha(old/'checkpoints'/name) for name in ('ckpt_epoch000.bin','ckpt_epoch001.bin')})
    record=dict(schema='shaping-threshold-coordinate-probe-v1',cells=runs,plan_sha256=sha(study/'PLAN.json'),
        groups_sha256=sha(groups),training_input_sha256=group_record['file_sha256'],build_sha256=sha(binaries/'BUILD.json'),
        trainer_sha256=TRAINER,driver_sha256=sha(__file__),epochs=2,pairs_per_epoch=50000,
        original_fits_replayed=False,scientific_selection=False,outer_results_read=False,
        purpose='Isolate corrected threshold coordinates on one previously collapsed and one surviving seed. '
                'Keep each original frozen bank, inputs and optimizer settings. Reuse old epoch 0/1 artifacts; never rerun the old source.',
        limitations='Purposive two-seed optimization diagnostic; no generalization or architecture ranking. '
                    'New executable includes qualified arithmetic-preserving training optimizations and full-state snapshots.',
        diagnostic='Compare per-epoch loss and nonzero output weights from native inspect. No quality-based checkpoint selection.',
        storage=dict(per_job_reservation_bytes=1024**3,shared_snapshot_cap_bytes=4*1024**3,minimum_free_bytes=30*1024**3))
    output.mkdir(exist_ok=False)
    (output/'REGISTRATION.json').write_bytes(encoded(record));return record


def inspect(binary,checkpoint):
    result=json.loads(subprocess.check_output([str(binary),'inspect','--in',str(checkpoint)],text=True))
    layer=result['layers'][-1]
    return dict(checkpoint_sha256=sha(checkpoint),active_output_coefficients=layer['stored_weights']-layer['zero_weights'])


def run(study,binaries,output,cell):
    from shaping_rescue_dispatch import reserve_worker,release_worker,bind_worker
    from shaping_storage import reservation
    if socket.gethostname()!='i134':raise ValueError('approved worker required')
    record=json.loads((output/'REGISTRATION.json').read_text())
    if (record['driver_sha256']!=sha(__file__) or record['plan_sha256']!=sha(study/'PLAN.json')
        or record['build_sha256']!=sha(binaries/'BUILD.json') or sha(binaries/'zensim_mlp_train')!=TRAINER
        or cell not in CELLS):raise ValueError('probe registration drift')
    for name,pin in json.loads((binaries/'BUILD.json').read_text())['binary_sha256'].items():
        if sha(binaries/name)!=pin:raise ValueError('qualified binary drift')
    old=study/'runs'/cell/'attempt-001';destination=output/cell
    for name,pin in record['cells'][cell]['original_sha256'].items():
        if sha(old/name)!=pin:raise ValueError('original evidence drift')
    for name,pin in record['training_input_sha256'].items():
        if sha(name)!=pin:raise ValueError('training input drift')
    original=json.loads((old/'RUN.json').read_text());command=command_for(original,binaries/'zensim_mlp_train',destination)
    declaration=Path(command[command.index('--derived-inputs')+1])
    if sha(declaration)!=DECLARATION_SHA256:raise ValueError('derived feature declaration drift')
    key='threshold-probe-'+cell
    while not reserve_worker(study,key)['can_start']:time.sleep(10)
    try:
        with reservation(study,destination,record['storage']['minimum_free_bytes']):
            destination.mkdir(exist_ok=False)
            env=os.environ.copy()
            for name in ('ZENSIM_MONOTONE_RESEARCH_MANIFEST','ZENSIM_MONOTONE_RESEARCH_SHA256',
                         'ZENSIM_TRAIN_RESUME','ZENSIM_TRAIN_RESUME_CHECKPOINT_ORIGIN'):env.pop(name,None)
            env.update(ZENSIM_INPUT_SHAPING_MANIFEST=str(old/'SHAPING.json'),ZENSIM_INPUT_SHAPING_SHA256=sha(old/'SHAPING.json'),
                       ZENSIM_TRAIN_SNAPSHOT_ROOT=str(study.parent/'shared-training-snapshots'),RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
            identity=dict(command=command,registration_sha256=sha(output/'REGISTRATION.json'),driver_sha256=sha(__file__),
                          snapshots=env['ZENSIM_TRAIN_SNAPSHOT_ROOT'],bank_sha256=sha(old/'SHAPING.json'))
            with (destination/'train.log').open('x') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                identity['pid']=process.pid;(destination/'RUN.json').write_bytes(encoded(identity));bind_worker(study,key,process.pid)
                code=process.wait()
            result=dict(identity,exit_code=code,scientific_selection=False,outer_results_read=False)
            if code==0:
                result['epochs']=[]
                for epoch in range(2):
                    name=f'ckpt_epoch{epoch:03}.bin'
                    if sha(old/'checkpoints'/name)!=record['cells'][cell]['checkpoint_sha256'][name]:raise ValueError('baseline checkpoint drift')
                    result['epochs'].append(dict(epoch=epoch,baseline=inspect(binaries/'bake_dial_refit',old/'checkpoints'/name),
                                                 corrected=inspect(binaries/'bake_dial_refit',destination/'checkpoints'/name)))
                result['epoch_logs']={phase:[line for line in path.read_text().splitlines() if re.search(r'epoch\s+[01]\s+\|',line)]
                    for phase,path in [('baseline',old/'train.log'),('corrected',destination/'train.log')]}
            (destination/'RESULT.json').write_bytes(encoded(result))
            if code:raise RuntimeError('native probe failed; preserve its artifacts')
            return result
    finally:release_worker(study,key)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','run'))
    for name in ('study','binaries','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cell',choices=CELLS);args=p.parse_args()
    if args.mode=='register':print(json.dumps(register(args.study,args.binaries,args.output)))
    else:
        if not args.cell:p.error('--cell required')
        print(json.dumps(run(args.study,args.binaries,args.output,args.cell)))
