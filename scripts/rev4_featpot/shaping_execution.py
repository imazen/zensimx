#!/usr/bin/env python3
"""Execute registered completion-study work through existing admission/training owners."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

# Each independent trainer gets a bounded single-thread lane. Concurrency is
# controlled by the caller's aggregate budget, not a global compute lock.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
os.environ['TMPDIR']=str(Path.home()/'tmp')
from shaping_completion import REPO, SEEDS, SETTINGS, encoded, sha


def load_registration(root):
    plan=json.loads((root/'PLAN.json').read_text())
    for name,pin in [('MATRIX.json','matrix_sha256'),('PARTITIONS.json','partition_sha256')]:
        if sha(root/name)!=plan[pin]:raise ValueError(f'registration drift: {name}')
    if sha(plan['protocol'])!=plan['protocol_sha256']:raise ValueError('protocol drift')
    return plan,json.loads((root/'MATRIX.json').read_text()),json.loads((root/'PARTITIONS.json').read_text())


def prepare_partition(root, partition, membership, plan):
    """Reuse the existing strict source-subset admission owner once per partition."""
    import numpy as np
    from shaped_linear_compare import prepare_neural
    from shaped_linear_study import OLD
    dest=root/'comparators'/partition
    receipt=dest/'GROUPS.json'
    if receipt.exists():
        record=json.loads(receipt.read_text())
        if record['membership_sha256']!=__import__('hashlib').sha256(encoded(membership)).hexdigest():
            raise ValueError('partition membership drift')
        for path,pin in record['file_sha256'].items():
            if sha(path)!=pin:raise ValueError(f'partition file drift: {path}')
        return record
    if dest.exists():raise ValueError(f'incomplete partition preserved: {dest}')
    data={'y':np.load(OLD/'data/y.npy',mmap_mode='r')}
    cell=dict(id=partition,train=membership['train'],test=membership['test'],mode='true',seed=0,
              outer=int(partition.split('-')[1]))
    groups,decision,_=prepare_neural(root,cell,data,plan)
    # Exposed development is never passed to this study's trainer. Selection
    # uses the externally evaluated inner rows and fixed final checkpoints.
    groups=[(name,str(path),tw,0.,mode) for name,path,tw,_,mode in groups if tw>0]
    files=[Path(decision),dest/'ADMISSION.json']
    for _,path,_,_,_ in groups:
        path=Path(path);files.extend((path,path.with_suffix('.keys.parquet'),Path(str(path)+'.manifest.json')))
    files.append(dest/'admission/wide/main/real/receipt.json')
    record=dict(partition=partition,groups=groups,decision=str(decision),
                membership_sha256=__import__('hashlib').sha256(encoded(membership)).hexdigest(),
                file_sha256={str(p):sha(p) for p in files},exposed_development_passed_to_trainer=False)
    receipt.write_bytes(encoded(record))
    return record


def admit(args):
    plan,cells,partitions=load_registration(args.study)
    binary=json.loads((args.binary_dir/'BUILD.json').read_text())
    for name,pin in binary['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('binary drift: '+name)
    for path,pin in plan['input_pins'].items():
        if sha(path)!=pin:raise ValueError('frozen input drift: '+path)
    return plan,cells,partitions


def run_native(root,binary_dir,cell,record,attempt,epochs,mem):
    if attempt<1:raise ValueError('attempt must be positive')
    dest=root/'runs'/cell['id']/f'attempt-{attempt:03}'
    if dest.exists():raise ValueError('destination exists; inspect its actual process/artifacts')
    dest.mkdir(parents=True)
    from shaped_linear_study import BASE,HEAVY
    import v2_common
    import v2_lodo_mlp as owner
    # The caller establishes these shared module settings before concurrency.
    if owner.EPOCHS!=epochs:raise ValueError('command builder epoch budget drift')
    recipe={**json.loads(BASE.read_text())['recipe'],'shared_hidden_activation':'relu','mse_weight':1}
    declaration=json.loads((REPO/'scripts/rev4_featpot/color_noise_fx2_declaration.json').read_text())
    keep=dest/'keep.txt';keep.write_text('\n'.join(map(str,declaration['direct']))+'\n')
    groups=[(n,Path(p),tw,vw,m) for n,p,tw,vw,m in record['groups']]
    argv=owner.train_command(groups,*SEEDS[cell['seed']],1853,keep,'N',dest/'best.bin',recipe,
                             strict_admission=True,data_role_decision=Path(record['decision']))
    if Path(argv[0]).resolve()!=(binary_dir/'zensim_mlp_train').resolve():
        raise ValueError('command builder did not select admitted binary')
    ckpt=dest/'checkpoints';ckpt.mkdir()
    lr,l2=SETTINGS[cell['setting']]
    argv+=['--lr',str(lr),'--l2',str(l2),'--no-sample-coverage',
           '--dump-checkpoints-every','1','--dump-checkpoints-dir',str(ckpt)]
    env=os.environ.copy()
    for key in ('ZENSIM_MONOTONE_RESEARCH_MANIFEST','ZENSIM_MONOTONE_RESEARCH_SHA256',
                'ZENSIM_INPUT_SHAPING_MANIFEST','ZENSIM_INPUT_SHAPING_SHA256'):
        env.pop(key,None)
    command=[str(HEAVY),'--mem',mem,'--jobs','1','--',*map(str,argv)]
    with (dest/'train.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        running=dict(cell=cell,command=command,pid=process.pid,started_utc=datetime.now(timezone.utc).isoformat(),
                     plan_sha256=sha(root/'PLAN.json'),build_sha256=sha(binary_dir/'BUILD.json'),
                     driver_sha256=sha(__file__),stage='running',epochs=epochs,pairs_per_epoch=50000,
                     scientific_selection=cell['stage']!='resource_pilot')
        (dest/'RUN.json').write_bytes(encoded(running))
        print(json.dumps({'run':str(dest),'pid':process.pid,'cell':cell['id']}),flush=True)
        rc=process.wait()
    final={**running,'stage':'trained' if rc==0 else 'failed','exit_code':rc,
           'finished_utc':datetime.now(timezone.utc).isoformat(),
           'artifact_bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())}
    if rc==0:
        bake=ckpt/f'ckpt_epoch{epochs-1:03}.bin'
        if not bake.is_file():raise ValueError('successful trainer omitted final native checkpoint')
        if len(list(ckpt.glob('ckpt_epoch*.bin')))!=epochs:raise ValueError('checkpoint inventory differs')
        final['bake_sha256']=sha(bake)
        if cell['stage']=='resource_pilot':final['stage']='complete'
    (dest/'RESULT.json').write_bytes(encoded(final))
    if rc:raise RuntimeError(f'training failed; preserve and inspect {dest}/train.log')
    print(json.dumps({'cell':cell['id'],'stage':final['stage'],'artifact_bytes':final['artifact_bytes']}),flush=True)
    return final


def execute(args):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    plan,cells,partitions=admit(args)
    if args.resource_pilot:
        selected=[c for c in cells if c['stage']=='resource_pilot'];epochs=1;workers=1;mem='4G'
    else:
        selected=[c for c in cells if c['stage']=='inner' and c['family']=='relu' and c['partition']==args.partition]
        epochs=240;workers=args.workers;mem='1G'
        if not selected or not 1<=workers<=16:raise ValueError('invalid partition or worker count')
    # Native pilot measured 606443 total artifact bytes, including two bakes.
    # Reserve a deliberately conservative fixed 1 GiB per admitted long run;
    # this is an admission allowance, not an extrapolated measured size.
    if shutil.disk_usage(args.study).free < plan['minimum_ssd_free_bytes']+(workers+2)*1024**3:
        raise ValueError('SSD reserve does not admit this wave')
    records={p:prepare_partition(args.study,p,partitions[p],plan) for p in {c['partition'] for c in selected}}
    os.environ['REV4_V2_BIN_DIR']=str(args.binary_dir)
    import v2_common
    import v2_lodo_mlp as owner
    v2_common.TRAINER=owner.TRAINER=args.binary_dir/'zensim_mlp_train'
    v2_common.FITBIN=owner.FITBIN=args.binary_dir/'bake_dial_refit'
    owner.EPOCHS=epochs;owner.PAIRS_PER_EPOCH=50000;owner.LOG_EVERY=1
    status=dict(required=len(selected),trained=0,failed=[],scientific_reports_completed=0,
                controller_pid=os.getpid(),stage='running',cells=[c['id'] for c in selected])
    status_path=args.study/f"{('pilot' if args.resource_pilot else args.partition)}-STATUS.json"
    status_path.write_bytes(encoded(status))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(run_native,args.study,args.binary_dir,c,records[c['partition']],args.attempt,epochs,mem):c for c in selected}
        for future in as_completed(pending):
            try:future.result();status['trained']+=1
            except Exception as error:status['failed'].append(dict(cell=pending[future]['id'],error=str(error)))
            status['checked_utc']=datetime.now(timezone.utc).isoformat();status_path.write_bytes(encoded(status))
    status['stage']='failed' if status['failed'] else 'training_finished';status_path.write_bytes(encoded(status))
    if status['failed']:raise RuntimeError('failed attempts preserved; see '+str(status_path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--binary-dir',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--resource-pilot',action='store_true')
    mode.add_argument('--inner-relu',action='store_true')
    p.add_argument('--partition',default='inner-0-0')
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--attempt',type=int,default=1)
    execute(p.parse_args())


if __name__=='__main__':main()
