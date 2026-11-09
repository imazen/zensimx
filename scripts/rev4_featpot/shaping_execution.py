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


DRIVER_SHA256=sha(__file__)

def load_registration(root,*,original=False):
    plan=json.loads((root/'PLAN.json').read_text())
    for name,pin in [('MATRIX.json','matrix_sha256'),('PARTITIONS.json','partition_sha256')]:
        if sha(root/name)!=plan[pin]:raise ValueError(f'registration drift: {name}')
    # A worker capsule may carry the exact frozen document beside PLAN.json.
    # Its registered content hash, never a host-specific path, owns admission.
    protocol=root/'PROTOCOL.md'
    if not protocol.exists():
        protocol=Path(plan['protocol'])
        if protocol.name=='SHAPING_COMPLETION_PROTOCOL.md' and protocol.parent.name=='docs':
            protocol=REPO/'docs'/protocol.name
    if sha(protocol)!=plan['protocol_sha256']:raise ValueError('protocol drift')
    cells=json.loads((root/'MATRIX.json').read_text())
    if not original and (root/'corrected_thresholds/ACTIVATION.json').exists():
        from shaping_corrected_view import activated_view
        cells=activated_view(root,cells)
    return plan,cells,json.loads((root/'PARTITIONS.json').read_text())


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
              outer=None if partition=='full-fit' else int(partition.split('-')[1]))
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
    from shaping_storage import reservation
    minimum=json.loads((root/'PLAN.json').read_text())['minimum_ssd_free_bytes']
    dest=root/'runs'/cell['id']/f'attempt-{attempt:03}'
    with reservation(root,dest,minimum):
        return _run_native(root,binary_dir,cell,record,attempt,epochs,mem)


def _run_native(root,binary_dir,cell,record,attempt,epochs,mem):
    from shaping_host_assignment import check
    check(root,[cell],sha(binary_dir/'BUILD.json'))
    from shaping_corrected_view import admit_fit
    corrected=admit_fit(root,cell,sha(binary_dir/'BUILD.json'),epochs)
    from shaping_reduced_scope import admit_cell
    admit_cell(root,cell)
    if attempt<1:raise ValueError('attempt must be positive')
    dest=root/'runs'/cell['id']/f'attempt-{attempt:03}'
    if dest.exists():raise ValueError('destination exists; inspect its actual process/artifacts')
    dest.mkdir(parents=True)
    from shaped_linear_study import BASE,HEAVY
    import v2_common
    import v2_lodo_mlp as owner
    # The caller establishes these shared module settings before concurrency.
    if owner.EPOCHS!=epochs:raise ValueError('command builder epoch budget drift')
    recipe={**json.loads(BASE.read_text())['recipe'],'shared_hidden_activation':'relu','mse_weight':record.get('mse_weight',1)}
    analytic_family=cell['family'] in ('softplus','silu_monotone','silu_diagnostic')
    if cell['family']=='silu_diagnostic':recipe['max_pairs']=0
    declaration=json.loads((REPO/'scripts/rev4_featpot/color_noise_fx2_declaration.json').read_text())
    keep=dest/'keep.txt';keep.write_text('\n'.join(map(str,declaration['direct']))+'\n')
    groups=[(n,Path(p),tw,vw,m) for n,p,tw,vw,m in record['groups']]
    seeds=SEEDS[cell['seed']]
    if '_pretrained_' in cell['family'] or cell['stage'].startswith('fresh_control'):
        seeds=tuple(seed+1000000 for seed in seeds)
    argv=owner.train_command(groups,*seeds,1853,keep,'N',dest/'best.bin',recipe,
                             strict_admission=True,data_role_decision=Path(record['decision']))
    if Path(argv[0]).resolve()!=(binary_dir/'zensim_mlp_train').resolve():
        raise ValueError('command builder did not select admitted binary')
    ckpt=dest/'checkpoints';ckpt.mkdir()
    lr,l2=SETTINGS[cell['setting']]
    argv+=['--lr',str(lr),'--l2',str(l2),'--no-sample-coverage',
           '--dump-checkpoints-every','1','--dump-checkpoints-dir',str(ckpt)]
    env=os.environ.copy()
    if corrected is not None:
        for key in list(env):
            if key.startswith(('ZENSIM_TRAIN_RESUME','ZENSIM_TRAIN_SNAPSHOT','ZENSIM_PLAIN_OPTIMIZER_')):env.pop(key,None)
        env['ZENSIM_TRAIN_SNAPSHOT_ROOT']=str(root.parent/'shared-training-snapshots')
    for key in ('ZENSIM_MONOTONE_RESEARCH_MANIFEST','ZENSIM_MONOTONE_RESEARCH_SHA256',
                'ZENSIM_INPUT_SHAPING_MANIFEST','ZENSIM_INPUT_SHAPING_SHA256'):
        env.pop(key,None)
    if cell['family'] in ('product_joint_identity','separate_joint_identity'):
        kind='separate' if cell['family'].startswith('separate') else 'product'
        initial=root/'shapes'/cell['partition']/kind/'INITIAL.json'
        receipt=json.loads((initial.parent/'RECEIPT.json').read_text())
        if sha(initial)!=receipt['manifest_sha256']:raise ValueError('initial shape drift')
        path=dest/'SHAPING.json';shutil.copy2(initial,path)
        env['ZENSIM_INPUT_SHAPING_MANIFEST']=str(path)
        env['ZENSIM_INPUT_SHAPING_SHA256']=sha(path)
    if '_pretrained_' in cell['family']:
        from shaping_pretraining import extract
        _,cells,_=load_registration(root)
        path=dest/'SHAPING.json'
        extract(root,binary_dir,cell,cells,path)
        env['ZENSIM_INPUT_SHAPING_MANIFEST']=str(path)
        env['ZENSIM_INPUT_SHAPING_SHA256']=sha(path)
    if analytic_family:
        tau,learn,bias=((.25,False,0.),(1.,False,0.),(1.,True,.25),(4.,True,1.))[cell['setting']]
        if cell['family']=='softplus' and cell['setting']==3:bias=-1.
        manifest=dict(schema='monotone-head-v1',family=cell['family'],widths=[64],
                      init_seed=SEEDS[cell['seed']][0],sample_seed=SEEDS[cell['seed']][1],
                      epochs=epochs,pairs_per_epoch=50000,first_width=128,max_pairs=recipe['max_pairs'],
                      analytic=dict(temperature=tau,learn_temperature=learn,bias=bias))
        if cell['family']=='silu_diagnostic':manifest['unrestricted_first']=True
        path=dest/'HEAD.json';path.write_bytes(encoded(manifest))
        env['ZENSIM_MONOTONE_RESEARCH_MANIFEST']=str(path)
        env['ZENSIM_MONOTONE_RESEARCH_SHA256']=sha(path)
    command=[str(HEAVY),'--mem',mem,'--jobs','1','--',*map(str,argv)]
    with (dest/'train.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        running=dict(cell=cell,command=command,pid=process.pid,started_utc=datetime.now(timezone.utc).isoformat(),
                     execution_host=__import__('socket').gethostname(),
                     plan_sha256=sha(root/'PLAN.json'),build_sha256=sha(binary_dir/'BUILD.json'),
                     driver_sha256=DRIVER_SHA256,stage='running',epochs=epochs,pairs_per_epoch=50000,
                     scientific_selection=cell['stage']!='resource_pilot')
        if corrected is not None:running['corrected_threshold_condition']=corrected
        (dest/'RUN.json').write_bytes(encoded(running))
        print(json.dumps({'run':str(dest),'pid':process.pid,'cell':cell['id']}),flush=True)
        if corrected is not None:
            from shaping_snapshot_retention import wait_and_bind
            rc=wait_and_bind(process,root.parent,dest)
        else:rc=process.wait()
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
        if args.family=='analytic':families=('softplus','silu_monotone','silu_diagnostic')
        elif args.family=='joint_shapes':families=('product_joint_identity','separate_joint_identity')
        elif args.family=='pretrained_shapes':families=('product_pretrained_frozen','product_pretrained_joint','separate_pretrained_frozen','separate_pretrained_joint')
        else:families=(args.family,)
        stage='fresh_control_inner' if args.fresh_control_inner else 'inner'
        selected=[c for c in cells if c['stage']==stage and c['family'] in families and c['partition']==args.partition]
        requested=getattr(args,'cell',None)
        if requested:
            selected=registered_subset(selected,requested)
        selected.sort(key=lambda c:(c['setting'],c['seed'],families.index(c['family'])))
        epochs=120 if stage=='fresh_control_inner' or all('_pretrained_' in c['family'] for c in selected) else 240
        workers=args.workers;mem='1G'
        if not selected or not 1<=workers<=16:raise ValueError('invalid partition or worker count')
        # A dispatcher may release individual children as their matching parents
        # finish. Never create a failed attempt merely because a parent is live.
        if any('_pretrained_' in c['family'] for c in selected):
            from shaping_pretraining import completed_parent
            for c in selected:completed_parent(args.study,c,cells)
    # Native pilot measured 606443 total artifact bytes, including two bakes.
    from shaping_host_assignment import check
    check(args.study,selected,sha(args.binary_dir/'BUILD.json'))
    # Reserve a deliberately conservative fixed 1 GiB per admitted long run;
    # this is an admission allowance, not an extrapolated measured size.
    if shutil.disk_usage(args.study).free < plan['minimum_ssd_free_bytes']+(workers+2)*1024**3:
        raise ValueError('SSD reserve does not admit this wave')
    records={p:prepare_partition(args.study,p,partitions[p],plan) for p in {c['partition'] for c in selected}}
    if not args.resource_pilot and any(c['family'].endswith('joint_identity') for c in selected):
        from shaping_inputs import prepare
        for partition in {c['partition'] for c in selected}:
            for separate in (False,True):prepare(args.study,partition,partitions[partition],plan,separate)
    os.environ['REV4_V2_BIN_DIR']=str(args.binary_dir)
    import v2_common
    import v2_lodo_mlp as owner
    v2_common.TRAINER=owner.TRAINER=args.binary_dir/'zensim_mlp_train'
    v2_common.FITBIN=owner.FITBIN=args.binary_dir/'bake_dial_refit'
    owner.EPOCHS=epochs;owner.PAIRS_PER_EPOCH=50000;owner.LOG_EVERY=1
    status=dict(required=len(selected),trained=0,failed=[],scientific_reports_completed=0,
                controller_pid=os.getpid(),stage='running',cells=[c['id'] for c in selected])
    status_path=args.study/f"{('pilot' if args.resource_pilot else args.partition+'-'+args.family+('-fresh' if args.fresh_control_inner else ''))}-STATUS.json"
    if getattr(args,'cell',None):
        suffix=__import__('hashlib').sha256(encoded(sorted(args.cell))).hexdigest()[:16]
        status_path=status_path.with_name(status_path.stem+'-'+suffix+status_path.suffix)
    status_path.write_bytes(encoded(status))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(run_native,args.study,args.binary_dir,c,records[c['partition']],args.attempt,epochs,mem):c for c in selected}
        for future in as_completed(pending):
            try:future.result();status['trained']+=1
            except Exception as error:status['failed'].append(dict(cell=pending[future]['id'],error=str(error)))
            status['checked_utc']=datetime.now(timezone.utc).isoformat();status_path.write_bytes(encoded(status))
    status['stage']='failed' if status['failed'] else 'training_finished';status_path.write_bytes(encoded(status))
    if status['failed']:raise RuntimeError('failed attempts preserved; see '+str(status_path))


def registered_subset(selected,requested):
    if len(set(requested))!=len(requested):raise ValueError('duplicate requested cell')
    if not set(requested)<=set(c['id'] for c in selected):raise ValueError('cell outside registered stage/family/partition')
    return [c for c in selected if c['id'] in requested]


def transported_control_live(root,run,directory,argv):
    """Bind the control CLI to its exact request rather than a module name."""
    scripts=[i for i,v in enumerate(argv) if Path(v).name=='shaping_control_worker.py']
    if len(scripts)!=1 or argv[scripts[0]+1:scripts[0]+2]!=['fit'] or argv.count('--request')!=1:return False
    try:
        request=Path(argv[argv.index('--request')+1]);record=json.loads(request.read_text())
        status=json.loads(request.with_name('STATUS.json').read_text())
        return (status.get('request_sha256')==sha(request)
            and record.get('execution_kind')=='control' and record.get('study')==str(root)
            and record.get('cell')==run['cell'] and record.get('result_output')==str(directory)
            and record.get('host')==run.get('execution_host')
            and record.get('plan_sha256')==run['plan_sha256'] and record.get('build_sha256')==run['build_sha256'])
    except (OSError,ValueError,IndexError,KeyError):return False


def live_status(root,*,persist=True,emit=True):
    import re
    rows=[]
    for path in sorted(root.glob('runs/*/attempt-*/RUN.json')):
        run=json.loads(path.read_text());pid=run['pid']
        try:
            argv=[v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]
            command=' '.join(argv)
            if run.get('process_role')=='control_worker':
                live=(str(root) in command and 'shaping_controls.py' in command) or transported_control_live(root,run,path.parent,argv)
            elif run.get('process_role')=='rescue_controller':live=str(root) in command and 'shaping_rescue_execution.py' in command and run['cell']['id'] in command
            elif run.get('process_role')=='extension_controller':live=str(root) in command and 'shaping_extension_execution.py' in command and run['cell']['id'] in command
            elif run.get('process_role')=='outer_controller':live=str(root) in command and 'shaping_outer_execution.py' in command and run['cell']['id'] in command
            elif run.get('process_role')=='full_fit_controller':live=str(root) in command and 'shaping_full_fit_execution.py' in command and run['cell']['id'] in command
            elif run.get('process_role')=='variant_controller':live=str(root) in command and 'shaping_variant_execution.py' in command and run['cell']['id'] in command
            else:live=str(path.parent) in command and 'run-heavy' in command
        except FileNotFoundError:live=False
        log=path.parent/run.get('fit_directory','.')/'train.log'
        text=log.read_text() if log.exists() else ''
        epochs=re.findall(r'epoch\s+(\d+)\s+\|',text)
        result=path.parent/'RESULT.json'
        terminal=json.loads(result.read_text())['stage'] if result.exists() else None
        if terminal is not None:live=False
        rows.append(dict(cell=run['cell']['id'],stage=run['cell']['stage'],pid=pid,
                         verified_live=live,terminal=terminal,last_epoch=int(epochs[-1]) if epochs else None,
                         required_epochs=run.get('epochs',run['cell'].get('epochs')),run=str(path.parent)))
    out=dict(checked_utc=datetime.now(timezone.utc).isoformat(),runs=rows,
             ssd_free_bytes=shutil.disk_usage(root).free,
             live=sum(r['verified_live'] for r in rows),
             unexpectedly_stopped=[r['cell'] for r in rows if not r['verified_live'] and r['terminal'] is None])
    if persist:
        temporary=root/'LIVE_STATUS.json.writing';temporary.write_bytes(encoded(out));temporary.replace(root/'LIVE_STATUS.json')
    if emit:print(json.dumps(out),flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--binary-dir',type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--resource-pilot',action='store_true')
    mode.add_argument('--inner-relu',action='store_true')
    mode.add_argument('--inner-family',action='store_true')
    mode.add_argument('--fresh-control-inner',action='store_true')
    p.add_argument('--family',choices=['relu','softplus','silu_monotone','silu_diagnostic','analytic','product_joint_identity','separate_joint_identity','joint_shapes','pretrained_shapes','product_pretrained_frozen','product_pretrained_joint','separate_pretrained_frozen','separate_pretrained_joint'],default='relu')
    p.add_argument('--cell',action='append',help='dispatch only these registered IDs; does not change study requirements')
    mode.add_argument('--status',action='store_true')
    p.add_argument('--partition',default='inner-0-0')
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--attempt',type=int,default=1)
    args=p.parse_args()
    if args.status:live_status(args.study)
    else:
        if args.binary_dir is None:p.error('--binary-dir is required for execution')
        execute(args)


if __name__=='__main__':main()
