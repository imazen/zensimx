"""Native neutral-control replay of a completed analytic prefix; never selection."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_execution import load_registration
from shaping_rescue import controls


def qualify(args):
    import pyarrow.parquet as pq
    _,cells,_=load_registration(args.study)
    original=json.loads((args.run/'RUN.json').read_text());result=json.loads((args.run/'RESULT.json').read_text())
    cell=original['cell']
    if cell not in cells or cell['stage']!='inner' or cell['family'] not in ('softplus','silu_monotone','silu_diagnostic'):
        raise ValueError('registered analytic inner fit required')
    if result['stage']!='trained' or result['exit_code']!=0 or original['plan_sha256']!=sha(args.study/'PLAN.json'):
        raise ValueError('completed frozen fit required')
    if not 1<=args.epochs<=original['epochs']:raise ValueError('invalid prefix exposure')
    final=args.run/'checkpoints'/f"ckpt_epoch{original['epochs']-1:03}.bin"
    if sha(final)!=result['bake_sha256']:raise ValueError('original final checkpoint drift')
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('qualification binary drift')
    groups=json.loads((args.study/'comparators'/cell['partition']/'GROUPS.json').read_text())
    for name,pin in groups['file_sha256'].items():
        if sha(name)!=pin:raise ValueError('admitted table drift')
    families=[]
    for name,path,weight,validation,mode in groups['groups']:
        if validation!=0 or weight<=0:raise ValueError('qualification must retain TRAIN-only groups')
        # Random order with balance none never uses these categories for draws or weights.
        # One category is an explicit neutral-control fixture, not a family analysis.
        families.append(dict(name=name,families=[0]*pq.read_metadata(path).num_rows))
    neutral=controls('init_half',True,groups=families)
    neutral.update(init_scale=1.,bias_scale=1.)
    head=json.loads((args.run/'HEAD.json').read_text());head['epochs']=args.epochs;head['training']=neutral
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'checkpoints').mkdir()
    manifest=args.output/'HEAD.json';manifest.write_bytes(encoded(head))
    from shaping_rescue_execution import portable_command
    cmd=portable_command(original['command'],args.run);cmd[cmd.index('--mem')+1]='2G'
    index=next(i for i,v in enumerate(cmd) if Path(v).name=='zensim_mlp_train');cmd[index]=str(args.binary_dir/'zensim_mlp_train')
    for flag,value in [('--epochs',args.epochs),('--out',args.output/'best.bin'),('--dump-checkpoints-dir',args.output/'checkpoints')]:
        cmd[cmd.index(flag)+1]=str(value)
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('ZENSIM_MONOTONE_RESEARCH_','ZENSIM_INPUT_SHAPING_','ZENSIM_PLAIN_OPTIMIZER_')):env.pop(key)
    env.update(ZENSIM_MONOTONE_RESEARCH_MANIFEST=str(manifest),ZENSIM_MONOTONE_RESEARCH_SHA256=sha(manifest),
               ZENSIM_OPTIMIZER_DIAGNOSTICS='1',TMPDIR=str(Path.home()/'tmp'))
    if getattr(args,'snapshot_root',None):
        env['ZENSIM_TRAIN_SNAPSHOT_ROOT']=str(args.snapshot_root)
    receipt=dict(original_run=str(args.run),original_result_sha256=sha(args.run/'RESULT.json'),command=cmd,
                 epochs=args.epochs,selection_candidate=False,build_sha256=sha(args.binary_dir/'BUILD.json'),
                 neutral_control=neutral,driver_sha256=sha(__file__))
    with (args.output/'train.log').open('w') as log:
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        receipt['pid']=process.pid;(args.output/'RUN.json').write_bytes(encoded(receipt));rc=process.wait()
    if rc:raise RuntimeError('neutral qualification failed; preserve log')
    hashes=[];epoch=args.epochs-1
    for label,path in [('original',args.run/'checkpoints'/f'ckpt_epoch{epoch:03}.bin'),('neutral',args.output/'checkpoints'/f'ckpt_epoch{epoch:03}.bin')]:
        for i,key in enumerate(('zentrain.repro','zentrain.monotone_research')):
            dest=args.output/f'{label}-{i}.bin'
            subprocess.run([str(args.binary_dir/'bake_dial_refit'),'strip','--in',str(path),'--out',str(dest),'--key',key],check=True)
            path=dest
        hashes.append(sha(path))
    receipt.update(canonical_sha256=hashes,match=hashes[0]==hashes[1])
    (args.output/'RESULT.json').write_bytes(encoded(receipt))
    if not receipt['match']:raise ValueError('neutral controls changed native weights or scaler')
    print(json.dumps(dict(match=True,epochs=args.epochs,canonical_sha256=hashes[0])),flush=True)


def qualify_on_worker(args):
    """Use the same CPU/SSD admission as scientific fits, with no fit retries."""
    from shaping_reduced_scope import admit_compute
    from shaping_rescue_dispatch import reserve_worker,release_worker,bind_worker
    from shaping_storage import reservation
    admit_compute(args.study)
    if args.output.exists():raise ValueError('preserve existing qualification attempt')
    # The native command retains this control ID in --keep-features. Shared
    # admission can therefore recognize its running trainer instead of counting
    # the qualification claim as another pending CPU in addition to that trainer.
    key=json.loads((args.run/'RUN.json').read_text())['cell']['id']
    if (len(key)!=20 or any(c not in '0123456789abcdef' for c in key)
        or args.run.parent.name!=key):raise ValueError('registered control directory required')
    if args.human_only_cell:
        key=__import__('hashlib').sha256(encoded(dict(source_cell=key,human_only_cell=args.human_only_cell,
            build_sha256=sha(args.binary_dir/'BUILD.json')))).hexdigest()[:20]
        if args.output.name!=key:raise ValueError('human qualification output must bind its unique admission identity')
    while True:
        if args.output.exists():raise ValueError('qualification appeared while waiting; preserve it')
        observation=reserve_worker(args.study,key)
        if observation['can_start']:break
        print(json.dumps(dict(stage='qualification_waiting_for_capacity',admission=observation)),flush=True)
        time.sleep(30)
    try:
        bind_worker(args.study,key,os.getpid())
        minimum=json.loads((args.study/'PLAN.json').read_text())['minimum_ssd_free_bytes']
        with reservation(args.study,args.output,minimum,bytes_required=1024**3):
            return qualify_human(args) if args.human_only_cell else qualify(args)
    finally:release_worker(args.study,key)


def human_command(command,groups,binary_dir,output):
    """Retain optimizer/feature arguments while replacing admitted TRAIN groups."""
    if len(groups)!=4 or {g[0] for g in groups}!={'human_kadid','human_tid2013','human_konfig','human_cid22_a25'}:
        raise ValueError('exact four human populations required')
    if any(weight<=0 or validation!=0 or mode!='withinref,rank' for _,_,weight,validation,mode in groups):
        raise ValueError('human-only qualification requires TRAIN-only ranking')
    cmd=[];i=0
    while i<len(command):
        if command[i]=='--group':i+=2;continue
        cmd.append(command[i]);i+=1
    index=next(i for i,v in enumerate(cmd) if Path(v).name=='zensim_mlp_train')
    cmd[index]=str(binary_dir/'zensim_mlp_train')
    for flag,value in [('--mem','2G'),('--epochs',1),('--mse-weight',0),('--out',output/'best.bin'),('--dump-checkpoints-dir',output/'checkpoints')]:
        if cmd.count(flag)!=1:raise ValueError('missing or duplicated command flag: '+flag)
        cmd[cmd.index(flag)+1]=str(value)
    for name,path,weight,validation,mode in groups:
        cmd.extend(['--group',f'{name}:{path}:{weight}:{validation}:{mode}'])
    return cmd


def human_context(args):
    from shaping_variant_inputs import load_variant,verify_prepared
    _,cell,identifier,view=load_variant(args.study,args.human_only_cell)
    if cell['stage']!='human_only' or view['teacher_training_rows']!=0:
        raise ValueError('registered human-only membership required')
    path=args.study/'comparators'/f'variant-{identifier[:20]}-true-0'/'GROUPS.json'
    record=verify_prepared(path,args.study,identifier,view,cell)
    original=json.loads((args.run/'RUN.json').read_text())
    head=json.loads((args.run/'HEAD.json').read_text())
    if head['family'] not in ('softplus','silu_monotone','silu_diagnostic'):
        raise ValueError('analytic head required')
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('qualification binary drift')
    return cell,view,path,record,original,head


def qualify_human(args):
    import pyarrow.parquet as pq
    cell,view,path,record,original,head=human_context(args)
    groups=[dict(name=n,families=[0]*pq.read_metadata(p).num_rows) for n,p,_,_,_ in record['groups']]
    neutral=controls('init_half',True,groups=groups,human_only=True)
    neutral.update(init_scale=1.,bias_scale=1.)
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema='human-only-native-neutral-qualification-v1',family=head['family'],
        source_run=str(args.run),source_run_sha256=sha(args.run/'RUN.json'),
        groups=str(path),groups_sha256=sha(path),membership_cell=cell,
        teacher_rows=0,training_rows=view['training_rows'],epochs=1,pairs_per_epoch=50000,
        scientific_selection=False,outer_results_read=False,build_sha256=sha(args.binary_dir/'BUILD.json'),
        driver_sha256=sha(__file__),arms={})
    for label,training in [('streaming',None),('human_only_controls',neutral)]:
        out=args.output/label;out.mkdir();(out/'checkpoints').mkdir()
        manifest=dict(head,epochs=1);manifest.pop('training',None)
        if training is not None:manifest['training']=training
        mp=out/'HEAD.json';mp.write_bytes(encoded(manifest))
        from shaping_rescue_execution import portable_command
        cmd=human_command(portable_command(original['command'],args.run),record['groups'],args.binary_dir,out)
        env=os.environ.copy()
        for key in list(env):
            if key.startswith(('ZENSIM_MONOTONE_RESEARCH_','ZENSIM_INPUT_SHAPING_','ZENSIM_PLAIN_OPTIMIZER_')):env.pop(key)
        env.update(ZENSIM_MONOTONE_RESEARCH_MANIFEST=str(mp),ZENSIM_MONOTONE_RESEARCH_SHA256=sha(mp),TMPDIR=str(Path.home()/'tmp'))
        with (out/'train.log').open('w') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            receipt['arms'][label]=dict(command=cmd,pid=process.pid,manifest_sha256=sha(mp))
            (args.output/'RUN.json').write_bytes(encoded(receipt));rc=process.wait()
        receipt['arms'][label]['exit_code']=rc
        (args.output/'PROGRESS.json').write_bytes(encoded(receipt))
        if rc:raise RuntimeError('human-only native qualification failed; preserve logs')
        model=out/'checkpoints/ckpt_epoch000.bin'
        for i,key in enumerate(('zentrain.repro','zentrain.monotone_research')):
            dest=out/f'canonical-{i}.bin'
            subprocess.run([str(args.binary_dir/'bake_dial_refit'),'strip','--in',str(model),'--out',str(dest),'--key',key],check=True)
            model=dest
        receipt['arms'][label]['canonical_sha256']=sha(model)
    receipt['match']=len({v['canonical_sha256'] for v in receipt['arms'].values()})==1
    (args.output/'RESULT.json').write_bytes(encoded(receipt))
    if not receipt['match']:raise ValueError('human-only neutral controls changed weights/scales')
    print(json.dumps(dict(family=head['family'],match=True,teacher_rows=0,training_rows=view['training_rows'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','run','binary-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--human-only-cell')
    p.add_argument('--worker',action='store_true');p.add_argument('--snapshot-root',type=Path)
    args=p.parse_args();(qualify_on_worker if args.worker else qualify_human if args.human_only_cell else qualify)(args)
