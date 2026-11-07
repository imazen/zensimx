"""Native neutral-control replay of a completed analytic prefix; never selection."""
from __future__ import annotations
import argparse,json,os,subprocess
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
    cmd=original['command'].copy();cmd[cmd.index('--mem')+1]='2G'
    index=next(i for i,v in enumerate(cmd) if Path(v).name=='zensim_mlp_train');cmd[index]=str(args.binary_dir/'zensim_mlp_train')
    for flag,value in [('--epochs',args.epochs),('--out',args.output/'best.bin'),('--dump-checkpoints-dir',args.output/'checkpoints')]:
        cmd[cmd.index(flag)+1]=str(value)
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('ZENSIM_MONOTONE_RESEARCH_','ZENSIM_INPUT_SHAPING_','ZENSIM_PLAIN_OPTIMIZER_')):env.pop(key)
    env.update(ZENSIM_MONOTONE_RESEARCH_MANIFEST=str(manifest),ZENSIM_MONOTONE_RESEARCH_SHA256=sha(manifest),
               ZENSIM_OPTIMIZER_DIAGNOSTICS='1',TMPDIR=str(Path.home()/'tmp'))
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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','run','binary-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=1);qualify(p.parse_args())
