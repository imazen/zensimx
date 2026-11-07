"""Repeat a completed fit for optimizer traces; refuse any canonical model drift."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
from shaping_completion import encoded,sha
from shaping_execution import load_registration


def canonical_pairs(binary_dir,pairs,output):
    """Require every requested checkpoint, including transient states, to match."""
    rows=[]
    for epoch,old,new in pairs:
        pins=[]
        for label,path in [('original',old),('diagnostic',new)]:
            dest=output/(f'epoch-{epoch:03}-'+label+'-canonical.bin')
            subprocess.run([str(binary_dir/'bake_dial_refit'),'strip','--in',str(path),'--out',str(dest),'--key','zentrain.repro'],check=True)
            pins.append(sha(dest))
        rows.append(dict(epoch=epoch,canonical_sha256=pins,match=pins[0]==pins[1]))
    return rows


def first_weight_traces(log,epochs):
    rows=[json.loads(line.split(" ",1)[1]) for line in log.splitlines() if line.startswith("FIRST_WEIGHT_DIAGNOSTICS ")]
    if [r["epoch"] for r in rows]!=list(range(epochs)):
        raise ValueError("incomplete first-weight diagnostic trace")
    return rows


def replay(args):
    from shaping_storage import reservation
    plan=json.loads((args.study/'PLAN.json').read_text())
    if args.output.exists():raise ValueError('preserve existing replay')
    with reservation(args.study,args.output,plan['minimum_ssd_free_bytes']):
        return admitted_replay(args)


def admitted_replay(args):
    plan,cells,_=load_registration(args.study)
    original=json.loads((args.run/'RUN.json').read_text())
    result=json.loads((args.run/'RESULT.json').read_text())
    if original['cell'] not in cells:raise ValueError('registered fit required')
    epochs=original.get('epochs',original['cell'].get('epochs'))
    if epochs is None:raise ValueError('missing registered epoch budget')
    pilot=getattr(args,'resource_pilot',False)
    if pilot:
        if original['cell']['stage']!='resource_pilot' or epochs!=1 or result['stage']!='complete':
            raise ValueError('only completed one-epoch resource pilot is a qualification replay')
    elif original['cell']['stage']!='inner' or result['stage']!='trained':raise ValueError('completed registered inner fit required')
    if result['exit_code']!=0:raise ValueError('failed fit cannot qualify a replay')
    if original['plan_sha256']!=sha(args.study/'PLAN.json'):raise ValueError('original plan drift')
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('diagnostic binary drift')
    old=args.run/'checkpoints'/f"ckpt_epoch{epochs-1:03}.bin"
    if sha(old)!=result['bake_sha256']:raise ValueError('original checkpoint drift')
    if args.output.exists():raise ValueError('preserve existing replay')
    if getattr(args,'all_checkpoints',False):
        if not all((args.run/'checkpoints'/f'ckpt_epoch{epoch:03}.bin').is_file() for epoch in range(epochs)):
            raise ValueError('all-checkpoint replay requires complete original checkpoint sequence')
    args.output.mkdir(parents=True);(args.output/'checkpoints').mkdir()
    cmd=original['command'].copy();start=next(i for i,v in enumerate(cmd) if Path(v).name=='zensim_mlp_train')
    cmd[start]=str(args.binary_dir/'zensim_mlp_train')
    for flag,value in [('--out',args.output/'best.bin'),('--dump-checkpoints-dir',args.output/'checkpoints')]:
        cmd[cmd.index(flag)+1]=str(value)
    env=os.environ.copy()
    for key in ('ZENSIM_MONOTONE_RESEARCH_MANIFEST','ZENSIM_MONOTONE_RESEARCH_SHA256',
                'ZENSIM_INPUT_SHAPING_MANIFEST','ZENSIM_INPUT_SHAPING_SHA256',
                'ZENSIM_PLAIN_OPTIMIZER_MANIFEST','ZENSIM_PLAIN_OPTIMIZER_SHA256'):
        env.pop(key,None)
    for name,prefix in [('HEAD.json','ZENSIM_MONOTONE_RESEARCH'),('SHAPING.json','ZENSIM_INPUT_SHAPING')]:
        path=args.run/name
        if path.exists():env[prefix+'_MANIFEST']=str(path);env[prefix+'_SHA256']=sha(path)
    env['ZENSIM_OPTIMIZER_DIAGNOSTICS']='1'
    receipt=dict(original_run=str(args.run),original_result_sha256=sha(args.run/'RESULT.json'),
                 original_build_sha256=original['build_sha256'],diagnostic_build_sha256=sha(args.binary_dir/'BUILD.json'),
                 command=cmd,selection_candidate=False,purpose='optimizer diagnostics with unchanged fit',driver_sha256=sha(__file__))
    with (args.output/'train.log').open('w') as log:
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        receipt['pid']=process.pid;(args.output/'RUN.json').write_bytes(encoded(receipt));rc=process.wait()
    if rc:raise RuntimeError('diagnostic replay failed; preserve train.log')
    new=args.output/'checkpoints'/old.name
    canonical=[]
    for label,path in [('original',old),('diagnostic',new)]:
        dest=args.output/(label+'-canonical.bin')
        subprocess.run([str(args.binary_dir/'bake_dial_refit'),'strip','--in',str(path),'--out',str(dest),'--key','zentrain.repro'],check=True)
        canonical.append(sha(dest))
    traces=[]
    for line in (args.output/'train.log').read_text().splitlines():
        if line.startswith('TRAINING_DIAGNOSTICS '):traces.append(json.loads(line.split(' ',1)[1]))
    if [r['epoch'] for r in traces]!=list(range(epochs)):raise ValueError('incomplete diagnostic trace')
    (args.output/'DIAGNOSTICS.json').write_bytes(encoded(traces))
    if getattr(args,'require_weight_diagnostics',False):
        weight_traces=first_weight_traces((args.output/'train.log').read_text(),epochs)
        (args.output/'FIRST_WEIGHT_DIAGNOSTICS.json').write_bytes(encoded(weight_traces))
    checkpoint_matches=[]
    if getattr(args,'all_checkpoints',False):
        pairs=[(epoch,args.run/'checkpoints'/f'ckpt_epoch{epoch:03}.bin',args.output/'checkpoints'/f'ckpt_epoch{epoch:03}.bin') for epoch in range(epochs)]
        checkpoint_matches=canonical_pairs(args.binary_dir,pairs,args.output)
    matched=canonical[0]==canonical[1] and all(r['match'] for r in checkpoint_matches)
    (args.output/'RESULT.json').write_bytes(encoded(dict(**receipt,canonical_sha256=canonical,match=matched,epochs=len(traces),checkpoint_matches=checkpoint_matches)))
    if not matched:raise ValueError('diagnostics changed canonical model; do not attribute traces to original')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','binary-dir','run','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--all-checkpoints',action='store_true',help='require canonical parity at every scheduled checkpoint, not only final')
    p.add_argument('--require-weight-diagnostics',action='store_true',help='require every epoch of sampled first-layer weight diagnostics')
    p.add_argument('--resource-pilot',action='store_true',help='qualification only, excluded from scientific selection')
    replay(p.parse_args())
