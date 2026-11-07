"""Repeat a completed fit for optimizer traces; refuse any canonical model drift."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
from shaping_completion import encoded,sha
from shaping_execution import load_registration


def replay(args):
    plan,cells,_=load_registration(args.study)
    original=json.loads((args.run/'RUN.json').read_text())
    result=json.loads((args.run/'RESULT.json').read_text())
    if original['cell'] not in cells or original['cell']['stage']!='inner':raise ValueError('registered inner fit required')
    if result['stage']!='trained' or result['exit_code']!=0:raise ValueError('completed fit required')
    if original['plan_sha256']!=sha(args.study/'PLAN.json'):raise ValueError('original plan drift')
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('diagnostic binary drift')
    old=args.run/'checkpoints'/f"ckpt_epoch{original['epochs']-1:03}.bin"
    if sha(old)!=result['bake_sha256']:raise ValueError('original checkpoint drift')
    if args.output.exists():raise ValueError('preserve existing replay')
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
    if [r['epoch'] for r in traces]!=list(range(original['epochs'])):raise ValueError('incomplete diagnostic trace')
    (args.output/'DIAGNOSTICS.json').write_bytes(encoded(traces))
    (args.output/'RESULT.json').write_bytes(encoded(dict(**receipt,canonical_sha256=canonical,match=canonical[0]==canonical[1],epochs=len(traces))))
    if canonical[0]!=canonical[1]:raise ValueError('diagnostics changed canonical model; do not attribute traces to original')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','binary-dir','run','output'):p.add_argument('--'+name,type=Path,required=True)
    replay(p.parse_args())
