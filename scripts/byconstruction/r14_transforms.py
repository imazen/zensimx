"""Local R14 jobs reusing the R13 shared-lock/resource and fit receipt owner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import r13_width as owner

X=owner.X
O=Path('/var/tmp/byc-transforms-20261104')
owner.O=O
owner.SCRATCH=Path.home()/'tmp/zensim-transforms'
owner.required_fits=lambda:3
ARMS=('T1a','T1b','T1c')


def args_for(arm,dest,seed=0):
    return [sys.executable,'-B',X/'scripts/rev4_featpot/r14_transforms_fit.py',
        '--r14-arm',arm,'--r14-table',O/f'screen/{arm}.tsv',
        '--root',owner.R,'--spec','sel:45acffa603e8@h32:H128:cv16:cf98:fx2:mono:thr:mp32',
        '--columns',(owner.H/'tmp/hc1-profile/cols_fx2.txt').read_text().strip(),
        '--head','N','--seed-index',seed,'--strict-admission',
        '--data-role-decision',owner.R/'human_role_decision.json','--dest',dest,'--train-only',
        '--human-sources','tid2013','--exclude-datasets',
        'pipal,live-md,cid-iq,cid-oa,irccyn-ivc,colourlab-rating-variability-2022','--mos-share','36.035']


def pack(arm):
    from v2_production_pack import pack_production
    cell=O/f'fits/{arm}-s0';dest=cell/'continuation/pack_e059'
    (dest/'refit').mkdir(parents=True,exist_ok=False)
    row=pack_production(cell/'continuation/ckpt/ckpt_epoch059.bin',dest,
        owner.R/'wide/main/real/cid22_fit.parquet',e33_output_stage=True,no_prune=True)
    bake=Path(row['packed_model'])
    cert=json.loads(subprocess.check_output([str(O/'bin/bake_dial_refit'),'certify-monotone','--bake',str(bake)]))
    if not cert['certified']:raise ValueError('R14 final certificate failed')
    inspect=O/'bin/prune_forward_bench'
    subprocess.run([str(inspect),'--inspect',str(bake),str(cell/'HEAD.json')],check=True)
    row.update(certificate=cert,capacity=owner.quick_capacity(cell/'HEAD.json'))
    owner.write(cell/'PACK.json',row)


def fit(arm):
    cell=O/f'fits/{arm}-s0';cell.mkdir(parents=True,exist_ok=True)
    if (cell/'FIT.json').exists():raise ValueError('fit already frozen; preserve it')
    owner.job(f'{arm}-s0-base',args_for(arm,cell/'base')+['--pack-production','--e33-output-stage'])
    owner.job(f'{arm}-s0-continuation',args_for(arm,cell/'continuation')+[
        '--init-from',cell/'base/refit/last.bin','--fine-tune-epochs',60,'--fine-tune-lr','0.00001',
        '--aic-public-train','/mnt/v/output/zensim/aic-finetune-20261026/folds-rev7/aic_full_fit.parquet',
        '--aic-weight',256,'--diagnostic-population','mixed-aic256-xref'])
    owner.job(f'{arm}-s0-pack',[sys.executable,'-B',__file__,'pack','--arm',arm])
    row=json.loads((cell/'PACK.json').read_text());times=[]
    for stage in ('base','continuation','pack'):
        raw=(O/'logs'/f'{arm}-s0-{stage}.time.txt').read_text()
        elapsed=next(v.split(': ',1)[1].strip() for v in raw.splitlines() if 'Elapsed (wall clock)' in v)
        parts=[float(v) for v in elapsed.split(':')]
        times.append(dict(stage=stage,seconds=sum(v*60**i for i,v in enumerate(reversed(parts)))))
    row.update(arm=arm,seed=0,wall_seconds=sum(v['seconds'] for v in times),stage_times=times,finished=owner.now())
    owner.write(cell/'FIT.json',row)
    print('PACKED '+json.dumps(dict(arm=arm,wall_seconds=row['wall_seconds'],capacity=row['capacity'])),flush=True)


def freeze():
    models={};rows=[]
    for arm in ARMS:
        row=json.loads((O/f'fits/{arm}-s0/FIT.json').read_text());bake=Path(row['packed_model'])
        if owner.sha(bake)!=row['packed_model_sha256']:raise ValueError('fitted bake changed')
        row['packed_sha256']=owner.sha(bake);row['packed_bytes']=bake.stat().st_size
        models[f'{arm}_s0']=str(bake);rows.append(row)
    owner.write(O/'MODELS.json',models)
    owner.write(O/'FROZEN_FITS.json',dict(models=rows,source_sha256=owner.sha(__file__)))


def development():
    import numpy as np
    import pyarrow.parquet as pq
    models=json.loads((O/'MODELS.json').read_text())
    models['F_s0']=str(owner.H/'tmp/e36test/r10/runs/mixF3-s0/pack_e059/refit/production-f16.bin')
    dest=O/'development';dest.mkdir(exist_ok=True);rows={}
    for group in ('safesyn','cid22'):
        table=owner.R/f'wide/main/real/{group}_dev.parquet'
        target=np.asarray(pq.read_table(table,columns=['human_score'])['human_score'])
        if not np.isfinite(target).all():raise ValueError('invalid teacher development target')
        rows[group]=dict(rows=len(target),table_sha256=owner.sha(table),mae={})
        for name,bake in models.items():
            path=dest/f'{group}-{name}.tsv'
            subprocess.run([str(O/'bin/bake_dial_refit'),'predict','--bake',bake,'--corpus',str(table),
                '--score-units','--out',str(path)],check=True)
            y=np.loadtxt(path,delimiter='\t',skiprows=1,usecols=[1])
            if len(y)!=len(target) or not np.isfinite(y).all():raise ValueError('invalid complete development predictions')
            rows[group]['mae'][name]=float(abs(y-target).mean())
    total=sum(v['rows'] for v in rows.values())
    pooled={name:sum(g['mae'][name]*g['rows'] for g in rows.values())/total for name in models}
    best=min(ARMS,key=lambda a:pooled[f'{a}_s0'])
    inputs=json.loads((O/'screen/INPUTS.json').read_text())
    with (O/f'screen/{best}.tsv').open() as f:
        import csv
        table=list(csv.DictReader(f,delimiter='\t'))
    trainable=[i for i,r in enumerate(table) if r['best_transform']=='soft_clip'
        and inputs[i]['p99']>max(0.,inputs[i]['p05'])]
    result=dict(groups=rows,pooled=pooled,best_arm=best,T2_feasible=pooled[f'{best}_s0']<=pooled['F_s0'],
        trainable_soft_clip_inputs=trainable,rule='best T1 pooled teacher-development MAE <= A0; zero allowance',
        units='stored human_score points, target scale1',role='TRAIN development only; no EVAL selection')
    owner.write(O/'DEVELOPMENT.json',result)
    print(json.dumps(result),flush=True)


def capacity():
    import r13_width_analysis as audit
    pins=json.loads((O/'SOURCE_PINS.json').read_text())
    owner.EXPORT=O/'bin/zensim-viz-export'
    audit.O=O
    audit.capacity(exporter_sha256=pins['binary_sha256']['zensim-viz-export'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('fit','pack','freeze','development','capacity','job'))
    ap.add_argument('--arm',choices=ARMS)
    ap.add_argument('--label')
    a,argv=ap.parse_known_args()
    if argv and a.command!='job':ap.error('unexpected arguments')
    for p in (O/'logs',owner.SCRATCH):p.mkdir(parents=True,exist_ok=True)
    if a.command in ('fit','pack'):
        if not a.arm:ap.error('--arm required')
        globals()[a.command](a.arm)
    elif a.command=='development':
        owner.job('development',[sys.executable,'-B',__file__,'development-worker'])
    elif a.command=='job':
        if not a.label:ap.error('--label required')
        owner.job(a.label,argv[1:] if argv[:1]==['--'] else argv)
    elif a.command=='capacity':capacity()
    else:freeze()


if __name__=='__main__':
    if sys.argv[1:]==['development-worker']:development()
    else:main()
