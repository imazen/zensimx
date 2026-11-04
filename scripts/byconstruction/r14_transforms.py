"""Local R14 jobs reusing the R13 shared-lock/resource and fit receipt owner."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys

import r13_width as owner

X=owner.X
O=Path('/var/tmp/byc-transforms-20261104')
owner.O=O
owner.SCRATCH=Path.home()/'tmp/zensim-transforms'
owner.required_fits=lambda:3
ARMS=('T1a','T1b','T1c')
SPEC='sel:45acffa603e8@h32:H128:cv16:cf98:fx2:mono:thr:mp32'


def validate_owners(pins,replacement):
    for relative in ('r14_transforms_fit.py','v2_confirm_fit.py','v2_lodo_mlp.py',
                     'v2_production_pack.py','subjective_training_leg.py',
                     'v2_teacher.py','v2c_wide.py','v2_common.py'):
        source=f'scripts/rev4_featpot/{relative}'
        observed=owner.sha(X/source)
        if observed==pins['source_sha256'][source]:continue
        if relative!='v2_common.py' or replacement is None:
            raise ValueError(f'fit owner changed since native freeze: {source}')
        old=subprocess.check_output(['jj','file','show','-r',pins['source_commit'],source],cwd=X)
        new=subprocess.check_output(['jj','file','show','-r',replacement['source_commit'],source],cwd=X)
        if hashlib.sha256(new).hexdigest()!=observed:raise ValueError('owner differs from successor trainer source')
        def outside_recipe(data):
            tree=ast.parse(data)
            tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='recipe_of']
            return ast.dump(tree,include_attributes=False)
        if outside_recipe(old)!=outside_recipe(new):raise ValueError('shared owner changed outside recipe parser')
        def recipe(data):
            namespace={'__name__':'r14_owner_compatibility','__file__':str(X/source)}
            exec(compile(data,str(X/source),'exec'),namespace)
            return namespace['recipe_of'](SPEC)
        original,current=recipe(old),recipe(new)
        if original!=current:raise ValueError('registered R14 recipe changed')
        receipt=dict(source=source,original_sha256=pins['source_sha256'][source],successor_sha256=observed,
            successor_commit=replacement['source_commit'],outside_recipe_ast='identical',
            registered_spec=SPEC,recipe=current,role='metadata-only compatibility; no data or model selection')
        out=O/f'owner-compat-v2-common-{observed[:12]}.json'
        if out.exists() and json.loads(out.read_text())!=receipt:raise ValueError('preserve prior owner compatibility receipt')
        owner.write(out,receipt)


def args_for(arm,dest,seed=0):
    return [sys.executable,'-B',X/'scripts/rev4_featpot/r14_transforms_fit.py',
        '--r14-arm',arm,'--r14-table',O/f'screen/{arm}.tsv',
        '--root',owner.R,'--spec',SPEC,
        '--columns',(owner.H/'tmp/hc1-profile/cols_fx2.txt').read_text().strip(),
        '--head','N','--seed-index',seed,'--strict-admission',
        '--data-role-decision',owner.R/'human_role_decision.json','--dest',dest,'--train-only',
        '--human-sources','tid2013','--exclude-datasets',
        'pipal,live-md,cid-iq,cid-oa,irccyn-ivc,colourlab-rating-variability-2022','--mos-share','36.035']


def pack(arm,continuation_name='continuation'):
    from v2_production_pack import pack_production
    cell=O/f'fits/{arm}-s0';continuation=cell/continuation_name;dest=continuation/'pack_e059'
    (dest/'refit').mkdir(parents=True,exist_ok=False)
    row=pack_production(continuation/'ckpt/ckpt_epoch059.bin',dest,
        owner.R/'wide/main/real/cid22_fit.parquet',e33_output_stage=True,no_prune=True)
    bake=Path(row['packed_model'])
    cert=json.loads(subprocess.check_output([str(O/'bin/bake_dial_refit'),'certify-monotone','--bake',str(bake)]))
    if not cert['certified']:raise ValueError('R14 final certificate failed')
    inspect=O/'bin/prune_forward_bench'
    subprocess.run([str(inspect),'--inspect',str(bake),str(cell/'HEAD.json')],check=True)
    row.update(certificate=cert,capacity=owner.quick_capacity(cell/'HEAD.json'))
    owner.write(cell/'PACK.json',row)


def fit(arm,trainer_dir=None,continuation_retry=None):
    driver_sha256=owner.sha(__file__)
    pins=json.loads((O/'SOURCE_PINS.json').read_text())
    screen=json.loads((O/'screen/SUMMARY.json').read_text())
    proof=json.loads((O/'native-noop/PROOF.json').read_text())
    if (owner.sha(O/'bin/zensim_mlp_train')!=pins['binary_sha256']['zensim_mlp_train']
            or owner.sha(O/f'screen/{arm}.tsv')!=screen['tables'][arm]['sha256']
            or proof['source_commit']!=pins['source_commit']
            or proof['new_tool']['sha256']!=pins['binary_sha256']['predict_features_with_bake']
            or set(proof['banks'])!=set(owner.BANKS)
            or any(any(b['differences'].values()) for b in proof['banks'].values())):
        raise ValueError('fit requires frozen screen, native trainer and complete exact A no-op proof')
    cell=O/f'fits/{arm}-s0';cell.mkdir(parents=True,exist_ok=True)
    if (cell/'FIT.json').exists():raise ValueError('fit already frozen; preserve it')
    native=dict(path=str(O/'bin/zensim_mlp_train'),sha256=pins['binary_sha256']['zensim_mlp_train'],source_commit=pins['source_commit'])
    active=O/'ACTIVE_TRAINER.json'
    if trainer_dir is None and active.exists():
        selected=json.loads(active.read_text())
        trainer_dir=Path(selected['trainer_dir'])
        if (selected['schema']!='r14-selected-caller-loader-v1'
                or owner.sha(trainer_dir/'TRAINER_PINS.json')!=selected['trainer_pins_sha256']):
            raise ValueError('active trainer selection differs')
    prefix=[]
    replacement=None
    if trainer_dir:
        trainer_dir=trainer_dir.resolve()
        replacement=json.loads((trainer_dir/'TRAINER_PINS.json').read_text())
        if (replacement['registration']!='5b5a013482db4944042883995ad6892f92244c69'
                or replacement['serving_pins_sha256']!=owner.sha(O/'SOURCE_PINS.json')
                or owner.sha(trainer_dir/'zensim_mlp_train')!=replacement['trainer_sha256']):
            raise ValueError('replacement trainer identity differs')
        native=dict(path=str(trainer_dir/'zensim_mlp_train'),sha256=replacement['trainer_sha256'],source_commit=replacement['source_commit'])
        prefix=['env',f'REV4_V2_BIN_DIR={trainer_dir}']
    validate_owners(pins,replacement)
    labels={stage:f'{arm}-s0-{stage}' for stage in ('base','continuation','pack')}
    continuation_name='continuation'
    failed_attempts=[]
    if continuation_retry:
        if continuation_retry!='r2' or trainer_dir is None:
            raise ValueError('only the documented pre-optimizer continuation repair is admitted')
        old=json.loads((O/'logs'/f'{arm}-s0-continuation.receipt.json').read_text())
        oldlog=cell/'continuation/train.log'
        import re
        if (old['rc']==0 or re.search(r'epoch=\d+',oldlog.read_text())
                or list((cell/'continuation/ckpt').glob('ckpt_epoch*.bin'))
                or 'null feature value' not in oldlog.read_text()):
            raise ValueError('repair requires the retained unused-null failure before any optimizer step')
        failed_attempts.append(dict(label=f'{arm}-s0-continuation',rc=old['rc'],
            receipt_sha256=owner.sha(O/'logs'/f'{arm}-s0-continuation.receipt.json'),
            log_sha256=owner.sha(oldlog),optimizer_steps=0))
        labels['continuation']+=f'-{continuation_retry}'
        continuation_name+=f'-{continuation_retry}'
    base_done=(O/'logs'/f'{arm}-s0-base.receipt.json').exists()
    base_native_path=cell/'BASE_NATIVE.json'
    if base_native_path.exists():
        base_native=json.loads(base_native_path.read_text())
    else:
        base_native=native
        if base_done:
            previous=json.loads((O/'logs'/f'{arm}-s0-base.receipt.json').read_text())
            if previous['argv'][0]=='env':raise ValueError('missing prior overridden base trainer identity')
            base_native=dict(path=str(O/'bin/zensim_mlp_train'),sha256=pins['binary_sha256']['zensim_mlp_train'],source_commit=pins['source_commit'])
        owner.write(base_native_path,base_native)
    if owner.sha(base_native['path'])!=base_native['sha256']:raise ValueError('base trainer identity changed')
    owner.job(labels['base'],prefix+args_for(arm,cell/'base')+['--pack-production','--e33-output-stage'])
    owner.job(labels['continuation'],prefix+args_for(arm,cell/continuation_name)+[
        '--init-from',cell/'base/refit/last.bin','--fine-tune-epochs',60,'--fine-tune-lr','0.00001',
        '--aic-public-train','/mnt/v/output/zensim/aic-finetune-20261026/folds-rev7/aic_full_fit.parquet',
        '--aic-weight',256,'--diagnostic-population','mixed-aic256-xref'])
    owner.job(labels['pack'],[sys.executable,'-B',__file__,'pack','--arm',arm,'--continuation-name',continuation_name])
    row=json.loads((cell/'PACK.json').read_text());times=[]
    for stage in ('base','continuation','pack'):
        raw=(O/'logs'/f'{labels[stage]}.time.txt').read_text()
        elapsed=next(v.split(': ',1)[1].strip() for v in raw.splitlines() if 'Elapsed (wall clock)' in v)
        parts=[float(v) for v in elapsed.split(':')]
        times.append(dict(stage=stage,label=labels[stage],seconds=sum(v*60**i for i,v in enumerate(reversed(parts)))))
    row.update(arm=arm,seed=0,wall_seconds=sum(v['seconds'] for v in times),stage_times=times,
        native_stages=dict(base=base_native,continuation=native),driver_sha256=driver_sha256,
        failed_attempts=failed_attempts,continuation_name=continuation_name,finished=owner.now())
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
        table=list(csv.DictReader(f,delimiter='\t'))
    trainable=[i for i,r in enumerate(table) if r['best_transform']=='soft_clip'
        and inputs[i]['p99']>max(0.,inputs[i]['p05'])]
    result=dict(groups=rows,pooled=pooled,best_arm=best,T2_feasible=pooled[f'{best}_s0']<=pooled['F_s0'],
        trainable_soft_clip_inputs=trainable,rule='best T1 pooled teacher-development MAE <= A0; zero allowance',
        units='stored human_score points, target scale1',role='TRAIN development only; no EVAL selection')
    owner.write(O/'DEVELOPMENT.json',result)
    print(json.dumps(result),flush=True)



def development_final():
    """Report-only development reads after the conditional T2 decision is final."""
    import numpy as np
    import pyarrow.parquet as pq
    release=json.loads((O/'T2_RELEASE.json').read_text())
    frozen=json.loads((O/'FROZEN_FITS.json').read_text())
    models=json.loads((O/'MODELS.json').read_text())
    if len(frozen['models'])!=3+int(release['run']) or len(models)!=len(frozen['models']):
        raise ValueError('final development reads require completed T1/T2 inventory')
    for row in frozen['models']:
        if owner.sha(row['packed_model'])!=row['packed_sha256']:raise ValueError('frozen bake changed')
    models.update({f'F_s{s}':str(owner.H/f'tmp/e36test/r10/runs/mixF3-s{s}/pack_e059/refit/production-f16.bin') for s in range(3)})
    dest=O/'development-final';dest.mkdir(exist_ok=False);groups={}
    for group in ('safesyn','cid22'):
        table=owner.R/f'wide/main/real/{group}_dev.parquet'
        target=np.asarray(pq.read_table(table,columns=['human_score'])['human_score'])
        if not np.isfinite(target).all():raise ValueError('invalid development target')
        groups[group]=dict(rows=len(target),table_sha256=owner.sha(table),mae={})
        for name,bake in models.items():
            path=dest/f'{group}-{name}.tsv'
            subprocess.run([str(O/'bin/bake_dial_refit'),'predict','--bake',bake,'--corpus',str(table),
                '--score-units','--out',str(path)],check=True)
            prediction=np.loadtxt(path,delimiter='\t',skiprows=1,usecols=[1])
            if len(prediction)!=len(target) or not np.isfinite(prediction).all():raise ValueError('incomplete predictions')
            groups[group]['mae'][name]=float(abs(prediction-target).mean())
    n=sum(v['rows'] for v in groups.values())
    pooled={name:sum(v['mae'][name]*v['rows'] for v in groups.values())/n for name in models}
    initial=json.loads((O/'DEVELOPMENT.json').read_text())
    if any(pooled[name]!=value for name,value in initial['pooled'].items()):
        raise ValueError('frozen T1/A0 development statistic changed')
    owner.write(O/'FINAL_DEVELOPMENT.json',dict(groups=groups,pooled=pooled,
        units=initial['units'],role='report only; conditional T2 release already fixed',
        T2_release_sha256=owner.sha(O/'T2_RELEASE.json')))


def capacity():
    import r13_width_analysis as audit
    pins=json.loads((O/'SOURCE_PINS.json').read_text())
    owner.EXPORT=O/'bin/zensim-viz-export'
    audit.O=O
    audit.capacity(exporter_sha256=pins['binary_sha256']['zensim-viz-export'])


def reads():
    common=[sys.executable,'-B',X/'scripts/byconstruction/r12_linear_reads.py',
        '--models',O/'MODELS.json','--out',O/'reads','--bin-dir',O/'bin']
    for stage in ('eval','e36','chroma','hdr'):
        owner.job('read-'+stage,common+['--only',stage])
    for name in json.loads((O/'MODELS.json').read_text()):
        for stage in ('steer','dial','nearid'):
            owner.job(f'read-{stage}-{name}',common+['--only',stage,'--include-model',name])


def finalize():
    import r13_width_finalize as strip
    owner.HEAD=O/'bin/prune_forward_bench'
    strip.worker(registered_hidden=128)
    frozen=json.loads((O/'FROZEN_FITS.json').read_text())
    for row in frozen['models']:
        for key in ('zentrain.feature_transforms','zentrain.feature_transform_params'):
            values=[subprocess.check_output([str(O/'bin/bake_dial_refit'),'read-meta',
                '--in',str(row[k]),'--key',key]) for k in ('packed_model','shipped_model')]
            if values[0]!=values[1]:raise ValueError('stripping changed transform metadata')
        row['stripped_transform_metadata']='exactly equal to packed table and params'
    owner.write(O/'FROZEN_FITS.json',frozen)


def fit_progress(destination):
    pins=json.loads((O/'SOURCE_PINS.json').read_text())
    screen=json.loads((O/'screen/SUMMARY.json').read_text())
    rows=[];pending=[]
    for arm in ARMS:
        cell=O/f'fits/{arm}-s0'
        if not (cell/'FIT.json').exists():
            pending.append(arm);continue
        fit=json.loads((cell/'FIT.json').read_text())
        if owner.sha(fit['packed_model'])!=fit['packed_model_sha256'] or not fit['certificate']['certified']:
            raise ValueError('frozen fit identity/certificate differs')
        transforms=fit['certificate']['feature_transforms']
        tokens=dict(Counter(t['token'] for t in transforms))
        if len(transforms)!=984 or tokens!=screen['token_counts'][arm]:
            raise ValueError('packed transform table token inventory differs')
        with (O/f'screen/{arm}.tsv').open() as f:
            expected=list(csv.DictReader(f,delimiter='\t'))
        f32=lambda values:b''.join(struct.pack('<f',float(v)) for v in values)
        for i,(actual,wanted) in enumerate(zip(transforms,expected,strict=True)):
            if (actual['caller_input']!=i or actual['token']!=wanted['best_transform']
                    or f32(actual['params'])!=f32(v for v in wanted['params_csv'].split(',') if v)):
                raise ValueError(f'packed transform parameters differ at caller input {i}')
        if any(t['value_at_zero']!=0 or not t['finite_positive_one_sided_derivative']
               or t['derivative_at_zero']<=0 for t in transforms):
            raise ValueError('packed transform zero-slope contract differs')
        resources={}
        stage_labels={v['stage']:v.get('label',f'{arm}-s0-{v["stage"]}') for v in fit['stage_times']}
        for stage in ('base','continuation','pack'):
            lines=[v for v in (O/'logs'/f'{stage_labels[stage]}.log').read_text().splitlines()
                   if v.startswith('run-heavy: done')]
            if len(lines)!=1:raise ValueError('unique complete resource receipt required')
            resources[stage]=lines[0]
        rows.append(dict(arm=arm,seed=0,packed_sha256=fit['packed_model_sha256'],
            packed_bytes=Path(fit['packed_model']).stat().st_size,
            wall_seconds_excluding_lock_waits=fit['wall_seconds'],stage_times=fit['stage_times'],
            capacity={k:fit['capacity'][k] for k in ('dead','residual','live')},
            certified=True,certified_max_pairs=fit['certificate']['max_pairs'],
            token_counts=tokens,resource_receipts=resources,native_stages=fit.get('native_stages'),
            failed_attempts=fit.get('failed_attempts',[]),finished=fit['finished']))
    owner.write(destination,dict(registration='5b5a013482db4944042883995ad6892f92244c69',
        trainer_source_commit=pins['source_commit'],trainer_sha256=pins['binary_sha256']['zensim_mlp_train'],
        screen_summary_sha256=owner.sha(O/'screen/SUMMARY.json'),models=rows,pending=pending,
        completed=len(rows),required_T1_fits=3,
        role='packed-fit identities only; no AIC-2026 quality statistics or EVAL selection'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('fit','pack','freeze','development','development-final','capacity','reads','finalize','fit-progress','job'))
    ap.add_argument('--arm',choices=ARMS)
    ap.add_argument('--label')
    ap.add_argument('--out',type=Path)
    ap.add_argument('--trainer-dir',type=Path)
    ap.add_argument('--continuation-retry',choices=('r2',))
    ap.add_argument('--continuation-name',default='continuation',choices=('continuation','continuation-r2'))
    a,argv=ap.parse_known_args()
    if argv and a.command!='job':ap.error('unexpected arguments')
    for p in (O/'logs',owner.SCRATCH):p.mkdir(parents=True,exist_ok=True)
    if a.command in ('fit','pack'):
        if not a.arm:ap.error('--arm required')
        if a.command=='fit':fit(a.arm,a.trainer_dir,a.continuation_retry)
        else:pack(a.arm,a.continuation_name)
    elif a.command=='development':
        owner.job('development',[sys.executable,'-B',__file__,'development-worker'])
    elif a.command=='development-final':
        owner.job('development-final',[sys.executable,'-B',__file__,'development-final-worker'])
    elif a.command=='job':
        if not a.label:ap.error('--label required')
        owner.job(a.label,argv[1:] if argv[:1]==['--'] else argv)
    elif a.command=='capacity':capacity()
    elif a.command=='reads':reads()
    elif a.command=='finalize':
        owner.job('metadata-strip',[sys.executable,'-B',__file__,'finalize-worker'])
    elif a.command=='fit-progress':
        if not a.out:ap.error('--out required')
        fit_progress(a.out)
    else:freeze()


if __name__=='__main__':
    if sys.argv[1:]==['development-worker']:development()
    elif sys.argv[1:]==['finalize-worker']:finalize()
    elif sys.argv[1:]==['development-final-worker']:development_final()
    else:main()
