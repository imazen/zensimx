"""Release and run the registered conditional T2; never select on EVAL."""
import argparse
import csv
import json
from pathlib import Path
import struct
import sys

import r14_transforms as fixed

owner=fixed.owner
O=fixed.O
X=fixed.X


def release():
    frozen=json.loads((O/'FROZEN_FITS.json').read_text())
    models=json.loads((O/'MODELS.json').read_text())
    if set(models)!={f'{a}_s0' for a in fixed.ARMS} or len(frozen['models'])!=3:
        raise ValueError('T2 release requires all three frozen T1 fits')
    for row in frozen['models']:
        if owner.sha(row['packed_model'])!=row['packed_sha256']:
            raise ValueError('T1 frozen bake changed')
    development=json.loads((O/'DEVELOPMENT.json').read_text())
    best=min(fixed.ARMS,key=lambda a:development['pooled'][f'{a}_s0'])
    feasible=development['pooled'][f'{best}_s0']<=development['pooled']['F_s0']
    if development['best_arm']!=best or development['T2_feasible']!=feasible:
        raise ValueError('inconsistent registered development decision')
    eligible=development['trainable_soft_clip_inputs']
    run=feasible and bool(eligible)
    result=dict(registration='5b5a013482db4944042883995ad6892f92244c69',run=run,
        best_arm=best,trainable_soft_clip_inputs=eligible,
        reason=('registered feasibility and eligible soft-clip slots' if run else
                'best T1 misses zero-allowance development feasibility' if not feasible else
                'best T1 has no eligible soft-clip slots'),
        development_sha256=owner.sha(O/'DEVELOPMENT.json'),
        frozen_T1_sha256=owner.sha(O/'FROZEN_FITS.json'),
        source_sha256=owner.sha(__file__))
    path=O/'T2_RELEASE.json'
    if path.exists() and json.loads(path.read_text())!=result:
        raise ValueError('preserve prior T2 release decision')
    owner.write(path,result)
    return result


def fit():
    decision=release()
    print(json.dumps(decision),flush=True)
    if not decision['run']:return
    selected=json.loads((O/'T2_TRAINER.json').read_text())
    native_dir=Path(selected['trainer_dir'])
    native=json.loads((native_dir/'TRAINER_PINS.json').read_text())
    if (selected['schema']!='r14-t2-f32-bounds-v1'
            or owner.sha(native_dir/'TRAINER_PINS.json')!=selected['trainer_pins_sha256']
            or owner.sha(native_dir/'zensim_mlp_train')!=native['trainer_sha256']
            or native['serving_pins_sha256']!=owner.sha(O/'SOURCE_PINS.json')):
        raise ValueError('T2 native trainer identity differs')
    fixed.validate_owners(json.loads((O/'SOURCE_PINS.json').read_text()),native)
    wrapper='scripts/rev4_featpot/r14_learn_knees_fit.py'
    for source in (wrapper,'zensim-validate/src/mlp_train/r14_knees.rs'):
        if owner.sha(X/source)!=native['source_sha256'][source]:
            raise ValueError('T2 learner differs from native source freeze')
    cell=O/'fits/T2-s0'
    cell.mkdir(exist_ok=True)
    if (cell/'FIT.json').exists():raise ValueError('preserve completed T2 fit')

    def arguments(dest):
        argv=fixed.args_for(decision['best_arm'],dest)
        argv[2]=X/wrapper
        argv[argv.index('--r14-arm')+1]='T2'
        return ['env',f'REV4_V2_BIN_DIR={native_dir}',*argv,
            '--r14-knee-bounds',O/'screen/INPUTS.json',
            '--r14-development',O/'DEVELOPMENT.json']

    owner.job('T2-s0-base',arguments(cell/'base')+['--pack-production','--e33-output-stage'])
    owner.job('T2-s0-continuation',arguments(cell/'continuation')+[
        '--init-from',cell/'base/refit/last.bin','--fine-tune-epochs',60,'--fine-tune-lr','0.00001',
        '--aic-public-train','/mnt/v/output/zensim/aic-finetune-20261026/folds-rev7/aic_full_fit.parquet',
        '--aic-weight',256,'--diagnostic-population','mixed-aic256-xref'])
    owner.job('T2-s0-pack',[sys.executable,'-B',__file__,'pack-worker'])
    row=json.loads((cell/'PACK.json').read_text())
    times=[]
    for stage in ('base','continuation','pack'):
        label=f'T2-s0-{stage}'
        raw=(O/'logs'/f'{label}.time.txt').read_text()
        elapsed=next(v.split(': ',1)[1].strip() for v in raw.splitlines() if 'Elapsed (wall clock)' in v)
        times.append(dict(stage=stage,label=label,
            seconds=sum(float(v)*60**i for i,v in enumerate(reversed(elapsed.split(':'))))))
    identity=dict(path=str(native_dir/'zensim_mlp_train'),sha256=native['trainer_sha256'],source_commit=native['source_commit'])
    row.update(arm='T2',seed=0,wall_seconds=sum(t['seconds'] for t in times),stage_times=times,
        native_stages=dict(base=identity,continuation=identity),driver_sha256=owner.sha(__file__),
        release_sha256=owner.sha(O/'T2_RELEASE.json'),finished=owner.now())
    owner.write(cell/'FIT.json',row)
    print('PACKED '+json.dumps(dict(arm='T2',wall_seconds=row['wall_seconds'],capacity=row['capacity'])),flush=True)


def freeze():
    decision=json.loads((O/'T2_RELEASE.json').read_text())
    if decision['development_sha256']!=owner.sha(O/'DEVELOPMENT.json'):
        raise ValueError('T2 development decision changed')
    if not decision['run']:return
    path=O/'FROZEN_FITS.json'
    if owner.sha(path)!=decision['frozen_T1_sha256']:
        raise ValueError('preserve prior freeze; T1 inventory changed')
    frozen=json.loads(path.read_text())
    row=json.loads((O/'fits/T2-s0/FIT.json').read_text())
    if owner.sha(row['packed_model'])!=row['packed_model_sha256']:
        raise ValueError('T2 final bake changed')
    with (O/f"screen/{decision['best_arm']}.tsv").open() as f:
        initial=list(csv.DictReader(f,delimiter='\t'))
    inputs=json.loads((O/'screen/INPUTS.json').read_text())
    final=row['certificate']['feature_transforms']
    if len(final)!=984 or not row['certificate']['certified']:
        raise ValueError('incomplete T2 final certificate')
    f32=lambda x:struct.unpack('<f',struct.pack('<f',float(x)))[0]
    changed=[];lower=[];upper=[]
    eligible=set(decision['trainable_soft_clip_inputs'])
    for i,(a,b) in enumerate(zip(final,initial,strict=True)):
        original=[f32(v) for v in b['params_csv'].split(',') if v]
        if a['caller_input']!=i or a['token']!=b['best_transform']:
            raise ValueError('T2 changed a transform kind')
        if i in eligible:
            knee,soft=a['params']
            raw_lo,raw_hi=max(0.,inputs[i]['p05']),inputs[i]['p99']
            lo,hi=f32(raw_lo),f32(raw_hi)
            bits=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
            number=lambda n:struct.unpack('<f',struct.pack('<I',n))[0]
            if lo<raw_lo:lo=number(bits(lo)+1)
            if hi>raw_hi:hi=number(bits(hi)-1)
            if knee!=soft or not (knee>0 and raw_lo<=knee<=raw_hi):
                raise ValueError('T2 knee outside registered positive TRAIN bounds')
            if knee!=original[0]:changed.append(i)
            if knee==lo:lower.append(i)
            if knee==hi:upper.append(i)
        elif a['params']!=original:
            raise ValueError('T2 changed a fixed transform parameter')
    row.update(packed_sha256=row['packed_model_sha256'],packed_bytes=Path(row['packed_model']).stat().st_size,
        learned_knees=dict(eligible=len(eligible),changed=len(changed),lower_bound=len(lower),upper_bound=len(upper),
            changed_callers=changed,lower_bound_callers=lower,upper_bound_callers=upper))
    frozen['models'].append(row)
    frozen['T2_release_sha256']=owner.sha(O/'T2_RELEASE.json')
    models=json.loads((O/'MODELS.json').read_text())
    if 'T2_s0' in models:raise ValueError('T2 already frozen')
    models['T2_s0']=row['packed_model']
    owner.write(O/'MODELS.json',models)
    owner.write(path,frozen)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','fit','freeze','pack-worker'))
    args=ap.parse_args()
    if args.command=='prepare':
        decision=release()
        if decision['run']:
            owner.job('T2-trainer-build',['bash',X/'scripts/byconstruction/r14_trainer_build.sh','t2-bounds'])
    elif args.command=='fit':fit()
    elif args.command=='freeze':freeze()
    else:fixed.pack('T2')


if __name__=='__main__':main()
