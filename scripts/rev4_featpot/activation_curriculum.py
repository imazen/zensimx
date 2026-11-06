#!/usr/bin/env python3
"""Independent scalar/schedule probes and strictly admitted native optimization pilots.

References are not metric scorers. Training, baking, predictions and statistics
remain owned by the existing Rust tools and v2 admission/command builders.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys


ARMS = ('control', 'walk', 'lr_low', 'l2_high', 'teacher_half', 'teacher_double')
SEEDS = ((1101, 101), (1107, 101), (1101, 200000101))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def activation(name, x, tau=0.125):
    if name == 'relu':
        return max(x, 0.), float(x > 0)
    if name == 'smooth_hinge':
        if x <= 0:
            return 0., 0.
        if x < tau:
            return x*x/(2*tau), x/tau
        return x-tau/2, 1.
    if name == 'leaky':
        return (x, 1.) if x >= 0 else (0.1*x, 0.1)
    if name == 'softplus':
        u = x/tau
        sigmoid = 1/(1+math.exp(-u)) if u >= 0 else math.exp(u)/(1+math.exp(u))
        return max(x, 0.) + tau*math.log1p(math.exp(-abs(u))), sigmoid
    if name == 'gaussian':
        u = x/tau
        cdf = 0.5*math.erfc(-u/math.sqrt(2))
        pdf = math.exp(-u*u/2)/math.sqrt(2*math.pi)
        return x*cdf+tau*pdf, cdf
    if name == 'rational':
        return (x*x/(x+tau), x*(x+2*tau)/(x+tau)**2) if x > 0 else (0., 0.)
    if name == 'linear_tanh':
        t = math.tanh(x/tau)
        return x+tau*t, 2-t*t
    if name == 'spline_reference':
        # Continuous, positive-slope reference with both signs of curvature.
        if x < -1:
            return -2+0.2*(x+1), 0.2
        if x < 0:
            return 2*x, 2.
        if x < 1:
            return 0.5*x, 0.5
        return 0.5+1.5*(x-1), 1.5
    if name == 'silu_negative_control':
        s = 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))
        return x*s, s+x*s*(1-s)
    if name == 'gelu_negative_control':
        cdf = 0.5*math.erfc(-x/math.sqrt(2))
        pdf = math.exp(-x*x/2)/math.sqrt(2*math.pi)
        return x*cdf, cdf+x*pdf
    raise ValueError(name)


def anchored(name, z, bias):
    return activation(name, bias+z)[0]-activation(name, bias)[0]


def audit(out):
    names = ('relu', 'smooth_hinge', 'leaky', 'softplus', 'gaussian',
             'rational', 'linear_tanh', 'spline_reference', 'silu_negative_control', 'gelu_negative_control')
    records = {}
    for name in names:
        rng = random.Random(20261106)
        errors_z, errors_b = [], []
        for _ in range(1000):
            z, b = rng.uniform(0.01, 8), rng.uniform(-8, 8)
            # Nondifferentiable reference points are deliberately excluded from finite differences.
            eps = 1e-6
            if min(abs(x-knot) for x in (b,b+z) for knot in (-1.,0.,0.125,1.)) < 10*eps:
                continue
            dz = (anchored(name, z+eps, b)-anchored(name, z-eps, b))/(2*eps)
            db = (anchored(name, z, b+eps)-anchored(name, z, b-eps))/(2*eps)
            errors_z.append(abs(dz-activation(name, b+z)[1]))
            errors_b.append(abs(db-(activation(name, b+z)[1]-activation(name, b)[1])))
        grid = [i/128 for i in range(-4096, 4097)]
        values = [activation(name, x)[0] for x in grid]
        rounded = [f32(x) for x in values]
        drops = [a-b for a,b in zip(rounded, rounded[1:]) if b < a]
        biases = [-100., -8., -1., -0., 0., 0.125, 1., 8., 100.]
        identity = all(f32(100-f32(f32(activation(name, f32(b))[0])-
                                          f32(activation(name, f32(b))[0]))) == 100 for b in biases)
        records[name] = {'grid_points': len(grid), 'minimum_derivative': min(activation(name,x)[1] for x in grid),
                         'f32_rounded_grid_decreases': len(drops), 'largest_decrease': max(drops,default=0.),
                         'identity_all_bias_probes': identity,
                         'max_gradient_error_z': max(errors_z), 'max_gradient_error_bias': max(errors_b)}
        if not identity or max(errors_z+errors_b) > 1e-7:
            raise ValueError(f'analytic reference gradient/identity failure: {name}')
        if not name.endswith('negative_control') and (drops or records[name]['minimum_derivative'] < 0):
            raise ValueError(f'positive control lost monotonicity: {name}')
        if name.endswith('negative_control') and not drops:
            raise ValueError(f'negative control failed to detect nonmonotonicity: {name}')
    # A certified max construction: ReLU(u-v)+ReLU(v)=max(u,v), u,v>=0.
    # Dropping only the second term makes increasing v reduce the response.
    independent_mask = [max(1-v,0.) for v in (0.,0.5,1.)]
    tied_mask = [max(1-v,0.)+max(v,0.) for v in (0.,0.5,1.)]
    assert independent_mask[1] < independent_mask[0] and tied_mask == [1.,1.,1.]
    # Freeze a multiset and a shared consolidation tail. Only prefix order changes.
    events = [(family, severity, repeat) for family in range(7) for severity in range(5) for repeat in range(8)]
    rng = random.Random(1101)
    rng.shuffle(events)
    prefix, tail = events[:210], events[210:]
    schedules = {'random': prefix+tail,
                 'easy_first': sorted(prefix,key=lambda x:-x[1])+tail,
                 'hard_first': sorted(prefix,key=lambda x:x[1])+tail,
                 'family_blocks': sorted(prefix,key=lambda x:x[0])+tail}
    for schedule in schedules.values():
        assert Counter(schedule) == Counter(events) and schedule[-70:] == tail
    # Also verify a nonzero floor for a time-varying proposal; this is a sampler
    # arithmetic check, not a claim about learned quality or actual dataset taxonomy.
    replay_weights = [(0.2/7 + (0.8 if family == stage else 0.)) for stage in range(7) for family in range(7)]
    result = {'scope':'independent numerical feasibility; not Rust/f32 runtime qualification or a fit',
              'code_sha256':sha(__file__), 'activations':records,
              'dropout_counterexample':{'independent_mask':independent_mask,'tied_mask':tied_mask},
              'schedule_probe':{'events':len(events),'schedules':list(schedules),
                                'identical_multisets':True,'shared_consolidation_events':len(tail),
                                'minimum_replay_proposal_mass':min(replay_weights)},
              'limitations':['f32-rounded scalar values are not native f32 activation evaluation',
                             'schedule fixtures are not dataset category assignments',
                             'custom activations require native trainer and bake/runtime implementation']}
    write(out,result)
    print(json.dumps(result,indent=2))


def native(plan_path, output, workers):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not 1 <= workers <= 3:
        raise ValueError('pilot concurrency must be 1..3 under the registered aggregate memory cap')
    base = json.loads(plan_path.read_text())
    for path,pin in base['inputs'].items():
        if sha(path) != pin['sha256']:
            raise ValueError(f'baseline input drift: {path}')
    prepared_path = Path(base['output'])/'PREPARED.json'
    prepared = json.loads(prepared_path.read_text())
    if prepared['plan_sha256'] != sha(plan_path):
        raise ValueError('prepared table plan mismatch')
    expected = {'human_kadid','human_tid2013','human_konfig','human_cid22_a25','cid22',
                'human_kadid_development','human_tid2013_development','human_konfig_development','human_cid22_a25_development'}
    if {g['name'] for g in prepared['groups']} != expected:
        raise ValueError('pilot requires exactly the admitted four human populations and CID22 teacher')
    output.mkdir(parents=True,exist_ok=False)
    os.environ.update(REV4_V2_ROOT=base['root'],REV4_V2_BIN_DIR=base['binary_dir'],
                      ZEN_PANEL_BIN=str(Path(base['binary_dir'])/'panel'),
                      RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    from v2_common import acceptance_weight, derived_declaration
    import v2_lodo_mlp as owner
    owner.EPOCHS,owner.PAIRS_PER_EPOCH,owner.LOG_EVERY = 12,5000,1
    groups = []
    for g in prepared['groups']:
        p = Path(g['path'])
        for path,key in [(p,'sha256'),(Path(str(p)+'.manifest.json'),'manifest_sha256'),
                         (p.with_suffix('.keys.parquet'),'keys_sha256')]:
            if sha(path) != g[key]:
                raise ValueError(f'prepared input drift: {path}')
        refs = pq.read_table(p,columns=['ref_basename'])['ref_basename'].to_pylist()
        groups.append((g['name'],p,acceptance_weight(1.,refs) if g['split']=='fit' else 0.,
                       1. if g['split']=='dev' else 0.,'withinref,both' if g['name']=='cid22' else 'withinref,rank'))
    _,_,declaration = derived_declaration('fx2')
    run_plan = {'scope':'short optimization diagnostic, not full-budget model selection',
                'base_plan_sha256':sha(plan_path),'prepared_sha256':sha(prepared_path),
                'script_sha256':sha(__file__),'epochs':12,'pairs_per_epoch':5000,
                'arms':ARMS,'seed_pairs':SEEDS,'workers':workers,
                'binary_sha256':sha(Path(base['binary_dir'])/'zensim_mlp_train')}
    write(output/'PLAN.json',run_plan)

    def cell(arm, index):
        dest = output/arm/f's{index}'
        dest.mkdir(parents=True)
        keep = dest/'keep.txt'
        keep.write_text('\n'.join(map(str,declaration['direct']))+'\n')
        selected = [(n,p,w*({'teacher_half':0.5,'teacher_double':2.}.get(arm,1.) if n=='cid22' else 1.),v,m)
                    for n,p,w,v,m in groups]
        recipe = {**base['recipe'],'shared_hidden_activation':'relu','mse_weight':1}
        cmd = owner.train_command(selected,*SEEDS[index],1853,keep,'N',dest/'best.bin',recipe,
                                  strict_admission=True,data_role_decision=Path(base['output'])/'root/human_role_decision.json')
        if arm == 'walk':
            cmd[cmd.index('--pair-sampling')+1] = 'walk'
        if arm == 'lr_low':
            cmd += ['--lr','0.0003']
        if arm == 'l2_high':
            cmd += ['--l2','0.0001']
        (dest/'ckpt').mkdir()
        cmd += ['--dump-checkpoints-every','11','--dump-checkpoints-dir',str(dest/'ckpt')]
        write(dest/'RUN.json',{'command':cmd,'arm':arm,'seed_index':index,'stage':'training'})
        owner.run(cmd,dest/'train.log')
        curve = owner.read_curve(dest/'train.log')
        if set(curve) != set(range(12)):
            raise ValueError('incomplete native training curve')
        bake = dest/'ckpt/ckpt_epoch011.bin'
        panels = {}
        for n,p,_,v,_ in groups:
            if not v:
                continue
            prediction = owner.predict(bake,p,dest/f'{n}.tsv')
            targets = pq.read_table(p,columns=['human_score','ref_basename'])
            table = dest/f'{n}.panel.parquet'
            pq.write_table(pa.table({'predicted':prediction,'target':targets['human_score'],
                                    'band':targets['ref_basename']}),table,compression='zstd')
            panels[n] = {}
            for mode in ('signed-quality','per-group','scatter'):
                command = [str(Path(base['binary_dir'])/'panel'),'--input',str(table),'--json','--'+mode]
                proc = subprocess.run(command,text=True,capture_output=True)
                (dest/f'{n}.{mode}.log').write_text(proc.stdout+'\n'+proc.stderr)
                proc.check_returncode()
                panels[n][mode] = json.loads(proc.stdout)
            q = panels[n]['signed-quality']
            if q['n'] != len(prediction) or q['n_dropped'] != 0 or panels[n]['scatter']['status'] != 'MEASURED':
                raise ValueError('incomplete panel coverage')
        result = {'arm':arm,'seed_index':index,'init_seed':SEEDS[index][0],'sample_seed':SEEDS[index][1],
                  'bake_sha256':sha(bake),'plan_sha256':sha(output/'PLAN.json'),'panels':panels}
        write(dest/'RESULT.json',result)
        return {'arm':arm,'seed_index':index,'result_sha256':sha(dest/'RESULT.json'),
                'metrics':{n:{'pooled_srocc':p['signed-quality']['srocc_signed'],
                              'within_reference_srocc':p['per-group']['per_group']['mean']} for n,p in panels.items()}}
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(cell,arm,i):(arm,i) for arm in ARMS for i in range(len(SEEDS))}
        for task in as_completed(tasks):
            arm,i = tasks[task]
            try:
                results.append(task.result())
            except Exception as error:
                failures.append({'arm':arm,'seed_index':i,'error':repr(error)})
            record = {'checked_utc':datetime.now(timezone.utc).isoformat(),'required':18,
                      'complete':len(results),'results':results,'failures':failures}
            write(output/'PROGRESS.json',record)
            print(json.dumps({'complete':len(results),'required':18,'failures':failures}),flush=True)
    write(output/'SUMMARY.json',record)
    if failures:
        raise RuntimeError(f'{len(failures)} pilot cells failed; preserved in SUMMARY.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    p = sub.add_parser('audit')
    p.add_argument('--output',type=Path,required=True)
    p = sub.add_parser('native')
    p.add_argument('--baseline-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if args.action == 'audit':
        if args.output.exists():
            raise ValueError('preserve existing audit before rerunning')
        audit(args.output)
    else:
        native(args.baseline_plan,args.output,args.workers)


if __name__ == '__main__':
    main()
