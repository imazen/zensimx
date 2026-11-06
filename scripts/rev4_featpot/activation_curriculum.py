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
ARCHITECTURE_SEEDS = ((1101, 101), (1107, 200000101), (1129, 400000101))
ARCHITECTURE_ARMS = {
    'shared_relu': None,
    'shared_smooth': None,
    'shared_caps': ('caps', [64]),
    'shared_switch': ('switch', [64]),
    'shared_switch_deep': ('switch', [64, 32]),
    'shared_spline': ('spline', [64]),
    'shared_lipschitz': ('lipschitz', [64]),
    'single_ramps': None,
}

LEAKY_ARMS = {'switch_leaky_05': {'negative_slope': .05}, 'switch_leaky_10': {'negative_slope': .1}}
TRAINING_ARMS = {
    'control': {},
    'init_half': {'init_scale': .5}, 'init_double': {'init_scale': 2.},
    'bias_zero': {'bias_scale': 0.}, 'bias_double': {'bias_scale': 2.},
    'l2_none': {'l2': 0.}, 'l2_high': {'l2': 1e-4},
    'adamw_low': {'l2': 0., 'adamw': 1e-5}, 'adamw_high': {'l2': 0., 'adamw': 1e-4},
    'dropout_05': {'dropout': .05}, 'dropout_15': {'dropout': .15},
    'easy_first': {'order': 'easy'}, 'hard_first': {'order': 'hard'},
    'family_blocks': {'order': 'family'}, 'teacher_warm': {'order': 'teacher_warm'},
    'family_balance': {'balance': 'family'}, 'source_family_balance': {'balance': 'source_family'},
    'teacher_zero': {'teacher_mass': 0.}, 'teacher_half': {'teacher_mass': .5}, 'teacher_double': {'teacher_mass': 2.},
    'lr_constant': {'schedule': 'constant'}, 'lr_consolidate': {'schedule': 'consolidate'},
}


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


def native(plan_path, output, workers, *, architecture=False, binary_dir=None, metadata=None, long=False, prepare_only=False, training=False, replay_control=False, leaky=False, stage_replay=False):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    if sum((leaky,replay_control,stage_replay))>1:
        raise ValueError('leaky arms, neutral replay and stage replay are separate registered runs')
    if not 1 <= workers <= (12 if architecture and long else 6 if architecture else 3):
        raise ValueError('pilot concurrency exceeds the registered aggregate memory budget')
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
    source_audit = None
    families = {}
    if architecture:
        if binary_dir is None or metadata is None:
            raise ValueError('architecture fits require explicit frozen binaries and source metadata')
        # Reuse the existing verified keyed join; read no additional peer labels.
        receipt = json.loads(metadata.with_name('PEERS.json').read_text())
        if sha(metadata) != receipt['metadata_sha256']:
            raise ValueError('source metadata drift')
        source = pq.read_table(metadata, columns=['pair_key','ref_pixels_sha256','development','codec']).to_pydict()
        offset, by_role = 0, {'fit':set(), 'dev':set()}
        for g in prepared['groups']:
            keys = pq.read_table(Path(g['path']).with_suffix('.keys.parquet'), columns=['pair_key'])['pair_key'].to_pylist()
            end = offset+len(keys)
            if source['pair_key'][offset:end] != keys or any(v != (g['split']=='dev') for v in source['development'][offset:end]):
                raise ValueError('source metadata order/role mismatch: '+g['name'])
            by_role[g['split']].update(source['ref_pixels_sha256'][offset:end])
            labels=source['codec'][offset:end]
            if any(not isinstance(v,str) or not v for v in labels):
                raise ValueError('missing family label: '+g['name'])
            families[g['name']]=labels
            offset=end
        if offset != len(source['pair_key']) or by_role['fit'] & by_role['dev']:
            raise ValueError('decoded reference overlap or unmatched source rows')
        source_audit = {'metadata_sha256':sha(metadata), 'receipt_sha256':sha(metadata.with_name('PEERS.json')),
                        'rows':offset, 'fit_sources':len(by_role['fit']), 'development_sources':len(by_role['dev']),
                        'decoded_reference_overlap':0}
    selected_bin = Path(binary_dir or base['binary_dir'])
    output.mkdir(parents=True,exist_ok=False)
    driver_source=Path(__file__).read_bytes()
    (output/'DRIVER.py').write_bytes(driver_source)
    os.environ.update(REV4_V2_ROOT=base['root'],REV4_V2_BIN_DIR=str(selected_bin),
                      ZEN_PANEL_BIN=str(selected_bin/'panel'),
                      RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    from v2_common import acceptance_weight, derived_declaration
    import v2_lodo_mlp as owner
    epochs, draws = (120,5000) if training and not replay_control else (120,50000) if long else (12,5000)
    owner.EPOCHS,owner.PAIRS_PER_EPOCH,owner.LOG_EVERY = epochs,draws,1
    training_options = LEAKY_ARMS if leaky else TRAINING_ARMS
    arms = (('control',) if replay_control else tuple(training_options)) if training else tuple(ARCHITECTURE_ARMS) if architecture else ARMS
    if stage_replay:arms=('control','easy_first','hard_first','family_blocks','teacher_warm')
    seeds = ARCHITECTURE_SEEDS if architecture else SEEDS
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
    probe = None
    if architecture:
        required=sorted(set(declaration['direct']) | {v for pair in declaration['products'] for v in pair})
        # Feature-space diagnostics, not generated images or admitted training data.
        fit_table=next(p for _,p,w,_,_ in groups if w>0)
        fit=pq.read_table(fit_table,columns=[f'f{i}' for i in required])
        median=np.array([np.median(fit[f'f{i}'].to_numpy()) for i in required],dtype=np.float32)
        if not np.isfinite(median).all() or (median<0).any():
            raise ValueError('nonnegative feature probe assumption failed')
        median=np.maximum(median,np.float32(1e-6))
        random=np.random.default_rng(20261106)
        small=(random.uniform(.01,2,(128,len(required)))*median).astype(np.float32)
        larger=small.copy()
        columns=random.integers(len(required),size=128)
        larger[np.arange(128),columns]+=median[columns]*np.float32(.25)
        ray=median[None,:]*np.array([0,1e-8,1e-6,1e-4,.01,.1,1],dtype=np.float32)[:,None]
        values=np.concatenate([small,larger,ray])
        fields={f'f{i}':pa.array(values[:,required.index(i)] if i in required else np.zeros(len(values),dtype=np.float32)) for i in range(1853)}
        fields.update(human_score=pa.array(np.zeros(len(values))),ref_basename=pa.array(['feature_probe']*len(values)))
        probe=output/'feature-probes.parquet'
        pq.write_table(pa.table(fields),probe,compression='zstd')
        write(Path(str(probe)+'.manifest.json'),dict(formula_revision=7,scope='feature-space diagnostic only',rows=len(values)))
        write(output/'PROBES.json',dict(table_sha256=sha(probe),fit_table_sha256=sha(fit_table),
              seed=20261106,pairs=128,ray_scales=[0,1e-8,1e-6,1e-4,.01,.1,1],required_raw_ids=required,
              note='raw nonnegative coordinates; derived products recomputed by native prediction'))
    run_plan = {'scope':'fixed-budget architecture development comparison' if architecture else 'short optimization diagnostic, not full-budget model selection',
                'base_plan_sha256':sha(plan_path),'prepared_sha256':sha(prepared_path),
                'script_sha256':hashlib.sha256(driver_source).hexdigest(),'epochs':epochs,'pairs_per_epoch':draws,
                'arms':arms,'seed_pairs':seeds,'workers':workers,'source_audit':source_audit,
                'selection':'fixed final epoch; no development selection',
                'binary_sha256':{name:sha(selected_bin/name) for name in ('zensim_mlp_train','bake_dial_refit','panel')},
                'architecture':architecture,'training':training,'training_options':training_options if training else None,
                'stage_replay':stage_replay,'checkpoint_interval':1 if stage_replay else epochs-1,
                'family_taxonomy':{n:sorted(set(v)) for n,v in families.items()},'per_training_child_memory_cap':'2G' if architecture else None}
    write(output/'PLAN.json',run_plan)
    if prepare_only:
        print(json.dumps({'stage':'preparation verified; zero optimizer steps','source_audit':source_audit}),flush=True)
        return

    def cell(arm, index):
        dest = output/arm/f's{index}'
        dest.mkdir(parents=True)
        keep = dest/'keep.txt'
        keep.write_text('\n'.join(map(str,declaration['direct']))+'\n')
        selected = [(n,p,w*({'teacher_half':0.5,'teacher_double':2.}.get(arm,1.) if n=='cid22' and not training else 1.),v,m)
                    for n,p,w,v,m in groups]
        recipe = {**base['recipe'],'shared_hidden_activation':'relu','mse_weight':1}
        if arm == 'shared_smooth':
            recipe['shared_hidden_activation']='smooth-hinge'
        if arm == 'single_ramps':
            recipe.pop('shared_hidden_activation')
            recipe['ramps']=16
        cmd = owner.train_command(selected,*seeds[index],1853,keep,'N',dest/'best.bin',recipe,
                                  strict_admission=True,data_role_decision=Path(base['output'])/'root/human_role_decision.json')
        if arm == 'walk':
            cmd[cmd.index('--pair-sampling')+1] = 'walk'
        if arm == 'lr_low':
            cmd += ['--lr','0.0003']
        if arm == 'l2_high' and not training:
            cmd += ['--l2','0.0001']
        if training:
            cmd += ['--l2',str(training_options[arm].get('l2',1e-5)),'--no-sample-coverage']
        (dest/'ckpt').mkdir()
        cmd += ['--dump-checkpoints-every',str(1 if stage_replay else epochs-1),'--dump-checkpoints-dir',str(dest/'ckpt')]
        env = os.environ.copy()
        for key in ('ZENSIM_MONOTONE_RESEARCH_MANIFEST','ZENSIM_MONOTONE_RESEARCH_SHA256'):
            env.pop(key,None)
        manifest = None
        if architecture and (training or ARCHITECTURE_ARMS[arm] is not None):
            family,widths=('switch',[64]) if training else ARCHITECTURE_ARMS[arm]
            manifest=dest/'HEAD.json'
            head_config=dict(schema='monotone-head-v1',family=family,widths=widths,
                init_seed=seeds[index][0],sample_seed=seeds[index][1],epochs=epochs,
                pairs_per_epoch=draws,first_width=recipe['hidden'],max_pairs=recipe['max_pairs'])
            if training:
                opts={k:v for k,v in training_options[arm].items() if k not in ('l2','negative_slope')}
                if 'negative_slope' in training_options[arm]:
                    head_config['negative_slope']=training_options[arm]['negative_slope']
                head_config['training']=dict(schema='monotone-training-v1',init_scale=1.,bias_scale=1.,
                    dropout=0.,dropout_seed=9000001+index*100000,adamw=0.,schedule='cosine',order='random',
                    teacher_mass=1.,balance='none',groups=[dict(name=n,families=[sorted(set(families[n])).index(v) for v in families[n]]) for n,_,w,_,_ in groups if w>0])
                head_config['training'].update(opts)
            write(manifest,head_config)
            env.update(ZENSIM_MONOTONE_RESEARCH_MANIFEST=str(manifest),ZENSIM_MONOTONE_RESEARCH_SHA256=sha(manifest))
        launch = ([str(Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','2G','--jobs','1','--']+cmd) if architecture else cmd
        with (dest/'train.log').open('w') as log:
            proc=subprocess.Popen(launch,stdout=log,stderr=subprocess.STDOUT,env=env)
            write(dest/'RUN.json',{'command':cmd,'launch':launch,'arm':arm,'seed_index':index,'stage':'training',
                'pid':proc.pid,'head_manifest_sha256':sha(manifest) if manifest else None})
            rc=proc.wait()
        if rc:
            raise RuntimeError(f'native training rc={rc}: {dest}/train.log')
        curve = owner.read_curve(dest/'train.log')
        if set(curve) != set(range(epochs)):
            raise ValueError('incomplete native training curve')
        bake = dest/f'ckpt/ckpt_epoch{epochs-1:03d}.bin'
        panels = {}
        for n,p,_,v,_ in groups:
            if not v and not architecture:
                continue
            prediction = owner.predict(bake,p,dest/f'{n}.tsv')
            targets = pq.read_table(p,columns=['human_score','ref_basename'])
            table = dest/f'{n}.panel.parquet'
            pq.write_table(pa.table({'predicted':prediction,'target':targets['human_score'],
                                    'band':targets['ref_basename']}),table,compression='zstd')
            panels[n] = {}
            for mode in ('signed-quality','per-group','scatter'):
                command = [str(selected_bin/'panel'),'--input',str(table),'--json','--'+mode]
                proc = subprocess.run(command,text=True,capture_output=True)
                (dest/f'{n}.{mode}.log').write_text(proc.stdout+'\n'+proc.stderr)
                proc.check_returncode()
                panels[n][mode] = json.loads(proc.stdout)
            if training:
                family_table=dest/f'{n}.families.parquet'
                pq.write_table(pa.table({'predicted':prediction,'target':targets['human_score'],'band':families[n]}),family_table,compression='zstd')
                proc=subprocess.run([str(selected_bin/'panel'),'--input',str(family_table),'--json','--per-group'],text=True,capture_output=True,check=True)
                panels[n]['families']=json.loads(proc.stdout)
                (dest/f'{n}.families.log').write_text(proc.stdout+'\n'+proc.stderr)
            q = panels[n]['signed-quality']
            if q['n'] != len(prediction) or q['n_dropped'] != 0 or panels[n]['scatter']['status'] != 'MEASURED':
                raise ValueError('incomplete panel coverage')
        probes=None
        if probe is not None:
            prediction=owner.predict(bake,probe,dest/'feature-probes.tsv')
            delta=prediction[128:256]-prediction[:128]
            probes=dict(identity=float(prediction[256]),near_identity_ray=prediction[256:].tolist(),
                        coordinate_pairs=128,positive_deltas=int(np.sum(delta>0)),
                        violations_over_1e_5=int(np.sum(delta>1e-5)),max_increase=float(max(0.,delta.max())),
                        ceiling_violations=int(np.sum(prediction>100.)))
            write(dest/'PROBES.json',probes)
            if probes['identity']!=100. or probes['violations_over_1e_5'] or probes['ceiling_violations']:
                raise ValueError(f'native feature gate failed: {dest}/PROBES.json')
        result = {'arm':arm,'seed_index':index,'init_seed':seeds[index][0],'sample_seed':seeds[index][1],
                  'bake_sha256':sha(bake),'plan_sha256':sha(output/'PLAN.json'),'panels':panels}
        if probes is not None:
            result['feature_probes']=probes
        write(dest/'RESULT.json',result)
        return {'arm':arm,'seed_index':index,'result_sha256':sha(dest/'RESULT.json'),
                'metrics':{n:{'pooled_srocc':p['signed-quality']['srocc_signed'],
                              'within_reference_srocc':p['per-group']['per_group']['mean']} for n,p in panels.items()}}
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(cell,arm,i):(arm,i) for arm in arms for i in range(len(seeds))}
        for task in as_completed(tasks):
            arm,i = tasks[task]
            try:
                results.append(task.result())
            except Exception as error:
                failures.append({'arm':arm,'seed_index':i,'error':repr(error)})
            record = {'checked_utc':datetime.now(timezone.utc).isoformat(),'required':len(tasks),
                      'complete':len(results),'results':results,'failures':failures}
            write(output/'PROGRESS.json',record)
            print(json.dumps({'complete':len(results),'required':len(tasks),'failures':failures}),flush=True)
    write(output/'SUMMARY.json',record)
    if failures:
        raise RuntimeError(f'{len(failures)} pilot cells failed; preserved in SUMMARY.json')


def replay_controls(frozen, binaries, output):
    """Replay all three frozen native compatibility cells; only paths/provenance differ."""
    original=json.loads((frozen/'RESULT.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    results={}
    for arm in ('new-single','deep-relu','deep-smooth'):
        dest=output/arm
        (dest/'ckpt').mkdir(parents=True)
        cmd=json.loads((frozen/arm/'command.json').read_text())
        cmd[0]=str(binaries/'zensim_mlp_train')
        cmd[cmd.index('--out')+1]=str(dest/'best.bin')
        cmd[cmd.index('--dump-checkpoints-dir')+1]=str(dest/'ckpt')
        env=os.environ.copy()
        env.update(RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(Path.home()/'tmp'))
        for key in ('ZENSIM_MONOTONE_RESEARCH_MANIFEST','ZENSIM_MONOTONE_RESEARCH_SHA256'):
            env.pop(key,None)
        write(dest/'COMMAND.json',cmd)
        with (dest/'train.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        stripped=dest/'without-repro.bin'
        with (dest/'strip.log').open('w') as log:
            subprocess.run([str(binaries/'bake_dial_refit'),'strip','--in',str(dest/'ckpt/ckpt_epoch001.bin'),
                '--out',str(stripped),'--key','zentrain.repro'],stdout=log,stderr=subprocess.STDOUT,check=True)
        actual=sha(stripped)
        expected=original['results'][arm]['without_repro_sha256']
        results[arm]=dict(expected=expected,actual=actual,match=actual==expected)
        write(output/'REPLAY.json',results)
        if actual!=expected:
            raise ValueError('baseline replay changed: '+arm)
    print(json.dumps(results),flush=True)


def compare_neutral(reference, candidate, binary, output, epoch=11, strip_research=True):
    output.mkdir(parents=True,exist_ok=False)
    result=[]
    for index in range(3):
        hashes=[]
        for label,root in [('reference',reference),('candidate',candidate)]:
            original=root/f's{index}'/f'ckpt/ckpt_epoch{epoch:03d}.bin'
            current=original
            for step,key in enumerate(('zentrain.repro','zentrain.monotone_research','zentrain.sample_coverage')):
                if key=='zentrain.monotone_research' and not strip_research:continue
                if key=='zentrain.sample_coverage' and '--no-sample-coverage' in json.loads((root/f's{index}'/'RUN.json').read_text())['command']:
                    continue
                destination=output/f'{label}-{index}-{step}.bin'
                with (output/f'{label}-{index}-{step}.log').open('w') as log:
                    subprocess.run([str(binary),'strip','--in',str(current),'--out',str(destination),'--key',key],stdout=log,stderr=subprocess.STDOUT,check=True)
                current=destination
            hashes.append(sha(current))
        result.append(dict(seed_index=index,reference=hashes[0],candidate=hashes[1],match=hashes[0]==hashes[1]))
    write(output/'COMPARISON.json',result)
    if not all(r['match'] for r in result):
        raise ValueError('neutral control changes native weights/scaler or retained metadata')
    print(json.dumps(result),flush=True)


def family_panels(root, prepared_path, metadata, binary, output):
    import numpy as np
    import pyarrow.parquet as pq
    os.environ.update(ZEN_PANEL_BIN=str(binary),TMPDIR=str(Path.home()/'tmp'))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from lib.zen_stats import panel_batch_indexed
    plan=json.loads((root/'PLAN.json').read_text())
    complete=json.loads((root/'SUMMARY.json').read_text())
    if complete['complete']!=complete['required'] or complete['failures']:
        raise ValueError('signed breakdowns require a completed, successful study')
    if sha(prepared_path)!=plan['prepared_sha256'] or sha(metadata)!=plan['source_audit']['metadata_sha256']:
        raise ValueError('family metadata/preparation drift')
    prepared=json.loads(prepared_path.read_text())
    meta=pq.read_table(metadata,columns=['pair_key','codec','ref_pixels_sha256']).to_pydict()
    offset=0;groups={}
    for group in prepared['groups']:
        keys_path=Path(group['path']).with_suffix('.keys.parquet')
        if sha(keys_path)!=group['keys_sha256']:raise ValueError('family key drift')
        keys=pq.read_table(keys_path,columns=['pair_key'])['pair_key'].to_pylist();end=offset+len(keys)
        if meta['pair_key'][offset:end]!=keys:raise ValueError('family key order drift')
        groups[group['name']]={k:np.array(meta[k][offset:end]) for k in ('codec','ref_pixels_sha256')};offset=end
    if offset!=len(meta['pair_key']):raise ValueError('unconsumed metadata')
    output.mkdir(parents=True,exist_ok=False)
    records={}
    for path in sorted(root.glob('*/s*/RESULT.json')):
        result=json.loads(path.read_text());dest=output/path.parent.relative_to(root);dest.mkdir(parents=True)
        panels={};sources={};source_families={};table_hashes={}
        for name,labels in groups.items():
            table=path.parent/(name+'.panel.parquet')
            data=pq.read_table(table).to_pydict();pred=np.array(data['predicted']);target=np.array(data['target'])
            if len(pred)!=len(labels['codec']):raise ValueError('family prediction length')
            jobs=[];descriptions={}
            for scope in ('family','source','source_family'):
                keys=(list(labels['codec']) if scope=='family' else list(labels['ref_pixels_sha256'])
                      if scope=='source' else list(zip(labels['codec'],labels['ref_pixels_sha256'])))
                buckets={}
                for i,key in enumerate(keys):buckets.setdefault(key,[]).append(i)
                for key,ids in sorted(buckets.items()):
                    label=str(len(jobs));jobs.append((label,'p','t',ids));descriptions[label]=(scope,key)
            rows=panel_batch_indexed({'p':pred,'t':target},jobs,stats='srocc')
            if any(r['n_dropped'] for r in rows):raise ValueError('family panel dropped rows')
            grouped={'family':[],'source':[],'source_family':[]}
            for row in rows:
                scope,key=descriptions[row['label']]
                grouped[scope].append(dict(label=key,n=row['n'],srocc_signed=row['srocc_signed'] if np.isfinite(row['srocc_signed']) else None,
                    status='measured' if np.isfinite(row['srocc_signed']) else 'undefined correlation'))
            panels[name]=grouped['family'];sources[name]=grouped['source'];source_families[name]=grouped['source_family']
            table_hashes[name]=sha(table)
        write(dest/'FAMILIES.json',dict(schema='signed-source-family-v1',source_result_sha256=sha(path),panel_binary_sha256=sha(binary),
            prediction_table_sha256=table_hashes,panels=panels,sources=sources,source_families=source_families))
        records[str(path.parent.relative_to(root))]=sha(dest/'FAMILIES.json')
    if len(records)!=complete['required']:raise ValueError('signed breakdown cell count mismatch')
    write(output/'INDEX.json',dict(plan_sha256=sha(root/'PLAN.json'),metadata_sha256=sha(metadata),cells=records))
    print(json.dumps(dict(family_cells=len(records),index_sha256=sha(output/'INDEX.json'))))


def capacity(root, binary, output):
    results={}
    for p in sorted(root.glob('*/s*/RESULT.json')):
        data=json.loads(p.read_text());plan=json.loads((root/'PLAN.json').read_text())
        model=p.parent/f"ckpt/ckpt_epoch{plan['epochs']-1:03d}.bin"
        if sha(model)!=data['bake_sha256']: raise ValueError('capacity model drift')
        info=json.loads(subprocess.check_output([str(binary),'inspect','--in',str(model)],text=True))
        command=json.loads(p.with_name('RUN.json').read_text())['command']
        mp=int(command[command.index('--max-pairs')+1])
        ramps=int(command[command.index('--monotone-ramps')+1]) if '--monotone-ramps' in command else 0
        nf,nh=info['layers'][0]['in_dim'],info['layers'][0]['out_dim']
        first=nf*(nh-ramps)+nh-mp
        if p.with_name('HEAD.json').exists():
            head=json.loads(p.with_name('HEAD.json').read_text());ni=nh-mp;count=first
            for width in head['widths']:
                family=head['family']
                count+=ni*width
                if family in ('switch','caps','lipschitz'): count+=width
                if family=='caps':count+=width//2
                if family=='spline':count+=width*6
                if family=='lipschitz':count+=width
                ni=1 if family=='lipschitz' else width
            count+=ni
        elif len(info['layers'])==3:
            width=info['layers'][1]['out_dim'];count=first+(nh-mp)*width+2*width
        elif len(info['layers'])==2:
            count=first+nh-mp-ramps
        else:raise ValueError('unsupported capacity parameterization')
        info['independent_trainable_coefficients']=count
        info['trainable_count_scope']='nominal coefficients after exact ties, frozen biases and unused cap thresholds; excludes scaler/fixed routing; boundary-zero weights remain trainable'
        results[str(p.parent.relative_to(root))]=info
    write(output,dict(inspector_sha256=sha(binary),cells=results))
    print(json.dumps({name:v['independent_trainable_coefficients'] for name,v in results.items()}))


def input_closure(root, plan_path, metadata):
    import shutil
    plan=json.loads(plan_path.read_text())
    pin=json.loads((root/'screen/PLAN.json').read_text())
    if sha(plan_path)!=pin['base_plan_sha256']: raise ValueError('baseline plan drift')
    prepared_path=Path(plan['output'])/'PREPARED.json'
    if sha(prepared_path)!=pin['prepared_sha256']: raise ValueError('prepared receipt drift')
    prepared=json.loads(prepared_path.read_text())
    wanted={p:v['sha256'] for p,v in plan['inputs'].items()}
    wanted.update({str(plan_path):sha(plan_path),str(prepared_path):sha(prepared_path),
                   str(metadata):pin['source_audit']['metadata_sha256'],str(metadata.with_name('PEERS.json')):pin['source_audit']['receipt_sha256']})
    for g in prepared['groups']:
        p=Path(g['path'])
        wanted.update({str(p):g['sha256'],str(p)+'.manifest.json':g['manifest_sha256'],
                       str(p.with_suffix('.keys.parquet')):g['keys_sha256']})
    role=Path(plan['output'])/'root/human_role_decision.json'
    wanted[str(role)]=sha(role)
    dest=root/'input-closure';dest.mkdir(exist_ok=False)
    records=[]
    for path,expected in sorted(wanted.items()):
        if sha(path)!=expected: raise ValueError('input closure drift: '+path)
        target=dest/expected
        if not target.exists(): shutil.copyfile(path,target)
        if sha(target)!=expected: raise ValueError('closure copy mismatch')
        records.append(dict(source=path,sha256=expected,bytes=target.stat().st_size,blob=expected))
    write(dest/'INDEX.json',records)
    print(json.dumps(dict(files=len(records),unique_blobs=len(set(wanted.values())),index_sha256=sha(dest/'INDEX.json'))))


def qualify_receipts(root, binary, output, watch=False):
    """Preserve historical bakes; remove invalid legacy reconstruction from final copies."""
    import time
    output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((root/'PLAN.json').read_text())
    required=len(plan['arms'])*len(plan['seed_pairs'])
    records={}
    while True:
        for p in sorted(root.glob('*/s*/RESULT.json')):
            relative=p.parent.relative_to(root)
            if str(relative) in records: continue
            result=json.loads(p.read_text())
            source=p.parent/f"ckpt/ckpt_epoch{plan['epochs']-1:03d}.bin"
            if sha(source)!=result['bake_sha256']: raise ValueError('changed source checkpoint')
            events=[json.loads(line.split(' ',1)[1]) for line in p.with_name('train.log').read_text().splitlines() if line.startswith('TRAINING_EVENTS ')]
            if len(events)!=1 or events[0]['events']!=plan['epochs']*plan['pairs_per_epoch']:
                raise ValueError('missing exact full-budget event receipt')
            command=json.loads(p.with_name('RUN.json').read_text())['command']
            omitted='--no-sample-coverage' in command
            if omitted and '[coverage] zentrain.sample_coverage SKIPPED (--no-sample-coverage)' not in p.with_name('train.log').read_text():
                raise ValueError('coverage omission lacks native log receipt')
            dest=output/relative;dest.mkdir(parents=True)
            receipt=dict(scope='full scheduled draw stream, including skips; not accepted-gradient or row-coverage statistics',
                raw_bake_sha256=sha(source),head_manifest_sha256=sha(p.with_name('HEAD.json')),
                training_log_sha256=sha(p.with_name('train.log')),events=events[0],
                correction='legacy entry not emitted; actual native event receipt appended' if omitted else
                    'legacy uniform reconstruction removed; actual native event receipt retained')
            write(dest/'EVENTS.json',receipt)
            def call(name,args):
                with (dest/(name+'.log')).open('w') as log:
                    subprocess.run([str(binary),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
            base=source
            if not omitted:
                base=dest/'without-legacy.bin'
                call('remove-legacy',['strip','--in',str(source),'--out',str(base),'--key','zentrain.sample_coverage'])
            call('append-actual',['append-meta','--in',str(base),'--out',str(dest/'final.bin'),
                 '--key','zentrain.training_events','--value-file',str(dest/'EVENTS.json')])
            call('verify-sections',['strip','--in',str(dest/'final.bin'),'--out',str(dest/'roundtrip.bin'),'--key','zentrain.training_events'])
            if sha(dest/'roundtrip.bin')!=sha(base):
                raise ValueError('receipt correction changed retained model sections')
            records[str(relative)]={'raw_sha256':sha(source),'qualified_sha256':sha(dest/'final.bin'),
                'retained_sections_identical':True,'events_sha256':sha(dest/'EVENTS.json')}
            write(output/'QUALIFICATION.json',dict(required=required,complete=len(records),cells=records))
            print(json.dumps(dict(qualified=len(records),required=required,cell=str(relative))),flush=True)
        if len(records)==required or not watch: break
        if (root/'SUMMARY.json').exists():
            raise ValueError('run terminated without every required result; preserve qualification record')
        time.sleep(5)


def progress(root):
    import re
    cells=[]
    for p in sorted(root.glob('*/**/RUN.json')):
        run=json.loads(p.read_text())
        log=p.with_name('train.log').read_text()
        epochs=re.findall(r'epoch\s+(\d+)\s*\|',log)
        proc=Path('/proc')/str(run['pid'])/'cmdline'
        command=proc.read_bytes().replace(b'\x00',b' ').decode(errors='replace') if proc.exists() else ''
        alive='run-heavy' in command and str(p.parent) in command
        cells.append(dict(cell=str(p.parent.relative_to(root)),pid=run['pid'],verified_wrapper_alive=alive,
            final_result=p.with_name('RESULT.json').exists(),last_epoch=int(epochs[-1]) if epochs else None,
            wrapper_exit=re.findall(r'run-heavy: done rc=(\d+)',log)))
    stages={}
    for p in sorted(root.glob('*/PLAN.json')):
        plan=json.loads(p.read_text())
        if 'arms' not in plan or 'seed_pairs' not in plan:continue
        name=p.parent.name;selected=[c for c in cells if c['cell'].startswith(name+'/')]
        required=len(plan['arms'])*len(plan['seed_pairs'])
        progress_path=p.with_name('PROGRESS.json')
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
        stages[name]=dict(required=required,complete=sum(c['final_result'] for c in selected),
            running=sum(c['verified_wrapper_alive'] for c in selected),not_started=required-len(selected),
            stopped_without_result=[c['cell'] for c in selected if not c['verified_wrapper_alive'] and not c['final_result']],
            recorded_failures=progress.get('failures',[]),plan_sha256=sha(p),binary_sha256=plan['binary_sha256'])
    report=dict(checked_utc=datetime.now(timezone.utc).isoformat(),stages=stages,cells=cells)
    write(root/'SUPERVISION.json',report)
    print(json.dumps(report),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    p = sub.add_parser('audit')
    p.add_argument('--output',type=Path,required=True)
    p = sub.add_parser('native')
    p.add_argument('--baseline-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    p = sub.add_parser('architectures')
    p.add_argument('--baseline-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--long',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p=sub.add_parser('training')
    p.add_argument('--baseline-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--replay-control',action='store_true')
    p.add_argument('--leaky-switches',action='store_true')
    p.add_argument('--stage-replay',action='store_true')
    p=sub.add_parser('compare-neutral')
    for name in ('reference','candidate','binary','output'): p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('families')
    for name in ('root','prepared','metadata','binary','output'):p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('capacity')
    for name in ('root','binary','output'): p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('closure')
    for name in ('root','baseline-plan','metadata'): p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('qualify-receipts')
    for name in ('root','binary','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    p=sub.add_parser('progress')
    p.add_argument('--root',type=Path,required=True)
    p=sub.add_parser('replay')
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'audit':
        if args.output.exists():
            raise ValueError('preserve existing audit before rerunning')
        audit(args.output)
    elif args.action == 'families':
        family_panels(args.root,args.prepared,args.metadata,args.binary,args.output)
    elif args.action == 'capacity':
        capacity(args.root,args.binary,args.output)
    elif args.action == 'closure':
        input_closure(args.root,args.baseline_plan,args.metadata)
    elif args.action == 'qualify-receipts':
        qualify_receipts(args.root,args.binary,args.output,args.watch)
    elif args.action == 'progress':
        progress(args.root)
    elif args.action == 'compare-neutral':
        compare_neutral(args.reference,args.candidate,args.binary,args.output)
    elif args.action == 'replay':
        replay_controls(args.frozen,args.binary_dir,args.output)
    elif args.action == 'training':
        native(args.baseline_plan,args.output,args.workers,architecture=True,training=True,
               binary_dir=args.binary_dir,metadata=args.metadata,replay_control=args.replay_control,leaky=args.leaky_switches,stage_replay=args.stage_replay)
    elif args.action == 'architectures':
        native(args.baseline_plan,args.output,args.workers,architecture=True,
               binary_dir=args.binary_dir,metadata=args.metadata,long=args.long,prepare_only=args.prepare_only)
    else:
        native(args.baseline_plan,args.output,args.workers)


if __name__ == '__main__':
    main()
