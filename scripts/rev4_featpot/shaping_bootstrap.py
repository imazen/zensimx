"""Primary outer source-bootstrap runner; admission precedes every score read."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from shaping_completion import FAMILIES,ROLES,encoded,seeds,sha
from shaping_outer_evidence import collect,collect_matched,primary_cells
from shaping_bootstrap_plan import SEED,REPLICATES,population_groups,component_draws
from shaping_bootstrap_stats import paired_stratum
from shaping_bootstrap_reduce import POPULATIONS,reduce


def verify_draws(registration,draws,partitions,populations,components):
    if registration['seed']!=SEED or registration['replicates']!=REPLICATES or registration['scores_read']:
        raise ValueError('bootstrap registration settings differ')
    expected=[];rng=np.random.default_rng(SEED);seen=set()
    for outer in range(3):
        groups=population_groups(partitions[f'outer-{outer}']['test'],populations,components)
        if tuple(groups)!=POPULATIONS:raise ValueError('bootstrap populations differ')
        current={c for group in groups.values() for c in group}
        if current&seen:raise ValueError('bootstrap outer sources overlap')
        seen|=current
        for pi,(population,group) in enumerate(groups.items()):
            key=f'outer{outer}_population{pi}'
            expected.append(dict(outer=outer,population=population,array=key,groups=group,
                source_components=len(group),rows=sum(map(len,group.values()))))
            actual=draws[key];reference=component_draws(group,rng)
            if actual.dtype!=reference.dtype or not np.array_equal(actual,reference):
                raise ValueError('registered bootstrap draws cannot be reproduced')
    if registration['strata']!=expected or set(draws)!=set(s['array'] for s in expected):
        raise ValueError('registered bootstrap source membership differs')


def compute(outputs,registration,draws,labels,progress=lambda *args:None):
    reduced=any('role' in row for row in outputs)
    families=ROLES if reduced else FAMILIES
    if reduced:
        expected={(r,f'outer-{o}',s) for r in ROLES for o in range(3) for s in seeds(r)}
        actual=[(row.get('role'),row['cell']['partition'],row['cell']['seed']) for row in outputs]
        if len(actual)!=48 or set(actual)!=expected:
            raise ValueError('complete reduced role/fold/seed coverage required')
    else:primary_cells([row['cell'] for row in outputs])
    samples={f:np.full((3,len(tuple(seeds(f))),5,2,REPLICATES),np.nan) for f in families}
    points={f:np.full((3,len(tuple(seeds(f))),5,2),np.nan) for f in families}
    for stratum in registration['strata']:
        outer=stratum['outer'];population=stratum['population'];pi=POPULATIONS.index(population)
        selected=[r for r in outputs if r['cell']['partition']==f'outer-{outer}']
        rows=[row for group in stratum['groups'].values() for row in group]
        positions={row:i for i,row in enumerate(rows)}
        groups=[np.array([positions[row] for row in group]) for group in stratum['groups'].values()]
        predictions=[]
        for row in selected:
            lookup={original:i for i,original in enumerate(row['indices'])}
            predictions.append(row['predictions'][[lookup[i] for i in rows]])
        result=paired_stratum(predictions,labels[rows],groups,draws[stratum['array']])
        for k,row in enumerate(selected):
            family=row['role'] if reduced else row['cell']['family'];seed=row['cell']['seed'];si=0 if seed is None else seed
            summary=row['report']['outer_test']['summary'][population]
            expected=np.array([summary['pooled'],summary['within']],dtype=float)
            if not np.isfinite(expected).all() or not np.allclose(result['point'][k],expected,rtol=0,atol=1e-9):
                raise ValueError('bootstrap point ranks differ from completed report')
            samples[family][outer,si,pi]=np.stack((result['pooled'][k],result['within'][k]))
            points[family][outer,si,pi]=result['point'][k]
        progress(stratum,result)
    return samples,points,reduce(samples,points,families=families)



def compute_matched(arms,registration,draws,labels,seed_ids,progress=lambda *args:None):
    """Matched outer arms; caller must verify source/model/admission evidence."""
    from shaping_bootstrap_reduce import reduce_matched
    seed_ids=tuple(seed_ids)
    if set(arms)!={'candidate','control'} or seed_ids not in ((0,1,2),(None,)):
        raise ValueError('complete registered matched arms and seeds required')
    expected={(f'outer-{fold}',seed) for fold in range(3) for seed in seed_ids}
    for rows in arms.values():
        coverage=[(r['cell']['partition'],r['cell']['seed']) for r in rows]
        if set(coverage)!=expected or len(coverage)!=len(expected):raise ValueError('incomplete or duplicate matched fold/seed coverage')
    samples={k:np.full((3,len(seed_ids),5,2,REPLICATES),np.nan) for k in arms}
    points={k:np.full((3,len(seed_ids),5,2),np.nan) for k in arms}
    for stratum in registration['strata']:
        outer=stratum['outer'];population=stratum['population'];pi=POPULATIONS.index(population)
        selected=[(key,row) for key,rows in arms.items() for row in rows if row['cell']['partition']==f'outer-{outer}']
        rows=[i for group in stratum['groups'].values() for i in group];positions={row:i for i,row in enumerate(rows)}
        groups=[np.array([positions[i] for i in group]) for group in stratum['groups'].values()]
        predictions=[]
        for _,row in selected:
            lookup={original:i for i,original in enumerate(row['indices'])}
            if len(lookup)!=len(row['indices']) or not set(rows)<=set(lookup):raise ValueError('matched outer source rows differ')
            predictions.append(row['predictions'][[lookup[i] for i in rows]])
        result=paired_stratum(predictions,labels[rows],groups,draws[stratum['array']])
        for k,(key,row) in enumerate(selected):
            si=seed_ids.index(row['cell']['seed']);summary=row['report']['outer_test']['summary'][population]
            point=np.array([summary['pooled'],summary['within']],dtype=float)
            if not np.isfinite(point).all() or not np.allclose(result['point'][k],point,rtol=0,atol=1e-9):
                raise ValueError('matched bootstrap point differs from completed report')
            samples[key][outer,si,pi]=np.stack((result['pooled'][k],result['within'][k]));points[key][outer,si,pi]=result['point'][k]
        progress(stratum,result)
    return samples,points,reduce_matched(samples,points,seed_ids)


def admit(args,*,host=None):
    """Reproduce the complete evidence gate before bootstrap computation."""
    from shaping_reduced_scope import admit_compute
    admit_compute(args.study,host=host)
    comparison=getattr(args,'comparison',None);spec=None
    if comparison is None:
        plan,partitions,outputs,pins,qualifications=collect(args.study)
    else:
        from shaping_bootstrap_plan import load_comparison
        spec,comparison_pins=load_comparison(args.study,comparison)
        plan,partitions,arms,pins,qualifications,seed_ids=collect_matched(args.study,**spec)
        outputs=[row for rows in arms.values() for row in rows]
        policy=Path(__file__).resolve().parents[2]/'docs/SHAPING_MATCHED_UNCERTAINTY.md'
        pins.update(comparison_pins)
    if args.output.parent.stat().st_dev!=args.study.stat().st_dev:
        raise ValueError('active bootstrap output must share the study SSD')
    for name in ('shaping_bootstrap.py','shaping_bootstrap_plan.py','shaping_bootstrap_stats.py',
                 'shaping_bootstrap_reduce.py','shaping_outer_evidence.py','shaping_prediction_evidence.py'):
        path=Path(__file__).with_name(name);pins[str(path)]=sha(path)
    registration_path=args.draws/'REGISTRATION.json';registration=json.loads(registration_path.read_text())
    if registration['plan_sha256']!=sha(args.study/'PLAN.json'):raise ValueError('draw registration study differs')
    draw_path=args.draws/'draws.npz'
    if sha(draw_path)!=registration['draws_sha256']:raise ValueError('registered draw file drift')
    pins.update({str(registration_path):sha(registration_path),str(draw_path):sha(draw_path)})
    for name,pin in registration['input_sha256'].items():
        if plan['input_pins'].get(name)!=pin or sha(name)!=pin:raise ValueError('bootstrap metadata drift')
        pins[name]=pin
    from shaped_linear_study import OLD
    paths={'y':OLD/'data/y.npy','pop':OLD/'data/pop.npy',
           'component':Path(plan['original_plan']).parent/'data/component.npy'}
    for path in paths.values():
        pin=sha(path)
        if plan['input_pins'].get(str(path))!=pin:raise ValueError('bootstrap frozen array drift')
        pins[str(path)]=pin
    data={key:np.load(path,allow_pickle=False) for key,path in paths.items()}
    with np.load(draw_path,allow_pickle=False) as archive:draws={key:archive[key] for key in archive.files}
    verify_draws(registration,draws,partitions,data['pop'],data['component'])
    build=args.binary_dir/'BUILD.json';binary=args.binary_dir/'panel'
    if sha(binary)!=json.loads(build.read_text())['binary_sha256']['panel']:raise ValueError('native panel drift')
    pins.update({str(build):sha(build),str(binary):sha(binary)})
    return dict(plan=plan,partitions=partitions,outputs=outputs,pins=pins,qualifications=qualifications,
        registration=registration,draws=draws,data=data,binary=binary,comparison=comparison,spec=spec,
        arms=arms if comparison is not None else None,seed_ids=seed_ids if comparison is not None else None)


def run(args):
    context=admit(args)
    plan=context['plan'];outputs=context['outputs'];pins=context['pins'];qualifications=context['qualifications']
    registration=context['registration'];draws=context['draws'];data=context['data'];binary=context['binary']
    comparison=context['comparison'];spec=context['spec'];arms=context['arms'];seed_ids=context['seed_ids']
    os.environ['ZEN_PANEL_BIN']=str(binary)
    from shaping_storage import reservation
    with reservation(args.study,args.output,plan['minimum_ssd_free_bytes']):
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'INPUTS.json').write_bytes(encoded(dict(input_sha256=pins,qualifications=qualifications)))
        def progress(stratum,result):
            np.savez_compressed(args.output/(stratum['array']+'.npz'),**result)
            print(json.dumps(dict(completed_stratum=stratum['array'],replicates=REPLICATES)),flush=True)
        try:
            if comparison is None:
                samples,points,summary=compute(outputs,registration,draws,data['y'],progress)
            else:
                samples,points,summary=compute_matched(arms,registration,draws,data['y'],seed_ids,progress)
            for name,pin in pins.items():
                if sha(name)!=pin:raise ValueError('bootstrap input changed during computation')
            arrays={**{f+'__samples':v for f,v in samples.items()},**{f+'__points':v for f,v in points.items()}}
            np.savez_compressed(args.output/'samples.npz',**arrays)
            summary.update(input_sha256=pins,convergence_qualifications=qualifications,
                outer_population_reports={r['cell']['id']:r['report']['outer_test'] for r in outputs},
                samples_sha256=sha(args.output/'samples.npz'),driver_sha256=sha(__file__),numpy_version=np.__version__,
                outer_results_read=True,final_holdout_read=False,scientific_scope=('48 reduced primary outer cells; six inner-selected roles' if any('role' in r for r in outputs) else '102 primary outer cells only; variants and fresh controls require separate comparisons') if comparison is None else 'one fully paired registered matched outer comparison')
            if comparison is not None:summary['comparison']=spec
            (args.output/'SUMMARY.json').write_bytes(encoded(summary))
            (args.output/'COMPLETE.json').write_bytes(encoded(dict(summary_sha256=sha(args.output/'SUMMARY.json'))))
        except Exception as error:
            (args.output/'FAILURE.json').write_bytes(encoded(dict(error=str(error),complete=False)))
            raise


def ready_cells(study,cells,admissions):
    """Metadata-only readiness; the runner repeats full scientific admission."""
    from shaping_completion import NEURAL
    from shaping_evaluation import VARIANT_STAGES
    for cell in cells:
        outer=int(cell['partition'].split('-')[1]);admission=admissions[outer]
        family=admission['finalist_roles'][cell['family']] if cell['stage'] in VARIANT_STAGES else cell['family']
        attempts=list((study/'runs'/cell['id']).glob('attempt-*'))
        complete=[]
        for attempt in attempts:
            result=attempt/'RESULT.json'
            if not result.exists():return False
            record=json.loads(result.read_text())
            if record['exit_code']!=0:continue
            if record['cell']!=cell or record['stage']!=('trained' if family in NEURAL else 'complete'):
                raise ValueError('bootstrap readiness terminal identity differs')
            representation=admission.get('evaluation_representations',{}).get(family,{'representation':'original'})
            panel=study/'reexport_max_pairs'/cell['id']/attempt.name/'final' if representation['representation']!='original' else attempt/'evaluation'
            if not (panel/'COMPLETE.json').exists():return False
            complete.append(attempt)
        if len(complete)>1:raise ValueError('ambiguous completed bootstrap cell')
        if not complete:return False
    return True


def complete_output(output):
    if not (output/'COMPLETE.json').exists():return False
    marker=json.loads((output/'COMPLETE.json').read_text());summary=json.loads((output/'SUMMARY.json').read_text())
    if marker['summary_sha256']!=sha(output/'SUMMARY.json') or summary['samples_sha256']!=sha(output/'samples.npz'):
        raise ValueError('completed bootstrap output drift')
    return True


def archive_output(output,destination,logs):
    from shaping_completion import REPO
    receipt=output/'MIRROR_VERIFIED.json'
    if not receipt.exists():
        if destination.exists() or (output/'SHA256SUMS').exists():raise ValueError('unfinished bootstrap archive needs reconciliation')
        command=[sys.executable,'-B',str(REPO/'scripts/byconstruction/dataset_mirror.py'),'--root',str(output),
                 '--nfs','--cold-readback','--destination',str(destination)]
        with (logs/('archive-'+output.name+'.log')).open('x') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    record=json.loads(receipt.read_text())
    if not record['all_artifact_sha256_verified'] or record['destination']!=str(destination) or sha(output/'SHA256SUMS')!=sha(destination/'SHA256SUMS'):
        raise ValueError('bootstrap archive readback incomplete')


def watch_sweep(args):
    from shaping_execution import load_registration
    from shaping_bootstrap_plan import load_comparison
    from shaping_outer_admission import resolve
    from shaping_outer_evidence import matched_cells
    _,cells,_=load_registration(args.study)
    registry=args.matched_registry/'MATCHED_REGISTRATION.json'
    entries=json.loads(registry.read_text())['comparisons']
    # Validate the entire original or owner-amended registry before any job.
    if not entries:raise ValueError('empty matched bootstrap registry')
    load_comparison(args.study,args.matched_registry/entries[0]['file'])
    jobs=[('primary',None,None)]+[(Path(row['file']).stem,args.matched_registry/row['file'],row['spec']) for row in entries]
    from shaping_reduced_scope import coverage
    reduced=coverage(args.study)
    required=25 if reduced else 89
    if len(jobs)!=required:raise ValueError('complete primary plus matched bootstrap queue required')
    args.output.mkdir(parents=True,exist_ok=True)
    failures=args.output/'failures';failures.mkdir(exist_ok=True)
    logs=args.output/'logs';logs.mkdir(exist_ok=True)
    status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),required=required,completed=[],archived=[],failed=[],waiting=[],driver_sha256=sha(__file__))
    admissions=None
    for name,comparison,spec in jobs:
        output=args.output/name;failure=failures/(name+'.json')
        if complete_output(output):
            status['completed'].append(name)
            archive_output(output,args.archive_destination/name,logs);status['archived'].append(name)
            continue
        if failure.exists():status['failed'].append(name);continue
        if output.exists():raise ValueError('incomplete bootstrap output requires explicit reconciliation: '+str(output))
        if not all((args.study/'selection'/f'outer-{o}'/'OUTER_ADMISSION.json').exists() for o in range(3)):
            status['waiting'].append(name);continue
        if admissions is None:admissions={o:resolve(args.study,o)[0] for o in range(3)}
        if spec is None:wanted=primary_cells(cells,admissions)
        else:
            pairs,_=matched_cells(cells,admissions,**spec)
            wanted=list({c['id']:c for pair in pairs for c in (pair['candidate'],pair['control'])}.values())
        if not ready_cells(args.study,wanted,admissions):status['waiting'].append(name);continue
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--study',str(args.study),'--draws',str(args.draws),
                 '--binary-dir',str(args.binary_dir),'--output',str(output)]
        if comparison is not None:command.extend(['--comparison',str(comparison)])
        launch=dict(name=name,pid=os.getpid(),command=command,started_utc=datetime.now(timezone.utc).isoformat(),stage='computing')
        (args.output/'ACTIVE.json').write_bytes(encoded(launch));print(json.dumps(launch),flush=True)
        with (logs/(name+'.log')).open('x') as stream:
            child=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if child.returncode or not complete_output(output):
            failure.write_bytes(encoded(dict(**launch,exit_code=child.returncode,complete=False,log=str(logs/(name+'.log')))))
            status['failed'].append(name)
        else:
            status['completed'].append(name)
            archive_output(output,args.archive_destination/name,logs);status['archived'].append(name)
        (args.output/'ACTIVE.json').write_bytes(encoded(dict(**launch,finished_utc=datetime.now(timezone.utc).isoformat(),exit_code=child.returncode)))
    status.update(checked_utc=datetime.now(timezone.utc).isoformat(),stage='complete' if len(status['completed'])==required else 'waiting_or_failed')
    (args.output/'STATUS.json').write_bytes(encoded(status));print(json.dumps(status),flush=True)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','draws','binary-dir','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--comparison',type=Path,help='JSON role/condition or fresh_family; full matched provenance required')
    p.add_argument('--watch',action='store_true')
    p.add_argument('--matched-registry',type=Path)
    p.add_argument('--archive-destination',type=Path)
    args=p.parse_args()
    if args.watch:
        if args.comparison is not None or args.matched_registry is None or args.archive_destination is None:p.error('watch requires matched registry and archive destination, and excludes individual comparison')
        while True:
            status=watch_sweep(args)
            if status['stage']=='complete':break
            time.sleep(30)
    else:
        if args.matched_registry is not None or args.archive_destination is not None:p.error('matched registry and archive destination require watch')
        run(args)
