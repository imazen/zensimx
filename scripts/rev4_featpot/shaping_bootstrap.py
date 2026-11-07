"""Primary outer source-bootstrap runner; admission precedes every score read."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
from shaping_completion import FAMILIES,encoded,seeds,sha
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
    primary_cells([row['cell'] for row in outputs])
    samples={f:np.full((3,len(tuple(seeds(f))),5,2,REPLICATES),np.nan) for f in FAMILIES}
    points={f:np.full((3,len(tuple(seeds(f))),5,2),np.nan) for f in FAMILIES}
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
            family=row['cell']['family'];seed=row['cell']['seed'];si=0 if seed is None else seed
            summary=row['report']['outer_test']['summary'][population]
            expected=np.array([summary['pooled'],summary['within']],dtype=float)
            if not np.isfinite(expected).all() or not np.allclose(result['point'][k],expected,rtol=0,atol=1e-9):
                raise ValueError('bootstrap point ranks differ from completed report')
            samples[family][outer,si,pi]=np.stack((result['pooled'][k],result['within'][k]))
            points[family][outer,si,pi]=result['point'][k]
        progress(stratum,result)
    return samples,points,reduce(samples,points)



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


def run(args):
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
                outer_results_read=True,final_holdout_read=False,scientific_scope='102 primary outer cells only; variants and fresh controls require separate comparisons' if comparison is None else 'one fully paired registered matched outer comparison')
            if comparison is not None:summary['comparison']=spec
            (args.output/'SUMMARY.json').write_bytes(encoded(summary))
            (args.output/'COMPLETE.json').write_bytes(encoded(dict(summary_sha256=sha(args.output/'SUMMARY.json'))))
        except Exception as error:
            (args.output/'FAILURE.json').write_bytes(encoded(dict(error=str(error),complete=False)))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','draws','binary-dir','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--comparison',type=Path,help='JSON role/condition or fresh_family; full matched provenance required')
    run(p.parse_args())
