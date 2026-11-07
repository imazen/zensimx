"""Freeze paired component draws from membership metadata, without reading scores."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from shaping_completion import NEURAL,REPO,encoded,sha
from shaping_execution import load_registration

SEED=20261107
REPLICATES=2000


def population_groups(indices,populations,components):
    indices=np.asarray(indices,dtype=np.int64)
    if indices.ndim!=1 or len(set(indices.tolist()))!=len(indices) or not len(indices):
        raise ValueError('unique nonempty row membership required')
    if len(populations)!=len(components) or indices.min()<0 or indices.max()>=len(populations):
        raise ValueError('metadata alignment differs')
    ownership={};groups={}
    for i in indices:
        pop=str(populations[i]);component=str(components[i])
        if component in ownership and ownership[component]!=pop:
            raise ValueError('component spans populations; independent stratification is invalid')
        ownership[component]=pop
        groups.setdefault(pop,{}).setdefault(component,[]).append(int(i))
    return {p:{c:groups[p][c] for c in sorted(groups[p])} for p in sorted(groups)}


def component_draws(groups,rng,replicates=REPLICATES):
    if not groups or replicates<1:raise ValueError('nonempty groups and positive replicates required')
    return rng.integers(len(groups),size=(replicates,len(groups)),dtype=np.int32)


def expand(groups,draw):
    """Whole sources, including repeat multiplicity; shared by every model."""
    rows=list(groups.values())
    if len(draw)!=len(rows) or any(i<0 or i>=len(rows) for i in draw):
        raise ValueError('component draw outside registered stratum')
    return np.asarray([row for i in draw for row in rows[i]],dtype=np.int64)


def register(study,output):
    plan,_,partitions=load_registration(study)
    from shaped_linear_study import OLD
    paths={'population':OLD/'data/pop.npy',
           'component':Path(plan['original_plan']).parent/'data/component.npy'}
    pins={str(p):sha(p) for p in paths.values()}
    if any(plan['input_pins'].get(p)!=pin for p,pin in pins.items()):
        raise ValueError('bootstrap membership metadata drift')
    metadata={k:np.load(p,allow_pickle=False) for k,p in paths.items()}
    rng=np.random.default_rng(SEED);arrays={};strata=[];seen=set()
    for fold in range(3):
        ids=partitions[f'outer-{fold}']['test']
        groups=population_groups(ids,metadata['population'],metadata['component'])
        if set(groups)!={'kadid','tid2013','konfig','cid22_a25','cid22_ssim2'}:
            raise ValueError('registered outer population coverage differs')
        current={c for g in groups.values() for c in g}
        if current&seen:raise ValueError('outer source components overlap')
        seen|=current
        for pi,(population,group) in enumerate(groups.items()):
            key=f'outer{fold}_population{pi}'
            arrays[key]=component_draws(group,rng)
            strata.append(dict(outer=fold,population=population,array=key,groups=group,
                               source_components=len(group),rows=sum(map(len,group.values()))))
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'draws.npz',**arrays)
    record=dict(schema='shaping-source-bootstrap-plan-v1',seed=SEED,replicates=REPLICATES,
        strata=strata,draws_sha256=sha(output/'draws.npz'),plan_sha256=sha(study/'PLAN.json'),
        input_sha256=pins,driver_sha256=sha(__file__),scores_read=False,
        pairing='same component draws for every candidate and control; independent RNG draws across fold/population strata',
        interpretation='population-stratified source bootstrap, conditional on fitted models and observed source components; no parameter selection')
    (output/'REGISTRATION.json').write_bytes(encoded(record))
    return record


def matched_specs(cells):
    fields={'source_curve':('stage','fraction','order_seed'),'row_curve':('stage','fraction','order_seed'),
            'shuffle':('stage','permutation','permutation_seed'),'human_only':('stage',)}
    specs={}
    for cell in cells:
        if cell['stage'] in fields:
            spec=dict(role=cell['family'],condition={k:cell[k] for k in fields[cell['stage']]})
            specs[encoded(spec)]=spec
    for family in NEURAL:
        if '_pretrained_' in family:
            spec=dict(fresh_family=family);specs[encoded(spec)]=spec
    if len(specs)!=88:raise ValueError('complete 88-comparison matched analysis required')
    return [specs[k] for k in sorted(specs)]


def register_matched(study,output):
    _,cells,_=load_registration(study);specs=matched_specs(cells)
    output.mkdir(parents=True,exist_ok=False);entries=[]
    for i,spec in enumerate(specs):
        path=output/f'comparison-{i:03}.json';path.write_bytes(encoded(spec))
        entries.append(dict(file=path.name,spec=spec,sha256=sha(path)))
    policy=REPO/'docs/SHAPING_MATCHED_UNCERTAINTY.md'
    record=dict(schema='shaping-matched-bootstrap-plan-v1',plan_sha256=sha(study/'PLAN.json'),
                policy_sha256=sha(policy),comparisons=entries,scores_read=False,outer_results_read=False,
                seed=SEED,replicates=REPLICATES,driver_sha256=sha(__file__))
    (output/'MATCHED_REGISTRATION.json').write_bytes(encoded(record));return record


def load_comparison(study,path):
    _,cells,_=load_registration(study)
    registration=path.parent/'MATCHED_REGISTRATION.json';record=json.loads(registration.read_text())
    policy=REPO/'docs/SHAPING_MATCHED_UNCERTAINTY.md'
    if (record['plan_sha256']!=sha(study/'PLAN.json') or record['policy_sha256']!=sha(policy)
        or record['scores_read'] or record['outer_results_read'] or record['seed']!=SEED or record['replicates']!=REPLICATES):
        raise ValueError('matched analysis registration drift')
    entries=record['comparisons']
    if [r['spec'] for r in entries]!=matched_specs(cells):raise ValueError('matched comparison coverage differs')
    pins={str(registration):sha(registration),str(policy):sha(policy)};selected=[]
    for i,row in enumerate(entries):
        if row['file']!=f'comparison-{i:03}.json':raise ValueError('matched comparison file identity differs')
        source=path.parent/row['file']
        if sha(source)!=row['sha256'] or json.loads(source.read_text())!=row['spec']:raise ValueError('matched comparison specification drift')
        pins[str(source)]=sha(source)
        if source==path:selected.append(row['spec'])
    if len(selected)!=1:raise ValueError('comparison not in complete registration')
    return selected[0],pins


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--matched',action='store_true')
    a=p.parse_args();record=(register_matched if a.matched else register)(a.study,a.output)
    print(json.dumps(dict(comparisons=len(record.get('comparisons',[])),strata=len(record.get('strata',[])),replicates=record['replicates'],scores_read=False)),flush=True)
