"""Freeze paired component draws from membership metadata, without reading scores."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from shaping_completion import encoded,sha
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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();record=register(a.study,a.output)
    print(json.dumps(dict(strata=len(record['strata']),replicates=record['replicates'],scores_read=False)),flush=True)
