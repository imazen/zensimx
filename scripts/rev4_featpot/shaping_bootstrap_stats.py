"""Native signed-rank panels on paired whole-component bootstrap draws."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch_indexed


def paired_stratum(predictions,target,groups,draws,*,panel=panel_batch_indexed):
    """No fit selection: callers must admit and pin every scientific input first.

    Predictions are model-by-row native quality scores. Groups partition the
    rows; draws index these groups with replacement. Seed performance is kept
    separate here; averaging predictions would describe a different estimator.
    """
    predictions=np.asarray(predictions,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    draws=np.asarray(draws)
    if predictions.ndim!=2 or target.ndim!=1 or predictions.shape[1]!=len(target) or not len(predictions):
        raise ValueError('aligned model-by-row predictions and targets required')
    if not np.isfinite(predictions).all() or not np.isfinite(target).all():
        raise ValueError('finite native scores and targets required')
    groups=[np.asarray(g) for g in groups]
    if not groups or any(g.ndim!=1 or not len(g) or g.dtype.kind not in 'iu' for g in groups):
        raise ValueError('nonempty integer component groups required')
    members=np.concatenate(groups)
    if len(members)!=len(target) or not np.array_equal(np.sort(members),np.arange(len(target))):
        raise ValueError('components must partition every row exactly once')
    # The study's metrics owner marks these correlations undefined; the native
    # panel's numeric zero convention must not become a measured bootstrap rank.
    for group in groups:
        if len(group)<2 or np.ptp(target[group])==0 or (np.ptp(predictions[:,group],axis=1)==0).any():
            raise ValueError('undefined within-component rank; cannot compute complete paired intervals')
    if draws.ndim!=2 or not len(draws) or draws.shape[1]!=len(groups) or draws.dtype.kind not in 'iu':
        raise ValueError('integer component draws with registered stratum width required')
    if draws.min()<0 or draws.max()>=len(groups):raise ValueError('component draw outside stratum')
    models=len(predictions);replicates=len(draws)
    bases={f'p{k}':p for k,p in enumerate(predictions)};bases['target']=target

    def ranks(jobs):
        result=panel(bases,jobs,stats='srocc')
        expected=[job[0] for job in jobs]
        if len(result)!=len(expected) or [row['label'] for row in result]!=expected:
            raise ValueError('native panel response identity differs')
        values=[]
        for row in result:
            if row['n_dropped'] or not np.isfinite(row['srocc_signed']):
                raise ValueError('native bootstrap panel dropped rows or returned undefined rank')
            values.append(row['srocc_signed'])
        return np.asarray(values)

    source=ranks([(f's{k}_{j}',f'p{k}','target',g)
                  for k in range(models) for j,g in enumerate(groups)]).reshape(models,len(groups))
    pooled_point=ranks([(f'point{k}',f'p{k}','target',None) for k in range(models)])
    point=np.stack((pooled_point,source.mean(axis=1)),axis=1)
    pooled=np.empty((models,replicates),dtype=np.float64)
    # Bounded batches keep the replicated index payload out of persistent arrays.
    for start in range(0,replicates,32):
        stop=min(start+32,replicates);jobs=[]
        for b in range(start,stop):
            ids=np.concatenate([groups[int(g)] for g in draws[b]])
            jobs.extend((f'b{k}_{b}',f'p{k}','target',ids) for k in range(models))
        values=ranks(jobs).reshape(stop-start,models)
        pooled[:,start:stop]=values.T
    within=source[:,draws].mean(axis=2)
    return dict(pooled=pooled,within=within,composite=(pooled+within)/2,point=point)
