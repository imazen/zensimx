"""Paired percentile intervals; mean seed performance, never mean predictions."""
from __future__ import annotations
import numpy as np
from shaping_completion import FAMILIES,seeds
from shaping_bootstrap_plan import REPLICATES

POPULATIONS=('cid22_a25','cid22_ssim2','kadid','konfig','tid2013')
HUMANS=(0,2,3,4)


def interval(samples,point,control,control_point):
    return dict(point=float(point),percentile95=np.quantile(samples,[.025,.975]).tolist(),
                difference_from_relu=float(point-control_point),
                difference_percentile95=np.quantile(samples-control,[.025,.975]).tolist())


def reduce(samples,points):
    """Input axes: fold, model seed, population, pooled/within, replicate."""
    if set(samples)!=set(FAMILIES) or set(points)!=set(FAMILIES):
        raise ValueError('all registered families required for primary reduction')
    averaged={};point_average={}
    for family in FAMILIES:
        s=np.asarray(samples[family]);p=np.asarray(points[family])
        shape=(3,len(tuple(seeds(family))),len(POPULATIONS),2)
        if s.shape!=(*shape,REPLICATES) or p.shape!=shape:
            raise ValueError('complete fold/seed/population/replicate axes required')
        if (not np.isfinite(s).all() or not np.isfinite(p).all()
            or (np.abs(s)>1+1e-12).any() or (np.abs(p)>1+1e-12).any()):
            raise ValueError('finite signed rank correlations required')
        averaged[family]=s.mean(axis=1);point_average[family]=p.mean(axis=1)
    result={}
    for name,folds in [('outer-0',[0]),('outer-1',[1]),('outer-2',[2]),('mean_outer',[0,1,2])]:
        control=averaged['relu'][folds].mean(axis=0);control_point=point_average['relu'][folds].mean(axis=0)
        families={}
        for family in FAMILIES:
            s=averaged[family][folds].mean(axis=0);p=point_average[family][folds].mean(axis=0)
            populations={}
            for i,population in enumerate(POPULATIONS):
                populations[population]={metric:interval(s[i,j],p[i,j],control[i,j],control_point[i,j])
                                         for j,metric in enumerate(('pooled','within'))}
                populations[population]['composite']=interval(s[i].mean(axis=0),p[i].mean(),control[i].mean(axis=0),control_point[i].mean())
            human=interval(s[list(HUMANS)].mean(axis=(0,1)),p[list(HUMANS)].mean(),
                           control[list(HUMANS)].mean(axis=(0,1)),control_point[list(HUMANS)].mean())
            families[family]=dict(populations=populations,equal_human_composite=human)
        result[name]=families
    return dict(scopes=result,replicates=REPLICATES,confidence_level=.95,
        estimator='mean seed rank performance; equal outer folds; equal four human populations and pooled/within ranks',
        teacher_in_human_headline=False,pooled_oof_used=False,selection_performed=False,
        uncertainty='population-stratified source bootstrap conditional on fitted models and observed sources; training-seed uncertainty is not estimated')


def reduce_matched(samples,points,seed_ids):
    """Paired variant/control intervals with identical registered model seeds."""
    seed_ids=tuple(seed_ids)
    if seed_ids not in ((0,1,2),(None,)):
        raise ValueError('all three paired seeds or the deterministic control required')
    if set(samples)!={'candidate','control'} or set(points)!=set(samples):
        raise ValueError('one complete matched candidate/control pair required')
    averaged={};point_average={};shape=(3,len(seed_ids),len(POPULATIONS),2)
    for key in ('candidate','control'):
        s=np.asarray(samples[key]);p=np.asarray(points[key])
        if s.shape!=(*shape,REPLICATES) or p.shape!=shape:
            raise ValueError('complete matched fold/seed/population/replicate axes required')
        if not np.isfinite(s).all() or not np.isfinite(p).all() or (np.abs(s)>1+1e-12).any() or (np.abs(p)>1+1e-12).any():
            raise ValueError('finite signed matched rank correlations required')
        averaged[key]=s.mean(axis=1);point_average[key]=p.mean(axis=1)
    def paired(s,p,c,cp):
        value=interval(s,p,c,cp)
        value['difference_from_control']=value.pop('difference_from_relu')
        value['control_point']=float(cp)
        return value
    scopes={}
    for name,folds in [('outer-0',[0]),('outer-1',[1]),('outer-2',[2]),('mean_outer',[0,1,2])]:
        s=averaged['candidate'][folds].mean(axis=0);p=point_average['candidate'][folds].mean(axis=0)
        c=averaged['control'][folds].mean(axis=0);cp=point_average['control'][folds].mean(axis=0)
        populations={}
        for i,population in enumerate(POPULATIONS):
            populations[population]={metric:paired(s[i,j],p[i,j],c[i,j],cp[i,j]) for j,metric in enumerate(('pooled','within'))}
            populations[population]['composite']=paired(s[i].mean(axis=0),p[i].mean(),c[i].mean(axis=0),cp[i].mean())
        human=paired(s[list(HUMANS)].mean(axis=(0,1)),p[list(HUMANS)].mean(),c[list(HUMANS)].mean(axis=(0,1)),cp[list(HUMANS)].mean())
        scopes[name]=dict(populations=populations,equal_human_composite=human)
    return dict(scopes=scopes,replicates=REPLICATES,seed_ids=list(seed_ids),confidence_level=.95,
        estimator='paired mean seed rank performance; equal outer folds; equal four human populations and pooled/within ranks',
        teacher_in_human_headline=False,selection_performed=False,
        uncertainty='source-bootstrap conditional on fitted models; model-seed and membership-order uncertainty are not estimated')
