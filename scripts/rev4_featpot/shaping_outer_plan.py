"""Resolve inner-selected refit schedules and exact matched fresh controls."""
from __future__ import annotations
import hashlib,math
from shaping_completion import FAMILIES,NEURAL,RESCUES,encoded


def finalist_roles(decisions):
    if set(decisions)!=set(FAMILIES):raise ValueError('all twelve family decisions required')
    def best(families):
        scores={f:decisions[f]['inner_score'] for f in families}
        if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in scores.values()):
            raise ValueError('finite inner human scores required')
        maximum=max(scores.values())
        return next(f for f in families if maximum-scores[f]<=1e-12)
    return dict(relu='relu',extra_trees='extra_trees',shaped_additive='shaped_additive',
        best_analytic=best([f for f in FAMILIES if f in ('softplus','silu_monotone','silu_diagnostic')]),
        best_product=best([f for f in FAMILIES if f.startswith('product_')]),
        best_separate=best([f for f in FAMILIES if f.startswith('separate_')]))


def plan(decisions,extended_families,outer,cells):
    """Pure planning: caller must validate complete selection/extension evidence."""
    if outer not in range(3) or set(decisions)!=set(FAMILIES):raise ValueError('complete registered fold decisions required')
    extended=set(extended_families)
    if not extended<=set(NEURAL) or (extended and 'relu' not in extended):
        raise ValueError('extensions require registered neural families and paired ReLU')
    schedules={}
    for family in FAMILIES:
        d=decisions[family]
        if d['family']!=family or d['outer']!=outer or d['outer_results_read']:
            raise ValueError('decision identity or exposure differs')
        if d['setting'] not in range(4) or d['optimizer_choice'] not in ('base',*RESCUES):
            raise ValueError('unregistered setting or optimizer choice')
        if family not in NEURAL and d['optimizer_choice']!='base':raise ValueError('neural rescue applied to nonneural control')
        pretrained='_pretrained_' in family
        phases=([240,240] if family in extended else [120,120]) if pretrained else ([480] if family in extended else [240])
        if family not in NEURAL:phases=[]
        schedules[family]=dict(setting=d['setting'],optimizer_choice=d['optimizer_choice'],
                              phase_epochs=phases,total_pair_draws=sum(phases)*50000,
                              exposure_extended=family in extended)
    existing=[c for c in cells if c['stage']=='fresh_control_outer' and c['partition']==f'outer-{outer}']
    if len(existing)!=3 or {c['seed'] for c in existing}!={0,1,2}:raise ValueError('original fresh-control pairing differs')
    relu=schedules['relu'];base=(relu['setting'],relu['optimizer_choice'],120)
    required={base:[]}
    for family,s in schedules.items():
        if '_pretrained_' not in family:continue
        required.setdefault((s['setting'],s['optimizer_choice'],s['phase_epochs'][-1]),[]).append(family)
    controls=[];additional=[]
    for key,paired in sorted(required.items()):
        setting,choice,epochs=key
        for seed in range(3):
            if key==base:cell=next(c for c in existing if c['seed']==seed)
            else:
                cell=dict(stage='fresh_control_outer_matched',family='relu',partition=f'outer-{outer}',
                          seed=seed,setting=setting,optimizer_choice=choice,epochs=epochs)
                cell['id']=hashlib.sha256(encoded(cell)).hexdigest()[:20];additional.append(cell)
            controls.append(dict(cell=cell,setting=setting,optimizer_choice=choice,epochs=epochs,
                                 paired_families=paired,original_registered_cell=key==base))
    return dict(outer=outer,schedules=schedules,finalist_roles=finalist_roles(decisions),
                fresh_controls=controls,additional_cells=additional,outer_results_read=False,
                outer_evaluation_admitted=False,status='plan_requires_verified_selection_and_extension_evidence')
