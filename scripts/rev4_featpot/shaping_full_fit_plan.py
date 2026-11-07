"""Resolve full-FIT finalists and fixed optimizer/exposure from inner evidence."""
from __future__ import annotations
import math
from shaping_completion import FAMILIES,NEURAL,RESCUES,ROLES
from shaping_outer_plan import finalist_roles
from shaping_selection import choose,HUMANS


def plan(folds,cells):
    if set(folds)!={0,1,2}:raise ValueError('all three complete inner-evidence folds required')
    summaries={};schedules={}
    for family in FAMILIES:
        records=[]
        for outer in range(3):
            if set(folds[outer])!=set(FAMILIES):raise ValueError('all family evidence required')
            row=folds[outer][family]
            if row['family']!=family or row['outer']!=outer or row['outer_results_read']:
                raise ValueError('full-FIT planning requires unexposed inner decisions')
            if not math.isfinite(row['inner_score']):raise ValueError('nonfinite inner score')
            records.append(row)
        summaries[family]=dict(inner_score=sum(r['inner_score'] for r in records)/3)
        if family not in NEURAL:
            schedules[family]=dict(optimizer_choice='base',phase_epochs=[],total_pair_draws=0)
            continue
        choices=set.intersection(*(set(r['choice_scores']) for r in records))
        if 'base' not in choices or not choices<=set(('base',*RESCUES)):
            raise ValueError('optimizer choices lack complete paired evidence')
        means={c:sum(r['choice_scores'][c] for r in records)/3 for c in choices}
        if not all(math.isfinite(v) for v in means.values()):raise ValueError('nonfinite optimizer score')
        best=max(means.values());choice=next(c for c in ('base',*RESCUES) if c in means and best-means[c]<=1e-12)
        allowed=([120,120],[240,240]) if '_pretrained_' in family else ([240],[480])
        if any(r['phase_epochs'] not in allowed for r in records):raise ValueError('unregistered full-FIT exposure')
        phases=max((r['phase_epochs'] for r in records),key=sum)
        schedules[family]=dict(optimizer_choice=choice,phase_epochs=phases,total_pair_draws=sum(phases)*50000,
                               eligible_choice_scores=means)
    roles=finalist_roles(summaries)
    selection=[c for c in cells if c['stage']=='full_fit_selection']
    final=[c for c in cells if c['stage']=='full_fit']
    expected={(role,f'outer-{fold}',setting,seed) for role in ROLES for fold in range(3) for setting in range(4)
              for seed in ((None,) if role=='shaped_additive' else (0,1,2))}
    if len(selection)!=len(expected) or {(c['family'],c['partition'],c['setting'],c['seed']) for c in selection}!=expected:
        raise ValueError('registered full-FIT search coverage differs')
    expected_final={(role,seed) for role in ROLES for seed in ((None,) if role=='shaped_additive' else (0,1,2))}
    if len(final)!=len(expected_final) or {(c['family'],c['seed']) for c in final}!=expected_final or any(c['partition']!='full-fit' for c in final):
        raise ValueError('registered full-FIT refit coverage differs')
    return dict(finalist_roles=roles,schedules={f:schedules[f] for f in roles.values()},
                family_inner_scores={f:v['inner_score'] for f,v in summaries.items()},
                selection_cells=selection,final_cells=final,development_results_read=False,
                outer_results_used_for_planning=False,status='requires_verified_inner_evidence_receipt')


def select(cells,reports,role,resolved_family):
    expected=[c for c in cells if c['stage']=='full_fit_selection' and c['family']==role]
    required={(setting,f'outer-{fold}',seed) for setting in range(4) for fold in range(3)
              for seed in ((None,) if role=='shaped_additive' else (0,1,2))}
    if len(expected)!=len(required) or {(c['setting'],c['partition'],c['seed']) for c in expected}!=required:
        raise ValueError('full-FIT selection pairing differs')
    if set(reports)!={c['id'] for c in expected}:raise ValueError('missing or extra full-FIT reports')
    scores={s:[] for s in range(4)};representations=set()
    for cell in expected:
        row=reports[cell['id']]
        if row['cell']!=cell or row['resolved_family']!=resolved_family or row['selection_scope']!='full_fit_only' or row['development_results_read']:
            raise ValueError('full-FIT selection report scope differs')
        panel=row['full_fit_validation'];values=[panel['summary'][p][k] for p in HUMANS for k in ('pooled','within')]
        if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in values):raise ValueError('undefined human panel')
        score=sum(values)/len(values)
        if not math.isclose(score,panel['selection_score'],rel_tol=0.,abs_tol=1e-12):raise ValueError('human composite differs')
        representations.add((row.get('representation','original'),row.get('evaluation_build_sha256')))
        scores[cell['setting']].append(score)
    if len(representations)!=1:raise ValueError('mixed full-FIT evaluation representations')
    means={s:sum(v)/len(v) for s,v in scores.items()}
    return dict(role=role,resolved_family=resolved_family,selected_setting=choose(means),setting_scores=means,
                reported_cells=sorted(reports),selection_scope='full_fit_only',development_results_read=False)
