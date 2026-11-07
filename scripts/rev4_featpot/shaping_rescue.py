"""Resolve registered rescue settings using complete inner-only evidence."""
from __future__ import annotations
import json
from pathlib import Path
from shaping_completion import NEURAL,RESCUES,sha
from shaping_execution import load_registration
from shaping_selection import aggregate


def selected_base(root,family,outer,cells):
    path=root/'selection'/f'outer-{outer}'/(family+'-base.json')
    record=json.loads(path.read_text())
    if record['plan_sha256']!=sha(root/'PLAN.json') or record['outer_results_read']:
        raise ValueError('selection plan or exposure differs')
    reports={}
    for name,pin in record['evidence_sha256'].items():
        evidence=Path(name)
        if sha(evidence)!=pin:raise ValueError('selection evidence drift')
        if evidence.name=='REPORT.json':
            report=json.loads(evidence.read_text());reports[report['cell']['id']]=report
    actual=aggregate(cells,reports,family,outer)
    if any(record[k]!=actual[k] for k in ('selected_setting','reported_cells','required_reports')):
        raise ValueError('base selection cannot be reproduced')
    for cell_id in actual['reported_cells']:
        candidates=[Path(p) for p in record['evidence_sha256'] if Path(p).name=='CONVERGENCE.json'
                    and json.loads(Path(p).read_text())['cell']['id']==cell_id]
        if len(candidates)!=1:raise ValueError('missing paired convergence evidence')
        convergence=json.loads(candidates[0].read_text())
        if convergence['outer_results_read']:raise ValueError('outer-exposed convergence')
        for name,pin in convergence['report_sha256'].items():
            if sha(name)!=pin:raise ValueError('convergence input drift')
    return actual['selected_setting'],str(path),sha(path)


def resolve(root,cell_id):
    _,cells,_=load_registration(root)
    cell=next(c for c in cells if c['id']==cell_id)
    if cell['stage']!='rescue' or cell['family'] not in NEURAL or cell['rescue'] not in RESCUES:
        raise ValueError('not a registered neural rescue')
    outer=int(cell['partition'].split('-')[1]);trigger_path=root/f'RESCUE_TRIGGER_OUTER_{outer}.json'
    trigger=json.loads(trigger_path.read_text())
    expected={c['id'] for c in cells if c['stage']=='rescue' and c['partition'].startswith(f'inner-{outer}-')}
    if set(trigger['required_rescue_cells'])!=expected or trigger['outer_results_read']:
        raise ValueError('rescue trigger coverage or exposure differs')
    paths=list((root/'runs'/trigger['trigger_cell']['id']).glob('attempt-*/CONVERGENCE.json'))
    if len(paths)!=1 or sha(paths[0])!=trigger['convergence_sha256']:
        raise ValueError('rescue trigger evidence differs')
    convergence=json.loads(paths[0].read_text())
    if not convergence['investigation_required'] or convergence['cell']!=trigger['trigger_cell']:
        raise ValueError('rescue trigger is not an investigated registered fit')
    for name,pin in convergence['report_sha256'].items():
        if sha(name)!=pin:raise ValueError('trigger report drift')
    setting,path,pin=selected_base(root,cell['family'],outer,cells)
    return dict(cell=cell,resolved_setting=setting,base_selection=path,base_selection_sha256=pin,
                trigger_sha256=sha(trigger_path),plan_sha256=sha(root/'PLAN.json'),outer_results_read=False)


def controls(rescue,analytic=False,*,groups=None):
    if rescue not in RESCUES:raise ValueError('unregistered rescue')
    scale={'init_half':.5,'init_double':2.}.get(rescue,1.)
    schedule={'constant_lr':'constant','consolidate_lr':'consolidate'}.get(rescue,'cosine')
    if not analytic:return dict(schema='plain-shared-optimizer-v1',init_scale=scale,schedule=schedule)
    if not groups or sum(g['name']=='cid22' for g in groups)!=1:
        raise ValueError('analytic training controls require exact admitted row-family groups and one teacher')
    if len({g['name'] for g in groups})!=len(groups) or any(not g['families'] for g in groups):
        raise ValueError('empty or duplicate analytic training groups')
    # Native Events validates each name and row count against admitted tables.
    # Match plain owner's weight AND bias scaling; retain teacher mass and pair order.
    return dict(schema='monotone-training-v1',init_scale=scale,bias_scale=scale,
                dropout=0.,dropout_seed=0,adamw=0.,schedule=schedule,order='random',
                teacher_mass=1.,balance='none',groups=groups)
