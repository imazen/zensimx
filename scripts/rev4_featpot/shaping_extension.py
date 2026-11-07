"""Register required doubled-exposure cells from complete inner-only selections."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from shaping_completion import NEURAL,REPO,encoded,sha
from shaping_execution import load_registration
from shaping_rescue import selected_base
from shaping_rescue_selection import aggregate

IMPROVEMENT={'loss_improvement_over_one_percent_or_rounding_boundary',
             'same_cycle_inner_composite_improvement_over_0.002'}


def matrix(decisions,outer):
    if set(decisions)!=set(NEURAL):raise ValueError('all neural family selections required')
    triggered=[]
    for family,decision in decisions.items():
        if decision['family']!=family or decision['outer']!=outer or decision['outer_results_read']:
            raise ValueError('selection identity or exposure differs')
        selected=list(decision['selected_convergence'].values())
        if len(selected)!=6 or {(c['cell']['partition'],c['cell']['seed']) for c in selected}!={
                (f'inner-{outer}-{fold}',seed) for fold in range(2) for seed in range(3)}:
            raise ValueError('selected convergence pairing differs')
        if any(IMPROVEMENT & set(c['reasons']) for c in selected):triggered.append(family)
    families=[f for f in NEURAL if f in triggered or (f=='relu' and triggered)]
    cells=[]
    def add(family,decision,partition,seed,role,targets):
        pretrained='_pretrained_' in family
        phases=[240,240] if pretrained else [240] if role=='fresh_control' else [480]
        cell=dict(stage='exposure_extension',family=family,partition=partition,seed=seed,
                  setting=decision['base_setting'],optimizer_choice=decision['selected_choice'],
                  role=role,paired_families=targets,phase_epochs=phases,pairs_per_epoch=50000,
                  total_pair_draws=sum(phases)*50000)
        cell['id']=hashlib.sha256(encoded(cell)).hexdigest()[:20];cells.append(cell)
    for family in families:
        for fold in range(2):
            for seed in range(3):add(family,decisions[family],f'inner-{outer}-{fold}',seed,'primary',[family])
    fresh={}
    for family in families:
        if '_pretrained_' not in family:continue
        d=decisions[family];fresh.setdefault((d['base_setting'],d['selected_choice']),[]).append(family)
    for targets in fresh.values():
        for fold in range(2):
            for seed in range(3):add('relu',decisions[targets[0]],f'inner-{outer}-{fold}',seed,'fresh_control',targets)
    if len({c['id'] for c in cells})!=len(cells):raise ValueError('extension identity collision')
    return dict(triggered_families=triggered,extended_families=families,cells=cells,
                outer=outer,outer_results_read=False,outer_evaluation_admitted=False)


def verify_decision(root,family,outer,cells):
    setting,_,_=selected_base(root,family,outer,cells)
    path=root/'selection'/f'outer-{outer}'/(family+'-rescue.json')
    decision=json.loads(path.read_text())
    if decision['plan_sha256']!=sha(root/'PLAN.json'):raise ValueError('rescue plan drift')
    reports={};convergence={}
    for name,pin in decision['evidence_sha256'].items():
        evidence=Path(name)
        if sha(evidence)!=pin:raise ValueError('rescue evidence drift')
        if evidence.name in ('REPORT.json','CONVERGENCE.json'):
            row=json.loads(evidence.read_text())
            (reports if evidence.name=='REPORT.json' else convergence)[row['cell']['id']]=row
    actual=aggregate(cells,reports,family,outer,setting)
    for key in ('family','outer','base_setting','choice_scores','selected_choice','required_reports','reported_cells','outer_results_read'):
        if actual[key]!=decision[key]:raise ValueError('rescue decision cannot be reproduced: '+key)
    selected={c['id'] for c in cells if c['id'] in reports and
              ('base' if c['stage']=='inner' else c['rescue'])==actual['selected_choice']}
    if {k:v for k,v in convergence.items() if k in selected}!=decision['selected_convergence']:
        raise ValueError('selected convergence differs from pinned evidence')
    for row in decision['selected_convergence'].values():
        if row['outer_results_read']:raise ValueError('outer-exposed diagnostics')
        for name,pin in row['report_sha256'].items():
            if sha(name)!=pin:raise ValueError('diagnostic curve drift')
    return decision,path


def register(root,outer):
    _,cells,_=load_registration(root);decisions={};pins={}
    for family in NEURAL:
        decision,path=verify_decision(root,family,outer,cells)
        decisions[family]=decision;pins[str(path)]=sha(path)
    result=matrix(decisions,outer)
    result.update(plan_sha256=sha(root/'PLAN.json'),decision_sha256=pins,
                  policy_sha256=sha(REPO/'docs/SHAPING_EXPOSURE_EXTENSION.md'),driver_sha256=sha(__file__))
    out=root/'extensions'/f'outer-{outer}';out.mkdir(parents=True,exist_ok=False)
    (out/'REGISTRATION.json').write_bytes(encoded(result));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--outer',type=int,choices=range(3),required=True);a=p.parse_args();register(a.study,a.outer)
