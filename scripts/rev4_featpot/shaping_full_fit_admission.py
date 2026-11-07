"""Bind global full-FIT planning to all three complete inner-evidence admissions."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from shaping_completion import FAMILIES,NEURAL,REPO,encoded,sha
from shaping_execution import load_registration
from shaping_outer_admission import resolve as resolve_outer
from shaping_full_fit_plan import plan

POLICY=REPO/'docs/SHAPING_FULL_FIT_SELECTION.md'


def build(root):
    _,cells,_=load_registration(root);folds={};pins={};representations={}
    for outer in range(3):
        admission,path=resolve_outer(root,outer);pins[str(path)]=sha(path)
        pins.update(admission['decision_sha256']);folds[outer]={}
        directory=root/'selection'/f'outer-{outer}'
        for family in FAMILIES:
            base_path=directory/(family+'-base.json')
            if pins.get(str(base_path))!=sha(base_path):raise ValueError('full-FIT base evidence is not admitted')
            base=json.loads(base_path.read_text());schedule=admission['schedules'][family]
            choices={'base':base['setting_scores'][str(schedule['setting'])]}
            if family in NEURAL:
                rescue_path=directory/(family+'-rescue.json')
                if rescue_path.exists():
                    if pins.get(str(rescue_path))!=sha(rescue_path):raise ValueError('full-FIT rescue evidence is not admitted')
                    rescue=json.loads(rescue_path.read_text());choices=rescue['choice_scores']
                identity=admission['evaluation_representations'][family]
                if family in representations and representations[family]!=identity:
                    raise ValueError('full-FIT planning requires uniform evaluation representations across folds')
                representations[family]=identity
            selected=schedule['optimizer_choice']
            if selected not in choices:raise ValueError('selected optimizer lacks paired evidence')
            folds[outer][family]=dict(family=family,outer=outer,outer_results_read=False,
                inner_score=choices[selected],choice_scores=choices,phase_epochs=schedule['phase_epochs'])
    record=plan(folds,cells)
    record.update(plan_sha256=sha(root/'PLAN.json'),policy_sha256=sha(POLICY),
        evidence_sha256=pins,evaluation_representations=representations,driver_sha256=sha(__file__),
        status='full_fit_search_admitted_from_complete_inner_evidence')
    return record


def register(root):
    record=build(root);directory=root/'full_fit';directory.mkdir(exist_ok=True)
    with (directory/'PLAN.json').open('xb') as stream:stream.write(encoded(record))
    return record


def resolve(root):
    path=root/'full_fit/PLAN.json';saved=json.loads(path.read_text());actual=build(root)
    if {k:v for k,v in saved.items() if k!='driver_sha256'}!={k:v for k,v in actual.items() if k!='driver_sha256'}:
        raise ValueError('full-FIT admission cannot be reproduced')
    return saved,path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--register',action='store_true');a=p.parse_args()
    print(json.dumps((register if a.register else build)(a.study)),flush=True)
