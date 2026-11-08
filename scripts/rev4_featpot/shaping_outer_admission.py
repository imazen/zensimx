"""Complete inner evidence gate before outer refits; never opens outer responses."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from shaping_completion import FAMILIES,NEURAL,REPO,encoded,sha
from shaping_execution import load_registration
from shaping_selection import aggregate
from shaping_rescue_selection import verified_candidate
from shaping_rescue import resolve as resolve_rescue
from shaping_extension import matrix as extension_matrix,verify_decision
from shaping_outer_plan import plan as outer_plan

POLICY=REPO/'docs/SHAPING_OUTER_ADMISSION.md'


def base_decision(root,family,outer,cells):
    from shaping_corrected_view import base_path
    path=base_path(root,family,outer)
    record=json.loads(path.read_text())
    if record['plan_sha256']!=sha(root/'PLAN.json') or record['outer_results_read']:
        raise ValueError('base decision plan or exposure differs')
    reports={};pins={str(path):sha(path)}
    for name,pin in record['evidence_sha256'].items():
        p=Path(name)
        if sha(p)!=pin:raise ValueError('base decision evidence drift')
        pins[name]=pin
        if p.name=='REPORT.json':
            report=json.loads(p.read_text());reports[report['cell']['id']]=report
    actual=aggregate(cells,reports,family,outer,record.get('selection_policy'))
    if record.get('selection_policy'):
        policy=REPO/'docs/SHAPING_UNDEFINED_SETTING_POLICY.md'
        if record['selection_policy_sha256']!=sha(policy):raise ValueError('base eligibility policy drift')
        if actual['ineligible_settings']!=record['ineligible_settings']:raise ValueError('base eligibility reasons differ')
        from shaping_policy_snapshot import policy_path
        policy=policy_path(root,policy)
        pins[str(policy)]=sha(policy)
    for key in ('family','outer','selected_setting','required_reports','reported_cells','setting_scores'):
        if json.loads(encoded(actual[key]))!=record[key]:raise ValueError('base decision cannot be reproduced: '+key)
    return actual,record,pins


def build(root,outer):
    _,cells,_=load_registration(root);decisions={};evidence={};neural={};representations={};evaluators={}
    from shaping_reduced_scope import coverage,shortlist
    amendment=coverage(root);frozen=None;shortlist_path=None
    if amendment is not None:
        frozen,shortlist_path=shortlist(root,outer)
        active=set(frozen['roles'].values())
        required_neural=tuple(f for f in NEURAL if f in active)
        evidence.update(frozen['evidence_sha256'])
        evidence[str(shortlist_path)]=sha(shortlist_path)
    else:
        active=set(FAMILIES);required_neural=NEURAL
    def check_evaluator(family,row):
        identity=dict(representation='research-relu-max-pairs-v1' if 'reexport_receipt' in row else 'original',
                      evaluation_build_sha256=row.get('evaluation_build_sha256'))
        if identity['representation']!=representations[family]:raise ValueError('candidate representation differs from selection')
        if family in evaluators and evaluators[family]!=identity:raise ValueError('mixed candidate evaluation runtimes')
        evaluators[family]=identity
    trigger=root/f'RESCUE_TRIGGER_OUTER_{outer}.json'
    triggered=trigger.exists()
    if triggered:
        t=json.loads(trigger.read_text())
        required={c['id'] for c in cells if c['stage']=='rescue' and c['partition'].startswith(f'inner-{outer}-')}
        if t['outer']!=outer or t['outer_results_read'] or set(t['required_rescue_cells'])!=required:
            raise ValueError('rescue trigger scope differs')
        resolve_rescue(root,next(c['id'] for c in cells if c['stage']=='rescue' and c['partition']==f'inner-{outer}-0'))
        evidence[str(trigger)]=sha(trigger)
    for family in FAMILIES:
        actual,base,pins=base_decision(root,family,outer,cells);evidence.update(pins)
        setting=actual['selected_setting'];representation=base.get('representation','original')
        representations[family]=representation
        decision=dict(family=family,outer=outer,setting=setting,optimizer_choice='base',
                      inner_score=actual['setting_scores'][setting],outer_results_read=False)
        if family in required_neural:
            # All base settings support the equal-search decision; some also
            # supply a fresh template/pretraining parent at another family's
            # selected setting. Pin every primary source, not only the winner.
            primary={}
            excluded={cid for entry in actual.get('ineligible_settings',{}).values() for cid in entry['undefined_cells']}
            if excluded and not triggered:raise ValueError('excluded undefined fits require equal-opportunity rescues')
            for c in cells:
                if c['id'] in actual['reported_cells']:
                    if c['id'] in excluded:
                        row,_,failed_pins=verified_candidate(root,c,representation,allow_undefined_failure=True)
                        evidence.update(failed_pins);check_evaluator(family,row)
                        continue
                    primary[c['id']]=verified_candidate(root,c,representation)
                    evidence.update(primary[c['id']][2]);check_evaluator(family,primary[c['id']][0])
            if triggered:
                record,path=verify_decision(root,family,outer,cells)
                evidence[str(path)]=sha(path);evidence.update(record['evidence_sha256'])
                decision.update(optimizer_choice=record['selected_choice'],inner_score=record['choice_scores'][record['selected_choice']])
                required=[c for c in cells if c['id'] in record['reported_cells']]
                selected={c['id'] for c in required if ('base' if c['stage']=='inner' else c['rescue'])==record['selected_choice']}
            else:
                required=[c for c in cells if c['id'] in actual['reported_cells']]
                selected={c['id'] for c in required if c['setting']==setting}
                record=dict(family=family,outer=outer,base_setting=setting,selected_choice='base',outer_results_read=False)
            convergence={}
            for cell in required:
                row,diagnostic,pins=(primary[cell['id']] if cell['id'] in primary else verified_candidate(root,cell,representation));evidence.update(pins)
                check_evaluator(family,row)
                if not triggered and diagnostic['investigation_required']:
                    raise ValueError('investigation requires registered equal-opportunity rescues')
                if cell['id'] in selected:convergence[cell['id']]=diagnostic
            if triggered and convergence!=record['selected_convergence']:
                raise ValueError('selected diagnostic evidence differs')
            record['selected_convergence']=convergence;neural[family]=record
            decision['selected_convergence']=convergence
        decisions[family]=decision
    extension=extension_matrix(neural,outer,candidate_families=required_neural);extended=extension['extended_families'];extension_results={}
    if extension['cells']:
        path=root/'extensions'/f'outer-{outer}'/'REGISTRATION.json';registration=json.loads(path.read_text())
        if registration['plan_sha256']!=sha(root/'PLAN.json') or registration['policy_sha256']!=sha(REPO/'docs/SHAPING_EXPOSURE_EXTENSION.md'):
            raise ValueError('extension registration plan or policy drift')
        if any(registration[k]!=v for k,v in extension.items()):raise ValueError('extension matrix differs')
        for name,pin in registration['decision_sha256'].items():
            if sha(name)!=pin or evidence.get(name)!=pin:raise ValueError('extension decision provenance differs')
        evidence[str(path)]=sha(path)
        for cell in extension['cells']:
            row,diagnostic,pins=verified_candidate(root,cell,representations[cell['family']]);evidence.update(pins)
            check_evaluator(cell['family'],row)
            run_path=next(Path(p) for p in pins if Path(p).name=='RUN.json')
            run=json.loads(run_path.read_text())
            if run['resolution']['registration_sha256']!=sha(path) or run['resolution']['cell']!=cell:
                raise ValueError('extension fit resolution differs')
            extension_results[cell['id']]=diagnostic
    resolved=outer_plan(decisions,extended,outer,cells,**({} if frozen is None else
                        dict(active_families=active,frozen_roles=frozen['roles'])))
    qualifications={}
    for family in required_neural:
        diagnostics=(list(decisions[family]['selected_convergence'].values()) if family not in extended else
                     [extension_results[c['id']] for c in extension['cells'] if c['family']==family and c['role']=='primary'])
        if len(diagnostics)!=6:raise ValueError('final diagnostic pairing differs')
        reasons=sorted({reason for d in diagnostics for reason in d['reasons']})
        qualifications[family]=dict(status='inconclusive_after_registered_optimization' if reasons else 'registered_triggers_absent',
                                    reasons=reasons,not_a_global_convergence_proof=True)
    resolved.update(outer_evaluation_admitted=True,status='admitted_with_explicit_convergence_classifications',
                    qualifications=qualifications,evaluation_representations=evaluators,plan_sha256=sha(root/'PLAN.json'),policy_sha256=sha(POLICY),
                    decision_sha256=evidence,driver_sha256=sha(__file__))
    if amendment is not None:
        resolved.update(reduced_scope_sha256=amendment['registration_sha256'],
                        shortlist_sha256=sha(shortlist_path))
    return resolved


def register(root,outer):
    record=build(root,outer);path=root/'selection'/f'outer-{outer}'/'OUTER_ADMISSION.json'
    with path.open('xb') as stream:stream.write(encoded(record))
    return record


def resolve(root,outer):
    path=root/'selection'/f'outer-{outer}'/'OUTER_ADMISSION.json'
    saved=json.loads(path.read_text());actual=build(root,outer)
    # Driver provenance may change under a verified refactor; every scientific
    # decision and pinned policy/input must still reproduce exactly.
    if {k:v for k,v in saved.items() if k!='driver_sha256'}!={k:v for k,v in actual.items() if k!='driver_sha256'}:
        raise ValueError('outer admission cannot be reproduced')
    return saved,path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--outer',type=int,choices=range(3),required=True);p.add_argument('--register',action='store_true')
    a=p.parse_args();print(json.dumps((register if a.register else build)(a.study,a.outer)),flush=True)
