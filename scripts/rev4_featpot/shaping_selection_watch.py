"""Advance complete paired inner evidence through existing selection/admission gates."""
from __future__ import annotations
import argparse,fcntl,json,os,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from shaping_completion import MAX_PAIR_FAMILIES,FAMILIES,NEURAL,encoded,sha
from shaping_execution import load_registration
from shaping_rescue_selection import select as select_rescue
from shaping_extension import register as register_extension,verify_decision
from shaping_outer_admission import register as register_outer,resolve as resolve_outer


def ready(root,cell,representation):
    repaired=representation=='research-relu-max-pairs-v1'
    if representation not in ('original','research-relu-max-pairs-v1'):raise ValueError('unknown selection representation')
    if repaired and cell['family'] not in MAX_PAIR_FAMILIES:raise ValueError('unqualified max-pair family')
    directory=root/('reexport_max_pairs' if repaired else 'runs')/cell['id']
    markers=list(directory.glob('attempt-*/final/COMPLETE.json' if repaired else 'attempt-*/evaluation/COMPLETE.json'))
    if len(markers)>1:raise ValueError('ambiguous completed candidate')
    # Presence releases the full existing provenance gate; it is not proof itself.
    return bool(markers and (markers[0].parent.parent/'CONVERGENCE.json').exists())


def sweep(root,outer):
    _,cells,_=load_registration(root);directory=root/'selection'/f'outer-{outer}'
    if not (root/f'RESCUE_TRIGGER_OUTER_{outer}.json').exists():
        return dict(stage='waiting_for_rescue_trigger',completed_rescue_selections=[])
    decisions={};created=[]
    for family in NEURAL:
        base=directory/(family+'-base.json');decision=directory/(family+'-rescue.json')
        if not base.exists():continue
        if not decision.exists():
            representation=json.loads(base.read_text()).get('representation','original')
            required=[c for c in cells if c['stage']=='rescue' and c['family']==family and
                      c['partition'] in (f'inner-{outer}-0',f'inner-{outer}-1')]
            if len(required)!=24:raise ValueError('complete registered rescue matrix required')
            if not all(ready(root,c,representation) for c in required):continue
            select_rescue(SimpleNamespace(study=root,family=family,outer=outer));created.append(family)
        decisions[family]=verify_decision(root,family,outer,cells)[0]
    result=dict(stage='waiting_for_paired_rescue_evidence',completed_rescue_selections=sorted(decisions),
                created_rescue_selections=created,outer_results_read=False)
    if set(decisions)!=set(NEURAL):return result
    path=root/'extensions'/f'outer-{outer}'/'REGISTRATION.json'
    if not path.exists():register_extension(root,outer)
    registration=json.loads(path.read_text())
    # Reproduce even an empty extension matrix, which has no cell to resolve.
    from shaping_extension import matrix
    from shaping_completion import MAX_PAIR_FAMILIES,REPO
    expected=matrix(decisions,outer)
    if any(registration[k]!=v for k,v in expected.items()) or registration['plan_sha256']!=sha(root/'PLAN.json') or registration['policy_sha256']!=sha(REPO/'docs/SHAPING_EXPOSURE_EXTENSION.md'):
        raise ValueError('extension registration cannot be reproduced')
    for family in NEURAL:
        source=directory/(family+'-rescue.json')
        if registration['decision_sha256'].get(str(source))!=sha(source):raise ValueError('extension decision pin differs')
    result.update(stage='waiting_for_extension_evidence',required_extension_cells=len(registration['cells']))
    for cell in registration['cells']:
        if not ready(root,cell,decisions[cell['family']]['representation']):return result
    if not all((directory/(family+'-base.json')).exists() for family in FAMILIES):return result
    admission=directory/'OUTER_ADMISSION.json'
    if not admission.exists():register_outer(root,outer)
    verified,_=resolve_outer(root,outer)
    result.update(stage='outer_admitted',admission_sha256=sha(admission),qualifications=verified['qualifications'])
    return result


def watch(args):
    directory=args.study/'dispatch/selection';directory.mkdir(parents=True,exist_ok=True)
    with (directory/f'outer-{args.outer}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            identity=dict(pid=os.getpid(),outer=args.outer,driver_sha256=sha(__file__),checked_utc=datetime.now(timezone.utc).isoformat())
            try:result=sweep(args.study,args.outer)
            except Exception as error:
                (directory/f'outer-{args.outer}-STATUS.json').write_bytes(encoded(dict(identity,stage='failed_preserved_for_reconciliation',error=str(error))))
                raise
            (directory/f'outer-{args.outer}-STATUS.json').write_bytes(encoded(dict(identity,**result)))
            if not args.watch or result['stage']=='outer_admitted':return
            time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--outer',type=int,choices=range(3),required=True);p.add_argument('--watch',action='store_true')
    watch(p.parse_args())
