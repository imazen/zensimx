"""Compare the selected base against all four registered paired rescue choices."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from shaping_completion import MAX_PAIR_FAMILIES,NEURAL,RESCUES,REPO,encoded,sha
from shaping_execution import load_registration
from shaping_rescue import selected_base
from shaping_selection import human_score


def aggregate(cells,reports,family,outer,setting):
    if family not in NEURAL or setting not in range(4):raise ValueError('registered neural setting required')
    expected=[c for c in cells if c['family']==family and c['partition'] in (f'inner-{outer}-0',f'inner-{outer}-1')
              and (c['stage']=='rescue' or (c['stage']=='inner' and c['setting']==setting))]
    choices=('base',*RESCUES)
    key=lambda c:('base' if c['stage']=='inner' else c['rescue'],c['partition'],c['seed'])
    required={(choice,f'inner-{outer}-{fold}',seed) for choice in choices for fold in range(2) for seed in range(3)}
    if len(expected)!=30 or {key(c) for c in expected}!=required:raise ValueError('registered rescue coverage differs')
    if set(reports)!={c['id'] for c in expected}:raise ValueError('all paired base/rescue reports required')
    representations={(('research-relu-max-pairs-v1',r['evaluation_build_sha256'])
                      if 'reexport_receipt' in r else ('original',None)) for r in reports.values()}
    if len(representations)!=1:raise ValueError('mixed base/rescue representations')
    scores={choice:[] for choice in choices}
    for cell in expected:
        row=reports[cell['id']]
        if row['cell']!=cell:raise ValueError('rescue report identity differs')
        scores[key(cell)[0]].append(human_score(row))
    means={choice:sum(values)/len(values) for choice,values in scores.items()}
    best=max(means.values())
    winner=next(choice for choice in choices if best-means[choice]<=1e-12)
    return dict(family=family,outer=outer,base_setting=setting,choice_scores=means,selected_choice=winner,
                required_reports=30,reported_cells=sorted(reports),outer_results_read=False)


def verified_candidate(root,cell,representation='original',*,allow_undefined_failure=False):
    """Shared final-checkpoint and diagnostic provenance gate for neural fits."""
    if representation not in ('original','research-relu-max-pairs-v1'):
        raise ValueError('unknown candidate representation')
    repaired=representation=='research-relu-max-pairs-v1'
    if repaired and cell['family'] not in MAX_PAIR_FAMILIES:raise ValueError('unqualified max-pair family')
    pins={}
    directory=root/('reexport_max_pairs' if repaired else 'runs')/cell['id']
    markers=list(directory.glob('attempt-*/final/COMPLETE.json' if repaired else 'attempt-*/evaluation/COMPLETE.json'))
    if len(markers)!=1:raise ValueError('one completed candidate required for '+cell['id'])
    marker=markers[0];path=marker.parent/'REPORT.json'
    if json.loads(marker.read_text())['report_sha256']!=sha(path):raise ValueError('candidate report drift')
    row=json.loads(path.read_text())
    if row['plan_sha256']!=sha(root/'PLAN.json'):raise ValueError('candidate plan drift')
    run_dir=root/'runs'/cell['id']/marker.parent.parent.name
    run=json.loads((run_dir/'RUN.json').read_text());result=json.loads((run_dir/'RESULT.json').read_text())
    if run['cell']!=cell or result['cell']!=cell or run['plan_sha256']!=sha(root/'PLAN.json'):
        raise ValueError('candidate training identity differs')
    if cell['stage']=='rescue' and run['selection_policy_sha256']!=sha(REPO/'docs/SHAPING_RESCUE_SELECTION.md'):
        raise ValueError('rescue selection policy drift')
    if result['stage']!='trained' or result['exit_code']!=0 or row['checkpoint_epoch']!=run['epochs']-1:
        raise ValueError('candidate is not a completed final checkpoint')
    original=run_dir/run.get('fit_directory','.')/'checkpoints'/f"ckpt_epoch{run['epochs']-1:03}.bin"
    if sha(original)!=result['bake_sha256']:raise ValueError('candidate source model drift')
    if result['plan_sha256']!=run['plan_sha256'] or result['epochs']!=run['epochs']:
        raise ValueError('candidate terminal provenance differs')
    cpath=marker.parent.parent/'CONVERGENCE.json'
    if allow_undefined_failure and not cpath.exists():
        if repaired or cell['stage']!='inner':raise ValueError('undefined failure retention requires original inner fit')
        if row['cell']!=cell or row['outer_results_read'] or row['checkpoint_sha256']!=sha(original):
            raise ValueError('excluded candidate identity, exposure or model differs')
        from shaping_selection import verify_undefined_failure
        failure_path=marker.parent.parent/'CONVERGENCE_FAILURE.json'
        verify_undefined_failure(failure_path,cell,row)
        failure=json.loads(failure_path.read_text());pins.update(failure['evidence_sha256'])
        for p in (path,marker,failure_path,run_dir/'RUN.json',run_dir/'RESULT.json',original,
                  run_dir/run.get('fit_directory','.')/'train.log'):pins[str(p)]=sha(p)
        return row,failure,pins
    c=json.loads(cpath.read_text())
    if c['cell']!=cell or c['outer_results_read']:raise ValueError('candidate convergence identity differs')
    if sha(run_dir/run.get('fit_directory','.')/'train.log')!=c['training_log_sha256']:
        raise ValueError('candidate training log drift')
    for name,pin in c['report_sha256'].items():
        if sha(name)!=pin:raise ValueError('candidate curve drift')
    if repaired:
        receipt_path=Path(row['reexport_receipt'])
        if sha(receipt_path)!=row['reexport_receipt_sha256']:raise ValueError('candidate re-export receipt drift')
        receipt=json.loads(receipt_path.read_text())
        if receipt['original_sha256']!=sha(original) or receipt['training_build_sha256']!=run['build_sha256']:
            raise ValueError('candidate re-export source drift')
        if c['representation']!=representation or c['evaluation_build_sha256']!=row['evaluation_build_sha256']:
            raise ValueError('candidate diagnostic representation differs')
        for name,pin in receipt['file_sha256'].items():
            if sha(name)!=pin:raise ValueError('candidate re-export artifact drift')
        pins[str(receipt_path)]=sha(receipt_path)
    elif row['checkpoint_sha256']!=sha(original):raise ValueError('candidate report model differs')
    pins.update({str(p):sha(p) for p in (path,cpath,run_dir/'RUN.json',run_dir/'RESULT.json')})
    if row['cell']!=cell or row['outer_results_read']:
        raise ValueError('candidate report identity or exposure differs')
    return row,c,pins


def select(args):
    _,cells,_=load_registration(args.study)
    setting,base_path,base_pin=selected_base(args.study,args.family,args.outer,cells)
    base=json.loads(Path(base_path).read_text());representation=base.get('representation','original')
    expected=[c for c in cells if c['family']==args.family and c['partition'] in (f'inner-{args.outer}-0',f'inner-{args.outer}-1')
              and (c['stage']=='rescue' or (c['stage']=='inner' and c['setting']==setting))]
    reports={};pins={base_path:base_pin};convergence={}
    for cell in expected:
        row,c,evidence=verified_candidate(args.study,cell,representation)
        reports[cell['id']]=row;convergence[cell['id']]=c;pins.update(evidence)
    decision=aggregate(cells,reports,args.family,args.outer,setting)
    winner=decision['selected_choice']
    selected=[c for c in expected if ('base' if c['stage']=='inner' else c['rescue'])==winner]
    extension=any(convergence[c['id']]['investigation_required'] for c in selected)
    decision.update(representation=representation,evidence_sha256=pins,plan_sha256=sha(args.study/'PLAN.json'),
                    selected_convergence={c['id']:convergence[c['id']] for c in selected},
                    further_investigation_required=extension,outer_evaluation_admitted=False,
                    status='selected_pending_convergence_resolution' if extension else 'selected_pending_outer_fold_gate')
    path=args.study/'selection'/f'outer-{args.outer}'/(args.family+'-rescue.json')
    if path.exists():raise ValueError('preserve existing rescue decision')
    path.write_bytes(encoded(decision));print(json.dumps(decision),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--family',choices=NEURAL,required=True)
    p.add_argument('--outer',type=int,choices=range(3),required=True)
    select(p.parse_args())
