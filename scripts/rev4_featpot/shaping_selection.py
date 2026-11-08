"""Inner-only base selection with complete paired coverage and teacher exclusion."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from shaping_completion import MAX_PAIR_FAMILIES,FAMILIES,NEURAL,REPO,encoded,sha
from shaping_execution import load_registration

HUMANS=('kadid','tid2013','konfig','cid22_a25')
INELIGIBILITY_POLICY='whole-setting-undefined-human-v1'


class UndefinedHumanPanel(ValueError):
    pass


def human_score(report):
    if report['outer_results_read']:raise ValueError('outer-exposed report cannot select settings')
    panel=report['inner_validation'];summary=panel['summary']
    values=[summary[p][k] for p in HUMANS for k in ('pooled','within')]
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
        raise UndefinedHumanPanel('undefined human panel')
    score=sum(values)/len(values)
    if not math.isclose(score,panel['selection_score'],rel_tol=0.,abs_tol=1e-12):
        raise ValueError('selection score differs from equal-human pooled/within composite')
    return score


def choose(scores):
    if set(scores)!=set(range(4)):raise ValueError('all four settings required')
    if not all(math.isfinite(v) for v in scores.values()):raise ValueError('nonfinite setting score')
    best=max(scores.values())
    return min(setting for setting,value in scores.items() if best-value<=1e-12)


def aggregate(cells,reports,family,outer,policy=None):
    if policy not in (None,INELIGIBILITY_POLICY):raise ValueError('unknown selection policy')
    expected=[c for c in cells if c['stage']=='inner' and c['family']==family
              and c['partition'] in (f'inner-{outer}-0',f'inner-{outer}-1')]
    seeds=(None,) if family=='shaped_additive' else (0,1,2)
    coverage={(c['setting'],c['partition'],c['seed']) for c in expected}
    required={(setting,f'inner-{outer}-{fold}',seed) for setting in range(4) for fold in range(2) for seed in seeds}
    if coverage!=required or len(expected)!=len(required):raise ValueError('registered tuning coverage differs')
    if set(reports)!={c['id'] for c in expected}:raise ValueError('missing or extra paired reports')
    representations={(('research-relu-max-pairs-v1',r['evaluation_build_sha256'])
                      if 'reexport_receipt' in r else ('original',None)) for r in reports.values()}
    if len(representations)!=1:raise ValueError('mixed evaluation representations or repaired runtimes')
    scores={setting:[] for setting in range(4)}
    undefined={}
    for cell in expected:
        report=reports[cell['id']]
        if report['cell']!=cell:raise ValueError('report identity differs')
        try:score=human_score(report)
        except UndefinedHumanPanel:
            if policy is None:raise
            undefined.setdefault(cell['setting'],[]).append(cell['id'])
        else:scores[cell['setting']].append(score)
    means={setting:None if setting in undefined else sum(values)/len(values) for setting,values in scores.items()}
    eligible={k:v for k,v in means.items() if v is not None}
    if not eligible:raise ValueError('no fully defined setting remains')
    best=max(eligible.values())
    winner=min(k for k,v in eligible.items() if best-v<=1e-12)
    extra={} if policy is None else dict(selection_policy=policy,
        ineligible_settings={str(k):dict(reason='undefined human panel in paired fit',undefined_cells=sorted(v)) for k,v in undefined.items()})
    return dict(family=family,outer=outer,setting_scores=means,selected_setting=winner,**extra,
                required_reports=len(required),reported_cells=sorted(reports),
                outer_results_read=False,outer_evaluation_admitted=False,
                status='base_selected_pending_neural_convergence_and_rescue_resolution')


def verify_undefined_failure(path,cell,final_report):
    """Retain a failed diagnostic as failure, never as convergence qualification."""
    try:human_score(final_report)
    except UndefinedHumanPanel:pass
    else:raise ValueError('finite final panel still requires convergence evidence')
    failure=json.loads(path.read_text())
    if (failure['cell']!=cell or failure['outer_results_read'] or
        failure['scientifically_complete'] or not failure['requires_investigation']):
        raise ValueError('invalid undefined convergence failure')
    for name,pin in failure['evidence_sha256'].items():
        if sha(name)!=pin:raise ValueError('undefined convergence evidence drift')
    from shaping_convergence import diagnostics,UndefinedInnerComposite
    from shaping_evaluation import diagnostic_epochs
    import re
    run=path.parent
    identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
    if (identity['cell']!=cell or result['cell']!=cell or result['stage']!='trained' or result['exit_code']!=0
        or final_report['plan_sha256']!=identity['plan_sha256']
        or final_report['checkpoint_epoch']!=identity['epochs']-1):
        raise ValueError('undefined fit identity or exposure differs')
    scores={};active={}
    for name in failure['evidence_sha256']:
        candidate=Path(name)
        if candidate.name!='REPORT.json':continue
        row=json.loads(candidate.read_text())
        marker=json.loads(candidate.with_name('DIAGNOSTIC_COMPLETE.json').read_text())
        if row['cell']!=cell or row['outer_results_read'] or marker['report_sha256']!=sha(candidate):
            raise ValueError('undefined diagnostic identity differs')
        epoch=row['checkpoint_epoch'];scores[epoch]=row['inner_validation']['selection_score']
        active[epoch]=row['active_output_coefficients']
    if sorted(scores)!=list(diagnostic_epochs(identity['epochs'])):raise ValueError('incomplete undefined diagnostic coverage')
    log=(run/identity.get('fit_directory','.')/'train.log').read_text()
    losses={int(k):float(v) for k,v in re.findall(r'epoch\s+(\d+)\s+\|[^\n]*?loss=([^ |]+)',log)}
    if sorted(losses)!=list(range(identity['epochs'])):raise ValueError('incomplete undefined fit exposure')
    try:diagnostics([losses[k] for k in sorted(losses)],scores,active)
    except UndefinedInnerComposite as error:
        if str(error)!=failure['reason']:raise ValueError('undefined failure reason differs')
    else:raise ValueError('undefined diagnostic failure could not be reproduced')


def select(args):
    from shaping_corrected_view import FAMILIES as corrected_families,require_activation
    if args.family in corrected_families:require_activation(args.study)
    _,cells,_=load_registration(args.study)
    repaired=getattr(args,'reexport_max_pairs',False)
    if repaired and args.family not in MAX_PAIR_FAMILIES:raise ValueError('unqualified max-pair family')
    expected=[c for c in cells if c['stage']=='inner' and c['family']==args.family
              and c['partition'] in (f'inner-{args.outer}-0',f'inner-{args.outer}-1')]
    reports={};pins={};convergence={}
    for cell in expected:
        candidates=list((args.study/('reexport_max_pairs' if repaired else 'runs')/cell['id']).glob(
            'attempt-*/final/COMPLETE.json' if repaired else 'attempt-*/evaluation/COMPLETE.json'))
        if len(candidates)!=1:raise ValueError('one complete report required for '+cell['id'])
        marker=candidates[0];path=marker.parent/'REPORT.json';receipt=json.loads(marker.read_text())
        if receipt['report_sha256']!=sha(path):raise ValueError('report drift')
        row=json.loads(path.read_text())
        if row['plan_sha256']!=sha(args.study/'PLAN.json'):raise ValueError('report plan differs')
        reports[cell['id']]=row;pins[str(path)]=sha(path)
        if args.family in NEURAL:
            cpath=marker.parent.parent/'CONVERGENCE.json'
            if not cpath.exists():
                failure=marker.parent.parent/'CONVERGENCE_FAILURE.json'
                verify_undefined_failure(failure,cell,row)
                convergence[cell['id']]=True;pins[str(failure)]=sha(failure)
                continue
            c=json.loads(cpath.read_text())
            if c['cell']!=cell or c['outer_results_read']:raise ValueError('convergence evidence differs')
            if repaired:
                if c['representation']!='research-relu-max-pairs-v1' or c['evaluation_build_sha256']!=row['evaluation_build_sha256']:
                    raise ValueError('final and diagnostic representations differ')
                receipt_path=Path(row['reexport_receipt'])
                if sha(receipt_path)!=row['reexport_receipt_sha256']:raise ValueError('final re-export receipt drift')
                receipt=json.loads(receipt_path.read_text())
                for name,pin in receipt['file_sha256'].items():
                    if sha(name)!=pin:raise ValueError('final re-export artifact drift')
                original=Path(c['original_training_run'])
                result=json.loads((original/'RESULT.json').read_text())
                if receipt['original_sha256']!=result['bake_sha256'] or row['checkpoint_epoch']!=result['epochs']-1:
                    raise ValueError('re-export did not use final scheduled checkpoint')
            for report_path,pin in c['report_sha256'].items():
                if sha(report_path)!=pin:raise ValueError('convergence input drift')
            convergence[cell['id']]=c['investigation_required'];pins[str(cpath)]=sha(cpath)
    result=aggregate(cells,reports,args.family,args.outer,INELIGIBILITY_POLICY)
    result.update(plan_sha256=sha(args.study/'PLAN.json'),evidence_sha256=pins,
                  selection_policy_sha256=sha(REPO/'docs/SHAPING_UNDEFINED_SETTING_POLICY.md'),
                  investigation_triggered=any(convergence.values()),convergence=convergence,driver_sha256=sha(__file__))
    result['representation']='research-relu-max-pairs-v1' if repaired else 'original'
    dest=args.study/'selection'/f'outer-{args.outer}';dest.mkdir(parents=True,exist_ok=True)
    from shaping_corrected_view import base_path
    path=base_path(args.study,args.family,args.outer)
    if path.exists():raise ValueError('preserve existing selection receipt')
    path.write_bytes(encoded(result));print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--family',choices=FAMILIES,required=True)
    parser.add_argument('--outer',type=int,choices=range(3),required=True)
    parser.add_argument('--reexport-max-pairs',action='store_true')
    select(parser.parse_args())
