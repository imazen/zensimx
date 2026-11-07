"""Inner-only base selection with complete paired coverage and teacher exclusion."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from shaping_completion import FAMILIES,NEURAL,encoded,sha
from shaping_execution import load_registration

HUMANS=('kadid','tid2013','konfig','cid22_a25')


def human_score(report):
    if report['outer_results_read']:raise ValueError('outer-exposed report cannot select settings')
    panel=report['inner_validation'];summary=panel['summary']
    values=[summary[p][k] for p in HUMANS for k in ('pooled','within')]
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
        raise ValueError('undefined human panel')
    score=sum(values)/len(values)
    if not math.isclose(score,panel['selection_score'],rel_tol=0.,abs_tol=1e-12):
        raise ValueError('selection score differs from equal-human pooled/within composite')
    return score


def choose(scores):
    if set(scores)!=set(range(4)):raise ValueError('all four settings required')
    if not all(math.isfinite(v) for v in scores.values()):raise ValueError('nonfinite setting score')
    best=max(scores.values())
    return min(setting for setting,value in scores.items() if best-value<=1e-12)


def aggregate(cells,reports,family,outer):
    expected=[c for c in cells if c['stage']=='inner' and c['family']==family
              and c['partition'] in (f'inner-{outer}-0',f'inner-{outer}-1')]
    seeds=(None,) if family=='shaped_additive' else (0,1,2)
    coverage={(c['setting'],c['partition'],c['seed']) for c in expected}
    required={(setting,f'inner-{outer}-{fold}',seed) for setting in range(4) for fold in range(2) for seed in seeds}
    if coverage!=required or len(expected)!=len(required):raise ValueError('registered tuning coverage differs')
    if set(reports)!={c['id'] for c in expected}:raise ValueError('missing or extra paired reports')
    scores={setting:[] for setting in range(4)}
    for cell in expected:
        report=reports[cell['id']]
        if report['cell']!=cell:raise ValueError('report identity differs')
        scores[cell['setting']].append(human_score(report))
    means={setting:sum(values)/len(values) for setting,values in scores.items()}
    return dict(family=family,outer=outer,setting_scores=means,selected_setting=choose(means),
                required_reports=len(required),reported_cells=sorted(reports),
                outer_results_read=False,outer_evaluation_admitted=False,
                status='base_selected_pending_neural_convergence_and_rescue_resolution')


def select(args):
    _,cells,_=load_registration(args.study)
    expected=[c for c in cells if c['stage']=='inner' and c['family']==args.family
              and c['partition'] in (f'inner-{args.outer}-0',f'inner-{args.outer}-1')]
    reports={};pins={};convergence={}
    for cell in expected:
        candidates=list((args.study/'runs'/cell['id']).glob('attempt-*/evaluation/COMPLETE.json'))
        if len(candidates)!=1:raise ValueError('one complete report required for '+cell['id'])
        marker=candidates[0];path=marker.parent/'REPORT.json';receipt=json.loads(marker.read_text())
        if receipt['report_sha256']!=sha(path):raise ValueError('report drift')
        row=json.loads(path.read_text())
        if row['plan_sha256']!=sha(args.study/'PLAN.json'):raise ValueError('report plan differs')
        reports[cell['id']]=row;pins[str(path)]=sha(path)
        if args.family in NEURAL:
            cpath=marker.parent.parent/'CONVERGENCE.json'
            c=json.loads(cpath.read_text())
            if c['cell']!=cell or c['outer_results_read']:raise ValueError('convergence evidence differs')
            for report_path,pin in c['report_sha256'].items():
                if sha(report_path)!=pin:raise ValueError('convergence input drift')
            convergence[cell['id']]=c['investigation_required'];pins[str(cpath)]=sha(cpath)
    result=aggregate(cells,reports,args.family,args.outer)
    result.update(plan_sha256=sha(args.study/'PLAN.json'),evidence_sha256=pins,
                  investigation_triggered=any(convergence.values()),convergence=convergence,driver_sha256=sha(__file__))
    dest=args.study/'selection'/f'outer-{args.outer}';dest.mkdir(parents=True,exist_ok=True)
    path=dest/(args.family+'-base.json')
    if path.exists():raise ValueError('preserve existing selection receipt')
    path.write_bytes(encoded(result));print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--family',choices=FAMILIES,required=True)
    parser.add_argument('--outer',type=int,choices=range(3),required=True)
    select(parser.parse_args())
