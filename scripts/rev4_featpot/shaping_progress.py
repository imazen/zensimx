"""Count registered evidence stages without treating process exit as scientific completion."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import NEURAL,encoded,sha
from shaping_execution import load_registration


def report_marker(path,cell,plan):
    if not path.exists():return False
    receipt=json.loads(path.read_text());report=path.parent/'REPORT.json'
    if receipt['report_sha256']!=sha(report):raise ValueError('report drift: '+str(report))
    row=json.loads(report.read_text())
    if row['cell']!=cell or row['plan_sha256']!=plan:raise ValueError('report identity drift')
    return True


def audit(root,workers):
    _,cells,_=load_registration(root);plan_pin=sha(root/'PLAN.json')
    required={c['id'] for c in cells if c['stage'] not in ('rescue','resource_pilot')}
    for path in root.glob('RESCUE_TRIGGER_OUTER_*.json'):
        trigger=json.loads(path.read_text());outer=int(path.stem.rsplit('_',1)[1])
        expected={c['id'] for c in cells if c['stage']=='rescue' and c['partition'].startswith(f'inner-{outer}-')}
        if set(trigger['required_rescue_cells'])!=expected or trigger['outer_results_read']:raise ValueError('rescue trigger differs')
        required|=expected
    for path in (root/'extensions').glob('outer-*/REGISTRATION.json'):
        extension=json.loads(path.read_text())
        from shaping_extension import resolve
        for cell in extension['cells']:
            if resolve(root,cell['id'])['cell']!=cell:raise ValueError('extension drift')
            cells.append(cell);required.add(cell['id'])
    rows=[]
    for cell in cells:
        directory=root/'runs'/cell['id'];attempts=[]
        for path in sorted(directory.glob('attempt-*/RESULT.json')):
            result=json.loads(path.read_text())
            if result['cell']!=cell or result['plan_sha256']!=plan_pin:raise ValueError('result identity drift')
            run=path.parent
            original=report_marker(run/'evaluation/COMPLETE.json',cell,plan_pin)
            supplemental_root=root/'reexport_max_pairs'/cell['id']/run.name
            supplemental=report_marker(supplemental_root/'final/COMPLETE.json',cell,plan_pin)
            diagnostics={}
            for label,base in [('original',run),('supplemental',supplemental_root)]:
                p=base/'CONVERGENCE.json'
                if not p.exists():continue
                c=json.loads(p.read_text())
                if c['cell']!=cell or c['outer_results_read']:raise ValueError('convergence identity drift')
                for name,pin in c['report_sha256'].items():
                    if sha(name)!=pin:raise ValueError('diagnostic report drift')
                diagnostics[label]=dict(investigation_required=c['investigation_required'],reasons=c['reasons'])
            attempts.append(dict(attempt=run.name,fit_success=result['exit_code']==0 and result['stage'] in ('trained','complete'),
                                 original_final_report=original,supplemental_final_report=supplemental,
                                 original_failed_gates=(run/'EVALUATION_FAILURES.json').exists(),diagnostics=diagnostics))
        archive=False
        if (directory/'MIRROR_VERIFIED.json').exists():
            receipt=json.loads((directory/'MIRROR_VERIFIED.json').read_text());destination=Path(receipt['destination'])
            if not receipt['all_artifact_sha256_verified'] or sha(directory/'SHA256SUMS')!=sha(destination/'SHA256SUMS'):
                raise ValueError('archive receipt/manifest differs')
            archive=True
        rows.append(dict(cell=cell['id'],stage=cell['stage'],family=cell['family'],required=cell['id'] in required,
                         attempts=attempts,archive_receipt_verified=archive))
    counts={}
    for stage in sorted({r['stage'] for r in rows}):
        selected=[r for r in rows if r['stage']==stage and r['required']]
        counts[stage]=dict(required=len(selected),successful_fit_cells=sum(any(a['fit_success'] for a in r['attempts']) for r in selected),
            original_final_report_cells=sum(any(a['original_final_report'] for a in r['attempts']) for r in selected),
            supplemental_final_report_cells=sum(any(a['supplemental_final_report'] for a in r['attempts']) for r in selected),
            neural_diagnostic_receipt_cells=sum(r['family'] in NEURAL and any(a['diagnostics'] for a in r['attempts']) for r in selected),
            cells_with_investigation_trigger=sum(any(d['investigation_required'] for a in r['attempts'] for d in a['diagnostics'].values()) for r in selected),
            cells_with_original_failed_gates=sum(any(a['original_failed_gates'] for a in r['attempts']) for r in selected),
            archive_receipt_cells=sum(r['archive_receipt_verified'] for r in selected))
    observed={};now=datetime.now(timezone.utc)
    for host,path in workers.items():
        status=json.loads(path.read_text());age=(now-datetime.fromisoformat(status['checked_utc'])).total_seconds()
        observed[host]=dict(status_sha256=sha(path),checked_utc=status['checked_utc'],age_seconds=age,
            fresh=0<=age<=120,observed_live=status['live'],unexpectedly_stopped=status['unexpectedly_stopped'],
            ssd_free_bytes=status['ssd_free_bytes'])
    return dict(checked_utc=now.isoformat(),plan_sha256=plan_pin,required_cells=len(required),
                counts=counts,workers=observed,cells=rows,overall_experiments_complete=False,
                interpretation='Fit, final-report, convergence and archive counts are distinct. Inner reports do not complete outer/variant comparisons. Cell-level archive receipts and manifest identity checked here; central batch archives are not counted. File readback occurred during archival.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--worker-status',action='append',default=[]);a=p.parse_args()
    workers={}
    for item in a.worker_status:
        host,path=item.split('=',1)
        if host in workers:p.error('duplicate host')
        workers[host]=Path(path)
    result=audit(a.study,workers)
    temporary=a.study/'PROGRESS_AUDIT.json.writing';temporary.write_bytes(encoded(result));temporary.replace(a.study/'PROGRESS_AUDIT.json')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'}),flush=True)
