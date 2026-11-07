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


SUPERVISOR_SCRIPTS={
    'inner_evaluation':'shaping_supervision.py','later_evaluation':'shaping_supervision.py',
    'relu_evaluation':'shaping_reexport_supervision.py','softplus_evaluation':'shaping_reexport_supervision.py',
    'primary_archive':'shaping_archive_supervision.py','supplemental_archive':'shaping_archive_supervision.py',
    'later_neural_archive':'shaping_archive_supervision.py','later_control_archive':'shaping_archive_supervision.py',
    'failed_original_archive':'shaping_archive_supervision.py','bootstrap':'shaping_bootstrap.py',
}
ARCHIVE_FLAGS={'primary_archive':None,'supplemental_archive':'--supplemental',
    'later_neural_archive':'--later-neural','later_control_archive':'--later-controls',
    'failed_original_archive':'--failed-original'}


def supervisor_observation(root,role,path,proc=Path('/proc'),now=None):
    """Observe the recorded process only; never restart or infer death from age."""
    if role not in SUPERVISOR_SCRIPTS:raise ValueError('unknown supervisor role')
    row=dict(status_file=str(path),expected_script=SUPERVISOR_SCRIPTS[role],verified_live=False)
    if not path.exists():return dict(row,process_state='status_missing')
    status=json.loads(path.read_text());pid=status['pid']
    if type(pid) is not int or pid<=0:raise ValueError('positive supervisor PID required')
    now=now or datetime.now(timezone.utc)
    age=(now-datetime.fromisoformat(status['checked_utc'])).total_seconds()
    row.update(pid=pid,status_sha256=sha(path),reported_stage=status['stage'],
               checked_utc=now.isoformat(),status_age_seconds=age,status_fresh=0<=age<=120)
    try:command=(proc/str(pid)/'cmdline').read_bytes().decode().split('\0')[:-1]
    except FileNotFoundError:return dict(row,process_state='recorded_process_missing')
    except PermissionError:return dict(row,verified_live=None,process_state='observation_unavailable')
    def argument(flag,default=None):
        if flag not in command:return default
        if command.count(flag)!=1 or command.index(flag)+1>=len(command):return None
        return command[command.index(flag)+1]
    script=next((v for v in command[1:] if not v.startswith('-')),'')
    matches=(bool(command) and Path(command[0]).name.startswith('python')
             and '-c' not in command and '-m' not in command
             and Path(script).name==SUPERVISOR_SCRIPTS[role]
             and argument('--study')==str(root) and '--watch' in command)
    if role in ('inner_evaluation','later_evaluation'):
        matches=matches and ('--later-only' in command)==(role=='later_evaluation')
    if role in ('relu_evaluation','softplus_evaluation'):
        matches=matches and argument('--family','relu')==role.split('_',1)[0]
    if role in ARCHIVE_FLAGS:
        flags=[flag for flag in ARCHIVE_FLAGS.values() if flag is not None and flag in command]
        matches=matches and flags==([] if ARCHIVE_FLAGS[role] is None else [ARCHIVE_FLAGS[role]])
    return dict(row,verified_live=bool(matches),
                process_state='verified_live' if matches else 'recorded_process_identity_differs')


def audit(root,workers,supervisors=None):
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
    for path in (root/'selection').glob('outer-*/OUTER_ADMISSION.json'):
        from shaping_outer_admission import resolve as resolve_outer
        outer=int(path.parent.name.rsplit('-',1)[1]);admission,_=resolve_outer(root,outer)
        for cell in admission['additional_cells']:
            if cell['id'] in {c['id'] for c in cells}:raise ValueError('conditional outer cell identity collision')
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
            failure_path=run/'CONVERGENCE_FAILURE.json'
            convergence_failure=False
            if failure_path.exists():
                failure=json.loads(failure_path.read_text())
                if failure['cell']!=cell or failure['outer_results_read'] or failure['scientifically_complete'] or not failure['requires_investigation']:
                    raise ValueError('convergence failure identity differs')
                for name,pin in failure['evidence_sha256'].items():
                    if sha(name)!=pin:raise ValueError('convergence failure evidence drift')
                convergence_failure=True
            attempts.append(dict(attempt=run.name,fit_success=result['exit_code']==0 and result['stage'] in ('trained','complete'),
                                 original_final_report=original,supplemental_final_report=supplemental,
                                 original_failed_gates=(run/'EVALUATION_FAILURES.json').exists(),convergence_failure=convergence_failure,diagnostics=diagnostics))
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
            cells_with_convergence_failure=sum(any(a['convergence_failure'] for a in r['attempts']) for r in selected),
            archive_receipt_cells=sum(r['archive_receipt_verified'] for r in selected))
    observed={};now=datetime.now(timezone.utc)
    for host,path in workers.items():
        status=json.loads(path.read_text());age=(now-datetime.fromisoformat(status['checked_utc'])).total_seconds()
        observed[host]=dict(status_sha256=sha(path),checked_utc=status['checked_utc'],age_seconds=age,
            fresh=0<=age<=120,observed_live=status['live'],unexpectedly_stopped=status['unexpectedly_stopped'],
            ssd_free_bytes=status['ssd_free_bytes'])
    supervising={role:supervisor_observation(root,role,path,now=now)
                 for role,path in (supervisors or {}).items()}
    return dict(checked_utc=now.isoformat(),plan_sha256=plan_pin,required_cells=len(required),
                counts=counts,workers=observed,cells=rows,overall_experiments_complete=False,
                supervisors=supervising,
                all_requested_supervisor_processes_verified_live=all(v['verified_live'] is True for v in supervising.values()) if supervising else None,
                interpretation='Fit, final-report, convergence and archive counts are distinct. Inner reports do not complete outer/variant comparisons. Cell-level archive receipts and manifest identity checked here; central batch archives are not counted. File readback occurred during archival.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--worker-status',action='append',default=[])
    p.add_argument('--supervisor-status',action='append',default=[],metavar='ROLE=PATH',
                   help='verify a recorded local PID without restarting it; roles: '+', '.join(SUPERVISOR_SCRIPTS))
    a=p.parse_args()
    workers={}
    for item in a.worker_status:
        host,path=item.split('=',1)
        if host in workers:p.error('duplicate host')
        workers[host]=Path(path)
    supervisors={}
    for item in a.supervisor_status:
        role,path=item.split('=',1)
        if role not in SUPERVISOR_SCRIPTS:p.error('unknown supervisor role: '+role)
        if role in supervisors:p.error('duplicate supervisor role')
        supervisors[role]=Path(path)
    result=audit(a.study,workers,supervisors)
    temporary=a.study/'PROGRESS_AUDIT.json.writing';temporary.write_bytes(encoded(result));temporary.replace(a.study/'PROGRESS_AUDIT.json')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'}),flush=True)
