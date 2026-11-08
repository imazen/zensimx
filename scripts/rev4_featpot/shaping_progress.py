"""Count registered evidence stages without treating process exit as scientific completion."""
from __future__ import annotations
import argparse,json,math,statistics,os,time,hashlib,fcntl
from functools import lru_cache
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import NEURAL,encoded,sha
from shaping_execution import load_registration


LABELS={
    'relu':'ReLU · unshaped','softplus':'Anchored Softplus',
    'silu_monotone':'Monotone SiLU','silu_diagnostic':'Ordinary SiLU · diagnostic',
    'product_joint_identity':'Product shaping · joint from identity',
    'product_pretrained_frozen':'Product shaping · pretrained, frozen',
    'product_pretrained_joint':'Product shaping · pretrained, joint',
    'separate_joint_identity':'Separate damage/fragility · joint from identity',
    'separate_pretrained_frozen':'Separate shaping · pretrained, frozen',
    'separate_pretrained_joint':'Separate shaping · pretrained, joint',
    'shaped_additive':'Shaped additive · linear solve','extra_trees':'Extra Trees',
}


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def panel_values(panel):
    from shaping_selection import HUMANS,human_score,UndefinedHumanPanel
    try:human=human_score(dict(outer_results_read=False,inner_validation=panel))
    except UndefinedHumanPanel:human=None
    data={}
    for name in (*HUMANS,'cid22_ssim2'):
        summary=panel['summary'][name]
        pooled=summary['pooled'];within=summary['within']
        data[name]=dict(pooled=pooled if finite(pooled) else None,within=within if finite(within) else None,
            mean=(pooled+within)/2 if finite(pooled) and finite(within) else None,
            rows=summary['rows'],sources=summary['sources'])
    return dict(human=human,datasets=data)


@lru_cache(maxsize=8192)
def saved_panel(path,pin,mtime,size):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('dashboard report drift: '+path)
    row=json.loads(raw)
    if row['outer_results_read']:raise ValueError('dashboard inner panel has outer exposure')
    return dict(cell=row['cell'],plan_sha256=row['plan_sha256'],
        epoch=row.get('checkpoint_epoch'),train=panel_values(row['train']),
        validation=panel_values(row['inner_validation']),
        active=row.get('active_output_coefficients'),gate=row.get('feature_gate'),
        structure=row.get('structure'),representation='max-pairs re-export' if 'reexport_receipt' in row else 'original')


def read_saved_panel(path,pin,cell,plan_pin):
    stat=path.stat();value=saved_panel(str(path),pin,stat.st_mtime_ns,stat.st_size)
    if value['cell']!=cell or value['plan_sha256']!=plan_pin:raise ValueError('dashboard panel identity differs')
    return value


def average(values,expected):
    good=[v for v in values if finite(v)]
    complete=len(values)==expected and len(good)==expected
    return dict(mean=statistics.fmean(good) if complete else None,
        sd=statistics.stdev(good) if complete and expected>1 else None,
        finite=len(good),required=expected)


def paired_delta(rows,baseline):
    left={r['cell']['seed']:r for r in rows};right={r['cell']['seed']:r for r in baseline}
    if None in left or set(left)!=set(right) or not left:return None
    values=[]
    for seed in left:
        a=left[seed].get('panel');b=right[seed].get('panel')
        if not a or not b or not finite(a['validation']['human']) or not finite(b['validation']['human']):return None
        values.append(a['validation']['human']-b['validation']['human'])
    return average(values,len(values))


def comparison_rows(records):
    groups={}
    for record in records:
        c=record['cell'];key=(c['stage'],c['partition'],c['setting'],c.get('rescue'),c['family'])
        groups.setdefault(key,[]).append(record)
    undefined=set()
    for r in records:
        c=r['cell']
        if c['stage']=='inner' and r.get('panel') and r['panel']['validation']['human'] is None:
            undefined.add((c['family'],c['partition'].split('-')[1],c['setting']))
    result=[]
    for (stage,partition,setting,rescue,family),items in groups.items():
        n=len(items);panels=[r['panel'] for r in items if r.get('panel')]
        row=dict(stage=stage,partition=partition,setting=setting,rescue=rescue,family=family,
            label=LABELS.get(family,family),required=n,reported=len(panels),
            trained=sum(r['trained'] for r in items),live=sum(r['state']=='running' for r in items),
            human=average([p['validation']['human'] for p in panels],n),
            train=average([p['train']['human'] for p in panels],n),
            ineligible=(family,partition.split('-')[1],setting) in undefined if stage=='inner' else False,
            original_failures=sum(r['original_failed_gate'] for r in items),
            investigations=sum(r['investigation'] for r in items),datasets={},
            active_coefficients=statistics.median([p['active'] for p in panels if p['active'] is not None]) if any(p['active'] is not None for p in panels) else None,
            cells=[r['cell']['id'] for r in items])
        for dataset in ('kadid','tid2013','konfig','cid22_a25','cid22_ssim2'):
            row['datasets'][dataset]={metric:average([p['validation']['datasets'][dataset][metric] for p in panels],n)
                                     for metric in ('pooled','within','mean')}
        baseline=groups.get(('inner',partition,setting,None,'relu'),[])
        row['delta']=paired_delta(items,baseline) if stage=='inner' and not row['ineligible'] else None
        result.append(row)
    return result


def corrected_program(root):
    path=root/'corrected_thresholds/REGISTRATION.json'
    if not path.exists():return None
    record=json.loads(path.read_text())
    if (record['schema']!='corrected-shaped-threshold-registration-v1' or record['core_fit_target']!=1461
        or record['corrected_condition_count']!=288 or len(record['omitted_original_conditions'])!=108
        or record['outer_results_read'] or record['settings']!=[0,1,2,3] or record['paired_seeds']!=[0,1,2]):
        raise ValueError('corrected dashboard registration differs')
    matches=[(Path(p),pin) for p,pin in record['evidence_sha256'].items()
             if Path(p).name=='THRESHOLD_COORDINATE_APPROVAL_20261108.json']
    if len(matches)!=1 or sha(matches[0][0])!=matches[0][1]:raise ValueError('corrected dashboard approval differs')
    approval=json.loads(matches[0][0].read_text())
    if approval['answer']!='Approve four settings: 1,461 fits':raise ValueError('corrected dashboard owner choice differs')
    result=dict(core_fit_target=1461,corrected_condition_count=288,omitted_original_conditions=108,
                registration_sha256=sha(path),execution_activated=(path.parent/'ACTIVATION.json').exists())
    compatibility=root.parent/'corrected-control-qualification-20261108/REGISTRATION.json'
    if compatibility.exists():
        from shaping_corrected_view import trajectory
        cases=json.loads(compatibility.read_text())['cases'];qualified=[]
        for family,case in cases.items():
            proof=(root.parent/'optimized-runtime-neutral-qualification-20261108/verified-evidence-v1/trajectory-check-20261108/RESULT.json'
                if family=='silu_monotone' else root.parent/'control-compatibility-collected-20261108'/family/'RESULT.json')
            if not proof.exists():continue
            value=json.loads(proof.read_text());trajectory(value,family,case)
            if value.get('original_result_sha256',value.get('source_result_sha256'))!=sha(Path(case['run'])/'RESULT.json'):
                raise ValueError('dashboard compatibility source result differs')
            qualified.append(family)
        result.update(compatibility_classes_required=len(cases),compatibility_classes_qualified=sorted(qualified))
    observation=root.parent/'native-supervision-20261108/r5600g.json'
    if observation.exists():
        native=json.loads(observation.read_text());age=(datetime.now(timezone.utc)-datetime.fromisoformat(native['checked_utc'])).total_seconds()
        result['control_observation']=dict(checked_utc=native['checked_utc'],fresh=0<=age<=120,
            running={row['cell']['family']:row['checkpoint_count'] for row in native['native_jobs']
                if row.get('cell') and 'corrected-control-qualification-20261108' in Path(row['run']).parts})
    return result


def dashboard_data(root,result,workers):
    from shaping_completion import FAMILIES,MAX_PAIR_FAMILIES
    from shaping_execution import live_status
    plan,cells,_=load_registration(root);plan_pin=result['plan_sha256']
    local=live_status(root,persist=False,emit=False)
    observations={r['cell']:dict(**r,host='dev',fresh=True) for r in local['runs']}
    fleet=[dict(host='dev',live=local['live'],ssd_free_bytes=local['ssd_free_bytes'],fresh=True,checked_utc=local['checked_utc'])]
    for host,path in workers.items():
        status=json.loads(path.read_text());health=result['workers'][host]
        fleet.append(dict(host=host,live=status['live'],ssd_free_bytes=status['ssd_free_bytes'],
                          fresh=health['fresh'],checked_utc=status['checked_utc']))
        for row in status['runs']:
            if row['verified_live']:observations[row['cell']]=dict(**row,host=host,fresh=health['fresh'])
    audited={r['cell']:r for r in result['cells']};records=[]
    for cell in cells:
        if cell['stage'] not in ('inner','fresh_control_inner','rescue') or not audited[cell['id']]['required']:continue
        directory=root/'runs'/cell['id'];a=audited[cell['id']]
        trained=any(v['fit_success'] for v in a['attempts']);seen=observations.get(cell['id'],{})
        state='trained' if trained else 'running' if seen.get('verified_live') and seen.get('fresh') else 'last observed running' if seen.get('verified_live') else seen.get('terminal') or 'pending'
        record=dict(cell=cell,trained=trained,state=state,host=seen.get('host'),
            last_epoch=seen.get('last_epoch'),required_epochs=seen.get('required_epochs'),
            original_failed_gate=any(v['original_failed_gates'] for v in a['attempts']),
            investigation=any(v['convergence_failure'] or any(d['investigation_required'] for d in v['diagnostics'].values()) for v in a['attempts']),
            diagnostics_ready=cell['family'] not in NEURAL or any(v['convergence_failure'] or v['diagnostics'] for v in a['attempts']),
            archive=a['archive_receipt_verified'],panel=None,curve=[],reasons=[])
        repaired=cell['family'] in MAX_PAIR_FAMILIES
        base=root/'reexport_max_pairs'/cell['id'] if repaired else directory
        markers=list(base.glob('attempt-*/final/COMPLETE.json' if repaired else 'attempt-*/evaluation/COMPLETE.json'))
        if len(markers)>1:raise ValueError('ambiguous dashboard final: '+cell['id'])
        if markers:
            marker=markers[0];path=marker.parent/'REPORT.json';pin=json.loads(marker.read_text())['report_sha256']
            record['panel']=read_saved_panel(path,pin,cell,plan_pin)
            run=directory/marker.parent.parent.name
            identity=json.loads((run/'RUN.json').read_text())
            if cell['family'] in NEURAL and record['panel']['epoch']!=identity['epochs']-1:
                raise ValueError('dashboard requires registered final checkpoint')
            for name in ('CONVERGENCE.json','CONVERGENCE_FAILURE.json'):
                conv=marker.parent.parent/name
                if not conv.exists():continue
                content=json.loads(conv.read_text())
                if content['cell']!=cell or content['outer_results_read']:raise ValueError('dashboard diagnostic identity differs')
                record['reasons']=content.get('reasons',[])
                pins=content.get('report_sha256',content.get('evidence_sha256',{}))
                points={}
                for name,pin in pins.items():
                    path=Path(name)
                    if path.name!='REPORT.json':continue
                    if not path.resolve().is_relative_to(root.resolve()):raise ValueError('dashboard diagnostic outside study')
                    p=read_saved_panel(path,pin,cell,plan_pin)
                    if p['epoch'] is not None:points[p['epoch']]=dict(epoch=p['epoch'],train=p['train']['human'],validation=p['validation']['human'],active=p['active'])
                record['curve']=[points[k] for k in sorted(points)]
                break
        records.append(record)
    comparisons=comparison_rows(records)
    primary=[r for r in comparisons if r['stage']=='inner' and r['setting']==0]
    coverage={p:sum(r['reported']/r['required'] for r in primary if r['partition']==p) for p in {r['partition'] for r in primary}}
    default=min(coverage,key=lambda p:(-coverage[p],p))
    return dict(schema='zensim-shaping-stats-v1',generated_utc=result['checked_utc'],plan_sha256=plan_pin,
        counts=result['counts'],required_cells=result['required_cells'],fleet=fleet,
        scope_amendment=result.get('scope_amendment'),
        corrected_program=corrected_program(root),
        supervisors=result['supervisors'],all_supervisors_live=result['all_requested_supervisor_processes_verified_live'],
        comparisons=comparisons,records=records,default_partition=default,
        neural_settings=plan['neural_settings'],families=list(FAMILIES),
        methodology=dict(human='Equal mean of pooled and within-source SROCC on each of four human datasets. Teacher excluded.',
            uncertainty='Mean and sample SD across all registered paired seeds; not source-bootstrap confidence intervals.',
            exposure='Primary neural fits: 240 epochs; pretrained shapes: 120-epoch prefix plus 120-epoch fresh head. Fresh controls: 120 epochs. 50,000 pairs per epoch.',
            scope='Source-disjoint inner validation. Outer and memorization comparisons remain pending until their registered gates pass.',
            completeness='An incomplete or undefined seed group has no aggregate score. A single undefined paired human fit makes its entire outer-group setting ineligible.'))


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
    'rescue_archive':'shaping_archive_supervision.py',
    'statistics_dashboard':'shaping_progress.py',
    'worker_evaluation':'shaping_evaluation_worker.py',
    'native_training_observer':'shaping_native_watch.py',
    **{f'selection_outer_{outer}':'shaping_selection_watch.py' for outer in range(3)},
}
ARCHIVE_FLAGS={'primary_archive':None,'supplemental_archive':'--supplemental',
    'later_neural_archive':'--later-neural','later_control_archive':'--later-controls',
    'failed_original_archive':'--failed-original','rescue_archive':'--rescue'}


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
    if role.startswith('selection_outer_'):
        matches=matches and argument('--outer')==role.rsplit('_',1)[1]
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
    from shaping_reduced_scope import coverage
    return dict(checked_utc=now.isoformat(),plan_sha256=plan_pin,required_cells=len(required),
                scope_amendment=coverage(root),
                counts=counts,workers=observed,cells=rows,overall_experiments_complete=False,
                supervisors=supervising,
                all_requested_supervisor_processes_verified_live=all(v['verified_live'] is True for v in supervising.values()) if supervising else None,
                interpretation='Fit, final-report, convergence and archive counts are distinct. Inner reports do not complete outer/variant comparisons. Cell-level archive receipts and manifest identity checked here; central batch archives are not counted. File readback occurred during archival.')


def atomic_json(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.writing')
    temporary.write_text(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n')
    temporary.replace(path)


def dashboard_claim(directory):
    directory.mkdir(parents=True,exist_ok=True)
    stream=(directory/'.publisher.lock').open('a')
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close();raise ValueError('another statistics publisher owns this output')
    return stream


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--worker-status',action='append',default=[])
    p.add_argument('--supervisor-status',action='append',default=[],metavar='ROLE=PATH',
                   help='verify a recorded local PID without restarting it; roles: '+', '.join(SUPERVISOR_SCRIPTS))
    p.add_argument('--dashboard',type=Path,help='publish a local HTML/JSON view of saved inner-study results only')
    p.add_argument('--watch',action='store_true',help='refresh the requested view without fitting or scoring')
    p.add_argument('--poll-seconds',type=int,default=60)
    a=p.parse_args()
    if a.poll_seconds<10:p.error('poll-seconds must be at least 10')
    if a.watch and not a.dashboard:p.error('watch requires dashboard output')
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
    if a.dashboard:
        dashboard_lock=dashboard_claim(a.dashboard)
        template=Path(__file__).with_name('shaping_dashboard.html').read_text()
        temp=a.dashboard/f'index.html.{os.getpid()}.writing';temp.write_text(template);temp.replace(a.dashboard/'index.html')
        if a.watch:atomic_json(a.dashboard/'publisher-status.json',dict(pid=os.getpid(),stage='starting',
            checked_utc=datetime.now(timezone.utc).isoformat(),error=None))
    while True:
        try:
            result=audit(a.study,workers,supervisors)
            if a.dashboard:
                data=dashboard_data(a.study,result,workers)
                atomic_json(a.dashboard/'stats.json',data)
                atomic_json(a.dashboard/'publisher-status.json',dict(pid=os.getpid(),stage='watching' if a.watch else 'snapshot',
                    checked_utc=datetime.now(timezone.utc).isoformat(),last_success_utc=result['checked_utc'],error=None))
            atomic_json(a.study/'PROGRESS_AUDIT.json',result)
            print(json.dumps({k:v for k,v in result.items() if k!='cells'}),flush=True)
        except Exception as error:
            if not a.dashboard or not a.watch:raise
            atomic_json(a.dashboard/'publisher-status.json',dict(pid=os.getpid(),stage='observation_failed',
                checked_utc=datetime.now(timezone.utc).isoformat(),error=str(error)))
            print(json.dumps(dict(stage='observation_failed',error=str(error))),flush=True)
        if not a.watch:break
        time.sleep(a.poll_seconds)
    if a.dashboard:dashboard_lock.close()


if __name__=='__main__':main()
