"""Reproduce full-FIT setting decisions from complete pinned search evidence."""
from __future__ import annotations
import argparse,fcntl,json,os,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import NEURAL,ROLES,encoded,sha
from shaping_execution import load_registration
from shaping_full_fit_admission import resolve as resolve_admission
from shaping_full_fit_plan import select
from shaping_convergence import training_report


def candidate(root,cell,family,admission,admission_path):
    final=cell['stage']=='full_fit'
    if cell['stage'] not in ('full_fit_selection','full_fit'):raise ValueError('full-FIT candidate stage differs')
    final_schedule=None
    if final:
        from shaping_full_fit_execution import refit_cell
        _,_,_,verified,path,registered,resolved,final_schedule=refit_cell(root,cell['id'])
        if registered!=cell or resolved!=family or verified!=admission or path!=admission_path:
            raise ValueError('final model admission differs')
    repaired=family in NEURAL and admission['evaluation_representations'][family]['representation']=='research-relu-max-pairs-v1'
    directory=root/('reexport_max_pairs' if repaired else 'runs')/cell['id']
    markers=list(directory.glob('attempt-*/final/COMPLETE.json' if repaired else 'attempt-*/evaluation/COMPLETE.json'))
    if len(markers)!=1:raise ValueError('one completed full-FIT candidate required: '+cell['id'])
    marker=markers[0];path=marker.parent/'REPORT.json';row=json.loads(path.read_text())
    if json.loads(marker.read_text())['report_sha256']!=sha(path):raise ValueError('full-FIT report drift')
    run_dir=root/'runs'/cell['id']/marker.parent.parent.name
    run_path=run_dir/'RUN.json';result_path=run_dir/'RESULT.json'
    run=json.loads(run_path.read_text());result=json.loads(result_path.read_text());plan_pin=sha(root/'PLAN.json')
    for record in (run,result,row):
        scope=('descriptive_only' if record is row else 'full_fit_refit') if final else 'full_fit_only'
        development=final and record is row
        if (record['cell']!=cell or record['plan_sha256']!=plan_pin or record['resolved_family']!=family
            or record['selection_scope']!=scope or record['development_results_read']!=development):
            raise ValueError('full-FIT candidate scope differs')
    if run['admission_sha256']!=sha(admission_path) or result['admission_sha256']!=sha(admission_path) or row['admission_sha256']!=sha(admission_path):
        raise ValueError('full-FIT candidate admission differs')
    if result['exit_code']!=0:raise ValueError('failed full-FIT candidate')
    pins={str(p):sha(p) for p in (marker,path,run_path,result_path)}
    def pin(path,expected):
        if sha(path)!=expected:raise ValueError('full-FIT evidence drift: '+str(path))
        pins[str(path)]=expected
    classification=None
    if final:
        for name,expected in admission['full_fit_selection_sha256'].items():
            pin(Path(name),expected)
            if family in NEURAL:
                if run['file_sha256'].get(name)!=expected:raise ValueError('final fit selection provenance differs')
            elif run['selection_sha256'].get(name)!=expected:raise ValueError('final control selection provenance differs')
        if 'exposed_development' not in row or 'full_fit_validation' in row:
            raise ValueError('final descriptive report scope differs')
    if family in NEURAL:
        schedule=final_schedule if final else dict(admission['schedules'][family],setting=cell['setting'])
        if run['schedule']!=schedule or result['schedule']!=schedule or result['stage']!='trained':raise ValueError('full-FIT candidate schedule differs')
        if run['epochs']!=schedule['phase_epochs'][-1] or row['checkpoint_epoch']!=run['epochs']-1 or run['training_exposure_pair_draws']!=schedule['total_pair_draws']:
            raise ValueError('full-FIT final exposure differs')
        for name,expected in run['file_sha256'].items():pin(Path(name),expected)
        bake=run_dir/'fit/checkpoints'/f"ckpt_epoch{run['epochs']-1:03}.bin";pin(bake,result['bake_sha256'])
        identity=admission['evaluation_representations'][family]
        if row.get('representation','original')!=identity['representation']:raise ValueError('full-FIT representation differs')
        if repaired:
            receipt_path=Path(row['reexport_receipt']);pin(receipt_path,row['reexport_receipt_sha256'])
            receipt=json.loads(receipt_path.read_text())
            if (receipt['original_sha256']!=sha(bake) or receipt['training_build_sha256']!=run['build_sha256']
                or receipt['evaluation_build_sha256']!=identity['evaluation_build_sha256']
                or row['evaluation_build_sha256']!=identity['evaluation_build_sha256']):raise ValueError('full-FIT re-export provenance differs')
            for name,expected in receipt['file_sha256'].items():pin(Path(name),expected)
            bake=Path(receipt['bake'])
            if str(bake) not in receipt['file_sha256']:raise ValueError('unpinned full-FIT export')
        pin(bake,row['checkpoint_sha256'])
        gate=row['feature_gate']
        if not gate['finite'] or (family!='silu_diagnostic' and (gate['identity']!=100 or gate['damage_increase_violations']!=0)):
            raise ValueError('full-FIT native feature gate failed')
        diagnostic=run_dir/'TRAIN_CONVERGENCE.json';saved=json.loads(diagnostic.read_text());actual=training_report(run_dir,path)
        if {k:v for k,v in saved.items() if k!='driver_sha256'}!={k:v for k,v in actual.items() if k!='driver_sha256'}:
            raise ValueError('full-FIT convergence evidence cannot reproduce')
        pins[str(diagnostic)]=sha(diagnostic);pins.update(actual['evidence_sha256'])
        classification=dict(conclusion=actual['conclusion'],observed_training_triggers=actual['observed_training_triggers'])
    else:
        if result['stage']!='complete':raise ValueError('incomplete full-FIT control')
        q=row['qualification']
        if family=='extra_trees':
            gate=q['native_tree_gate']
            for suffix,key in [('opaque-vector.zcth','model_sha256'),('parity.npz','parity_sha256'),('parity.log','log_sha256')]:
                pin(run_dir/('model.'+suffix),gate[key])
            if gate['raw_ulp']!=0:raise ValueError('full-FIT tree parity failed')
        elif family=='shaped_additive':
            pin(run_dir/'native/model.bin',q['model_sha256'])
            if q['identity']!=100 or q['damage_increase_violations']!=0:raise ValueError('full-FIT additive gate failed')
        else:raise ValueError('unknown full-FIT control')
    return row,classification,pins


def build(root,role):
    plan,cells,_=load_registration(root);admission,path=resolve_admission(root)
    if role not in ROLES:raise ValueError('registered full-FIT role required')
    for name,pin in plan['input_pins'].items():
        if sha(name)!=pin:raise ValueError('full-FIT frozen input drift')
    family=admission['finalist_roles'][role];reports={};diagnostics={};pins={str(path):sha(path)}
    for cell in admission['selection_cells']:
        if cell['family']!=role:continue
        row,diagnostic,evidence=candidate(root,cell,family,admission,path)
        reports[cell['id']]=row;pins.update(evidence)
        if diagnostic is not None:diagnostics[cell['id']]=diagnostic
    result=select(cells,reports,role,family)
    result.update(admission_sha256=sha(path),plan_sha256=sha(root/'PLAN.json'),evidence_sha256=pins,
        training_only_classifications=diagnostics,driver_sha256=sha(__file__),
        status='setting_selected_with_explicit_convergence_limits')
    return json.loads(encoded(result))


def register(root,role):
    record=build(root,role);directory=root/'full_fit/selection';directory.mkdir(exist_ok=True)
    with (directory/(role+'.json')).open('xb') as stream:stream.write(encoded(record))
    return record


def resolve(root,role):
    path=root/'full_fit/selection'/(role+'.json');saved=json.loads(path.read_text());actual=build(root,role)
    if {k:v for k,v in saved.items() if k!='driver_sha256'}!={k:v for k,v in actual.items() if k!='driver_sha256'}:
        raise ValueError('full-FIT setting selection cannot reproduce')
    return saved,path


def readiness(root,admission):
    """Inspect completion metadata only; candidate() later verifies all evidence."""
    ready=[];missing=[]
    for cell in admission['selection_cells']:
        family=admission['finalist_roles'][cell['family']]
        repaired=family in NEURAL and admission['evaluation_representations'][family]['representation']=='research-relu-max-pairs-v1'
        directory=root/('reexport_max_pairs' if repaired else 'runs')/cell['id']
        markers=list(directory.glob('attempt-*/final/COMPLETE.json' if repaired else 'attempt-*/evaluation/COMPLETE.json'))
        if len(markers)>1:raise ValueError('ambiguous full-FIT completed candidates')
        if markers:
            attempt=markers[0].parent.parent.name
            if (markers[0].parent/'FAILED_GATE.json').exists():raise ValueError('full-FIT evaluation gate failed')
            if family not in NEURAL or (root/'runs'/cell['id']/attempt/'TRAIN_CONVERGENCE.json').exists():
                ready.append(cell['id']);continue
        else:
            for result in (root/'runs'/cell['id']).glob('attempt-*/RESULT.json'):
                row=json.loads(result.read_text())
                if row.get('exit_code')!=0 or row.get('stage')=='failed':raise ValueError('full-FIT search fit failed: '+cell['id'])
            if list(directory.glob('attempt-*/final/FAILED_GATE.json' if repaired else 'attempt-*/evaluation/FAILED_GATE.json')):
                raise ValueError('full-FIT evaluation gate failed: '+cell['id'])
        missing.append(cell['id'])
    return ready,missing


def watch(root):
    directory=root/'dispatch/full-fit-selection';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'OWNER.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output=directory/'STATUS.json'
        if output.exists():raise ValueError('existing full-FIT selection queue requires reconciliation')
        status=dict(pid=os.getpid(),driver_sha256=sha(__file__),development_results_read=False)
        def save(stage,**extra):
            status.update(stage=stage,checked_utc=datetime.now(timezone.utc).isoformat(),**extra);output.write_bytes(encoded(status))
        try:
            while not (root/'full_fit/PLAN.json').exists():
                save('waiting_for_full_fit_admission');time.sleep(30)
            admission,path=resolve_admission(root);pin=sha(path)
            while True:
                if sha(path)!=pin:raise ValueError('full-FIT admission changed during selection wait')
                ready,missing=readiness(root,admission)
                save('waiting_for_search_evaluation',ready_cells=ready,missing_cells=missing,admission_sha256=pin)
                if not missing:break
                time.sleep(30)
            # Validate every role before writing any new setting choice.
            for role in ROLES:build(root,role)
            for role in ROLES:
                path=root/'full_fit/selection'/(role+'.json')
                if not path.exists():register(root,role)
                resolve(root,role)
            pins={}
            for role in ROLES:
                _,path=resolve(root,role);pins[str(path)]=sha(path)
            save('all_role_settings_verified',selection_sha256=pins)
        except Exception as error:
            save('failed',error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--role',choices=ROLES);p.add_argument('--register',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.watch:
        if a.role or a.register:p.error('watch owns all roles and registration')
        watch(a.study)
    else:
        if not a.role:p.error('role required')
        print(json.dumps((register if a.register else build)(a.study,a.role)),flush=True)
