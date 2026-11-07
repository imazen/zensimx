"""Reproduce full-FIT setting decisions from complete pinned search evidence."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from shaping_completion import NEURAL,ROLES,encoded,sha
from shaping_execution import load_registration
from shaping_full_fit_admission import resolve as resolve_admission
from shaping_full_fit_plan import select
from shaping_convergence import training_report


def candidate(root,cell,family,admission,admission_path):
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
        if (record['cell']!=cell or record['plan_sha256']!=plan_pin or record['resolved_family']!=family
            or record['selection_scope']!='full_fit_only' or record['development_results_read']):
            raise ValueError('full-FIT candidate scope differs')
    if run['admission_sha256']!=sha(admission_path) or result['admission_sha256']!=sha(admission_path) or row['admission_sha256']!=sha(admission_path):
        raise ValueError('full-FIT candidate admission differs')
    if result['exit_code']!=0:raise ValueError('failed full-FIT candidate')
    pins={str(p):sha(p) for p in (marker,path,run_path,result_path)}
    def pin(path,expected):
        if sha(path)!=expected:raise ValueError('full-FIT evidence drift: '+str(path))
        pins[str(path)]=expected
    classification=None
    if family in NEURAL:
        schedule=dict(admission['schedules'][family],setting=cell['setting'])
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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--role',choices=ROLES,required=True);p.add_argument('--register',action='store_true');a=p.parse_args()
    print(json.dumps((register if a.register else build)(a.study,a.role)),flush=True)
