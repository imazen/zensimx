"""Archive immutable completed inner cells; never reclaim or copy active attempts."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import REPO,encoded,sha
from shaping_execution import load_registration


def eligible(directory,cells,plan_pin):
    attempts=list(directory.glob('attempt-*'))
    # Multiple attempts need explicit reconciliation of all retained failures.
    if len(attempts)!=1:return False
    run=attempts[0]
    if not all((run/name).is_file() for name in ('RUN.json','RESULT.json','CONVERGENCE.json','evaluation/COMPLETE.json')):return False
    identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
    cell=identity['cell']
    if cell['stage'] not in ('inner','fresh_control_inner'):return False
    if cell not in cells or cell['id']!=directory.name:raise ValueError('archive cell identity differs')
    if cell['stage'] not in ('inner','fresh_control_inner') or result['stage']!='trained':return False
    if identity['plan_sha256']!=plan_pin or result['cell']!=cell or result['exit_code']!=0:raise ValueError('archive fit differs')
    c=json.loads((run/'CONVERGENCE.json').read_text())
    if c['cell']!=cell or c['outer_results_read']:raise ValueError('archive convergence identity differs')
    for name,pin in c['report_sha256'].items():
        if sha(name)!=pin:raise ValueError('archive curve drift')
    final=run/'evaluation/REPORT.json'
    if json.loads((run/'evaluation/COMPLETE.json').read_text())['report_sha256']!=sha(final):raise ValueError('archive final report drift')
    if (run/'EVALUATION_FAILURES.json').exists():return False
    return True


def sweep(args):
    _,cells,_=load_registration(args.study);completed=0
    logs=args.study/'archive_supervision';logs.mkdir(exist_ok=True)
    for directory in sorted((args.study/'runs').iterdir()):
        if not directory.is_dir():continue
        if (directory/'MIRROR_VERIFIED.json').exists():continue
        if not eligible(directory,cells,sha(args.study/'PLAN.json')):continue
        dest=args.destination/('inner-'+directory.name)
        if dest.exists() or (directory/'SHA256SUMS').exists():
            raise ValueError('unfinished archive requires explicit reconciliation: '+str(directory))
        log=logs/(directory.name+'.log')
        command=['python3','-B',str(REPO/'scripts/byconstruction/dataset_mirror.py'),
                 '--root',str(directory),'--nfs','--cold-readback','--destination',str(dest)]
        print(json.dumps(dict(cell=directory.name,stage='archiving',destination=str(dest))),flush=True)
        with log.open('x') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        receipt=json.loads((directory/'MIRROR_VERIFIED.json').read_text())
        if not receipt['all_artifact_sha256_verified'] or receipt['destination']!=str(dest):raise ValueError('archive readback incomplete')
        if sha(directory/'SHA256SUMS')!=sha(dest/'SHA256SUMS'):raise ValueError('archive manifest differs')
        completed+=1
    return completed



def supplemental_sweep(args):
    """Mirror only complete, uniformly pinned supplemental checkpoint panels."""
    from shaping_evaluation import diagnostic_epochs
    _,cells,_=load_registration(args.study);completed=0
    logs=args.study/'archive_supervision';logs.mkdir(exist_ok=True)
    for cpath in sorted((args.study/'reexport_max_pairs').glob('*/attempt-*/CONVERGENCE.json')):
        root=cpath.parent;convergence=json.loads(cpath.read_text());cell=convergence['cell']
        if cell['stage'] not in ('inner','fresh_control_inner','rescue','exposure_extension'):continue
        if cell['stage']=='exposure_extension':
            from shaping_extension import resolve
            registered=resolve(args.study,cell['id'])['cell']
            if cell!=registered:raise ValueError('supplemental extension identity differs')
            if cell not in cells:cells.append(cell)
        if cell not in cells:raise ValueError('supplemental archive cell not registered')
        run=args.study/'runs'/cell['id']/root.name
        identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
        if identity['cell']!=cell or result['cell']!=cell or result['stage']!='trained' or result['exit_code']!=0:
            raise ValueError('supplemental archive training identity differs')
        if identity['plan_sha256']!=sha(args.study/'PLAN.json') or convergence['outer_results_read']:
            raise ValueError('supplemental archive scope differs')
        if convergence['original_training_run']!=str(run):raise ValueError('supplemental original run differs')
        expected={str(root/f'epoch-{e:03}'/'REPORT.json') for e in diagnostic_epochs(result['epochs'])}
        if set(convergence['report_sha256'])!=expected:raise ValueError('incomplete supplemental archive checkpoints')
        for name,pin in convergence['report_sha256'].items():
            if sha(name)!=pin:raise ValueError('supplemental report drift')
        for epoch in (None,*diagnostic_epochs(result['epochs'])):
            panel=root/('final' if epoch is None else f'epoch-{epoch:03}')
            path=panel/'REPORT.json';marker=panel/('COMPLETE.json' if epoch is None else 'DIAGNOSTIC_COMPLETE.json')
            if json.loads(marker.read_text())['report_sha256']!=sha(path):raise ValueError('supplemental marker drift')
            row=json.loads(path.read_text());actual=result['epochs']-1 if epoch is None else epoch
            if row['cell']!=cell or row['checkpoint_epoch']!=actual or row['outer_results_read']:
                raise ValueError('supplemental panel identity differs')
            receipt=Path(row['reexport_receipt'])
            if sha(receipt)!=row['reexport_receipt_sha256']:raise ValueError('supplemental receipt drift')
            conversion=json.loads(receipt.read_text())
            original=run/identity.get('fit_directory','.')/'checkpoints'/f'ckpt_epoch{actual:03}.bin'
            if conversion['original_sha256']!=sha(original) or conversion['training_build_sha256']!=identity['build_sha256']:
                raise ValueError('supplemental bake source drift')
            if conversion['evaluation_build_sha256']!=convergence['evaluation_build_sha256'] or row['evaluation_build_sha256']!=convergence['evaluation_build_sha256']:
                raise ValueError('supplemental runtime drift')
            for name,pin in conversion['file_sha256'].items():
                if sha(name)!=pin:raise ValueError('supplemental artifact drift')
            if row['checkpoint_sha256']!=sha(conversion['bake']):raise ValueError('supplemental scored bake drift')
        for kind,directory in [('panels',root.parent),('bakes',args.study/'reexports'/cell['id'])]:
            if (directory/'MIRROR_VERIFIED.json').exists():continue
            dest=args.destination/(f'supplemental-{kind}-'+cell['id'])
            if dest.exists() or (directory/'SHA256SUMS').exists():raise ValueError('unfinished supplemental archive needs reconciliation')
            log=logs/(f'supplemental-{kind}-'+cell['id']+'.log')
            command=['python3','-B',str(REPO/'scripts/byconstruction/dataset_mirror.py'),'--root',str(directory),'--nfs','--cold-readback','--destination',str(dest)]
            with log.open('x') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
            receipt=json.loads((directory/'MIRROR_VERIFIED.json').read_text())
            if not receipt['all_artifact_sha256_verified'] or sha(directory/'SHA256SUMS')!=sha(dest/'SHA256SUMS'):
                raise ValueError('supplemental archive incomplete')
            completed+=1
    return completed


def later_neural_bundles(study,directory,cells):
    """Resolve completed final-only evidence without opening prediction arrays."""
    from shaping_evaluation import outer_context,admitted_epoch
    from shaping_supervision import LATER_STAGES
    from shaping_convergence import training_report
    attempts=list(directory.glob('attempt-*'))
    if len(attempts)!=1:return []
    run=attempts[0]
    if not all((run/n).is_file() for n in ('RUN.json','RESULT.json')):return []
    identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
    cell=identity['cell']
    if cell['stage'] not in LATER_STAGES or result['stage']!='trained':return []
    plan_pin=sha(study/'PLAN.json')
    if cell['id']!=directory.name or result['cell']!=cell or identity['plan_sha256']!=plan_pin or result['plan_sha256']!=plan_pin:
        raise ValueError('later archive training identity differs')
    context=outer_context(study,identity,cells)
    if context is None:raise ValueError('later archive lacks admission')
    epochs=admitted_epoch(identity,result,cells,context)
    family=context.get('resolved_family',context.get('variant',{}).get('resolved_family',cell['family']))
    representation=context['admission']['evaluation_representations'][family]
    repaired=representation['representation']=='research-relu-max-pairs-v1'
    if representation['representation'] not in ('original','research-relu-max-pairs-v1'):
        raise ValueError('unknown later archive representation')
    panel=study/'reexport_max_pairs'/cell['id']/run.name/'final' if repaired else run/'evaluation'
    if not (panel/'COMPLETE.json').exists() or (panel/'FAILED_GATE.json').exists():return []
    marker=json.loads((panel/'COMPLETE.json').read_text());report=json.loads((panel/'REPORT.json').read_text())
    if marker['cell']!=cell['id'] or marker['report_sha256']!=sha(panel/'REPORT.json'):
        raise ValueError('later archive report drift')
    if (report['cell']!=cell or report['plan_sha256']!=plan_pin or report['admission_sha256']!=context['admission_sha256']
        or report['checkpoint_epoch']!=epochs-1 or report['outer_results_read'] is not True):
        raise ValueError('later archive report scope differs')
    expected_role='exposed_development' if context.get('descriptive') else 'full_fit_validation' if context.get('full_fit') else 'outer_test'
    if {k for k in ('outer_test','full_fit_validation','exposed_development','inner_validation') if k in report}!={expected_role}:
        raise ValueError('later archive response role differs')
    if sha(panel/'predictions.npz')!=report['predictions_sha256']:raise ValueError('later archive prediction drift')
    phases=result['phases'];expected=context['schedule']['phase_epochs']
    if [p['epochs'] for p in phases]!=expected or len(expected) not in (1,2):raise ValueError('later archive phase exposure differs')
    for name,phase in zip(['pretraining','fit'] if len(expected)==2 else ['fit'],phases):
        phase_root=run/name
        if phase['stage']!='trained' or phase['exit_code']!=0 or json.loads((phase_root/'RESULT.json').read_text())!=phase:
            raise ValueError('later archive phase incomplete')
        if sha(phase_root/'checkpoints'/f"ckpt_epoch{phase['epochs']-1:03}.bin")!=phase['bake_sha256']:
            raise ValueError('later archive phase model drift')
    if phases[-1]['bake_sha256']!=result['bake_sha256']:raise ValueError('later archive final model differs')
    if repaired:
        receipt_path=Path(report['reexport_receipt']);receipt=json.loads(receipt_path.read_text())
        if (sha(receipt_path)!=report['reexport_receipt_sha256'] or receipt['original_sha256']!=result['bake_sha256']
            or receipt['training_build_sha256']!=identity['build_sha256'] or receipt['representation']!=representation['representation']
            or receipt['evaluation_build_sha256']!=representation['evaluation_build_sha256']
            or report['evaluation_build_sha256']!=representation['evaluation_build_sha256']):
            raise ValueError('later archive conversion identity differs')
        if receipt['bake'] not in receipt['file_sha256']:raise ValueError('later archive unpinned bake')
        for name,pin in receipt['file_sha256'].items():
            if sha(name)!=pin:raise ValueError('later archive conversion artifact drift')
        if sha(receipt['bake'])!=report['checkpoint_sha256']:raise ValueError('later archive scored bake differs')
    elif report['checkpoint_sha256']!=result['bake_sha256']:raise ValueError('later archive scored model differs')
    if cell['stage'] in ('full_fit_selection','full_fit'):
        diagnostic=run/'TRAIN_CONVERGENCE.json'
        if not diagnostic.exists():return []
        saved=json.loads(diagnostic.read_text());actual=training_report(run,panel/'REPORT.json')
        if {k:v for k,v in saved.items() if k!='driver_sha256'}!={k:v for k,v in actual.items() if k!='driver_sha256'}:
            raise ValueError('later archive training diagnostic drift')
    bundles=[('run',directory)]
    if repaired:bundles.extend([('panels',panel.parent.parent),('bakes',study/'reexports'/cell['id'])])
    return bundles


def later_control_bundles(study,directory,cells):
    """Reproduce control admission and hash native artifacts without scoring."""
    from shaping_controls import selected_cells,full_fit_cells,final_refit_cells
    from shaping_evaluation import VARIANT_STAGES
    attempts=list(directory.glob('attempt-*'))
    if len(attempts)!=1:return []
    run=attempts[0]
    if not all((run/n).is_file() for n in ('RUN.json','RESULT.json','evaluation/COMPLETE.json')):return []
    identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
    cell=identity['cell'];family=cell['family'];stage=cell['stage']
    if family not in ('extra_trees','shaped_additive') or stage not in ('outer','full_fit_selection','full_fit',*VARIANT_STAGES) or result['stage']!='complete':return []
    if stage in VARIANT_STAGES:
        from shaping_variant_execution import resolve
        from shaping_variant_inputs import load_variant
        variant=resolve(study,cell['id']);load_variant(study,cell['id'])
        if variant['resolved_family']!=family:raise ValueError('control archive resolved family differs')
        chosen=[variant['cell']]
        context=dict(admission_sha256=variant['admission_sha256'],selected_setting=variant['schedule']['setting'],variant=variant)
    else:
        resolver=final_refit_cells if stage=='full_fit' else full_fit_cells if stage=='full_fit_selection' else selected_cells
        chosen,context=resolver(study,cell['partition'],family,cells)
    plan_pin=sha(study/'PLAN.json')
    if cell not in chosen or directory.name!=cell['id']:raise ValueError('control archive cell differs')
    for record in (identity,result):
        if record['cell']!=cell or record['plan_sha256']!=plan_pin or any(record.get(k)!=v for k,v in context.items()):
            raise ValueError('control archive admission differs')
    if result['exit_code']!=0 or result['build_sha256']!=identity['build_sha256']:raise ValueError('control archive terminal identity differs')
    panel=run/'evaluation';marker=json.loads((panel/'COMPLETE.json').read_text());report=json.loads((panel/'REPORT.json').read_text())
    if marker['cell']!=cell['id'] or marker['report_sha256']!=sha(panel/'REPORT.json'):raise ValueError('control archive report drift')
    if (report['cell']!=cell or report['plan_sha256']!=plan_pin or report['admission_sha256']!=context['admission_sha256']
        or report['outer_results_read'] is not True):raise ValueError('control archive report identity differs')
    role='exposed_development' if context.get('descriptive') else 'full_fit_validation' if context.get('full_fit') else 'outer_test'
    if {k for k in ('outer_test','full_fit_validation','exposed_development','inner_validation') if k in report}!={role}:
        raise ValueError('control archive response role differs')
    if sha(panel/'predictions.npz')!=report['predictions_sha256']:raise ValueError('control archive prediction drift')
    qualification=report['qualification']
    model=run/('model.opaque-vector.zcth' if family=='extra_trees' else 'native/model.bin')
    pin=qualification['native_tree_gate']['model_sha256'] if family=='extra_trees' else qualification['model_sha256']
    if sha(model)!=pin:raise ValueError('control archive native model drift')
    return [('control',directory)]


def later_sweep(args):
    _,cells,_=load_registration(args.study);completed=0
    logs=args.study/'archive_supervision';logs.mkdir(exist_ok=True)
    for directory in sorted((args.study/'runs').iterdir()):
        if not directory.is_dir():continue
        inspect=later_control_bundles if getattr(args,'later_controls',False) else later_neural_bundles
        for kind,source in inspect(args.study,directory,cells):
            if (source/'MIRROR_VERIFIED.json').exists():continue
            dest=args.destination/(f'later-{kind}-'+directory.name)
            if dest.exists() or (source/'SHA256SUMS').exists():raise ValueError('unfinished later archive needs reconciliation')
            command=['python3','-B',str(REPO/'scripts/byconstruction/dataset_mirror.py'),'--root',str(source),'--nfs','--cold-readback','--destination',str(dest)]
            with (logs/(f'later-{kind}-'+directory.name+'.log')).open('x') as stream:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
            receipt=json.loads((source/'MIRROR_VERIFIED.json').read_text())
            if not receipt['all_artifact_sha256_verified'] or receipt['destination']!=str(dest) or sha(source/'SHA256SUMS')!=sha(dest/'SHA256SUMS'):
                raise ValueError('later archive readback incomplete')
            completed+=1
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','destination'):p.add_argument('--'+key,type=Path,required=True)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--supplemental',action='store_true')
    modes.add_argument('--later-neural',action='store_true')
    modes.add_argument('--later-controls',action='store_true')
    p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        completed=later_sweep(args) if args.later_neural or args.later_controls else supplemental_sweep(args) if args.supplemental else sweep(args)
        status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),
                    newly_archived=completed,stage='watching' if args.watch else 'sweep_finished',driver_sha256=sha(__file__))
        status_name='LATER_CONTROL_ARCHIVE_STATUS.json' if args.later_controls else 'LATER_NEURAL_ARCHIVE_STATUS.json' if args.later_neural else 'SUPPLEMENTAL_ARCHIVE_STATUS.json' if args.supplemental else 'ARCHIVE_SUPERVISOR_STATUS.json'
        (args.study/status_name).write_bytes(encoded(status));print(json.dumps(status),flush=True)
        if not args.watch:return
        time.sleep(30)


if __name__=='__main__':main()
