"""Registered inner diagnostics; never use outer scores or choose checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import numpy as np
from shaping_completion import encoded,sha
from shaping_evaluation import diagnostic_epochs


class UndefinedInnerComposite(ValueError):
    """A measured undefined rank panel cannot establish convergence."""


def loss_window(losses):
    if len(losses)<40 or not np.isfinite(losses).all():
        raise ValueError('requires at least forty finite scheduled losses')
    before=float(np.mean(losses[-40:-20]));after=float(np.mean(losses[-20:]))
    # Native epoch logs round loss to four decimals. Preserve the precision limit;
    # a threshold-straddling interval triggers investigation, not convergence.
    lower=(before-.00005)*.99-(after+.00005)
    upper=(before+.00005)*.99-(after-.00005)
    return dict(preceding_20_mean_loss=before,final_20_mean_loss=after,
                loss_log_rounding_absolute=.00005,loss_threshold_margin_interval=[lower,upper])


def diagnostics(losses,scores,active):
    window=loss_window(losses)
    before=window['preceding_20_mean_loss'];after=window['final_20_mean_loss']
    lower,upper=window['loss_threshold_margin_interval']
    pairs=[(a,b) for a in sorted(scores) for b in sorted(scores) if a<b and a%50==b%50]
    if not pairs:raise ValueError('no same-cycle checkpoint comparison')
    a,b=max(pairs,key=lambda pair:(pair[1],pair[0]))
    if any(scores[k] is None or not np.isfinite(scores[k]) for k in (a,b)):
        raise UndefinedInnerComposite(f'undefined inner composite at same-cycle epochs {a}, {b}')
    improvement=scores[b]-scores[a]
    if not np.isfinite(improvement):raise UndefinedInnerComposite('undefined inner composite')
    collapse=any(v==1 for v in active.values())
    reasons=[]
    if upper>0:reasons.append('loss_improvement_over_one_percent_or_rounding_boundary')
    if improvement>.002:reasons.append('same_cycle_inner_composite_improvement_over_0.002')
    if collapse:reasons.append('one_output_coefficient')
    return dict(investigation_required=bool(reasons),reasons=reasons,
                preceding_20_mean_loss=before,final_20_mean_loss=after,
                loss_log_rounding_absolute=.00005,loss_threshold_margin_interval=[lower,upper],
                same_cycle_epochs=[a,b],same_cycle_human_improvement=improvement,
                active_output_coefficients=active,
                conclusion='investigate' if reasons else 'registered triggers absent; not a proof of global convergence')


def report(run,reexport_root=None,analytic_diagnostic=False):
    identity=json.loads((run/'RUN.json').read_text())
    if identity['cell']['stage'] not in ('inner','fresh_control_inner','rescue','exposure_extension'):raise ValueError('only inner diagnostics admitted')
    result=json.loads((run/'RESULT.json').read_text())
    if result['stage']!='trained':raise ValueError('fit incomplete')
    fit_root=run/identity.get('fit_directory','.')
    lines=re.findall(r'epoch\s+(\d+)\s+\|[^\n]*?loss=([^\s|]+)',(fit_root/'train.log').read_text())
    losses={int(k):float(v) for k,v in lines}
    if sorted(losses)!=list(range(identity['epochs'])):raise ValueError('missing or extra scheduled loss epochs')
    scores={};active={};pins={};builds=set()
    if reexport_root is not None:
        from shaping_reexport import admit_family
        admit_family(identity['cell']['family'],analytic_diagnostic)
    curve_root=reexport_root if reexport_root is not None else run/'curve_evaluation'
    for path in sorted(curve_root.glob('epoch-*/REPORT.json')):
        marker=path.parent/'DIAGNOSTIC_COMPLETE.json'
        if json.loads(marker.read_text())['report_sha256']!=sha(path):raise ValueError('diagnostic report drift')
        row=json.loads(path.read_text())
        if row['cell']!=identity['cell'] or row['outer_results_read']:raise ValueError('diagnostic identity differs')
        if reexport_root is not None:
            receipt=Path(row['reexport_receipt'])
            if sha(receipt)!=row['reexport_receipt_sha256']:raise ValueError('re-export receipt drift')
            r=json.loads(receipt.read_text())
            original=fit_root/'checkpoints'/f"ckpt_epoch{row['checkpoint_epoch']:03}.bin"
            if r['original_sha256']!=sha(original) or r['training_build_sha256']!=identity['build_sha256']:
                raise ValueError('re-export training identity drift')
            if r['representation']!='research-relu-max-pairs-v1':raise ValueError('representation differs')
            for name,pin in r['file_sha256'].items():
                if sha(name)!=pin:raise ValueError('re-export artifact drift')
            if row['checkpoint_sha256']!=sha(r['bake']):raise ValueError('re-export bake drift')
            builds.add(row['evaluation_build_sha256'])
            if r['evaluation_build_sha256']!=row['evaluation_build_sha256']:raise ValueError('re-export evaluator differs')
        epoch=row['checkpoint_epoch'];scores[epoch]=row['inner_validation']['selection_score'];active[epoch]=row['active_output_coefficients'];pins[str(path)]=sha(path)
    expected=list(diagnostic_epochs(identity['epochs']))
    if sorted(scores)!=expected:raise ValueError('incomplete registered checkpoint panels')
    if reexport_root is not None and len(builds)!=1:raise ValueError('mixed re-export evaluators')
    out=diagnostics([losses[i] for i in sorted(losses)],scores,active)
    out.update(cell=identity['cell'],report_sha256=pins,training_log_sha256=sha(fit_root/'train.log'),outer_results_read=False)
    if reexport_root is not None:
        out.update(representation='research-relu-max-pairs-v1',evaluation_build_sha256=next(iter(builds)),
                   original_training_run=str(run))
    path=(reexport_root if reexport_root is not None else run)/'CONVERGENCE.json'
    if path.exists():raise ValueError('preserve existing convergence receipt')
    path.write_bytes(encoded(out));print(json.dumps(out),flush=True)


def training_report(run,final_report):
    identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
    cell=identity['cell']
    final_refit=cell['stage']=='full_fit'
    expected_scope='full_fit_refit' if final_refit else 'full_fit_only'
    if cell['stage'] not in ('full_fit_selection','full_fit') or identity['selection_scope']!=expected_scope or identity['development_results_read']:
        raise ValueError('training-only full-FIT selection diagnostics required')
    if result['cell']!=cell or result['stage']!='trained' or result['exit_code']!=0:
        raise ValueError('full-FIT training incomplete')
    final=json.loads(final_report.read_text())
    if (final['cell']!=cell or final['selection_scope']!=('descriptive_only' if final_refit else 'full_fit_only') or final['development_results_read']!=final_refit
        or final['checkpoint_epoch']!=identity['epochs']-1 or final['plan_sha256']!=identity['plan_sha256']):
        raise ValueError('full-FIT final report identity differs')
    marker=final_report.parent/'COMPLETE.json'
    if json.loads(marker.read_text())['report_sha256']!=sha(final_report):raise ValueError('full-FIT report drift')
    active=final['active_output_coefficients']
    if not isinstance(active,int) or isinstance(active,bool) or active<0:raise ValueError('invalid output activity')
    expected=identity['schedule']['phase_epochs'];phases=result['phases']
    if [p['epochs'] for p in phases]!=expected:raise ValueError('full-FIT phase schedule differs')
    names=['pretraining','fit'] if len(phases)==2 else ['fit'];windows={};pins={}
    for name,phase in zip(names,phases):
        directory=run/name;log=directory/'train.log'
        if phase['stage']!='trained' or phase['exit_code']!=0 or json.loads((directory/'RESULT.json').read_text())!=phase:
            raise ValueError('full-FIT phase incomplete')
        entries=re.findall(r'epoch\s+(\d+)\s+\|[^\n]*?loss=([^\s|]+)',log.read_text())
        losses={int(k):float(v) for k,v in entries}
        if len(entries)!=phase['epochs'] or sorted(losses)!=list(range(phase['epochs'])):raise ValueError('full-FIT loss epoch coverage differs')
        windows[name]=loss_window([losses[k] for k in sorted(losses)])
        checkpoint=directory/'checkpoints'/f"ckpt_epoch{phase['epochs']-1:03}.bin"
        if sha(checkpoint)!=phase['bake_sha256']:raise ValueError('full-FIT phase model drift')
        pins.update({str(p):sha(p) for p in (log,directory/'RESULT.json',checkpoint)})
    reasons=[name+':loss_improvement_over_one_percent_or_rounding_boundary' for name,w in windows.items() if w['loss_threshold_margin_interval'][1]>0]
    if active==1:reasons.append('one_output_coefficient_at_final_epoch')
    record=dict(cell=cell,plan_sha256=identity['plan_sha256'],phase_loss_windows=windows,
        active_output_coefficients_at_final=active,observed_training_triggers=reasons,
        conclusion='inconclusive_training_only',validation_scores_used=False,
        same_cycle_human_trend='not measured; validation checkpoint diagnosis prohibited',
        selection_scope=expected_scope,development_results_used=False,
        evidence_sha256={**pins,**{str(p):sha(p) for p in (run/'RUN.json',run/'RESULT.json',final_report,marker)}},
        driver_sha256=sha(__file__))
    return record


def save_training_report(run,final_report):
    record=training_report(run,final_report)
    with (run/'TRAIN_CONVERGENCE.json').open('xb') as stream:stream.write(encoded(record))
    print(json.dumps(record),flush=True)


def structure_trace(run,binary_dir,output):
    """Inspect every scheduled bake without scoring or selecting checkpoints."""
    import subprocess
    identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
    if result['stage']!='trained' or result['exit_code']!=0 or result['cell']!=identity['cell']:
        raise ValueError('completed matching fit required')
    build=binary_dir/'BUILD.json';reader=binary_dir/'bake_dial_refit'
    if sha(build)!=identity['build_sha256'] or sha(reader)!=json.loads(build.read_text())['binary_sha256']['bake_dial_refit']:
        raise ValueError('native checkpoint reader identity differs')
    fit=run/identity.get('fit_directory','.')
    checkpoints=[fit/'checkpoints'/f'ckpt_epoch{epoch:03}.bin' for epoch in range(identity['epochs'])]
    if not all(p.is_file() for p in checkpoints):raise ValueError('complete scheduled checkpoint sequence required')
    if output.exists():raise ValueError('preserve previous structure trace')
    pins={str(p):sha(p) for p in [run/'RUN.json',run/'RESULT.json',build,reader,*checkpoints]}
    rows=[]
    for epoch,path in enumerate(checkpoints):
        structure=json.loads(subprocess.check_output([str(reader),'inspect','--in',str(path)],text=True))
        layers=structure['layers']
        if not layers:raise ValueError('native checkpoint contains no layers')
        active=[layer['stored_weights']-layer['zero_weights'] for layer in layers]
        if any(v<0 for v in active):raise ValueError('invalid native weight counts')
        rows.append(dict(epoch=epoch,active_stored_weights_by_layer=active,structure=structure))
    shape_path=run/'SHAPING.json';shapes=None
    if shape_path.exists():
        pins[str(shape_path)]=sha(shape_path);config=json.loads(shape_path.read_text())
        shapes=dict(trainable=config['trainable'],terms=len(config['terms']),
                    terms_with_positive_slope=sum(any(v>0 for v in t['slopes']) for t in config['terms']))
    for name,pin in pins.items():
        if sha(name)!=pin:raise ValueError('checkpoint evidence changed during inspection')
    record=dict(cell=identity['cell'],epochs=identity['epochs'],checkpoints=rows,input_shapes=shapes,
        evidence_sha256=pins,driver_sha256=sha(__file__),outer_results_read=False,
        scores_computed=False,checkpoint_selection=False,
        interpretation='Stored coefficients only; no claim about per-row activation, gradients, or independent parameter count.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(encoded(record))
    print(json.dumps(dict(epochs=len(rows),first_zero_output_epoch=next((r['epoch'] for r in rows if r['active_stored_weights_by_layer'][-1]==0),None),input_shapes=shapes)),flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reexport-root',type=Path)
    parser.add_argument('--analytic-diagnostic',action='store_true')
    parser.add_argument('--training-only',action='store_true');parser.add_argument('--final-report',type=Path)
    parser.add_argument('--structure-trace',type=Path);parser.add_argument('--binary-dir',type=Path)
    args=parser.parse_args()
    if args.structure_trace:
        if args.training_only or args.final_report or args.reexport_root or not args.binary_dir:parser.error('structure trace requires only run and binary-dir')
        structure_trace(args.run,args.binary_dir,args.structure_trace)
    elif args.training_only:
        if not args.final_report or args.reexport_root:parser.error('training-only requires final-report and no reexport-root')
        save_training_report(args.run,args.final_report)
    else:report(args.run,args.reexport_root,args.analytic_diagnostic)
