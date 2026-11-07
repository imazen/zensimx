"""Registered inner diagnostics; never use outer scores or choose checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import numpy as np
from shaping_completion import encoded,sha
from shaping_evaluation import diagnostic_epochs


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
    improvement=scores[b]-scores[a]
    if not np.isfinite(improvement):raise ValueError('undefined inner composite')
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


def report(run,reexport_root=None):
    identity=json.loads((run/'RUN.json').read_text())
    if identity['cell']['stage'] not in ('inner','fresh_control_inner','rescue','exposure_extension'):raise ValueError('only inner diagnostics admitted')
    result=json.loads((run/'RESULT.json').read_text())
    if result['stage']!='trained':raise ValueError('fit incomplete')
    fit_root=run/identity.get('fit_directory','.')
    lines=re.findall(r'epoch\s+(\d+)\s+\|[^\n]*?loss=([^\s|]+)',(fit_root/'train.log').read_text())
    losses={int(k):float(v) for k,v in lines}
    if sorted(losses)!=list(range(identity['epochs'])):raise ValueError('missing or extra scheduled loss epochs')
    scores={};active={};pins={};builds=set()
    if reexport_root is not None and identity['cell']['family']!='relu':
        raise ValueError('max-pair supplemental convergence requires ReLU')
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
    if cell['stage']!='full_fit_selection' or identity['selection_scope']!='full_fit_only' or identity['development_results_read']:
        raise ValueError('training-only full-FIT selection diagnostics required')
    if result['cell']!=cell or result['stage']!='trained' or result['exit_code']!=0:
        raise ValueError('full-FIT training incomplete')
    final=json.loads(final_report.read_text())
    if (final['cell']!=cell or final['selection_scope']!='full_fit_only' or final['development_results_read']
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
        selection_scope='full_fit_only',development_results_used=False,
        evidence_sha256={**pins,**{str(p):sha(p) for p in (run/'RUN.json',run/'RESULT.json',final_report,marker)}},
        driver_sha256=sha(__file__))
    return record


def save_training_report(run,final_report):
    record=training_report(run,final_report)
    with (run/'TRAIN_CONVERGENCE.json').open('xb') as stream:stream.write(encoded(record))
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reexport-root',type=Path)
    parser.add_argument('--training-only',action='store_true');parser.add_argument('--final-report',type=Path)
    args=parser.parse_args()
    if args.training_only:
        if not args.final_report or args.reexport_root:parser.error('training-only requires final-report and no reexport-root')
        save_training_report(args.run,args.final_report)
    else:report(args.run,args.reexport_root)
