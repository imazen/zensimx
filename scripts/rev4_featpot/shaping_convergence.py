"""Registered inner diagnostics; never use outer scores or choose checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import numpy as np
from shaping_completion import encoded,sha


def diagnostics(losses,scores,active):
    if len(losses)<40 or not np.isfinite(losses).all():
        raise ValueError('requires at least forty finite scheduled losses')
    before=float(np.mean(losses[-40:-20]));after=float(np.mean(losses[-20:]))
    # Native epoch logs round loss to four decimals. Preserve the precision limit;
    # a threshold-straddling interval triggers investigation, not convergence.
    lower=(before-.00005)*.99-(after+.00005)
    upper=(before+.00005)*.99-(after-.00005)
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


def report(run):
    identity=json.loads((run/'RUN.json').read_text())
    if identity['cell']['stage']!='inner':raise ValueError('only inner diagnostics admitted')
    result=json.loads((run/'RESULT.json').read_text())
    if result['stage']!='trained':raise ValueError('fit incomplete')
    lines=re.findall(r'epoch\s+(\d+)\s+\|[^\n]*?loss=([^ |]+)',(run/'train.log').read_text())
    losses={int(k):float(v) for k,v in lines}
    if sorted(losses)!=list(range(identity['epochs'])):raise ValueError('missing or extra scheduled loss epochs')
    scores={};active={};pins={}
    for path in sorted((run/'curve_evaluation').glob('epoch-*/REPORT.json')):
        marker=path.parent/'DIAGNOSTIC_COMPLETE.json'
        if json.loads(marker.read_text())['report_sha256']!=sha(path):raise ValueError('diagnostic report drift')
        row=json.loads(path.read_text())
        if row['cell']!=identity['cell'] or row['outer_results_read']:raise ValueError('diagnostic identity differs')
        epoch=row['checkpoint_epoch'];scores[epoch]=row['inner_validation']['selection_score'];active[epoch]=row['active_output_coefficients'];pins[str(path)]=sha(path)
    expected=[e for e in (49,99,119,149,199,219,239) if e<identity['epochs']]
    if sorted(scores)!=expected:raise ValueError('incomplete registered checkpoint panels')
    out=diagnostics([losses[i] for i in sorted(losses)],scores,active)
    out.update(cell=identity['cell'],report_sha256=pins,training_log_sha256=sha(run/'train.log'),outer_results_read=False)
    path=run/'CONVERGENCE.json'
    if path.exists():raise ValueError('preserve existing convergence receipt')
    path.write_bytes(encoded(out));print(json.dumps(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    report(parser.parse_args().run)
