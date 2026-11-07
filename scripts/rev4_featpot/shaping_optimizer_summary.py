"""Summarize complete native optimizer traces without using outer responses."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_rescue_selection import verified_candidate
from shaping_execution import load_registration


def summarize(text,epochs):
    rows=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith('TRAINING_DIAGNOSTICS ')]
    if [r['epoch'] for r in rows]!=list(range(epochs)):raise ValueError('complete ordered optimizer trace required')
    compact=[]
    for row in rows:
        width=row['last_units'];counts=row['last_unit_active_counts'];weights=row['output_amplitudes']
        if len(counts)!=width or len(weights)!=width:raise ValueError('native hidden/output widths differ')
        if row['sampled_responses']<=0 or row['first_activation_values']<=0:raise ValueError('training activity sample required')
        if any(not isinstance(n,int) or not 0<=n<=row['sampled_responses'] for n in counts):raise ValueError('invalid sampled activity count')
        if not 0<=row['first_zero_activations']<=row['first_activation_values']:raise ValueError('invalid first activation count')
        gradients=[row['mean_squared_first_activation_gradient'],row['mean_squared_regularized_output_gradient']]
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in weights+gradients):
            raise ValueError('nonfinite native optimizer evidence; preserve trace and investigate')
        if any(v<0 for v in gradients):raise ValueError('negative squared gradient')
        active=sum(v!=0 for v in weights)
        if width-active!=row['output_zero_weights']:raise ValueError('native output activity count differs')
        compact.append(dict(epoch=row['epoch'],nonzero_output_coefficients=active,
            sampled_active_last_units=sum(v>0 for v in counts),sampled_responses=row['sampled_responses'],
            first_zero_activations=row['first_zero_activations'],first_activation_values=row['first_activation_values'],
            mean_squared_first_activation_gradient=gradients[0],mean_squared_regularized_output_gradient=gradients[1],
            output_projection_hits=row['output_projection_hits']))
    collapsed=[r['epoch'] for r in compact if r['nonzero_output_coefficients']<=1]
    return dict(epochs=epochs,first=compact[0],final=compact[-1],
        epochs_with_at_most_one_output=collapsed,total_output_projection_hits=sum(r['output_projection_hits'] for r in compact),
        sampled_activity_is_not_global_deadness=True,first_activation_gradient_is_not_first_weight_gradient=True)


def report(args):
    _,cells,_=load_registration(args.study);identity=json.loads((args.run/'RUN.json').read_text());cell=identity['cell']
    if cell not in cells or cell['stage'] not in ('inner','fresh_control_inner','rescue'):
        raise ValueError('registered completed inner fit required')
    if args.run.resolve().parent.parent!=(args.study/'runs').resolve() or args.run.parent.name!=cell['id']:
        raise ValueError('optimizer run location differs')
    _,convergence,pins=verified_candidate(args.study,cell,args.representation)
    log=args.run/identity.get('fit_directory','.')/'train.log'
    if sha(log)!=convergence['training_log_sha256']:raise ValueError('training trace provenance differs')
    result=dict(cell=cell,scope='final training phase only',summary=summarize(log.read_text(),identity['epochs']),
                training_log_sha256=sha(log),evidence_sha256=pins,plan_sha256=sha(args.study/'PLAN.json'),
                driver_sha256=sha(__file__),selection_candidate=False,outer_results_read=False)
    with args.output.open('xb') as stream:stream.write(encoded(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','run','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--representation',choices=['original','research-relu-max-pairs-v1'],default='original')
    print(json.dumps(report(p.parse_args())),flush=True)
