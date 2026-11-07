"""Descriptive learned input responses across a complete paired inner setting."""
from __future__ import annotations
import argparse,json,math,subprocess,struct
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_execution import load_registration
from shaping_rescue_selection import verified_candidate
from shaping_pretraining import structural_terms
from shaping_evaluation import evaluation_epoch


def response(term,x,f):
    a,b=term['knots'];s=term['slopes']
    g=s[0]*min(x,a)+s[1]*min(max(x-a,0),b-a)+s[2]*max(x-b,0)
    if not term['separate']:return g
    h=term['gate'];gate=h[0]+2*f*(h[1]-h[0]) if f<=.5 else h[1]+2*(f-.5)*(h[2]-h[1])
    return g*gate


def compare(configs):
    identities=[[(t['column'],t['damage'],t['reference'],t['separate']) for t in c['terms']] for c in configs]
    if not configs or any(c['schema']!='zensim-input-shaping-v1' for c in configs) or any(x!=identities[0] for x in identities):
        raise ValueError('same native shaping terms required')
    rows=[]
    for terms in zip(*(c['terms'] for c in configs)):
        for t in terms:
            if any(not math.isfinite(v) or v<0 for v in t['knots']+t['slopes']+t['gate']) or not 0<t['knots'][0]<t['knots'][1]:
                raise ValueError('invalid native shape or gate')
        # serde's shortest f32 decimals parse as f64 in Python; recover exact wire values.
        terms=[dict(t,**{k:[struct.unpack('<f',struct.pack('<f',v))[0] for v in t[k]] for k in ('knots','slopes','gate')}) for t in terms]
        # Domain and normalization belong to this panel: adding fits can extend
        # the knot union, so spread/strength are not monotone across panels.
        grid=sorted({0.,*(v for t in terms for v in t['knots'])})
        references=[0.,.5,1.] if terms[0]['separate'] else [1.]
        curves=[[response(t,x,f) for x in grid for f in references] for t in terms]
        strengths=[max(c) for c in curves];normalized=[[v/s for v in c] for c,s in zip(curves,strengths) if s>0]
        spread=max(max(v)-min(v) for v in zip(*normalized)) if len(normalized)>=2 else None
        rows.append(dict(column=terms[0]['column'],damage=terms[0]['damage'],reference=terms[0]['reference'],
            separate=terms[0]['separate'],active_fits=len(normalized),total_fits=len(terms),
            raw_domain_end=grid[-1],normalized_response_max_spread=spread,response_strength=strengths,
            strength_ratio=max(strengths)/min(strengths) if all(s>0 for s in strengths) else None,
            slope_min=min(v for t in terms for v in t['slopes']),slope_max=max(v for t in terms for v in t['slopes']),
            gate_min=min(v for t in terms for v in t['gate']) if terms[0]['separate'] else None,
            gate_max=max(v for t in terms for v in t['gate']) if terms[0]['separate'] else None))
    return rows


def selected_cells(cells,family,outer,settings,inner_fold=None,partitions=None):
    families=['product_joint_identity','separate_joint_identity'] if family=='direct_joint_both' else [family]
    if (not settings or len(set(settings))!=len(settings)
        or any(v not in range(4) for v in settings) or inner_fold not in (None,0,1)):
        raise ValueError('registered distinct settings and inner folds required')
    if partitions is None:
        if outer not in range(3):raise ValueError('registered outer fold required')
        folds=(0,1) if inner_fold is None else (inner_fold,)
        partitions=[f'inner-{outer}-{fold}' for fold in folds]
    else:
        allowed={f'inner-{o}-{i}' for o in range(3) for i in range(2)}
        if (outer is not None or inner_fold is not None or not partitions or
                len(partitions)!=len(set(partitions)) or not set(partitions)<=allowed):
            raise ValueError('distinct registered partitions without fold selectors required')
    expected={(f,partition,seed,setting) for f in families for partition in partitions for seed in range(3) for setting in settings}
    selected=[c for c in cells if c['stage']=='inner' and (c['family'],c['partition'],c['seed'],c['setting']) in expected]
    if len(selected)!=len(expected) or {(c['family'],c['partition'],c['seed'],c['setting']) for c in selected}!=expected:
        raise ValueError('complete paired settings required')
    return sorted(selected,key=lambda c:(c['family'],c['partition'],c['setting'],c['seed']))


def parameter_activity(config):
    """Count stored f32 zero paths, not predictive utility or input rank."""
    counts=dict(terms=len(config['terms']),zero_damage_terms=0,zero_reference_gates=0,
                zero_parameter_paths=0,interaction_terms=0)
    for term in config['terms']:
        zero=lambda values:all(struct.unpack('<f',struct.pack('<f',v))[0]==0 for v in values)
        damage_zero=zero(term['slopes']);gate_zero=term['separate'] and zero(term['gate'])
        counts['zero_damage_terms']+=damage_zero;counts['zero_reference_gates']+=gate_zero
        counts['zero_parameter_paths']+=damage_zero or gate_zero
        counts['interaction_terms']+=term['reference'] is not None
    counts['nonzero_parameter_paths']=counts['terms']-counts['zero_parameter_paths']
    return counts


def checkpoint_config(binary_dir,run,cell,epochs,requested,final_report,pins):
    epoch=evaluation_epoch(epochs,requested)
    model=run/'checkpoints'/f'ckpt_epoch{epoch:03}.bin'
    row=final_report
    if requested is not None:
        directory=run/'curve_evaluation'/f'epoch-{epoch:03}'
        marker=directory/'DIAGNOSTIC_COMPLETE.json';path=directory/'REPORT.json'
        row=json.loads(path.read_text())
        if json.loads(marker.read_text())['report_sha256']!=sha(path):raise ValueError('shape diagnostic report drift')
        for p in (marker,path):pins[str(p)]=sha(p)
    if (row['cell']!=cell or row['plan_sha256']!=final_report['plan_sha256']
            or row['checkpoint_epoch']!=epoch or row['outer_results_read']
            or row['checkpoint_sha256']!=sha(model)):
        raise ValueError('shape checkpoint/report identity differs')
    pins[str(model)]=sha(model)
    config=json.loads(subprocess.check_output([str(binary_dir/'bake_dial_refit'),'read-meta',
        '--in',str(model),'--key','zentrain.input_shaping'],text=True))
    return config,epoch,row['active_output_coefficients']


def report(args):
    _,cells,_=load_registration(args.study)
    settings=getattr(args,'settings',None) or [args.setting]
    selected=selected_cells(cells,args.family,args.outer,settings,getattr(args,'inner_fold',None),getattr(args,'partitions',None))
    args.output.mkdir(parents=True,exist_ok=False);configs=[];pins={};fits=[];activity=[]
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    if sha(args.binary_dir/'bake_dial_refit')!=build['binary_sha256']['bake_dial_refit']:raise ValueError('native metadata reader drift')
    for cell in selected:
        final_report,_,evidence=verified_candidate(args.study,cell);pins.update(evidence)
        run=next(Path(p).parent for p in evidence if Path(p).name=='RUN.json')
        identity=json.loads((run/'RUN.json').read_text())
        if identity['build_sha256']!=sha(args.binary_dir/'BUILD.json'):raise ValueError('native training build differs')
        config,epoch,output_active=checkpoint_config(args.binary_dir,run,cell,identity['epochs'],
            getattr(args,'checkpoint',None),final_report,pins)
        initial=json.loads((run/'SHAPING.json').read_text())
        if structural_terms(config)!=structural_terms(initial):raise ValueError('native shape structure differs from TRAIN initialization')
        pins[str(run/'SHAPING.json')]=sha(run/'SHAPING.json')
        if args.family=='direct_joint_both':config['terms']=[t for t in config['terms'] if t['reference'] is None]
        activity.append(dict(cell=cell['id'],checkpoint_epoch=epoch,
                             active_output_coefficients=output_active,**parameter_activity(config)))
        (args.output/(cell['id']+'.json')).write_bytes(encoded(config));configs.append(config);fits.append(cell)
    rows=compare(configs)
    result=dict(scope='Descriptive paired inner fit comparison; no quality selection, outer scores or transferable-utility claim.',
        method='Real-valued piecewise-linear response at native quantized parameters; maximum normalized spread on union-of-knot vertices. Separate responses use the damage/reference rectangle. Not a bit-exact runtime score comparison.',
        domain='Zero through largest TRAIN second damage knot; separate reference gates use [0,1]. No claim beyond this domain.',
        downstream_weights_included=False,zero_terms_are_not_useful_stability=True,fits=fits,input_sha256=pins,
        requested_checkpoint=getattr(args,'checkpoint',None),coefficient_activity=activity,
        checkpoint_scope='registered final' if getattr(args,'checkpoint',None) is None else 'registered diagnostic; excluded from checkpoint selection',
        coefficient_activity_scope='Stored f32 zero paths only; not measured input rank, optimizer state or predictive utility.',
        plan_sha256=sha(args.study/'PLAN.json'),driver_sha256=sha(__file__),terms=rows,
        active_in_all=sum(r['active_fits']==len(selected) for r in rows),zero_in_all=sum(r['active_fits']==0 for r in rows),total_terms=len(rows))
    (args.output/'CURVE_STABILITY.json').write_bytes(encoded(result))
    print(json.dumps({k:result[k] for k in ('active_in_all','zero_in_all','total_terms')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','binary-dir','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--family',choices=['product_joint_identity','separate_joint_identity','direct_joint_both'],required=True)
    scope=p.add_mutually_exclusive_group(required=True)
    scope.add_argument('--outer',type=int,choices=range(3))
    scope.add_argument('--partitions',nargs='+',choices=[f'inner-{o}-{i}' for o in range(3) for i in range(2)])
    p.add_argument('--inner-fold',type=int,choices=(0,1))
    p.add_argument('--checkpoint',type=int,help='registered diagnostic epoch; requires a completed and qualified parent fit')
    choices=p.add_mutually_exclusive_group(required=True)
    choices.add_argument('--setting',type=int,choices=range(4))
    choices.add_argument('--settings',type=int,choices=range(4),nargs='+')
    report(p.parse_args())
