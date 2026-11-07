"""Descriptive learned input responses across a complete paired inner setting."""
from __future__ import annotations
import argparse,json,math,subprocess
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_execution import load_registration
from shaping_rescue_selection import verified_candidate
from shaping_pretraining import structural_terms


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


def report(args):
    _,cells,_=load_registration(args.study)
    selected=[c for c in cells if c['stage']=='inner' and c['family']==args.family and c['setting']==args.setting and
              c['partition'] in (f'inner-{args.outer}-0',f'inner-{args.outer}-1')]
    if len(selected)!=6 or {(c['partition'],c['seed']) for c in selected}!={(f'inner-{args.outer}-{fold}',seed) for fold in range(2) for seed in range(3)}:
        raise ValueError('complete six-fit paired setting required')
    args.output.mkdir(parents=True,exist_ok=False);configs=[];pins={};fits=[]
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    if sha(args.binary_dir/'bake_dial_refit')!=build['binary_sha256']['bake_dial_refit']:raise ValueError('native metadata reader drift')
    for cell in sorted(selected,key=lambda c:(c['partition'],c['seed'])):
        _,_,evidence=verified_candidate(args.study,cell);pins.update(evidence)
        run=next(Path(p).parent for p in evidence if Path(p).name=='RUN.json')
        identity=json.loads((run/'RUN.json').read_text())
        if identity['build_sha256']!=sha(args.binary_dir/'BUILD.json'):raise ValueError('native training build differs')
        model=run/'checkpoints'/f"ckpt_epoch{identity['epochs']-1:03}.bin"
        config=json.loads(subprocess.check_output([str(args.binary_dir/'bake_dial_refit'),'read-meta','--in',str(model),'--key','zentrain.input_shaping'],text=True))
        initial=json.loads((run/'SHAPING.json').read_text())
        if structural_terms(config)!=structural_terms(initial):raise ValueError('native shape structure differs from TRAIN initialization')
        pins[str(model)]=sha(model);pins[str(run/'SHAPING.json')]=sha(run/'SHAPING.json')
        (args.output/(cell['id']+'.json')).write_bytes(encoded(config));configs.append(config);fits.append(cell)
    rows=compare(configs)
    result=dict(scope='Descriptive one-setting inner fit comparison; no quality selection, outer scores or transferable-utility claim.',
        method='Real-valued piecewise-linear response at native quantized parameters; maximum normalized spread on union-of-knot vertices. Separate responses use the damage/reference rectangle. Not a bit-exact runtime score comparison.',
        domain='Zero through largest TRAIN second damage knot; separate reference gates use [0,1]. No claim beyond this domain.',
        downstream_weights_included=False,zero_terms_are_not_useful_stability=True,fits=fits,input_sha256=pins,
        plan_sha256=sha(args.study/'PLAN.json'),driver_sha256=sha(__file__),terms=rows,
        active_in_all=sum(r['active_fits']==6 for r in rows),zero_in_all=sum(r['active_fits']==0 for r in rows),total_terms=len(rows))
    (args.output/'CURVE_STABILITY.json').write_bytes(encoded(result))
    print(json.dumps({k:result[k] for k in ('active_in_all','zero_in_all','total_terms')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','binary-dir','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--family',choices=['product_joint_identity','separate_joint_identity'],required=True)
    p.add_argument('--outer',type=int,choices=range(3),required=True);p.add_argument('--setting',type=int,choices=range(4),required=True)
    report(p.parse_args())
