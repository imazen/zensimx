"""Qualify a uniform stable export without changing fitting or numerical gates."""
import argparse,json,math
from pathlib import Path
from shaping_completion import encoded,sha

NAME='STABLE_MAX_PAIR_QUALIFICATION.json'
FAMILIES=('silu_monotone','product_joint_identity','product_pretrained_frozen','product_pretrained_joint',
          'separate_joint_identity','separate_pretrained_frozen','separate_pretrained_joint')
WITNESSES={'1a22976d73e29a7c2217':219,'d20acb0b9441fc96aab8':119,
           '7e28ac0fc83682e66022':119,'46c5fe4370e1e47e9b4a':99}
GROUPS={'human_kadid','human_tid2013','human_konfig','human_cid22_a25','cid22'}


def families(study):
    path=Path(study)/NAME
    if not path.exists():return ()
    value=json.loads(path.read_text())
    from shaping_corrected_view import BUILD_SHA256
    if (value['schema']!='qualified-stable-max-pair-families-v1' or tuple(value['families'])!=FAMILIES
        or value['plan_sha256']!=sha(Path(study)/'PLAN.json') or value['build_sha256']!=BUILD_SHA256
        or value['fitting_changed'] or value['numerical_threshold_changed'] or value['outer_results_read']
        or value['original_evidence_preserved'] is not True):
        raise ValueError('stable max-pair qualification scope differs')
    if len(value['cases'])!=8 or {q['cell']['family'] for q in value['cases']}!=set(FAMILIES):
        raise ValueError('stable max-pair qualification family coverage differs')
    if {q['cell']['id']:q['epoch'] for q in value['cases'] if q['witness']}!=WITNESSES:
        raise ValueError('original numerical witnesses differ')
    for name,pin in value['evidence_sha256'].items():
        if sha(name)!=pin:raise ValueError('stable max-pair qualification evidence drift: '+name)
    for case in value['cases']:
        matches=[Path(p) for p in value['evidence_sha256'] if Path(p).name=='RESULT.json'
                 and Path(p).parent.name=='comparison' and Path(p).parent.parent.name==case['cell']['id']]
        if len(matches)!=1:raise ValueError('one native comparison proof per case required')
        measured=json.loads(matches[0].read_text())
        if (measured['cell']!=case['cell'] or measured['checkpoint_epoch']!=case['epoch']
            or measured['all_unmarked_exact'] is not True or measured['outer_results_read']
            or case['population_rows']!=sum(r['rows'] for r in measured['comparisons'])
            or case['maximum_population_score_delta']!=max(r['max_abs_delta'] for r in measured['comparisons'])):
            raise ValueError('qualified case differs from native comparison')
    return FAMILIES


def register(study,root,source):
    from shaping_execution import load_registration
    from shaping_corrected_view import BUILD_SHA256,require_activation
    require_activation(study);_,cells,_=load_registration(study);registered={c['id']:c for c in cells}
    output=study/NAME
    if output.exists():raise ValueError('preserve existing stable max-pair qualification')
    initial=json.loads((root/'CASES.json').read_text());additional=json.loads((root/'ADDITIONAL_CASES.json').read_text())
    if {q['cell']:q['epoch'] for q in initial}!=WITNESSES or len(initial)!=4 or len(additional)!=4:
        raise ValueError('four original witnesses and four fixed pretrained controls required')
    evidence={str(study/'PLAN.json'):sha(study/'PLAN.json'),
              str(study/'corrected_thresholds/ACTIVATION.json'):sha(study/'corrected_thresholds/ACTIVATION.json')}
    trace=source/'scripts/rev4_featpot/shaping_probe_trace.py';comparison=source/'scripts/rev4_featpot/shaping_reexport.py'
    evidence.update({str(p):sha(p) for p in (trace,comparison,root/'CASES.json',root/'ADDITIONAL_CASES.json')})
    cases=[]
    for item in [*initial,*additional]:
        cid=item['cell'];cell=registered[cid];epoch=item['epoch'];folder=root/cid
        if cell['family'] not in FAMILIES or cell['stage']!='inner':raise ValueError('registered inner numerical case required')
        witness=cid in WITNESSES
        if not witness and (cell['family'] not in FAMILIES or '_pretrained_' not in cell['family'] or
            cell['partition']!='inner-0-0' or cell['setting']!=0 or cell['seed']!=1 or epoch!=119):
            raise ValueError('fixed setting-zero seed-one pretrained case required')
        probe_path=folder/'RESULT.json';compare_path=folder/'comparison/RESULT.json'
        probe=json.loads(probe_path.read_text());compare=json.loads(compare_path.read_text())
        if (probe['driver_sha256']!=sha(trace) or probe['identity']!=100. or probe['finite'] is not True
            or probe['damage_increase_violations']!=0 or not math.isfinite(probe['maximum_damage_increase'])
            or probe['maximum_damage_increase']>1e-5
            or probe['selection_candidate'] is not False):raise ValueError('native stable probe qualification failed')
        if (compare['driver_sha256']!=sha(comparison) or compare['cell']!=cell or compare['checkpoint_epoch']!=epoch
            or compare['all_unmarked_exact'] is not True or compare['outer_results_read']
            or compare['selection_candidate'] is not False):raise ValueError('native population comparison failed')
        rows=compare['comparisons']
        if len(rows)!=10 or {(q['group'],q['split']) for q in rows}!={(g,s) for g in GROUPS for s in ('train','validation')}:
            raise ValueError('complete TRAIN and inner-validation population comparison required')
        if any(q['rows']<=0 or not q['unmarked_exact'] or not math.isfinite(q['max_abs_delta'])
               or q['max_abs_delta']<0 for q in rows):raise ValueError('invalid native comparison measurements')
        # Read all native qualification inputs once when sealing this certificate.
        # Later admission binds these immutable proof receipts; each new export
        # still independently validates its own model, inputs and binary bytes.
        for proof in (probe,compare):
            for name,pin in proof['evidence_sha256'].items():
                if sha(name)!=pin:raise ValueError('native qualification input drift: '+name)
        build=study.parent/'bounded-resume-runtime-20261108/binaries-v3/BUILD.json'
        if sha(build)!=BUILD_SHA256:raise ValueError('qualified native build differs')
        evidence.update({str(p):sha(p) for p in (probe_path,compare_path,build)})
        if witness:
            old=study/'runs'/cid/'attempt-001/curve_evaluation'/f'epoch-{epoch:03}'/'FAILED_GATE.json'
            gate=json.loads(old.read_text())
            if gate['damage_increase_violations']<=0 or gate['maximum_damage_increase']<=1e-5:
                raise ValueError('original failed witness must remain a failure')
            evidence[str(old)]=sha(old)
        cases.append(dict(cell=cell,epoch=epoch,witness=witness,population_rows=sum(q['rows'] for q in rows),
                          maximum_population_score_delta=max(q['max_abs_delta'] for q in rows)))
    value=dict(schema='qualified-stable-max-pair-families-v1',families=list(FAMILIES),cases=cases,
        plan_sha256=sha(study/'PLAN.json'),build_sha256=BUILD_SHA256,evidence_sha256=evidence,
        fitting_changed=False,numerical_threshold_changed=False,outer_results_read=False,
        original_evidence_preserved=True,driver_sha256=sha(__file__),
        application='Uniform final and scheduled diagnostic exports for every paired cell of each admitted family; no per-cell repair selection.')
    with output.open('xb') as stream:stream.write(encoded(value))
    families(study)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','qualification','source'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();value=register(a.study,a.qualification,a.source)
    print(json.dumps(dict(families=value['families'],cases=len(value['cases']),fitting_changed=False)),flush=True)
