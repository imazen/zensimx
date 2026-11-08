"""Freeze approved corrected condition identities without altering the old study."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import REPO,encoded,sha
from shaping_control_compatibility import BUILD_SHA256,PROPOSAL_SHA256

MATRIX_SHA256='7102573bee15e35e119eb6c4d7eeb541987c946ff30f38dd568c1617f3206ee9'
FAMILIES=('product_pretrained_frozen','product_pretrained_joint','separate_pretrained_frozen','separate_pretrained_joint')
OBSERVATION_HOSTS=('r3500','r5600g','i134','r3800x','i265','i270')


def validate_matrix(matrix,cells):
    from shaping_pretraining import matching_parent
    entries=matrix['corrected_pretrained_conditions'];registered={c['id']:c for c in cells}
    expected={(f,f'inner-{outer}-{fold}',seed,setting) for f in FAMILIES
              for outer in range(3) for fold in range(2) for seed in range(3) for setting in range(4)}
    actual=set();ids=set()
    for entry in entries:
        old=entry['original_condition'];key=(old['family'],old['partition'],old['seed'],old['setting'])
        if (entry['schema']!='corrected-shaped-threshold-condition-v1' or old!=registered.get(old['id'])
            or old['stage']!='inner' or entry['parent']!=matching_parent(old,cells)
            or entry['source_build_sha256']!=BUILD_SHA256 or entry['epochs']!=120 or entry['pairs_per_epoch']!=50000
            or entry['initializer']!='quantiles-of-effective-shaped-training-input'
            or len(entry['id'])!=20 or any(c not in '0123456789abcdef' for c in entry['id'])
            or entry['id'] in registered or entry['id'] in ids or key in actual):
            raise ValueError('corrected condition identity, parent or exposure differs')
        actual.add(key);ids.add(entry['id'])
    if (len(entries)!=288 or actual!=expected or matrix['settings']!=[0,1,2,3]
        or matrix['paired_seeds']!=[0,1,2] or matrix['eligible_inner_cells']!=816):
        raise ValueError('complete four-setting paired corrected matrix required')
    reused=[c for c in cells if c['stage']=='inner' and c['family'] not in FAMILIES]
    fresh=[c for c in cells if c['stage']=='fresh_control_inner']
    for name,values in [('reused_original_primary_controls',reused),('matched_fresh_controls',fresh)]:
        if sorted(matrix[name],key=lambda c:c['id'])!=sorted(values,key=lambda c:c['id']):
            raise ValueError('matching original controls differ')
    return entries


def register(study,approval_path,proposal_path,binaries,observations,output):
    from shaping_execution import load_registration
    if sha(proposal_path)!=PROPOSAL_SHA256:raise ValueError('frozen approved proposal differs')
    approval=json.loads(approval_path.read_text());proposal=json.loads(proposal_path.read_text())
    if (approval['answer']!='Approve four settings: 1,461 fits' or approval['selected_option']!='four_settings'
        or approval['proposal_sha256']!=PROPOSAL_SHA256 or approval['matrix_sha256']!=MATRIX_SHA256
        or approval['corrected_fits']!=288 or approval['omitted_still_unstarted_originals']!=108
        or approval['total_core_fits']!=1461 or not approval['preserve_existing_and_active_fits']
        or not approval['qualification_required_before_control_reuse']):raise ValueError('four-setting owner approval required')
    option=proposal['options']['four_settings'];matrix_path=Path(option['matrix'])
    if sha(matrix_path)!=MATRIX_SHA256 or sha(binaries/'BUILD.json')!=BUILD_SHA256:
        raise ValueError('corrected matrix or native build differs')
    _,cells,_=load_registration(study);matrix=json.loads(matrix_path.read_text());entries=validate_matrix(matrix,cells)
    if proposal['plan_sha256']!=sha(study/'PLAN.json'):raise ValueError('original plan differs')
    omitted=proposal['unstarted_original_conditions'];ids={c['id'] for c in omitted}
    original={e['original_condition']['id']:e['original_condition'] for e in entries}
    if len(ids)!=108 or any(c!=original.get(c['id']) for c in omitted):raise ValueError('omission scope differs')
    if set(observations)!=set(OBSERVATION_HOSTS):raise ValueError('all current and drained worker observations required')
    snapshots={};now=datetime.now(timezone.utc)
    for host,path in observations.items():
        raw=path.read_bytes();record=json.loads(raw);age=(now-datetime.fromisoformat(record['checked_utc'])).total_seconds()
        if not 0<=age<=120 or record['unexpectedly_stopped'] or any(r['cell'] in ids for r in record['runs']):
            raise ValueError('omitted conditions require fresh unstarted worker evidence')
        snapshots[host]=raw
    if any((study/'runs'/cid).exists() for cid in ids):raise ValueError('preserve every existing original attempt')
    compatibility=study.parent/'corrected-control-qualification-20261108/REGISTRATION.json'
    related=json.loads(compatibility.read_text())
    if related['approval_sha256']!=sha(approval_path) or related['build_sha256']!=BUILD_SHA256:
        raise ValueError('compatibility registration is not bound to this approval')
    pins={str(p):sha(p) for p in (approval_path,proposal_path,matrix_path,binaries/'BUILD.json',compatibility,
                                  REPO/'docs/SHAPING_THRESHOLD_AMENDMENT.md',study/'PLAN.json',study/'MATRIX.json',study/'PARTITIONS.json')}
    record=dict(schema='corrected-shaped-threshold-registration-v1',core_fit_target=1461,
        corrected_condition_count=288,omitted_original_conditions=omitted,original_evidence_preserved=True,
        evidence_sha256=pins,matrix_path=str(matrix_path),plan_sha256=sha(study/'PLAN.json'),
        source_build_sha256=BUILD_SHA256,settings=[0,1,2,3],paired_seeds=[0,1,2],
        selector='existing equal-human pooled/within composite; all paired folds and seeds; existing whole-setting undefined policy',
        original_pretrained_outcomes='preserved auxiliary evidence; never substituted for corrected outcomes',
        downstream_budgets_unchanged=True,corrected_execution_requires_control_compatibility=True,
        execution_activated=False,outer_results_read=False,driver_sha256=sha(__file__),
        observation_sha256={host:hashlib.sha256(raw).hexdigest() for host,raw in snapshots.items()})
    output.mkdir(parents=True,exist_ok=False);(output/'observations').mkdir()
    for host,raw in snapshots.items():(output/'observations'/(host+'.json')).write_bytes(raw)
    (output/'REGISTRATION.json').write_bytes(encoded(record));return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','approval','proposal','binary-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--worker-observation',action='append',default=[])
    a=p.parse_args();observations={}
    for value in a.worker_observation:
        host,path=value.split('=',1)
        if host in observations:p.error('duplicate worker observation')
        observations[host]=Path(path)
    result=register(a.study,a.approval,a.proposal,a.binary_dir,observations,a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('omitted_original_conditions','evidence_sha256')}),flush=True)
