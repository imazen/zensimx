"""Prepare fixed-role or verified shortlisted rescues through the existing owner."""
import argparse,json,subprocess
from pathlib import Path
from shaping_completion import encoded,sha


def protocol_snapshot(study,source):
    name='SHAPING_REDUCED_SCOPE.md';record=json.loads((study/'REDUCED_SCOPE.json').read_text())
    pins=[pin for path,pin in record['evidence_sha256'].items() if Path(path).name==name]
    if len(pins)!=1 or sha(source/'docs'/name)!=pins[0]:raise ValueError('frozen reduced policy differs')
    target=study/'REDUCED_PROTOCOL.md'
    if target.exists():
        if sha(target)!=pins[0]:raise ValueError('preserve changed reduced policy snapshot')
    else:
        with target.open('xb') as stream:stream.write((source/'docs'/name).read_bytes())
    return target


def prepare(study,source,binaries,qualification,outer,placements,output):
    from shaping_execution import load_registration
    from shaping_rescue_dispatch import base_ready
    from shaping_rescue_worker import HOSTS
    _,cells,_=load_registration(study)
    selected=sorted([c for c in cells if c['stage']=='rescue' and c['family']=='relu'
                     and c['partition'].startswith(f'inner-{outer}-') and c['rescue'] in ('init_half','constant_lr')],key=lambda c:c['id'])
    ids={c['id'] for c in selected}
    if (outer not in range(3) or len(selected)!=12 or set(placements)!=ids
        or any(host not in HOSTS for host in placements.values())):raise ValueError('complete approved fixed-role ReLU wave required')
    if not base_ready(study,cells,'relu',outer):raise ValueError('paired ReLU base is not ready')
    if any((study/'runs'/cid).exists() for cid in ids):raise ValueError('preserve existing attempts')
    for path in (study/'downstream_assignments').glob('*.json'):
        if ids&set(json.loads(path.read_text())['cells']):raise ValueError('preserve existing placements')
    return emit(study,source,binaries,qualification,outer,'relu',selected,placements,output)


def selected_conditions(study,outer,family):
    """Resolve the registered shortlist without changing or retrying any fit."""
    from shaping_reduced_scope import shortlist,RESCUE_ROLES
    from shaping_completion import NEURAL
    from shaping_rescue import resolve
    scope,path=shortlist(study,outer)
    if family not in NEURAL or family not in {scope['roles'][role] for role in RESCUE_ROLES}:
        raise ValueError('family is outside the verified rescue shortlist')
    selected=sorted((c for c in scope['cells'] if c['stage']=='rescue' and c['family']==family),key=lambda c:c['id'])
    expected={(f'inner-{outer}-{fold}',seed,choice) for fold in range(2) for seed in range(3)
              for choice in ('init_half','constant_lr')}
    if len(selected)!=12 or {(c['partition'],c['seed'],c['rescue']) for c in selected}!=expected:
        raise ValueError('complete paired shortlisted rescue matrix required')
    for cell in selected:
        if resolve(study,cell['id'])['cell']!=cell:raise ValueError('rescue resolution differs from shortlist')
    occupied={};selected_ids={c['id'] for c in selected}
    for directory in ('worker_assignments','recovery_assignments','downstream_assignments'):
        for item in (study/directory).glob('*.json'):
            for cid in json.loads(item.read_text())['cells']:
                if cid not in selected_ids:continue
                if cid in occupied:raise ValueError('ambiguous existing rescue placement')
                occupied[cid]=str(item)
    pending=[];preserved=[]
    for cell in selected:
        attempt=study/'runs'/cell['id']
        if attempt.exists() or cell['id'] in occupied:
            preserved.append(dict(cell=cell,attempt_directory=str(attempt) if attempt.exists() else None,
                                  assignment=occupied.get(cell['id']),completion_claim=False))
        else:pending.append(cell)
    return pending,dict(shortlist_path=str(path),shortlist_sha256=sha(path),
                        registered_conditions=selected,preserved_conditions=preserved)


def prepare_selected(study,source,binaries,qualification,outer,family,placements,output):
    from shaping_rescue_worker import HOSTS
    selected,scope=selected_conditions(study,outer,family)
    if set(placements)!={c['id'] for c in selected} or any(h not in HOSTS for h in placements.values()):
        raise ValueError('exact unstarted and unassigned shortlisted placements required')
    if not selected:
        return dict(stage='no_unassigned_conditions',jobs=[],waves=[],**scope,completion_claim=False)
    return emit(study,source,binaries,qualification,outer,family,selected,placements,output,scope=scope)


def emit(study,source,binaries,qualification,outer,family,selected,placements,output,scope=None):
    protocol_snapshot(study,source)
    output.mkdir(parents=True,exist_ok=False)
    (study/'downstream_assignments').mkdir(exist_ok=True)
    record=dict(source=str(source),source_driver_sha256=sha(source/'scripts/rev4_featpot/shaping_rescue_worker.py'),
                outer=outer,family=family,jobs=[],stage='registering',driver_sha256=sha(__file__))
    if scope is not None:record.update(selected_scope=scope,training_replayed=False,outer_results_read=False)
    journal=output/'REGISTRATION.json';journal.write_bytes(encoded(record))
    try:
        for cell in selected:
            host=placements[cell['id']];job=output/'jobs'/cell['id']
            command=['python3','-B',str(source/'scripts/rev4_featpot/shaping_rescue_worker.py'),'register',
                '--study',str(study),'--binaries',str(binaries),'--qualification',str(qualification),
                '--cell',cell['id'],'--host',host,'--output',str(job)]
            with (output/(cell['id']+'.log')).open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=source)
            assignment=study/'downstream_assignments'/f"{family}-outer-{outer}-{cell['id']}-{host}.json"
            with assignment.open('xb') as stream:stream.write((job/'ASSIGNMENT.json').read_bytes())
            record['jobs'].append(dict(cell=cell,host=host,request=str(job/'REQUEST.json'),request_sha256=sha(job/'REQUEST.json'),
                assignment=str(assignment),assignment_sha256=sha(assignment)))
            journal.write_bytes(encoded(record))
        record['waves']=[]
        for host in sorted(set(placements.values())):
            requests=[Path(j['request']) for j in record['jobs'] if j['host']==host]
            for offset in range(0,len(requests),4):
                chosen=requests[offset:offset+4];path=output/'waves'/host/f'{offset//4:02}'/'MANIFEST.json'
                path.parent.mkdir(parents=True)
                with path.open('xb') as stream:stream.write(encoded(dict(workers=len(chosen),requests=list(map(str,chosen)),
                    request_sha256={str(p):sha(p) for p in chosen})))
                record['waves'].append(dict(host=host,manifest=str(path),manifest_sha256=sha(path),workers=len(chosen)))
        record['stage']='registered'
    except Exception as error:
        record.update(stage='failed_preserved_for_reconciliation',error=str(error));raise
    finally:journal.write_bytes(encoded(record))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','binaries','qualification','placements','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--selected-family',help='require the frozen shortlist and preserve all existing conditions')
    p.add_argument('--outer',type=int,required=True);a=p.parse_args()
    args=(a.study,a.source,a.binaries,a.qualification,a.outer)
    placements=json.loads(a.placements.read_text())
    record=(prepare_selected(*args,a.selected_family,placements,a.output) if a.selected_family else prepare(*args,placements,a.output))
    print(json.dumps(dict(stage=record['stage'],jobs=len(record['jobs']))),flush=True)
