"""Prepare a complete fixed-role ReLU rescue wave through the existing owner."""
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
    protocol_snapshot(study,source)
    output.mkdir(parents=True,exist_ok=False)
    record=dict(source=str(source),source_driver_sha256=sha(source/'scripts/rev4_featpot/shaping_rescue_worker.py'),
                outer=outer,family='relu',jobs=[],stage='registering',driver_sha256=sha(__file__))
    journal=output/'REGISTRATION.json';journal.write_bytes(encoded(record))
    try:
        for cell in selected:
            host=placements[cell['id']];job=output/'jobs'/cell['id']
            command=['python3','-B',str(source/'scripts/rev4_featpot/shaping_rescue_worker.py'),'register',
                '--study',str(study),'--binaries',str(binaries),'--qualification',str(qualification),
                '--cell',cell['id'],'--host',host,'--output',str(job)]
            with (output/(cell['id']+'.log')).open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=source)
            assignment=study/'downstream_assignments'/f"relu-outer-{outer}-{cell['id']}-{host}.json"
            with assignment.open('xb') as stream:stream.write((job/'ASSIGNMENT.json').read_bytes())
            record['jobs'].append(dict(cell=cell,host=host,request=str(job/'REQUEST.json'),request_sha256=sha(job/'REQUEST.json'),
                assignment=str(assignment),assignment_sha256=sha(assignment)))
            journal.write_bytes(encoded(record))
        record['stage']='registered'
    except Exception as error:
        record.update(stage='failed_preserved_for_reconciliation',error=str(error));raise
    finally:journal.write_bytes(encoded(record))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','binaries','qualification','placements','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--outer',type=int,required=True);a=p.parse_args()
    record=prepare(a.study,a.source,a.binaries,a.qualification,a.outer,json.loads(a.placements.read_text()),a.output)
    print(json.dumps(dict(stage=record['stage'],jobs=len(record['jobs']))),flush=True)
