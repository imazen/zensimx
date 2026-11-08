"""Seal never-started corrected conditions into bounded, explicitly placed waves."""
import argparse,json,subprocess
from collections import defaultdict
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_corrected_view import registration
from shaping_rescue_worker import HOSTS


def prepare(study,source,binaries,placements,output):
    from shaping_execution import load_registration
    _,original,_=load_registration(study,original=True)
    record,matrix,cells=registration(study,original)
    if not (study/'corrected_thresholds/ACTIVATION.json').exists():
        raise ValueError('corrected activation required before wave registration')
    corrected={e['id'] for e in matrix['corrected_pretrained_conditions']}
    if (not 1<=len(placements)<=44 or not set(placements)<=corrected
        or any(host not in HOSTS for host in placements.values())):
        raise ValueError('one to 44 distinct approved corrected placements required')
    selected={c['id']:c for c in cells if c['id'] in placements}
    if set(selected)!=set(placements):raise ValueError('corrected cell coverage differs')
    if any((study/'runs'/cid).exists() for cid in placements):raise ValueError('preserve all existing attempts')
    for directory in ('worker_assignments','downstream_assignments','recovery_assignments'):
        for path in (study/directory).glob('*.json'):
            if set(placements)&set(json.loads(path.read_text())['cells']):
                raise ValueError('preserve existing cell placements')
    if sha(source/'scripts/rev4_featpot/shaping_corrected_worker.py')!=sha(Path(__file__).with_name('shaping_corrected_worker.py')):
        raise ValueError('wave and exported corrected owners differ')
    output.mkdir(parents=True,exist_ok=False)
    journal=output/'REGISTRATION.json'
    value=dict(schema='corrected-inner-worker-wave-v1',stage='registering',jobs=[],waves=[],source=str(source),
        placements=placements,activation_sha256=sha(study/'corrected_thresholds/ACTIVATION.json'),
        corrected_registration_sha256=sha(study/'corrected_thresholds/REGISTRATION.json'),
        build_sha256=sha(binaries/'BUILD.json'),outer_results_read=False,driver_sha256=sha(__file__))
    journal.write_bytes(encoded(value));groups=defaultdict(list)
    try:
        # Install non-circular placement guards first, so every request captures
        # its guard. The request-bearing ASSIGNMENT stays beside each capsule.
        directory=study/'worker_assignments';directory.mkdir(exist_ok=True)
        for host in sorted(set(placements.values())):
            ids=sorted(cid for cid,h in placements.items() if h==host)
            guard=dict(host=host,cells=ids,plan_sha256=record['plan_sha256'],build_sha256=value['build_sha256'])
            path=directory/(output.name+'-'+host+'.json')
            with path.open('xb') as stream:stream.write(encoded(guard))
        for cid in sorted(placements):
            host=placements[cid];job=output/'jobs'/cid
            command=['python3','-B',str(source/'scripts/rev4_featpot/shaping_rescue_worker.py'),'register',
                '--execution-kind','corrected_inner','--study',str(study),'--binaries',str(binaries),
                '--qualification',str(study.parent/'corrected-control-qualification-20261108'),
                '--cell',cid,'--host',host,'--output',str(job)]
            with (output/(cid+'.log')).open('x') as log:
                subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
            request=job/'REQUEST.json';groups[host].append(request)
            value['jobs'].append(dict(cell=selected[cid],host=host,request=str(request),request_sha256=sha(request)))
            journal.write_bytes(encoded(value))
        # Four workers per controller retains the existing bounded owner. The
        # independent controllers still share actual CPU/RAM/SSD admission.
        for host,requests in sorted(groups.items()):
            for offset in range(0,len(requests),4):
                chosen=requests[offset:offset+4];path=output/'waves'/host/f'{offset//4:02}'/'MANIFEST.json'
                path.parent.mkdir(parents=True)
                manifest=dict(workers=len(chosen),requests=list(map(str,chosen)),request_sha256={str(p):sha(p) for p in chosen})
                with path.open('xb') as stream:stream.write(encoded(manifest))
                value['waves'].append(dict(host=host,manifest=str(path),manifest_sha256=sha(path),workers=len(chosen)))
        value['stage']='registered'
    except Exception as error:
        value.update(stage='failed_preserved_for_reconciliation',error=str(error));raise
    finally:journal.write_bytes(encoded(value))
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','binaries','placements','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=prepare(a.study,a.source,a.binaries,json.loads(a.placements.read_text()),a.output)
    print(json.dumps(dict(stage=result['stage'],jobs=len(result['jobs']),waves=len(result['waves']))),flush=True)
