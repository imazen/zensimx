"""Opt-in resource sizing for measured streaming families; no numerical changes."""
import json,time
from pathlib import Path
from shaping_completion import sha

NAME='STREAMING_MEMORY_QUALIFICATION.json'
FAMILIES=('relu','product_joint_identity','separate_joint_identity')
CAP=2*1024**3


def family(record):
    return record['resolution'].get('resolution',{}).get('resolved_family',record['cell']['family'])


def qualified(study):
    path=study/NAME
    if not path.exists():return None
    q=json.loads(path.read_text())
    if (q['schema']!='streaming-480-memory-qualification-v1' or q['plan_sha256']!=sha(study/'PLAN.json')
        or q['cap_bytes']!=CAP or q['families']!=list(FAMILIES) or q['epochs']!=480
        or q['scientific_inputs_changed'] or q['native_binaries_changed'] or q['numerical_gates_changed']
        or set(q['cases'])!=set(FAMILIES)):
        raise ValueError('streaming resource qualification differs')
    for name,pin in q['evidence_sha256'].items():
        if sha(name)!=pin:raise ValueError('streaming resource evidence drift')
    for name,case in q['cases'].items():
        result=json.loads(Path(case['result']).read_text())
        if (result['cell']['family']!=name or result['stage']!='trained' or result['exit_code']!=0
            or result['epochs']!=480 or result['build_sha256']!=case['build_sha256']
            or str(case['result']) not in q['evidence_sha256']
            or str(case['service_log']) not in q['evidence_sha256']
            or not case['whole_worker_receipt'].startswith('run-heavy: done rc=0 ')
            or 'peak-RSS 0.17GiB' not in case['whole_worker_receipt']
            or case['whole_worker_receipt'] not in Path(case['service_log']).read_text().splitlines()):
            raise ValueError('completed streaming phase resource evidence differs')
    return path,q


def attach(record,study,category,source):
    if record.get('execution_kind') not in ('outer','variant'):return
    if family(record) not in FAMILIES or record['resolution']['schedule']['phase_epochs']!=[480]:return
    value=qualified(study)
    if value is None:return
    path,q=value;name=family(record)
    if record['build_sha256']!=q['cases'][name]['build_sha256']:return
    record['streaming_memory']=dict(family=name,cap_bytes=CAP,qualification=str(path),qualification_sha256=sha(path))
    pins={str(path):sha(path),**q['evidence_sha256'],str(source):sha(source)}
    for name,pin in pins.items():
        p=Path(name);kind,relative=category(p)
        record['files'][name]=dict(kind=kind,name=relative,bytes=p.stat().st_size,sha256=pin)
    record['input_bytes']=sum(row['bytes'] for row in record['files'].values())


def validate(record):
    hint=record.get('streaming_memory')
    if hint is None:return None
    study=Path(record['study']);value=qualified(study)
    if value is None:raise ValueError('streaming resource qualification required')
    path,q=value;name=family(record)
    if (record.get('execution_kind') not in ('outer','variant') or name not in FAMILIES
        or record['resolution']['schedule']['phase_epochs']!=[480]
        or record['build_sha256']!=q['cases'][name]['build_sha256']
        or hint!=dict(family=name,cap_bytes=CAP,qualification=str(path),qualification_sha256=sha(path))
        or record['files'].get(str(path),{}).get('sha256')!=sha(path)
        or any(record['files'].get(p,{}).get('sha256')!=pin for p,pin in q['evidence_sha256'].items())):
        raise ValueError('streaming resource hint differs from sealed qualification')
    return CAP


def native_cap(family_name,requested):
    if requested is None:return None
    if requested!='2G' or family_name not in FAMILIES:raise ValueError('measured streaming family and two-GiB cap required')
    return requested


def wait_for_first_checkpoint(process,study,key):
    from shaping_rescue_dispatch import worker_claims,OWNER_TOKEN
    checkpoint=study/'runs'/key/'attempt-001/fit/checkpoints/ckpt_epoch000.bin'
    while process.poll() is None:
        if checkpoint.is_file():
            with worker_claims(study) as claims:
                if claims[key]['owner']!=OWNER_TOKEN:raise ValueError('warmup claim owner differs')
                claims[key].pop('warmup_until_checkpoint',None)
            break
        time.sleep(1)
    return process.wait()
