"""Owner-approved aggregate admission for independent one-thread trainers."""
from __future__ import annotations
import json,os,socket
from pathlib import Path
from shaping_completion import encoded,sha

NAME='TRAINING_CPU_BUDGET.json'


def register(study,approval):
    row=json.loads(approval.read_text())
    if row['approved'] is not True or not row['per_job_thread_caps_unchanged'] or not row['ram_and_storage_guards_unchanged']:
        raise ValueError('explicit aggregate-only resource approval required')
    for value in row['limits'].values():
        if value not in (dict(logical_cpus=6,max_trainers=5),dict(logical_cpus=12,max_trainers=11),dict(logical_cpus=16,max_trainers=14)):
            raise ValueError('aggregate limit outside approved headroom')
    record=dict(schema='shaping-training-cpu-budget-v1',limits=row['limits'],approval_path=str(approval),
                approval_sha256=sha(approval),plan_sha256=sha(study/'PLAN.json'))
    with (study/NAME).open('xb') as out:out.write(encoded(record))
    return record


def training_limit(study=None,cpu_count=None,*,host=None):
    count=len(os.sched_getaffinity(0)) if cpu_count is None else cpu_count
    path=Path(study)/NAME if study is not None else None
    if path is None or not path.exists():return max(1,count-4)
    record=json.loads(path.read_text())
    if record['schema']!='shaping-training-cpu-budget-v1' or record['plan_sha256']!=sha(Path(study)/'PLAN.json'):
        raise ValueError('aggregate CPU budget identity differs')
    approval=Path(record['approval_path'])
    if sha(approval)!=record['approval_sha256']:raise ValueError('aggregate CPU approval drift')
    row=json.loads(approval.read_text())
    if (row.get('approved') is not True or row.get('per_job_thread_caps_unchanged') is not True
        or row.get('ram_and_storage_guards_unchanged') is not True or row['limits']!=record['limits']):
        raise ValueError('aggregate-only CPU approval differs')
    name=host or socket.gethostname()
    if name not in record['limits']:
        from shaping_compute_pool import allowed
        hosts=allowed(study)
        if hosts is None or name not in hosts:raise ValueError('host has no approved compute budget')
        # Newly admitted machines retain the ordinary four-thread headroom.
        # The four previously approved tighter aggregate ceilings stay intact.
        return max(1,count-4)
    limit=record['limits'][name]
    if limit['logical_cpus']!=count or not 1<=limit['max_trainers']<count:
        raise ValueError('observed CPU availability differs from approved budget')
    return limit['max_trainers']
