"""Read a hash-bound placement amendment without changing scientific scope."""
import json
from pathlib import Path
from shaping_completion import sha

NAME='COMPUTE_POOL.json'
EXCLUDED={'dev','i265','r5900xt'}


def allowed(study):
    path=Path(study)/NAME
    if not path.exists():return None
    record=json.loads(path.read_text());approval=Path(record['approval_path'])
    if (record['schema']!='shaping-expanded-compute-pool-v1' or record['plan_sha256']!=sha(Path(study)/'PLAN.json')
        or sha(approval)!=record['approval_sha256'] or set(record['excluded_hosts'])!=EXCLUDED
        or not record['scientific_protocol_unchanged']):raise ValueError('expanded compute placement identity differs')
    choice=json.loads(approval.read_text());hosts=set(record['allowed_hosts'])
    if (choice.get('approved') is not True or choice.get('all_other_known_hosts_authorized') is not True
        or set(choice['excluded_hosts'])!=EXCLUDED or hosts&EXCLUDED or not hosts):
        raise ValueError('expanded owner compute authorization differs')
    withdrawal=Path(study)/'COMPUTE_WITHDRAWALS.json'
    if withdrawal.exists():
        row=json.loads(withdrawal.read_text());proof=Path(row['direction_path'])
        if (row['schema']!='shaping-compute-withdrawals-v1' or row['plan_sha256']!=record['plan_sha256']
            or sha(proof)!=row['direction_sha256'] or not set(row['hosts'])<=hosts):
            raise ValueError('compute withdrawal identity differs')
        direction=json.loads(proof.read_text())
        if direction.get('new_compute_authorization_withdrawn') is not True or set(row['hosts'])!={direction['host']}:
            raise ValueError('compute withdrawal direction differs')
        hosts-=set(row['hosts'])
    return hosts
