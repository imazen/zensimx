"""Commit-sized progress receipt for packed R13 fits; no assessment reads."""
import argparse
import json
from pathlib import Path

import r13_width as owner


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    rows=[]
    pending=[]
    for seed in range(3):
        for width in owner.ORDER:
            name=f'W{width}-s{seed}'
            path=owner.O/'fits'/name/'FIT.json'
            if not path.exists():
                pending.append(name)
                continue
            d=json.loads(path.read_text())
            if owner.sha(d['packed_model'])!=d['packed_model_sha256'] or not d['certificate']['certified']:
                raise ValueError('packed fit identity/certificate differs')
            resources={}
            for stage in ('base','continuation','pack'):
                matches=[v for v in (owner.O/'logs'/f'{name}-{stage}.log').read_text().splitlines()
                         if v.startswith('run-heavy: done')]
                if len(matches)!=1:raise ValueError('complete unique resource receipt required')
                resources[stage]=matches[0]
            rows.append(dict(name=name,hidden=width,initial_max_pairs=width//4,seed=seed,
                 packed_sha256=d['packed_model_sha256'],packed_bytes=Path(d['packed_model']).stat().st_size,
                 certified=d['certificate']['certified'],certified_max_pairs=d['certificate']['max_pairs'],
                 capacity={k:d['capacity'][k] for k in ('dead','residual','live')},
                 wall_seconds_excluding_lock_waits=d['wall_seconds'],stage_times=d['stage_times'],
                 resource_receipts=resources,finished=d['finished']))
    pins=json.loads((owner.O/'SOURCE_PINS.json').read_text())
    owner.write(a.out,dict(schema='r13-packed-fit-progress-v1',registration='25e772922fd5a40d576e12af47778ce81bed5e08',
          trainer_source_commit=pins['source_commit'],reporting_source_sha256=owner.sha(__file__),
          complete_fits=len(rows),required_fits=15,models=rows,pending=pending,
          artifacts=dict(host='dev',root=str(owner.O),source_pins_sha256=owner.sha(owner.O/'SOURCE_PINS.json')),
          role='packed-fit identities/capacity only; all fifteen must freeze before R13 EVAL assessments'))


if __name__=='__main__':main()
