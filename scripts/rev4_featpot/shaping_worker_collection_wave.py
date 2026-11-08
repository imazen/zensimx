"""Collect a fixed observed terminal worker wave through existing verified tools."""
from __future__ import annotations
import argparse,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import REPO,encoded,sha


def run(args):
    status=json.loads(args.status.read_text())
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(status['checked_utc'])).total_seconds()
    if not 0<=age<=120:raise ValueError('refresh worker process observation before collection')
    assignments={}
    for path in (args.study/'worker_assignments').glob('*.json'):
        record=json.loads(path.read_text())
        if record['host']!=args.host:continue
        for cell in record['cells']:
            if cell in assignments:raise ValueError('ambiguous worker assignment')
            assignments[cell]=path
    pending=[]
    for row in status['runs']:
        if row['terminal']!='trained':continue
        cell=row['cell'];local=args.study/'runs'/cell/Path(row['run']).name
        if local.exists():continue
        if cell not in assignments:raise ValueError('terminal cell has no host assignment')
        attempt=Path(row['run']).name
        if attempt!='attempt-001':
            from shaping_external_recovery import authorize_attempt
            authorize_attempt(args.study,cell,attempt)
        pending.append(dict(cell=cell,attempt=attempt,assignment=str(assignments[cell]),assignment_sha256=sha(assignments[cell])))
    if len({r['cell'] for r in pending})!=len(pending):raise ValueError('duplicate terminal attempts')
    args.output.mkdir(parents=True,exist_ok=False)
    record=dict(host=args.host,source=str(args.source),status_sha256=sha(args.status),
                observed_utc=status['checked_utc'],pending=pending,completed=[],stage='collecting',driver_sha256=sha(__file__))
    receipt=args.output/'WAVE.json';receipt.write_bytes(encoded(record))
    try:
        for row in pending:
            cell=row['cell'];assignment=row['assignment'];attempt=row['attempt']
            remote=args.study.parent/'worker-bundles'/f'{cell}-{attempt}.tar'
            destination=args.destination/f'{args.host}-{cell}-{attempt}';bundle=destination/remote.name
            with (args.output/(cell+'.log')).open('x') as log:
                def call(command):
                    log.write(json.dumps(command)+'\n');log.flush()
                    subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
                call(['just','shaping-completion-pack-worker',args.host,str(args.source),str(args.study),assignment,cell,str(remote),attempt])
                call(['just','shaping-completion-fetch-worker',args.host,str(remote),str(destination)])
                pin=json.loads(Path(str(bundle)+'.json').read_text())['bundle_sha256']
                for action in ('verify-transfer','collect'):
                    call(['just','shaping-completion-worker-collect',action,str(args.study),assignment,str(bundle),'--sha256',pin])
            record['completed'].append(dict(cell=cell,bundle=str(bundle),bundle_sha256=pin))
            receipt.write_bytes(encoded(record));print(json.dumps(record['completed'][-1]),flush=True)
        record['stage']='collected'
    except Exception as error:
        record.update(stage='failed',error=str(error));raise
    finally:
        record['checked_utc']=datetime.now(timezone.utc).isoformat();receipt.write_bytes(encoded(record))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True)
    for key in ('source','study','status','destination','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
