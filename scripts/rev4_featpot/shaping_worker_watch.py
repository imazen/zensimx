"""Observe worker processes and collect terminal waves through verified owners."""
from __future__ import annotations
import argparse,fcntl,json,os,shlex,shutil,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from shaping_completion import REPO,encoded,sha
from shaping_worker_collection_wave import run as collect_wave


def pending(study,status):
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(status['checked_utc'])).total_seconds()
    if not 0<=age<=120:raise ValueError('fresh worker observation required')
    if status['unexpectedly_stopped']:raise ValueError('worker has stopped attempts requiring reconciliation')
    return [r for r in status['runs'] if r['terminal']=='trained' and
            not (study/'runs'/r['cell']/Path(r['run']).name).exists()]


def observe(args):
    command=['python3','-B',str(args.status_source/'scripts/rev4_featpot/shaping_execution.py'),
             '--study',str(args.study),'--status']
    remote=shlex.join(command)
    result=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=15',
                           '-o','ServerAliveCountMax=2',args.host,remote],
                          capture_output=True,text=True,timeout=45)
    if result.returncode:raise subprocess.CalledProcessError(result.returncode,command,output=result.stdout,stderr=result.stderr)
    return json.loads(result.stdout)


def cycle(args,observation,sequence):
    rows=pending(args.study,observation)
    if not rows:return None
    stamp=f'wave-{sequence:06}'
    status=args.output/(stamp+'-observation.json')
    with status.open('xb') as stream:stream.write(encoded(observation))
    wave=args.output/stamp
    # Transport containers are working copies. The native archive owner preserves
    # their imported payload separately; do not duplicate that payload on the NAS.
    transfer_root=getattr(args,'transfer_root',None) or args.destination/'worker-transfers'
    collect_wave(SimpleNamespace(host=args.host,source=args.source,study=args.study,status=status,
                 destination=transfer_root,output=wave))
    record=json.loads((wave/'WAVE.json').read_text())
    if record['stage']!='collected' or len(record['completed'])!=len(rows):raise ValueError('worker wave incomplete')
    archive=args.destination/f'worker-watch-{args.host}-{args.output.name}-{stamp}'
    with (args.output/(stamp+'-archive.log')).open('x') as log:
        subprocess.run(['just','shaping-completion-archive',str(wave),str(archive)],cwd=REPO,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=json.loads((wave/'MIRROR_VERIFIED.json').read_text())
    if not receipt['all_artifact_sha256_verified'] or receipt['destination']!=str(archive):
        raise ValueError('worker wave archive not verified')
    if sha(wave/'SHA256SUMS')!=sha(archive/'SHA256SUMS'):raise ValueError('worker wave archive manifest differs')
    return dict(wave=str(wave),completed=len(rows),archive=receipt)


def watch(args):
    missing=[name for name in ('just','ssh','scp') if shutil.which(name) is None]
    if missing:raise ValueError('worker collection tools missing from PATH: '+', '.join(missing))
    ownership=args.study/'dispatch/worker-collection';ownership.mkdir(parents=True,exist_ok=True)
    with (ownership/(args.host+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.output.mkdir(parents=True,exist_ok=False)
        record=dict(pid=os.getpid(),host=args.host,source=str(args.source),status_source=str(args.status_source),
                    driver_sha256=sha(__file__),completed_waves=[],observation_failures=0,
                    transfer_root=str(args.transfer_root) if getattr(args,'transfer_root',None) else None)
        sequence=0
        def save(stage,**values):
            record.update(stage=stage,checked_utc=datetime.now(timezone.utc).isoformat(),**values)
            temporary=args.output/'STATUS.json.writing';temporary.write_bytes(encoded(record));temporary.replace(args.output/'STATUS.json')
        while True:
            sequence+=1
            try:
                observation=observe(args)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                # An unreachable observation says nothing about the remote fit's state.
                record['observation_failures']+=1
                save('observation_unavailable_training_state_unknown',error=str(error),stderr=str(getattr(error,'stderr','')))
                if not args.watch:raise
                time.sleep(60);continue
            latest=args.output/'LIVE_STATUS.json.writing';latest.write_bytes(encoded(observation));latest.replace(args.output/'LIVE_STATUS.json')
            save('checking_terminal_fits',worker_checked_utc=observation['checked_utc'],worker_live=observation['live'],error=None)
            try:
                result=cycle(args,observation,sequence)
                if result:record['completed_waves'].append(result)
            except Exception as error:
                save('collection_failed_preserved_for_reconciliation',error=str(error));raise
            save('watching' if args.watch else 'sweep_finished')
            if not args.watch:return
            time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',choices=['i265','i270','r7900x','r3800x','i134','r5600g','r3500'],required=True)
    for key in ('source','status-source','study','destination','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--transfer-root',type=Path,help='SSD directory for transient transfer containers; native archive destination is unchanged')
    p.add_argument('--watch',action='store_true');watch(p.parse_args())
