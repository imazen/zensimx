"""Freeze queue admissions for placement maintenance; native children keep running."""
from contextlib import ExitStack
from datetime import datetime, timezone
import argparse,fcntl,json,os,signal,socket,time
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_host_assignment import process_identity


def queues(study,proc=Path('/proc')):
    found=[]
    for directory in proc.glob('[0-9]*'):
        try:
            argv=[v.decode() for v in (directory/'cmdline').read_bytes().split(b'\0') if v]
            if not argv or Path(argv[0]).name not in ('python3','python'):continue
            owners=[v for v in argv if Path(v).name=='shaping_rescue_worker.py' or
                    Path(v).name.startswith('shaping_compact_worker-')]
            if len(owners)!=1 or 'wave' not in argv:continue
            compact=Path(owners[0]).name!='shaping_rescue_worker.py'
            option='--wave' if compact else '--request'
            manifest=Path(argv[argv.index(option)+1]);wave=json.loads(manifest.read_text())
            requests=[json.loads(Path(v).read_text()) for v in wave['requests']]
            if not requests or {r['study'] for r in requests}!={str(study)}:continue
            if any(sha(v)!=wave['request_sha256'][v] for v in wave['requests']):
                raise ValueError('queue request drift')
            status=(Path(argv[argv.index('--output')+1]) if compact else manifest.parent)/'WAVE_STATUS.json'
            row=json.loads(status.read_text());pid=int(directory.name);state,start=process_identity(proc,pid)
            if row['pid']!=pid or row['manifest_sha256']!=sha(manifest) or state in ('T','Z','X'):
                raise ValueError('queue status or process identity differs')
            found.append(dict(pid=pid,start_ticks=start,command=argv,manifest=str(manifest),
                manifest_sha256=sha(manifest),cells=[r['cell']['id'] for r in requests]))
        except (FileNotFoundError,ProcessLookupError):continue
    if not found:raise ValueError('no live queues for this study')
    return found


def resume(record,proc=Path('/proc')):
    for row in record['queues']:
        state,start=process_identity(proc,row['pid'])
        if start!=row['start_ticks'] or state!='T':raise ValueError('stopped queue identity differs')
    for row in record['queues']:os.kill(row['pid'],signal.SIGCONT)


def freeze(study,output,proc=Path('/proc')):
    if output.exists():raise ValueError('preserve existing queue maintenance receipt')
    selected=queues(study,proc);stopped=[]
    record=dict(schema='shaping-queue-placement-freeze-v1',host=socket.gethostname(),study=str(study),
                queues=selected,stage='prepared',native_processes_signalled=[],driver_sha256=sha(__file__))
    with ExitStack() as stack:
        for name in ('cpu-admission','storage-admission'):
            lock=stack.enter_context((study/name/'transaction.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            for row in selected:
                pid=row['pid'];state,start=process_identity(proc,pid)
                if start!=row['start_ticks'] or state in ('T','Z','X'):raise ValueError('queue changed before freeze')
                os.kill(pid,signal.SIGSTOP);stopped.append(row)
                for _ in range(200):
                    states=[p.read_text().rsplit(')',1)[1].split()[0] for p in (proc/str(pid)/'task').glob('*/stat')]
                    if states and all(s=='T' for s in states):break
                    time.sleep(.01)
                else:raise ValueError('queue threads did not reach stopped boundary')
                row['children']=(proc/str(pid)/'task'/str(pid)/'children').read_text().split()
            record.update(stage='queue_admissions_frozen',checked_utc=datetime.now(timezone.utc).isoformat())
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('xb') as stream:stream.write(encoded(record))
        except BaseException:
            # No placement has changed. Return every successfully stopped parent.
            for row in stopped:
                if process_identity(proc,row['pid'])[1]==row['start_ticks']:os.kill(row['pid'],signal.SIGCONT)
            raise
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('freeze','resume'))
    p.add_argument('--study',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    if a.mode=='freeze':value=freeze(a.study,a.receipt)
    else:
        value=json.loads(a.receipt.read_text())
        if value['study']!=str(a.study) or value['host']!=socket.gethostname():raise ValueError('maintenance host or study differs')
        resume(value);value.update(stage='queues_resumed',checked_utc=datetime.now(timezone.utc).isoformat())
        destination=a.receipt.with_name(a.receipt.stem+'-RESUMED.json')
        with destination.open('xb') as stream:stream.write(encoded(value))
    print(json.dumps(dict(stage=value['stage'],queues=len(value['queues']),native_processes_signalled=[])),flush=True)
