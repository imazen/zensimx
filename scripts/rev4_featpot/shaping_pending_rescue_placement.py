"""Retire only unclaimed, childless rescue waiters before changing placement."""
from contextlib import ExitStack
from datetime import datetime,timezone
import argparse,fcntl,json,os,signal,socket,time
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_host_assignment import process_identity


def idle(study,cid,pid,claims,storage,proc=Path('/proc')):
    return (cid not in claims and not (study/'runs'/cid).exists()
            and not any('/runs/'+cid+'/' in key for key in storage)
            and not (proc/str(pid)/'task'/str(pid)/'children').read_text().strip())


def retire(study,limit,output,proc=Path('/proc')):
    if not 1<=limit<=20:raise ValueError('bounded pending placement count required')
    output.mkdir(parents=True,exist_ok=False)
    record=dict(stage='checking_unstarted_waiters',host=socket.gethostname(),study=str(study),
                rows=[],native_processes_signalled=[],driver_sha256=sha(__file__),training_replayed=False)
    journal=output/'STOPPED.json';journal.write_bytes(encoded(record));stopped=[]
    with ExitStack() as stack:
        for name in ('cpu-admission','storage-admission'):
            lock=stack.enter_context((study/name/'transaction.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX)
        claims_path=study/'cpu-admission/CLAIMS.json';storage_path=study/'storage-admission/RESERVATIONS.json'
        claims=json.loads(claims_path.read_text()) if claims_path.exists() else {}
        storage=json.loads(storage_path.read_text()) if storage_path.exists() else {}
        try:
            for directory in sorted(proc.glob('[0-9]*'),key=lambda p:int(p.name)):
                try:
                    argv=[v.decode() for v in (directory/'cmdline').read_bytes().split(b'\0') if v]
                    positions=[i for i,v in enumerate(argv) if Path(v).name=='shaping_rescue_worker.py']
                    if len(positions)!=1 or argv[positions[0]+1:positions[0]+2]!=['work'] or '--request' not in argv:continue
                    if not Path(os.readlink(directory/'exe')).name.startswith('python'):continue
                    request=Path(argv[argv.index('--request')+1]);q=json.loads(request.read_text());cid=q['cell']['id'];pid=int(directory.name)
                    if q['study']!=str(study) or q['cell']['stage']!='rescue' or q['host']!=socket.gethostname():continue
                    if q['driver_sha256']!=sha(argv[positions[0]]) or q['plan_sha256']!=sha(study/'PLAN.json'):
                        raise ValueError('waiter source or study identity differs')
                    state,start=process_identity(proc,pid)
                    if state in ('T','Z','X') or request.with_name('STATUS.json').exists() or not idle(study,cid,pid,claims,storage,proc):continue
                    os.kill(pid,signal.SIGSTOP)
                    stopped.append(dict(pid=pid,start_ticks=start,request=str(request),request_sha256=sha(request),cell=q['cell']))
                    for _ in range(200):
                        state,current=process_identity(proc,pid)
                        if current!=start:raise ValueError('waiter PID changed')
                        if state=='T':break
                        time.sleep(.01)
                    else:raise ValueError('waiter did not stop')
                    if not idle(study,cid,pid,claims,storage,proc):raise ValueError('waiter became claimed or acquired scientific work')
                    if len(stopped)>=limit:break
                except (FileNotFoundError,ProcessLookupError):continue
            if not stopped:raise ValueError('no unclaimed unstarted waiters')
            record.update(stage='verified_unstarted_waiters',rows=stopped,checked_utc=datetime.now(timezone.utc).isoformat())
            journal.write_bytes(encoded(record))
            for row in stopped:
                pid=row['pid'];state,start=process_identity(proc,pid)
                if state!='T' or start!=row['start_ticks'] or not idle(study,row['cell']['id'],pid,claims,storage,proc):
                    raise ValueError('stopped waiter identity changed')
                os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
                for _ in range(200):
                    try:
                        state,start=process_identity(proc,pid)
                        if start!=row['start_ticks']:raise ValueError('waiter PID reused')
                        if state in ('Z','X'):break
                    except FileNotFoundError:break
                    time.sleep(.01)
                else:raise ValueError('retired waiter remains live')
                row['retired_before_fit']=True;journal.write_bytes(encoded(record))
            record.update(stage='unstarted_waiters_retired',checked_utc=datetime.now(timezone.utc).isoformat())
            journal.write_bytes(encoded(record));return record
        finally:
            for row in stopped:
                try:
                    state,start=process_identity(proc,row['pid'])
                    if state=='T' and start==row['start_ticks']:os.kill(row['pid'],signal.SIGCONT)
                except FileNotFoundError:pass


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--limit',type=int,required=True);a=p.parse_args();r=retire(a.study,a.limit,a.output)
    print(json.dumps(dict(stage=r['stage'],retired=len(r['rows']),native_processes_signalled=[])),flush=True)
