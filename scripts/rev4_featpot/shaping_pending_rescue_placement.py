"""Retire only unclaimed, childless rescue waiters before changing placement."""
from contextlib import ExitStack
from datetime import datetime,timezone
import argparse,fcntl,json,os,signal,socket,time,subprocess
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


def prepare_target(study,proofs,source,binaries,qualification,target,output):
    from shaping_rescue_worker import HOSTS
    if target not in HOSTS:raise ValueError('approved target required')
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for proof in proofs:
        record=json.loads(proof.read_text())
        if record['stage']!='unstarted_waiters_retired' or record['native_processes_signalled'] or record['host'] not in HOSTS:
            raise ValueError('verified source retirement required')
        rows.extend(record['rows'])
    if not rows or len({r['cell']['id'] for r in rows})!=len(rows) or any(not r.get('retired_before_fit') for r in rows):
        raise ValueError('distinct retired unstarted cells required')
    jobs=[];placements=[]
    for row in rows:
        old_path=Path(row['request']);old=json.loads(old_path.read_text());cid=row['cell']['id']
        if sha(old_path)!=row['request_sha256'] or old['cell']!=row['cell'] or old['host']==target or (study/'runs'/cid).exists():
            raise ValueError('source request or attempt changed')
        program='''import json
from pathlib import Path
root=Path(STUDY);request=REQUEST;cid=CELL
assert not (root/'runs'/cid).exists()
claims=root/'cpu-admission/CLAIMS.json'
assert not claims.exists() or cid not in json.loads(claims.read_text())
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=[s.decode() for s in p.read_bytes().split(b'\\0') if s]
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 assert not ('work' in a and request in a and any(Path(s).name=='shaping_rescue_worker.py' for s in a))
'''
        program='STUDY='+repr(str(study))+'\nREQUEST='+repr(str(old_path))+'\nCELL='+repr(cid)+'\n'+program
        subprocess.run(['ssh',old['host'],'python3 -B -'],input=program,text=True,check=True)
        matches=[p for p in (study/'downstream_assignments').glob('*.json') if cid in json.loads(p.read_text())['cells']]
        if len(matches)!=1 or json.loads(matches[0].read_text())['host']!=old['host']:raise ValueError('one original placement required')
        job=output/'jobs'/cid
        with (output/(cid+'.log')).open('x') as log:
            subprocess.run(['python3','-B',str(source/'scripts/rev4_featpot/shaping_rescue_worker.py'),'register',
                '--study',str(study),'--binaries',str(binaries),'--qualification',str(qualification),
                '--cell',cid,'--host',target,'--output',str(job)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=source)
        request=job/'REQUEST.json';new=json.loads(request.read_text())
        if new!={**old,'host':target}:raise ValueError('placement change altered scientific request')
        assignment=study/'downstream_assignments'/('rebalanced-'+cid+'-'+target+'.json')
        if assignment.exists():raise ValueError('preserve existing replacement placement')
        jobs.append(dict(cell=new['cell'],host=target,request=str(request),request_sha256=sha(request),
                         previous_request=str(old_path),previous_request_sha256=sha(old_path),
                         assignment=str(assignment),assignment_sha256=sha(job/'ASSIGNMENT.json')))
        placements.append(dict(original=str(matches[0]),original_sha256=sha(matches[0]),
                               backup=str(output/'original-placements'/matches[0].name),new=str(assignment),content=str(job/'ASSIGNMENT.json')))
    record=dict(stage='prepared',source=str(source),source_driver_sha256=sha(source/'scripts/rev4_featpot/shaping_rescue_worker.py'),
                jobs=jobs,placements=placements,retirement_proof_sha256={str(p):sha(p) for p in proofs},
                scientific_requests_unchanged_except_host=True,training_replayed=False)
    journal=output/'REGISTRATION.json';journal.write_bytes(encoded(record))
    (output/'original-placements').mkdir()
    for item in placements:
        path=Path(item['original'])
        if sha(path)!=item['original_sha256']:raise ValueError('placement changed before installation')
        path.rename(item['backup']);Path(item['new']).write_bytes(Path(item['content']).read_bytes())
    record['stage']='registered';journal.write_bytes(encoded(record));return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=('retire','target'),default='retire');p.add_argument('--limit',type=int)
    p.add_argument('--proof',type=Path,action='append',default=[]);p.add_argument('--target')
    for key in ('source','binaries','qualification'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    r=(retire(a.study,a.limit,a.output) if a.mode=='retire' else prepare_target(a.study,a.proof,a.source,a.binaries,a.qualification,a.target,a.output))
    print(json.dumps(dict(stage=r['stage'],cells=len(r.get('rows',r.get('jobs',[]))),training_replayed=False)),flush=True)
