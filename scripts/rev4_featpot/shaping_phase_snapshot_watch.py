"""Archive and retire only completed, collected native phase state copies."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

from shaping_completion import REPO,encoded,sha
from shaping_rescue_worker import HOSTS


def collected(study,host):
    result={}
    for marker in sorted((study/'runs').glob('*/attempt-*/WORKER_COLLECTION.json')):
        receipt=json.loads(marker.read_text())
        if receipt['source_host']!=host or receipt.get('continued_collection'):continue
        run=marker.parent;parent=json.loads((run/'RUN.json').read_text())
        terminal=json.loads((run/'RESULT.json').read_text())
        if terminal.get('stage')!='trained' or terminal.get('exit_code')!=0:continue
        pins={};phases=[]
        for phase in ('','pretraining','fit'):
            identity=run/phase/'RUN.json'
            if phase:
                if not identity.exists() or not json.loads(identity.read_text()).get('snapshot_phase'):continue
            elif not (run/'SNAPSHOT_BINDING.json').exists():continue
            phases.append(phase)
            for name in ('RUN.json','RESULT.json','SNAPSHOT_BINDING.json'):
                path=run/phase/name;relative=path.relative_to(run).as_posix()
                if relative not in receipt['files'] or sha(path)!=receipt['files'][relative]['sha256']:
                    raise ValueError('collected phase state evidence differs: '+str(path))
                pins[relative]=sha(path)
        if not phases:continue
        for name in ('RUN.json','RESULT.json'):
            if sha(run/name)!=receipt['files'][name]['sha256']:
                raise ValueError('collected parent identity differs')
            pins[name]=sha(run/name)
        key=parent['cell']['id']
        if key in result:raise ValueError('ambiguous collected phase attempts')
        result[key]=dict(attempt=run.name,phases=phases,files=pins,collection_sha256=sha(marker))
    return result


def pending(study,allowed):
    result=[]
    for key,record in allowed.items():
        if not re.fullmatch('[0-9a-f]{20}',key) or not re.fullmatch('attempt-[0-9]{3}',record['attempt']):
            raise ValueError('registered phase run identity required')
        run=study/'runs'/key/record['attempt']
        for name,pin in record['files'].items():
            if name not in ('RUN.json','RESULT.json','SNAPSHOT_BINDING.json') and name not in {
                phase+'/'+filename for phase in ('fit','pretraining')
                for filename in ('RUN.json','RESULT.json','SNAPSHOT_BINDING.json')}:
                raise ValueError('unexpected collected phase evidence path')
            if sha(run/name)!=pin:raise ValueError('worker phase evidence differs from collection')
        found=False
        for phase in record['phases']:
            if phase not in ('','fit','pretraining'):raise ValueError('unknown native snapshot phase')
            binding=run/phase/'SNAPSHOT_BINDING.json';bound=json.loads(binding.read_text())
            if bound['host']!=socket.gethostname():raise ValueError('phase state host differs')
            store=Path(bound['snapshot_directory'])
            if store.exists():found=True
            else:
                journal=binding.with_name('SNAPSHOT_RETIREMENT.json')
                previous=json.loads(journal.read_text()) if journal.exists() else {}
                if (previous.get('stage')!='completed_store_retired'
                    or previous.get('binding_sha256')!=sha(binding)
                    or previous.get('snapshot_directory')!=str(store)):
                    raise ValueError('phase store disappeared without retirement proof')
        if found:result.append(key)
    return dict(host=socket.gethostname(),pending_cells=result)


def watch(args):
    if not args.destination.is_relative_to(args.archive_mount):
        raise ValueError('durable destination must be within its read-only worker mount')
    args.output.mkdir(parents=True,exist_ok=False)
    status=dict(pid=os.getpid(),driver_sha256=sha(__file__),batches=[],scientific_training_launched=False)
    def save(stage,**fields):
        status.update(stage=stage,checked_utc=datetime.now(timezone.utc).isoformat(),**fields)
        (args.output/'STATUS.json').write_bytes(encoded(status))
    def execute(command,log):
        with log.open('a') as stream:
            stream.write(json.dumps(command)+'\n');stream.flush()
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            save(status['stage'],child_pid=child.pid,command=command)
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:save(status['stage'])
        save(status['stage'],child_pid=None)
        if code:raise RuntimeError('snapshot cleanup command failed; preserve '+str(log))
    def remote(host,command,log):
        payload=['env','TMPDIR='+str(Path.home()/'tmp'),*map(str,command)]
        execute(['ssh',host,shlex.join(payload)],log)
    heavy=Path.home()/'work/claudehints/scripts/run-heavy'
    helper=REPO/'scripts/rev4_featpot/shaping_snapshot_batch.py'
    namespace=REPO/'scripts/rev4_featpot/shaping_archive_namespace.sh'
    with (args.output/'OWNER.lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                save('watching_completed_collections')
                cycle_pending=0
                for host in (getattr(args,'worker_hosts',None) or HOSTS):
                    allowed=collected(args.study,host)
                    if not allowed:continue
                    command=['python3','-B',str(Path(__file__)),'probe','--study',str(args.study)]
                    probe=subprocess.run(['ssh',host,shlex.join(command)],input=json.dumps(allowed),
                        capture_output=True,text=True,check=True,timeout=30)
                    observed=json.loads(probe.stdout)
                    if observed['host']!=host:raise ValueError('resolved snapshot worker identity differs')
                    cells=observed['pending_cells'];cycle_pending+=len(cells)
                    if not cells:continue
                    index=len(status['batches'])+1;root=args.output/host/f'batch-{index:04}'
                    root.parent.mkdir(parents=True,exist_ok=True);log=root.with_name(root.name+'.log')
                    retirement=root.with_name(root.name+'-RETIRED.json');worker_trim=root.with_name(root.name+'-WORKER_TRIMMED.json')
                    request=root.with_name(root.name+'-REQUEST.json')
                    request.write_bytes(encoded(dict(host=host,cells=cells,collection_evidence={c:allowed[c] for c in cells})))
                    save('staging_completed_states',host=host,batch=str(root),required_cells=cells)
                    phases_only=all('' not in allowed[c]['phases'] for c in cells)
                    command=[heavy,'--mem','512M','--jobs','1','--','python3','-B',helper,'stage',
                        '--source',REPO,'--study',args.study,'--root',root]
                    if phases_only:command+=['--phases-only']
                    for cell in cells:command+=['--cell',cell]
                    remote(host,command,log)
                    execute(['rsync','-rt','--ignore-existing',host+':'+str(root)+'/',str(root)+'/'],log)
                    batch=json.loads((root/'BATCH.json').read_text())
                    if batch['phases_only']!=phases_only or batch['host']!=host or set(batch['selected_cells'])!=set(cells):
                        raise ValueError('staged phase batch scope differs')
                    save('archiving_completed_states')
                    mirror=args.destination/host/root.name
                    mirror.parent.mkdir(parents=True,exist_ok=True)
                    execute([str(heavy),'--mem','512M','--jobs','1','--','python3','-B',str(REPO/'scripts/byconstruction/dataset_mirror.py'),
                        '--root',str(root),'--nfs','--cold-readback','--artifact-kind','training','--destination',str(mirror)],log)
                    for name in ('MIRROR_VERIFIED.json','SHA256SUMS'):
                        execute(['scp',str(root/name),host+':'+str(root/name)],log)
                    save('retiring_verified_worker_states')
                    command=[heavy,'--mem','512M','--jobs','1','--','bash',namespace,args.archive_source,args.archive_mount,
                        'python3','-B',Path(__file__),'finish','--root',root,'--retirement',retirement,'--trim-receipt',worker_trim]
                    remote(host,command,log)
                    for path in (retirement,worker_trim):execute(['scp',host+':'+str(path),str(path)],log)
                    from shaping_snapshot_batch import trim
                    import shaping_snapshot_retention as owner
                    save('trimming_verified_coordinator_duplicates')
                    if host==socket.gethostname():
                        # Worker and coordinator are the same directory here;
                        # finish already trimmed it after durable readback.
                        proof=json.loads(worker_trim.read_text())
                        if (proof.get('stage')!='worker_global_and_staged_states_retired'
                            or proof.get('archive_root')!=str(root)
                            or proof.get('retirement_sha256')!=sha(retirement)):
                            raise ValueError('local completed-state trim proof differs')
                        result=dict(bytes=0)
                    else:
                        result=trim(root,retirement,root.with_name(root.name+'-COORDINATOR_TRIMMED.json'),owner)
                    status['batches'].append(dict(host=host,archive_root=str(root),durable_destination=str(mirror),
                        archive_receipt_sha256=sha(root/'MIRROR_VERIFIED.json'),stores=len(batch['rows']),
                        retired_stores=json.loads(retirement.read_text())['stores'],trimmed_local_bytes=result['bytes']))
                    save('watching_completed_collections',host=None,batch=None)
                save('watching_completed_collections',last_cycle_pending_cells=cycle_pending)
                if not args.watch:return status
                time.sleep(60)
        except BaseException as error:
            save('failed_with_preserved_evidence',error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('watch','probe','finish'))
    for name in ('study','output','destination','archive-mount','root','retirement','trim-receipt'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--worker-host',action='append',dest='worker_hosts',choices=HOSTS);p.add_argument('--archive-source');p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.mode=='probe':result=pending(a.study,json.load(sys.stdin))
    elif a.mode=='finish':
        import shaping_snapshot_batch as batch
        import shaping_snapshot_retention as owner
        result=batch.retire(a.root,a.retirement,owner,trim_staged=True)
        a.trim_receipt.write_bytes(encoded(dict(stage='worker_global_and_staged_states_retired',
            archive_root=str(a.root),retirement_sha256=sha(a.retirement))))
    else:result=watch(a)
    print(json.dumps(result),flush=True)
