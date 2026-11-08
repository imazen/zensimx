"""Collect terminal compatibility bakes and snapshots through existing owners.

Never launches training, alters a native attempt or removes a snapshot store.
"""
import argparse,fcntl,json,os,shlex,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import REPO,encoded,sha
from shaping_control_compatibility import FAMILIES
from shaping_control_evidence import verify
from shaping_storage import reservation


def remote(host,command,**kwargs):
    return subprocess.run(['ssh',host,shlex.join(list(map(str,command)))],check=True,**kwargs)


def collect(args,family):
    base=args.study.parent/'corrected-control-qualification-20261108'
    run=base/family;destination=args.output/family;inventory=destination/'INVENTORY.json'
    # Absence is a running qualification, never a successful or failed fit.
    probe=remote(args.host,['python3','-c',
        'import pathlib,sys; p=pathlib.Path(sys.argv[1]); print(p.read_text() if p.exists() else "null")',
        run/'RESULT.json'],capture_output=True,text=True,timeout=30)
    result=json.loads(probe.stdout)
    if result is None:return dict(stage='waiting_for_terminal_qualification')
    if result.get('stage')!='qualified' or not result.get('all_240_checkpoints_equal'):
        raise ValueError('native compatibility did not qualify: '+family)
    stage=base/'evidence-inventories'/family
    heavy=Path.home()/'work/claudehints/scripts/run-heavy'
    owner=args.source/'scripts/rev4_featpot'
    log=args.output/(family+'.log')
    with log.open('a') as stream:
        def call(command):
            stream.write(json.dumps(list(map(str,command)))+'\n');stream.flush()
            return subprocess.run(list(map(str,command)),stdout=stream,stderr=subprocess.STDOUT,check=True)
        # A preserved staging directory is reusable only through its verified
        # immutable inventory. Never overwrite an existing inventory or attempt.
        exists=remote(args.host,['test','-f',stage/'INVENTORY.json'],stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL) if inventory.exists() else None
        if exists is None:
            command=[heavy,'--mem','512M','--jobs','1','--','python3','-B',owner/'shaping_control_evidence.py',
                'prepare','--registration',base/'REGISTRATION.json','--family',family,'--run',run,'--output',stage]
            remote(args.host,command,stdout=stream,stderr=subprocess.STDOUT)
        with reservation(args.study,destination,30*1024**3):
            destination.mkdir(parents=True,exist_ok=True)
            if not inventory.exists():
                call(['scp',args.host+':'+str(stage/'INVENTORY.json'),str(inventory)])
            record=json.loads(inventory.read_text())
            if record['family']!=family or record['source_run']!=str(run):raise ValueError('control inventory identity differs')
            paths=destination/'FILES.txt';paths.write_text(''.join(name+'\n' for name in record['files']))
            for name in record['files']:
                p=Path(name)
                if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe native evidence path')
            call(['rsync','-rt','--ignore-existing','--files-from='+str(paths),args.host+':'+str(run)+'/',str(destination)+'/'])
            verified=verify(inventory,destination)
            if sha(destination/'RESULT.json')!=__import__('hashlib').sha256(encoded(result)).hexdigest():
                raise ValueError('terminal result changed during collection')
            # Bound states using the native store owner, including its native
            # flock and completed-run identity checks. Store retirement is separate.
            snapshots=destination/'snapshots';remote_snapshots=stage/'snapshots'
            if not snapshots.exists():
                remote(args.host,[heavy,'--mem','512M','--jobs','1','--','python3','-B',owner/'shaping_snapshot_retention.py',
                    'stage','--binding',run/'SNAPSHOT_BINDING.json','--output',remote_snapshots],stdout=stream,stderr=subprocess.STDOUT)
                snapshots.mkdir()
            call(['rsync','-rt','--ignore-existing',args.host+':'+str(remote_snapshots)+'/',str(snapshots)+'/'])
            manifest=json.loads((snapshots/'SNAPSHOTS.json').read_text())
            if manifest['schema']!='completed-native-snapshot-stage-v1':raise ValueError('snapshot stage differs')
            for name,pin in manifest['files'].items():
                if Path(name).name!=name or sha(snapshots/name)!=pin:raise ValueError('snapshot collection drift')
            mirror=args.destination/family
            receipt=destination/'MIRROR_VERIFIED.json'
            if not receipt.exists():
                call([heavy,'--mem','512M','--jobs','1','--','python3','-B',REPO/'scripts/byconstruction/dataset_mirror.py',
                    '--root',destination,'--nfs','--cold-readback','--artifact-kind','training','--destination',mirror])
            archived=json.loads(receipt.read_text())
            if (archived['destination']!=str(mirror) or not archived['all_artifact_sha256_verified']
                or sha(destination/'SHA256SUMS')!=sha(mirror/'SHA256SUMS')):
                raise ValueError('durable compatibility evidence differs')
    return dict(stage='archived',result=str(destination/'RESULT.json'),result_sha256=sha(destination/'RESULT.json'),
        native=verified,durable_destination=str(mirror),archive_receipt_sha256=sha(receipt),snapshots_preserved=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','output','destination'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--host',choices=('r3500','r3800x','i134','r5600g'),required=True)
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'ARCHIVE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results={}
        while True:
            for family in FAMILIES:
                if family=='silu_monotone':continue  # Its separate qualified archive is already complete.
                if results.get(family,{}).get('stage')=='archived':continue
                results[family]=collect(a,family)
            complete=all(v['stage']=='archived' for v in results.values())
            status=dict(pid=os.getpid(),stage='archived' if complete else 'watching',
                checked_utc=datetime.now(timezone.utc).isoformat(),families=results,driver_sha256=sha(__file__))
            temp=a.output/'STATUS.writing';temp.write_bytes(encoded(status));temp.replace(a.output/'STATUS.json')
            if complete or not a.watch:print(json.dumps(status),flush=True);return
            time.sleep(60)


if __name__=='__main__':main()
