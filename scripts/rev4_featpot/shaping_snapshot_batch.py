"""Batch the existing completed-snapshot archive and retirement gates."""
import argparse,json,socket,sys
from contextlib import ExitStack
from datetime import datetime,timezone
from pathlib import Path


def save(path,value):
    temporary=path.with_suffix('.writing');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def stage(study,output,owner,storage,*,phases_only=False,cells=None):
    if output.exists():raise ValueError('preserve the existing snapshot batch')
    output.mkdir(parents=True);rows=[];deferred=[];nonlocal_count=unfinished=0;total=0
    with ExitStack() as stack:
        for index in range(4):stack.enter_context(storage.reservation(study,output/('reservation-'+str(index)),30*1024**3))
        bindings=list((study/'runs').glob('*/attempt-*/SNAPSHOT_BINDING.json')) if not phases_only else []
        bindings+=list((study/'runs').glob('*/attempt-*/*/SNAPSHOT_BINDING.json'))
        for binding in sorted(bindings):
            bound=json.loads(binding.read_text())
            identity=json.loads(binding.with_name('RUN.json').read_text())
            if phases_only and not identity.get('snapshot_phase'):continue
            if cells is not None and identity['cell']['id'] not in cells:continue
            if bound['host']!=socket.gethostname():nonlocal_count+=1;continue
            result=binding.with_name('RESULT.json')
            if not result.exists():unfinished+=1;continue
            terminal=json.loads(result.read_text())
            if terminal.get('stage')!='trained' or terminal.get('exit_code')!=0:unfinished+=1;continue
            directory=Path(bound['snapshot_directory'])
            if not directory.exists():
                previous=binding.with_name('SNAPSHOT_RETIREMENT.json')
                if not previous.exists() or json.loads(previous.read_text()).get('stage')!='completed_store_retired':
                    raise ValueError('completed store is missing without its retirement journal')
                continue
            destination=output/directory.name
            if destination.exists():raise ValueError('two bindings refer to the same native snapshot store')
            try:manifest=owner.stage(binding,destination)
            except BlockingIOError:deferred.append(str(binding));continue
            size=sum(p.stat().st_size for p in destination.iterdir() if p.is_file());total+=size
            if total>4*1024**3:raise ValueError('snapshot archive batch exceeds its four-GiB reservation')
            rows.append(dict(binding=str(binding),binding_sha256=owner.sha(binding),stage=str(destination),
                             snapshot_directory=str(directory),bytes=size))
        if not rows:raise ValueError('no completed native snapshot stores were staged')
        value=dict(schema='completed-snapshot-batch-v1',stage='complete',host=socket.gethostname(),
                   study=str(study),rows=rows,deferred=deferred,skipped_nonlocal=nonlocal_count,
                   skipped_unfinished_or_failed=unfinished,bytes=total,helper_sha256=owner.sha(owner.__file__),
                   checked_utc=datetime.now(timezone.utc).isoformat(),scientific_training_launched=False,
                   phases_only=phases_only,selected_cells=sorted(cells) if cells is not None else None)
        save(output/'BATCH.json',value)
    return dict(stage='complete',stores=len(rows),bytes=total,deferred=len(deferred))


def retire(root,receipt,owner):
    if receipt.exists() or receipt.is_relative_to(root):raise ValueError('fresh retirement receipt outside the immutable archive required')
    batch=json.loads((root/'BATCH.json').read_text())
    if (batch['schema']!='completed-snapshot-batch-v1' or batch['stage']!='complete'
        or batch['host']!=socket.gethostname() or batch['helper_sha256']!=owner.sha(owner.__file__)):
        raise ValueError('snapshot batch host or helper identity differs')
    retired=[]
    for row in batch['rows']:
        binding=Path(row['binding']);stage_root=Path(row['stage']);journal=binding.with_name('SNAPSHOT_RETIREMENT.json')
        if owner.sha(binding)!=row['binding_sha256']:raise ValueError('snapshot binding changed after archival')
        if journal.exists():
            old=json.loads(journal.read_text())
            if (old['stage']!='completed_store_retired' or old['binding_sha256']!=row['binding_sha256']
                or old['archive_root']!=str(root) or old['archive_manifest_sha256']!=owner.sha(root/'SHA256SUMS')
                or Path(row['snapshot_directory']).exists()):raise ValueError('partial snapshot retirement requires explicit reconciliation')
        else:owner.retire(binding,stage_root,journal,root)
        retired.append(dict(binding=str(binding),journal=str(journal),journal_sha256=owner.sha(journal)))
    save(receipt,dict(schema='completed-snapshot-batch-retirement-v1',stage='complete',archive_root=str(root),
         archive_manifest_sha256=owner.sha(root/'SHA256SUMS'),retired=retired,stores=len(retired),
         checked_utc=datetime.now(timezone.utc).isoformat(),scientific_training_launched=False))
    return dict(stage='complete',retired_stores=len(retired))


def trim(root,retirement,receipt,owner):
    """Remove only cold-verified staged state duplicates after store retirement."""
    from shaping_checkpoint_retention import archive_inventory,no_users
    import os,hashlib,re
    if root.resolve()!=root or receipt.exists() or receipt.is_relative_to(root):
        raise ValueError('canonical stage and fresh external trim receipt required')
    batch=json.loads((root/'BATCH.json').read_text());retired=json.loads(retirement.read_text())
    if (batch['schema']!='completed-snapshot-batch-v1' or batch['stage']!='complete'
        or retired['schema']!='completed-snapshot-batch-retirement-v1' or retired['stage']!='complete'
        or retired['archive_root']!=str(root) or retired['archive_manifest_sha256']!=owner.sha(root/'SHA256SUMS')
        or retired['stores']!=len(batch['rows'])):
        raise ValueError('completed store retirement and exact archive required')
    mirror,pins=archive_inventory(root);rows=[]
    for row in batch['rows']:
        directory=Path(row['stage'])
        if directory.parent!=root or not re.fullmatch('[0-9a-f]{64}',directory.name):
            raise ValueError('bounded snapshot stage directory required')
        manifest=json.loads((directory/'SNAPSHOTS.json').read_text())
        for name in sorted(owner.STATES & set(manifest['files'])):
            path=directory/name;owner.regular(path);relative=path.relative_to(root).as_posix()
            pin=manifest['files'][name]
            if pins.get(relative)!=pin or owner.sha(path)!=pin:raise ValueError('staged snapshot drift')
            remote=mirror/relative;owner.regular(remote)
            with remote.open('rb') as stream:
                os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                if hashlib.file_digest(stream,'sha256').hexdigest()!=pin:
                    raise ValueError('staged state cold readback differs')
            rows.append(dict(path=str(path),durable_path=str(remote),sha256=pin,bytes=path.stat().st_size))
    if not rows:raise ValueError('no staged snapshot payload to trim')
    no_users(rows)
    value=dict(schema='completed-snapshot-stage-trim-v1',stage='verified_before_removal',rows=rows,
        archive_root=str(root),archive_manifest_sha256=owner.sha(root/'SHA256SUMS'),
        retirement_sha256=owner.sha(retirement),bytes=sum(r['bytes'] for r in rows),
        checked_utc=datetime.now(timezone.utc).isoformat())
    with receipt.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    for row in rows:Path(row['path']).unlink()
    value['stage']='archived_stage_payload_trimmed';save(receipt,value)
    return dict(stage=value['stage'],files=len(rows),bytes=value['bytes'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('stage','retire','trim'))
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--study',type=Path)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--receipt',type=Path)
    parser.add_argument('--retirement',type=Path)
    parser.add_argument('--phases-only',action='store_true');parser.add_argument('--cell',action='append')
    args=parser.parse_args();sys.path.insert(0,str(args.source/'scripts/rev4_featpot'))
    import shaping_snapshot_retention as owner
    import shaping_storage as storage
    if args.mode=='stage':value=stage(args.study,args.root,owner,storage,phases_only=args.phases_only,cells=set(args.cell) if args.cell is not None else None)
    elif args.mode=='retire':value=retire(args.root,args.receipt,owner)
    else:value=trim(args.root,args.retirement,args.receipt,owner)
    print(json.dumps(value),flush=True)
