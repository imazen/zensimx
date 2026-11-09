"""Retain needed local epochs; reread durable checkpoints before removing duplicates."""
from __future__ import annotations
import argparse,json,os,re,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_execution import load_registration
from shaping_evaluation import diagnostic_epochs
from shaping_archive_supervision import eligible,rescue_eligible


from shaping_artifact_retention import archive_inventory,open_targets,no_users

DRIVER_SHA256=sha(__file__)


def retained_epochs(epochs):
    return {0,epochs-1,*diagnostic_epochs(epochs),*( [119] if epochs>=120 else [] )}


def rescue_phases(run):
    identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
    if identity['cell']['stage']!='rescue':raise ValueError('rescue phase retention requires a rescue run')
    pretrained='_pretrained_' in identity['cell']['family']
    layout=[('pretraining',120),('fit',120)] if pretrained else [('fit',240)]
    if result['stage']!='trained' or [p['epochs'] for p in result['phases']]!=[n for _,n in layout]:
        raise ValueError('complete registered rescue phases required')
    for phase,epochs in layout:
        recorded=json.loads((run/phase/'RUN.json').read_text())
        if recorded['epochs']!=epochs or recorded['pairs_per_epoch']!=50000:
            raise ValueError('rescue phase exposure differs')
    return layout




def candidates(root,rescue=False):
    _,cells,_=load_registration(root);rows=[];plan_pin=sha(root/'PLAN.json')
    input_pins=json.loads((root/'PLAN.json').read_text())['input_pins']
    for receipt in sorted(root.glob('runs/*/MIRROR_VERIFIED.json')):
        directory=receipt.parent
        if (directory/'LOCAL_CHECKPOINT_RETENTION.json').exists():continue
        if not (rescue_eligible(root,directory,cells) if rescue else eligible(directory,cells,plan_pin)):continue
        mirror,hashes=archive_inventory(directory)
        run=directory/'attempt-001';record=json.loads((run/'RUN.json').read_text())
        phases=rescue_phases(run) if rescue else [(None,record['epochs'])]
        for phase,epochs in phases:
            base=run/phase if phase else run
            paths=list((base/'checkpoints').glob('ckpt_epoch*.bin'))
            expected={f'ckpt_epoch{epoch:03}.bin' for epoch in range(epochs)}
            if {p.name for p in paths}!=expected:raise ValueError('unaccounted partial local checkpoint inventory')
            keep=retained_epochs(epochs)
            for path in paths:
                epoch=int(re.fullmatch(r'ckpt_epoch(\d+)\.bin',path.name)[1])
                if epoch in keep or path.stat().st_nlink!=1:continue
                relative=path.relative_to(directory).as_posix()
                if str(path) in input_pins:raise ValueError('active study input selected for removal')
                row=dict(path=str(path),durable_path=str(mirror/relative),sha256=hashes[relative],
                         cell=directory.name,epoch=epoch,retained_epochs=sorted(keep),
                         bytes=path.stat().st_size,allocated_bytes=path.stat().st_blocks*512)
                if phase:row['phase']=phase
                rows.append(row)
    return rows






def reclaim(root,rows,output):
    if output.exists():raise ValueError('preserve existing retention receipt')
    if not rows:raise ValueError('no new archived checkpoint duplicates')
    if root.stat().st_dev!=output.parent.stat().st_dev:raise ValueError('receipt and study must share the experiment filesystem')
    names=set();rescue_layouts={}
    phase_cells={row['cell'] for row in rows if row.get('phase') is not None}
    if phase_cells:
        _,cells,_=load_registration(root)
        for cell in phase_cells:
            directory=root/'runs'/cell
            if not rescue_eligible(root,directory,cells):raise ValueError('rescue retention candidate is not qualified')
            mirror,hashes=archive_inventory(directory)
            rescue_layouts[cell]=(dict(rescue_phases(directory/'attempt-001')),mirror,hashes)
    for row in rows:
        local=Path(row['path']);remote=Path(row['durable_path'])
        run=root/'runs'/row['cell']/'attempt-001';phase=row.get('phase')
        if phase is not None:
            layout,mirror,hashes=rescue_layouts[row['cell']]
            if phase not in layout:raise ValueError('unregistered rescue checkpoint phase')
            expected=run/phase/'checkpoints'/f"ckpt_epoch{row['epoch']:03}.bin";epochs=layout[phase]
            relative=expected.relative_to(run.parent).as_posix()
            if remote!=mirror/relative or row['sha256']!=hashes.get(relative):raise ValueError('rescue checkpoint archive binding differs')
        else:
            expected=run/'checkpoints'/f"ckpt_epoch{row['epoch']:03}.bin"
            epochs=json.loads((run/'RUN.json').read_text())['epochs']
        if local!=expected or local.resolve()!=local or remote.resolve()==local.resolve():
            raise ValueError('checkpoint path outside exact local retention scope')
        if row['epoch'] not in range(epochs) or row['epoch'] in retained_epochs(epochs):
            raise ValueError('required checkpoint selected for removal')
        if str(local) in names:raise ValueError('duplicate checkpoint removal')
        names.add(str(local))
    no_users(rows);identities={}
    for i,row in enumerate(rows):
        local=Path(row['path']);remote=Path(row['durable_path'])
        if local.is_symlink() or remote.is_symlink():raise ValueError('checkpoint symlink refused')
        if sha(local)!=row['sha256']:raise ValueError('local checkpoint changed')
        with remote.open('rb') as stream:
            os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            actual=__import__('hashlib').file_digest(stream,'sha256').hexdigest()
        if actual!=row['sha256']:raise ValueError('durable checkpoint readback differs')
        stat=local.stat();identities[str(local)]=(stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if i%256==0:print(json.dumps(dict(verified=i+1,required=len(rows))),flush=True)
    no_users(rows)
    for name,identity in identities.items():
        stat=Path(name).stat()
        if (stat.st_ino,stat.st_size,stat.st_mtime_ns)!=identity:raise ValueError('checkpoint changed during verification')
    record=dict(stage='verified_before_removal',checked_utc=datetime.now(timezone.utc).isoformat(),
        rows=rows,driver_sha256=DRIVER_SHA256,readback='all removed checkpoints: local SHA-256 and durable SHA-256 after POSIX_FADV_DONTNEED',
        filesystem_free_before_bytes=shutil.disk_usage(root).free)
    with output.open('xb') as stream:stream.write(encoded(record))
    for row in rows:Path(row['path']).unlink()
    record.update(stage='local_duplicates_removed',filesystem_free_after_bytes=shutil.disk_usage(root).free)
    output.write_bytes(encoded(record))
    for cell in sorted({row['cell'] for row in rows}):
        removed=[r for r in rows if r['cell']==cell]
        note=dict(partial_local_checkpoint_copy=True,retention_receipt=str(output),receipt_sha256=sha(output),
                  removed_epochs=sorted({r['epoch'] for r in removed}),retained_epochs=removed[0]['retained_epochs'],
                  durable_checkpoints_preserved=True)
        if any('phase' in row for row in removed):
            note['phase_retention']={phase:dict(removed_epochs=sorted(r['epoch'] for r in removed if r.get('phase')==phase),
                retained_epochs=next(r['retained_epochs'] for r in removed if r.get('phase')==phase))
                for phase in sorted({r['phase'] for r in removed})}
        with (root/'runs'/cell/'LOCAL_CHECKPOINT_RETENTION.json').open('xb') as stream:stream.write(encoded(note))
    print(json.dumps(dict(cells=len({r['cell'] for r in rows}),removed_files=len(rows),
        removed_logical_bytes=sum(r['bytes'] for r in rows),filesystem_free_before_bytes=record['filesystem_free_before_bytes'],
        filesystem_free_after_bytes=record['filesystem_free_after_bytes'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--apply',action='store_true')
    p.add_argument('--rescue',action='store_true',help='qualified completed rescue phases only; preserves failed originals')
    a=p.parse_args();rows=candidates(a.study,rescue=a.rescue)
    if a.apply:reclaim(a.study,rows,a.output)
    else:
        with a.output.open('xb') as stream:stream.write(encoded(dict(rows=rows,files=len(rows),logical_bytes=sum(r['bytes'] for r in rows),removed=False)))
