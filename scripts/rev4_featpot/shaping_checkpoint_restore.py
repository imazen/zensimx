"""Restore a completed control's missing epochs from its verified native archive."""
import argparse,hashlib,json,os
from pathlib import Path
from shaping_completion import encoded,sha

ALLOWANCE=128*1024**2


def restore(study,cell,output):
    from shaping_execution import load_registration
    from shaping_checkpoint_retention import archive_inventory
    from shaping_storage import reservation
    plan,cells,_=load_registration(study);directory=study/'runs'/cell;run=directory/'attempt-001'
    identity=json.loads((run/'RUN.json').read_text());terminal=json.loads((run/'RESULT.json').read_text())
    if (identity['cell'] not in cells or identity['cell']['id']!=cell or identity['cell']['stage']!='inner'
        or identity['epochs']!=240 or terminal['stage']!='trained' or terminal['exit_code']!=0
        or identity['plan_sha256']!=sha(study/'PLAN.json')):
        raise ValueError('completed registered 240-epoch control required')
    mirror,inventory=archive_inventory(directory)
    paths={f'attempt-001/checkpoints/ckpt_epoch{e:03}.bin':inventory[f'attempt-001/checkpoints/ckpt_epoch{e:03}.bin'] for e in range(240)}
    # The native archive also contains reports; reserve only the named missing
    # checkpoint payload. The streaming guard below still bounds actual writes.
    needed=sum((mirror/name).stat().st_size for name in paths if not (directory/name).exists())
    if needed>ALLOWANCE:raise ValueError('restore exceeds its finite allowance')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(cell=identity['cell'],archive=str(mirror),archive_manifest_sha256=sha(directory/'SHA256SUMS'),
                restored={},already_present=[],copied_bytes=0,outer_responses_read=False)
    with reservation(study,output/'admission',plan['minimum_ssd_free_bytes'],bytes_required=ALLOWANCE):
        for name,pin in paths.items():
            target=directory/name
            if any(p.is_symlink() for p in [target,*target.parents]):raise ValueError('symlink in restoration target')
            if target.exists():
                if sha(target)!=pin:raise ValueError('existing original checkpoint differs')
                result['already_present'].append(name);continue
            temporary=target.with_suffix('.bin.restoring');digest=hashlib.sha256();size=0
            with (mirror/name).open('rb') as source,temporary.open('xb') as dest:
                while chunk:=source.read(1024*1024):
                    size+=len(chunk);result['copied_bytes']+=len(chunk)
                    if size>16*1024**2 or result['copied_bytes']>ALLOWANCE:raise ValueError('restore exceeds its finite allowance')
                    dest.write(chunk);digest.update(chunk)
                dest.flush();os.fsync(dest.fileno())
            if digest.hexdigest()!=pin:raise ValueError('archived checkpoint differs; preserve partial restoration')
            temporary.rename(target);result['restored'][name]=dict(sha256=pin,bytes=size)
        if any(sha(directory/name)!=pin for name,pin in paths.items()):raise ValueError('restored original inventory differs')
    result['all_240_original_checkpoints_verified']=True
    (output/'RESTORED.json').write_bytes(encoded(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--cell',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=restore(a.study,a.cell,a.output)
    print(json.dumps(dict(restored=len(result['restored']),already_present=len(result['already_present']),copied_bytes=result['copied_bytes'])),flush=True)
