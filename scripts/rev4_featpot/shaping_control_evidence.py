"""Inventory finalized compatibility evidence without copying canonical scratch."""
import argparse,json
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_corrected_view import trajectory


def prepare(registration,family,run,output):
    record=json.loads(registration.read_text());case=record['cases'][family]
    result=json.loads((run/'RESULT.json').read_text())
    if (result['registration_sha256']!=sha(registration)
        or result['source_result_sha256']!=sha(Path(case['run'])/'RESULT.json')):
        raise ValueError('control evidence registration or source result differs')
    trajectory(result,family,case)
    expected={f'ckpt_epoch{epoch:03}.bin' for epoch in range(240)}
    if {p.name for p in (run/'checkpoints').glob('ckpt_epoch*.bin')}!=expected:
        raise ValueError('complete native control bakes required')
    required={'RUN.json','RESULT.json','TRAINING_RESULT.json','train.log','best.bin','best.bin.spec.json'}
    optional={'SNAPSHOT_BINDING.json','HEAD.json','SHAPING.json'}
    names=required|{name for name in optional if (run/name).exists()}|{'checkpoints/'+name for name in expected}
    rows={}
    for name in sorted(names):
        path=run/name
        if path.is_symlink() or not path.is_file() or path.resolve()!=path:raise ValueError('regular finalized evidence required')
        pin=sha(path)
        if name.startswith('checkpoints/') and pin!=result['trajectory']['compared'][path.name]['input_sha256']['neutral']:
            raise ValueError('finalized native checkpoint changed after comparison')
        rows[name]=dict(bytes=path.stat().st_size,sha256=pin)
    if sum(row['bytes'] for row in rows.values())>384*1024**2:raise ValueError('control evidence exceeds bounded copy allowance')
    output.mkdir(parents=True,exist_ok=False)
    inventory=dict(schema='native-compatibility-evidence-inventory-v1',family=family,source_run=str(run),
        registration_path=str(registration),registration_sha256=sha(registration),files=rows,
        source_control=case,canonical_derivatives_copied=False,original_source_bakes_copied=False,
        shared_training_inputs_copied=False,driver_sha256=sha(__file__))
    (output/'INVENTORY.json').write_bytes(encoded(inventory))
    (output/'FILES.txt').write_text(''.join(name+'\n' for name in rows))
    return inventory


def verify(inventory,directory):
    record=json.loads(inventory.read_text())
    if record['schema']!='native-compatibility-evidence-inventory-v1':raise ValueError('control evidence inventory differs')
    for name,row in record['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe control evidence path')
        path=directory/relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('collected control evidence drift: '+name)
    return dict(files=len(record['files']),bytes=sum(row['bytes'] for row in record['files'].values()),all_files_verified=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','verify'))
    for name in ('registration','run','output','inventory','directory'):p.add_argument('--'+name,type=Path)
    p.add_argument('--family');a=p.parse_args()
    result=prepare(a.registration,a.family,a.run,a.output) if a.mode=='prepare' else verify(a.inventory,a.directory)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','source_control')}),flush=True)
