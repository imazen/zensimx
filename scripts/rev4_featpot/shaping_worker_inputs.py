"""Prepare a registered TRAIN partition and package its immutable worker inputs."""
from __future__ import annotations
import argparse,json,tarfile
from pathlib import Path
from shaping_completion import sha,encoded
from shaping_execution import load_registration,prepare_partition


def prepare(study,assignment_path,output):
    plan,cells,partitions=load_registration(study)
    assignment=json.loads(assignment_path.read_text())
    if assignment['plan_sha256']!=sha(study/'PLAN.json'):raise ValueError('assignment drift')
    selected=[c for c in cells if c['id'] in assignment['cells']]
    if len(selected)!=len(assignment['cells']) or any(c['partition']!=assignment['partition'] for c in selected):
        raise ValueError('assignment is not one registered partition')
    partition=assignment['partition']
    record=prepare_partition(study,partition,partitions[partition],plan)
    receipt=study/'comparators'/partition/'GROUPS.json'
    files={receipt,assignment_path,*map(Path,record['file_sha256'])}
    files.update(p for p in receipt.parent.rglob('*') if p.is_file())
    shapes=study/'shapes'/partition
    if any(c['family'].endswith('joint_identity') for c in selected):
        from shaping_inputs import prepare as prepare_shapes
        for separate in (False,True):prepare_shapes(study,partition,partitions[partition],plan,separate)
        files.update(p for p in shapes.rglob('*') if p.is_file())
    rows=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    output.mkdir(parents=True,exist_ok=False)
    (output/'INPUTS.json').write_bytes(encoded(rows))
    bundle=output/'inputs.tar'
    with tarfile.open(bundle,'x',dereference=True) as archive:
        for row in rows:archive.add(row['path'],arcname=row['path'].lstrip('/'),recursive=False)
    for row in rows:
        if sha(row['path'])!=row['sha256']:raise ValueError('input changed during packaging')
    result=dict(files=len(rows),bytes=sum(x['bytes'] for x in rows),archive_sha256=sha(bundle),
                assignment_sha256=sha(assignment_path),manifest_sha256=sha(output/'INPUTS.json'))
    (output/'BUNDLE.json').write_bytes(encoded(result));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','assignment','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.study,a.assignment,a.output)
