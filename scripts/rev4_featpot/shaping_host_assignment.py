"""Content-bound cell placement; never an exclusive compute lock."""
from __future__ import annotations
import argparse,json,socket
from pathlib import Path
from shaping_completion import encoded,sha


def check(root,cells,build_pin,host=None):
    host=host or socket.gethostname();selected={c['id'] for c in cells};seen=set()
    for path in (root/'worker_assignments').glob('*.json'):
        record=json.loads(path.read_text());ids=set(record['cells'])
        if seen&ids:raise ValueError('overlapping worker assignments')
        seen|=ids
        if not selected&ids:continue
        if record['plan_sha256']!=sha(root/'PLAN.json') or record['build_sha256']!=build_pin:
            raise ValueError('worker assignment scientific identity differs')
        if record['host']!=host:raise ValueError('cell assigned to another worker')


def main():
    from shaping_execution import load_registration
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--partition',required=True);p.add_argument('--host',required=True)
    p.add_argument('--family',choices=('joint_shapes','analytic'),default='joint_shapes')
    args=p.parse_args();_,cells,_=load_registration(args.study)
    families=(('product_joint_identity','separate_joint_identity') if args.family=='joint_shapes'
              else ('softplus','silu_monotone','silu_diagnostic'))
    selected=[c for c in cells if c['stage']=='inner' and c['partition']==args.partition
              and c['family'] in families]
    if len(selected)!=12*len(families):raise ValueError('complete registered family wave required')
    if any((args.study/'runs'/c['id']).exists() for c in selected):raise ValueError('existing attempts cannot be reassigned')
    build_pin=sha(args.binary_dir/'BUILD.json');check(args.study,selected,build_pin,args.host)
    directory=args.study/'worker_assignments';directory.mkdir(exist_ok=True)
    out=directory/(args.partition+'-'+args.family.replace('_','-')+'.json')
    record=dict(host=args.host,cells=[c['id'] for c in selected],plan_sha256=sha(args.study/'PLAN.json'),
                build_sha256=build_pin,partition=args.partition,scientific_inputs_unchanged=True)
    with out.open('xb') as stream:stream.write(encoded(record))
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
