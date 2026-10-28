#!/usr/bin/env python3
"""Replay source-built SDR teacher on the actual host's baked worker image."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time
from chroma_pilot import save

def replay(a):
    old='/var/tmp/byc-chroma-20261028';nas=Path('/mnt/tower/output/zensim/byc-chroma-20261028')
    rows=list(map(json.loads,(nas/'b/replay-pairs.jsonl').read_text().splitlines()))
    for r in rows:
        for k in ['reference','distorted']:r[k]=r[k].replace(old,str(nas),1);assert Path(r[k]).is_file(),r[k]
    root=Path(old)/'b'/('replay-'+a.host);root.mkdir()
    args=['docker','run','--pull=never','--name','chroma-replay-'+a.host,'--network','none','--cpu-shares=2048','--memory=4g','--memory-swap=4g','-i','-v',str(nas)+':'+str(nas)+':ro','--entrypoint','nice',a.image,'-n','19','ionice','-c','3','/opt/chroma/bin/chroma-frozen-teacher']
    start=time.monotonic()
    with (root/'scores.jsonl').open('w') as stdout,(root/'stderr.log').open('w') as stderr:subprocess.run(args,input=''.join(json.dumps(r)+'\n' for r in rows),text=True,stdout=stdout,stderr=stderr,check=True)
    actual=list(map(json.loads,(root/'scores.jsonl').read_text().splitlines()));assert len(actual)==len(rows)
    failures=[]
    for e,r in zip(rows,actual):
        assert (e['reference'],e['distorted'])==(r['reference'],r['distorted'])
        if struct.pack('>d',e['expected']).hex()!=r['value_bits'] or r['distorted_pixels_sha256']!=e['expected_dist_hash'] or (e['expected_ref_hash'] and r['reference_pixels_sha256']!=e['expected_ref_hash']):failures.append(dict(expected=e,actual=r))
    save(root/'PARITY.json',dict(passed=not failures,matched=len(rows)-len(failures),required=len(rows),host=a.host,image=a.image,seconds=time.monotonic()-start,failures=failures))
    assert not failures,'host teacher replay differs'
    print(a.host,len(rows),'exact replays',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('host');p.add_argument('image');replay(p.parse_args())
