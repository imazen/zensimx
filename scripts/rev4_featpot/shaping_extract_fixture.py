"""Freeze deterministic extraction fixtures before native qualification or timing."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from shaping_completion import encoded,sha


def pixel_hash(size,salt):
    digest=hashlib.sha256()
    # Bound temporary arrays independently of image dimensions. u64 wrap matches
    # the native instrument before the explicit modulus.
    for start in range(0,size*size,65536):
        i=np.arange(start,min(start+65536,size*size),dtype=np.uint64)
        k=((i+np.uint64(salt*7919))*np.uint64(2654435761))%np.uint64(65521)
        rgb=np.stack((k%256,(k//3)%256,(k//7)%256),axis=1).astype(np.uint8)
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def prepare(source,output):
    if output.exists():raise ValueError('preserve existing extraction fixture')
    w=json.loads(source.read_text())
    if w['schema']!='shaping-cost-v1':raise ValueError('native scoring workload required')
    ids=w['raw_ids']
    if not ids or len(ids)!=len(set(ids)) or any(type(v) is not int or not 0<=v<1853 for v in ids):
        raise ValueError('invalid revision-7 read set')
    output.mkdir(parents=True)
    files={}
    for size in (256,1024,4096):
        pixels=[[pixel_hash(size,2*i+1),pixel_hash(size,2*i+2)] for i in range(3)]
        for threads in (1,4):
            path=output/f'{size}-{threads}.json'
            path.write_bytes(encoded(dict(schema='shaping-extract-cost-v1',
                admission_sha256=w['admission_sha256'],raw_ids=ids,size=size,threads=threads,pixel_sha256=pixels)))
            files[path.name]=sha(path)
    (output/'REGISTRATION.json').write_bytes(encoded(dict(source=str(source),source_sha256=sha(source),
        generator_sha256=sha(__file__),file_sha256=files,
        scope='extraction instrument fixtures; finalist timing still requires full-FIT model admission',
        generator='For pixel i and salt s: k=((u64(i)+s*7919)*2654435761 mod 2^64) mod 65521; RGB=(k%256,(k//3)%256,(k//7)%256); pair j salts 2*j+1 and 2*j+2.',
        generated_before_native_qualification=True)))
    return dict(output=str(output),workloads=len(files))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.output)))
