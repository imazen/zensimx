"""Compare frozen A0/A1/A2 served f64 bits across old and R14 native scorers.

Only the two registered TRAIN audit banks are read. Prediction is delegated to
the existing native predict_features_with_bake --production --score-bits owner.
"""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
import pyarrow.parquet as pq

import r14_transforms as run


def main():
    root=run.O
    out=root/'native-noop'
    out.mkdir(exist_ok=False)
    old=Path('/var/tmp/byc-linear-20261104/extra/bin/predict_features_with_bake')
    old_pin=json.loads(Path('/var/tmp/byc-linear-20261104/extra/proof/TOOL.json').read_text())
    new=root/'bin/predict_features_with_bake'
    pins=json.loads((root/'SOURCE_PINS.json').read_text())
    if run.owner.sha(old)!=old_pin['sha256'] or run.owner.sha(new)!=pins['binary_sha256'][new.name]:
        raise ValueError('no-op scorer identity changed')
    models={f'A{s}':run.owner.H/f'tmp/e36test/r10/runs/mixF3-s{s}/pack_e059/refit/production-f16.bin' for s in range(3)}
    report=dict(source_commit=pins['source_commit'],old_tool=old_pin,
        new_tool=dict(path=str(new),sha256=run.owner.sha(new)),
        models={n:dict(path=str(p),sha256=run.owner.sha(p)) for n,p in models.items()},banks={})
    for name,(path,digest) in run.owner.BANKS.items():
        if run.owner.sha(path)!=digest:raise ValueError('no-op TRAIN bank changed')
        pf=pq.ParquetFile(path)
        schema=set(pf.schema_arrow.names)
        prefix='f' if 'f0' in schema else 'feat_'
        ids=sorted(int(c[len(prefix):]) for c in schema if c.startswith(prefix) and c[len(prefix):].isdigit())
        if ids!=list(range(len(ids))) or len(ids)<720:raise ValueError('incomplete raw caller bank')
        rows=0;differences={n:0 for n in models}
        hashes={n:[hashlib.sha256(),hashlib.sha256()] for n in models}
        for batch in pf.iter_batches(batch_size=2048,columns=[f'{prefix}{i}' for i in ids]):
            matrix=np.column_stack([batch.column(i).to_numpy(zero_copy_only=False) for i in ids]).astype('<f8')
            with tempfile.NamedTemporaryFile(dir=out,suffix='.wire') as f:
                f.write(struct.pack('<II',len(ids),len(matrix))+matrix.tobytes());f.flush()
                for name_model,bake in models.items():
                    scores=[]
                    for index,tool in enumerate((old,new)):
                        proc=subprocess.run([str(tool),'--bake',str(bake),'--features-file',f.name,
                            '--f64-wire','--production','--score-bits'],capture_output=True,text=True,check=True)
                        bits=np.fromstring(proc.stdout,sep=' ',dtype='<u8')
                        if len(bits)!=len(matrix) or not np.isfinite(bits.view('<f8')).all():
                            raise ValueError('incomplete or nonfinite native no-op prediction')
                        hashes[name_model][index].update(bits.tobytes());scores.append(bits)
                    differences[name_model]+=int(np.count_nonzero(scores[0]!=scores[1]))
            rows+=len(matrix)
        if rows!=pf.metadata.num_rows:raise ValueError('incomplete no-op bank')
        report['banks'][name]=dict(path=str(path),sha256=digest,rows=rows,raw_columns=len(ids),
            differences=differences,score_bit_sha256={n:[h.hexdigest() for h in hs] for n,hs in hashes.items()})
        run.owner.write(out/'PROOF.json',report)
        print(json.dumps(report['banks'][name]),flush=True)
        if any(differences.values()):raise ValueError('A scores moved; stop R14')


if __name__=='__main__':main()
