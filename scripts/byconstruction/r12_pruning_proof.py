"""Strict native served f64-bit proof for registered R12 L0 fixtures.

Uses predict_features_with_bake --score-bits, the production BakeScorer owner;
no prediction or feature arithmetic is reproduced here. L0r is an explicit
approximation diagnostic, never part of the exact-pruning assertion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
import pyarrow.parquet as pq


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    a=ap.parse_args();r=a.root;out=r/'extra/proof';out.mkdir(parents=True,exist_ok=False)
    tool=r/'extra/bin/predict_features_with_bake'
    models={f'F{s}':Path.home()/f'tmp/e36test/r10/runs/mixF3-s{s}/pack_e059/refit/production-f16.bin' for s in range(3)}
    models.update({f'L0{s}':r/f'extra/bakes/L0_s{s}.bin' for s in range(3)})
    models['L0r']=r/'extra/bakes/L0r_s0.bin'
    fixtures={
        'EVAL':r/'reads/eval/features.parquet',
        'E36':Path('/mnt/v/zensim/e36-features/2026-11-01/bank/features.parquet'),
        'GCHROMA':Path('/var/tmp/byc-fchroma-20261029/c/merged-bank/features.parquet'),
    }
    instruments=Path('/var/tmp/byc-fx2-20261028/stage2/fx2-gate-instruments-run02')
    for grid in ('standard','ladder','negtail','identity'):fixtures[f'C-{grid}']=instruments/f'{grid}.parquet'
    for pop in ('4kdtb_eval','ladder_eval','hdrteach_val','upiq_dev'):fixtures[f'HDR-{pop}']=r/f'reads/hdr-panel/{pop}.parquet'
    fixtures['UPIQ-all']=r/'reads/hdr-upiq/upiq_all.parquet'
    result={}
    for name,path in fixtures.items():
        pf=pq.ParquetFile(path);schema=set(pf.schema_arrow.names)
        prefix='f' if 'f0' in schema else 'feat_'
        ids=sorted(int(c[len(prefix):]) for c in schema if c.startswith(prefix) and c[len(prefix):].isdigit())
        # The registered dial grids contain the full 720-column revision-7
        # raw read set. Other banks carry additional columns. Every R12
        # direct/product operand is <=719; preserve each fixture's real width.
        if ids!=list(range(len(ids))) or len(ids)<720:raise ValueError(f'{name}: complete contiguous native raw read set required')
        columns=[f'{prefix}{i}' for i in ids]
        # G-CHROMA must use the registered dev views >64px, never calibration.
        wanted=None;offset=0
        if name=='GCHROMA':
            keys=pq.read_table(path.with_name('keys.parquet'),columns=['mlic_role','width','height']).to_pydict()
            wanted=np.array([role=='dev' and max(w,h)>64 for role,w,h in zip(keys['mlic_role'],keys['width'],keys['height'])])
        record=dict(rows=0,differences=[0,0,0],L0r_differences=0,L0r_abs_sum=0.,L0r_max_abs=0.,
                    path=str(path),feature_metadata_rows=pf.metadata.num_rows)
        record['raw_feature_columns']=len(columns)
        for batch in pf.iter_batches(batch_size=10000,columns=columns):
            n=batch.num_rows
            matrix=np.column_stack([batch.column(i).to_numpy(zero_copy_only=False) for i in ids]).astype('<f8')
            if wanted is not None:matrix=matrix[wanted[offset:offset+n]]
            offset+=n
            if len(matrix)==0:continue  # explicit source-role mask, not a missing-input skip
            with tempfile.NamedTemporaryFile(dir=out,suffix='.wire') as f:
                f.write(struct.pack('<II',len(columns),len(matrix))+matrix.tobytes());f.flush()
                scores={}
                for label,bake in models.items():
                    p=subprocess.run([str(tool),'--bake',str(bake),'--features-file',f.name,
                        '--f64-wire','--production','--score-bits'],capture_output=True,text=True,check=True)
                    bits=np.fromstring(p.stdout,sep=' ',dtype=np.uint64)
                    if len(bits)!=len(matrix):raise ValueError('incomplete native bit output')
                    scores[label]=bits
            record['rows']+=len(matrix)
            for s in range(3):record['differences'][s]+=int(np.sum(scores[f'F{s}']!=scores[f'L0{s}']))
            delta=np.abs(scores['F0'].view(np.float64)-scores['L0r'].view(np.float64))
            if not np.isfinite(delta).all():raise ValueError('nonfinite approximation difference')
            record['L0r_differences']+=int(np.sum(scores['F0']!=scores['L0r']))
            record['L0r_abs_sum']+=float(delta.sum());record['L0r_max_abs']=max(record['L0r_max_abs'],float(delta.max()))
        record['L0r_mean_abs']=record.pop('L0r_abs_sum')/record['rows']
        result[name]=record
        (out/'FEATURE_PROOF.json').write_text(json.dumps(result,indent=2)+'\n')
        print(name,record,flush=True)
        if any(record['differences']):raise ValueError(f'{name}: L0 is not bit-identical')
    (out/'TOOL.json').write_text(json.dumps(dict(path=str(tool),sha256=hashlib.sha256(tool.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
