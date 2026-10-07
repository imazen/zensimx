#!/usr/bin/env python3
"""Pinned registered pixel identity and CHROMAQ diagnostic, no new encodes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/v/output/zensim/prodqual-b-2026-10-07')
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PINS = json.loads((OUT/'PROGRAM_MODEL_PINS.json').read_text())
SCORER = OUT/'bin/serve_custom_bake'


def scores(pairs, label, dest):
    env = dict(os.environ, ZENSIM_FORMULA_REV='5' if label=='seed0' else '1',
               RAYON_NUM_THREADS='1',TMPDIR=str(Path.home()/'tmp/prodqual-b'))
    cmd = [str(SCORER),'--pairs',str(pairs),str(OUT/'models'/f'{label}.bin')]
    Path(str(dest)+'.command.json').write_text(json.dumps({'argv':cmd,'formula_revision':env['ZENSIM_FORMULA_REV']})+'\n')
    with dest.open('w') as out, Path(str(dest)+'.log').open('w') as log:
        subprocess.run(cmd,env=env,stdout=out,stderr=log,check=True)


def identity():
    f=json.loads((ROOT/'benchmarks/prodqual_b_registration_2026-10-07.json').read_text())
    record=next(r for r in f['inputs'] if r['name']=='identity')
    keys=Path(record['path'].replace('.parquet','.keys.parquet'))
    assert SHA(keys)==record['keys_sha256']
    rows=pq.read_table(keys).to_pylist();assert len(rows)==38
    assert all(r['pixels_identical'] and r['ref_path']==r['dist_path'] and
               Path(r['ref_path']).parent==Path('/mnt/v/input/zensim/sources') for r in rows)
    dest=OUT/'pixel-identity';dest.mkdir(exist_ok=True)
    (dest/'PREREAD.json').write_text(json.dumps({'model_pins':PINS,'rows':rows,'pixel_bar':100},indent=2)+'\n')
    for r in rows:assert SHA(r['ref_path'])==r['reference_file_sha256']
    pairs=dest/'pairs.tsv'
    with pairs.open('w') as f:
        f.write('ref_path\tdist_path\n')
        for r in rows:f.write(r['ref_path']+'\t'+r['dist_path']+'\n')
    summary={}
    for label in ['seed0','B']:
        p=dest/f'{label}.tsv';scores(pairs,label,p)
        with p.open() as f:rs=list(csv.reader(f,delimiter='\t'))[1:]
        assert len(rs)==38
        vs=[float(r[2]) for r in rs]
        summary[label]={'n':38,'min':min(vs),'max':max(vs),'exact100':sum(v==100 for v in vs)}
    (dest/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)


def chromaq():
    freeze=json.loads((ROOT/'benchmarks/prodqual_b_chromaq_registration_2026-10-07.json').read_text())
    dest=OUT/'chromaq';dest.mkdir(exist_ok=True)
    rows=[]
    for p in freeze['pairs']:
        with Path(p).open() as f:rs=list(csv.DictReader(f,delimiter='\t'))
        for r in rs:
            for field in ['ref_path','dist_path']:
                assert r[field].startswith('/w/')
                r[field]=str(Path.home()/'tmp/chromaq'/r[field][3:])
            rows.append(r)
    refs={r['ref_path'] for r in rows};assert len(rows)==1760 and len(refs)==32
    (dest/'PREREAD.json').write_text(json.dumps({'registration':freeze,'pins':PINS,
        'paths':[{'ref_path':r['ref_path'],'dist_path':r['dist_path']} for r in rows]},indent=2)+'\n')
    inputs=sorted(refs|{r['dist_path'] for r in rows})
    (dest/'PIXEL_INPUT_HASHES.json').write_text(json.dumps({p:SHA(p) for p in inputs},indent=2)+'\n')
    parts=[]
    for i in range(4):
        p=dest/f'pairs-{i}.tsv';parts.append(p)
        with p.open('w') as f:
            f.write('ref_path\tdist_path\n')
            for r in rows[i::4]:f.write(r['ref_path']+'\t'+r['dist_path']+'\n')
    def one(p):
        target=p.with_name(p.stem+'-seed0.tsv');scores(p,'seed0',target);print('CHROMAQ scored',p.name,flush=True);return target
    with ThreadPoolExecutor(max_workers=4) as pool: outputs=list(pool.map(one,parts))
    merged=dest/'scores.tsv'
    with merged.open('w') as out:
        for i,p in enumerate(outputs):
            lines=p.read_text().splitlines(True);out.writelines(lines if i==0 else lines[1:])
    cmd=[sys.executable,str(ROOT/'benchmarks/chromaq_2026-10-04/analyze.py'),*freeze['sweeps'],str(merged),str(dest/'summary.json')]
    with (dest/'analyze.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('CHROMAQ complete: 1760 registered cells, rule pending',flush=True)


if __name__=='__main__':
    for p,h in PINS.items():assert SHA(OUT/p)==h
    if sys.argv[1]=='identity':identity()
    elif sys.argv[1]=='chromaq':chromaq()
    else:raise SystemExit('choose identity or chromaq')
