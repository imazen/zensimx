#!/usr/bin/env python3
"""Replay only the preregistered engineering instruments on frozen seed zero."""
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
FREEZE = json.loads((ROOT / 'benchmarks/prodqual_b_registration_2026-10-07.json').read_text())
PINS = json.loads((OUT / 'PROGRAM_MODEL_PINS.json').read_text())
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ENV = dict(os.environ, ZENSIM_FORMULA_REV='5', ZENSIM_PREPARED_STEERING='1',
           RAYON_NUM_THREADS='1', TMPDIR=str(Path.home() / 'tmp/prodqual-b'))
for key in ['ZENSIM_NEIGHBOUR_EXACT', 'ZENSIM_FINITE_MOMENTS', 'ZENSIM_REPAIR_ALPHA',
            'ZENSIM_REPAIR_SOURCE', 'ZENSIM_STEERING_BIN', 'ZENSIM_V2_DIAG']:
    ENV.pop(key, None)


def run(cmd, log, env=ENV):
    with Path(log).open('w') as f:
        f.write('COMMAND ' + json.dumps(cmd) + '\n'); f.flush()
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return result.returncode


def inventory(name):
    row = next(r for r in FREEZE['inputs'] if r['name'] == name)
    assert SHA(row['declaration']) == row['declaration_sha256']
    assert SHA(row['path']) == row['sha256']
    return row


def dial(name):
    tables = {n: inventory(n) for n in [name, 'negtail', 'identity']}
    dest = OUT / name; dest.mkdir(exist_ok=True)
    command = [str(OUT / 'bin/bake_verdict'), '--bake', str(OUT / 'models/seed0.bin'),
               '--regime', '720', '--features-root', str(OUT / 'no-human-corpus'),
               '--corpora', 'none', '--dial-grid', tables[name]['path'],
               '--negtail-probe', tables['negtail']['path'], '--identity-probe', tables['identity']['path'],
               '--corruption-grid', str(OUT / 'no-integrity-grid'),
               '--perpair-metrics', str(OUT / 'no-perpair-grid'), '--require-feature-set-match',
               '--gaddr-json', str(dest / 'gaddr.json'), '--fulleval', str(dest / 'fulleval.json'),
               '--output', str(dest / 'verdict.md')]
    (dest / 'COMMAND.json').write_text(json.dumps(command, indent=2)+'\n')
    rc = run(command, dest / 'verdict.log')
    (dest / 'EXIT.json').write_text(json.dumps({'rc': rc})+'\n')
    print(name, 'rc', rc, flush=True)


def steering(label="seed0"):
    env = dict(ENV, ZENSIM_FORMULA_REV="5" if label == "seed0" else "1")
    suffix = "" if label == "seed0" else "_B"
    jobs = []
    for name in ['broad', 'owner', 'jpeg8', 'm3']:
        row = next(r for r in FREEZE['inputs'] if r['name'] == name)
        path = Path(row['path'].replace('.parquet', '.keys.parquet'))
        assert SHA(path) == row['keys_sha256']
        rows = pq.read_table(path).to_pylist(); assert len(rows) == row['rows']
        for r in rows:
            assert r['source_role'] in ['TRAIN', 'design-released KADID SELECT; previously exposed development',
                                       'historical M3 fixture; diagnostic role/overlap limits retained']
            for field in ['ref_path', 'dist_path']:
                assert Path(r[field]).is_relative_to('/var/tmp/shippath7/delivery')
            jobs.append(dict(r, instrument=name))
    (OUT / ('STEERING_PREREAD'+suffix+'.json')).write_text(json.dumps({'model':PINS['models/'+label+'.bin'],
        'binary':PINS['bin/diffmap_block_coherence'],'rules':{'m2':.99,'m3f':.70},'cases':jobs},indent=2)+'\n')
    for r in jobs:
        assert SHA(r['ref_path']) == r['reference_file_sha256']
        assert SHA(r['dist_path']) == r['distorted_file_sha256']
    dest = OUT / ('steering' + suffix); dest.mkdir(exist_ok=True)
    def one(r):
        p = dest / (r['instrument']+'-'+str(r['row_id'])+'.json')
        cmd = [str(OUT/'bin/diffmap_block_coherence'),r['ref_path'],r['dist_path'],
               '--block',str(r['block']),'--bake',str(OUT/'models'/f'{label}.bin'),'--json',str(p)]
        rc = run(cmd, p.with_suffix('.log'), env)
        d = json.loads(p.read_text()) if p.exists() else {}
        supported = d.get('refinement_available') is True and not d.get('refinement_unsupported_ids')
        status = 'blocked' if rc or not supported else ('PASS' if d['m2']>=.99 and d['m3f']>=.70 else 'FAIL')
        return {'instrument':r['instrument'],'row':r['row_id'],'case':r['case'],'rc':rc,
                'status':status,'m2':d.get('m2'),'m3f':d.get('m3f'),'report':str(p),
                'sha256':SHA(p) if p.exists() else None}
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(one,jobs):
            results.append(r)
            (OUT/('STEERING_PARTIAL'+suffix+'.json')).write_text(json.dumps(results,indent=2)+'\n')
            print('steering',len(results),'/',len(jobs),r['status'],flush=True)
    (OUT/('STEERING_RESULTS'+suffix+'.json')).write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    for path, sha in PINS.items():
        assert SHA(OUT/path) == sha
    if sys.argv[1] == 'dial':
        for name in ['standard','ladder']: dial(name)
    elif sys.argv[1] == 'steering': steering()
    elif sys.argv[1] == 'steering-B': steering('B')
    else: raise SystemExit('choose dial or steering')
