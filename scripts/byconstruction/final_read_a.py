#!/usr/bin/env python3
"""Registered frozen-A assessment orchestration; inference/statistics stay in Rust owners.

keys/stage/decode/bank/predict/peers never read a human target. The separate read
entry is permitted only after its committed pins and per-population exposure.
"""
import argparse
import ast
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
LIB = Path('/mnt/tower/input/iqa/datasets')
EXTRACTOR = Path('/var/tmp/byconstruction-bin/b77883be18db-rev7/extract_features_372col')
EXTRACTOR_SHA = 'ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8'
SETS = ('cid22_b', 'csiq', 'live', 'konjnd_terminal', 'kadid_terminal', 'aic4', 'mcljci', 'clic22')
DATASETS = dict(live='live-iqa', mcljci='mcl-jci', clic22='clic-2022-perceptual')
EXPECTED = dict(cid22_b=2100, csiq=866, live=779, konjnd_terminal=100,
                kadid_terminal=2000, aic4=300, mcljci=5000, clic22=57300)
CURRENT = None


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def progress(a, state, child=None, error=None):
    p=a.root/'PROGRESS.json'
    value=json.loads(p.read_text()) if p.exists() else {}
    for old in ('held_population','verified_process_observation','labels_parsed','sets_scored'):
        value.pop(old,None)
    reads={}
    for name in SETS:
        receipt=a.root/'scores'/name/'LABEL_READS.json'
        if receipt.exists():reads[name]=json.loads(receipt.read_text())['state']
    value.update(stage=f'{a.set}:{a.action}:{state}',last_check=datetime.now(timezone.utc).isoformat(),
        controller_pid=os.getpid() if state=='running' else None,child_pid=child,
        label_reads=reads,completed_sets=[name for name in SETS if (a.root/'scores'/name/'RESULTS.json').exists()],
        required_sets=list(SETS),failure=error)
    value['extractor_sha256']=EXTRACTOR_SHA
    for file,field in [('bake_dial_refit','predictor_sha256'),('trust-decode','decoder_sha256')]:
        binary=a.root/'bin'/file
        if binary.is_file():value[field]=sha(binary)
    value['next_action']={
        'stage':'keys','keys':'decode','decode':'bank','verify_pixels':'ledger after predictions and peers','bank':'predict','predict':'peers',
        'peers':'comparators then ledger' if a.set=='kadid_terminal' else 'ledger',
        'comparators':'ledger','ledger':'commit/push/verify exposure, then read','read':'stats',
        'stats':'publish results and continue in registered population order',
    }[a.action] if state=='complete' else f'supervise {a.set}:{a.action}'
    write(p,value)


def command(argv, log, stderr=None, env=None):
    print(json.dumps(dict(command=list(map(str, argv)), log=str(log))), flush=True)
    with log.open('w') as f:
        if stderr is None:
            child=subprocess.Popen(list(map(str, argv)), stdout=f, stderr=subprocess.STDOUT, env=env)
            if CURRENT is not None:progress(CURRENT,'running',child=child.pid)
            code=child.wait()
        else:
            with stderr.open('w') as errors:
                child=subprocess.Popen(list(map(str, argv)), stdout=f, stderr=errors, env=env)
                if CURRENT is not None:progress(CURRENT,'running',child=child.pid)
                code=child.wait()
        if code:raise subprocess.CalledProcessError(code,list(map(str,argv)))


def key_rows(path, columns):
    """mos_confirm.keys discipline: select key strings only, never a target."""
    with open(path, newline='') as f:
        reader = csv.reader(f, delimiter=',' if Path(path).suffix == '.csv' else '\t')
        header = next(reader)
        indices = [header.index(c) for c in columns]
        for i, row in enumerate(reader):
            yield i, [row[j] for j in indices]


def label_spec(name):
    pin = json.loads((REPO/'benchmarks/rev4_featpot_effaudit/v2c_setcompare_pin_r7a_2026-10-04.json').read_text())
    if name in ('cid22_b', 'csiq', 'aic4', 'konjnd_terminal'):
        spec = pin['labels']['konjnd_jpeg_terminal' if name == 'konjnd_terminal' else name]
        return dict(path=spec['path'], sha256=spec['sha256'], target='human_score')
    paths = dict(live='/mnt/v/datasets/LIVE/live_r2_pairs.tsv',
                 kadid_terminal='/mnt/v/dataset/kadid10k/dmos.csv',
                 clic22=str(LIB/'clic-2022-perceptual/pairs.tsv'))
    digests = dict(live='49fcd7d23c9e35b147b52e3361103dbbf6a81aa8ae9d7b16f4763c7e7051fe5d',
                   kadid_terminal='573e2ed98fdaa2a5aed7b50ad716906e125e1001ed0e97c8e93e79d7b997ba99',
                   clic22='03300f61963c2fc96e5daf3cad3a0e8a43438637959875b904d7a944db295db1')
    path = paths[name]
    assert sha(path) == digests[name], 'registered label-file SHA changed before keys'
    target = dict(live='human_score', kadid_terminal='dmos',
                  clic22='majority(n_a,n_b), tied majority=0.5')[name]
    return dict(path=path, sha256=digests[name], target=target)


def stage(a):
    dataset = DATASETS[a.set]
    src, dst = LIB/dataset, a.root/'stage'/dataset
    dst.mkdir(parents=True, exist_ok=True)
    receipt = dst/'STAGING.json'
    if receipt.exists():
        raise RuntimeError('staging already verified; do not overwrite its receipt')
    command(['rsync', '-aL', '--no-owner', '--no-group', '--info=progress2',
             str(src/'content')+'/', str(dst/'content')+'/'], dst/'rsync.log')
    expected = {}
    for line in (src/'SHA256SUMS').read_text().splitlines():
        digest, rel = line.split('  ', 1)
        if rel.startswith('content/'):
            expected[rel] = digest
    actual = {str(p.relative_to(dst)) for p in (dst/'content').rglob('*') if p.is_file()}
    assert actual == set(expected), (len(actual), len(expected), 'staging coverage')
    total = 0
    for i, (rel, digest) in enumerate(expected.items(), 1):
        p = dst/rel
        assert sha(p) == digest, str(p)
        total += p.stat().st_size
        if i % 100 == 0:
            print(f'verified staging {i}/{len(expected)}', flush=True)
    write(receipt, dict(source=str(src/'content'), files=len(expected), bytes=total,
                        dataset_sha256sums_sha256=sha(src/'SHA256SUMS'), every_file_verified=True,
                        labels_parsed=False))
    print(receipt.read_text())


def keys(a):
    out = a.root/'inputs'/a.set
    out.mkdir(parents=True, exist_ok=False)
    rows, questions = [], []
    def add(r, d, source_row, **extra):
        k = a.set+':'+hashlib.sha256((str(r)+'\t'+str(d)).encode()).hexdigest()
        rows.append(dict(pair_key=k, source_row=source_row, original_ref=str(r),
                         original_dist=str(d), origin=Path(r).stem, **extra))
    if a.set == 'mcljci':
        base = a.root/'stage/mcl-jci/content/MCL-JCI'
        for s in range(1, 51):
            for q in range(1, 101):
                r = base/f'source_images/[BMP]/ImageJND_SRC{s:02d}.bmp'
                d = base/f'distorted_image/ImageJND_SRC{s:02d}/ImageJND_SRC{s:02d}_{q:03d}.jpg'
                add(r, d, len(rows), source=s, q=q)
        spec = dict(path=str(LIB/'mcl-jci/labels/JND_samples'), raw_pin_sha256=sha(a.root/'pins/MCL_RAW_SHA256.json'),
                    target='registered mean subject-chain JND count')
    elif a.set == 'clic22':
        spec = label_spec(a.set)
        seen = {}
        for i, (r, x, y) in key_rows(spec['path'], ['ref', 'a', 'b']):
            questions.append(dict(source_row=i, ref=r, a=x, b=y))
            for d in (x, y):
                if (r, d) not in seen:
                    seen[r, d] = True
                    add(r, d, i)
    else:
        spec = label_spec(a.set)
        assert sha(spec['path']) == spec['sha256'], 'label file changed before keys'
        columns = ['ref_img', 'dist_img'] if a.set == 'kadid_terminal' else ['ref_path', 'dist_path']
        bref = set()
        if a.set == 'cid22_b':
            bank = a.root/'pins/CID22_B_APPROVED_POPULATION_KEYS.parquet'
            bref = {Path(x).stem for x in pq.read_table(bank, columns=['ref_path'])['ref_path'].to_pylist()}
            assert len(bref) == 24
            spec['population_keys_sha256'] = sha(bank)
        for i, (r, d) in key_rows(spec['path'], columns):
            if a.set == 'cid22_b' and Path(r).stem not in bref:
                continue
            if a.set == 'konjnd_terminal' and int(Path(r).stem[3:]) % 10 not in (7, 9):
                continue
            if a.set == 'kadid_terminal':
                if int(Path(r).stem[1:]) % 10 not in (7, 9):
                    continue
                r, d = str(Path(spec['path']).parent/'images'/r), str(Path(spec['path']).parent/'images'/d)
            add(r, d, i)
    def source(path):
        if a.set == 'clic22':
            return a.root/'stage/clic-2022-perceptual/content'/path
        if a.set == 'live':
            return a.root/'stage/live-iqa/content/LIVEIQA_release2'/Path(path).parent.name/Path(path).name
        return Path(path)
    for row in rows:
        row['ref_path'], row['dist_path'] = str(source(row['original_ref'])), str(source(row['original_dist']))
        for c in ('ref_path', 'dist_path'):
            assert Path(row[c]).is_file(), row[c]
    assert len(questions or rows) == EXPECTED[a.set], (a.set, len(questions or rows))
    assert len({r['pair_key'] for r in rows}) == len(rows), 'duplicate pair keys'
    write(out/'KEYS.json', dict(set=a.set, rows=rows, questions=questions, label_spec=spec, labels_parsed=False))
    paths = dict.fromkeys(p for r in rows for p in (r['ref_path'], r['dist_path']))
    (out/'decode-list.txt').write_text(''.join(p+'\n' for p in paths))
    print(json.dumps(dict(set=a.set, pairs=len(rows), questions=len(questions), references=len({r['origin'] for r in rows}),
                          stimuli=len(paths), keys_sha256=sha(out/'KEYS.json'), labels_parsed=False)), flush=True)


def decode(a):
    out = a.root/'inputs'/a.set
    with (out/'decode-map.tsv').open('x') as f:
        subprocess.run([str(a.root/'bin/trust-decode'), str(out/'decode-list.txt'),
                        str(a.root/'pixels'/a.set), '--preserve-rgb8-png', '--fast-png'], stdout=f, check=True)
    verify_pixels(a)


def verify_pixels(a):
    """Independent readback of the exact PNG buffers consumed by every scorer."""
    from PIL import Image, __version__ as pillow_version
    out=a.root/'inputs'/a.set
    with (out/'decode-map.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
    source=json.loads((out/'KEYS.json').read_text())
    assert {r['input'] for r in rows} == {r[c] for r in source['rows'] for c in ('ref_path','dist_path')}
    seen={}
    for r in rows:
        expected=(int(r['width']),int(r['height']),r['rgb8_sha256'])
        if r['png'] in seen:
            assert seen[r['png']]==expected,'CAS geometry/hash ambiguity'
            continue
        with Image.open(r['png']) as image:
            assert image.size==expected[:2],r['png']
            assert hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()==expected[2],r['png']
        seen[r['png']]=expected
        if len(seen)%100==0:print(f'independent PNG readback {len(seen)}',flush=True)
    receipt=dict(images=len(seen),source_stimuli=len(rows),all_geometry_and_rgb_hashes_match=True,
                 decode_map_sha256=sha(out/'decode-map.tsv'),oracle='Pillow '+pillow_version,labels_parsed=False)
    write(out/'INDEPENDENT_PIXEL_READBACK.json',receipt);print(json.dumps(receipt),flush=True)


def bank(a):
    out = a.root/'inputs'/a.set
    source = json.loads((out/'KEYS.json').read_text())
    with (out/'decode-map.tsv').open() as f:
        decoded = {r['input']:r for r in csv.DictReader(f, delimiter='\t')}
    rows = source['rows']
    table = pa.table(dict(row_id=list(range(len(rows))), pair_key=[r['pair_key'] for r in rows],
        ref_path=[decoded[r['ref_path']]['png'] for r in rows], dist_path=[decoded[r['dist_path']]['png'] for r in rows],
        origin=[r['origin'] for r in rows]))
    kp = out/'keys.parquet'
    pq.write_table(table, kp, compression='zstd')
    assert sha(EXTRACTOR) == EXTRACTOR_SHA
    instrument = dict(schema='rev5-assessment-keys-v1', labels_read=False, rows=len(rows),
        keys=dict(path=str(kp),sha256=sha(kp)), extractor=dict(path=str(EXTRACTOR),sha256=EXTRACTOR_SHA),
        immutable_roots=[str(out),str(a.root/'pixels'/a.set)],
        instrument=dict(name=a.set,source=source['label_spec'],selection='registered label-free population'),
        exposure_freeze=dict(state='features-only',labels_read=False,scope='final_read_a_2026-11-05.md'))
    write(out/'instrument.json',instrument)
    command(['python3','-B',REPO/'scripts/rev4_featpot/rev5_bank.py','extract',a.set,
             '--bin',EXTRACTOR,'--build-commit','b77883be18db5933a9db4b9e2670a2ba88a9729b',
             '--revision','7','--era','rev7_rowlanes','--threads','1','--chunk','5000',
             '--retain-feature-csv',
             '--instrument-manifest',out/'instrument.json','--out',a.root/'banks'],out/'bank.log')
    audited = pq.read_table(a.root/'banks'/a.set/'keys.parquet').to_pylist()
    for row, audit in zip(rows, audited, strict=True):
        for c, field in [('ref_path','reference_pixels_sha256'),('dist_path','distorted_pixels_sha256')]:
            assert decoded[row[c]]['rgb8_sha256'] == audit[field], (row['pair_key'],field,'decode parity')
    print(json.dumps(dict(set=a.set,rows=len(rows),decode_hashes_verified=2*len(rows))),flush=True)


def predict(a):
    out = a.root/'scores'/a.set
    out.mkdir(parents=True,exist_ok=True)
    evidence=pq.read_table(a.root/'banks'/a.set/'keys.parquet',columns=['pixels_identical'])
    count=sum(evidence['pixels_identical'].to_pylist())
    if count:
        proof=json.loads((a.root/'inputs'/a.set/'CACHED_IDENTITY_PARITY.json').read_text())
        assert proof['rows']==count and proof['all_cached_equal_native'] and proof['all_scores_exactly_100']
        for field,path in [('bank_keys_sha256',a.root/'banks'/a.set/'keys.parquet'),
                           ('feature_bank_sha256',a.root/'banks'/a.set/'features.parquet'),
                           ('model_sha256',a.root/'models/A.bin'),
                           ('predictor_sha256',a.root/'bin/bake_dial_refit'),
                           ('native_scorer_sha256',a.root/'bin/serve_custom_bake')]:
            assert proof[field]==sha(path), 'identity parity proof does not bind this instrument'
    tab = pq.read_table(a.root/'banks'/a.set/'features.parquet').drop(['row_id'])
    tab = tab.rename_columns(['ref_basename' if c=='pair_key' else c for c in tab.column_names])
    tab = tab.append_column('human_score',pa.array([0.0]*tab.num_rows))
    corpus = out/'features.parquet';pq.write_table(tab,corpus,compression='zstd')
    command([a.root/'bin/bake_dial_refit','predict','--bake',a.root/'source-public/zensim/weights/rev7_mix_h128_2026-11-03.bin',
             '--corpus',corpus,'--score-units','--out',out/'pred_A.tsv'],out/'predict.log',
             env={**os.environ,'ZENSIM_FORMULA_REV':'7'})


def peers(a):
    keys = pq.read_table(a.root/'inputs'/a.set/'keys.parquet').to_pylist()
    out = a.root/'scores'/a.set;out.mkdir(parents=True,exist_ok=True)
    pairs = out/'peer-pairs.tsv'
    with pairs.open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['id','ref','dist'])
        writer.writerows((r['pair_key'],r['ref_path'],r['dist_path']) for r in keys)
    threads=min(8,int(os.environ.get('RAYON_NUM_THREADS','1')))
    assert threads>=1
    env={**os.environ,'AIC4_MODEL_DIR':'/mnt/v/zenmetrics-private/aic4/models/ecec4e0'}
    if threads>1:
        control=out/'peer-thread-parity';control.mkdir(exist_ok=False)
        selected=np.linspace(0,len(keys)-1,min(64,len(keys)),dtype=int)
        probe=control/'pairs.tsv'
        with probe.open('x') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['id','ref','dist'])
            w.writerows((keys[i]['pair_key'],keys[i]['ref_path'],keys[i]['dist_path']) for i in selected)
        scores_by_thread={}
        for n in (1,threads):
            dest=control/f'scores-{n}.tsv'
            command([a.root/'bin/jnd-ladder-score','--configs','','--threads',str(n),probe,dest],control/f'run-{n}.log',env=env)
            with dest.open() as f:
                scores_by_thread[n]={r['id']:(r['ssim2'],r['ba_p3']) for r in csv.DictReader(f,delimiter='\t')}
        assert scores_by_thread[1]==scores_by_thread[threads] and len(scores_by_thread[1])==len(selected), 'peer thread parity failed before labels'
        write(control/'PARITY.json',dict(rows=len(selected),threads=[1,threads],all_registered_metric_outputs_exact=True,
              input_sha256=sha(probe),binary_sha256=sha(a.root/'bin/jnd-ladder-score'),labels_parsed=False))
    command([a.root/'bin/jnd-ladder-score','--configs','','--threads',str(threads),pairs,out/'peers.tsv'],out/'peers.log',env=env)


def comparators(a):
    assert a.set=='kadid_terminal'
    out=a.root/'scores'/a.set
    keys=pq.read_table(a.root/'inputs'/a.set/'keys.parquet').to_pylist()
    pairs=out/'comparator-pairs.tsv'
    with pairs.open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['ref_path','dist_path','pair_key'])
        w.writerows((r['ref_path'],r['dist_path'],r['pair_key']) for r in keys)
    models=json.loads((a.root/'pins/MODELS.json').read_text())
    for name in ('B','by_v2fy'):
        if name not in models:continue
        spec=models[name];assert sha(spec['path'])==spec['sha256']
        command([a.root/'bin/serve_custom_bake','--native-pairs',pairs,spec['path']],out/f'pred_{name}.tsv',
                stderr=out/f'predict_{name}.log',env={**os.environ,'ZENSIM_FORMULA_REV':str(spec['revision'])})


def scores(a):
    out=a.root/'scores'/a.set
    rows=json.loads((a.root/'inputs'/a.set/'KEYS.json').read_text())['rows']
    with (out/'pred_A.tsv').open() as f:
        predictions=list(csv.DictReader(f,delimiter='\t'))
    assert len(predictions)==len(rows)
    if predictions and 'row_idx' in predictions[0]:
        assert [int(r['row_idx']) for r in predictions]==list(range(len(rows))), 'prediction row order'
    values={'A':{r['pair_key']:float(p['pred']) for r,p in zip(rows,predictions,strict=True)}}
    with (out/'peers.tsv').open() as f:
        peer=list(csv.DictReader(f,delimiter='\t'))
    values['SSIMULACRA2']={r['id']:float(r['ssim2']) for r in peer}
    values['butteraugli3_quality']={r['id']:-float(r['ba_p3']) for r in peer}
    for name in ('B','by_v2fy'):
        p=out/f'pred_{name}.tsv'
        if p.exists():
            with p.open() as f:pred=list(csv.DictReader(f,delimiter='\t'))
            assert len(pred)==len(rows)
            values[name]={r['pair_key']:float(list(v.values())[-1]) for r,v in zip(rows,pred,strict=True)}
    expected={r['pair_key'] for r in rows}
    for name,v in values.items():
        assert set(v)==expected and np.isfinite(list(v.values())).all(), (name,'prediction coverage')
    if a.set=='kadid_terminal':
        assert 'B' in values, 'KADID B comparator required before labels'
        if 'by_v2fy' in json.loads((a.root/'pins/MODELS.json').read_text()):
            assert 'by_v2fy' in values, 'the located and approved Rev4 comparator is required'
    return values


def ledger(a):
    out=a.root/'scores'/a.set
    source=json.loads((a.root/'inputs'/a.set/'KEYS.json').read_text())
    scores(a)  # Refuse any incomplete prediction surface before spending labels.
    parity=json.loads((a.root/'parity/contract.json').read_text())
    assert parity['missing']==parity['extra']==parity['different']==0
    oracle=a.root/'inputs'/a.set/'INDEPENDENT_PIXEL_READBACK.json'
    independent=json.loads(oracle.read_text())
    assert independent['all_geometry_and_rgb_hashes_match']
    assert independent['decode_map_sha256']==sha(a.root/'inputs'/a.set/'decode-map.tsv')
    path=out/'LABEL_READS.json'
    assert not path.exists(), 'a registered or spent read already exists; coordinator required'
    pin=dict(set=a.set,state='authorized-before-read',utc=datetime.now(timezone.utc).isoformat(),
        registration='benchmarks/final_read_a_2026-11-05.md',label_spec=source['label_spec'],
        rows=len(source['questions'] or source['rows']),
        key_sha256=sha(a.root/'inputs'/a.set/'KEYS.json'),
        decode_map_sha256=sha(a.root/'inputs'/a.set/'decode-map.tsv'),
        feature_bank_sha256=sha(a.root/'banks'/a.set/'features.parquet'),
        independent_pixel_readback_sha256=sha(oracle),
        score_files={p.name:sha(p) for p in out.glob('*.tsv')},
        binary_sha256={p.name:sha(p) for p in (a.root/'bin').iterdir() if p.is_file()},
        bootstrap=dict(draws=10000,seed=20261105,unit='whole reference',interval='95% percentile, type7'),
        orientation='A/SSIM2 quality; butteraugli quality = -p3; target orientation disclosed per collection',
        cid22_scale='pinned MCOS/100 multiplied by100 for collection-scale MCOS/RMSE',
        clic_ties='every question retained; tied majority or tied predictions receive half credit')
    models=json.loads((a.root/'pins/MODELS.json').read_text())
    for spec in models.values():assert sha(spec['path'])==spec['sha256']
    pin['models']=models
    with path.open('x') as f:
        json.dump(pin,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    with (REPO/'docs/DATA_SPLITS.md').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write(f"\n### Frozen-A {a.set}: individual pre-open exposure — {pin['utc']}\n\n")
        f.write(f"Authorized frozen read of {pin['rows']} rows; before target parsing. LABEL_READS.json SHA256 `{sha(path)}`; label specification `{json.dumps(pin['label_spec'],sort_keys=True)}`. Prediction/decode/binary pins retained in that receipt. Bootstrap10000 seed20261105; no tuning or selection. Status: authorized-before-read; any partial/failed read spends this population.\n")
        f.flush();os.fsync(f.fileno())
    print(json.dumps(pin),flush=True)


def read(a):
    out=a.root/'scores'/a.set;path=out/'LABEL_READS.json'
    pin=json.loads(path.read_text());assert pin['state']=='authorized-before-read'
    assert not (out/'eval_rows.json').exists()
    source=json.loads((a.root/'inputs'/a.set/'KEYS.json').read_text())
    assert sha(a.root/'inputs'/a.set/'KEYS.json')==pin['key_sha256']
    pin['state']='spent-parse-started';write(path,pin)
    try:
        rows=source['rows'];spec=source['label_spec'];targets=[]
        if a.set=='mcljci':
            raw=json.loads((a.root/'pins/MCL_RAW_SHA256.json').read_text())['files']
            chains={}
            for index,entry in enumerate(raw,1):
                assert sha(entry['path'])==entry['sha256']
                with open(entry['path']) as f:
                    ch=[ast.literal_eval(line.strip().split(' ',1)[1]) for line in f if line.strip()]
                assert len(ch)==30 and all(all(isinstance(v,int) and 0<=v<=100 for v in x) for x in ch)
                chains[index]=ch
            for row in rows:
                q=row['q']
                y=0.0 if q==100 else sum(sum(v>=q for v in chain) for chain in chains[row['source']])/30
                targets.append(dict(a=row['pair_key'],group=row['origin'],target=y,source=row['source'],q=q))
            rebuilt=out/'rebuilt_jnd_dist.tsv'
            with rebuilt.open('x') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['source','q','jnd_dist'])
                w.writerows((r['source'],r['q'],r['target']) for r in targets)
            pin['rebuilt_table_sha256']=sha(rebuilt)
            pin['lost_table_byte_identity']='cannot be verified'
        elif a.set=='clic22':
            assert sha(spec['path'])==spec['sha256']
            bysource={r['source_row']:r for r in source['questions']}
            with open(spec['path'],newline='') as f:
                for i,r in enumerate(csv.DictReader(f,delimiter='\t')):
                    q=bysource[i];na,nb=float(r['n_a']),float(r['n_b'])
                    assert na+nb>0
                    pair=lambda d:a.set+':'+hashlib.sha256((q['ref']+'\t'+d).encode()).hexdigest()
                    targets.append(dict(a=pair(q['a']),b=pair(q['b']),group=q['ref'],
                        target=1.0 if na>nb else 0.0 if na<nb else 0.5,source_row=i))
        else:
            assert sha(spec['path'])==spec['sha256']
            bysource={r['source_row']:r for r in rows}
            with open(spec['path'],newline='') as f:
                for i,r in enumerate(csv.DictReader(f,delimiter=',' if a.set=='kadid_terminal' else '\t')):
                    if i not in bysource:continue
                    row=bysource[i];y=float(r[spec['target']])
                    if a.set=='cid22_b':y*=100
                    item=dict(a=row['pair_key'],group=row['origin'],target=y,source_row=i)
                    if a.set=='kadid_terminal':item['distortion']=Path(row['original_dist']).stem.split('_')[1]
                    if a.set=='aic4':item['codec']=Path(row['original_dist']).stem.rsplit('_',1)[0].split('_',2)[-1]
                    targets.append(item)
        assert len(targets)==EXPECTED[a.set] and np.isfinite([r['target'] for r in targets]).all()
        write(out/'eval_rows.json',targets)
        pin.update(state='read',rows_read=len(targets),target_cache_sha256=sha(out/'eval_rows.json'))
        write(path,pin)
        print(json.dumps(pin),flush=True)
    except BaseException as error:
        pin.update(state='failed-spent',error=repr(error));write(path,pin)
        raise


def interval(values):
    finite=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
    if not len(finite):return dict(lo=None,hi=None,valid_draws=0)
    lo,hi=np.quantile(finite,[.025,.975],method='linear')
    return dict(lo=float(lo),hi=float(hi),valid_draws=len(finite))


def mos_stats(a, rows, values, name):
    out=a.root/'scores'/a.set/name;out.mkdir(parents=True,exist_ok=False)
    groups=sorted({r['group'] for r in rows});ix={g:np.array([i for i,r in enumerate(rows) if r['group']==g]) for g in groups}
    y=np.array([r['target'] for r in rows]);x={m:np.array([s[r['a']] for r in rows]) for m,s in values.items()}
    manifest=out/'bootstrap.batch.tsv'
    rng=np.random.default_rng(20261105)
    within_draws=[]
    with manifest.open('w') as f:
        f.write('#def\ty\t'+','.join(map(repr,y.tolist()))+'\n')
        for m,p in x.items():f.write('#def\t'+m+'\t'+','.join(map(repr,p.tolist()))+'\n')
        for m in x:f.write(f'{m}:point\t@{m}:@y\t*\n')
        for b in range(10000):
            draw=rng.integers(0,len(groups),len(groups));within_draws.append(draw)
            indices=np.concatenate([ix[groups[k]] for k in draw]);index_text=','.join(map(str,indices.tolist()))
            for m in x:f.write(f'{m}:boot:{b}\t@{m}:@y\t{index_text}\n')
    command([a.root/'bin/panel','--batch',manifest,'--signed-quality'],out/'bootstrap.jsonl',stderr=out/'bootstrap.stderr.log')
    # Rust stderr is separate from the machine-readable JSONL.
    records=[]
    for line in (out/'bootstrap.jsonl').read_text().splitlines():
        records.append(json.loads(line))
    sys.path.insert(0,str(REPO/'scripts'))
    from lib.zen_stats import panel_batch
    os.environ['ZEN_PANEL_BIN']=str(a.root/'bin/panel')
    result=dict(rows=len(rows),references=len(groups),bootstrap_draws=10000,seed=20261105,models={})
    job=[]
    for m,p in x.items():
        for j,g in enumerate(groups):
            if len(ix[g])>=3:job.append((f'{m}:{j}',p[ix[g]].tolist(),y[ix[g]].tolist()))
    per={r['label']:r['srocc_signed'] for r in panel_batch(job,stats='srocc')} if job else {}
    for m in x:
        point=next(r for r in records if r['label']==f'{m}:point')
        boots=[r for r in records if r['label'].startswith(f'{m}:boot:')]
        result['models'][m]={k:dict(value=point[k],**interval([r[k] for r in boots]))
                                for k in ('srocc_signed','krocc','plcc','rmse')}
        within=np.array([per.get(f'{m}:{j}',np.nan) for j in range(len(groups))])
        valid=np.isfinite(within)
        bs=[float(np.nanmean(within[d])) if np.isfinite(within[d]).any() else None for d in within_draws]
        result['models'][m]['within_reference_srocc']=dict(value=float(within[valid].mean()) if valid.any() else None,
                                                         references_defined=int(valid.sum()),**interval(bs))
    if a.set=='kadid_terminal':
        indexed={r['label']:r for r in records}
        delta=np.array([indexed[f'A:boot:{i}']['srocc_signed']-indexed[f'B:boot:{i}']['srocc_signed'] for i in range(10000)])
        point=result['models']['A']['srocc_signed']['value']-result['models']['B']['srocc_signed']['value']
        se=float(np.std(delta,ddof=1));rule=dict(delta_A_minus_B=point,paired_bootstrap_se=se,
            lower_limit=-2*se,pass_B=point>=-2*se,**interval(delta))
        if 'by_v2fy' in x:
            d=result['models']['A']['srocc_signed']['value']-result['models']['by_v2fy']['srocc_signed']['value']
            rule.update(delta_A_minus_by_v2fy=d,by_v2fy_limit=-.005,pass_by_v2fy=d>=-.005)
            rule['by_v2fy_delta_interval']=interval([indexed[f'A:boot:{i}']['srocc_signed']-
                indexed[f'by_v2fy:boot:{i}']['srocc_signed'] for i in range(10000)])
        else:rule['deviation']='missing Rev4 by_v2fy comparator, coordinator-approved fallback'
        rule['passed']=rule['pass_B'] and rule.get('pass_by_v2fy',True);result['confirmation']=rule
        from lib.zen_stats import reference_srocc_interval, scatter
        result['per_distortion_srocc']={}
        for distortion in sorted({r['distortion'] for r in rows}):
            keep=[i for i,r in enumerate(rows) if r['distortion']==distortion]
            result['per_distortion_srocc'][distortion]={}
            for m,p in x.items():
                result['per_distortion_srocc'][distortion][m]=reference_srocc_interval(
                    p[keep].tolist(),y[keep].tolist(),[groups.index(rows[i]['group']) for i in keep],resamples=10000,seed=20261105)
        result['scatter']={}
        for m,p in x.items():
            dest=out/f'scatter-{m}.json';write(dest,scatter(p.tolist(),y.tolist()))
            result['scatter'][m]=dict(path=str(dest),sha256=sha(dest))
    write(out/'RESULTS.json',result);print(json.dumps(result),flush=True)
    return result


def stats(a):
    out=a.root/'scores'/a.set;pin=json.loads((out/'LABEL_READS.json').read_text())
    assert pin['state']=='read' and sha(out/'eval_rows.json')==pin['target_cache_sha256']
    rows=json.loads((out/'eval_rows.json').read_text());values=scores(a)
    if a.set=='clic22':
        sys.path.insert(0,str(REPO/'scripts/byconstruction'))
        from dataset_eval import pair_agreement
        os.environ['ZEN_PANEL_BIN']=str(a.root/'bin/panel')
        result=dict(rows=len(rows),references=len({r['group'] for r in rows}),models={})
        for m,s in values.items():
            value,lo,hi,n=pair_agreement(rows,s,resamples=10000,seed=20261105)
            result['models'][m]=dict(agreement=value,lo=lo,hi=hi,questions=n)
        write(out/'RESULTS.json',result);print(json.dumps(result),flush=True);return
    full=mos_stats(a,rows,values,'all')
    if a.set=='live':
        sys.path.insert(0,str(REPO/'scripts/rev4_featpot'))
        from external_sets import CONTENT_OVERLAP
        clean=[r for r in rows if r['group']+'.bmp' not in CONTENT_OVERLAP['live']]
        assert len(clean)==267 and len({r['group'] for r in clean})==10
        mos_stats(a,clean,values,'clean10')
    if a.set=='aic4':
        subset=[r for r in rows if r['codec']=='JPEG-AI'];assert len(subset)==50
        mos_stats(a,subset,values,'SDR25')
    if a.set=='mcljci':
        excluded={1,3,6,7,9,17,28,31,45,50}
        clean=[r for r in rows if r['source'] not in excluded];assert len(clean)==4000
        mos_stats(a,clean,values,'R7a_clean40')
    write(out/'RESULTS.json',full)


def main():
    global CURRENT, EXTRACTOR
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['stage','keys','decode','verify_pixels','bank','predict','peers','comparators','ledger','read','stats'])
    parser.add_argument('set',choices=SETS)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    assert args.root.is_absolute()
    CURRENT=args
    EXTRACTOR=args.root/'bin/extract_features_372col'
    progress(args,'running')
    try:
        globals()[args.action](args)
    except BaseException as error:
        progress(args,'failed',error=repr(error));raise
    progress(args,'complete')


if __name__ == '__main__':
    main()
