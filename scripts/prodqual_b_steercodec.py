#!/usr/bin/env python3
"""Frozen codec-swap diagnostic; Rust owns maps, composites and scoring.

Uses the existing STEERCODEC statistic owner unchanged. Original CLIC inputs
are read-only links; every generated artifact goes to the qualification root.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np

REPO=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/v/output/zensim/prodqual-b-2026-10-07')
ROOT=OUT/'steercodec'
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    freeze=json.loads((REPO/'benchmarks/prodqual_b_steercodec_registration_2026-10-07.json').read_text())
    pins=json.loads((OUT/'PROGRAM_MODEL_PINS.json').read_text())
    for p,h in pins.items():assert SHA(OUT/p)==h
    if (ROOT/'PREREAD.json').exists():
        raise SystemExit('qualification artifacts already exist; preserve them before a fresh run')
    ROOT.mkdir(exist_ok=True)
    for name in ['src','dec','swap','comp','score']:(ROOT/name).mkdir(exist_ok=True)
    (ROOT/'PREREAD.json').write_text(json.dumps({'registration':freeze,'pins':pins},indent=2)+'\n')
    hashes={p:SHA(p) for p in freeze['paths']}
    (ROOT/'INPUT_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for src in freeze['paths']:
        p=Path(src);target=ROOT/p.relative_to('/var/tmp/steercodec')
        target.parent.mkdir(exist_ok=True,parents=True)
        target.symlink_to(p)
    base=dict(os.environ,ZENSIM_FORMULA_REV='5',ZENSIM_PREPARED_STEERING='1',RAYON_NUM_THREADS='1',TMPDIR=str(Path.home()/'tmp/prodqual-b'))
    for k in ['ZENSIM_NEIGHBOUR_EXACT','ZENSIM_REPAIR_SOURCE','ZENSIM_FINITE_MOMENTS','ZENSIM_REPAIR_ALPHA','ZENSIM_STEERING_BIN','ZENSIM_V2_DIAG']:base.pop(k,None)
    jobs=[(c,s['name'],lo,hi) for c in freeze['codecs'] for s in freeze['sources'] for lo,hi in freeze['swaps']]
    def swap(job):
        c,img,lo,hi=job;name=f'{c}-{img}-q{lo}to{hi}-seed0-x0';p=ROOT/'swap'/f'{name}.json'
        env=dict(base,ZENSIM_REPAIR_SOURCE=str(ROOT/'dec'/c/f'{img}_q{hi}.png'))
        cmd=[str(OUT/'bin/diffmap_block_coherence'),str(ROOT/'src'/f'{img}.png'),str(ROOT/'dec'/c/f'{img}_q{lo}.png'),'--bake',str(OUT/'models/seed0.bin'),'--block','8','--json',str(p)]
        with p.with_suffix('.log').open('w') as log:
            log.write('COMMAND '+json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return p
    with ThreadPoolExecutor(max_workers=4) as pool:
        maps=[]
        for p in pool.map(swap,jobs):maps.append(p);print('codec maps',len(maps),'/',len(jobs),flush=True)
    compose=[]
    for p in maps:
        c,img,qq=p.stem.split('-')[:3];lo,hi=qq[1:].split('to');d=json.loads(p.read_text());assert d['refinement_available'] and not d['refinement_unsupported_ids'];bs=d['blocks'];k=int(round(.25*len(bs)))
        rankings={'map':[-b['refinement_gain'] for b in bs],'oracle':[-b['score_delta'] for b in bs],'random':list(np.random.default_rng(20261004).permutation(len(bs))),'anti':[b['refinement_gain'] for b in bs]}
        for ranking,v in rankings.items():
            compose.append({'low':str(ROOT/'dec'/c/f'{img}_q{lo}.png'),'high':str(ROOT/'dec'/c/f'{img}_q{hi}.png'),'output':str(ROOT/'comp'/f'{p.stem}-{ranking}.png'),'bounds':[b['bounds'] for b in bs],'selected':[int(i) for i in np.argsort(v,kind='stable')[:k]]})
    packet=ROOT/'COMPOSE.json';packet.write_text(json.dumps(compose)+'\n')
    with (ROOT/'compose.log').open('w') as log:subprocess.run([str(OUT/'bin/quality_swap_composites'),str(packet)],stdout=log,stderr=subprocess.STDOUT,check=True)
    # Cross-check every new native PNG using an independent reader against
    # explicit low/high pixels and the frozen selected rectangles.
    from PIL import Image
    for j in compose:
        lo=np.asarray(Image.open(j['low']).convert('RGB')).copy();hi=np.asarray(Image.open(j['high']).convert('RGB'))
        for i in j['selected']:
            x0,y0,x1,y1=j['bounds'][i];lo[y0:y1,x0:x1]=hi[y0:y1,x0:x1]
        assert np.array_equal(lo,np.asarray(Image.open(j['output']).convert('RGB')))
    print('native composites independently verified',len(compose),flush=True)
    pairs=[]
    for c in freeze['codecs']:
        for s in freeze['sources']:
            for q in [30,60,85]:pairs.append((str(ROOT/'src'/f"{s['name']}.png"),str(ROOT/'dec'/c/f"{s['name']}_q{q}.png")))
    for j in compose:pairs.append((str(ROOT/'src'/f"{Path(j['output']).stem.split('-')[1]}.png"),j['output']))
    path=ROOT/'score/pairs.tsv'
    with path.open('w') as f:
        f.write('ref_path\tdist_path\n')
        for a,b in pairs:f.write(a+'\t'+b+'\n')
    with (ROOT/'score/zensim-seed0.tsv').open('w') as out,(ROOT/'score/seed0.log').open('w') as log:
        subprocess.run([str(OUT/'bin/serve_custom_bake'),'--pairs',str(path),str(OUT/'models/seed0.bin')],env=base,stdout=out,stderr=log,check=True)
    scratch=ROOT/'scratch';scratch.mkdir(exist_ok=True)
    image='ghcr.io/imazen/zenfleet-worker:exec-cvvdp-0a61830d'
    imageid=subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()
    assert imageid=='sha256:c1a97ef66885697efe4af3f34466b8895350e21756b8e98194950018c2fac330'
    for metric in ['ssim2','butteraugli']:
        cmd=['docker','run','--rm','-u',f'{os.getuid()}:{os.getgid()}','--cpus','8','--memory','12g','--cpu-shares','256','-v',f'{ROOT}:{ROOT}','-v','/var/tmp/steercodec:/var/tmp/steercodec:ro','-e',f'TMPDIR={scratch}','--entrypoint','zenmetrics',image,'score-pairs','--metric',metric,'--pairs-tsv',str(path),'--out-parquet',str(ROOT/'score'/f'{metric}.parquet')]
        (ROOT/f'{metric}.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        with (ROOT/f'{metric}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('independent judge complete',metric,flush=True)
    owner=REPO/'benchmarks/steercodec_2026-10-04/steer_codec.py';spec=importlib.util.spec_from_file_location('steercodec_owner',owner);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=ROOT
    import argparse
    module.cmd_summary(argparse.Namespace(out=str(ROOT/'summary.json')))
    print('STEERCODEC complete; additional pass rule pending',flush=True)


if __name__=='__main__':main()
