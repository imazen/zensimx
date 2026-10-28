#!/usr/bin/env python3
"""Compare EVIF's pinned judge on the four stored SafeSyn controls."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image
from chroma_pilot import save,sha
root=Path(sys.argv[1]);base=root.parent
os.environ.update(ZENSIM_FORMULA_REV='3',ZENSIM_ROOT_FORM='sqrt')
spec=json.loads((base/'pilot-run04/SOURCES.json').read_text());out=root/'judge-replay';out.mkdir()
tools=root/'bin/chroma_tools';records=[]
for i,c in enumerate(spec['controls']):
    ref=base/'raw'/c['raw'];original=base/'raw'/c['dist_raw'];png=out/f'control{i}.png'
    subprocess.run([str(tools),'decode-jpeg',str(original),str(png)],check=True)
    scores=json.loads(subprocess.check_output([str(tools),'ssim',str(ref),str(png)],text=True))
    with Image.open(png) as im:actual_hash=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
    records.append(dict(reference=str(ref),distorted=str(png),original_distorted=str(original),expected_safesyn_score=c['expected_ssim2'],expected_distorted_pixels_sha256=c['distorted_pixels_sha256'],actual_distorted_pixels_sha256=actual_hash,evif_judge=scores[0]))
pairs=out/'pairs.tsv'
with pairs.open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['ref_path','dist_path','human_score'])
    for r in records:w.writerow([r['reference'],r['distorted'],0])
with (out/'teacher.log').open('w') as f:subprocess.run([str(base/'bin/teacher'),'--corpus','pairs-tsv','--path',str(pairs),'--out',str(out/'teacher.csv'),'--full-944','--audit-jsonl',str(out/'teacher.jsonl'),'--audit-ssim2'],stdout=f,stderr=subprocess.STDOUT,check=True)
audit={r['distorted']:r for r in map(json.loads,(out/'teacher.jsonl').read_text().splitlines())}
for r in records:
    r['frozen_teacher_on_same_pixels']=audit[r['distorted']]['peer_ssim2']['score']
    r['same_metric_value']=r['frozen_teacher_on_same_pixels']==r['evif_judge']['value']
    r['original_decode_pixels_match']=r['expected_distorted_pixels_sha256']==r['actual_distorted_pixels_sha256']
    r['safesyn_replay_exact']=r['original_decode_pixels_match'] and r['evif_judge']['value']==r['expected_safesyn_score']
save(out/'REPLAY.json',dict(controls=records,exact_replays=sum(r['safesyn_replay_exact'] for r in records),required=4,frozen_teacher_sha256=sha(base/'bin/teacher'),diagnostic_only=True))
print(json.dumps(records,indent=2),flush=True)
