#!/usr/bin/env python3
"""Qualify every prepared PNG through the frozen teacher's exact RGB projection."""
import json
from pathlib import Path
import subprocess
import sys
from chroma_pilot import save,sha
nas=Path('/mnt/tower/output/zensim/byc-chroma-20261028');root=nas/'b/full';binary=Path('/var/tmp/byc-chroma-20261028/b/bin/chroma-frozen-teacher')
assert sha(binary)=='2e6d5f77f54062c9a08c6390668eb075474ff8772926720b5475d8ca179eb3a6'
views=json.loads((root/'VIEWS.json').read_text())
pairs=''.join(json.dumps(dict(reference=v['path'],distorted=v['path']))+'\n' for v in views)
with (root/'frozen-reference-scores.jsonl').open('x') as stdout,(root/'frozen-reference.log').open('w') as stderr:subprocess.run([str(binary)],input=pairs,text=True,stdout=stdout,stderr=stderr,check=True)
actual=list(map(json.loads,(root/'frozen-reference-scores.jsonl').read_text().splitlines()));assert len(actual)==len(views)
rows=[]
for v,r in zip(views,actual):
    assert r['reference']==v['path']==r['distorted'];assert r['reference_pixels_sha256']==r['distorted_pixels_sha256']
    assert r['reference_pixels_sha256']==v['pixels_sha256'],'independent source hash differs from frozen teacher projection'
    rows.append(dict(id=v['id'],path=v['path'],sha256=v['sha256'],reference_pixels_sha256=r['reference_pixels_sha256']))
save(root/'FROZEN_REFERENCE_PIXELS.json',rows);save(root/'REFERENCE_REPLAY.json',dict(passed=True,views=len(rows),frozen_helper_sha256=sha(binary),checks=['every frozen PNG decode projects to independently checked source RGB8 pixels','all prepared source byte identities bound before declaration']))
print(json.dumps(dict(passed=True,views=len(rows))),flush=True)
