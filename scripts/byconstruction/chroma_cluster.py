#!/usr/bin/env python3
"""Replay the existing K500_even selection owner before ranking substitutes."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('owner',root/'owner.py');owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
Z,paths,crops,classes,widths,heights,ctx=owner.load(root/'imazen26_features_2026-06-13.parquet',0)
km=owner.KMeans(n_clusters=500,n_init=10,random_state=0).fit(Z)
np=owner.np
original=list(csv.DictReader((root/'imazen26_representatives_K500_even_2026-06-18.tsv').open(),delimiter='\t'))
def key(path,crop):return (Path(path).name,crop)
expected={key(r.get('image_path',r.get('url','')),r['crop_label']):r for r in original}
actual={}; members={}
for c in range(500):
 idx=np.where(km.labels_==c)[0];d=np.linalg.norm(Z[idx]-km.cluster_centers_[c],axis=1)
 ranked=idx[np.argsort(d,kind='stable')]
 k=key(paths[int(ranked[0])],crops[int(ranked[0])]);actual[k]=c
 members[c]=[dict(image_path=paths[int(i)],crop_label=crops[int(i)],content_class=classes[int(i)],width=widths[int(i)],height=heights[int(i)],distance=float(np.linalg.norm(Z[int(i)]-km.cluster_centers_[c]))) for i in ranked]
missing=set(expected)-set(actual); extra=set(actual)-set(expected)
result=dict(expected_reps=len(expected),replayed_reps=len(actual),matching_reps=len(set(expected)&set(actual)),missing=sorted(missing),extra=sorted(extra),numpy=np.__version__,sklearn=__import__('sklearn').__version__)
for k,r in expected.items():
 if k[0].startswith('2402_'):
  result['target']=r
  if k in actual:result['ranked_members']=members[actual[k]]
(root/'CLUSTER_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
if missing or extra:raise ValueError('K500 representative replay differs; do not silently substitute a new cluster')
