#!/usr/bin/env python3
"""Owner-authorized nearest-feature substitute; never claims cluster membership."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from chroma_pilot import save,sha
root=Path(sys.argv[1]);base=root.parent
spec=importlib.util.spec_from_file_location('owner',root/'owner.py');owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
Z,paths,crops,classes,widths,heights,ctx=owner.load(root/'imazen26_features_2026-06-13.parquet',0)
np=owner.np
cls='2400-unsplash-textures';target_id=2402;target_crop='c50_bl'
units=[i for i,c in enumerate(classes) if c==cls]
origins=sorted({owner.leading_id(paths[i]) for i in units})
logs=root/'substitution';logs.mkdir(exist_ok=True)
for row in json.loads((root/'ADDED_SOURCE_PROVENANCE.json').read_text()):
    if sha(row['raw'])!=row['lfs_sha256']:raise ValueError('canonical LFS identity mismatch')
eligible=[];guards=[]
for identity in origins:
    raw=base/'raw'/f'im{identity}.png'
    if not raw.exists():raise ValueError(f'canonical TRAIN source must be staged: {identity}')
    dest=logs/f'im{identity}-guard.png'
    with (logs/f'im{identity}-guard.log').open('w') as f:
        code=subprocess.run([str(base/'bin/prepare'),'resize','--in',str(raw),'--out',str(dest),'--max','32'],stdout=f,stderr=subprocess.STDOUT).returncode
    guards.append(dict(origin=identity,raw=str(raw),sha256=sha(raw),exit_code=code))
    if code==0 and identity!=target_id:eligible.append(identity)
targets=[i for i in units if owner.leading_id(paths[i])==target_id and crops[i]==target_crop]
if targets:
    if len(targets)!=1:raise ValueError('target selection feature row is ambiguous')
    center=Z[targets[0]];method='nearest to im2402/c50_bl selection-feature row'
else:
    center=Z[units].mean(0);method='nearest to TRAIN content-class centroid; target feature row absent'
candidates=[]
for i in units:
    identity=owner.leading_id(paths[i])
    if identity in eligible:
        candidates.append(dict(origin=identity,crop_label=crops[i],image_path=paths[i],distance=float(np.linalg.norm(Z[i]-center)),width=widths[i],height=heights[i]))
candidates.sort(key=lambda r:(r['distance'],r['origin'],r['crop_label'],r['image_path']))
if not candidates:raise ValueError('no eligible same-class TRAIN source')
replay=json.loads((root/'CLUSTER_REPLAY.json').read_text())
result=dict(method=method,certified_cluster_mate=False,target_origin=target_id,target_crop=target_crop,content_class=cls,selected=candidates[0],candidates=candidates,decoder_guards=guards,feature_parquet_sha256=sha(root/'imazen26_features_2026-06-13.parquet'),selection_owner_sha256=sha(root/'owner.py'),content_feature_count=Z.shape[1],replay_matching_representatives=replay['matching_reps'],replay_total_representatives=replay['expected_reps'],geometry_exclusion=list(owner.GEOM),train_parity=0)
save(root/'SUBSTITUTION.json',result);print(json.dumps(result,indent=2),flush=True)
