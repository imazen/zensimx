"""Reuse only a completed, matching identity-joint training prefix."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
from shaping_completion import encoded, sha


def matching_parent(child, cells):
    if child['stage']!='inner' or '_pretrained_' not in child['family']:
        raise ValueError('only registered inner pretrained cells supported')
    family=child['family'].split('_pretrained_')[0]+'_joint_identity'
    found=[c for c in cells if c['stage']=='inner' and c['family']==family
           and all(c[k]==child[k] for k in ('partition','setting','seed'))]
    if len(found)!=1:raise ValueError('missing or ambiguous matching prefix')
    return found[0]


def completed_parent(root,child,cells):
    parent=matching_parent(child,cells)
    candidates=[]
    for path in (root/'runs'/parent['id']).glob('attempt-*/RESULT.json'):
        result=json.loads(path.read_text())
        if result['stage']=='trained' and result['exit_code']==0:candidates.append((path.parent,result))
    if len(candidates)!=1:raise ValueError('requires one completed matching parent; never reuse mutable checkpoints')
    source,result=candidates[0]
    return parent,source,result


def extract(root,binary_dir,child,cells,destination):
    parent,source,result=completed_parent(root,child,cells)
    run=json.loads((source/'RUN.json').read_text())
    if run['cell']!=parent or run['epochs']!=240 or run['pairs_per_epoch']!=50000:
        raise ValueError('parent exposure or registered identity differs')
    if run['plan_sha256']!=sha(root/'PLAN.json') or run['build_sha256']!=sha(binary_dir/'BUILD.json'):
        raise ValueError('parent plan/binary differs')
    if sha(source/'checkpoints/ckpt_epoch239.bin')!=result['bake_sha256']:
        raise ValueError('completed parent bake drift')
    prefix=source/'checkpoints/ckpt_epoch119.bin'
    def meta(key):
        return subprocess.check_output([str(binary_dir/'bake_dial_refit'),'read-meta','--in',str(prefix),'--key',key],text=True).strip()
    if meta('zentrain.input_shaping_initial_sha256')!=sha(source/'SHAPING.json'):
        raise ValueError('prefix initial shapes differ')
    config=json.loads(meta('zentrain.input_shaping'))
    if config['schema']!='zensim-input-shaping-v1' or not config['trainable']:
        raise ValueError('prefix is not jointly learned shaping')
    initial=json.loads((source/'SHAPING.json').read_text())
    structural=lambda c:[{k:v for k,v in t.items() if k not in ('slopes','gate')} for t in c['terms']]
    if structural(config)!=structural(initial):raise ValueError('prefix shape layout or knots differ')
    config['trainable']=child['family'].endswith('_joint')
    destination.write_bytes(encoded(config))
    receipt=dict(parent_cell=parent,source_run=str(source),prefix_sha256=sha(prefix),
                 parent_result_sha256=sha(source/'RESULT.json'),manifest_sha256=sha(destination),
                 pretraining_epochs=120,pretraining_pair_draws=6000000,fresh_head_epochs=120,
                 schedule='50-epoch cosine restart; independent of total epochs',
                 scaler='same training-complement raw scale-only owner; shaping applied after scaling statistics')
    destination.with_suffix('.provenance.json').write_bytes(encoded(receipt))
    return receipt
