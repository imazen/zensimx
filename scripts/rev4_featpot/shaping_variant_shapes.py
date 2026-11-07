"""Initialize shapes independently on each registered generalization TRAIN view."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_variant_inputs import load_variant,verify_prepared
from shaping_inputs import prepare
from shaping_storage import reservation


def initialize(root):
    output=root/'VARIANT_SHAPES.json'
    if output.exists():raise ValueError('preserve shape registration')
    audit_path=root/'VARIANT_INPUT_AUDIT.json';audit=json.loads(audit_path.read_text())
    progress=root/'VARIANT_INPUT_PROGRESS.json'
    if audit['plan_sha256']!=sha(root/'PLAN.json') or audit['progress_sha256']!=sha(progress):
        raise ValueError('input audit identity drift')
    if not audit['feature_bits_and_keys_unchanged'] or not audit['training_labels_match_registration']:
        raise ValueError('input audit not qualified')
    records=json.loads(progress.read_text())['prepared_views'];seen={};bindings={}
    for record in records:
        plan,cell,identifier,view=load_variant(root,record['representative_cell'])
        group_path=Path(record['path'])
        if sha(group_path)!=record['sha256']:raise ValueError('prepared groups drift')
        verify_prepared(group_path,root,identifier,view,cell)
        if identifier in seen:continue
        partition='variant-'+identifier[:20]
        with reservation(root,root/'shapes'/partition,plan['minimum_ssd_free_bytes']):
            paths=[prepare(root,partition,view,plan,separate) for separate in (False,True)]
        seen[identifier]=dict(training_rows=len(view['train']),teacher_training_rows=view['teacher_training_rows'],
            manifests={p.parent.name:dict(path=str(p),sha256=sha(p),receipt_sha256=sha(p.with_name('RECEIPT.json'))) for p in paths})
        print(json.dumps(dict(prepared_views=len(seen),view_identifier=identifier)),flush=True)
    membership=json.loads((root/'VARIANT_MEMBERSHIPS.json').read_text())
    if set(seen)!=set(membership['views']):raise ValueError('shape view coverage differs')
    bindings=membership['cells']
    output.write_bytes(encoded(dict(plan_sha256=sha(root/'PLAN.json'),audit_sha256=sha(audit_path),
        membership_sha256=sha(root/'VARIANT_MEMBERSHIPS.json'),driver_sha256=sha(__file__),
        views=seen,cells=bindings,learned_shapes_reused=False,supervised_pretraining_fits=0,
        label_independent_initializations_only=True,outer_responses_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    initialize(p.parse_args().study)
