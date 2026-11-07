"""Verify prepared TRAIN rows, feature bits, keys and registered label permutations."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from shaping_completion import encoded,sha
from shaping_variant_inputs import load_variant,verify_prepared


def same_table(actual,expected,exclude=()):
    if actual.schema!=expected.schema or actual.num_rows!=expected.num_rows:
        raise ValueError('table schema or row coverage differs')
    for name in actual.column_names:
        if name in exclude:continue
        a=actual[name].combine_chunks();b=expected[name].combine_chunks()
        missing=a.is_null().to_numpy(zero_copy_only=False)
        if not np.array_equal(missing,b.is_null().to_numpy(zero_copy_only=False)):
            raise ValueError('column validity differs: '+name)
        x=a.to_numpy(zero_copy_only=False)[~missing];y=b.to_numpy(zero_copy_only=False)[~missing]
        equal=x.dtype==y.dtype and (x.tobytes()==y.tobytes() if x.dtype.kind in 'biufc' else np.array_equal(x,y))
        if not equal:raise ValueError('column values or numeric bits differ: '+name)


def shuffled_indices(ids,pop,family,mode,seed):
    """Reconstruct the registered permutation without using the preparer's owner."""
    if mode not in ('true','population','conditional'):raise ValueError('unregistered label permutation')
    result=np.array(ids,copy=True)
    if mode=='true':return result
    rng=np.random.default_rng(seed)
    for population in sorted(set(pop[ids])):
        positions=np.flatnonzero(pop[ids]==population)
        groups=[positions] if mode=='population' else [positions[family[ids[positions]]==f] for f in sorted(set(family[ids[positions]]))]
        for group in groups:result[group]=rng.permutation(ids[group])
    return result


def audit(root,output):
    from shaped_linear_study import PREP,OLD
    progress=root/'VARIANT_INPUT_PROGRESS.json';status=json.loads(progress.read_text())
    if status['stage']!='prepared':raise ValueError('all registered input views must finish first')
    if output.exists():raise ValueError('preserve existing audit')
    plan=json.loads((root/'PLAN.json').read_text())
    for name,pin in plan['input_pins'].items():
        if sha(name)!=pin:raise ValueError('frozen input drift')
    original=Path(plan['original_plan']).parent
    pop=np.load(OLD/'data/pop.npy',allow_pickle=False)
    family=np.load(original/'data/family.npy',allow_pickle=False)
    labels=np.load(OLD/'data/y.npy',allow_pickle=False)
    source=json.loads(PREP.read_text());cache={};rows=[]
    for prepared in status['prepared_views']:
        path=Path(prepared['path'])
        if sha(path)!=prepared['sha256']:raise ValueError('prepared receipt drift')
        _,cell,identifier,view=load_variant(root,prepared['representative_cell'])
        receipt=verify_prepared(path,root,identifier,view,cell)
        train=np.asarray(view['train']);label_ids=shuffled_indices(train,pop,family,receipt['permutation'],receipt['permutation_seed'])
        expected_labels=dict(zip(map(int,train),labels[label_ids]))
        groups={g[0]:g for g in receipt['groups']}
        if len(groups)!=len(receipt['groups']):raise ValueError('duplicate population group')
        offset=0;seen=set();covered=[];changed=0
        for group in source['groups']:
            start=offset;offset+=group['rows']
            if group['split']!='fit':continue
            ids=train[(train>=start)&(train<offset)]
            if not len(ids):
                if group['name'] in groups:raise ValueError('absent population reintroduced')
                continue
            seen.add(group['name']);covered.extend(map(int,ids))
            name,table_path,weight,validation_weight,mode=groups[group['name']]
            if weight<=0 or validation_weight!=0:raise ValueError('training/evaluation role differs')
            if name not in cache:
                src=Path(group['path']);key=src.with_suffix('.keys.parquet')
                if sha(src)!=group['sha256'] or sha(key)!=group['keys_sha256']:raise ValueError('source table drift')
                cache[name]=(pq.read_table(src),pq.read_table(key))
            source_table,source_keys=cache[name]
            expected=source_table.take(ids-start);actual=pq.read_table(table_path)
            same_table(actual,expected,exclude=('human_score',))
            same_table(pq.read_table(Path(table_path).with_suffix('.keys.parquet')),source_keys.take(ids-start))
            values=actual['human_score'].to_numpy();wanted=np.array([expected_labels[int(i)] for i in ids],dtype=values.dtype)
            if not np.array_equal(values,wanted):raise ValueError('training label permutation differs')
            changed+=int(np.sum(values!=expected['human_score'].to_numpy()))
        if set(groups)!=seen or sorted(covered)!=sorted(view['train']):raise ValueError('training row coverage differs')
        if cell['stage']=='human_only' and ('cid22' in groups or np.any(pop[train]=='cid22_ssim2')):
            raise ValueError('teacher entered human-only preparation')
        rows.append(dict(path=str(path),receipt_sha256=sha(path),rows=len(train),changed_labels=changed,
                         stage=cell['stage'],permutation=receipt['permutation'],permutation_seed=receipt['permutation_seed'],
                         teacher_rows=int(np.sum(pop[train]=='cid22_ssim2'))))
        print(json.dumps(dict(verified_views=len(rows),total_views=len(status['prepared_views']))),flush=True)
    # Bind coverage independently to the registered matrix, not only the progress list.
    registered=json.loads((root/'VARIANT_MEMBERSHIPS.json').read_text())
    cells=json.loads((root/'MATRIX.json').read_text())
    expected={(registered['cells'][c['id']],c.get('permutation','true'),c.get('permutation_seed',0)) for c in cells if c['id'] in registered['cells']}
    actual=set()
    for row in rows:
        receipt=json.loads(Path(row['path']).read_text())
        actual.add((receipt['view_identifier'],row['permutation'],row['permutation_seed']))
    if actual!=expected or len(rows)!=len(expected):raise ValueError('registered view coverage differs')
    output.write_bytes(encoded(dict(plan_sha256=sha(root/'PLAN.json'),progress_sha256=sha(progress),
        driver_sha256=sha(__file__),views=rows,registered_variant_cells=len(registered['cells']),
        feature_bits_and_keys_unchanged=True,training_labels_match_registration=True,
        outer_responses_used=False,models_fitted=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.study,a.output)
