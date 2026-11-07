"""Metadata-only membership expansion; never reads scores or final holdouts."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from shaping_completion import encoded,sha,validate_partition
from shaping_execution import load_registration


def digest_order(seed,value):return hashlib.sha256(f'{seed}|{value}'.encode()).digest()


def membership(cell,base,component,pop):
    original=sorted(base['train']);train=original.copy();stage=cell['stage']
    if stage=='source_curve':
        sources=sorted(set(component[train]),key=lambda name:(digest_order(cell['order_seed'],name),str(name)))
        n=math.floor(len(sources)*cell['fraction'])
        if not 0<n<=len(sources):raise ValueError('empty or oversized source budget')
        selected=set(sources[:n]);train=[i for i in train if component[i] in selected]
    elif stage=='row_curve':
        target=math.floor(len(train)*cell['fraction'])
        order=sorted(train,key=lambda i:(digest_order(cell['order_seed'],i),i))
        representatives={}
        for i in order:representatives.setdefault(component[i],i)
        if target<len(representatives):raise ValueError('row budget cannot retain all training sources')
        mandatory=set(representatives.values())
        train=sorted(mandatory)
        # Fill the exact remaining budget in one fixed hash order, preserving nesting.
        chosen=set(train)
        for i in order:
            if len(chosen)==target:break
            chosen.add(i)
        train=sorted(chosen)
    elif stage=='human_only':train=[i for i in train if pop[i]!='cid22_ssim2']
    elif stage!='shuffle':raise ValueError('unsupported variant stage')
    if not train or not set(train)<=set(original):raise ValueError('invalid training subset')
    if set(component[train])&set(component[base['test']]):raise ValueError('source leakage')
    return dict(train=train,test=base['test'],base_partition=cell['partition'],stage=stage,
                training_rows=len(train),training_sources=len(set(component[train])),
                base_training_rows=len(original),base_training_sources=len(set(component[original])),
                training_population_rows={str(p):int(np.sum(pop[train]==p)) for p in sorted(set(pop[original]))},
                absent_training_populations=sorted(str(p) for p in set(pop[original])-set(pop[train])),
                labels_permuted=stage=='shuffle',teacher_training_rows=int(np.sum(pop[train]=='cid22_ssim2')))


def register(args):
    plan,cells,partitions=load_registration(args.study)
    original=Path(plan['original_plan']).parent
    paths={'component':original/'data/component.npy','pop':Path('/mnt/v/output/zensim/tree-study-20261106/search/data/pop.npy')}
    for path in paths.values():
        if sha(path)!=plan['input_pins'][str(path)]:raise ValueError('metadata drift')
    arrays={k:np.load(p,allow_pickle=False) for k,p in paths.items()}
    selected=[c for c in cells if c['stage'] in ('source_curve','row_curve','human_only','shuffle')]
    dest=args.study/'VARIANT_MEMBERSHIPS.json'
    if dest.exists():raise ValueError('preserve registered memberships')
    views={};mapping={};cache={}
    dev=partitions['full-fit']['descriptive_development']
    for c in selected:
        key=(c['stage'],c['partition'],c.get('fraction'),c.get('order_seed'))
        if key not in cache:
            d=membership(c,partitions[c['partition']],**arrays)
            validate_partition(arrays['component'],d['train'],d['test'],dev)
            identifier=hashlib.sha256(encoded(d)).hexdigest()
            views[identifier]=d;cache[key]=identifier
        mapping[c['id']]=cache[key]
    dest.write_bytes(encoded(dict(schema='shaping-variant-memberships-v1',cells=mapping,views=views)))
    policy=Path(__file__).resolve().parents[2]/'docs/SHAPING_VARIANT_MEMBERSHIP.md'
    receipt=dict(plan_sha256=sha(args.study/'PLAN.json'),membership_sha256=sha(dest),driver_sha256=sha(__file__),
                 policy_sha256=sha(policy),metadata_sha256={str(p):sha(p) for p in paths.values()},cells=len(mapping),unique_views=len(views),
                 scores_read=False,outer_results_read=False)
    (args.study/'VARIANT_MEMBERSHIPS_RECEIPT.json').write_bytes(encoded(receipt))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    register(p.parse_args())
