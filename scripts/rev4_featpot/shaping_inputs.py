"""Training-complement-only initialization for native input-shaping manifests."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from shaping_completion import encoded, sha


def knots(values):
    values=np.asarray(values,dtype=np.float32)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('nonfinite or negative damage input')
    positive=values[values>0]
    a,b=np.quantile(positive,[.5,.9]).astype(np.float32) if len(positive) else (np.float32(1),np.float32(2))
    if b<=a:b=np.float32(2)*a
    if not (0<a<b and np.isfinite(b)):raise ValueError('unrepresentable knot widening')
    return [float(a),float(b)]


def initial_config(raw,raw_ids,train,declaration,base,separate):
    """Each logical input owns its own g; product gates are independent too."""
    train=np.asarray(train,dtype=np.int64)
    if len(train)==0 or len(set(train.tolist()))!=len(train):raise ValueError('empty or duplicate training selection')
    positions={int(v):i for i,v in enumerate(raw_ids)}
    if len(positions)!=len(raw_ids):raise ValueError('duplicate raw field ids')
    def column(identifier):return np.asarray(raw[train,positions[identifier]],dtype=np.float32)
    terms=[]
    for d in declaration['direct']:
        terms.append(dict(column=d,damage=d,reference=None,separate=False,
                          knots=knots(column(d)),slopes=[1.,1.,1.],gate=[0.,.5,1.]))
    for j,(d,f) in enumerate(declaration['products']):
        damage,reference=column(d),column(f)
        if not np.isfinite(reference).all() or (reference<0).any():raise ValueError('invalid reference domain')
        if separate and (reference>1).any():raise ValueError('separate fragility exceeds one')
        values=damage if separate else np.multiply(damage,reference,dtype=np.float32)
        terms.append(dict(column=base+j,damage=d,reference=f,separate=separate,
                          knots=knots(values),slopes=[1.,1.,1.],gate=[0.,.5,1.]))
    return dict(schema='zensim-input-shaping-v1',trainable=True,terms=terms)


def prepare(root,partition,membership,plan,separate):
    from tree_study import RAW_IDS
    from shaped_linear_study import OLD
    from shaping_completion import REPO
    kind='separate' if separate else 'product'
    dest=root/'shapes'/partition/kind
    selected_sha=hashlib.sha256(encoded(membership['train'])).hexdigest()
    if dest.exists():
        receipt=json.loads((dest/'RECEIPT.json').read_text())
        if receipt['training_indices_sha256']!=selected_sha or sha(dest/'INITIAL.json')!=receipt['manifest_sha256']:
            raise ValueError('shaping initialization cache drift')
        return dest/'INITIAL.json'
    raw_path=OLD/'data/raw.npy'
    if sha(raw_path)!=plan['input_pins'][str(raw_path)]:raise ValueError('raw feature drift')
    raw=np.load(raw_path,mmap_mode='r',allow_pickle=False)
    declaration_path=REPO/'scripts/rev4_featpot/color_noise_fx2_declaration.json'
    declaration=json.loads(declaration_path.read_text())
    config=initial_config(raw,RAW_IDS,membership['train'],declaration,1853,separate)
    dest.mkdir(parents=True)
    path=dest/'INITIAL.json';path.write_bytes(encoded(config))
    receipt=dict(schema='shaping-initialization-v1',partition=partition,kind=kind,
                 training_indices_sha256=selected_sha,training_rows=len(membership['train']),
                 raw_sha256=sha(raw_path),declaration_sha256=sha(declaration_path),
                 driver_sha256=sha(__file__),plan_sha256=sha(root/'PLAN.json'),
                 manifest_sha256=sha(path),knots_fit_on='training complement only; no labels',
                 shapes_independent_per_logical_input=True,fragility_monotonicity_claim=False,
                 independent_trainable_shape_parameters=sum(6 if t['separate'] else 3 for t in config['terms']))
    (dest/'RECEIPT.json').write_bytes(encoded(receipt))
    return path


def main():
    import argparse
    from shaping_execution import load_registration
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--partition',required=True)
    args=p.parse_args()
    plan,_,partitions=load_registration(args.study)
    if args.partition not in partitions:raise ValueError('unregistered partition')
    for separate in (False,True):
        path=prepare(args.study,args.partition,partitions[args.partition],plan,separate)
        print(json.dumps({'manifest':str(path),'sha256':sha(path)}),flush=True)


if __name__=='__main__':main()
