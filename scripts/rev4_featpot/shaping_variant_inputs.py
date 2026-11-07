"""Training-only admitted views for already registered variant memberships."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from shaping_completion import encoded,sha
from shaping_execution import load_registration


def load_variant(root,cell_id):
    plan,cells,_=load_registration(root)
    cell=next(c for c in cells if c['id']==cell_id)
    receipt=json.loads((root/'VARIANT_MEMBERSHIPS_RECEIPT.json').read_text())
    if receipt['plan_sha256']!=sha(root/'PLAN.json') or receipt['membership_sha256']!=sha(root/'VARIANT_MEMBERSHIPS.json'):
        raise ValueError('variant registration drift')
    policy=Path(__file__).resolve().parents[2]/'docs/SHAPING_VARIANT_MEMBERSHIP.md'
    if sha(policy)!=receipt['policy_sha256']:raise ValueError('variant policy drift')
    for path,pin in receipt['metadata_sha256'].items():
        if sha(path)!=pin:raise ValueError('membership metadata drift')
    data=json.loads((root/'VARIANT_MEMBERSHIPS.json').read_text());identifier=data['cells'][cell_id]
    view=data['views'][identifier]
    if __import__('hashlib').sha256(encoded(view)).hexdigest()!=identifier:raise ValueError('view identity differs')
    return plan,cell,identifier,view


def prepare(root,cell_id):
    from shaped_linear_study import OLD,PREP
    from shaped_linear_compare import prepare_neural
    plan,cell,identifier,view=load_variant(root,cell_id)
    mode=cell.get('permutation','true');seed=cell.get('permutation_seed',0)
    # Seed/model-independent preprocessing is shared; shuffle labels distinguish views.
    key=f"variant-{identifier[:20]}-{mode}-{seed}"
    dest=root/'comparators'/key
    if dest.exists():raise ValueError('preserve existing variant preparation')
    for path,pin in plan['input_pins'].items():
        if sha(path)!=pin:raise ValueError('frozen input drift')
    selected=set(view['train']);offset=0;absent=[]
    for group in json.loads(PREP.read_text())['groups']:
        included=len(selected&set(range(offset,offset+group['rows'])));offset+=group['rows']
        if group['split']=='fit' and included==0:absent.append(group['name'])
    data={'y':np.load(OLD/'data/y.npy',mmap_mode='r')}
    if mode!='true':
        original=Path(plan['original_plan']).parent
        data.update(pop=np.load(OLD/'data/pop.npy',mmap_mode='r'),family=np.load(original/'data/family.npy',mmap_mode='r'))
    request=dict(id=key,train=view['train'],test=view['test'],mode=mode,seed=seed,outer=int(cell['partition'].split('-')[1]))
    groups,decision,_=prepare_neural(root,request,data,plan,absent_fit_groups=absent)
    groups=[(n,str(p),tw,0.,m) for n,p,tw,_,m in groups if tw>0]
    if cell['stage']=='human_only' and any(n=='cid22' for n,_,_,_,_ in groups):raise ValueError('teacher entered human-only fit')
    files=[Path(decision),dest/'ADMISSION.json',dest/'admission/wide/main/real/receipt.json']
    for _,path,_,_,_ in groups:
        p=Path(path);files.extend((p,p.with_suffix('.keys.parquet'),Path(str(p)+'.manifest.json')))
    receipt=dict(groups=groups,decision=str(decision),view_identifier=identifier,permutation=mode,permutation_seed=seed,
                 explicit_absent_fit_groups=absent,view_sha256=sha(root/'VARIANT_MEMBERSHIPS.json'),
                 file_sha256={str(p):sha(p) for p in files},mse_weight=1 if any('both' in g[4] for g in groups) else 0,
                 exposed_development_passed_to_trainer=False,outer_results_read=False)
    (dest/'GROUPS.json').write_bytes(encoded(receipt));print(json.dumps(dict(path=str(dest/'GROUPS.json'),groups=[g[0] for g in groups],absent=absent,mse_weight=receipt['mse_weight'])),flush=True)
    return dest/'GROUPS.json'


def verify_prepared(path,root,identifier,view,cell):
    receipt=json.loads(path.read_text())
    expected=dict(view_identifier=identifier,permutation=cell.get('permutation','true'),
                  permutation_seed=cell.get('permutation_seed',0),
                  view_sha256=sha(root/'VARIANT_MEMBERSHIPS.json'),
                  exposed_development_passed_to_trainer=False,outer_results_read=False)
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('prepared variant identity differs')
    admission=json.loads(path.with_name('ADMISSION.json').read_text())
    for key in ('train','test'):
        if admission['cell'][key]!=view[key]:raise ValueError('prepared membership differs')
    for name,pin in receipt['file_sha256'].items():
        if sha(name)!=pin:raise ValueError('prepared variant input drift')
    return receipt


def prepare_all(root):
    from shaping_storage import reservation
    plan,cells,_=load_registration(root)
    selected=[c for c in cells if c['stage'] in ('source_curve','row_curve','human_only','shuffle')]
    seen=set();records=[]
    for cell in selected:
        _,_,identifier,view=load_variant(root,cell['id'])
        key=f"variant-{identifier[:20]}-{cell.get('permutation','true')}-{cell.get('permutation_seed',0)}"
        if key in seen:continue
        seen.add(key);dest=root/'comparators'/key;path=dest/'GROUPS.json'
        with reservation(root,dest,plan['minimum_ssd_free_bytes']):
            if not path.exists():path=prepare(root,cell['id'])
            verify_prepared(path,root,identifier,view,cell)
        records.append(dict(path=str(path),sha256=sha(path),representative_cell=cell['id']))
        status=dict(prepared_views=records,registered_variant_cells=len(selected),
                    stage='preparing',outer_results_read=False,models_fitted=0)
        (root/'VARIANT_INPUT_PROGRESS.json').write_bytes(encoded(status))
        print(json.dumps(dict(prepared_views=len(records),path=str(path))),flush=True)
    status['stage']='prepared'
    (root/'VARIANT_INPUT_PROGRESS.json').write_bytes(encoded(status))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--cell');mode.add_argument('--all',action='store_true')
    args=p.parse_args()
    if args.all:prepare_all(args.study)
    else:prepare(args.study,args.cell)
