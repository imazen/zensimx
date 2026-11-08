"""Publish scalar scatter inputs from pinned, completed inner reports only."""
import argparse,json,math,os
from pathlib import Path
from functools import lru_cache
import numpy as np
from shaping_completion import MAX_PAIR_FAMILIES,encoded,sha
from shaping_execution import load_registration

MAX_BYTES=2*1024**3
POPULATIONS=('kadid','tid2013','konfig','cid22_a25','cid22_ssim2')


def write_atomic(path,value):
    temporary=path.with_suffix('.writing');temporary.write_text(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n');temporary.replace(path)


def report_path(study,record):
    cell=record['cell'];repaired=cell['family'] in MAX_PAIR_FAMILIES
    root=study/('reexport_max_pairs' if repaired else 'runs')/cell['id']
    markers=list(root.glob('attempt-*/final/COMPLETE.json' if repaired else 'attempt-*/evaluation/COMPLETE.json'))
    if len(markers)!=1:raise ValueError('one completed scatter representation required')
    return markers[0].parent/'REPORT.json',markers[0]


@lru_cache(maxsize=4096)
def archived_inventory(root):
    from shaping_checkpoint_retention import archive_inventory
    return archive_inventory(Path(root))[1]


def prediction_pin(study,path,report,pin,cell):
    predicted=path.parent/'predictions.npz';actual=sha(predicted)
    if report.get('predictions_sha256') is not None:
        if report['predictions_sha256']!=actual:raise ValueError('saved prediction evidence differs')
        return actual
    # Historical reports predate the inline prediction pin. Their completed
    # archive binds both REPORT and NPZ; retain that original report unchanged.
    candidates=[]
    for parent in path.parents:
        if parent==study.parent:break
        if (parent/'MIRROR_VERIFIED.json').exists():candidates.append((parent,path.relative_to(parent)))
    if cell['family'] in ('extra_trees','shaped_additive'):
        name='tree-inner-archive' if cell['family']=='extra_trees' else 'linear-inner-archive'
        candidates.append((study.parent/name,path.relative_to(study)))
    for root,relative in candidates:
        if not (root/'MIRROR_VERIFIED.json').exists():continue
        inventory=archived_inventory(str(root));npz=relative.with_name('predictions.npz').as_posix()
        if relative.as_posix() not in inventory or npz not in inventory:continue
        if inventory[relative.as_posix()]!=pin or inventory[npz]!=actual:
            raise ValueError('historical scatter archive evidence differs')
        return actual
    return None


def prepare(study,dashboard,output):
    plan,cells,partitions=load_registration(study);registered={c['id']:c for c in cells}
    snapshot=json.loads((dashboard/'stats.json').read_text());plan_pin=sha(study/'PLAN.json')
    if snapshot['plan_sha256']!=plan_pin:raise ValueError('statistics snapshot registration differs')
    from shaped_linear_study import OLD,mapped_labels
    original=Path(plan['original_plan']).parent
    paths={**{k:OLD/'data'/f'{k}.npy' for k in ('y','pop')},
           **{k:original/'data'/f'{k}.npy' for k in ('component','family')}}
    for path in paths.values():
        if sha(path)!=plan['input_pins'].get(str(path)):raise ValueError('frozen scatter labels differ')
    arrays={k:np.load(path,allow_pickle=False,mmap_mode='r') for k,path in paths.items()}
    output.mkdir(parents=True,exist_ok=True);points=output/'points';points.mkdir(exist_ok=True)
    rows={};jobs=[];pending=[];total=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    records=[r for r in snapshot['records'] if r.get('panel')]
    # The default partition/setting gets detailed native panels first; all other
    # completed settings and paired seeds remain available in the same index.
    records.sort(key=lambda r:(r['cell']['partition']!=snapshot['default_partition'],r['cell']['setting']!=0,
                               r['cell']['seed'] not in (0,None),r['cell']['family'],r['cell']['id']))
    for record in records:
        cell=record['cell'];cid=cell['id']
        if cell!=registered.get(cid) or cell['stage'] not in ('inner','fresh_control_inner','rescue'):
            raise ValueError('scatter candidate is outside registered inner scope')
        path,marker=report_path(study,record);pin=sha(path);report=json.loads(path.read_text())
        if (json.loads(marker.read_text())['report_sha256']!=pin or report['cell']!=cell
            or report['plan_sha256']!=plan_pin or report['outer_results_read'] is not False):
            raise ValueError('completed inner scatter report differs')
        destination=points/(pin+'.json');url='details/points/'+destination.name
        if not destination.exists():
            predictions=path.parent/'predictions.npz'
            predicted_pin=prediction_pin(study,path,report,pin,cell)
            if predicted_pin is None:
                pending.append(dict(cell=cid,reason='awaiting archived prediction provenance'));continue
            membership=partitions[cell['partition']]
            expected=sorted(set(membership['train'])|set(membership['test']))
            with np.load(predictions,allow_pickle=False) as table:
                if set(table.files)!={'original_indices','native'}:raise ValueError('scatter prediction columns differ')
                ids=table['original_indices'];native=table['native']
            if (ids.tolist()!=expected or native.shape!=ids.shape or not np.isfinite(native).all()):
                raise ValueError('scatter membership, finite coverage or alignment differs')
            d={k:a[ids] for k,a in arrays.items()};mapped=mapped_labels(d['y'],d['pop'],report['maps'])*100
            positions={int(row):i for i,row in enumerate(ids)};populations={}
            for pop in POPULATIONS:
                roles={}
                for role,key,report_key in (('train','train','train'),('validation','test','inner_validation')):
                    selected=np.array([positions[row] for row in membership[key] if d['pop'][positions[row]]==pop],dtype=np.int64)
                    sources=sorted(set(d['component'][selected]));families=sorted(set(d['family'][selected]))
                    source_codes={str(v):i for i,v in enumerate(sources)};family_codes={str(v):i for i,v in enumerate(families)}
                    summary=report[report_key]['summary'][pop]
                    if len(selected)!=summary['rows']:raise ValueError('scatter row denominator differs')
                    roles[role]=dict(summary=summary,panels=[p for p in report[report_key]['panels'] if p['label'].startswith(pop+'|')],
                        indices=ids[selected].tolist(),predicted=native[selected].tolist(),target=d['y'][selected].tolist(),
                        scaled_target=mapped[selected].tolist(),sources=list(map(str,sources)),families=list(map(str,families)),
                        source=[source_codes[str(v)] for v in d['component'][selected]],family=[family_codes[str(v)] for v in d['family'][selected]])
                populations[pop]=dict(teacher=pop=='cid22_ssim2',roles=roles)
            value=dict(schema='zensim-inner-scatter-v1',cell=cell,plan_sha256=plan_pin,report_sha256=pin,
                source_report='../'+path.relative_to(study.parent).as_posix(),predictions_sha256=predicted_pin,
                representation=record['panel']['representation'],populations=populations,
                feature_gate=report.get('feature_gate'),active_output_coefficients=report.get('active_output_coefficients'),
                native_panel_url='details/native/'+pin+'.json',outer_results_read=False,
                target_scaling='Existing TRAIN-only population range mapping; held-apart values are not clipped.',
                protocol_summary='Inner source-disjoint validation; no final holdout, no new inference, no pixel maps.')
            payload=json.dumps(value,allow_nan=False,separators=(',',':')).encode()+b'\n'
            if total+len(payload)>MAX_BYTES:raise ValueError('scatter publication exceeds its 2-GiB disk budget')
            with destination.open('xb') as stream:stream.write(payload)
            total+=len(payload)
        else:
            cached=json.loads(destination.read_text())
            if cached['report_sha256']!=pin or cached['cell']!=cell or cached['outer_results_read']:
                raise ValueError('preserve changed scatter publication')
        rows[cid]=dict(url=url,report_sha256=pin,native_url='details/native/'+pin+'.json')
        jobs.append(dict(cell=cell,path=str(destination),sha256=sha(destination),report_sha256=pin))
    index=dict(schema='zensim-inner-scatter-index-v1',plan_sha256=plan_pin,cells=rows,pending=pending,outer_results_read=False)
    write_atomic(output/'INDEX.json',index)
    manifest=dict(schema='zensim-native-dashboard-jobs-v1',study=str(study),jobs=jobs,
                  source_snapshot_sha256=sha(dashboard/'stats.json'),source_driver_sha256=sha(__file__),outer_results_read=False)
    write_atomic(output/'NATIVE_JOBS.json',manifest)
    return dict(scatter_reports=len(rows),pending=len(pending),published_bytes=total,budget_bytes=MAX_BYTES,outer_results_read=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','dashboard','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.study,a.dashboard,a.output)),flush=True)
