"""Reconcile primary-fit coverage against existing full-readback archive manifests."""
import argparse,json
from pathlib import Path
from shaping_completion import NEURAL,encoded,sha
from shaping_artifact_retention import archive_inventory


def evidence(study,cell):
    candidates=[]
    for path in sorted((study/'runs'/cell['id']).glob('attempt-*/RESULT.json')):
        result=json.loads(path.read_text())
        if result.get('exit_code')==0 and result.get('stage') in ('trained','complete'):candidates.append((path.parent,result))
    if len(candidates)!=1:raise ValueError('one successful primary attempt required: '+cell['id'])
    run,result=candidates[0];identity=json.loads((run/'RUN.json').read_text())
    if (identity['cell']!=cell or result['cell']!=cell or identity['plan_sha256']!=sha(study/'PLAN.json')
        or result['plan_sha256']!=identity['plan_sha256']
        or result['stage']!=('trained' if cell['family'] in NEURAL else 'complete')):
        raise ValueError('primary archive scientific identity differs')
    pins={str(p):sha(p) for p in (run/'RUN.json',run/'RESULT.json')}
    if cell['family'] in NEURAL:
        model=run/identity.get('fit_directory','.')/'checkpoints'/f"ckpt_epoch{result['epochs']-1:03}.bin"
        pins[str(model)]=result['bake_sha256']
    else:
        report=run/'evaluation/REPORT.json';marker=report.with_name('COMPLETE.json');value=json.loads(report.read_text())
        if (value['cell']!=cell or value['plan_sha256']!=identity['plan_sha256']
            or json.loads(marker.read_text())['report_sha256']!=sha(report)):
            raise ValueError('primary control report identity differs')
        q=value['qualification'];tree=cell['family']=='extra_trees'
        pins[str(run/('model.opaque-vector.zcth' if tree else 'native/model.bin'))]=q['native_tree_gate']['model_sha256'] if tree else q['model_sha256']
        pins.update({str(p):sha(p) for p in (report,marker)})
    return run,pins


def coverage(study,cells,batches):
    """Read metadata and manifests; do not repeat full model/payload readbacks."""
    primary=[c for c in cells if c['stage']=='inner']
    if len({c['id'] for c in primary})!=len(primary):raise ValueError('duplicate primary registration')
    cached={}
    def archive(root):
        if root not in cached:
            mirror,pins=archive_inventory(root)
            if root.resolve()==mirror.resolve():raise ValueError('distinct durable archive required')
            cached[root]=dict(root=str(root),destination=str(mirror),manifest_sha256=sha(root/'SHA256SUMS'),
                receipt_sha256=sha(root/'MIRROR_VERIFIED.json'),pins=pins)
        return cached[root]
    for root in batches:archive(root)
    rows=[];missing=[]
    for cell in primary:
        run,pins=evidence(study,cell);directory=run.parent;matches=[]
        candidates=[(b,study,'batch') for b in batches]
        if (directory/'MIRROR_VERIFIED.json').exists():candidates.insert(0,(directory,directory,'individual'))
        for root,relative_to,kind in candidates:
            value=archive(root);expected={Path(p).relative_to(relative_to).as_posix():pin for p,pin in pins.items()}
            if not any(p in value['pins'] for p in expected):continue
            if any(value['pins'].get(p)!=pin for p,pin in expected.items()):
                raise ValueError('archive does not bind the successful primary fit: '+cell['id'])
            matches.append(dict(archive_root=str(root),kind=kind,manifest_sha256=value['manifest_sha256'],
                destination=value['destination'],required_file_sha256=expected))
        rows.append(dict(cell=cell,run=str(run),archives=matches))
        if not matches:missing.append(cell['id'])
    records=[{k:v for k,v in item.items() if k!='pins'} for item in cached.values()]
    individual=sum(any(a['kind']=='individual' for a in row['archives']) for row in rows)
    return dict(schema='shaping-primary-archive-coverage-v1',plan_sha256=sha(study/'PLAN.json'),
        primary_cells=len(primary),covered_cells=len(rows)-len(missing),individual_archive_cells=individual,
        batch_only_cells=sum(bool(row['archives']) and not any(a['kind']=='individual' for a in row['archives']) for row in rows),
        complete=not missing,missing_cells=missing,cells=rows,archives=records,
        scope='Successful effective primary fits only. Archived RUN/RESULT and final-model identities are matched to manifests whose complete durable readbacks were previously verified; matching current source/destination manifests are checked here. Not a fresh payload scrub or a completion claim for later stages.',
        training_replayed=False,scientific_evaluation_started=False)


if __name__=='__main__':
    from shaping_execution import load_registration
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--batch',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve existing archive coverage report')
    _,cells,_=load_registration(a.study);result=coverage(a.study,cells,a.batch);a.output.write_bytes(encoded(result))
    print(json.dumps({k:result[k] for k in ('primary_cells','covered_cells','individual_archive_cells','batch_only_cells','complete')}))
    if not result['complete']:raise ValueError('primary archive coverage remains incomplete')
