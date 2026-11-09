"""Freeze the scalar viewer and linked reports without rerunning metric/model code."""
import argparse,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha,max_pair_families


def freeze(study,dashboard,output):
    if output.exists() or output.is_relative_to(dashboard):raise ValueError('distinct fresh viewer snapshot required')
    index=json.loads((dashboard/'details/INDEX.json').read_text());plan_pin=sha(study/'PLAN.json')
    if index['schema']!='zensim-inner-scatter-index-v1' or index['plan_sha256']!=plan_pin or index['outer_results_read']:
        raise ValueError('registered inner scatter index required')
    if any(p.is_symlink() for p in dashboard.rglob('*')):raise ValueError('regular publication assets required')
    repaired=set(max_pair_families(study));reports=[]
    for cid,row in index['cells'].items():
        cell=row['cell']
        if cell['id']!=cid or cell['stage'] not in ('inner','fresh_control_inner','rescue','exposure_extension'):
            raise ValueError('viewer snapshot contains another scientific scope')
        root=study/('reexport_max_pairs' if cell['family'] in repaired else 'runs')/cid
        paths=list(root.glob('attempt-*/final/REPORT.json' if cell['family'] in repaired else 'attempt-*/evaluation/REPORT.json'))
        if len(paths)!=1 or sha(paths[0])!=row['report_sha256']:raise ValueError('linked scatter report differs')
        for key in ('url','native_url'):
            path=dashboard/row[key]
            if not path.resolve().is_relative_to((dashboard/'details').resolve()) or path.is_symlink() or not path.is_file():
                raise ValueError('complete regular scalar publication required')
        reports.append(paths[0])
    output.mkdir(parents=True)
    # Published content-addressed scalar assets and completed reports are
    # immutable. Hard links retain them without another local byte copy.
    shutil.copytree(dashboard,output/'stats',copy_function=os.link,symlinks=False,
                    ignore=shutil.ignore_patterns('*.writing','*.publishing','*.lock','.owner','publisher-status.json'))
    # Mutable publication metadata must have independent inodes, including
    # lock/status files that a live publisher may update in place.
    mutable=[p for p in dashboard.iterdir() if p.is_file()]
    mutable.extend(dashboard/'details'/name for name in ('INDEX.json','NATIVE_JOBS.json'))
    for source in mutable:
        target=output/'stats'/source.relative_to(dashboard)
        if not target.exists():continue
        target.unlink();shutil.copy2(source,target)
    for report in reports:
        target=output/report.relative_to(study.parent);target.parent.mkdir(parents=True,exist_ok=True)
        os.link(report,target)
    frozen_index=json.loads((output/'stats/details/INDEX.json').read_text())
    if frozen_index!=index:raise ValueError('scatter publication changed during snapshot; preserve and reconcile')
    result=dict(schema='zensim-scalar-viewer-snapshot-v1',checked_utc=datetime.now(timezone.utc).isoformat(),
                plan_sha256=plan_pin,published_cells=len(index['cells']),linked_reports=len(reports),
                statistics_snapshot_sha256=sha(output/'stats/stats.json'),
                model_inference_replayed=False,metric_statistics_replayed=False,
                files=sum(p.is_file() for p in output.rglob('*')),
                file_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                interpretation='Frozen viewer snapshot; live research may remain incomplete.')
    (output/'SNAPSHOT.json').write_bytes(encoded(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','dashboard','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    from contextlib import ExitStack
    from shaping_storage import reservation
    with ExitStack() as stack:
        for part in range(3):stack.enter_context(reservation(a.study,a.output.with_name(a.output.name+f'-reservation-{part}'),30*1024**3))
        print(json.dumps(freeze(a.study,a.dashboard,a.output)))
