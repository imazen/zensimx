"""Seal a fixed scalar-export wave, preserving prior assignments and all fits."""
import argparse,json
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_stable_pairs import families,NAME
from shaping_evaluation import diagnostic_epochs
from shaping_execution import load_registration


def rescue_inputs(study,source,run,identity,result,paths):
    """Preserve phase receipts and the exact inner selection dependency closure."""
    from shaping_rescue_worker import reads
    from shaping_rescue import resolve
    from shaping_completion import REPO
    with reads() as dependencies:
        resolution=resolve(study,identity['cell']['id'])
    if resolution!=identity['resolution']:raise ValueError('rescue resolution drift')
    for path in dependencies:
        if REPO in path.parents:
            target=source/path.relative_to(REPO)
            if sha(path)!=sha(target):raise ValueError('exported rescue policy source differs')
            path=target
        paths.add(path)
    names=['pretraining','fit'] if len(result['phases'])==2 else ['fit']
    for name,phase in zip(names,result['phases']):
        directory=run/name
        if json.loads((directory/'RESULT.json').read_text())!=phase:raise ValueError('rescue phase receipt differs')
        if phase['stage']!='trained' or phase['exit_code']!=0 or phase['pairs_per_epoch']!=50000:
            raise ValueError('rescue phase training incomplete')
        paths.update((directory/'RUN.json',directory/'RESULT.json',directory/'train.log'))
        checkpoint=directory/'checkpoints'/f"ckpt_epoch{phase['epochs']-1:03}.bin"
        if sha(checkpoint)!=phase['bake_sha256']:raise ValueError('rescue phase model differs')
        paths.add(checkpoint)
        for key in ('manifest_sha256','source_file_sha256'):
            for path,pin in phase[key].items():
                if sha(path)!=pin:raise ValueError('rescue phase input differs')
                paths.add(Path(path))


def prepare(study,source,host,selected_families,previous,output,*,stage='inner',cell_ids=None,evaluation_inputs=None):
    from shaping_reduced_scope import admit_compute
    admit_compute(study,host=host);qualified=set(families(study))
    if not selected_families or not set(selected_families)<=qualified:raise ValueError('explicit qualified families required')
    if stage not in ('inner','rescue'):raise ValueError('registered inner or rescue stage required')
    if stage=='rescue' and (not cell_ids or evaluation_inputs is None):raise ValueError('explicit rescue scope and inner evaluation closure required')
    if output.exists():raise ValueError('preserve existing scalar-export wave')
    owner=source/'scripts/rev4_featpot/shaping_reexport_supervision.py'
    if sha(owner)!=sha(Path(__file__).with_name(owner.name)):raise ValueError('exported supplemental owner differs')
    _,cells,_=load_registration(study);certificate=study/NAME;pin=sha(certificate);assigned=set();previous_pins={}
    if cell_ids is not None and not set(cell_ids)<={c['id'] for c in cells if c['stage']==stage and c['family'] in selected_families}:
        raise ValueError('explicit scalar scope contains unregistered or wrong-stage cells')
    for path in previous:
        record=json.loads(path.read_text())
        if record['schema']!='fixed-qualified-supplemental-wave-v1' or record['plan_sha256']!=sha(study/'PLAN.json') or record['qualification_sha256']!=pin:
            raise ValueError('prior scalar-export registration differs')
        ids={q['cell']['id'] for q in record['jobs']}
        if assigned&ids:raise ValueError('overlapping prior scalar-export waves')
        assigned|=ids;previous_pins[str(path)]=sha(path)
    q=json.loads(certificate.read_text());paths={certificate,*map(Path,q['evidence_sha256'])};jobs=[]
    if evaluation_inputs is not None:
        from shaping_evaluation_inputs import validate
        closure=json.loads(evaluation_inputs.read_text());validate(study,closure);paths.add(evaluation_inputs)
        for name,row in closure['files'].items():
            if sha(name)!=row['sha256']:raise ValueError('inner evaluation input differs')
            paths.add(Path(name))
    for partition in sorted({c['partition'] for c in cells if c['stage']=='inner'}):
        path=study/'comparators'/partition/'GROUPS.json';group=json.loads(path.read_text());paths.add(path)
        for name,expected in group['file_sha256'].items():
            if sha(name)!=expected:raise ValueError('registered scalar-export table drift')
            paths.add(Path(name))
    for cell in sorted(cells,key=lambda c:(c['partition'],str(c['setting']),str(c['seed']))):
        if (cell['stage']!=stage or cell['family'] not in selected_families or cell['id'] in assigned
            or (cell_ids is not None and cell['id'] not in cell_ids)):continue
        complete=list((study/'reexport_max_pairs'/cell['id']).glob('attempt-*/CONVERGENCE.json'))
        if complete:raise ValueError('completed supplemental cell lacks a prior-wave declaration')
        runs=[p.parent for p in (study/'runs'/cell['id']).glob('attempt-*/RESULT.json') if json.loads(p.read_text()).get('stage')=='trained']
        if not runs:continue  # The caller registers only observed terminal models.
        if len(runs)!=1:raise ValueError('one successful original attempt required')
        run=runs[0];identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
        if identity['cell']!=cell or result['cell']!=cell:raise ValueError('original model identity differs')
        fit=run/identity.get('fit_directory','.')
        if stage=='rescue':rescue_inputs(study,source,run,identity,result,paths)
        for name in ('RUN.json','RESULT.json','train.log','HEAD.json','SHAPING.json','keep.txt','COLLECTION_DERIVED_INPUTS.json','WORKER_COLLECTION.json'):
            path=run/name
            if path.exists():paths.add(path)
        epochs=identity['epochs']
        for epoch in {epochs-1,*diagnostic_epochs(epochs)}:paths.add(fit/'checkpoints'/f'ckpt_epoch{epoch:03}.bin')
        if sha(fit/'checkpoints'/f'ckpt_epoch{epochs-1:03}.bin')!=result['bake_sha256']:raise ValueError('original final bake drift')
        jobs.append(dict(cell=cell,run=str(run),epochs=epochs,report_positions=1+len(diagnostic_epochs(epochs))))
    if not jobs:raise ValueError('no new terminal scalar-export models')
    for path in paths:
        if not path.is_file() or path.is_symlink():raise ValueError('canonical scalar-export input required: '+str(path))
    files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)}
    if len(files)>10000 or sum(q['bytes'] for q in files.values())>2*1024**3:raise ValueError('scalar-export wave exceeds input allowance')
    record=dict(schema='fixed-qualified-supplemental-wave-v1',source=str(source),source_owner_sha256=sha(owner),host=host,
        families=list(selected_families),jobs=jobs,plan_sha256=sha(study/'PLAN.json'),qualification_sha256=pin,
        previous_wave_sha256=previous_pins,files=files,training_replayed=False,outer_or_final_holdout_read=False)
    if stage!='inner':record['cell_stage']=stage
    output.mkdir(parents=True,exist_ok=False);(output/'REGISTRATION.json').write_bytes(encoded(record))
    (output/'INPUT_PATHS.txt').write_bytes(b''.join(str(p).lstrip('/').encode()+b'\0' for p in sorted(paths)))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--family',action='append',required=True)
    p.add_argument('--stage',choices=('inner','rescue'),default='inner')
    p.add_argument('--cell-list',type=Path);p.add_argument('--evaluation-inputs',type=Path)
    p.add_argument('--previous-wave',type=Path,action='append',default=[]);a=p.parse_args()
    selected=json.loads(a.cell_list.read_text()) if a.cell_list else None
    value=prepare(a.study,a.source,a.host,a.family,a.previous_wave,a.output,
                  stage=a.stage,cell_ids=selected,evaluation_inputs=a.evaluation_inputs)
    print(json.dumps(dict(models=len(value['jobs']),report_positions=sum(q['report_positions'] for q in value['jobs']),files=len(value['files']))),flush=True)
