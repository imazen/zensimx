"""Seal a fixed scalar-export wave, preserving prior assignments and all fits."""
import argparse,json
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_stable_pairs import families,NAME
from shaping_evaluation import diagnostic_epochs
from shaping_execution import load_registration


def prepare(study,source,host,selected_families,previous,output):
    from shaping_reduced_scope import admit_compute
    admit_compute(study,host=host);qualified=set(families(study))
    if not selected_families or not set(selected_families)<=qualified:raise ValueError('explicit qualified families required')
    if output.exists():raise ValueError('preserve existing scalar-export wave')
    owner=source/'scripts/rev4_featpot/shaping_reexport_supervision.py'
    if sha(owner)!=sha(Path(__file__).with_name(owner.name)):raise ValueError('exported supplemental owner differs')
    _,cells,_=load_registration(study);certificate=study/NAME;pin=sha(certificate);assigned=set();previous_pins={}
    for path in previous:
        record=json.loads(path.read_text())
        if record['schema']!='fixed-qualified-supplemental-wave-v1' or record['plan_sha256']!=sha(study/'PLAN.json') or record['qualification_sha256']!=pin:
            raise ValueError('prior scalar-export registration differs')
        ids={q['cell']['id'] for q in record['jobs']}
        if assigned&ids:raise ValueError('overlapping prior scalar-export waves')
        assigned|=ids;previous_pins[str(path)]=sha(path)
    q=json.loads(certificate.read_text());paths={certificate,*map(Path,q['evidence_sha256'])};jobs=[]
    for partition in sorted({c['partition'] for c in cells if c['stage']=='inner'}):
        path=study/'comparators'/partition/'GROUPS.json';group=json.loads(path.read_text());paths.add(path)
        for name,expected in group['file_sha256'].items():
            if sha(name)!=expected:raise ValueError('registered scalar-export table drift')
            paths.add(Path(name))
    for cell in sorted(cells,key=lambda c:(c['partition'],str(c['setting']),str(c['seed']))):
        if cell['stage']!='inner' or cell['family'] not in selected_families or cell['id'] in assigned:continue
        complete=list((study/'reexport_max_pairs'/cell['id']).glob('attempt-*/CONVERGENCE.json'))
        if complete:raise ValueError('completed supplemental cell lacks a prior-wave declaration')
        runs=[p.parent for p in (study/'runs'/cell['id']).glob('attempt-*/RESULT.json') if json.loads(p.read_text()).get('stage')=='trained']
        if not runs:continue  # The caller registers only observed terminal models.
        if len(runs)!=1:raise ValueError('one successful original attempt required')
        run=runs[0];identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
        if identity['cell']!=cell or result['cell']!=cell:raise ValueError('original model identity differs')
        for name in ('RUN.json','RESULT.json','train.log','HEAD.json','SHAPING.json','keep.txt','COLLECTION_DERIVED_INPUTS.json','WORKER_COLLECTION.json'):
            path=run/name
            if path.exists():paths.add(path)
        epochs=identity['epochs']
        for epoch in {epochs-1,*diagnostic_epochs(epochs)}:paths.add(run/'checkpoints'/f'ckpt_epoch{epoch:03}.bin')
        if sha(run/'checkpoints'/f'ckpt_epoch{epochs-1:03}.bin')!=result['bake_sha256']:raise ValueError('original final bake drift')
        jobs.append(dict(cell=cell,run=str(run),epochs=epochs,report_positions=1+len(diagnostic_epochs(epochs))))
    if not jobs:raise ValueError('no new terminal scalar-export models')
    for path in paths:
        if not path.is_file() or path.is_symlink():raise ValueError('canonical scalar-export input required: '+str(path))
    files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)}
    if len(files)>10000 or sum(q['bytes'] for q in files.values())>2*1024**3:raise ValueError('scalar-export wave exceeds input allowance')
    record=dict(schema='fixed-qualified-supplemental-wave-v1',source=str(source),source_owner_sha256=sha(owner),host=host,
        families=list(selected_families),jobs=jobs,plan_sha256=sha(study/'PLAN.json'),qualification_sha256=pin,
        previous_wave_sha256=previous_pins,files=files,training_replayed=False,outer_or_final_holdout_read=False)
    output.mkdir(parents=True,exist_ok=False);(output/'REGISTRATION.json').write_bytes(encoded(record))
    (output/'INPUT_PATHS.txt').write_bytes(b''.join(str(p).lstrip('/').encode()+b'\0' for p in sorted(paths)))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--family',action='append',required=True)
    p.add_argument('--previous-wave',type=Path,action='append',default=[]);a=p.parse_args()
    value=prepare(a.study,a.source,a.host,a.family,a.previous_wave,a.output)
    print(json.dumps(dict(models=len(value['jobs']),report_positions=sum(q['report_positions'] for q in value['jobs']),files=len(value['files']))),flush=True)
