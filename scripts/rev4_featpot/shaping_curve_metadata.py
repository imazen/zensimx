"""Extend descriptive curve agreement using already collected native metadata."""
import argparse,ast,json,os,time
from contextlib import ExitStack
from pathlib import Path
from shaping_completion import encoded,sha


def extract(value):
    entries=[r for r in value['metadata'] if r['key']=='zentrain.input_shaping']
    if len(entries)!=1:raise ValueError('one native shaping metadata entry required')
    row=entries[0]
    if row['kind']!='utf8' or len(row['value_text'].encode())!=row['value_len']:
        raise ValueError('native shaping metadata length or encoding differs')
    result=json.loads(row['value_text'])
    if result['schema']!='zensim-input-shaping-v1':raise ValueError('native shaping schema differs')
    return result


def register(study,collection,output):
    from shaping_execution import load_registration
    _,cells,_=load_registration(study)
    receipt=json.loads((collection/'COLLECTION_VERIFIED.json').read_text())
    if not receipt['all_transport_sha256_verified'] or not receipt['all_candidate_provenance_verified']:
        raise ValueError('completed original collection required')
    manifest=collection/'MANIFEST.json'
    if sha(manifest)!=receipt['manifest_sha256']:raise ValueError('collection manifest differs')
    files=json.loads(manifest.read_text())['files'];jobs=[]
    for cell in cells:
        if cell['stage']!='inner' or cell['family'] not in ('product_joint_identity','separate_joint_identity'):continue
        paths=list((study/'reexports'/cell['id']).glob('attempt-*/final/RECEIPT.json'))
        if len(paths)!=1:raise ValueError('one collected native final export required')
        path=paths[0]
        if files[str(path)]['sha256']!=sha(path):raise ValueError('native receipt differs from verified collection')
        row=json.loads(path.read_text());model=path.parent/'original-inspection.json'
        if files[str(model)]['sha256']!=row['file_sha256'][str(model)]:raise ValueError('inspection pin differs')
        run=study/'runs'/cell['id']/path.parent.parent.name;result=json.loads((run/'RESULT.json').read_text())
        if result['cell']!=cell or result['stage']!='trained' or result['bake_sha256']!=row['original_sha256']:
            raise ValueError('original final training identity differs')
        initial=run/'SHAPING.json';report=study/'reexport_max_pairs'/cell['id']/run.name/'final/REPORT.json'
        if files[str(report)]['sha256']!=sha(report):raise ValueError('collected final report drift')
        jobs.append(dict(cell=cell,inspection=str(model),inspection_sha256=files[str(model)]['sha256'],
            initial=str(initial),initial_sha256=sha(initial),original_model_sha256=row['original_sha256'],
            active_output_coefficients=json.loads(report.read_text())['active_output_coefficients']))
    if len(jobs)!=144:raise ValueError('all 72 paired fits of both joint families required')
    output.mkdir(exist_ok=False)
    value=dict(schema='collected-curve-stability-inputs-v1',jobs=jobs,study=str(study),
        plan_sha256=sha(study/'PLAN.json'),collection_manifest_sha256=sha(manifest),
        driver_sha256=sha(__file__),comparison_driver_sha256=sha(Path(__file__).with_name('shaping_curve_stability.py')),
        operation='Descriptive curve agreement only; no fit, quality rescoring, model selection or transferred bank.',
        outer_scores_read=False)
    (output/'INPUTS.json').write_bytes(encoded(value));return dict(models=len(jobs))


def analyse(study,manifest,source,output):
    import sys
    sys.path.insert(0,str(source/'scripts/rev4_featpot'))
    from shaping_curve_stability import compare,parameter_activity
    from shaping_pretraining import structural_terms
    from shaping_rescue_dispatch import reserve_worker,release_worker
    from shaping_reduced_scope import admit_compute
    from shaping_storage import reservation
    admit_compute(study);q=json.loads(manifest.read_text())
    if (q['schema']!='collected-curve-stability-inputs-v1' or q['driver_sha256']!=sha(__file__)
        or q['plan_sha256']!=sha(study/'PLAN.json') or q['outer_scores_read']
        or q['comparison_driver_sha256']!=sha(source/'scripts/rev4_featpot/shaping_curve_stability.py')):
        raise ValueError('curve analysis source or input scope differs')
    key='curve-metadata-all-144'
    while not reserve_worker(study,key)['can_start']:time.sleep(10)
    resources=ExitStack()
    try:
        resources.enter_context(reservation(study,output,30*1024**3,bytes_required=512*1024**2))
        output.mkdir(exist_ok=False);configs=[];cases=[]
        (output/'STATUS.json').write_bytes(encoded(dict(pid=os.getpid(),stage='reading_collected_parameters',models=144)))
        for job in q['jobs']:
            raw=Path(job['inspection']).read_bytes()
            if __import__('hashlib').sha256(raw).hexdigest()!=job['inspection_sha256']:raise ValueError('native inspection drift')
            config=extract(json.loads(raw));del raw
            if sha(job['initial'])!=job['initial_sha256']:raise ValueError('initial shape drift')
            if structural_terms(config)!=structural_terms(json.loads(Path(job['initial']).read_text())):
                raise ValueError('native curve structure differs from TRAIN initialization')
            name=job['cell']['id']+'.json';(output/name).write_bytes(encoded(config));configs.append(config)
            cases.append(dict(cell=job['cell'],path=name,sha256=sha(output/name),activity=parameter_activity(config),active_output_coefficients=job['active_output_coefficients']))
        specs=[]
        for family in ('product_joint_identity','separate_joint_identity'):
            ids=[i for i,j in enumerate(q['jobs']) if j['cell']['family']==family]
            specs.append(dict(name=family+'-all-72',indices=ids,direct_only=False))
            for setting in range(4):specs.append(dict(name=family+f'-setting-{setting}',indices=[i for i in ids if q['jobs'][i]['cell']['setting']==setting],direct_only=False))
        specs.append(dict(name='direct-both-all-144',indices=list(range(144)),direct_only=True))
        summaries=[]
        for spec in specs:
            (output/'STATUS.json').write_bytes(encoded(dict(pid=os.getpid(),stage='descriptive_response_comparison',panel=spec['name'],completed_panels=len(summaries))))
            selected=[configs[i] for i in spec['indices']]
            if spec['direct_only']:selected=[dict(c,terms=[t for t in c['terms'] if t['reference'] is None]) for c in selected]
            rows=compare(selected);active=sum(r['active_fits']==len(selected) for r in rows);zero=sum(r['active_fits']==0 for r in rows)
            result=dict(scope=q['operation'],spec=spec,terms=rows,total_terms=len(rows),fits=len(selected),active_in_all=active,zero_in_all=zero,
                downstream_weights_included=False,zero_terms_are_not_useful_stability=True,outer_scores_read=False)
            (output/(spec['name']+'.json')).write_bytes(encoded(result));summaries.append({k:result[k] for k in ('spec','total_terms','fits','active_in_all','zero_in_all')})
            print(json.dumps(summaries[-1]),flush=True)
        (output/'CASES.json').write_bytes(encoded(cases));(output/'SUMMARY.json').write_bytes(encoded(summaries))
        # Preserve the existing mathematics verbatim as a stdlib-only replay.
        text=(source/'scripts/rev4_featpot/shaping_curve_stability.py').read_text();tree=ast.parse(text)
        functions=[ast.get_source_segment(text,node) for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('response','compare')]
        if len(functions)!=2:raise ValueError('curve mathematics replay closure differs')
        (output/'curve_math.py').write_text('import math,struct\n\n'+'\n\n'.join(functions)+'\n')
        (output/'REPRODUCE.py').write_text('''import json,hashlib
from pathlib import Path
from curve_math import compare
r=Path(__file__).resolve().parent;cases=json.loads((r/'CASES.json').read_text());configs=[]
for case in cases:
 p=r/case['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==case['sha256'];configs.append(json.loads(p.read_text()))
for summary in json.loads((r/'SUMMARY.json').read_text()):
 spec=summary['spec'];expected=json.loads((r/(spec['name']+'.json')).read_text());selected=[configs[i] for i in spec['indices']]
 if spec['direct_only']:selected=[dict(c,terms=[t for t in c['terms'] if t['reference'] is None]) for c in selected]
 assert compare(selected)==expected['terms'],spec['name']
 print(spec['name'],'MATCH')
''')
        result=dict(models=len(cases),panels=len(specs),training_or_prediction_replayed=False,pid=os.getpid(),stage='complete')
        (output/'STATUS.json').write_bytes(encoded(result));return result
    except Exception as error:
        if output.exists():(output/'STATUS.json').write_bytes(encoded(dict(pid=os.getpid(),stage='failed_with_preserved_evidence',error=str(error))))
        raise
    finally:resources.close();release_worker(study,key)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','analyse'))
    for name in ('study','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('collection','manifest','source'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();value=register(a.study,a.collection,a.output) if a.mode=='register' else analyse(a.study,a.manifest,a.source,a.output)
    print(json.dumps(value),flush=True)
