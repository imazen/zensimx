"""Attach native structure, training parameters and observed costs to finalists."""
import argparse,json,math,subprocess
from pathlib import Path
from shaping_completion import NEURAL,encoded,sha


def inputs(study,rows):
    root=study.parent
    binary=root/'model-inventory-bin/shaping_model_inventory'
    build=binary.with_name('BUILD.json')
    qualification=root/'model-inventory-qualification/QUALIFICATION.json'
    q=json.loads(qualification.read_text())
    if (q['native_inventory_verified'] is not True or q['models']!=8
        or sha(binary)!=json.loads(build.read_text())['binary_sha256']
        or q['source_pins'].get(str(binary))!=sha(binary)):
        raise ValueError('qualified native model inventory required')
    inspector=root/'optimizer-bin/bake_dial_refit'
    if q['source_pins'].get(str(inspector))!=sha(inspector):
        raise ValueError('qualified parameter inspector differs')
    pins={str(p):sha(p) for p in (binary,build,qualification,inspector)}
    for row in rows:
        run=Path(row['run_directory']);identity=json.loads((run/'RUN.json').read_text())
        result=json.loads((run/'RESULT.json').read_text())
        paths=[run/'RUN.json',run/'RESULT.json']
        if row['cell']['family'] in NEURAL:
            fit=run/identity.get('fit_directory','.')
            paths.extend(p for p in (fit/'HEAD.json',fit/'SHAPING.json') if p.exists())
            phases=['pretraining','fit'] if len(result.get('phases',[]))==2 else ['fit'] if result.get('phases') else [identity.get('fit_directory','.')]
            for phase in phases:
                directory=run/phase;native=json.loads((directory/'RESULT.json').read_text())
                paths.extend([directory/'RUN.json',directory/'RESULT.json',directory/'train.log',
                    directory/'checkpoints'/f"ckpt_epoch{native['epochs']-1:03}.bin"])
        elif row['cell']['family']=='shaped_additive':paths.append(run/'native/RESULT.json')
        pins.update({str(p):sha(p) for p in paths})
    return pins,binary,inspector


def write_fold(study,rows,workload,directory,binary,inspector):
    from shaping_parameter_count import count_run
    from shaping_training_resources import report as resources
    directory.mkdir();native=directory/'NATIVE.json'
    with (directory/'native.log').open('xb') as log:
        subprocess.run([str(binary),str(workload),str(native)],stdout=log,stderr=subprocess.STDOUT,check=True)
    output=[]
    for row in rows:
        cell=row['cell'];run=Path(row['run_directory']);role=row['role']
        name=role+'-'+('deterministic' if cell['seed'] is None else str(cell['seed']))
        item=dict(id=name,cell=cell,role=role,resolved_family=cell['family'])
        if cell['family'] in NEURAL:
            parameter=directory/(name+'-PARAMETERS.json');training=directory/(name+'-TRAINING.json')
            p=count_run(run,inspector,parameter);r=resources(run,training)
            item.update(trainable_scalar_entries=p['trainable_scalar_entries'],
                frozen_response_shape_values=p['frozen_response_shape_values'],
                parameters=dict(path=parameter.name,sha256=sha(parameter)),
                training_resources=dict(path=training.name,sha256=sha(training)),
                training_wrapper_seconds=r['directly_executed_phase_wrapper_seconds'],
                parameter_scope=p['interpretation'])
        elif cell['family']=='shaped_additive':
            source=run/'native/RESULT.json';values=json.loads(source.read_text())['coefficients']
            if not values or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
                raise ValueError('finite nonnegative fitted additive coefficients required')
            item.update(adjustable_basis_coefficients=len(values),nonzero_basis_coefficients=sum(v!=0 for v in values),
                coefficient_source=str(source),coefficient_source_sha256=sha(source),
                parameter_scope='Fitted nonnegative basis coefficients; fixed knots, transforms and score intercept excluded; not effective rank.')
        else:item.update(parameter_scope='Tree/node structure in NATIVE.json; scalar training-parameter equivalence is not asserted.')
        output.append(item)
    record=dict(schema='shaping-finalist-complexity-v1',workload_sha256=sha(workload),
        native_inventory_sha256=sha(native),native_inventory_binary_sha256=sha(binary),
        models=output,timing_measured=False,training_replayed=False,
        resource_scope='Observed concurrent-run training receipts, separate from isolated inference timing. No control training peak RSS is inferred.')
    (directory/'SUMMARY.json').write_bytes(encoded(record))
    return record


def validate_fold(directory,workload,cells,pins):
    record=json.loads((directory/'SUMMARY.json').read_text());native=directory/'NATIVE.json'
    w=json.loads(workload.read_text());n=json.loads(native.read_text())
    tools=[p for p in pins if Path(p).name=='shaping_model_inventory']
    if (len(tools)!=1 or record['schema']!='shaping-finalist-complexity-v1' or record['workload_sha256']!=sha(workload)
        or record['native_inventory_sha256']!=sha(native) or record['training_replayed'] is not False
        or record['timing_measured'] is not False or n['workload_sha256']!=sha(workload)
        or n['binary_sha256']!=record['native_inventory_binary_sha256']
        or record['native_inventory_binary_sha256']!=pins[tools[0]]
        or n['schema']!='shaping-model-inventory-v1' or n['timing_measured'] is not False):
        raise ValueError('finalist complexity identity differs')
    expected={a['id']:a for a in w['arms']};observed={a['id']:a for a in n['models']}
    by_id={a['id']:a for a in record['models']}
    if (len(record['models'])!=16 or len(n['models'])!=16 or set(expected)!=set(observed) or set(expected)!=set(by_id)
        or sorted((a['cell'] for a in record['models']),key=lambda c:c['id'])!=sorted((a['cell'] for a in cells),key=lambda c:c['id'])):
        raise ValueError('all sixteen finalist complexity records required')
    for name,arm in expected.items():
        item=by_id[name];model=workload.parent/arm['model']
        expected_cells=[c for c in cells if c['cell']==item['cell']]
        if (len(expected_cells)!=1 or item['role']!=expected_cells[0]['role']
            or name!=item['role']+'-'+('deterministic' if item['cell']['seed'] is None else str(item['cell']['seed']))):
            raise ValueError('complexity finalist role differs')
        if observed[name]['model_sha256']!=arm['sha256'] or observed[name]['model_bytes']!=model.stat().st_size:
            raise ValueError('complexity model differs from timed representation')
        if item['resolved_family']!=item['cell']['family']:raise ValueError('complexity family differs')
        if item['cell']['family'] in NEURAL:
            for field in ('parameters','training_resources'):
                path=directory/item[field]['path']
                if path.parent!=directory or sha(path)!=item[field]['sha256']:raise ValueError('complexity report drift')
                proof=json.loads(path.read_text())
                if proof['cell']!=item['cell'] or any(pins.get(p)!=v for p,v in proof['source_pins'].items()):
                    raise ValueError('unbound finalist parameter/resource evidence')
            p=json.loads((directory/item['parameters']['path']).read_text())
            t=json.loads((directory/item['training_resources']['path']).read_text())
            if (p['trainable_scalar_entries']!=item['trainable_scalar_entries']
                or p['frozen_response_shape_values']!=item['frozen_response_shape_values']
                or t['directly_executed_phase_wrapper_seconds']!=item['training_wrapper_seconds']):
                raise ValueError('finalist complexity summary differs')
        elif item['cell']['family']=='shaped_additive':
            source=Path(item['coefficient_source'])
            if pins.get(str(source))!=item['coefficient_source_sha256'] or sha(source)!=item['coefficient_source_sha256']:
                raise ValueError('unbound additive coefficient evidence')
            values=json.loads(source.read_text())['coefficients']
            if (item['adjustable_basis_coefficients']!=len(values)
                or item['nonzero_basis_coefficients']!=sum(v!=0 for v in values)):
                raise ValueError('additive coefficient summary differs')


def qualification_inputs(study,fixture,expected):
    from shaping_execution import load_registration
    from shaping_rescue_worker import reads
    with reads() as observed:
        _,registered,_=load_registration(study)
        provenance=json.loads((fixture/'PROVENANCE.json').read_text())
        workload=fixture/'WORKLOAD.json';payload=json.loads(workload.read_text())
        if provenance['workload_sha256']!=sha(workload) or len(payload['arms'])!=8:
            raise ValueError('existing eight-model instrument fixture required')
        rows=[]
        for cell in provenance['cells']:
            if (cell not in registered or cell['stage']!='inner' or cell['partition']!='inner-0-0'
                or cell['setting']!=0 or cell['seed']!=(None if cell['family']=='shaped_additive' else 0)):
                raise ValueError('fixed inner-only inventory qualification required')
            rows.append(dict(cell=cell,role=cell['family'],run_directory=str(study/'runs'/cell['id']/'attempt-001')))
        pins,binary,inspector=inputs(study,rows)
        reference=study.parent/'model-inventory-qualification/INVENTORY.json'
        qualification=json.loads((study.parent/'model-inventory-qualification/QUALIFICATION.json').read_text())
        if qualification['inventory_sha256']!=sha(reference):raise ValueError('native inventory reference differs')
        pins[str(reference)]=sha(reference)
        for arm in payload['arms']:
            path=fixture/arm['model']
            if sha(path)!=arm['sha256']:raise ValueError('qualification model differs')
            pins[str(path)]=sha(path)
        for row in rows:
            if row['cell']['family'] in NEURAL:
                path=expected/(row['cell']['family']+'.json');proof=json.loads(path.read_text())
                if proof['cell']!=row['cell']:raise ValueError('parameter reference specimen differs')
                pins[str(path)]=sha(path)
    pins.update({str(p):sha(p) for p in observed})
    return dict(rows=rows,pins=pins,binary=str(binary),inspector=str(inspector),
        fixture=str(fixture),expected=str(expected),reference=str(reference),scientific_fit_count=0)


def qualify(study,fixture,expected,output):
    q=qualification_inputs(study,fixture,expected)
    if output.exists():raise ValueError('preserve existing finalist inventory qualification')
    output.mkdir(parents=True);payload=json.loads((fixture/'WORKLOAD.json').read_text())
    for arm in payload['arms']:
        arm['model']=str(fixture/arm['model'])
        arm['id']+='-'+('deterministic' if arm['kind']=='shaped_additive' else '0')
    workload=output/'WORKLOAD.json';workload.write_bytes(encoded(payload))
    result=write_fold(study,q['rows'],workload,output/'complexity',Path(q['binary']),Path(q['inspector']))
    observed=json.loads((output/'complexity/NATIVE.json').read_text())
    reference=json.loads(Path(q['reference']).read_text());by_name={r['id']:r for r in reference['models']}
    for row in observed['models']:
        original=by_name[row['id'].removesuffix('-deterministic').removesuffix('-0')]
        if any(row[k]!=original[k] for k in ('model_sha256','model_bytes','structure')):
            raise ValueError('native inventory differs from qualified fixture')
    neural=[]
    for row in result['models']:
        if row['cell']['family'] in NEURAL:
            original=json.loads((expected/(row['cell']['family']+'.json')).read_text())
            for key in ('trainable_scalar_entries','frozen_response_shape_values'):
                if row[key]!=original[key]:raise ValueError('parameter count differs from qualified native specimen')
            neural.append(row['cell']['id'])
    for p,pin in q['pins'].items():
        if sha(p)!=pin:raise ValueError('qualification inputs changed')
    receipt=dict(models=len(result['models']),neural_parameter_and_resource_reports=len(neural),
        native_inventory_equal=True,parameter_counts_equal=True,scientific_fit_count=0,
        final_holdout_read=False,timing_measured=False,source_pins=q['pins'],driver_sha256=sha(__file__))
    (output/'QUALIFICATION.json').write_bytes(encoded(receipt));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('inputs','qualify'))
    for name in ('study','fixture','expected','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.mode=='inputs':a.output.write_bytes(encoded(qualification_inputs(a.study,a.fixture,a.expected)))
    else:print(json.dumps(qualify(a.study,a.fixture,a.expected,a.output)))
