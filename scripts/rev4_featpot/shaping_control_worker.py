"""Transport admitted tree/linear controls without changing their fitting owners."""
import argparse,json,shutil,socket
from pathlib import Path
from types import SimpleNamespace
import shaping_controls as owner
from shaping_completion import DECLARATION_SHA256,REPO,encoded,sha


def packages():
    import numpy,scipy,sklearn,pyarrow
    return {m.__name__:m.__version__ for m in (numpy,scipy,sklearn,pyarrow)}


def context(study,cell_id,binaries,*,host=None):
    from shaping_execution import load_registration
    _,cells,_=load_registration(study);found=[c for c in cells if c['id']==cell_id]
    if len(found)!=1:raise ValueError('one registered control required')
    cell=found[0]
    if cell['family'] not in ('extra_trees','shaped_additive') or cell['stage'] not in ('outer','source_curve','shuffle','human_only'):
        raise ValueError('approved outer or reduced variant control required')
    if (study/'runs'/cell_id).exists():raise ValueError('preserve existing control attempts; never replay')
    args=SimpleNamespace(study=study,binary_dir=binaries,partition=cell['partition'],tree=cell['family']=='extra_trees',
        cell=cell_id if cell['stage']!='outer' else None,selected_cell=cell_id if cell['stage']=='outer' else None,
        attempt=1,full_fit_selection=False,full_fit_refit=False,native_mem='2G')
    plan,partitions,family,chosen,outer=owner.admit(args,host=host)
    if chosen!=[cell] or outer is None:raise ValueError('one complete admitted control scope required')
    for path,pin in plan['input_pins'].items():
        if sha(path)!=pin:raise ValueError('frozen control input drift')
    build=json.loads((binaries/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(binaries/name)!=pin:raise ValueError('native control binary drift')
    if sha(REPO/'scripts/rev4_featpot/color_noise_fx2_declaration.json')!=DECLARATION_SHA256:
        raise ValueError('frozen control feature declaration drift')
    if family=='extra_trees':sha(REPO/'scripts/v_next/train_corruption_head.py')
    import shaped_linear_study,tree_study
    membership=partitions[cell['partition']]
    scope=dict(cell=cell,outer=outer,plan_sha256=sha(study/'PLAN.json'),build_sha256=sha(binaries/'BUILD.json'),
        expected_indices=sorted(set(membership['train'])|set(membership['test'])),package_versions=packages(),
        native_memory_cap='2G',python_memory_cap='2G',outer_results_read=False)
    return dict(args=args,scope=scope,cell=cell,outer=outer)


def register(study,cell_id,binaries,host,output):
    from shaping_rescue_worker import HOSTS,reads,seal_inputs
    if host not in HOSTS:raise ValueError('approved control worker required')
    packages()  # Load third-party modules outside the project-file read capture.
    with reads() as paths:admitted=context(study,cell_id,binaries,host=host)
    record=seal_inputs(paths,study,admitted['cell'],host,binaries,None,admitted['scope'],execution_kind='control')
    record.update(result_output=str(study/'runs'/cell_id/'attempt-001'),control_driver_sha256=sha(__file__))
    output.mkdir(parents=True,exist_ok=False);(output/'REQUEST.json').write_bytes(encoded(record))
    (output/'ASSIGNMENT.json').write_bytes(encoded(dict(host=host,cells=[cell_id],plan_sha256=record['plan_sha256'],build_sha256=record['build_sha256'])))
    return record


def reproduce(record,*,host=None):
    if record['control_driver_sha256']!=sha(__file__):raise ValueError('control worker source differs')
    study=Path(record['study']);cell_id=record['cell']['id']
    if record['result_output']!=str(study/'runs'/cell_id/'attempt-001'):raise ValueError('control output differs')
    admitted=context(study,cell_id,Path(record['binary_dir']),host=socket.gethostname() if host is None else host)
    # Each controller/fit gets its own two-GiB scope; its native linear child
    # also has two GiB. The shared CPU claim reserves the combined allowance.
    heavy=Path.home()/'work/claudehints/scripts/run-heavy'
    command=[str(heavy),'--mem','2G','--jobs','1','--','python3','-B',str(REPO/'scripts/rev4_featpot/shaping_control_worker.py'),
        'fit','--request',record['_request_path']]
    return admitted['scope'],command


def fit(request):
    from shaping_rescue_worker import verify
    record=json.loads(request.read_text());verify(record)
    admitted=context(Path(record['study']),record['cell']['id'],Path(record['binary_dir']))
    if admitted['scope']!=record['resolution']:raise ValueError('control gate changed before fitting')
    owner.run(admitted['args']);result=Path(record['result_output']);copies={}
    if record['cell']['family']=='shaped_additive':
        source=Path(record['study'])/'control_inputs'/owner.control_cache_key(record['cell']['partition'],admitted['outer'])/'linear'
        target=result/'native-inputs';target.mkdir()
        for name in ('inputs.npz','gram.npz','SPEC.json','INPUTS.json'):
            pin=sha(source/name);shutil.copyfile(source/name,target/name)
            if sha(target/name)!=pin:raise ValueError('native control replay input changed')
            copies[name]=dict(original_path=str(source/name),sha256=pin)
    (result/'WORKER_EXECUTION.json').write_bytes(encoded(dict(request_sha256=sha(request),execution_host=socket.gethostname(),
        native_replay_inputs=copies,package_versions=packages(),driver_sha256=sha(__file__))))


def validate_result(record,directory):
    from shaping_prediction_evidence import read
    run=json.loads((directory/'RUN.json').read_text());terminal=json.loads((directory/'RESULT.json').read_text())
    scope=record['resolution'];cell=record['cell']
    for value in (run,terminal):
        if (value['cell']!=cell or value['plan_sha256']!=scope['plan_sha256'] or value['build_sha256']!=scope['build_sha256']
            or any(value.get(k)!=v for k,v in scope['outer'].items())):raise ValueError('terminal control scope differs')
    if terminal['stage']!='complete' or terminal['exit_code']!=0:raise ValueError('successful completed control required')
    report,_,_=read(directory/'evaluation',cell,scope['plan_sha256'],scope['outer']['admission_sha256'],scope['expected_indices'])
    execution=json.loads((directory/'WORKER_EXECUTION.json').read_text())
    if execution['execution_host']!=record['host'] or execution['package_versions']!=scope['package_versions']:
        raise ValueError('control execution environment differs')
    qualification=report['qualification']
    if cell['family']=='extra_trees':
        gate=qualification['native_tree_gate']
        for name,key in [('model.opaque-vector.zcth','model_sha256'),('model.parity.npz','parity_sha256'),('model.parity.log','log_sha256')]:
            if sha(directory/name)!=gate[key]:raise ValueError('native tree gate artifact drift')
        if gate['raw_ulp']!=0:raise ValueError('native tree parity differs')
    else:
        if sha(directory/'native/model.bin')!=qualification['model_sha256']:raise ValueError('native linear model drift')
        if set(execution['native_replay_inputs'])!={'inputs.npz','gram.npz','SPEC.json','INPUTS.json'}:
            raise ValueError('complete code-free linear replay inputs required')
        for name,row in execution['native_replay_inputs'].items():
            if sha(directory/'native-inputs'/name)!=row['sha256']:raise ValueError('native linear replay input drift')


def pack(request,bundle):
    from shaping_terminal_transport import pack as original
    record=json.loads(request.read_text());execution=json.loads((Path(record['result_output'])/'WORKER_EXECUTION.json').read_text())
    if execution['request_sha256']!=sha(request):raise ValueError('control execution request differs')
    return original(request,bundle,validate_result)


def collect(request,bundle,pin):
    from shaping_terminal_transport import collect as original
    def replay(record,*,host):return reproduce(dict(record,_request_path=str(request)),host=host)
    def validate(record,directory):
        if json.loads((directory/'WORKER_EXECUTION.json').read_text())['request_sha256']!=sha(request):
            raise ValueError('collected control request differs')
        validate_result(record,directory)
    return dict(original(request,bundle,pin,replay,validate),scientific_fit_count=1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','fit','pack','collect'))
    for name in ('study','binary-dir','output','request','bundle'):p.add_argument('--'+name,type=Path)
    p.add_argument('--cell');p.add_argument('--host');p.add_argument('--bundle-sha256');a=p.parse_args()
    if a.mode=='register':r=register(a.study,a.cell,a.binary_dir,a.host,a.output);r=dict(cell=r['cell'],files=len(r['files']),bytes=r['input_bytes'])
    elif a.mode=='fit':r=fit(a.request)
    elif a.mode=='pack':r=pack(a.request,a.bundle)
    else:r=collect(a.request,a.bundle,a.bundle_sha256)
    print(json.dumps({k:v for k,v in r.items() if k!='files'} if isinstance(r,dict) else r),flush=True)
