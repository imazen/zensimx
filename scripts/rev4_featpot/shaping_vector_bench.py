"""Time all admitted outer scoring models, preserving paired rounds and failures."""
import argparse,json,os,platform,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import ROLES,encoded,sha


def register(study,prepared,request,binary,qualification,launcher,cpu_tool,output):
    from shaping_cost_worker import validate_result
    if output.exists():raise ValueError('preserve existing vector timing registration')
    record=json.loads(request.read_text());collection=json.loads((prepared/'WORKER_COLLECTION.json').read_text())
    if (record['execution_kind']!='cost_prepare' or record['study']!=str(study)
        or record['result_output']!=str(prepared) or collection['request_sha256']!=sha(request)):
        raise ValueError('collected outer cost preparation required')
    validate_result(record,prepared)
    q=json.loads(qualification.read_text());build=binary.with_name('BUILD.json')
    if (q['native_instrument_qualified'] is not True or q['models']!=8 or q['rows']!=40
        or q['binary_sha256']!=sha(binary) or json.loads(build.read_text())['binary_sha256']!=sha(binary)):
        raise ValueError('qualified native vector instrument required')
    workloads=[];pins={};expected={role+'-'+('deterministic' if seed is None else str(seed))
        for role in ROLES for seed in ((None,) if role=='shaped_additive' else range(3))}
    for outer in range(3):
        path=prepared/f'outer-{outer}'/'WORKLOAD.json';payload=json.loads(path.read_text())
        if {a['id'] for a in payload['arms']}!=expected or len(payload['arms'])!=16:
            raise ValueError('all sixteen finalist role/seed timing arms required')
        workloads.append(str(path))
        for source in (path,path.with_name('PROVENANCE.json'),*(path.parent/a['model'] for a in payload['arms'])):
            if source.is_symlink() or source.resolve()!=source:raise ValueError('canonical vector input required')
            pins[str(source)]=sha(source)
    output.mkdir(parents=True);(output/'inputs').mkdir()
    for source,name in ((binary,binary.name),(build,'BUILD.json'),(qualification,'QUALIFICATION.json'),
        (launcher,launcher.name),(cpu_tool,cpu_tool.name),(request,'PREPARED_REQUEST.json'),
        (prepared/'WORKER_COLLECTION.json','PREPARED_COLLECTION.json')):
        target=output/'inputs'/name;shutil.copy2(source,target);pins[str(target)]=sha(source)
    (output/'REGISTRATION.json').write_bytes(encoded(dict(schema='shaping-outer-vector-timing-v1',
        study=str(study),plan_sha256=sha(study/'PLAN.json'),prepared=str(prepared),workloads=workloads,
        input_sha256=pins,driver_sha256=sha(__file__),models=48,rows_per_workload=40,
        instrument=str(output/'inputs'/binary.name),launcher=str(output/'inputs'/launcher.name),
        scientific_fit_count=0,final_holdout_read=False,timing_measured=False)))
    return dict(output=str(output),models=48,workloads=workloads)


def verify(root,study):
    record=json.loads((root/'REGISTRATION.json').read_text())
    if (record['schema']!='shaping-outer-vector-timing-v1' or record['driver_sha256']!=sha(__file__)
        or record['study']!=str(study) or record['plan_sha256']!=sha(study/'PLAN.json')
        or record['models']!=48 or record['rows_per_workload']!=40 or len(record['workloads'])!=3
        or record['scientific_fit_count']!=0 or record['final_holdout_read'] is not False):
        raise ValueError('outer vector timing identity differs')
    for name,pin in record['input_sha256'].items():
        if sha(name)!=pin:raise ValueError('outer vector input drift: '+name)
    return record


def run(root,study,cpus):
    from shaping_cost_report import report
    from shaping_reduced_scope import admit_compute
    from shaping_rescue_dispatch import reserve_worker,bind_worker,release_worker,worker_slot
    from shaping_storage import reservation
    admit_compute(study)
    if platform.node().split('.')[0] in ('dev','r7900x'):
        raise ValueError('vector timing requires another eligible benchmark host')
    record=verify(root,study)
    if (root/'STATUS.json').exists():raise ValueError('preserve previous vector timing owner; no implicit replay')
    key='vector-cost-'+sha(root/'REGISTRATION.json')[:16];held=False
    status=dict(pid=os.getpid(),host=platform.node(),registration_sha256=sha(root/'REGISTRATION.json'),
        cpus=cpus,metadata_python=sys.executable,completed_folds=[],stage='waiting_for_idle_benchmark_host')
    def save(stage,**values):
        status.update(stage=stage,checked_utc=datetime.now(timezone.utc).isoformat(),**values)
        (root/'STATUS.json').write_bytes(encoded(status))
    save('waiting_for_idle_benchmark_host')
    try:
        while True:
            # Benchmarks use zenbench's admission; do not hold a CPU claim ahead
            # of active training while waiting for a quiet benchmark host.
            if worker_slot(set(),study=study)['native_threads']==0:
                admission=reserve_worker(study,key,memory_allowance_bytes=4*1024**3)
                if admission['can_start']:held=True;break
            save('waiting_for_idle_benchmark_host');time.sleep(30)
        with reservation(study,root,30*1024**3):
            env=os.environ.copy()
            env.update(TMPDIR=str(Path.home()/'tmp'),PATH=str(root/'inputs')+':'+env['PATH'],
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1')
            for outer,path in enumerate(record['workloads']):
                verify(root,study);workload=Path(path);native=root/f'timing-outer-{outer}'
                command=[str(Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','4G','--jobs','1','--',
                    'bash',record['launcher'],cpus,'env','ZENBENCH_LAUNCHER_PIDS='+str(os.getpid()),
                    record['instrument'],'bench',str(workload),str(native)]
                save('measuring_vector_scoring',outer=outer,command=command)
                with (root/f'outer-{outer}.log').open('xb') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                    bind_worker(study,key,child.pid);code=child.wait()
                if code:raise RuntimeError('vector timing incomplete; preserve original attempt')
                report(native,workload,root/f'outer-{outer}-REPORT.json')
                save('measuring_fresh_process_memory',outer=outer)
                # Each arm loads in its own GNU-time-qualified native process.
                # Reuse the verified cpuset for these one-model observations.
                command=[str(Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','4G','--jobs','1','--',
                    'bash',record['launcher'],cpus,'env','PYTHONPATH='+str(Path(__file__).parent),
                    sys.executable,'-B',str(Path(__file__).with_name('shaping_cost_memory.py')),
                    '--binary',record['instrument'],'--workload',str(workload),'--output',str(root/f'memory-outer-{outer}')]
                with (root/f'outer-{outer}-memory.log').open('xb') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                    bind_worker(study,key,child.pid);code=child.wait()
                if code:raise RuntimeError('fresh model memory measurement failed; preserve original attempt')
                status['completed_folds'].append(outer);save('measuring_vector_scoring')
            verify(root,study);save('complete',timed_models=48,fresh_memory_models=48)
            return status
    except BaseException as error:
        save('failed_with_preserved_evidence',error=str(error));raise
    finally:
        if held:release_worker(study,key)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','run'))
    for name in ('study','prepared','request','binary','qualification','launcher','cpu-tool','output','root'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--cpus');a=p.parse_args()
    result=(run(a.root,a.study,a.cpus) if a.mode=='run' else register(a.study,a.prepared,a.request,
        a.binary,a.qualification,a.launcher,a.cpu_tool,a.output))
    print(json.dumps(result),flush=True)
