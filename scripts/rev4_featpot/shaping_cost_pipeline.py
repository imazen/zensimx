"""Prepare, time and archive actual outer finalists after their existing gates."""
import argparse,json,os,shlex,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import REPO,encoded,sha


def ready(study):
    from shaping_bootstrap_pipeline import ready as owner
    return owner(study,dict(spec=None))


def submit(args):
    from shaping_cost_worker import register
    from shaping_rescue_host_launch import launch
    wave=args.output/'preparation';request=wave/'request/REQUEST.json'
    q=register(args.study,args.binaries,args.output/'prepared',args.host,request.parent)
    assignment=args.study/'downstream_assignments'/('cost-prepare-'+q['cell']['id']+'-'+args.host+'.json')
    with assignment.open('xb') as stream:stream.write(request.with_name('ASSIGNMENT.json').read_bytes())
    job=dict(cell=q['cell'],host=args.host,request=str(request),request_sha256=sha(request),
        assignment=str(assignment),assignment_sha256=sha(assignment))
    (wave/'REGISTRATION.json').write_bytes(encoded(dict(stage='registered',source=str(REPO),jobs=[job],
        source_driver_sha256=sha(REPO/'scripts/rev4_featpot/shaping_rescue_worker.py'),scientific_fit_count=0)))
    launch([wave],REPO,args.host,wave/'launch')
    return request


def watch(args):
    from shaping_rescue_worker import HOSTS
    if args.host not in HOSTS or args.host=='r7900x':raise ValueError('eligible non-training benchmark host required')
    args.output.mkdir(parents=True,exist_ok=False);logs=args.output/'logs';logs.mkdir()
    status=dict(pid=os.getpid(),host=args.host,driver_sha256=sha(__file__),scientific_fit_count=0,
        final_holdout_read=False,stage='waiting_for_complete_outer_finalists')
    def save(stage,**fields):
        status.update(stage=stage,checked_utc=datetime.now(timezone.utc).isoformat(),**fields)
        (args.output/'STATUS.json').write_bytes(encoded(status))
    def execute(command,name):
        with (logs/(name+'.log')).open('x') as log:
            log.write(json.dumps(command)+'\n');log.flush()
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            save(status['stage'],child_pid=child.pid)
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:save(status['stage'])
        save(status['stage'],child_pid=None)
        if code:raise RuntimeError('cost pipeline command failed; preserve '+str(logs/(name+'.log')))
    def remote(command,name):
        execute(['ssh',args.host,shlex.join(['env','TMPDIR=/home/lilith/tmp',*map(str,command)])],name)
    def observed(path,script,unit,output):
        code='path='+repr(str(path))+'\nscript='+repr(script)+'\nunit='+repr(unit)+'\noutput='+repr(str(output))+'\n'+'''from pathlib import Path
import json,subprocess
p=Path(path);row=json.loads(p.read_text()) if p.exists() else {}
service=subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState,SubState,Result'],capture_output=True,text=True,check=True)
row['service']=dict(line.split('=',1) for line in service.stdout.splitlines() if '=' in line)
if row and row.get('stage') not in ('analysis_finished','complete','failed','failed_with_preserved_evidence'):
 if 'pid' in row:
  cmd=Path('/proc')/str(row['pid'])/'cmdline'
  try:argv=[v.decode() for v in cmd.read_bytes().split(b'\\0') if v]
  except FileNotFoundError:argv=[]
  row['recorded_process_verified_live']=any(Path(v).name==script for v in argv) and output in argv
print(json.dumps(row))
'''
        try:
            result=subprocess.run(['ssh',args.host,'python3 -B -'],input=code,text=True,capture_output=True,check=True,timeout=30)
            row=json.loads(result.stdout)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired,json.JSONDecodeError) as error:
            return dict(observation_error=str(error))
        if row.get('stage') in ('failed','failed_with_preserved_evidence'):
            raise RuntimeError('worker cost stage failed; preserve '+str(path))
        if row.get('stage') not in ('analysis_finished','complete') and row['service']['ActiveState'] in ('failed','inactive'):
            raise RuntimeError('cost owner service is terminal before completion; reconcile '+str(path))
        return row
    def archive(root,name):
        destination=args.destination/name;destination.parent.mkdir(parents=True,exist_ok=True)
        execute([str(Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','512M','--jobs','1','--',
            'python3','-B',str(REPO/'scripts/byconstruction/dataset_mirror.py'),'--root',str(root),'--nfs',
            '--cold-readback','--artifact-kind','training','--destination',str(destination)],'archive-'+name)
    save('waiting_for_complete_outer_finalists')
    try:
        while not ready(args.study):
            save('waiting_for_complete_outer_finalists');time.sleep(30)
        save('submitting_admitted_cost_preparation')
        request=submit(args);record=json.loads(request.read_text());prepared=Path(record['result_output'])
        prep_unit=json.loads((args.output/'preparation/launch/LAUNCH.json').read_text())['rows'][0]['unit']
        while True:
            row=observed(request.with_name('STATUS.json'),'shaping_cost_finalists.py',prep_unit,prepared)
            if row.get('stage'):
                if row['request_sha256']!=sha(request):raise ValueError('cost preparation request differs')
                if row['stage']=='analysis_finished':break
            save('waiting_for_cost_preparation',worker_observation=row);time.sleep(30)
        save('collecting_prepared_finalists')
        bundle=args.study.parent/'worker-bundles'/('cost-prepare-'+record['cell']['id']+'.tar')
        remote([str(Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','512M','--jobs','1','--',
            'python3','-B',REPO/'scripts/rev4_featpot/shaping_cost_worker.py','pack','--request',request,'--bundle',bundle],
            'pack-prepared')
        transport=args.output/'transport';transport.mkdir()
        execute(['rsync','-rt','--ignore-existing',args.host+':'+str(bundle),args.host+':'+str(bundle)+'.json',str(transport)+'/'],'collect-prepared')
        from shaping_cost_worker import collect
        collect(request,transport/bundle.name,json.loads((transport/(bundle.name+'.json')).read_text())['bundle_sha256'])
        archive(prepared,'prepared')
        save('registering_vector_timing')
        from shaping_vector_bench import register
        root=args.output/'vector'
        register(args.study,prepared,request,args.instrument,args.qualification,args.launcher,args.cpu_tool,root)
        # Existing prepared models stay shared on the worker; transfer only the
        # registered tools and metadata into a new benchmark directory.
        remote(['test','!','-e',root],'check-fresh-benchmark')
        execute(['rsync','-rt','--mkpath','--ignore-existing',str(root)+'/',args.host+':'+str(root)+'/'],'transfer-vector')
        unit='zensim-vector-cost-'+sha(root/'REGISTRATION.json')[:12]
        command=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=640M','--property=MemorySwapMax=0',
            '--property=Nice=19','--property=IOSchedulingClass=idle','--property=StandardOutput=append:'+str(root)+'.log',
            '--property=StandardError=append:'+str(root)+'.log',
            str(Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','512M','--jobs','1','--','python3','-B',
            str(REPO/'scripts/rev4_featpot/shaping_vector_bench.py'),'run','--study',str(args.study),'--root',str(root),'--cpus',args.cpus]
        (args.output/'BENCHMARK_SUBMITTED.json').write_bytes(encoded(dict(host=args.host,unit=unit,command=command,
            registration_sha256=sha(root/'REGISTRATION.json'))))
        save('submitting_vector_timing');remote(command,'submit-vector')
        while True:
            row=observed(root/'STATUS.json','shaping_vector_bench.py',unit,root)
            if row.get('stage'):
                if row['registration_sha256']!=sha(root/'REGISTRATION.json'):raise ValueError('vector registration differs')
                if row['stage']=='complete':break
            save('waiting_for_vector_timing',worker_observation=row);time.sleep(30)
        save('collecting_complete_vector_timing')
        # The terminal worker inventory binds every file before and after transfer.
        code='import sys,json;from pathlib import Path;sys.path.insert(0,'+repr(str(REPO/'scripts/rev4_featpot'))+');from shaping_worker_collect import inventory;print(json.dumps(inventory(Path('+repr(str(root))+'))))'
        def inventory():
            return json.loads(subprocess.run(['ssh',args.host,'python3 -B -c '+shlex.quote(code)],capture_output=True,text=True,check=True).stdout)
        before=inventory();execute(['rsync','-rt','--ignore-existing',args.host+':'+str(root)+'/',str(root)+'/'],'collect-vector')
        if before!=inventory():raise ValueError('terminal vector artifacts changed during collection')
        for name,pin in before.items():
            path=root/name
            if sha(path)!=pin['sha256'] or path.stat().st_size!=pin['bytes']:
                raise ValueError('collected vector artifact differs')
        (root/'WORKER_COLLECTION.json').write_bytes(encoded(dict(source_host=args.host,files=before,
            registration_sha256=sha(root/'REGISTRATION.json'),complete=True)))
        archive(root,'vector')
        save('complete',timed_models=48,fresh_memory_models=48,prepared_archive=str(args.destination/'prepared'),
            vector_archive=str(args.destination/'vector'))
        return status
    except BaseException as error:
        save('failed_with_preserved_evidence',error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','binaries','output','destination','instrument','qualification','launcher','cpu-tool'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--cpus',required=True)
    print(json.dumps(watch(p.parse_args())),flush=True)
