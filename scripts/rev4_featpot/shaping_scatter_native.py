"""Compute detailed Zenstats from published scalar pairs on an admitted worker."""
import argparse,csv,io,json,math,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
for name in ('RAYON_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from shaping_completion import encoded,sha
from shaping_rescue_dispatch import reserve_worker,release_worker
from shaping_storage import reservation
from shaping_reduced_scope import admit_compute
from shaping_scatter_prepare import write_atomic


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    return value


def one(study,panel,job,output):
    path=Path(job['path']);pin=job['report_sha256'];destination=output/(pin+'.json')
    if sha(path)!=job['sha256']:raise ValueError('scatter scalar-input hash differs')
    from shaping_scatter_prepare import read_points
    data=read_points(path)
    if data['schema']!='zensim-inner-scatter-v1' or data['outer_results_read'] or data['cell']!=job['cell'] or data['report_sha256']!=pin:
        raise ValueError('inner scalar-panel scope differs')
    if destination.exists():
        old=json.loads(destination.read_text())
        if old['input_sha256']!=job['sha256'] or old['panel_sha256']!=sha(panel):raise ValueError('preserve changed detailed panel')
        return dict(report=pin,stage='reused',path=str(destination))
    key='zenstats-'+pin[:24]
    while not reserve_worker(study,key)['can_start']:time.sleep(10)
    try:
        with reservation(study,output/(pin+'-evidence'),30*1024**3,bytes_required=64*1024**2):
            work=output/(pin+'-evidence');work.mkdir(exist_ok=False)
            vectors=[];labels=[];expected={}
            for pop,population in data['populations'].items():
                for role,row in population['roles'].items():
                    label=pop+'|'+role;x=row['predicted'];y=row['scaled_target']
                    if len(x)!=row['summary']['rows'] or len(y)!=len(x) or len(x)<3:
                        raise ValueError('detailed panel denominator differs')
                    if not all(math.isfinite(v) for v in [*x,*y]):raise ValueError('nonfinite scalar panel input')
                    vectors.append((label,x,y));labels.append(label);expected[label]=row['summary']
            batch=''.join(label+'\t'+','.join(map(repr,x))+'\t'+','.join(map(repr,y))+'\n' for label,x,y in vectors)
            (work/'pairs.batch.tsv').write_text(batch)
            results={}
            with (work/'native.log').open('x') as log:
                for mode,extra in (('full',['--raw-errors']),('signed',['--signed-quality'])):
                    command=[str(panel),'--batch',str(work/'pairs.batch.tsv'),'--stats','full',*extra]
                    result=subprocess.run(command,capture_output=True,text=True,timeout=600,check=True)
                    (work/(mode+'.stdout')).write_text(result.stdout);log.write(result.stderr)
                    if mode=='signed':rows=[json.loads(line) for line in result.stdout.splitlines()]
                    else:
                        reader=csv.DictReader(io.StringIO(result.stdout),delimiter='\t');rows=[]
                        for record in reader:
                            rows.append({k:(v if k=='label' else int(v) if k in ('n','n_dropped') else float(v)) for k,v in record.items()})
                    if [r['label'] for r in rows]!=labels:raise ValueError('native panel row order differs')
                    results[mode]={r['label']:r for r in rows}
                for label,x,y in vectors:
                    file=work/(label.replace('|','-')+'.tsv')
                    file.write_text('predicted\ttarget\n'+''.join(repr(a)+'\t'+repr(b)+'\n' for a,b in zip(x,y)))
                    command=[str(panel),'--input',str(file),'--json','--scatter']
                    result=subprocess.run(command,capture_output=True,text=True,timeout=600,check=True)
                    log.write(result.stderr);value=json.loads(result.stdout)
                    if value['n']!=len(x):raise ValueError('native scatter denominator differs')
                    results.setdefault('scatter',{})[label]=value
            for label in labels:
                if any(results[mode][label]['n']!=expected[label]['rows'] or results[mode][label]['n_dropped']!=0 for mode in ('full','signed')):
                    raise ValueError('native panel dropped rows')
                measured=expected[label]['pooled']
                if measured is not None and abs(results['signed'][label]['srocc_signed']-measured)>1e-12:
                    raise ValueError('native signed rank differs from saved population')
                results['signed'][label]['saved_correlation_status']='measured' if measured is not None else 'undefined_constant_or_singleton'
            result=clean(dict(schema='zensim-native-detailed-panel-v1',cell=data['cell'],report_sha256=pin,input_sha256=job['sha256'],
                panel_sha256=sha(panel),statistics=results,outer_results_read=False,
                target_scale=data['target_scaling'],owner='zenmetrics-stats through the frozen native panel CLI',
                signed_metrics='Quality-oriented signed SROCC, Kendall and increasing-logistic PLCC; separate from legacy magnitude-oriented full-panel fields.'))
            payload=json.dumps(result,allow_nan=False,separators=(',',':'))+'\n'
            if len(payload.encode())>16*1024**2:raise ValueError('detailed panel exceeds its 16-MiB allowance')
            with destination.open('x') as stream:stream.write(payload)
            return dict(report=pin,stage='complete',path=str(destination),sha256=sha(destination))
    finally:release_worker(study,key)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','panel','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();admit_compute(a.study)
    if not 1<=a.workers<=4:p.error('one to four native statistics workers required')
    manifest=json.loads(a.manifest.read_text())
    if manifest['schema']!='zensim-native-dashboard-jobs-v1' or manifest['outer_results_read'] or manifest['panel_sha256']!=sha(a.panel):raise ValueError('native detailed-panel manifest differs')
    a.output.mkdir(parents=True,exist_ok=True);record=dict(pid=os.getpid(),stage='running',completed=[],failed={},manifest_sha256=sha(a.manifest))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(one,a.study,a.panel,job,a.output):job['report_sha256'] for job in manifest['jobs']}
        for future in as_completed(futures):
            pin=futures[future]
            try:record['completed'].append(future.result())
            except Exception as error:record['failed'][pin]=str(error)
            write_atomic(a.output/'STATUS.json',record)
    record['stage']='failed_with_preserved_evidence' if record['failed'] else 'complete';write_atomic(a.output/'STATUS.json',record)
    print(json.dumps(dict(stage=record['stage'],completed=len(record['completed']),failed=record['failed'])),flush=True)
