"""Carry complete source-bootstrap evidence to an approved native statistics host."""
import argparse,hashlib,json,socket
from pathlib import Path
from types import SimpleNamespace
from shaping_completion import REPO,encoded,sha


def arguments(study,draws,binaries,result,comparison=None):
    return SimpleNamespace(study=study,draws=draws,binary_dir=binaries,output=result,comparison=comparison)


def scope(context):
    from shaping_rescue_worker import source_identity
    pins={source_identity(name):pin for name,pin in context['pins'].items()}
    if len(pins)!=len(context['pins']):raise ValueError('ambiguous bootstrap source identity')
    return dict(input_sha256=pins,comparison=context['spec'],
        cells=[row['cell'] for row in context['outputs']],qualifications=context['qualifications'],
        outer_results_read=True,final_holdout_read=False)


def output_scope(study,result):
    return (result.is_absolute() and '..' not in result.parts and result.resolve()==result
        and result.is_relative_to(study.parent) and not result.exists())


def register(study,draws,binaries,result,host,output,comparison=None):
    from shaping_bootstrap import admit
    from shaping_rescue_worker import HOSTS,reads,seal_inputs
    if host not in HOSTS:raise ValueError('approved bootstrap host required')
    if not output_scope(study,result):raise ValueError('new study-local analysis output required')
    result.parent.mkdir(parents=True,exist_ok=True)
    args=arguments(study,draws,binaries,result,comparison)
    with reads() as paths:context=admit(args,host=host)
    resolution=scope(context)
    identifier=hashlib.sha256(encoded(dict(plan=sha(study/'PLAN.json'),draws=sha(draws/'REGISTRATION.json'),
        build=sha(binaries/'BUILD.json'),comparison=context['spec']))).hexdigest()[:20]
    cell=dict(stage='bootstrap',family='source_bootstrap',partition='outer-all',id=identifier)
    record=seal_inputs(paths,study,cell,host,binaries,None,resolution,execution_kind='bootstrap')
    record.update(draws=str(draws),comparison=str(comparison) if comparison else None,result_output=str(result),
                  bootstrap_driver_sha256=sha(__file__),final_holdout_read=False)
    output.mkdir(parents=True,exist_ok=False);(output/'REQUEST.json').write_bytes(encoded(record))
    assignment=dict(host=host,cells=[identifier],plan_sha256=sha(study/'PLAN.json'),build_sha256=sha(binaries/'BUILD.json'))
    (output/'ASSIGNMENT.json').write_bytes(encoded(assignment));return record


def reproduce(record,*,host=None):
    from shaping_bootstrap import admit
    study=Path(record['study']);result=Path(record['result_output'])
    if (record['bootstrap_driver_sha256']!=sha(__file__) or record['final_holdout_read'] is not False
        or not output_scope(study,result)):
        raise ValueError('bootstrap source or new output scope differs')
    result.parent.mkdir(parents=True,exist_ok=True)
    args=arguments(study,Path(record['draws']),Path(record['binary_dir']),result,
        Path(record['comparison']) if record['comparison'] else None)
    resolution=scope(admit(args,host=socket.gethostname() if host is None else host))
    command=['python3','-B',str(REPO/'scripts/rev4_featpot/shaping_bootstrap.py'),'--study',str(study),
        '--draws',str(args.draws),'--binary-dir',str(args.binary_dir),'--output',str(result)]
    if args.comparison:command+=['--comparison',str(args.comparison)]
    return resolution,command


def validate_result(record,directory):
    from shaping_rescue_worker import source_identity
    marker=json.loads((directory/'COMPLETE.json').read_text());summary=json.loads((directory/'SUMMARY.json').read_text())
    if (marker['summary_sha256']!=sha(directory/'SUMMARY.json') or summary['final_holdout_read'] is not False
        or summary['outer_results_read'] is not True or summary['samples_sha256']!=sha(directory/'samples.npz')):
        raise ValueError('complete native bootstrap result identity differs')
    pins={source_identity(name,summary['source_root']):pin for name,pin in summary['input_sha256'].items()}
    if len(pins)!=len(summary['input_sha256']) or pins!=record['resolution']['input_sha256']:
        raise ValueError('bootstrap result inputs differ from reproduced admission')
    if summary['convergence_qualifications']!=record['resolution']['qualifications']:
        raise ValueError('bootstrap result convergence qualifications differ')
    if (summary.get('comparison')!=record['resolution']['comparison']
        or set(summary['outer_population_reports'])!={c['id'] for c in record['resolution']['cells']}):
        raise ValueError('bootstrap result comparison or paired model coverage differs')


def pack(request,bundle):
    from shaping_terminal_transport import pack as original
    return original(request,bundle,validate_result)


def collect(request,bundle,pin):
    from shaping_terminal_transport import collect as original
    result=original(request,bundle,pin,reproduce,validate_result)
    return dict(result,summary_sha256=sha(Path(result['result'])/'SUMMARY.json'),scientific_fit_count=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('register','pack','collect'))
    for name in ('study','draws','binary-dir','result-output','output','request','bundle'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--host');parser.add_argument('--comparison',type=Path);parser.add_argument('--bundle-sha256');a=parser.parse_args()
    if a.mode=='register':
        r=register(a.study,a.draws,a.binary_dir,a.result_output,a.host,a.output,a.comparison)
        r=dict(cell=r['cell'],input_bytes=r['input_bytes'],files=len(r['files']),host=r['host'])
    elif a.mode=='pack':r=pack(a.request,a.bundle)
    else:r=collect(a.request,a.bundle,a.bundle_sha256)
    print(json.dumps({k:v for k,v in r.items() if k!='files'}),flush=True)
