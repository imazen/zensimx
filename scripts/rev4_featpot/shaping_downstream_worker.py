"""Transport prepared outer/variant neural refits through their original gates."""
from pathlib import Path
from types import SimpleNamespace
import socket
from shaping_completion import DECLARATION_SHA256,REPO,sha


def arguments(study,cell,binaries,qualification,kind,human_qualification=None):
    if kind not in ('outer','variant','extension'):raise ValueError('registered downstream execution kind required')
    if kind=='variant' and human_qualification is None:raise ValueError('variant human qualification directory required')
    values=dict(study=study,cell=cell,binary_dir=binaries,neutral_qualification=qualification,
                human_qualification=human_qualification,attempt=1)
    if kind=='outer':
        from shaping_execution import load_registration
        _,cells,_=load_registration(study)
        found=[c for c in cells if c['id']==cell]
        if not found:
            # Conditional fresh-control identities are registered by admission.
            from shaping_outer_admission import resolve
            for outer in range(3):
                path=study/'selection'/f'outer-{outer}'/'OUTER_ADMISSION.json'
                if path.exists():
                    import json
                    # Discover the owning fold from its sealed metadata before
                    # resolving that fold's full evidence closure. Opening all
                    # three closures inflated one matched control to 3.66 GB.
                    raw = json.loads(path.read_text())
                    if any(c['id'] == cell for c in raw['additional_cells']):
                        found.extend(c for c in resolve(study,outer)[0]['additional_cells'] if c['id']==cell)
        if len(found)!=1:raise ValueError('one registered outer cell required')
        values['outer']=int(found[0]['partition'].split('-')[1])
    return SimpleNamespace(**values)


def admitted(args,kind,*,host=None):
    if kind=='outer':
        from shaping_outer_execution import admit
    elif kind=='variant':
        from shaping_variant_execution import admit
    else:
        from shaping_extension_worker import admit
    return admit(args,host=host)


def scope(context,kind):
    value=dict(cell=context['cell'],schedule=context['schedule'],outer_results_read=False,
        templates=[dict(source=str(path),run_sha256=sha(path/'RUN.json')) for path,_ in context['templates']])
    if kind=='outer':
        value.update(admission_sha256=sha(context['admission_path']),
            prepared_sha256=sha(context['admission_path'].parents[2]/'outer_inputs'/context['cell']['partition']/'PREPARED.json'))
    else:value['resolution']=context['resolution']
    return value


def resolve_inputs(study,cell,host,binaries,qualification,kind,human_qualification=None):
    from shaping_rescue_worker import HOSTS,reads,seal_inputs
    if host not in HOSTS:raise ValueError('approved downstream worker required')
    with reads() as paths:
        args=arguments(study,cell,binaries,qualification,kind,human_qualification)
        context=admitted(args,kind,host=host)
        if not (study/'policies/POLICIES.json').exists():
            raise ValueError('freeze public policy snapshots before downstream registration')
        if (study/'runs'/cell).exists():raise ValueError('preserve existing downstream attempts; never replay')
        resolution=scope(context,kind)
        for source,original in context['templates']:
            for name in ('RUN.json','RESULT.json','keep.txt','HEAD.json','SHAPING.json',
                         'COLLECTION_DERIVED_INPUTS.json','WORKER_COLLECTION.json'):
                if (source/name).exists():paths.add(source/name)
            declaration=Path(original['command'][original['command'].index('--derived-inputs')+1])
            if not (source/'COLLECTION_DERIVED_INPUTS.json').exists():
                if sha(declaration)!=DECLARATION_SHA256:raise ValueError('frozen declaration drift')
                paths.add(declaration)
        if kind=='outer':
            paths.add(study/'outer_inputs'/context['cell']['partition']/'PREPARED.json')
        elif kind=='variant':paths.update(map(Path,context['resolution']['file_sha256']))
    paths.add(Path(__file__).absolute())
    return seal_inputs(paths,study,context['cell'],host,binaries,qualification,resolution,
                       execution_kind=kind,human_qualification=human_qualification)


def reproduce(record,*,host=None):
    kind=record['execution_kind'];study=Path(record['study'])
    from shaping_runtime_equivalence import verify_binding
    verify_binding(study,record)
    args=arguments(study,record['cell']['id'],Path(record['binary_dir']),
                   Path(record['neutral_qualification']),kind,
                   Path(record['human_qualification']) if record['human_qualification'] else None)
    context=admitted(args,kind,host=socket.gethostname() if host is None else host)
    driver={'outer':'shaping_outer_execution.py','variant':'shaping_variant_execution.py','extension':'shaping_extension_execution.py'}[kind]
    command=['python3','-B',str(REPO/'scripts/rev4_featpot'/driver),
             '--study',str(study),'--binary-dir',record['binary_dir'],
             '--neutral-qualification',record['neutral_qualification'],'--cell',record['cell']['id']]
    if kind=='outer':command.extend(['--outer',str(args.outer)])
    elif kind=='variant':command.extend(['--human-qualification',record['human_qualification']])
    if record.get('streaming_memory') is not None:
        from shaping_streaming_memory import validate
        validate(record);command.extend(['--streaming-memory','2G'])
    return scope(context,kind),command
