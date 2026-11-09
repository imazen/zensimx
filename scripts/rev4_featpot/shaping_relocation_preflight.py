"""Read-only evidence for relocating an existing native phase; never signal or resume it."""
import argparse,json,os,re,shutil,socket,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_state_probe import MAX_BYTES,read_state


def native_identity(pid):
    root=Path('/proc')/str(pid)
    stat=(root/'stat').read_text().rsplit(')',1)[1].split()
    command=[os.fsdecode(v) for v in (root/'cmdline').read_bytes().split(b'\0') if v]
    if stat[0]=='Z' or not command or Path(command[0]).name!='zensim_mlp_train':
        raise ValueError('live native trainer required')
    return dict(pid=pid,start_ticks=stat[19],command=command)


def checkpoint_prefix(state,phase):
    rows=[]
    for line in state['checkpoint_receipts']:
        marker='checkpoint dump: '
        # Native state stores the JSON receipt, without the stdout prefix.
        value=json.loads(line.split(marker,1)[1] if marker in line else line)
        path=Path(value['path'])
        if (path.parent!=phase/'checkpoints' or not re.fullmatch('ckpt_epoch[0-9]{3}.bin',path.name)
            or value['epoch']>=state['epoch'] or path.stat().st_size!=value['bytes']
            or sha(path)!=value['sha256']):raise ValueError('snapshot checkpoint prefix differs')
        rows.append(value)
    if ({r['epoch'] for r in rows}!=set(range(state['epoch']))
        or len(rows)!=state['epoch']):raise ValueError('snapshot prefix coverage differs')
    return rows


def capture(study,cid,target,output):
    from shaping_snapshot_retention import phase_authority
    from shaping_extension import resolve
    from shaping_compute_pool import allowed
    hosts=allowed(study)
    if (not re.fullmatch('[0-9a-f]{20}',cid) or hosts is None
        or socket.gethostname() not in hosts or target not in hosts or target==socket.gethostname()):
        raise ValueError('distinct eligible workers and registered cell required')
    phase=study/'runs'/cid/'attempt-001'/'fit';binding=phase/'SNAPSHOT_BINDING.json'
    bound=json.loads(binding.read_text());identity=json.loads((phase/'RUN.json').read_text())
    parent=phase_authority(phase,identity);cell=parent['cell']
    if (cell['stage']!='exposure_extension' or cell['phase_epochs']!=[480]
        or resolve(study,cid)!=parent['resolution'] or parent['execution_host']!=socket.gethostname()
        or cell['id']!=cid or (phase/'RESULT.json').exists() or (phase.parent/'RESULT.json').exists()
        or bound['schema']!='native-snapshot-run-binding-v1' or bound['host']!=socket.gethostname()
        or bound['run']!=str(phase) or bound['run_sha256']!=sha(phase/'RUN.json')):
        raise ValueError('original live single-phase extension binding required')
    before=native_identity(bound['native_observation']['pid'])
    if (before['start_ticks']!=str(bound['native_observation']['start_ticks'])
        or before['command']!=bound['native_observation']['command']):raise ValueError('bound native owner differs')
    executable=Path(before['command'][0]);build=executable.parent/'BUILD.json'
    if sha(build)!=identity['build_sha256']:raise ValueError('native build differs')
    for name,pin in json.loads(build.read_text())['binary_sha256'].items():
        if sha(build.parent/name)!=pin:raise ValueError('native executable closure differs')
    store=Path(bound['snapshot_directory'])
    if store.parent!=study.parent/'shared-training-snapshots' or not re.fullmatch('[0-9a-f]{64}',store.name):
        raise ValueError('canonical original snapshot store required')
    # Open files stay intact across the native store's atomic replacement. A
    # disappearing directory entry is a bounded read race, not a trainer failure.
    for attempt in range(3):
        try:
            states=[(p,*read_state(p),p.stat().st_mtime_ns) for p in store.glob('state-*.snap')]
            if not 1<=len(states)<=2:raise ValueError('one or two committed native states required')
            selected=max(states,key=lambda row:row[1]);break
        except FileNotFoundError:
            if attempt==2:raise
    path,generation,state,pin,raw,mtime=selected
    if state['epoch']>=480 or not 0<=state['pair']<50000:raise ValueError('unfinished registered native cursor required')
    prefixes=checkpoint_prefix(state,phase)
    if native_identity(before['pid'])!=before:raise ValueError('native owner changed during preflight')
    output.mkdir(parents=True,exist_ok=False)
    (output/'witness.snap').write_bytes(raw)
    sources=[binding,phase/'RUN.json',phase.parent/'RUN.json',build]
    sources.extend(Path(p) for p in identity['manifest_sha256'])
    for name,expected in identity['manifest_sha256'].items():
        if sha(name)!=expected:raise ValueError('native phase manifest drift')
    files={}
    for i,source in enumerate(dict.fromkeys(sources)):
        dest=output/f'{i:02}-{source.name}';shutil.copy2(source,dest)
        if sha(source)!=sha(dest):raise ValueError('preflight metadata changed during capture')
        files[str(source)]=dict(copy=dest.name,sha256=sha(dest))
    env=dict(v.split('=',1) for v in (Path('/proc')/str(before['pid'])/'environ').read_bytes().decode().split('\0') if '=' in v)
    selected_env={k:v for k,v in env.items() if k.startswith(('ZENSIM_MONOTONE_RESEARCH_',
        'ZENSIM_INPUT_SHAPING_','ZENSIM_PLAIN_OPTIMIZER_','ZENSIM_TRAIN_'))
        or k in ('ZENSIM_OPTIMIZER_DIAGNOSTICS','ZENSIM_PHASE_SNAPSHOTS')}
    unknown=sorted(k for k in env if k.startswith('ZENSIM_') and k not in selected_env)
    if unknown:raise ValueError('unclassified scientific environment keys: '+', '.join(unknown))
    result=dict(schema='native-phase-relocation-preflight-v1',checked_utc=datetime.now(timezone.utc).isoformat(),
        cell=cell,source_host=socket.gethostname(),proposed_target_host=target,native=before,
        build_sha256=sha(build),trainer_sha256=sha(executable),binding_sha256=sha(binding),
        snapshot=dict(original_path=str(path),generation=generation,sha256=pin,bytes=len(raw),
            epoch=state['epoch'],pair=state['pair'],sampled_mtime_ns=mtime,
            sampled_age_seconds=(time.time_ns()-mtime)/1e9,payload_checksum_verified=True),
        checkpoint_prefix=prefixes,metadata=files,scientific_environment=selected_env,
        native_state_modified=False,native_signalled=False,training_started=False,training_replayed=False,
        authorization='Preflight only; does not authorize stopping a live fit or accepting a recovered attempt.',
        cursor_note='Captured witness only. Any authorized move must capture the then-current state after reconciling its original owner.')
    (output/'PREFLIGHT.json').write_bytes(encoded(result));return result


def preflight(study,placements,output):
    from shaping_storage import reservation
    values=json.loads(placements.read_text())
    if (values['plan_sha256']!=sha(study/'PLAN.json') or values['outer_results_read']
        or not 1<=len(values['placements'])<=8):raise ValueError('bounded plan-bound placement proposal required')
    allowance=len(values['placements'])*MAX_BYTES+16*1024**2
    if allowance>256*1024**2:raise ValueError('proposal exceeds staging bound')
    if output.exists():raise ValueError('preserve existing relocation evidence')
    minimum=json.loads((study/'PLAN.json').read_text())['minimum_ssd_free_bytes']
    with reservation(study,output,minimum,bytes_required=allowance):
        output.mkdir(parents=True)
        rows=[capture(study,cid,target,output/cid) for cid,target in values['placements'].items()]
        result=dict(schema='native-relocation-proposal-evidence-v1',checked_utc=datetime.now(timezone.utc).isoformat(),
            cells=rows,placements_sha256=sha(placements),driver_sha256=sha(__file__),
            reserved_staging_bytes=allowance,payload_bytes_before_summary=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
            native_signalled=False,training_started=False,scientific_completion_claimed=False)
        (output/'SUMMARY.json').write_bytes(encoded(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','placements','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=preflight(a.study,a.placements,a.output)
    print(json.dumps(dict(captured_cells=len(result['cells']),output=str(a.output),native_signalled=False)))
