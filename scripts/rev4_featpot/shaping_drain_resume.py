"""Run a preserved native continuation; scientific collection remains a separate gate."""
import argparse,json,os,shutil,socket,subprocess,time
from pathlib import Path
from shaping_completion import sha,encoded
from shaping_state_probe import read_state


def resume(capture,output,study):
    row=json.loads(capture.read_text());native=row['native'];cid=native['cell']['id']
    if socket.gethostname() not in ('r3500','r3800x','r5600g','i134') or output.exists():
        raise ValueError('eligible worker and fresh continuation output required')
    command=list(native['command']);phase=Path(row['phase']);original=Path(native['run'])
    staged=capture.parent/'original-attempt'
    if not staged.is_dir() or not row['exact_state_checksum_verified']:
        raise ValueError('complete preserved original attempt required')
    selected=Path(row['selected']['captured']);_,state,pin,_=read_state(selected)
    if pin!=row['selected']['sha256'] or state['epoch']!=row['selected']['epoch']:
        raise ValueError('preserved optimizer state differs')
    phase_record=json.loads((staged/row.get('recovery_phase','fit')/'RUN.json').read_text())
    build=Path(command[0]).parent/'BUILD.json'
    if sha(build)!=phase_record['build_sha256']:raise ValueError('original native build differs')
    for name,digest in json.loads(build.read_text())['binary_sha256'].items():
        if sha(build.parent/name)!=digest:raise ValueError('native executable closure differs')
    from shaping_rescue_dispatch import reserve_worker,bind_worker,release_worker
    from shaping_snapshot_retention import capacity
    memory=int(native['memory_max'])
    while not reserve_worker(study,cid,memory_allowance_bytes=memory,warmup_until_checkpoint=True,snapshot_phases=1)['can_start']:
        time.sleep(10)
    try:
        if shutil.disk_usage(study).free<34*1024**3:raise ValueError('continuation SSD reserve unavailable')
        original.parent.mkdir(parents=True,exist_ok=True)
        if original.exists():
            if sha(original/'RUN.json')!=sha(staged/'RUN.json'):raise ValueError('preserve different original attempt at target')
        else:shutil.copytree(staged,original)
        store=Path(native['snapshot_directory'])
        # Account for the new store under the same capacity gate as other fits.
        if not capacity(study,cid)['can_start']:raise ValueError('shared snapshot reservation unavailable')
        store.mkdir(exist_ok=True)
        for saved in row['snapshots']:
            path=Path(saved['captured']);dest=store/path.name
            if sha(path)!=saved['sha256']:raise ValueError('snapshot changed during transfer')
            if dest.exists():
                if sha(dest)!=saved['sha256']:raise ValueError('preserve different native snapshot identity')
            else:shutil.copy2(path,dest)
        for receipt in state['checkpoint_receipts']:
            value=json.loads(receipt.split('checkpoint dump: ',1)[1] if 'checkpoint dump: ' in receipt else receipt)
            if sha(value['path'])!=value['sha256']:raise ValueError('completed checkpoint prefix differs')
        for path,digest in phase_record['manifest_sha256'].items():
            if sha(path)!=digest:raise ValueError('original phase input differs')
        output.mkdir(parents=True);(output/'checkpoints').mkdir()
        for flag,path in [('--out',output/'best.bin'),('--dump-checkpoints-dir',output/'checkpoints')]:
            if command.count(flag)!=1:raise ValueError('unique native output argument required')
            command[command.index(flag)+1]=str(path)
        env=os.environ.copy();env.update(row['environment'])
        env.update(ZENSIM_TRAIN_RESUME='1',ZENSIM_TRAIN_RESUME_CHECKPOINT_ORIGIN=str(phase/'checkpoints'))
        record=dict(cell=native['cell'],source_capture=str(capture),source_capture_sha256=sha(capture),
            source_host=json.loads((staged/'RUN.json').read_text())['execution_host'],execution_host=socket.gethostname(),source_epoch=state['epoch'],source_pair=state['pair'],
            snapshot_sha256=pin,build_sha256=sha(build),pid=os.getpid(),stage='starting',command=command,
            scientific_collection_pending=True,completed_epochs_replayed=0)
        (output/'RUN.json').write_bytes(encoded(record))
        with (output/'train.log').open('x') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            bind_worker(study,cid,process.pid);record.update(stage='running',native_pid=process.pid)
            (output/'STATUS.json').write_bytes(encoded(record));code=process.wait()
        record.update(stage='native_complete_pending_scientific_collection' if code==0 else 'failed_preserved',exit_code=code)
        (output/'RESULT.json').write_bytes(encoded(record))
        if code:raise RuntimeError('preserved native continuation refused or failed')
    finally:release_worker(study,cid)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','output','study'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();resume(a.capture,a.output,a.study)
