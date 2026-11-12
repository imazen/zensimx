"""Recover externally interrupted same-host native phases from committed full state."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
from shaping_completion import encoded, sha


def capture(study, cid, output):
    from shaping_snapshot_retention import closed_store, phase_authority
    from shaping_state_probe import read_state
    from shaping_relocation_preflight import checkpoint_prefix
    host=socket.gethostname()
    if host not in ("i134","r3500","r3800x","r5600g"):
        raise ValueError("approved recovery host required")
    root=study/"runs"/cid/"attempt-001";phase=root/"fit"
    bound=json.loads((phase/"SNAPSHOT_BINDING.json").read_text())
    identity=json.loads((phase/"RUN.json").read_text())
    parent=phase_authority(phase,identity)
    boot_seconds=int(next(line.split()[1] for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime ")))
    observed=datetime.fromisoformat(bound["checked_utc"]).timestamp()
    if (bound["host"]!=host or parent["execution_host"]!=host or boot_seconds<=observed
        or bound["run_sha256"]!=sha(phase/"RUN.json") or bound["run"]!=str(phase)
        or (phase/"RESULT.json").exists() or (root/"RESULT.json").exists()
        or (root/"pretraining").exists()):
        raise ValueError("same-host single-phase interruption by a later reboot required")
    command=bound["native_observation"]["command"]
    if command!=identity["command"][identity["command"].index("--")+1:]:
        raise ValueError("native command binding differs")
    build=Path(command[0]).parent/"BUILD.json"
    if sha(build)!=identity["build_sha256"]:raise ValueError("frozen native build differs")
    for name,pin in json.loads(build.read_text())["binary_sha256"].items():
        if sha(build.parent/name)!=pin:raise ValueError("frozen executable differs")
    for name,pin in identity["manifest_sha256"].items():
        if sha(name)!=pin:raise ValueError("phase manifest differs")
    store=Path(bound["snapshot_directory"])
    if store.parent!=study.parent/"shared-training-snapshots":
        raise ValueError("registered snapshot root required")
    environment=dict(TMPDIR=str(Path.home()/"tmp"),OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",RAYON_NUM_THREADS="1",ZENSIM_OPTIMIZER_DIAGNOSTICS="1",
        ZENSIM_PHASE_SNAPSHOTS="1",ZENSIM_TRAIN_SNAPSHOT_ROOT=str(store.parent))
    for filename,prefix in (("OPTIMIZER.json","ZENSIM_PLAIN_OPTIMIZER"),
                            ("HEAD.json","ZENSIM_MONOTONE_RESEARCH"),
                            ("SHAPING.json","ZENSIM_INPUT_SHAPING")):
        path=phase/filename
        if path.exists():
            if identity["manifest_sha256"].get(str(path))!=sha(path):
                raise ValueError("scientific environment lacks its original manifest pin")
            environment[prefix+"_MANIFEST"]=str(path);environment[prefix+"_SHA256"]=sha(path)
    if output.exists():raise ValueError("preserve existing reboot capture")
    # Reject live stores and unfinished writes before copying anything.
    with closed_store(store):
        states=[(p,*read_state(p)) for p in store.glob("state-*.snap")]
        if not 1<=len(states)<=2:raise ValueError("one or two committed states required")
        selected=max(states,key=lambda r:r[1]);state=selected[2]
        if not 0<state["epoch"]<identity["epochs"] or state["pair"]!=0:
            raise ValueError("committed unfinished epoch-boundary state required")
        checkpoint_prefix(state,phase)
        size=sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        if size>2*1024**3 or shutil.disk_usage(study).free<size+34*1024**3:
            raise ValueError("bounded original-attempt capture and SSD headroom required")
        output.mkdir(parents=True)
        shutil.copytree(root,output/"original-attempt")
        saved=[]
        for path,generation,value,pin,raw in states:
            destination=output/store.name/path.name;destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(raw)
            saved.append(dict(captured=str(destination),sha256=pin,epoch=value["epoch"],generation=generation))
        chosen=next(r for r in saved if r["sha256"]==selected[3])
        row=dict(schema="native-reboot-capture-v1",source_host=host,phase=str(phase),
            native=bound["native_observation"],environment=environment,snapshots=saved,selected=chosen,
            exact_state_checksum_verified=True,completed_epochs_replayed=0,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            original_binding_sha256=sha(phase/"SNAPSHOT_BINDING.json"),
            checked_utc=datetime.now(timezone.utc).isoformat(),driver_sha256=sha(__file__))
        (output/"CAPTURE.json").write_bytes(encoded(row))
        return dict(cell=cid,epoch=state["epoch"],capture=str(output/"CAPTURE.json"),preserved_bytes=size)


def work(study, capture_path, output, request):
    from shaping_drain_resume import resume
    from shaping_continuation_finalize import finalize
    row=json.loads(capture_path.read_text());job=json.loads(request.read_text())
    if job["cell"]!=row["native"]["cell"] or job["host"]!=socket.gethostname():
        raise ValueError("original worker request differs")
    status=request.with_name("STATUS.json")
    prior=json.loads(status.read_text())
    if prior["request_sha256"]!=sha(request) or prior["stage"] not in ("running","starting"):
        raise ValueError("preserve conflicting worker status")
    archive=status.with_name("STATUS.before-reboot.json")
    if archive.exists():raise ValueError("preserve previous recovery attempt")
    archive.write_bytes(status.read_bytes())
    resume(capture_path,output,study)
    result=finalize(study,capture_path,output)
    proof=study/"runs"/job["cell"]["id"]/"attempt-001/CONTINUATION.json"
    status.write_bytes(encoded(dict(prior,stage="trained",exit_code=0,
        reboot_recovery=dict(pid=os.getpid(),capture_sha256=sha(capture_path),
            continuation_sha256=sha(proof),completed_epochs_replayed=result["completed_epochs_replayed"]))))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("capture","work"))
    for n in ("study","output","capture","request"):p.add_argument("--"+n,type=Path)
    p.add_argument("--cell");a=p.parse_args()
    result=capture(a.study,a.cell,a.output) if a.mode=="capture" else work(a.study,a.capture,a.output,a.request)
    print(json.dumps(result),flush=True)
