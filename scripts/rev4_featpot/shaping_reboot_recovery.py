"""Recover externally interrupted same-host native phases from committed full state."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
from shaping_completion import encoded, sha


def capture(study, cid, output, phase_name="fit"):
    from shaping_snapshot_retention import closed_store, phase_authority
    from shaping_state_probe import read_state
    from shaping_relocation_preflight import checkpoint_prefix
    host=socket.gethostname()
    if host not in ("i134","r3500","r3800x","r5600g"):
        raise ValueError("approved recovery host required")
    root=study/"runs"/cid/"attempt-001";phase=root/phase_name
    if phase_name not in ("fit","pretraining"):raise ValueError("registered recovery phase required")
    bound=json.loads((phase/"SNAPSHOT_BINDING.json").read_text())
    identity=json.loads((phase/"RUN.json").read_text())
    parent=phase_authority(phase,identity)
    boot_seconds=int(next(line.split()[1] for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime ")))
    observed=datetime.fromisoformat(bound["checked_utc"]).timestamp()
    if (bound["host"]!=host or parent["execution_host"]!=host or boot_seconds<=observed
        or bound["run_sha256"]!=sha(phase/"RUN.json") or bound["run"]!=str(phase)
        or (phase/"RESULT.json").exists() or (root/"RESULT.json").exists()
        or (phase_name=="fit" and (root/"pretraining").exists())
        or (phase_name=="pretraining" and (root/"fit").exists())):
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
        row=dict(schema="native-reboot-capture-v1",source_host=host,phase=str(phase),recovery_phase=phase_name,
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
    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    archive=status.with_name("STATUS.before-reboot-"+boot_id+".json")
    if archive.exists():raise ValueError("preserve previous recovery attempt")
    archive.write_bytes(status.read_bytes())
    # Native completion may precede an interruption in scientific publication.
    # The finalizer below verifies the existing result and its full closure;
    # never rerun a trainer when a terminal native receipt already exists.
    if not (output/"RESULT.json").exists():
        resume(capture_path,output,study)
    for item in row.get("predecessor_evidence",[]):
        source=Path(item["path"])
        if sha(source)!=item["sha256"]:raise ValueError("predecessor evidence changed")
        dest=study/"runs"/job["cell"]["id"]/"attempt-001/continuation-evidence/predecessors"/item["name"]
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if dest.stat().st_size != source.stat().st_size or sha(dest) != item["sha256"]:
                raise ValueError("preserve different predecessor recovery evidence")
        else:
            shutil.copy2(source,dest)
    result=finalize(study,capture_path,output)
    if row.get("recovery_phase")=="pretraining":
        finish_pretrained_controller(study,request)
    proof=study/"runs"/job["cell"]["id"]/"attempt-001"/("pretraining/CONTINUATION.json" if row.get("recovery_phase")=="pretraining" else "CONTINUATION.json")
    status.write_bytes(encoded(dict(prior,stage="trained",exit_code=0,
        reboot_recovery=dict(pid=os.getpid(),capture_sha256=sha(capture_path),
            continuation_sha256=sha(proof),completed_epochs_replayed=result["completed_epochs_replayed"]))))
    return result


def capture_continuation(study,cid,prior_capture,prior_native,output):
    """Preserve a later interruption without replaying the earlier continuation."""
    from shaping_snapshot_retention import closed_store
    from shaping_state_probe import read_state
    from shaping_relocation_preflight import checkpoint_prefix
    prior=json.loads(prior_capture.read_text());native=json.loads((prior_native/"RUN.json").read_text())
    root=study/"runs"/cid/"attempt-001";phase=root/"fit"
    boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if (boot==prior["boot_id"] or prior["native"]["cell"]["id"]!=cid
        or native["source_capture_sha256"]!=sha(prior_capture)
        or native["execution_host"]!=socket.gethostname()
        or (root/"RESULT.json").exists() or (prior_native/"RESULT.json").exists()):
        raise ValueError("later reboot of an unfinished original continuation required")
    store=Path(prior["native"]["snapshot_directory"])
    with closed_store(store):
        states=[(p,*read_state(p)) for p in store.glob("state-*.snap")]
        selected=max(states,key=lambda r:r[1]);state=selected[2]
        if not prior["selected"]["epoch"]<=state["epoch"]<json.loads((phase/"RUN.json").read_text())["epochs"]:
            raise ValueError("later committed optimizer cursor required")
        checkpoint_prefix(state,prior_native)
        size=sum(p.stat().st_size for p in (prior_capture.parent/"original-attempt").rglob("*") if p.is_file())
        size+=sum(p.stat().st_size for p in (prior_native/"checkpoints").glob("*.bin"))
        if output.exists() or size>2*1024**3 or shutil.disk_usage(study).free<size+34*1024**3:
            raise ValueError("fresh bounded continuation capture required")
        output.mkdir(parents=True)
        original=output/"original-attempt"
        shutil.copytree(prior_capture.parent/"original-attempt",original)
        for path in (prior_native/"checkpoints").glob("ckpt_epoch*.bin"):
            epoch=int(path.stem.removeprefix("ckpt_epoch"))
            if epoch>state["epoch"]:raise ValueError("unreconciled checkpoint tail")
            dest=original/"fit/checkpoints"/path.name
            if dest.exists():
                if sha(dest)!=sha(path):raise ValueError("predecessor raw checkpoint differs")
            else:shutil.copy2(path,dest)
        raw_log=original/"fit/train.log"
        raw_log.write_text(raw_log.read_text()+"\n"+(prior_native/"train.log").read_text())
        saved=[]
        for path,generation,value,pin,raw in states:
            dest=output/store.name/path.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(raw)
            saved.append(dict(captured=str(dest),sha256=pin,epoch=value["epoch"],generation=generation))
        row=dict(prior,boot_id=boot,snapshots=saved,
            selected=next(v for v in saved if v["sha256"]==selected[3]),
            previous_capture=str(prior_capture),previous_capture_sha256=sha(prior_capture),
            predecessor_evidence=[dict(path=str(p),sha256=sha(p),name=name) for p,name in (
                (prior_capture,"initial-CAPTURE.json"),(prior_native/"RUN.json","first-native-RUN.json"),
                (prior_native/"train.log","first-native-train.log"),
                (prior_capture.parent/"original-attempt/fit/train.log","initial-original-train.log"))],
            original_attempt_assembly="Original identity and manifests; exact verified checkpoint prefixes and both raw epoch logs.",
            checked_utc=datetime.now(timezone.utc).isoformat(),driver_sha256=sha(__file__))
        (output/"CAPTURE.json").write_bytes(encoded(row))
        return dict(cell=cid,epoch=state["epoch"],capture=str(output/"CAPTURE.json"))


def finish_pretrained_controller(study,request):
    """Run only the originally registered, unstarted fit after recovered pretraining."""
    import subprocess,time
    from shaping_downstream_worker import arguments,admitted,scope
    from shaping_rescue_execution import phase
    from shaping_variant_execution import command,optimizer,structural_terms
    from shaping_streaming_memory import native_cap
    from shaping_rescue_dispatch import reserve_worker,release_worker,bind_worker
    job=json.loads(request.read_text());cid=job["cell"]["id"];root=study/"runs"/cid/"attempt-001"
    initial=json.loads((root/"RUN.json").read_text());before=json.loads((root/"pretraining/RESULT.json").read_text())
    proof=root/"pretraining/CONTINUATION.json"
    if (job["execution_kind"]!="variant" or before["stage"]!="trained" or before["exit_code"]!=0
        or not proof.exists() or (root/"fit").exists() or (root/"RESULT.json").exists()):
        raise ValueError("completed recovered pretraining and untouched registered fit required")
    args=arguments(study,cid,Path(job["binary_dir"]),Path(job["neutral_qualification"]),
                   "variant",Path(job["human_qualification"]))
    context=admitted(args,"variant",host=job["host"])
    if scope(context,"variant")!=job["resolution"]:raise ValueError("original controller admission differs")
    schedule=context["schedule"]
    if schedule["phase_epochs"]!=[120,120]:raise ValueError("registered paired pretraining exposure required")
    shape=json.loads(Path(context["resolution"]["initial_shape"]).read_text())
    prefix=root/"pretraining/checkpoints/ckpt_epoch119.bin";binaries=Path(job["binary_dir"])
    learned=json.loads(subprocess.check_output([str(binaries/"bake_dial_refit"),"read-meta",
        "--in",str(prefix),"--key","zentrain.input_shaping"],text=True))
    if structural_terms(learned)!=structural_terms(shape):raise ValueError("pretraining shape layout differs")
    learned["trainable"]=context["family"].endswith("_joint")
    cap=native_cap(context["family"],None);record=context["record"];choice=schedule["optimizer_choice"]
    builder=lambda original,bins,dest,epochs:command(original,bins,dest,epochs,
        record["groups"],record["explicit_absent_fit_groups"],memory_cap=cap)
    from shaping_rescue_worker import memory_allowance
    memory=memory_allowance(job)
    while not reserve_worker(study,cid,memory_allowance_bytes=memory,snapshot_phases=2)["can_start"]:
        time.sleep(10)
    try:
        bind_worker(study,cid,os.getpid())
        os.environ["ZENSIM_PHASE_SNAPSHOTS"]="1"
        after=phase(study,binaries,context["source"],context["original"],root/"fit",
            initial["epochs"],choice,context["groups"],learned,
            optimizer=optimizer(choice,context["analytic"],context["groups"]),command_builder=builder)
        recovery=dict(original_run_sha256=sha(root/"RUN.json"),pretraining_continuation_sha256=sha(proof),
            registered_fit_epochs=initial["epochs"],additional_training_epochs=0,
            actual_recovery_driver_sha256=sha(__file__),original_request_sha256=sha(request))
        (root/"CONTROLLER_RECOVERY.json").write_bytes(encoded(recovery))
        (root/"RESULT.json").write_bytes(encoded(dict(initial,phases=[before,after],stage="trained",
            exit_code=0,bake_sha256=after["bake_sha256"],finished_utc=datetime.now(timezone.utc).isoformat(),
            controller_recovery_sha256=sha(root/"CONTROLLER_RECOVERY.json"))))
    finally:release_worker(study,cid)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("capture","work"))
    for n in ("study","output","capture","request"):p.add_argument("--"+n,type=Path)
    p.add_argument("--phase",choices=("fit","pretraining"),default="fit")
    p.add_argument("--cell");a=p.parse_args()
    result=capture(a.study,a.cell,a.output,a.phase) if a.mode=="capture" else work(a.study,a.capture,a.output,a.request)
    print(json.dumps(result),flush=True)
