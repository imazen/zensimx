"""Recover the fixed outer-2 deliveries; never execute or replay a trainer."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

ALLOWED = ("i134", "r3500", "r3800x", "r5600g")


def materialize_archived_components(study, cid):
    from shaping_artifact_retention import archive_inventory
    from shaping_completion import sha
    for kind in ("runs","reexport_max_pairs","reexports"):
        directory=study/kind/cid
        marker=directory/"MIRROR_VERIFIED.json"
        if not marker.exists():
            continue
        record=json.loads(marker.read_text());mirror=Path(record["destination"])
        manifest=directory/"SHA256SUMS"
        if not manifest.exists():
            with manifest.open("xb") as target,(mirror/"SHA256SUMS").open("rb") as source:
                shutil.copyfileobj(source,target)
        archive,files=archive_inventory(directory)
        for name,pin in files.items():
            relative=Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe qualified archive path")
            target=directory/relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as output,(archive/relative).open("rb") as source:
                shutil.copyfileobj(source,output,1024*1024)
            if sha(target)!=pin:
                raise ValueError("archive materialization readback differs")


def run(args):
    from shaping_completion import encoded, sha
    from shaping_execution import load_registration
    from shaping_outer_dispatch import selected_cells
    from shaping_variant_pipeline import selected
    from shaping_worker_collect import inventory
    study, out = args.study, args.output
    out.mkdir(parents=True, exist_ok=False)
    _, cells, _ = load_registration(study)
    admission = json.loads((study / "selection/outer-2/OUTER_ADMISSION.json").read_text())
    wanted = selected_cells(cells, admission)
    wanted += [c for c in cells if c["stage"] == "outer" and c["partition"] == "outer-2"
               and c["family"] in ("extra_trees", "shaped_additive")]
    wanted += selected(study, 2, "human_only")[0]
    if len(wanted) != 35 or len({c["id"] for c in wanted}) != 35:
        raise ValueError("complete frozen outer-2 delivery scope required")
    moves = {r["cell"]: r for r in json.loads(args.placements.read_text())}
    jobs = []
    for cell in wanted:
        cid = cell["id"]; kind = "outer" if cell["stage"] != "human_only" else "variant"
        paths = list((study / "downstream_assignments").glob(kind + "-" + cid + "-*.json"))
        if len(paths) != 1:
            raise ValueError("one original outer-2 assignment required: " + cid)
        a = json.loads(paths[0].read_text())
        host = moves[cid]["target"] if cid in moves else a["host"]
        if host not in ALLOWED:
            raise ValueError("outer-2 provider is outside current compute pool")
        default = study.parent / "downstream-cap-approved-20261109T2210" / kind / "outer-2"
        default = default / ("neural-fits" if kind == "outer" else "human_only") / "jobs" / cid / "REQUEST.json"
        jobs.append(dict(cell=cell, host=host, assignment=str(paths[0]),
                         request=a.get("worker_request_path", str(default)), continued=cid in moves))
    (out / "JOBS.json").write_bytes(encoded(jobs))
    done, failures = [], {}
    def status():
        (out / "STATUS.json").write_bytes(encoded(dict(pid=__import__("os").getpid(),
            checked_utc=datetime.now(timezone.utc).isoformat(), stage="complete" if len(done)==35 else "recovering",
            required=35, completed=done, failures=failures, training_started=False)))
    for job in jobs:
        cid, host = job["cell"]["id"], job["host"]
        root = study / "runs" / cid / "attempt-001"
        if (root / "WORKER_COLLECTION.json").exists():
            materialize_archived_components(study,cid)
            done.append(cid); status(); continue
        directory = out / cid; directory.mkdir()
        log = directory / "recovery.log"
        def call(command):
            with log.open("a") as stream:
                stream.write(shlex.join(command) + "\n"); stream.flush()
                return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        try:
            if not job["continued"]:
                request = Path(job["request"])
                if not request.exists():
                    remote = "test -f " + shlex.quote(str(request))
                    call(["ssh", host, remote])
                    request.parent.mkdir(parents=True, exist_ok=True)
                    call(["rsync", "-rlpt", "--ignore-existing", host + ":" + str(request), str(request)])
                q = json.loads(request.read_text())
                source = Path(q["source_root"])
                if source == Path("/home/lilith/work/zen/zensimx"):
                    source = args.dispatch / "source-f05"
                if not (source / "scripts/rev4_featpot/shaping_worker_collect.py").exists():
                    source.mkdir(parents=True, exist_ok=True)
                    call(["rsync", "-rlpt", "--ignore-existing", "--exclude=.git", "--exclude=.jj",
                          "--exclude=target", "--include=*/", "--include=*.py", "--include=*.json",
                          "--include=*.md", "--include=justfile", "--include=*.just", "--exclude=*",
                          host + ":" + str(source) + "/", str(source) + "/"])
                if host == socket.gethostname():
                    call([sys.executable, "-B", str(Path(__file__).with_name("shaping_local_collection.py")),
                          "--study", str(study), "--request", str(request), "--assignment", job["assignment"], "--source", str(source)])
                else:
                    # Metadata copied during the coordinator drain is preserved
                    # before the ordinary atomic worker collector publishes.
                    if root.parent.exists():
                        backup = directory / "coordinator-before-collection"
                        root.parent.rename(backup)
                    if q["execution_kind"] == "control":
                        driver=source/"scripts/rev4_featpot/shaping_control_worker.py"
                        remote_bundle=study.parent/"worker-bundles"/("legacy-control-"+cid+".tar")
                        remote=["env","TMPDIR=/home/lilith/tmp","PYTHONPATH="+str(study.parent/"python-runtime"),
                                "/home/lilith/work/claudehints/scripts/run-heavy","--mem","1G","--jobs","1","--",
                                "python3","-B",str(driver),"pack","--request",str(request),"--bundle",str(remote_bundle)]
                        call(["ssh",host,shlex.join(remote)])
                        call(["rsync","-rlpt","--ignore-existing",host+":"+str(remote_bundle),host+":"+str(remote_bundle)+".json",str(directory)+"/"])
                        local=directory/remote_bundle.name;pin=json.loads(Path(str(local)+".json").read_text())["bundle_sha256"]
                        call([sys.executable,"-B",str(driver),"collect","--request",str(request),"--bundle",str(local),"--bundle-sha256",pin])
                        done.append(cid);status();continue
                    spec = dict(cell=cid, host=host, assignment=job["assignment"], request=str(request), source=str(source))
                    plan = directory / "COLLECTION.json"; plan.write_bytes(encoded([spec]))
                    call([sys.executable, "-B", str(Path(__file__).with_name("shaping_resume_collect.py")),
                          "--study", str(study), "--jobs", str(plan), "--output", str(directory / "collection")])
            else:
                if host != socket.gethostname():
                    program = "root=" + repr(str(root)) + "\n" + """
from pathlib import Path
import json,hashlib
r=Path(root)
result=json.loads((r/'RESULT.json').read_text())
if result['stage']!='trained' or result['exit_code']!=0:raise ValueError('continuation finalization not complete')
files={}
for p in sorted(r.rglob('*')):
 if p.is_symlink():raise ValueError('symlink in finalized continuation')
 if p.is_file():
  with p.open('rb') as f:pin=hashlib.file_digest(f,'sha256').hexdigest()
  files[str(p.relative_to(r))]=dict(bytes=p.stat().st_size,sha256=pin)
print(json.dumps(files))
"""
                    result = subprocess.run(["ssh", host, "python3 -B -"], input=program, text=True,
                                            capture_output=True, check=True)
                    pins = json.loads(result.stdout)
                    staged = directory / "verified-attempt"; staged.mkdir()
                    call(["rsync", "-rlpt", "--ignore-existing", host + ":" + str(root) + "/", str(staged) + "/"])
                    if inventory(staged) != pins:
                        raise ValueError("continued fit destination readback differs")
                    (directory / "TRANSFER_VERIFIED.json").write_bytes(encoded(dict(source_host=host,files=pins)))
                    if root.parent.exists():
                        old = root / "RUN.json"
                        if old.exists() and sha(old) != sha(staged / "RUN.json"):
                            raise ValueError("preserve conflicting coordinator original identity")
                        root.parent.rename(directory / "coordinator-before-collection")
                    root.parent.mkdir()
                    staged.rename(root)
                seal_continued(study, root, host)
            materialize_archived_components(study,cid)
            done.append(cid)
        except Exception as error:
            failures[cid] = str(error)
            (directory / "FAILURE.json").write_bytes(encoded(dict(error=str(error), training_started=False)))
        status()
    return dict(completed=len(done),required=35,failures=failures)


def seal_continued(study, root, host):
    from shaping_completion import DECLARATION_SHA256, encoded, sha
    from shaping_execution import load_registration
    from shaping_evaluation import outer_context, admitted_epoch
    from shaping_worker_collect import inventory
    run=json.loads((root/"RUN.json").read_text());result=json.loads((root/"RESULT.json").read_text())
    proof_path=root/"CONTINUATION.json";proof=json.loads(proof_path.read_text())
    if (host not in ALLOWED or proof["resumed_execution_host"]!=host or proof["cell"]!=run["cell"]
            or proof["original_run_sha256"]!=sha(root/"RUN.json")
            or proof["original_phase_sha256"]!=sha(root/"fit/RUN.json")
            or result["continuation_sha256"]!=sha(proof_path)
            or result["stage"]!="trained" or result["exit_code"]!=0
            or run["plan_sha256"]!=sha(study/"PLAN.json")):
        raise ValueError("continued scientific identity differs")
    _,cells,_=load_registration(study);context=outer_context(study,run,cells)
    epochs=admitted_epoch(run,result,cells,context)
    phase=json.loads((root/"fit/RESULT.json").read_text())
    if (result["phases"]!=[phase] or phase["continuation_sha256"]!=sha(proof_path)
            or phase["epochs"]!=epochs or phase["pairs_per_epoch"]!=50000):
        raise ValueError("continued registered phase coverage differs")
    for name,pin in proof["checkpoint_sha256"].items():
        if sha(root/"fit/checkpoints"/name)!=pin:raise ValueError("continued checkpoint drift")
    for name,pin in proof["evidence_sha256"].items():
        if sha(root/name)!=pin:raise ValueError("continued retained evidence drift")
    initial=json.loads((root/"fit/RUN.json").read_text())
    for name,pin in initial["manifest_sha256"].items():
        if sha(name)!=pin:raise ValueError("continued phase manifest drift")
    declaration=Path(initial["command"][initial["command"].index("--derived-inputs")+1])
    if sha(declaration)!=DECLARATION_SHA256:raise ValueError("continued declaration drift")
    target=root/"COLLECTION_DERIVED_INPUTS.json"
    if not target.exists():
        with target.open("xb") as output,declaration.open("rb") as input:shutil.copyfileobj(input,output)
    if sha(target)!=DECLARATION_SHA256:raise ValueError("continued portable declaration drift")
    files=inventory(root)
    receipt=dict(continued_collection=True,source_host=host,initial_execution_host=run["execution_host"],
                 continuation_sha256=sha(proof_path),original_declaration_path=str(declaration),
                 files=files,verified_bytes=sum(v["bytes"] for v in files.values()),collect_driver_sha256=sha(__file__))
    with (root/"WORKER_COLLECTION.json").open("xb") as stream:stream.write(encoded(receipt))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("study","dispatch","placements","output"):parser.add_argument("--"+name,type=Path,required=True)
    print(json.dumps(run(parser.parse_args())))
