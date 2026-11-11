"""Seal terminal fits already on the coordinator using their frozen native gates."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import sys


def seal(study, request, assignment, source):
    sys.path.insert(0, str(source / "scripts/rev4_featpot"))
    from shaping_completion import DECLARATION_SHA256, encoded, sha
    from shaping_rescue_worker import verify
    from shaping_worker_collect import inventory, validate
    from shaping_execution import load_registration
    q = json.loads(request.read_text())
    if q["host"] != socket.gethostname():
        raise ValueError("only terminal coordinator-local fits may be sealed in place")
    root = study / "runs" / q["cell"]["id"] / "attempt-001"
    if (root / "WORKER_COLLECTION.json").exists():
        raise ValueError("preserve existing local collection receipt")
    verify(q)
    declaration = None
    if q["execution_kind"] == "control":
        from shaping_control_worker import validate_result
        execution = json.loads((root / "WORKER_EXECUTION.json").read_text())
        if execution["request_sha256"] != sha(request):
            raise ValueError("local control execution request drift")
        validate_result(q, root)
    else:
        a = json.loads(assignment.read_text())
        if a["host"] != socket.gethostname() or a["worker_request_sha256"] != sha(request):
            raise ValueError("local neural assignment differs")
        _, cells, _ = load_registration(study)
        run = validate(root, a, cells, study)
        from shaping_external_recovery import verify_attempt
        verify_attempt(study, run, "attempt-001")
        command = json.loads((root / "fit/RUN.json").read_text())["command"]
        declaration = Path(command[command.index("--derived-inputs") + 1])
        if sha(declaration) != DECLARATION_SHA256:
            raise ValueError("local frozen declaration drift")
        target = root / "COLLECTION_DERIVED_INPUTS.json"
        if target.exists():
            if sha(target) != DECLARATION_SHA256:
                raise ValueError("preserve conflicting local declaration")
        else:
            with target.open("xb") as output, declaration.open("rb") as input:
                shutil.copyfileobj(input, output)
    files = inventory(root)
    result = dict(local_collection=True, request_sha256=sha(request),
                  assignment_sha256=sha(assignment), source_host=socket.gethostname(),
                  collect_driver_sha256=sha(__file__), files=files,
                  verified_bytes=sum(r["bytes"] for r in files.values()))
    if declaration is not None:
        result["original_declaration_path"] = str(declaration)
    with (root / "WORKER_COLLECTION.json").open("xb") as stream:
        stream.write(encoded(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "request", "assignment", "source"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    r = seal(args.study, args.request, args.assignment, args.source)
    print(json.dumps(dict(local_collection=True, verified_bytes=r["verified_bytes"])))
