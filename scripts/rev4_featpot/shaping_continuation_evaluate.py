"""Evaluate the fixed ten collected continuations through the original final-only owner."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace
from shaping_completion import encoded, sha


def run(args):
    from shaping_supervision import later_sweep
    from shaping_evaluation_worker import capacity
    jobs=json.loads(args.placements.read_text())
    if len(jobs)!=10 or len({r["cell"] for r in jobs})!=10:raise ValueError("fixed ten migrated conditions required")
    args.output.mkdir(parents=True,exist_ok=True)
    for job in jobs:
        cid=job["cell"];root=args.study/"runs"/cid/"attempt-001";marker=root/"DIRECT_LATER_EVALUATION.json"
        if marker.exists():continue
        while not capacity(args.study):time.sleep(60)
        values=SimpleNamespace(study=args.study,cells=(cid,),binary_dir=args.study.parent/"study-bin",
            rescue_binary_dir=args.study.parent/"bounded-resume-runtime-20261108/binaries-v3",
            candidate_binary_dir=[args.study.parent/"paired-gradient-native-20261109/binaries"],
            repair_binary_dir=args.study.parent/"max-pair-bin",
            repair_inspector=args.study.parent/"first-weight-native-qualification/zenpredict-inspect")
        if later_sweep(values)!=1:raise ValueError("continued final-only evaluation not complete: "+cid)
        marker.write_bytes(encoded(dict(cell=cid,local_evaluation=True,continuation_sha256=sha(root/"CONTINUATION.json"),
            finished_utc=datetime.now(timezone.utc).isoformat(),driver_sha256=sha(__file__))))
    (args.output/"COMPLETE.json").write_bytes(encoded(dict(evaluated=10,additional_training=0,final_holdout_read=False)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("study","placements","output"):parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())
