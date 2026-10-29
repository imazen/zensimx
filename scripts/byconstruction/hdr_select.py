#!/usr/bin/env python3
"""Apply the registered TRAIN-side weight rule (docs/HDR_P2P3_EXECUTION_2026-10-29.md, amendment): per (path, scale) the unweighted mean
over legs of the mean over folds of the leg's within-reference Spearman, counting only eligible (leg, fold) pairs; larger wins, tie -> smaller scale.
Inputs: <runs>/dev-<path>-<scale>-f<k>/devscore.json produced by hdr_devfold_score.py. Reads nothing else."""
import json, sys
from pathlib import Path

runs = Path(sys.argv[1])
out = {}
for path in (sys.argv[2:] or ["p2", "p3"]):
    stats = {}
    for scale in ("s05", "s10"):
        legs = {}
        for f in (0, 1):
            p = runs / f"dev-{path}-{scale}-f{f}" / "devscore.json"
            for leg, m in json.loads(p.read_text()).items():
                if m["selection_eligible"]:
                    legs.setdefault(leg, []).append(m["within_ref_spearman_mean"])
        per_leg = {leg: sum(v) / len(v) for leg, v in legs.items()}
        stats[scale] = dict(per_leg=per_leg, folds_used={leg: len(v) for leg, v in legs.items()},
                            statistic=sum(per_leg.values()) / len(per_leg) if per_leg else None)
    a, b = stats["s05"]["statistic"], stats["s10"]["statistic"]
    chosen = "s10" if (a is not None and b is not None and b > a) else "s05"
    out[path] = dict(stats=stats, chosen=chosen, rule="larger statistic wins; exact tie -> smaller scale")
(runs / ("SELECTION_" + "_".join(sys.argv[2:] or ["p2", "p3"]) + ".json")).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({p: (v["chosen"], v["stats"]["s05"]["statistic"], v["stats"]["s10"]["statistic"]) for p, v in out.items()}))
