"""Bind a pretraining template through the admission's frozen base evidence."""
import json
from pathlib import Path
from shaping_completion import sha


def require_parent(study, admission, parent):
    path = parent / "RUN.json"
    if str(path) in admission["decision_sha256"]:
        if admission["decision_sha256"][str(path)] != sha(path):
            raise ValueError("outer pretraining template admission pin drift")
        return
    run = json.loads(path.read_text())
    cell = run["cell"]
    if (not cell["family"].endswith("_joint_identity") or cell["stage"] != "inner"
            or cell["partition"] != f"inner-{admission['outer']}-0"):
        raise ValueError("registered same-fold joint pretraining parent required")
    from shaping_outer_admission import base_decision
    from shaping_execution import load_registration
    from shaping_rescue_selection import verified_candidate
    _, cells, _ = load_registration(study)
    actual, base, pins = base_decision(study, cell["family"], admission["outer"], cells)
    if (cell["id"] not in actual["reported_cells"] or not pins
            or any(admission["decision_sha256"].get(name) != pin for name, pin in pins.items())):
        raise ValueError("pretraining parent base evidence is not pinned by admission")
    # An inactive family still supplies the explicitly registered command
    # template. Verify it through the same native candidate gate used when
    # the family is active; the worker request then pins its RUN bytes.
    _, _, verified = verified_candidate(study, cell, base.get("representation", "original"))
    if verified.get(str(path)) != sha(path):
        raise ValueError("verified pretraining parent lacks its native run pin")
