"""Recover exact two-vote sufficient statistics from released TRAIN judges.

The native missing-count gate remains unchanged. This separate source audit
reads only frozen TRAIN judge arrays and checks them against the normalized
probability and the published two-judge protocol. It never predicts a vote.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"rev4_featpot"))
from ring_root import dataset_family_guard, dataset_targets
from run_processes import compute_lock, process_snapshot

ARCHIVE_SHA = "b4f5c772c7cd88ac44bcb4ff0741745d434c05f0094f2bfb262ddcd0055bf393"
MEMBERSHIP_SHA = "96e5acc0d01c412d81e9a5259df5a6af903550a2d85138cc3b45fd6f75f9b669"
POLICY_SHA = "40407d74cf998da64400b36a7885b798333c886227c9c6aef5927cb1777e3e02"
ROOT_SHA = "a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recover(raw, label):
    """Lossless protocol decoding, with no rounding, tolerance or imputation."""
    if label["split"] != "train" or label["n"] != "2":
        raise ValueError("released TRAIN two-judge protocol required")
    if len(raw) > 4096:
        raise ValueError("unexpected native judge size")
    values = np.load(io.BytesIO(raw), allow_pickle=False)
    if values.size != 1 or values.dtype.kind != "f":
        raise ValueError("one native floating judgment aggregate required")
    p1 = float(values.reshape(-1)[0])
    if p1 not in (0., 0.5, 1.) or float(label["p_a"]) != 1-p1:
        raise ValueError("released aggregate differs from normalized preference")
    counts = dict(n_a=int(2*(1-p1)), n_b=int(2*p1), n_tie=0, n=2, p_a=1-p1)
    for name in ("n_a", "n_b", "n_tie"):
        if label.get(name, "") not in ("", str(counts[name])):
            raise ValueError("existing count contradicts released aggregate")
    return counts


def judge_path(member):
    ref = PurePosixPath(member["reference_path"])
    if (len(ref.parts) != 4 or ref.parts[0] != "train" or ref.parts[1] not in ("cnn", "mix", "traditional")
            or ref.parts[2] != "ref" or ref.suffix != ".png"
            or member["a_path"] != str(ref.parent.parent/"p0"/ref.name)
            or member["b_path"] != str(ref.parent.parent/"p1"/ref.name)):
        raise ValueError("released TRAIN triplet identity differs")
    return str(ref.parent.parent/"judge"/ref.with_suffix(".npy").name)


def audit(args):
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("source audit requires the serialized compute lock")
    if sha(args.root/"wide/frozen.json") != ROOT_SHA or sha(args.policy) != POLICY_SHA:
        raise ValueError("original Mlic root and corrected policy required")
    args.output.mkdir()
    state = dict(run_directory=str(args.output), pid=os.getpid(),
        start_ticks=process_snapshot(os.getpid())["start_ticks"], source_sha256=sha(__file__),
        compute_lock=compute_lock(os.getpid()), required=146003, complete=0,
        next_action="verify every frozen TRAIN judge before supplying recovered counts")
    def progress(stage, **details):
        state.update(stage=stage,checked_utc=datetime.now(timezone.utc).isoformat(),**details)
        (args.output/"CURRENT_RUN.json").write_text(json.dumps(state,indent=2)+"\n")
    progress("metadata-admission")
    guard, memberships = dataset_family_guard(args.admission, args.root, args.output/"FAMILY_GUARD.json")
    if guard["membership_sha256"]["bapps"] != MEMBERSHIP_SHA or guard["inherited_leaks"]:
        raise ValueError("frozen TRAIN membership/family guard changed")
    members = [r for r in memberships["bapps"] if r["role"] == "TRAIN" and r["membership_state"] == "FROZEN"]
    if len(members) != 146003 or any(r["input_table"] != "pairs.tsv" for r in members):
        raise ValueError("registered BAPPS TRAIN population differs")
    wanted = {judge_path(m):m for m in members}
    if len(wanted) != len(members):
        raise ValueError("duplicate released judge identity")
    ledger = args.output/"LABEL_READS.json"
    ledger.write_text(json.dumps(dict(role="TRAIN", reads=[dict(kind="opaque-archive-integrity-hash",
        path=str(args.archive), expected_sha256=ARCHIVE_SHA, interpreted_judgments=0,
        parsed_judgment_members=sorted(wanted), excluded_judgments_parsed=False)]),indent=1)+"\n")
    progress("opaque-archive-integrity-hash")
    if sha(args.archive) != ARCHIVE_SHA:
        raise ValueError("released TRAIN archive pin changed")
    labels = dataset_targets(args.library/"bapps/pairs.tsv", {m["row_index"] for m in members},
        ["ref","a","b","p_a","n","split","n_a","n_b","n_tie"], ledger, "bapps")
    ledger_data = json.loads(ledger.read_text())
    ledger_data["reads"].append(dict(kind="released-native-TRAIN-judge-arrays", members=sorted(wanted),
                                    archive_sha256=ARCHIVE_SHA,state="authorized-before-read"))
    ledger.write_text(json.dumps(ledger_data,indent=1)+"\n")
    rows = {}
    progress("native-TRAIN-judge-audit")
    # The pinned .tar.gz is an uncompressed tar. Seek past image bodies and
    # excluded judge arrays; no extraction and no member cache.
    with tarfile.open(args.archive, "r:", stream=True) as archive:
        for entry in archive:
            name = entry.name.removeprefix("./")
            if name not in wanted:
                continue
            member = wanted[name]
            index = member["row_index"]
            if index in rows or not entry.isfile() or entry.size > 4096:
                raise ValueError("duplicate or malformed native judge member")
            label = labels[index]
            if (label["ref"],label["a"],label["b"]) != (member["reference_path"],member["a_path"],member["b_path"]):
                raise ValueError("normalized triplet differs from frozen membership")
            with archive.extractfile(entry) as stream:
                raw = stream.read()
            rows[index] = dict(row=index, source=f"bapps:pairs.tsv:{index}", judge_member=name,
                judge_sha256=hashlib.sha256(raw).hexdigest(), **recover(raw,label))
            if len(rows) % 10000 == 0:
                progress("native-TRAIN-judge-audit",complete=len(rows))
    if set(rows) != {m["row_index"] for m in members}:
        raise ValueError("released judge coverage incomplete")
    counts = args.output/"recovered-counts.json"
    counts.write_text(json.dumps(dict(schema="bapps-released-two-vote-counts-v1",
        rows=[rows[i] for i in sorted(rows)]),separators=(",",":"))+"\n")
    receipt = dict(schema="bapps-native-vote-source-audit-v1", role="TRAIN", state="verified",
        original_root_sha256=ROOT_SHA, membership_sha256=MEMBERSHIP_SHA, confidence_policy_sha256=POLICY_SHA,
        archive_path=str(args.archive),archive_sha256=ARCHIVE_SHA,
        count_path=str(counts),count_sha256=sha(counts),rows=len(rows),
        counts_by_n_a=dict(sorted(Counter(r["n_a"] for r in rows.values()).items())),
        interpretation="lossless sufficient statistics from released aggregate; exactly two binary judgments",
        source_script_sha256=sha(__file__),
        sources=["https://arxiv.org/html/1801.03924#Sx4",
                 "https://github.com/richzhang/PerceptualSimilarity#c-about-the-dataset"],
        excluded_train_judgments_parsed=0,eval_judgments_parsed=0)
    (args.output/"AUDIT.json").write_text(json.dumps(receipt,indent=2)+"\n")
    ledger_data["reads"][-1]["state"] = "verified"
    ledger.write_text(json.dumps(ledger_data,indent=1)+"\n")
    progress("verified",complete=len(rows),next_action="fresh vote preparation with independently pinned audit")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root","admission","library","policy","archive","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    try:
        audit(args)
    except Exception as error:
        progress_path = args.output/"CURRENT_RUN.json"
        if progress_path.exists():
            state = json.loads(progress_path.read_text())
            state.update(stage="failed",failure=repr(error),checked_utc=datetime.now(timezone.utc).isoformat())
            progress_path.write_text(json.dumps(state,indent=2)+"\n")
        raise
