#!/usr/bin/env python3
"""Append the four registered native-HDR legs to a frozen training root (paths P2/P3).

Registration: docs/HDR_TRAINING_PATHS_2026-10-29.md and docs/HDR_P2P3_EXECUTION_2026-10-29.md.
Legs (all TRAIN rows only; held-out EVAL/VAL/development rows never enter the root):
  hdr_upiq380         UPIQ-380 fit rows (330), human JOD, within-reference rank       (producer: upiq_pu_score)
  hdr_4kdtb_train     4Kdtb TRAIN half (48), human MOS, within-reference rank
  hdr_ladder_chroma   consumer-TRAIN HDR chroma-ladder rungs, natural SSIMULACRA2, within-ladder rank-and-scale
  hdr_teacher_hdrvdp3 HDRTEACH train agree rows (7390), 10*q_jod, within-reference rank-and-scale
The parent root's payloads stay byte-identical; only receipts/freeze move (as ring_root.py chroma does).
"""
import argparse
import collections
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ring_root import bank_transport, sha  # noqa: E402

FEATURES = "basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af"
HDR_EXTRACT_SHA = "63bed9a64ed265d2e8b8bad4aabaae5d952215afda935ec911a0c82d1c204aa7"
HDR_COMMIT = "a86520ca68dc92f98abfd8a526d81f8abf112de6"
UPIQ_SHA = "73d1374f1cbb2ca2f5166812d69ee49e22486174b56337263fa27d9689c79214"
UPIQ_COMMIT = "d86b3b44383433e56ebc6161ee6c8db12639c5af"
UPIQ_LABELS = "/mnt/v/output/zenmetrics/upiq-pu/upiq_cid_jod.csv"
UPIQ_LABELS_SHA = "e0b23f539d46845f6cc4a65591474bafbf7eb4faf02da7175fb342caa5df80ac"
# name: kind, mode, loss, target_transform, rows, base nominal mass, producer commit, extractor sha
LEGS = {
    "hdr_upiq380": dict(human=True, mode="withinref,rank", loss="within-source-rank",
                        transform="score=100+10*human_JOD;no-clipping", rows=330, mass=0.5,
                        commit=UPIQ_COMMIT, extractor=UPIQ_SHA),
    "hdr_4kdtb_train": dict(human=True, mode="withinref,rank", loss="within-source-rank",
                            transform="orient-native-quality;MOS-0-100", rows=48, mass=0.5,
                            commit=HDR_COMMIT, extractor=HDR_EXTRACT_SHA),
    "hdr_ladder_chroma": dict(human=False, mode="withinref,both", loss="within-source-rank-and-scale",
                              transform="identity-ssimulacra2-unclipped", rows=7177, mass=1.0,
                              commit=HDR_COMMIT, extractor=HDR_EXTRACT_SHA),
    "hdr_teacher_hdrvdp3": dict(human=False, mode="withinref,both", loss="within-source-rank-and-scale",
                                transform="score=10*q_jod;no-clipping", rows=7390, mass=0.5,
                                commit=HDR_COMMIT, extractor=HDR_EXTRACT_SHA),
}
SCHEMA = "hdr-native-leg-v1"
# TRAIN rows left after holding out the families of dev fold k (registered; mirrored in the admissions)
FOLD_ROWS = {"hdr_upiq380": [150, 180], "hdr_4kdtb_train": [0, 48], "hdr_ladder_chroma": [2709, 4468],
             "hdr_teacher_hdrvdp3": [4484, 2906]}


def fold_of(family):
    return int(hashlib.sha256(("hdr-p2p3-dev-fold-v1:" + family).encode()).hexdigest(), 16) % 2


def table_of(path, cols=None):
    return pq.read_table(path, columns=cols)


def upiq_rows(bank, admission, labels):
    """UPIQ-380 fit rows: native features joined to the approved isolated JOD labels (fit only)."""
    man = json.loads((bank / "_MANIFEST.json").read_text())
    a = json.loads(Path(admission).read_text())
    if sha(bank / "features.parquet") != man["features_parquet_sha256"] or sha(labels) != UPIQ_LABELS_SHA:
        raise ValueError("UPIQ bank or label file changed")
    jod = {r[0]: float(r[1]) for r in csv.reader(open(labels))}
    keys = pq.read_table(bank / "keys.parquet").to_pylist()
    feats = pq.read_table(bank / "features.parquet")
    index = {k["condition_id"]: i for i, k in enumerate(keys)}
    rows = []
    for r in a["rows"]:
        if r["split"] != "fit":
            continue  # the 50 development rows are never read here
        rows.append(dict(i=index[r["condition_id"]], obs=r["pair_key"], ref=f"upiq:{r['dataset']}:{r['content']:02d}",
                         family=f"hdr:upiq:{r['dataset']}:{r['content']:02d}", target=100.0 + 10.0 * jod[r["condition_id"]]))
    return feats, rows


def fourk_rows(bank, labels_tsv):
    keys = pq.read_table(bank / "keys.parquet").to_pylist()
    feats = pq.read_table(bank / "features.parquet")
    lab = {}
    for r in csv.DictReader(open(labels_tsv), delimiter="\t"):
        lab[r["item"]] = float(r["score"])
    rows = []
    for i, k in enumerate(keys):
        m = json.loads(k["metadata_json"])
        if m["role"] != "TRAIN":
            continue
        item = m["dist_path"].split("/content/")[1]
        rows.append(dict(i=i, obs="4kdtb:" + m["dist_path"].split("/content/")[1], ref="4kdtb:" + m["source_id"],
                         family="hdr:" + m["family_id"], target=lab[item]))
    return feats, rows


def ladder_rows(bank, cells):
    keys = pq.read_table(bank / "keys.parquet").to_pylist()
    feats = pq.read_table(bank / "features.parquet")
    cell = {}
    for c in pq.read_table(cells, columns=["encoded_sha256", "view_id", "quality", "natural_ssim2"]).to_pylist():
        cell[(c["view_id"], c["encoded_sha256"])] = c
    rows = []
    for i, k in enumerate(keys):
        if k["consumer_role"] != "train":
            continue
        c = cell[(k["view_id"], k["encoded_sha256"])]
        rows.append(dict(i=i, obs="hdrladder:" + k["view_id"] + ":" + k["encoded_sha256"],
                         ref=f"hdrladder:{k['view_id']}|q{c['quality']}", family="hdr:ladder:" + k["source"],
                         target=float(c["natural_ssim2"])))
    return feats, rows


def teacher_rows(bank, table):
    keys = pq.read_table(bank / "keys.parquet").to_pylist()
    feats = pq.read_table(bank / "features.parquet")
    t = {r["row_id"]: r for r in pq.read_table(table, columns=["row_id", "role", "agree", "ref_sha256", "source_family", "hdrvdp3_q_jod"]).to_pylist()}
    rows = []
    for i, k in enumerate(keys):
        m = json.loads(k["metadata_json"])
        r = t[m["row_id"]]
        if r["role"] != "train" or not r["agree"]:
            continue
        rows.append(dict(i=i, obs=f"hdrteach:{m['row_id']}", ref="hdrteach:" + r["ref_sha256"][:16],
                         family="hdr:teacher:" + str(r["source_family"]), target=10.0 * float(r["hdrvdp3_q_jod"])))
    return feats, rows


def build_leg(name, feats, rows, bank, out, selected, width, scale, fold=None):
    spec = LEGS[name]
    registered = spec["rows"]
    if len(rows) != registered:
        raise ValueError(f"{name}: {len(rows)} rows, registered {registered}")
    if fold is not None:
        dev = [r for r in rows if fold_of(r["family"]) == fold]
        rows = [r for r in rows if fold_of(r["family"]) != fold]
        if len(rows) != FOLD_ROWS[name][fold]:
            raise ValueError(f"{name}: fold {fold} leaves {len(rows)} rows, registered {FOLD_ROWS[name][fold]}")
        if dev:
            devdir = out / "hdr_dev"
            devdir.mkdir(exist_ok=True)
            dtab = bank_transport(feats.take(pa.array([r["i"] for r in dev])).select([f"f{i}" for i in range(1825)]),
                                  [r["ref"] for r in dev], [r["target"] for r in dev], width)
            pq.write_table(dtab, devdir / f"{name}.parquet", compression="zstd")
            (devdir / f"{name}.keys.json").write_text(json.dumps([dict(row=i, ref=r["ref"], family=r["family"]) for i, r in enumerate(dev)]) + "\n")
        if not rows:
            return None, None
    man = json.loads((bank / "_MANIFEST.json").read_text())
    if man["build_commit"] != spec["commit"] or sha(bank / "features.parquet") != man["features_parquet_sha256"]:
        raise ValueError(f"{name}: bank producer/hash differs")
    take = feats.take(pa.array([r["i"] for r in rows]))
    take = take.select([f"f{i}" for i in range(1825)])
    tab = bank_transport(take, [r["ref"] for r in rows], [r["target"] for r in rows], width)
    legdir = out / "datasets"
    path = legdir / f"{name}.parquet"
    pq.write_table(tab, path, compression="zstd")
    counts = dict(sorted(collections.Counter(r["ref"] for r in rows).items()))
    if min(counts.values()) < 2:
        raise ValueError(f"{name}: a group has fewer than two rows")
    keys = [dict(row=i, observation=r["obs"], source=r["ref"], reference=r["ref"], family=r["family"], role="fit", tier="T2")
            for i, r in enumerate(rows)]
    if len({k["observation"] for k in keys}) != len(keys):
        raise ValueError(f"{name}: duplicate observation")
    kp = legdir / f"{name}.keys.json"
    kp.write_text(json.dumps(keys, separators=(",", ":")) + "\n")
    nominal = spec["mass"] * scale
    decl = dict(schema=SCHEMA, leg=name, loss=spec["loss"], nominal_weight=nominal, target_orientation="quality",
                target_transform=spec["transform"], target_column="human_score", selection="fixed-final-epoch",
                formula_revision=7, tier="v3", feature_ids=selected, feature_set_id=FEATURES, dev_fold=fold,
                producer_commit=spec["commit"], extractor_sha256=spec["extractor"], native_role="TRAIN", rows=len(rows),
                table_sha256=sha(path), keys_path=str(kp), keys_sha256=sha(kp), training_width=width, admission_root=str(out),
                feature_bank_manifest_path=str(bank / "_MANIFEST.json"), feature_bank_manifest_sha256=sha(bank / "_MANIFEST.json"))
    dp = Path(str(path) + ".manifest.json")
    dp.write_text(json.dumps(decl, indent=1) + "\n")
    total = 0.0
    for key in sorted(counts):
        total += 1.0 - 1.0 / counts[key]
    weight = nominal / (total / len(counts))
    record = dict(name=name, path=str(path), teacher=not spec["human"], subjective=spec["human"], train_weight=weight,
                  val_weight=0.0, mode=spec["mode"], table_sha256=decl["table_sha256"], manifest_sha256=sha(dp),
                  keys_path=str(kp), keys_sha256=sha(kp), rows=len(rows), source_counts=counts, collection=name,
                  groups=len(counts), nominal_weight=nominal)
    return record, decl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--upiq-bank", required=True)
    ap.add_argument("--upiq-admission", required=True)
    ap.add_argument("--fourk-bank", required=True)
    ap.add_argument("--fourk-labels", default="/mnt/tower/input/iqa/datasets/rousselot-4kdtb/labels.tsv")
    ap.add_argument("--ladder-bank", required=True)
    ap.add_argument("--ladder-cells", required=True)
    ap.add_argument("--teacher-bank", required=True)
    ap.add_argument("--teacher-table", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="registered scale on every leg's base nominal mass")
    ap.add_argument("--holdout-fold", type=int, choices=[0, 1], default=None, help="hold out the families of this registered TRAIN-side dev fold; held-out rows go to hdr_dev/ and never into a leg")
    ap.add_argument("--legs", default=",".join(LEGS))
    ap.add_argument("--build-commit", required=True)
    a = ap.parse_args()
    source, out = Path(a.root), Path(a.out)
    if out.exists():
        raise ValueError("destination must be fresh")
    chosen = a.legs.split(",")
    if not set(chosen) <= set(LEGS) or not a.scale > 0:
        raise ValueError("unregistered leg or scale")
    packet = json.loads((source / "datasets/receipt.json").read_text())
    selected = packet["feature_ids"]
    width = json.loads((source / "wide/main/real/receipt.json").read_text())["width"]
    parent_pin = sha(source / "wide/frozen.json")
    original_paths = Path(packet["family_guard_path"]).parent.parent
    shutil.copytree(source, out)
    banks = {"hdr_upiq380": Path(a.upiq_bank), "hdr_4kdtb_train": Path(a.fourk_bank),
             "hdr_ladder_chroma": Path(a.ladder_bank), "hdr_teacher_hdrvdp3": Path(a.teacher_bank)}
    loaders = {"hdr_upiq380": lambda: upiq_rows(banks["hdr_upiq380"], a.upiq_admission, UPIQ_LABELS),
               "hdr_4kdtb_train": lambda: fourk_rows(banks["hdr_4kdtb_train"], a.fourk_labels),
               "hdr_ladder_chroma": lambda: ladder_rows(banks["hdr_ladder_chroma"], a.ladder_cells),
               "hdr_teacher_hdrvdp3": lambda: teacher_rows(banks["hdr_teacher_hdrvdp3"], a.teacher_table)}
    for r in packet["groups"]:
        for field in ["path", "keys_path"]:
            r[field] = str(out / Path(r[field]).relative_to(original_paths))
        if sha(r["path"]) != r["table_sha256"] or sha(r["keys_path"]) != r["keys_sha256"]:
            raise ValueError("inherited leg changed")
    packet["family_guard_path"] = str(out / Path(packet["family_guard_path"]).relative_to(original_paths))
    guard = json.loads(Path(packet["family_guard_path"]).read_text())
    protected = set(guard["protected_families"])
    new = []
    for name in chosen:
        feats, rows = loaders[name]()
        if any(r["family"] in protected for r in rows):
            raise ValueError(f"{name}: family reserved by another role")
        record, decl = build_leg(name, feats, rows, banks[name], out, selected, width, a.scale, a.holdout_fold)
        if record is None:
            print(name, "absent in this fold (every family held out)", flush=True)
            continue
        packet["groups"].append(record)
        new.append(record)
        print(name, len(rows), "rows", len(record["source_counts"]), "groups", "weight", record["train_weight"], flush=True)
    packet["hdr_legs"] = dict(parent_frozen_sha256=parent_pin, scale=a.scale, legs=[r["name"] for r in new], holdout_fold=a.holdout_fold, build_only=True)
    pp = out / "datasets/receipt.json"
    pp.write_text(json.dumps(packet, indent=1) + "\n")
    frozen = json.loads((out / "wide/frozen.json").read_text())
    view = dict(frozen["admission_view"])
    view["immutable_input_roots"] = list(dict.fromkeys(view["immutable_input_roots"] + [str(source), str(out)] + [str(b) for b in banks.values()]))
    rp = out / "wide/main/real/receipt.json"
    receipt = json.loads(rp.read_text())
    receipt["admission_view"] = view
    rp.write_text(json.dumps(receipt, indent=1) + "\n")
    frozen.update(admission_view=view, wide_receipts={"main/real": sha(rp)}, dataset_legs_sha256=sha(pp), byc_hdr=packet["hdr_legs"])
    frozen["auxiliary_files"] = {rel: sha(out / rel) for rel in frozen["auxiliary_files"]}
    for rec in new:
        for p in (rec["path"], rec["path"] + ".manifest.json", rec["keys_path"]):
            frozen["auxiliary_files"][str(Path(p).relative_to(out))] = sha(p)
    frozen["auxiliary_files"]["datasets/receipt.json"] = sha(pp)
    (out / "wide/frozen.json").write_text(json.dumps(frozen, indent=1) + "\n")
    changed = {"wide/frozen.json", "wide/main/real/receipt.json", "datasets/receipt.json", "_MANIFEST.json"}
    for path in source.rglob("*"):
        if path.is_file() and str(path.relative_to(source)) not in changed and sha(path) != sha(out / path.relative_to(source)):
            raise ValueError("inherited root payload changed")
    (out / "_MANIFEST.json").write_text(json.dumps(dict(schema="hdr-derived-root-v1", build_commit=a.build_commit, parent_root=str(source),
        parent_frozen_sha256=parent_pin, frozen_sha256=sha(out / "wide/frozen.json"), legs={r["name"]: dict(rows=r["rows"], train_weight=r["train_weight"], nominal_weight=r["nominal_weight"]) for r in new},
        scale=a.scale, holdout_fold=a.holdout_fold, training_launched=False), indent=1) + "\n")


if __name__ == "__main__":
    main()
