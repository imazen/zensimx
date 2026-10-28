"""Receipt-bound unnamed subjective leg. Metadata/key admission precedes payloads.

The authority digest comes from the independently frozen program/argv, never
from the leg being admitted. This module contains no population identities.
"""
import hashlib
import json
from pathlib import Path
import math
import os
import sys


def safe(path):
    path = Path(path)
    for p in (path.absolute(), path.resolve()):
        for part in p.parts:
            token = part.lower()
            if any(s in token for s in ("_sealed", "holdout", "terminal")) or token.startswith("labels__"):
                raise ValueError("protected subjective input ancestry")
    return path


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def document(path, expected=None):
    path = safe(path)
    data = path.read_bytes()
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("subjective metadata/key pin changed")
    return json.loads(data)


def command_binding(authority_path,pin,cell_id,init_seed,sample_seed,width,head,recipe):
    authority = document(authority_path,pin)
    c = authority["cells"][cell_id]
    expected_width = c["width"] if authority["schema"] in ("subjective-training-authority-v2", "subjective-training-authority-v3",
                                                           "subjective-training-authority-v4") else 1825
    if (c.get("route") not in ("d1","full","loso")
            or not 0<=c["seed_index"]<(3 if c["route"]=="loso" else 10)
            or (c["epochs"],c["pairs_per_epoch"],c["selected_epoch"])!=(120,50000,119)
            or (c["init_seed"],c["sample_seed"])!=(init_seed,sample_seed)
            or width!=expected_width or head!="N" or (recipe or {}).get("hidden",32)!=128):
        raise ValueError("subjective registered cell/seed/full-budget/head binding differs")


def family_masks(c,records):
    if c.get("route")!="loso":
        return
    masks=document(c["all_leg_masks_path"],c["all_leg_masks_sha256"])
    if (masks.get("schema")!="source-family-native-mask-v1" or masks["excluded_families"]!=c["excluded_families"]
            or set(masks["groups"])!={r["name"] for r in records}):
        raise ValueError("LOSO every inherited and added group must be masked")
    for r in records:
        m=masks["groups"][r["name"]]
        if (m["manifest_sha256"]!=r["manifest_sha256"] or m["keys_sha256"]!=r["keys_sha256"]
                or len(m["families"])!=r["rows"] or any(not isinstance(f,str) or not f
                    or f in c["excluded_families"] for f in m["families"])):
            raise ValueError("LOSO inherited/added family leak or mask pin mismatch")
    for stage in ("calibration","preprocessing"):
        m=masks[stage]
        if (c[stage+"_manifest_sha256"]!=m["manifest_sha256"] or any(not isinstance(f,str) or not f
                or f in c["excluded_families"] for f in m["families"])):
            raise ValueError("LOSO scaler/calibration family leak or pin differs")
        document(m["manifest_path"],m["manifest_sha256"])


def admit(authority_path, authority_sha256, cell_id, groups, selected_ids):
    if len(authority_sha256) != 64:
        raise ValueError("independently pinned authority required")
    a = document(authority_path, authority_sha256)
    if a.get("schema") in ("subjective-training-authority-v2", "subjective-training-authority-v3",
                           "subjective-training-authority-v4"):
        return admit_datasets(a, cell_id, groups, selected_ids)
    if (a.get("schema") != "subjective-training-authority-v1" or a.get("state") != "approved"
            or a.get("artifact_visibility") != "private" or a.get("selection") != "fixed-final-epoch"):
        raise ValueError("subjective authority not admitted")
    c = a["cells"][cell_id]
    if c["feature_ids"] != selected_ids or c["formula_revision"] != 5 or c["tier"] != "v3":
        raise ValueError("subjective feature/producer contract differs")
    if len(groups) != len(c["groups"]) or len({g[0] for g in groups}) != len(groups):
        raise ValueError("complete unique cell group inventory required")
    records = {r["name"]:r for r in c["groups"]}
    pending = []
    # First ALL paths, then ALL manifests, then keys. Never target-bearing data.
    for name, path, tw, vw, mode in groups:
        r = records[name]
        if (str(safe(path).resolve()) != str(safe(r["path"]).resolve())
                or not all(math.isfinite(v) for v in (tw,vw))
                or (tw,vw,mode) != (r["train_weight"],r["val_weight"],r["mode"])):
            raise ValueError("subjective cell group path/mode/weight mismatch")
        safe(str(path)+".manifest.json")
        safe(r["keys_path"])
        pending.append((r, Path(str(path)+".manifest.json")))
    family_masks(c,c["groups"])
    declarations = [(r, document(p,r["manifest_sha256"])) for r,p in pending]
    admitted = {}
    for r,d in declarations:
        if not r.get("subjective",False):
            continue  # Existing strict D1/oracle owner remains mandatory.
        if (d.get("schema") != "subjective-training-leg-v1"
                or d.get("loss") != "within-source-rank" or d.get("nominal_weight") != 8
                or d.get("target_orientation") != "quality" or d.get("target_transform") != "negate-pointwise"
                or d.get("target_column") != "human_score" or d.get("selection") != "fixed-final-epoch"
                or d.get("formula_revision") != 5 or d.get("tier") != "v3"
                or d.get("feature_ids") != selected_ids or d.get("producer_commit") != c["producer_commit"]
                or r["name"] != "subjective_rank" or r["mode"] != "withinref,rank" or r["val_weight"] != 0
                or r["train_weight"] <= 0 or d.get("table_sha256") != r["table_sha256"]):
            raise ValueError("subjective role/loss/producer/target binding differs")
        admitted[r["name"]] = (r,d)
    if len(admitted) != 1:
        raise ValueError("authority requires one aggregated unnamed subjective leg")
    for r,d in admitted.values():
        keys = document(r["keys_path"],r["keys_sha256"])
        if len(keys) != r["rows"] or d["rows"] != r["rows"] or not keys:
            raise ValueError("subjective ordered key count differs")
        observations = set()
        source_counts = {}
        references = {}
        for i,key in enumerate(keys):
            if (set(key) != {"observation", "source", "family", "reference", "role", "tier", "row"}
                    or key["row"] != i or key["role"] != "fit" or key["tier"] != "T2"
                    or key["source"] not in c["sources"] or key["family"] in c["excluded_families"]
                    or key["observation"] in observations
                    or any(not isinstance(key[k], str) or not key[k] for k in ("observation","source","family","reference"))):
                raise ValueError("subjective source/observation/order/role mismatch")
            if references.setdefault(key["source"],key["reference"]) != key["reference"]:
                raise ValueError("subjective source must have one pairing reference")
            observations.add(key["observation"])
            source_counts[key["source"]] = source_counts.get(key["source"],0)+1
        if (source_counts != r["source_counts"] or any(n < 2 for n in source_counts.values())
                or set(source_counts) != set(c["sources"])
                or len(set(references.values())) != len(references)):
            raise ValueError("subjective source population differs from authority")
        expected = 8/(sum(1-1/source_counts[k] for k in sorted(source_counts))/len(source_counts))
        if r["train_weight"] != expected:
            raise ValueError("subjective nominal-eight acceptance correction differs")
    return {name:d for name,(_,d) in admitted.items()}


def admit_datasets(a, cell_id, groups, selected_ids):
    """Revision-7 TRAIN datasets; v1's registered experiment stays unchanged.

    Separate datasets have a declared positive nominal mass each. Uniform within-reference
    sampling uses the inherited acceptance correction (including self draws).
    Hard preferences contain exactly two rows per comparison, never a pooled
    list of scores for all distortions of the reference.
    """
    if (a.get("state") != "approved" or a.get("artifact_visibility") != "private"
            or a.get("selection") != "fixed-final-epoch"):
        raise ValueError("dataset authority not admitted")
    c = a["cells"][cell_id]
    aic_only = diagnostic_binding(a, cell_id, groups) if a["schema"] == "subjective-training-authority-v4" else False
    if a["schema"] in ("subjective-training-authority-v3", "subjective-training-authority-v4"):
        if "continuation" not in c:
            raise ValueError("continuation authority requires checkpoint binding")
        groups = [g for g in groups if g[0] != "aic_public_train"]
    elif "continuation" in c:
        raise ValueError("continuation requires version-3 authority")
    if c["feature_ids"] != selected_ids or c["formula_revision"] != 7 or c["tier"] != "v3":
        raise ValueError("dataset feature/producer contract differs")
    guard = document(c["family_guard_path"], c["family_guard_sha256"])
    if guard.get("schema") != "dataset-family-guard-v1" or guard.get("inherited_leaks") != 0:
        raise ValueError("dataset family guard differs")
    excluded = set(guard["protected_families"])
    records = {r["name"]: r for r in c["groups"]}
    if len(records) != len(groups) or set(records) != {g[0] for g in groups}:
        raise ValueError("complete unique dataset cell inventory required")
    pending = []
    for name, path, tw, vw, mode in groups:
        r = records[name]
        if (safe(path).resolve() != safe(r["path"]).resolve()
                or not all(math.isfinite(v) for v in (tw, vw))
                or (tw, vw, mode) != (r["train_weight"], r["val_weight"], r["mode"])):
            raise ValueError("dataset cell group path/mode/weight mismatch")
        pending.append((r, document(str(path)+".manifest.json", r["manifest_sha256"])))
    admitted = {}
    for r, d in pending:
        if not r.get("subjective"):
            continue
        votes = d.get("schema") == "subjective-training-leg-v3"
        legacy = (d.get("schema") == "subjective-training-leg-v2"
                  and d.get("loss") in ("within-source-rank", "hard-pair-preference")
                  and d.get("target_transform") == "orient-native-quality"
                  and r["name"].startswith("subjective_rank_"))
        vote_contract = (votes and d.get("loss") == "binomial-two-vote"
                         and d.get("target_transform") == "native-vote-probability"
                         and d.get("collection") == "bapps" and d.get("native_votes_per_pair") == 2
                         and d.get("nominal_weight") == 0.5 and r["name"] == "subjective_vote_bapps"
                         and d.get("keys_path") == r["keys_path"] and d.get("keys_sha256") == r["keys_sha256"])
        if (not (legacy or vote_contract)
                or not isinstance(d.get("nominal_weight"), (float, int))
                or not math.isfinite(d["nominal_weight"]) or d["nominal_weight"] <= 0
                or d.get("target_orientation") != "quality"
                or d.get("target_column") != "human_score" or d.get("formula_revision") != 7
                or d.get("tier") != "v3" or d.get("feature_ids") != selected_ids
                or d.get("selection") != "fixed-final-epoch" or d.get("native_role") != "TRAIN"
                or d.get("producer_commit") != c["producer_commit"]
                or r["mode"] != "withinref,rank"
                or r["val_weight"] != 0 or r["train_weight"] <= 0
                or d.get("table_sha256") != r["table_sha256"]):
            raise ValueError("dataset role/loss/producer/target binding differs")
        if votes:
            safe(d["native_votes_path"])
            if (not isinstance(d.get("native_votes_sha256"), str)
                    or len(d["native_votes_sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in d["native_votes_sha256"])):
                raise ValueError("native vote payload pin required")
        keys = document(r["keys_path"], r["keys_sha256"])
        if not keys or len(keys) != r["rows"] or d["rows"] != r["rows"]:
            raise ValueError("dataset ordered key count differs")
        counts, observations = {}, set()
        for i, k in enumerate(keys):
            if (set(k) != {"observation", "source", "family", "reference", "role", "tier", "row"}
                    or k["row"] != i or k["role"] != "fit" or k["tier"] != "T2"
                    or k["family"] in excluded or k["observation"] in observations
                    or k["source"] != k["reference"]
                    or any(not isinstance(k[f], str) or not k[f] for f in
                           ("observation", "source", "family", "reference"))):
                raise ValueError("dataset key role/order/family mismatch")
            observations.add(k["observation"])
            counts[k["source"]] = counts.get(k["source"], 0)+1
        if counts != r["source_counts"] or any(n < 2 for n in counts.values()):
            raise ValueError("dataset source population mismatch")
        if (d["loss"] == "hard-pair-preference" or votes) and any(n != 2 for n in counts.values()):
            raise ValueError("hard preference requires exactly two sides per comparison")
        weight = d["nominal_weight"]/(sum(1-1/counts[k] for k in sorted(counts))/len(counts))
        if r["train_weight"] != weight:
            raise ValueError("dataset nominal-weight acceptance correction differs")
        admitted[r["name"]] = d
    # The AIC-only diagnostic arm explicitly excludes every training leg; the
    # exclusion is bound to the unmodified base by diagnostic_binding.
    if not admitted and not aic_only:
        raise ValueError("dataset authority has no subjective groups")
    return admitted


# Registered continuation protocols beyond the frozen 0/4/8-epoch contract
# (byconstruction record, 2026-10-27): each names its populations with their
# public_train weight, and its retained checkpoint epochs. Every one runs 60
# epochs at constant LR 1e-5, fresh Adam, a checkpoint dump every epoch. The
# native owner holds the same table.
DIAGNOSTIC_PROTOCOL = "aic-matched-continuation-diagnostic-v1"
FRONTIER_PROTOCOL = "aic-share-frontier-v1"
# Stage D: the base mixture plus one rehearsal leg, the human leg again with its pairs drawn within each older
# dataset (the native owner names the datasets from the keys' member sets).
XREF_PROTOCOL = "aic-share-xref-v1"
WITHIN_SET_GROUP = "human_xref"
DIAGNOSTIC_PROTOCOLS = {
    DIAGNOSTIC_PROTOCOL: ({"mixed-aic16": 16.0, "aic-only": 16.0}, [7, 59]),
    FRONTIER_PROTOCOL: ({"mixed-aic64": 64.0, "mixed-aic256": 256.0, "mixed-aic1024": 1024.0}, [7, 15, 29, 59]),
    XREF_PROTOCOL: ({"mixed-aic256-xref": 256.0, "mixed-aic1024-xref": 1024.0}, [7, 15, 29, 59]),
}
DIAGNOSTIC_EPOCHS = 60
DIAGNOSTIC_POPULATIONS = tuple(p for populations, _ in DIAGNOSTIC_PROTOCOLS.values() for p in populations)


def diagnostic_protocol(population):
    """(protocol, public_train weight, retained checkpoint epochs) of one registered population."""
    for protocol, (populations, checkpoints) in DIAGNOSTIC_PROTOCOLS.items():
        if population in populations:
            return protocol, populations[population], list(checkpoints)
    raise ValueError("diagnostic population required")


def diagnostic_partition(base_groups, population):
    """(retained, excluded) base group records; the native owner applies the same rule."""
    diagnostic_protocol(population)
    if any(not isinstance(r.get("train_weight"), (int, float)) or not math.isfinite(r["train_weight"])
           or r["train_weight"] < 0 for r in base_groups):
        raise ValueError("diagnostic base train weights required")
    if diagnostic_protocol(population)[0] == XREF_PROTOCOL:
        return [*base_groups, xref_record(base_groups)], []
    if population != "aic-only":
        return list(base_groups), []
    return ([r for r in base_groups if r["train_weight"] == 0],
            [r for r in base_groups if r["train_weight"] > 0])


def xref_record(base_groups):
    """The rehearsal leg: the base within-reference human leg's own record under its name, paired by dataset."""
    human = [r for r in base_groups if r["name"] == "human"]
    if len(human) != 1 or human[0]["mode"] != "withinref,rank" or human[0].get("subjective"):
        raise ValueError("rehearsal requires the base within-reference human leg")
    return dict(human[0], name=WITHIN_SET_GROUP, mode="withinset,rank")


def diagnostic_binding(a, cell_id, groups):
    """Mirror of the native diagnostic admission; returns whether the arm is AIC-only."""
    base = document(a["base_authority"], a["base_authority_sha256"])
    if base.get("schema") != "subjective-training-authority-v2":
        raise ValueError("diagnostic base must be the original version-2 authority")
    outer = {"schema", "base_authority", "base_authority_sha256", "cells"}
    if {k: v for k, v in a.items() if k not in outer} != {k: v for k, v in base.items() if k not in outer}:
        raise ValueError("diagnostic authority changed a base field")
    if set(a["cells"]) != set(base["cells"]):
        raise ValueError("diagnostic cell inventory differs from its base")
    if any(a["cells"][k] != base["cells"][k] for k in a["cells"] if k != cell_id):
        raise ValueError("diagnostic authority changed another cell")
    c, b = a["cells"][cell_id], base["cells"][cell_id]
    inner = {"groups", "continuation"}
    if "continuation" in b or {k: v for k, v in c.items() if k not in inner} != {k: v for k, v in b.items() if k not in inner}:
        raise ValueError("diagnostic cell changed a base binding")
    w = c["continuation"]
    protocol, weight, checkpoints = diagnostic_protocol(w.get("population"))
    retained, excluded = diagnostic_partition(b["groups"], w["population"])
    aic_only = w["population"] == "aic-only"
    if (w.get("protocol") != protocol or w.get("epochs") != DIAGNOSTIC_EPOCHS
            or w.get("aic_weight") != weight or w.get("dump_checkpoints_every") != 1
            or w.get("retained_checkpoint_epochs") != checkpoints or w.get("optimizer_state") != "fresh"
            or c["groups"] != retained or w.get("excluded_groups") != excluded
            or (aic_only and (not retained or not excluded))):
        raise ValueError("diagnostic protocol/population/group partition differs")
    aic = [g for g in groups if g[0] == "aic_public_train"]
    if len(aic) != 1 or aic[0][2] != weight:
        raise ValueError(f"diagnostic requires one public_train leg at weight {weight:g}")
    return aic_only


def diagnostic_authority(binding, checkpoint, population, lr):
    """Bind one arm of the registered diagnostic to its unmodified version-2 base.

    Returns the new binding and the retained group names. The frozen
    continuation_authority contract and its 0/4/8-epoch budgets are unchanged.
    """
    path, pin, cell, selected = binding
    a = document(path, pin)
    if a["schema"] != "subjective-training-authority-v2" or lr != 0.00001:
        raise ValueError("diagnostic requires the original version-2 authority and LR 1e-5")
    checkpoint = safe(checkpoint).resolve()
    protocol, weight, checkpoints = diagnostic_protocol(population)
    retained, excluded = diagnostic_partition(a["cells"][cell]["groups"], population)
    a["schema"] = "subjective-training-authority-v4"
    a["base_authority"] = str(Path(path).resolve())
    a["base_authority_sha256"] = pin
    a["cells"][cell]["groups"] = retained
    a["cells"][cell]["continuation"] = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        epochs=DIAGNOSTIC_EPOCHS, lr=lr, lr_schedule="constant", optimizer_state="fresh",
        protocol=protocol, population=population, aic_weight=weight,
        dump_checkpoints_every=1, retained_checkpoint_epochs=checkpoints, excluded_groups=excluded)
    out = Path(path).with_name("diagnostic-authority.json")
    if out.exists():
        raise ValueError("preserve previous diagnostic authority")
    out.write_text(json.dumps(a, sort_keys=True, separators=(",", ":"))+"\n")
    return (out, sha(out), cell, selected), [r["name"] for r in retained]


def continuation_authority(binding, checkpoint, epochs, lr):
    """Bind a new continuation without changing the original full-fit authority."""
    path, pin, cell, selected = binding
    a = document(path, pin)
    if a["schema"] != "subjective-training-authority-v2" or epochs not in (0, 4, 8) or lr != 0.00001:
        raise ValueError("unregistered continuation budget")
    checkpoint = safe(checkpoint).resolve()
    a["schema"] = "subjective-training-authority-v3"
    a["base_authority_sha256"] = pin
    a["cells"][cell]["continuation"] = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        epochs=epochs, lr=lr, lr_schedule="constant", optimizer_state="fresh")
    out = path.with_name("continuation-authority.json")
    if out.exists():
        raise ValueError("preserve previous continuation authority")
    out.write_text(json.dumps(a, sort_keys=True, separators=(",", ":"))+"\n")
    return out, sha(out), cell, selected


def payloads(admitted, groups):
    """Called only after every group passed native/wrapper metadata admission."""
    for name,path,_,_,_ in groups:
        if name in admitted and sha(path) != admitted[name]["table_sha256"]:
            raise ValueError("subjective payload pin changed")
        if name in admitted and admitted[name]["schema"] == "subjective-training-leg-v3":
            import pyarrow.parquet as pq
            d = admitted[name]
            validate_native_votes(d, pq.read_table(path, columns=["human_score"])["human_score"].to_pylist())


def validate_native_votes(declaration, targets):
    """Payload phase only: bind native counts to both ordered TRAIN targets."""
    d = declaration
    votes = document(d["native_votes_path"], d["native_votes_sha256"])
    keys = document(d["keys_path"], d["keys_sha256"])
    comparisons = votes.get("comparisons", [])
    if (votes.get("schema") != "bapps-native-votes-v1" or not comparisons
            or len(comparisons)*2 != len(keys) or len(keys) != len(targets)):
        raise ValueError("native vote population differs")
    for i, v in enumerate(comparisons):
        if set(v) != {"source", "n_a", "n_b", "n_tie", "n", "p_a"}:
            raise ValueError("native vote count fields required")
        if (any(type(v[k]) is not int for k in ("n_a", "n_b", "n_tie", "n"))
                or min(v["n_a"], v["n_b"]) < 0 or v["n_a"]+v["n_b"] != 2
                or v["n_tie"] != 0 or v["n"] != 2 or v["p_a"] != v["n_a"]/2):
            raise ValueError("native two-vote count/probability binding differs")
        for side, probability, row in (("a", v["p_a"], 2*i), ("b", 1-v["p_a"], 2*i+1)):
            key = keys[row]
            if (key["source"] != v["source"] or key["observation"] != v["source"]+":"+side
                    or targets[row] != probability):
                raise ValueError("native vote ordered key/target binding differs")
        if keys[2*i]["family"] != keys[2*i+1]["family"]:
            raise ValueError("native vote pair family differs")


DATASET_FOLDS = {
    "pipal": ("pipal",),
    "pairs": ("bapps", "pieapp", "fg-iqa"),
    "mos7": ("mdid", "live-md", "cid-iq", "cid-oa", "vcl-fer", "irccyn-ivc",
             "colourlab-rating-variability-2022"),
}


def linked_subset_exclusions(root, fold):
    """Read the label-free, frozen linkage authority; never infer from labels."""
    path = root/"datasets/receipt.json"
    if not path.exists():
        return ()
    packet = document(path)
    link = packet.get("linked_subset_omissions")
    if link is None:
        if packet.get("subset_policy_sha256"):
            raise ValueError("subset holdouts require the frozen linked omission authority")
        return ()
    frozen = document(root/"wide/frozen.json")
    if frozen.get("dataset_legs_sha256") != sha(path):
        raise ValueError("linked subset root receipt is not frozen")
    relative = "datasets/linked-subset-omissions.json"
    if link["path"] != relative or frozen["auxiliary_files"].get(relative) != link["sha256"]:
        raise ValueError("linked subset authority is not bound by the frozen root")
    authority = document(root/relative, link["sha256"])
    if (authority.get("schema") != "dataset-linked-subset-omissions-v1"
            or authority.get("labels_read") is not False
            or authority["parent_frozen_sha256"] != link["parent_frozen_sha256"]
            or set(authority["folds"]) != {"kadid", "tid2013", "konfig", "cid22_a25", *DATASET_FOLDS}):
        raise ValueError("linked subset authority scope differs")
    added = {r["collection"] for r in packet["groups"] if "protocol" in r}
    if set(authority["added_collections"]) != added:
        raise ValueError("linked subset collection inventory differs")
    excluded = authority["folds"][fold]["excluded_subset_collections"]
    if len(excluded) != len(set(excluded)) or not set(excluded) <= added:
        raise ValueError("linked subset exclusion inventory differs")
    return tuple(excluded)


def log_dataset_read(dest, path, rows, columns, kind, **bindings):
    """Persist a per-cell authorization before interpreting TRAIN target values."""
    ledger = dest / "LABEL_READS.json"
    value = document(ledger) if ledger.exists() else dict(schema="dataset-cell-label-reads-v1", role="TRAIN", reads=[])
    value["reads"].append(dict(table=str(path), rows=dict(start=0, stop=rows), columns=columns,
                               kind=kind, state="authorized-before-read", **bindings))
    ledger.write_text(json.dumps(value, indent=1) + "\n")


def read_train_panel(path):
    """Optional pre-read ledger shared by the existing TRAIN diagnostic owners."""
    directory = os.environ.get("BYC_LABEL_LEDGER_DIR")
    if directory:
        dest = Path(directory) / (Path(sys.argv[0]).stem + "-" + str(os.getpid()))
        dest.mkdir(parents=True, exist_ok=True)
        log_dataset_read(dest, path, "all", ["datasets[*].target", "datasets[*].orientation"],
                         "E21-TRAIN-diagnostic", table_sha256=sha(path), argv=sys.argv)
    return document(path)


def dataset_groups(root, groups, selected, route, seed_index, init_seed, sample_seed, dest, exclude=(),
                   derived_inputs="fx1"):
    """Attach the frozen derived root's dataset legs to the canonical fitter.

    `derived_inputs` is the run's pinned derived-input recipe token; the root's legs must have been admitted
    for exactly its direct ids (`ring_root.py selection-rebind` binds a root to another one)."""
    frozen = document(root/"wide/frozen.json")
    pin = frozen.get("dataset_legs_sha256")
    if pin is None:
        return groups, None
    p = document(root/"datasets/receipt.json", pin)
    if p.get("schema") != "rev7-dataset-legs-v1" or p["feature_ids"] != selected:
        raise ValueError("frozen dataset leg recipe differs")
    guard = document(p["family_guard_path"],p["family_guard_sha256"])
    if guard.get("schema") != "dataset-family-guard-v1" or guard.get("inherited_leaks") != 0:
        raise ValueError("dataset family guard differs")
    exclude = tuple(exclude)
    supported = {c for group in DATASET_FOLDS.values() for c in group} | {r["collection"] for r in p["groups"]}
    if len(exclude) != len(set(exclude)) or not set(exclude) <= supported:
        raise ValueError("unique supported dataset exclusions required")
    if p.get("linked_subset_omissions"):
        folds = [route] if route in ("kadid", "tid2013", "konfig", "cid22_a25") else [
            f for f, cs in DATASET_FOLDS.items() if set(cs) <= set(exclude)]
        required = {c for fold in folds for c in linked_subset_exclusions(root, fold)}
        if not required <= set(exclude):
            raise ValueError("required linked subset was retained in the held-out fit")
    retained = [r for r in p["groups"] if r["collection"] not in exclude]
    selection = dict(schema="dataset-group-selection-v1", root=str(root), root_sha256=sha(root/"wide/frozen.json"),
                     receipt_sha256=pin, requested_exclusions=list(exclude),
                     omitted_collections=[r["collection"] for r in p["groups"] if r["collection"] in exclude],
                     retained_collections=[r["collection"] for r in retained])
    (dest/"DATASET_GROUP_SELECTION.json").write_text(json.dumps(selection, indent=1)+"\n")
    for name, path, tw, vw, mode in groups:
        declaration = document(str(path)+".manifest.json")
        log_dataset_read(dest, path, declaration.get("rows_kept", "all"), ["ref_basename", "human_score"],
                         "inherited-native-fit", name=name, train_weight=tw, val_weight=vw,
                         manifest_sha256=sha(str(path)+".manifest.json"))
    for r in retained:
        log_dataset_read(dest, Path(r["path"]), r["rows"], ["ref_basename", "human_score"], "derived-native-fit",
                         collection=r["collection"], table_sha256=r["table_sha256"],
                         manifest_sha256=r["manifest_sha256"])
        d = document(str(r["path"])+".manifest.json", r["manifest_sha256"])
        if d["schema"] == "subjective-training-leg-v3":
            log_dataset_read(dest, d["native_votes_path"], r["rows"]//2,
                             ["source", "n_a", "n_b", "n_tie", "n", "p_a"], "native-vote-counts",
                             table_sha256=d["native_votes_sha256"])
    if not retained:
        return groups, None
    records = []
    for name, path, tw, vw, mode in groups:
        records.append(dict(name=name, path=str(path), train_weight=tw, val_weight=vw, mode=mode,
                            manifest_sha256=sha(str(path)+".manifest.json"),
                            keys_path=str(path.with_suffix(".keys.parquet")), subjective=False))
    records += retained
    expanded = [(r["name"],Path(r["path"]),r["train_weight"],r["val_weight"],r["mode"]) for r in records]
    cell_id = f"{route}-s{seed_index}"
    cell = dict(route="full" if route == "full" else "d1",seed_index=seed_index,epochs=120,pairs_per_epoch=50000,
                selected_epoch=119,init_seed=init_seed,sample_seed=sample_seed,feature_ids=selected,
                formula_revision=7,tier="v3",producer_commit=p["producer_commit"],groups=records,
                width=document(root/"wide/main/real/receipt.json")["width"],
                family_guard_path=p["family_guard_path"],family_guard_sha256=p["family_guard_sha256"])
    from v2_common import derived_declaration
    direct = derived_declaration(derived_inputs)[2]
    if selected != direct["direct"]:
        raise ValueError("dataset M recipe requires its pinned direct columns")
    cell["read_ids"] = sorted(set(selected) | {b for _,b in direct["products"]})
    a = dict(schema="subjective-training-authority-v2",state="approved",artifact_visibility="private",
             selection="fixed-final-epoch",cells={cell_id:cell})
    ap = dest/"dataset-authority.json"
    ap.write_text(json.dumps(a,sort_keys=True,separators=(",",":"))+"\n")
    binding = (ap,sha(ap),cell_id,selected)
    admit(ap,binding[1],cell_id,expanded,selected)
    return expanded,binding
