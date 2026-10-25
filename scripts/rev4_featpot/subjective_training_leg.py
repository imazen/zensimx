"""Receipt-bound unnamed subjective leg. Metadata/key admission precedes payloads.

The authority digest comes from the independently frozen program/argv, never
from the leg being admitted. This module contains no population identities.
"""
import hashlib
import json
from pathlib import Path
import math


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
    expected_width = c["width"] if authority["schema"] == "subjective-training-authority-v2" else 1825
    if (c.get("route") not in ("d1","full","loso")
            or not 0<=c["seed_index"]<(3 if c["route"]=="loso" else 10)
            or (c["epochs"],c["pairs_per_epoch"],c["selected_epoch"])!=(120,50000,119)
            or (c["init_seed"],c["sample_seed"])!=(init_seed,sample_seed)
            or width!=expected_width or head!="N" or (recipe or {}).get("hidden",32)!=128):
        raise ValueError("subjective registered cell/seed/full-budget/head binding differs")


def family_masks(c,records):
    if c.get("route")!="loso": return
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
    if a.get("schema") == "subjective-training-authority-v2":
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

    Separate datasets have nominal 0.5 mass each. Uniform within-reference
    sampling uses the inherited acceptance correction (including self draws).
    Hard preferences contain exactly two rows per comparison, never a pooled
    list of scores for all distortions of the reference.
    """
    if (a.get("state") != "approved" or a.get("artifact_visibility") != "private"
            or a.get("selection") != "fixed-final-epoch"):
        raise ValueError("dataset authority not admitted")
    c = a["cells"][cell_id]
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
        if (d.get("schema") != "subjective-training-leg-v2" or d.get("loss") not in
                ("within-source-rank", "hard-pair-preference")
                or d.get("nominal_weight") != 0.5 or d.get("target_orientation") != "quality"
                or d.get("target_transform") != "orient-native-quality"
                or d.get("target_column") != "human_score" or d.get("formula_revision") != 7
                or d.get("tier") != "v3" or d.get("feature_ids") != selected_ids
                or d.get("selection") != "fixed-final-epoch" or d.get("native_role") != "TRAIN"
                or d.get("producer_commit") != c["producer_commit"]
                or not r["name"].startswith("subjective_rank_") or r["mode"] != "withinref,rank"
                or r["val_weight"] != 0 or r["train_weight"] <= 0
                or d.get("table_sha256") != r["table_sha256"]):
            raise ValueError("dataset role/loss/producer/target binding differs")
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
        if d["loss"] == "hard-pair-preference" and any(n != 2 for n in counts.values()):
            raise ValueError("hard preference requires exactly two sides per comparison")
        weight = 0.5/(sum(1-1/counts[k] for k in sorted(counts))/len(counts))
        if r["train_weight"] != weight:
            raise ValueError("dataset nominal-half acceptance correction differs")
        admitted[r["name"]] = d
    if not admitted:
        raise ValueError("dataset authority has no subjective groups")
    return admitted


def payloads(admitted, groups):
    """Called only after every group passed native/wrapper metadata admission."""
    for name,path,_,_,_ in groups:
        if name in admitted and sha(path) != admitted[name]["table_sha256"]:
            raise ValueError("subjective payload pin changed")


def dataset_groups(root, groups, selected, route, seed_index, init_seed, sample_seed, dest):
    """Attach the frozen derived root's dataset legs to the canonical fitter."""
    frozen = document(root/"wide/frozen.json")
    pin = frozen.get("dataset_legs_sha256")
    if pin is None:
        return groups, None
    p = document(root/"datasets/receipt.json", pin)
    if p.get("schema") != "rev7-dataset-legs-v1" or p["feature_ids"] != selected:
        raise ValueError("frozen dataset leg recipe differs")
    records = []
    for name, path, tw, vw, mode in groups:
        records.append(dict(name=name, path=str(path), train_weight=tw, val_weight=vw, mode=mode,
                            manifest_sha256=sha(str(path)+".manifest.json"),
                            keys_path=str(path.with_suffix(".keys.parquet")), subjective=False))
    records += p["groups"]
    expanded = [(r["name"],Path(r["path"]),r["train_weight"],r["val_weight"],r["mode"]) for r in records]
    cell_id = f"{route}-s{seed_index}"
    cell = dict(route="full" if route == "full" else "d1",seed_index=seed_index,epochs=120,pairs_per_epoch=50000,
                selected_epoch=119,init_seed=init_seed,sample_seed=sample_seed,feature_ids=selected,
                formula_revision=7,tier="v3",producer_commit=p["producer_commit"],groups=records,
                width=document(root/"wide/main/real/receipt.json")["width"],
                family_guard_path=p["family_guard_path"],family_guard_sha256=p["family_guard_sha256"])
    a = dict(schema="subjective-training-authority-v2",state="approved",artifact_visibility="private",
             selection="fixed-final-epoch",cells={cell_id:cell})
    ap = dest/"dataset-authority.json"
    ap.write_text(json.dumps(a,sort_keys=True,separators=(",",":"))+"\n")
    binding = (ap,sha(ap),cell_id,selected)
    admit(ap,binding[1],cell_id,expanded,selected)
    return expanded,binding
