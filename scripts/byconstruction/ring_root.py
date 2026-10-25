"""Derive a training root whose columns come from a measurement arm (by-construction exploration).

Exploration record: benchmarks/byconstruction_2026-10-22.md ("Ringing corrected: high-pass form, trained and tested";
"Coarse scales in linear light"). The source root (the v40r4 `v2e29` root every E21 arm trained on) stays untouched.
The new root holds the same tables, rows, keys, labels and split decisions; only the arm's columns are replaced with
values from a bank extracted under it. Every other measured column of the new bank must equal the stored table bit for
bit (float64 bank values cast to float32, as `v2c_wide` writes them); that check is what proves each row joined to the
right pixels and that the arm moved nothing else. Arms (`--arm`, default hpm):

  hpm     `ZENSIM_RING_FORM=hpm`: the 12 ringing columns (v2 `idx::RINGING` over the 12 scale x channel cells)
  linpyr  `ZENSIM_PYRAMID=linear`: every column of scales 1-3, and scale 0's edge-width change (a scale 0/1 ratio)
  rev7    `ZENSIM_FORMULA_REV=7` (raw ringing, owner 2026-10-25): every measured column; the tables declare formula
          revision 7 and the new extractor, and each replaced value must stay close to the stored Rev5 value

  pool   --bin EXTRACTOR --out DIR             extract the E15 coverage-pool selection (44,000 pairs) under the arm
  build  --bank DIR --pool DIR --out ROOT      write the derived root and re-pin its receipt and freeze
  globaff-pairs --bank DIR --pool DIR --out DIR   one pairs TSV per bank set (row_id = bank row) and the pool's
  globaff --root ROOT --bank DIR --sidecars DIR --out ROOT2
                                               widen a derived root with the byc_globaff descriptor at f1867-f1890
  orient --root ROOT --bank DIR --sidecars DIR --out ROOT2
                                               widen a root with the byc_orient damage (Haar H/V/D band-difference RMS
                                               per scale x channel) at f2301-f2336; sidecars from the globaff-pairs TSVs
  gate-table --table T.parquet --decl FX1G.json --out T2.parquet [--channels XYB|Y]
                                               the same gating for one width-1891 table (mixture tables, AIC folds)
  gated --root ROOT2 --decl FX1G.json --out ROOT3 [--channels XYB|Y]
                                               widen a descriptor root with gated damage from f1891: for each of fx1's
                                               differences d (ascending) in a gated channel, d x (1 - share) of its own
                                               scale and channel cell, as an f32 product of f32 factors. XYB (default)
                                               gates all 410 (f1891-f2300); Y only the 164 luminance-cell differences
                                               (f1891-f2054)
  gated-decl --decl FX1G.json --out DECL.json [--channels XYB|Y]
                                               the training declaration of a gated root: fx1's 410 differences and the
                                               gated ids direct, fx1's 410 fragility products

The HDR legs (`hdr_hb4`, `hdr_hc4`, leg `hdr_consensus`) are not re-extracted and are left out of the derived root.
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SOURCE = Path("/mnt/v/output/zensim/v40r4-2026-10-08/v2e29")
POOL_SOURCE = Path("/var/tmp/rev5-featpot/v2c/e15")
RING_IDS = [372 + 29 * cell + 26 for cell in range(12)]  # v2 block f372.., 29 per (scale, channel) cell, RINGING = 26
MEASURED = [i for lo, hi in ((0, 228), (372, 720)) for i in range(lo, hi)]
ERA = "rev5_localwin_ringhpm"
HDR = ("hdr_hb4", "hdr_hc4")


def scale_of(i: int) -> int:
    """Pyramid scale of measured feature `i`: (scale, channel) cells of 13 basic, 6 peak and 29 v2 features."""
    if i < 156:
        return i // 39
    if i < 228:
        return (i - 156) // 18
    return (i - 372) // 87


EDGE_WIDTH_S0 = [372 + 29 * ch + 28 for ch in range(3)]  # v2 EDGE_WIDTH_CHANGE = 28: scale 0's two-scale ratio
ARMS = {
    "hpm": {"env": {"ZENSIM_RING_FORM": "hpm"}, "manifest": ("ring_form", "hpm"), "era": ERA, "replaced": RING_IDS,
            "record": "byc_ring_root",
            "note": f"; ringing f{RING_IDS[0]}..f{RING_IDS[-1]} (12 ids) re-extracted under ZENSIM_RING_FORM=hpm (byconstruction)"},
    "linpyr": {"env": {"ZENSIM_PYRAMID": "linear"}, "manifest": ("pyramid", "linear"), "era": "rev5_localwin_linpyr",
               "replaced": sorted([i for i in MEASURED if scale_of(i) > 0] + EDGE_WIDTH_S0), "record": "byc_linpyr_root",
               "note": "; scales 1-3 and scale-0 edge-width change re-extracted under ZENSIM_PYRAMID=linear (byconstruction)"},
    "rev7": {"env": {"ZENSIM_FORMULA_REV": "7"}, "manifest": ("formula_revision", "Rev7"), "era": "rev7_rowlanes",
             "replaced": MEASURED, "record": "byc_rev7_root", "revision": 7, "close": True,
             "note": "; every measured slot re-extracted at formula revision 7 (rowlanes, raw ringing; byconstruction)"},
}


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


DATASET_MOS = ("pipal", "mdid", "live-md", "cid-iq", "cid-oa", "vcl-fer", "irccyn-ivc",
               "colourlab-rating-variability-2022")
DATASET_PAIR = ("bapps", "pieapp", "fg-iqa")


def dataset_family_guard(admission, source, out):
    """Label-free closure of new reservations through native hashes and D4 aliases.

    Include excluded reservations too: exclusion from an assessment does not
    grant fit permission. This audit refuses inherited collisions rather than
    silently rewriting the frozen revision-7 recipe's populations.
    """
    memberships, protected, protected_hashes = {}, set(), set()
    validation_path = admission/"validation-receipt.json"
    if sha(validation_path) != "a4013c0bb1a72b5b4990cc78eb7003acb0e87e2374a01c7d5c8a47bf1d82f9d8":
        raise ValueError("accepted membership receipt changed")
    accepted = json.loads(validation_path.read_text())["collections"]
    for p in sorted(admission.glob("*/membership.parquet")):
        if p.parent.name == "ls2n-ipi-sr":
            continue
        if sha(p) != accepted[p.parent.name]["membership_sha256"]:
            raise ValueError("frozen membership changed")
        rows = pq.read_table(p).to_pylist()
        memberships[p.parent.name] = rows
        for r in rows:
            if r["role"] in ("EVAL", "TEST", "CALIBRATION", "SOURCE_EXPOSED_TEST_DIAGNOSTIC"):
                protected.add(str(r["family_id"]))
                protected_hashes.update(h for h in r["source_audit_rgb8_sha256"] if h)
    d4 = Path("/mnt/v/zensim/d4-roles-private")
    freeze = json.loads((admission / "family-registry-freeze.json").read_text())
    for file, key in (("images.json", "source_authority_inventory_sha256"),
                      ("source-family-authority.json", "source_authority_sha256")):
        if sha(d4 / file) != freeze[key]:
            raise ValueError("D4 family authority changed since admission")
    images = json.loads((d4 / "images.json").read_text())
    authority = json.loads((d4 / "source-family-authority.json").read_text())["uid_to_family"]
    aliases = {images[int(uid)]["rgb_sha256"]: family for uid, family in authority.items()}
    del images
    reserved_aliases = {aliases[h] for h in protected_hashes if h in aliases}
    def reserved(h):
        return h in protected_hashes or aliases.get(h) in reserved_aliases
    bank = Path(json.loads((source / "wide/main/real/receipt.json").read_text())["admission_view"]["bank_root"])
    collisions = {}
    for p in sorted(bank.glob("*/keys.parquet")):
        keys = pq.read_table(p).to_pylist()
        collisions[p.parent.name] = {r["ref_group"] for r in keys if reserved(r["ref_pixels_sha256"])}
    inherited = {}
    for p in sorted((source / "wide/main/real").glob("*.keys.parquet")):
        if not p.name.endswith(("_fit.keys.parquet", "_dev.keys.parquet")):
            continue
        keys = pq.read_table(p).to_pylist()
        bad = [r for r in keys if r["ref_basename"] in collisions.get(r["member_set"], set())]
        inherited[p.name] = {"rows": len(keys), "reserved_rows": len(bad),
                             "references": sorted({r["ref_basename"] for r in bad})}
    # E15 uses source filenames, unlike the bank-keyed human/teacher legs.
    coverage = pq.read_table(source / "e15/coverage_pool.keys.parquet").to_pylist()
    source_images = json.loads((admission / "screen-v1/source-images.json").read_text())
    bad_names = {Path(p).name for r in source_images if reserved(r["rgb_sha256"]) for p in r["paths"]}
    del source_images
    coverage_bad = [r for r in coverage if r["source_filename"] in bad_names]
    guard = dict(schema="dataset-family-guard-v1", protected_families=sorted(protected),
                 protected_roles=["EVAL", "TEST", "CALIBRATION", "SOURCE_EXPOSED_TEST_DIAGNOSTIC"],
                 includes_excluded_memberships=True, inherited=inherited,
                 coverage_reserved_rows=len(coverage_bad), coverage_reserved_sources=sorted(
                     {r["source_filename"] for r in coverage_bad}),
                 inherited_leaks=sum(r["reserved_rows"] for r in inherited.values())+len(coverage_bad),
                 membership_sha256={c: sha(admission/c/"membership.parquet") for c in memberships},
                 source_frozen_sha256=sha(source/"wide/frozen.json"),
                 family_registry_sha256=sha(admission/"source-family-registry.tsv"),
                 scope="accepted exact native pixels and confirmed D4 aliases; bounded source screen, not perfect crop recall")
    out.write_text(json.dumps(guard, indent=1)+"\n")
    return guard, memberships


def cmd_dataset_guard(a):
    guard, _ = dataset_family_guard(Path(a.admission), Path(a.root), Path(a.out))
    log(f"inherited reserved rows: {guard['inherited_leaks']}")
    return 0


def dataset_targets(path, wanted, columns, ledger, collection):
    """Parse selected TRAIN lines only. Persist the preread ledger first."""
    reads = json.loads(ledger.read_text()) if ledger.exists() else {"role": "TRAIN", "reads": []}
    reads["reads"].append(dict(collection=collection, table=str(path), rows=sorted(wanted),
                               columns=columns, row_count=len(wanted), state="requested"))
    ledger.write_text(json.dumps(reads, indent=1)+"\n")
    result = {}
    with path.open() as stream:
        header = next(csv.reader([next(stream)], delimiter="\t"))
        positions = [header.index(c) for c in columns]
        for i, line in enumerate(stream):
            if i in wanted:
                row = next(csv.reader([line], delimiter="\t"))
                if len(row) != len(header):
                    raise ValueError("normalized label table requires one complete TSV record per line")
                result[i] = dict(zip(columns, (row[j] for j in positions)))
    if set(result) != wanted:
        raise ValueError("TRAIN label row selection incomplete")
    reads["reads"][-1]["state"] = "read"
    ledger.write_text(json.dumps(reads, indent=1)+"\n")
    return result


def dataset_observations(collection, members, mapping, labels):
    """Quality-oriented scalar rows and strictly pair-local hard preferences."""
    by_row = collections.defaultdict(dict)
    for r in mapping:
        if r["role"] == "TRAIN" and r["skip_reason"] is None:
            key = (r["input_table"], r["membership_row_index"])
            if r["side"] in by_row[key]:
                raise ValueError("duplicate membership-side join")
            by_row[key][r["side"]] = r["pair_key"]
    observations, dropped = [], collections.Counter()
    for r in members:
        key = (r["input_table"], r["row_index"])
        sides, lab = by_row[key], labels[key]
        if lab["ref"] != r["reference_path"]:
            raise ValueError("label reference does not match frozen membership")
        identity = f"{collection}:{r['input_table']}:{r['row_index']}"
        if collection in DATASET_PAIR:
            if set(sides) != {"a", "b"} or (lab["a"], lab["b"]) != (r["a_path"], r["b_path"]):
                raise ValueError("preference side binding differs")
            p = float(lab["p_a"])
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("invalid native preference probability")
            if p == 0.5:
                dropped["exact_probability_tie"] += 1
                continue
            for side in ("a", "b"):
                observations.append(dict(pair_key=sides[side], ref_basename=identity,
                    human_score=float((p > 0.5) == (side == "a")), observation=identity+":"+side,
                    family=str(r["family_id"])))
        else:
            if set(sides) != {"item"} or lab["item"] != r["item_path"]:
                raise ValueError("MOS item binding differs")
            direction = lab["higher_is_better"].lower()
            if direction not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError("unknown native MOS direction")
            score = float(lab["score"])
            if not np.isfinite(score):
                raise ValueError("nonfinite native MOS")
            ref = f"{collection}:{r['reference_path']}"
            observations.append(dict(pair_key=sides["item"], ref_basename=ref,
                human_score=score if direction in ("true", "1", "yes") else -score,
                observation=identity, family=str(r["family_id"])))
    counts = collections.Counter(r["ref_basename"] for r in observations)
    usable = [r for r in observations if counts[r["ref_basename"]] >= 2]
    dropped["singleton_rows"] = len(observations)-len(usable)
    return usable, dict(dropped)


def cmd_datasets(a):
    """Extend the existing derived root builder; never mutate frozen inputs."""
    source, out, admission, features = map(Path, (a.root, a.out, a.admission, a.features))
    if out.exists():
        raise ValueError("dataset root destination must be fresh")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Metadata/identity audit comes before the first new label value.
    guard_path = out.parent / "FAMILY_GUARD.json"
    guard, memberships = dataset_family_guard(admission, source, guard_path)
    shutil.copytree(source, out)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
    from v2_teacher import row_keys_sha, selection_sha
    from v2_common import admission_input_roots
    real = out / "wide/main/real"
    training_width = json.loads((real/"receipt.json").read_text())["width"]
    masked = {}
    # Excluded TEST memberships still reserve their known families. Preserve
    # every original row and receipt in the immutable source root.
    for key_name, rec in guard["inherited"].items():
        if not rec["reserved_rows"]:
            continue
        path = real / key_name.replace(".keys.parquet", ".parquet")
        kp = path.with_suffix(".keys.parquet")
        keys = pq.read_table(kp)
        keep = [i for i,r in enumerate(keys["ref_basename"].to_pylist()) if r not in rec["references"]]
        table = pq.read_table(path).take(pa.array(keep))
        keys = keys.take(pa.array(keep))
        write(table, path)
        pq.write_table(keys, kp, compression="zstd")
        dp = Path(str(path)+".manifest.json")
        decl = json.loads(dp.read_text())
        decl.update(table_sha256=sha(path), keys_sha256=sha(kp), row_keys_sha256=row_keys_sha(keys),
                    rows_kept=len(keep), row_selection_sha256=selection_sha(keep),
                    dataset_family_mask=dict(parent_table_sha256=decl["table_sha256"],
                                             removed_rows=rec["reserved_rows"], references=rec["references"]))
        dp.write_text(json.dumps(decl, indent=1)+"\n")
        masked[path.name] = dict(removed_rows=rec["reserved_rows"], kept_rows=len(keep))
    if guard["coverage_reserved_rows"]:
        raise ValueError("coverage source reservation needs a bound ordinal mask")
    remaining = sum(sum(ref in rec["references"] for ref in
        pq.read_table(real/name, columns=["ref_basename"])["ref_basename"].to_pylist())
        for name,rec in guard["inherited"].items())
    if remaining:
        raise ValueError("inherited family mask incomplete")
    guard["masked_inherited_tables"] = masked
    guard["inherited_leaks"] = remaining
    guard_path.write_text(json.dumps(guard, indent=1)+"\n")
    legdir = out / "datasets"
    legdir.mkdir()
    shutil.copyfile(guard_path, legdir/"FAMILY_GUARD.json")
    guard_path = legdir/"FAMILY_GUARD.json"
    ledger = out.parent/"LABEL_READS.json"
    selected = [int(x) for x in Path(a.columns).read_text().strip().split(",")]
    protected = set(guard["protected_families"])
    legs = []
    for c in (*DATASET_MOS, *DATASET_PAIR):
        members = [r for r in memberships[c] if r["membership_state"] == "FROZEN" and r["role"] == "TRAIN"]
        if any(str(r["family_id"]) in protected for r in members):
            raise ValueError(f"{c}: TRAIN family reserved by another role")
        mapping = pq.read_table(features/"inputs"/c/"membership-to-pairs.parquet").to_pylist()
        preparation = json.loads((features/"inputs"/c/"PREPARATION.json").read_text())
        if (preparation["membership_sha256"] != guard["membership_sha256"][c]
                or preparation["mapping_sha256"] != sha(features/"inputs"/c/"membership-to-pairs.parquet")):
            raise ValueError("feature-to-membership mapping changed")
        labels = {}
        columns = ["ref", "a", "b", "p_a"] if c in DATASET_PAIR else ["ref", "item", "score", "higher_is_better"]
        for table in sorted({r["input_table"] for r in members}):
            wanted = {r["row_index"] for r in members if r["input_table"] == table}
            labels.update({(table,i):v for i,v in dataset_targets(Path(a.library)/c/table, wanted, columns, ledger, c).items()})
        obs, dropped = dataset_observations(c, members, mapping, labels)
        bank = features/"bank"/c.replace("-", "_")
        manifest = json.loads((bank/"_MANIFEST.json").read_text())
        if (manifest["feature_set_id"] != "basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af"
                or manifest["build_commit"] != "b77883be18db5933a9db4b9e2670a2ba88a9729b"
                or sha(bank/"features.parquet") != manifest["features_parquet_sha256"]):
            raise ValueError("dataset feature producer/hash differs")
        tab = pq.read_table(bank/"features.parquet")
        index = {k:i for i,k in enumerate(tab["pair_key"].to_pylist())}
        if len(index) != tab.num_rows:
            raise ValueError("duplicate dataset bank key")
        tab = tab.take(pa.array([index[r["pair_key"]] for r in obs]))
        # Keep the canonical gapped width; absent slots stay NaN, never zero-filled.
        tab = tab.select([f"f{i}" for i in range(1825)]).cast(pa.schema(
            [(f"f{i}", pa.float32()) for i in range(1825)]))
        # The inherited v2 trainer transport is width1853 even though the
        # registered measurement surface ends at1824. Match that transport;
        # these unmeasured tail columns remain NaN and are never selected.
        for i in range(1825, training_width):
            tab = tab.append_column(f"f{i}",pa.array(np.full(len(obs),np.nan,dtype=np.float32)))
        tab = tab.append_column("ref_basename",pa.array([r["ref_basename"] for r in obs]))
        tab = tab.append_column("human_score",pa.array([r["human_score"] for r in obs]))
        path = legdir/f"{c}.parquet"
        write(tab, path)
        counts = dict(sorted(collections.Counter(r["ref_basename"] for r in obs).items()))
        keys = [dict(row=i, observation=r["observation"],source=r["ref_basename"],reference=r["ref_basename"],
                     family=r["family"],role="fit",tier="T2") for i,r in enumerate(obs)]
        kp = legdir/f"{c}.keys.json"
        kp.write_text(json.dumps(keys,separators=(",",":"))+"\n")
        decl = dict(schema="subjective-training-leg-v2",loss="hard-pair-preference" if c in DATASET_PAIR else "within-source-rank",
            nominal_weight=0.5,target_orientation="quality",target_transform="orient-native-quality",target_column="human_score",
            selection="fixed-final-epoch",formula_revision=7,tier="v3",feature_ids=selected,
            producer_commit=manifest["build_commit"],feature_set_id=manifest["feature_set_id"],
            native_role="TRAIN",rows=len(obs),table_sha256=sha(path),admission_root=str(out),
            training_width=training_width,
            membership_sha256=guard["membership_sha256"][c],feature_manifest_sha256=sha(bank/"_MANIFEST.json"))
        dp = Path(str(path)+".manifest.json")
        dp.write_text(json.dumps(decl,indent=1)+"\n")
        weight = 0.5/(sum(1-1/n for n in counts.values())/len(counts))
        legs.append(dict(name="subjective_rank_"+c.replace("-","_"),path=str(path),train_weight=weight,val_weight=0.,
            mode="withinref,rank",subjective=True,table_sha256=sha(path),manifest_sha256=sha(dp),
            keys_path=str(kp),keys_sha256=sha(kp),rows=len(obs),source_counts=counts,
            collection=c,membership_rows=len(members),dropped=dropped,groups=len(counts),
            pairs=sum(n*(n-1)//2 for n in counts.values())))
        log(f"{c}: {len(members)} memberships, {len(obs)} rows, {len(counts)} groups; dropped {dropped}")
    receipt_path = real/"receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for leg in receipt["legs"].values():
        for rec in leg.values():
            if isinstance(rec,dict) and "rel" in rec:
                p = out/rec["rel"]
                rec.update(sha256=sha(p),manifest_sha256=sha(str(p)+".manifest.json"),
                           keys_sha256=sha(p.with_suffix(".keys.parquet")))
                if p.name in masked:
                    rec["rows"] = masked[p.name]["kept_rows"]
    view = dict(receipt["admission_view"])
    view["immutable_input_roots"] = [str(p) for p in
        (*admission_input_roots(source),source,admission,features,out)]
    receipt["admission_view"] = view
    receipt_path.write_text(json.dumps(receipt,indent=1)+"\n")
    packet = dict(schema="rev7-dataset-legs-v1",groups=legs,feature_ids=selected,
                  producer_commit="b77883be18db5933a9db4b9e2670a2ba88a9729b",
                  family_guard_path=str(guard_path),family_guard_sha256=sha(guard_path),label_reads_path=str(ledger))
    pp = legdir/"receipt.json"
    pp.write_text(json.dumps(packet,indent=1)+"\n")
    frozen = json.loads((out/"wide/frozen.json").read_text())
    frozen["admission_view"] = view
    frozen["wide_receipts"] = {"main/real":sha(receipt_path)}
    frozen["dataset_legs_sha256"] = sha(pp)
    frozen["auxiliary_files"] = {rel:sha(out/rel) for rel in frozen["auxiliary_files"]}
    frozen["auxiliary_files"].update({str(p.relative_to(out)):sha(p) for p in legdir.iterdir()})
    (out/"wide/frozen.json").write_text(json.dumps(frozen,indent=1)+"\n")
    log(f"dataset root frozen: {out}")
    return 0


def cmd_dataset_report(a):
    root, output = Path(a.root), Path(a.out)
    packet = json.loads((root/"datasets/receipt.json").read_text())
    ledger_path = Path(packet["label_reads_path"])
    ledger = json.loads(ledger_path.read_text())
    result = {}
    for r in packet["groups"]:
        # Record this read of already-derived TRAIN targets separately from
        # the original source-table reads. No held-out target is accessed.
        ledger["reads"].append(dict(collection=r["collection"],table=r["path"],
            rows=list(range(r["rows"])),columns=["ref_basename","human_score"],
            row_count=r["rows"],state="requested",kind="derived-TRAIN-pair-count"))
        ledger_path.write_text(json.dumps(ledger,indent=1)+"\n")
        t = pq.read_table(r["path"],columns=["ref_basename","human_score"])
        grouped = collections.defaultdict(collections.Counter)
        for ref,score in zip(t["ref_basename"].to_pylist(),t["human_score"].to_pylist()):
            grouped[ref][score] += 1
        tied = sum(sum(n*(n-1)//2 for n in counts.values()) for counts in grouped.values())
        keys = json.loads(Path(r["keys_path"]).read_text())
        result[r["collection"]] = dict(membership_rows=r["membership_rows"],rows=r["rows"],groups=len(grouped),
            families=len({k["family"] for k in keys}),candidate_pairs=r["pairs"],
            informative_pairs=r["pairs"]-tied,exact_mos_tied_pairs=tied,dropped=r["dropped"],
            nominal_weight=0.5,train_weight=r["train_weight"])
        ledger["reads"][-1]["state"] = "read"
        ledger_path.write_text(json.dumps(ledger,indent=1)+"\n")
    output.write_text(json.dumps(result,indent=1)+"\n")
    print(json.dumps(result,indent=1))
    return 0


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def require_arm(manifests: list[Path], arm: dict) -> None:
    """Every research manifest of the extraction must say the arm ran (and its scale map must be `scale_of`)."""
    if not manifests:
        raise ValueError("no research manifests found")
    key, value = arm["manifest"]
    for m in manifests:
        man = json.loads(m.read_text())
        if str(man.get(key)) != str(value):
            raise ValueError(f"{m}: not extracted under {key}={value}")
        scales = {f["id"]: f["scale"] for f in man["features"]}
        if any(scales[i] != scale_of(i) for i in MEASURED):
            raise ValueError(f"{m}: feature scales disagree with scale_of")


def cmd_pool(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    tsv = out / "pairs.tsv"
    shutil.copyfile(POOL_SOURCE / "extract/pairs.tsv", tsv)
    csvp = out / "features.csv"
    arm = ARMS[a.arm]
    env = dict(os.environ, ZENSIM_FORMULA_REV="5", ZENSIM_ROOT_FORM="sqrt", RAYON_NUM_THREADS=str(a.threads))
    env.update(arm["env"])
    cmd = [a.bin, "--corpus", "pairs-tsv", "--path", str(tsv), "--out", str(csvp), "--restore-cuts", "basic,peaks,v2",
           "--input-contract", "legacy-rgb8", "--era-label", arm["era"], "--force-tier", "native"]
    t0 = time.time()
    with (out / "extract.log").open("w") as lg:
        subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT, check=True)
    require_arm([Path(f"{csvp}.research_manifest.json")], arm)
    names = ["row_id"] + [f"f{i}" for i in MEASURED]
    tab = pacsv.read_csv(csvp, convert_options=pacsv.ConvertOptions(
        column_types={"row_id": pa.int64(), **{f"f{i}": pa.float64() for i in MEASURED}}, include_columns=names))
    order = np.argsort(tab.column("row_id").to_numpy(), kind="stable")
    tab = tab.take(pa.array(order))
    if tab.column("row_id").to_pylist() != list(range(tab.num_rows)):
        raise ValueError("pool row_id join failed")
    pq.write_table(tab, out / "features.parquet", compression="zstd")
    meta = {"pairs_sha256": sha(tsv), "binary_sha256": sha(a.bin), "rows": tab.num_rows, "era": arm["era"],
            "extract_wall_s": round(time.time() - t0, 1), "features_sha256": sha(out / "features.parquet"),
            "research_manifest_sha256": sha(f"{csvp}.research_manifest.json")}
    csvp.unlink()
    (out / "_MANIFEST.json").write_text(json.dumps(meta, indent=1) + "\n")
    log(f"pool: {tab.num_rows} rows in {meta['extract_wall_s']}s")
    return 0


def load_bank(bank: Path, name: str, arm: dict) -> tuple[dict, np.ndarray]:
    require_arm(sorted((bank / "_work" / name).glob("chunk_*.csv.research_manifest.json")), arm)
    cols = ["pair_key"] + [f"f{i}" for i in MEASURED]
    tab = pq.read_table(bank / name / "features.parquet", columns=cols)
    index = {k: i for i, k in enumerate(tab.column("pair_key").to_pylist())}
    if len(index) != tab.num_rows:
        raise ValueError(f"{name}: duplicate pair_key in bank")
    X = np.stack([tab.column(f"f{i}").to_numpy() for i in MEASURED], axis=1).astype(np.float32)
    return index, X


def close_check(table: pa.Table, new: np.ndarray, what: str, replaced: list[int]) -> None:
    """A re-extraction at another formula revision (rev7) moves values: by rounding almost everywhere, and by more where
    revision 7's cancellation-free forms correct the old forms' noise on near-degenerate content (flat channels: v1
    texture and variance loss, v2 HF magnitude loss, edge-width change, blockiness at the noise floor). A join or
    extractor error instead moves most of a row's columns. Refuse a nonfinite value where the stored one is finite,
    or more than 0.1% of rows with over a quarter of their columns off by more than 1e-2 relative (+1e-6); log the
    relative-change distribution and the columns that moved most. (A first version refused more than 0.1% of cells;
    on safesyn_dev 0.66% of cells moved, at most 49 of a row's 576 columns, against a mean of 546 for rows joined to
    the wrong pair: benchmarks/byconstruction_2026-10-22.md, "Revision 7 with raw ringing".)"""
    pos = {i: j for j, i in enumerate(MEASURED)}
    old = np.stack([table.column(f"f{i}").to_numpy() for i in replaced], axis=1).astype(np.float64)
    cur = new[:, [pos[i] for i in replaced]].astype(np.float64)
    fin = np.isfinite(old)
    if (fin & ~np.isfinite(cur)).any():
        raise ValueError(f"{what}: a re-extracted value is nonfinite where the stored one is finite")
    both = fin & np.isfinite(cur)
    d = np.where(both, np.abs(cur - old), 0.0)
    far = both & (d > 1e-2 * np.abs(old) + 1e-6)
    rel = d[both] / np.maximum(np.abs(old[both]), 1e-12)
    per_row = far.sum(axis=1)
    bad_rows = int((per_row > len(replaced) / 4).sum())
    top = sorted(((int(far[:, j].sum()), replaced[j]) for j in range(len(replaced))), reverse=True)[:6]
    log(f"{what}: relative change median {np.median(rel):.2e} p99 {np.quantile(rel, 0.99):.2e}; "
        f"{int(far.sum())} of {int(both.sum())} cells beyond 1e-2 relative + 1e-6, at most {int(per_row.max())} of "
        f"{len(replaced)} in a row; most: " + ", ".join(f"f{i} {c}" for c, i in top if c))
    if bad_rows > 1e-3 * len(per_row):
        raise ValueError(f"{what}: {bad_rows} rows moved in over a quarter of their columns; the join is wrong")


def patch(table: pa.Table, new: np.ndarray, what: str, replaced: list[int], close: bool = False) -> pa.Table:
    """Check every measured column outside `replaced` bit for bit (and, with `close`, every replaced value against
    `close_check`), then swap in the `replaced` columns."""
    if close:
        close_check(table, new, what, replaced)
    pos = {i: j for j, i in enumerate(MEASURED)}
    other = [i for i in MEASURED if i not in replaced]
    for i in other:
        old = table.column(f"f{i}").to_numpy()
        if not np.array_equal(old.view(np.uint32), new[:, pos[i]].view(np.uint32)):
            bad = int((old.view(np.uint32) != new[:, pos[i]].view(np.uint32)).sum())
            raise ValueError(f"{what}: f{i} differs on {bad} rows; the join or the extractor is wrong")
    moved = 0
    for i in replaced:
        j = table.column_names.index(f"f{i}")
        col = np.ascontiguousarray(new[:, pos[i]])
        moved += int((table.column(j).to_numpy().view(np.uint32) != col.view(np.uint32)).sum())
        table = table.set_column(j, f"f{i}", pa.array(col))
    log(f"{what}: {table.num_rows} rows, {len(other)} columns identical, {moved} of {table.num_rows * len(replaced)} "
        f"replaced cells changed")
    return table


def write(table: pa.Table, path: Path) -> None:
    names = [c for c in table.column_names if c.startswith("f") and c[1:].isdigit()]
    pq.write_table(table, path, compression="zstd", use_dictionary=["ref_basename"],
                   use_byte_stream_split=["human_score", *names])


def cmd_build(a) -> int:
    arm = ARMS[a.arm]
    era, replaced = arm["era"], arm["replaced"]
    bank, pool, out = Path(a.bank), Path(a.pool), Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists")
    real = out / "wide/main/real"
    real.mkdir(parents=True)
    src_real = SOURCE / "wide/main/real"
    banks, bank_manifests = {}, {}
    fsids = set()
    key, value = arm["manifest"]
    record = {"source_root": str(SOURCE), "source_frozen_sha256": sha(SOURCE / "wide/frozen.json"), key: value,
              "replaced_ids": replaced, "bank": str(bank), "pool": str(pool), "tables": {}}
    for t in sorted(src_real.glob("*.parquet")):
        if t.name.endswith(".keys.parquet") or t.stem in HDR:
            continue
        keys = pq.read_table(src_real / f"{t.stem}.keys.parquet", columns=["pair_key", "member_set"])
        rows = []
        new = np.empty((keys.num_rows, len(MEASURED)), np.float32)
        members = keys.column("member_set").to_pylist()
        pks = keys.column("pair_key").to_pylist()
        for m in sorted(set(members)):
            if m not in banks:
                banks[m] = load_bank(bank, m, arm)
                man = json.loads((bank / m / "_MANIFEST.json").read_text())
                fsids.add(man["feature_set_id"])
                bank_manifests[m] = {"manifest_sha256": sha(bank / m / "_MANIFEST.json"),
                                     "features_sha256": man["features_parquet_sha256"], "keys_sha256": man["keys_sha256"],
                                     "rows": man["rows"], "feature_set_id": man["feature_set_id"],
                                     "binary_sha256": man["binary_sha256"], "build_commit": man["build_commit"]}
            index, X = banks[m]
            sel = [r for r, mm in enumerate(members) if mm == m]
            new[sel] = X[[index[pks[r]] for r in sel]]
        table = patch(pq.read_table(t), new, t.stem, replaced, arm.get("close", False))
        write(table, real / t.name)
        shutil.copyfile(src_real / f"{t.stem}.keys.parquet", real / f"{t.stem}.keys.parquet")
        record["tables"][t.name] = {"source_sha256": sha(t), "sha256": sha(real / t.name)}
    if len(fsids) != 1:
        raise ValueError(f"bank feature_set_id differs between sets: {fsids}")
    fsid = fsids.pop()
    if f"/{era}#" not in fsid:
        raise ValueError(f"bank was not extracted with era {era}: {fsid}")

    # Coverage pool: the stored pool is the selection minus identical and single-rung rows, in order. Match it as a
    # subsequence on its ladder fields, then let the bit-for-bit check confirm the alignment.
    import pandas as pd
    require_arm([pool / "features.csv.research_manifest.json"] if (pool / "features.csv.research_manifest.json").exists()
                else [], arm)
    sel = pd.read_parquet(POOL_SOURCE / "selection.parquet")
    pkeys = pd.read_parquet(SOURCE / "e15/coverage_pool.keys.parquet")
    fields = ["source_filename", "type", "severity_level", "severity", "sign"]
    want = list(pkeys[fields].astype(str).itertuples(index=False, name=None))
    have = list(sel[fields].astype(str).itertuples(index=False, name=None))
    take, j = [], 0
    for i, row in enumerate(have):
        if j < len(want) and row == want[j]:
            take.append(i)
            j += 1
    if j != len(want):
        raise ValueError(f"coverage pool: matched {j} of {len(want)} rows")
    ptab = pq.read_table(pool / "features.parquet")
    X = np.stack([ptab.column(f"f{i}").to_numpy() for i in MEASURED], axis=1)[take].astype(np.float32)
    table = patch(pq.read_table(SOURCE / "e15/coverage_pool.parquet"), X, "coverage_pool", replaced,
                  arm.get("close", False))
    (out / "e15").mkdir()
    write(table, out / "e15/coverage_pool.parquet")
    shutil.copyfile(SOURCE / "e15/coverage_pool.keys.parquet", out / "e15/coverage_pool.keys.parquet")
    record["tables"]["e15/coverage_pool.parquet"] = {"source_sha256": sha(SOURCE / "e15/coverage_pool.parquet"),
                                                     "sha256": sha(out / "e15/coverage_pool.parquet")}
    record["pool_manifest_sha256"] = sha(pool / "_MANIFEST.json")

    # Declarations: same rows, roles and decisions; new table hashes, feature identity and the derivation.
    note = arm["note"]
    bank_root = str(bank.resolve())
    new_roots = [str(SOURCE.resolve()), bank_root, str(pool.resolve()), str(POOL_SOURCE.resolve())]

    def declare(src_manifest: Path, table: Path) -> None:
        m = json.loads(src_manifest.read_text())
        m["byc_derived_from"] = {"table_sha256": m.get("table_sha256"), "manifest_sha256": sha(src_manifest)}
        m["table_sha256"] = sha(table)
        if "source_table_sha256" in m:
            m["source_table_sha256"] = m["table_sha256"]
        m["feature_set_id"] = fsid
        m["source_bank_feature_set_id"] = fsid
        m["feature_values_changed"] = True
        m[key] = value
        if "revision" in arm:
            # The tables now hold another revision's values: declare it, and the extractor that produced them.
            binaries = {b["binary_sha256"] for b in bank_manifests.values()}
            commits = {b["build_commit"] for b in bank_manifests.values()}
            if len(binaries) != 1 or len(commits) != 1:
                raise ValueError(f"bank sets were extracted by different binaries: {binaries} {commits}")
            m["formula_revision"] = arm["revision"]
            m["decoder_era"] = f"legacy-rgb8/extract_features_372col@sha256:{next(iter(binaries))}"
            m["extractor_build_commit"] = next(iter(commits))
        m["composite"] = m.get("composite", "") + note
        m["immutable_input_roots"] = [*m.get("immutable_input_roots", []), *new_roots]
        Path(f"{table}.manifest.json").write_text(json.dumps(m, indent=1) + "\n")

    for name in record["tables"]:
        if name.startswith("e15/"):
            declare(SOURCE / f"{name}.manifest.json", out / name)
        else:
            declare(src_real / f"{name}.manifest.json", real / name)

    for rel in ("human_role_decision.json", "source_bindings/frozen.json", "source_bindings/receipt.json",
                "wide/keep_lists.json"):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / rel, out / rel)

    receipt = json.loads((src_real / "receipt.json").read_text())
    receipt["legs"].pop("hdr_consensus", None)
    for leg in receipt["legs"].values():
        for part in leg.values():
            if isinstance(part, dict) and "rel" in part:
                p = out / part["rel"]
                part["sha256"] = sha(p)
                part["manifest_sha256"] = sha(Path(f"{p}.manifest.json"))
    receipt["byc_source_bank"] = receipt["bank"]
    receipt["bank"] = bank_manifests
    receipt["feature_set_id"] = fsid
    receipt["era"] = era
    receipt[key] = value
    if "revision" in arm:
        receipt["formula_revision"] = arm["revision"]
    view = dict(receipt["admission_view"])
    view["bank_root"] = bank_root
    view["immutable_input_roots"] = [*view["immutable_input_roots"], *new_roots]
    receipt["admission_view"] = view
    (real / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")

    frozen = json.loads((SOURCE / "wide/frozen.json").read_text())
    frozen["admission_view"] = view
    frozen["wide_receipts"] = {"main/real": sha(real / "receipt.json")}
    aux = {}
    for rel in frozen["auxiliary_files"]:
        if any(h in rel for h in HDR):
            continue
        aux[rel] = sha(out / rel)
    frozen["auxiliary_files"] = aux
    frozen["keep_lists_sha256"] = sha(out / "wide/keep_lists.json")
    frozen[arm["record"]] = record
    (out / "wide/frozen.json").write_text(json.dumps(frozen, indent=1) + "\n")
    log(f"root written: {out} (fsid {fsid})")
    return 0


GLOBAFF_IDS = list(range(1867, 1891))
GLOBAFF_WIDTH = GLOBAFF_IDS[-1] + 1
# `byc_fsid basic+peaks+v2 1891 rev5_localwin_ringhpm_globaff 0-227,372-719,1867-1890` (hash from zensim::feature_set_id).
GLOBAFF_FSID = "basic+peaks+v2@w1891/rev5_localwin_ringhpm_globaff#d56ad477"


def cmd_globaff_pairs(a) -> int:
    """Pairs TSVs for byc_globaff: each bank set's rows in bank order, and the coverage-pool selection."""
    bank, out = Path(a.bank), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    for d in sorted(p for p in (bank / "_work").iterdir() if p.is_dir()):
        chunks = sorted(d.glob("pairs_*.tsv"))
        lines = [chunks[0].read_text().splitlines()[0]]
        for c in chunks:
            lines += c.read_text().splitlines()[1:]
        (out / f"{d.name}.tsv").write_text("\n".join(lines) + "\n")
        log(f"{d.name}: {len(lines) - 1} pairs")
    shutil.copyfile(Path(a.pool) / "pairs.tsv", out / "pool.tsv")
    return 0


# Research columns a widened root appends (`widen_root`). The declarations list them under `byc_globaff_ids`, the key
# strict admission (Python and the Rust trainer) reads for research ids on a `byc_derived_from` root, whatever family.
WIDEN = {
    "globaff": {"ids": GLOBAFF_IDS, "prefix": "g", "fsid": GLOBAFF_FSID, "finite": slice(0, None, 2),
                "note": "; byc_globaff global affine-change descriptor at f1867-f1890 (share, explained per scale x channel)"},
    # `byc_fsid basic+peaks+v2 2337 rev5_localwin_orient 0-227,372-719,2301-2336` (hash from zensim::feature_set_id).
    "orient": {"ids": list(range(2301, 2337)), "prefix": "o",
               "fsid": "basic+peaks+v2@w2337/rev5_localwin_orient#24a93918", "finite": slice(None),
               "note": "; byc_orient Haar H/V/D band-difference RMS per scale x channel at f2301-f2336"},
}


def read_sidecar(path: Path, prefix: str = "g", n: int = len(GLOBAFF_IDS)) -> np.ndarray:
    tab = pacsv.read_csv(path)
    rid = tab.column("row_id").to_numpy()
    if not np.array_equal(np.sort(rid), np.arange(len(rid))):
        raise ValueError(f"{path}: row ids are not 0..n-1")
    X = np.stack([tab.column(f"{prefix}{i}").to_numpy() for i in range(n)], axis=1)
    out = np.empty_like(X)
    out[rid] = X
    return out.astype(np.float32)


def widen(table: pa.Table, values: np.ndarray, what: str, spec: dict | None = None) -> pa.Table:
    """Append NaN from the stored width (1853) up to the family's first id, then its columns."""
    spec = spec or WIDEN["globaff"]
    ids = spec["ids"]
    width = sum(1 for c in table.column_names if c.startswith("f") and c[1:].isdigit())
    if width != GLOBAFF_IDS[0] - 14 or not np.isfinite(values[:, spec["finite"]]).all():
        raise ValueError(f"{what}: width {width} or nonfinite values")
    nan = pa.array(np.full(table.num_rows, np.nan, np.float32))
    for i in range(width, ids[0]):
        table = table.append_column(f"f{i}", nan)
    for j, i in enumerate(ids):
        table = table.append_column(f"f{i}", pa.array(np.ascontiguousarray(values[:, j])))
    return table


def cmd_globaff(a) -> int:
    return widen_root(a, "globaff")


def cmd_orient(a) -> int:
    return widen_root(a, "orient")


def widen_root(a, name: str) -> int:
    spec = WIDEN[name]
    ids, prefix, fsid = spec["ids"], spec["prefix"], spec["fsid"]
    root, bank, side, out = Path(a.root), Path(a.bank), Path(a.sidecars), Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists")
    shutil.copytree(root, out)
    real = out / "wide/main/real"
    cache = {}
    record = {"source_root": str(root), "source_frozen_sha256": sha(root / "wide/frozen.json"), "ids": ids,
              "sidecars": {}, "tables": {}}
    # A stored root still carries the HDR legs; like `build`, the widened root leaves them out (no sidecar covers them).
    for p in sorted(real.iterdir()):
        if p.name.split(".")[0] in HDR:
            p.unlink()
    for t in sorted(real.glob("*.parquet")):
        if t.name.endswith(".keys.parquet"):
            continue
        keys = pq.read_table(real / f"{t.stem}.keys.parquet", columns=["pair_key", "member_set"])
        members, pks = keys.column("member_set").to_pylist(), keys.column("pair_key").to_pylist()
        vals = np.empty((keys.num_rows, len(ids)), np.float32)
        for m in sorted(set(members)):
            if m not in cache:
                order = pq.read_table(bank / m / "features.parquet", columns=["pair_key"]).column("pair_key").to_pylist()
                cache[m] = ({k: i for i, k in enumerate(order)}, read_sidecar(side / f"{m}.csv", prefix, len(ids)))
                record["sidecars"][m] = sha(side / f"{m}.csv")
            index, X = cache[m]
            sel = [r for r, mm in enumerate(members) if mm == m]
            vals[sel] = X[[index[pks[r]] for r in sel]]
        write(widen(pq.read_table(t), vals, t.stem, spec), t)
        record["tables"][t.name] = sha(t)
    # Coverage pool: the same subsequence alignment `build` checked bit for bit.
    import pandas as pd
    sel = pd.read_parquet(POOL_SOURCE / "selection.parquet")
    pkeys = pd.read_parquet(out / "e15/coverage_pool.keys.parquet")
    fields = ["source_filename", "type", "severity_level", "severity", "sign"]
    want = list(pkeys[fields].astype(str).itertuples(index=False, name=None))
    take, j = [], 0
    for i, row in enumerate(sel[fields].astype(str).itertuples(index=False, name=None)):
        if j < len(want) and row == want[j]:
            take.append(i)
            j += 1
    if j != len(want):
        raise ValueError("coverage pool alignment failed")
    pool = out / "e15/coverage_pool.parquet"
    write(widen(pq.read_table(pool), read_sidecar(side / "pool.csv", prefix, len(ids))[take], "coverage_pool", spec), pool)
    record["sidecars"]["pool"] = sha(side / "pool.csv")
    record["tables"]["e15/coverage_pool.parquet"] = sha(pool)

    note = spec["note"]
    for table_name in list(record["tables"]):
        path = out / table_name if table_name.startswith("e15/") else real / table_name
        mp = Path(f"{path}.manifest.json")
        m = json.loads(mp.read_text())
        derived = {"table_sha256": m.get("table_sha256"), "manifest_sha256": sha(mp)}
        m[f"byc_{name}_derived_from"] = derived
        m.setdefault("byc_derived_from", derived)  # a stored (not yet derived) root becomes a derived one here
        m["table_sha256"] = sha(path)
        if "source_table_sha256" in m:
            m["source_table_sha256"] = m["table_sha256"]
        m["byc_globaff_ids"] = ids
        m["feature_set_id"] = fsid
        m["source_bank_feature_set_id"] = fsid
        m["composite"] = m.get("composite", "") + note
        m["immutable_input_roots"] = [*m.get("immutable_input_roots", []), str(root.resolve()), str(side.resolve())]
        mp.write_text(json.dumps(m, indent=1) + "\n")
    receipt = json.loads((real / "receipt.json").read_text())
    receipt["legs"].pop("hdr_consensus", None)
    for leg in receipt["legs"].values():
        for part in leg.values():
            if isinstance(part, dict) and "rel" in part:
                q = out / part["rel"]
                part["sha256"] = sha(q)
                part["manifest_sha256"] = sha(Path(f"{q}.manifest.json"))
    receipt["width"] = ids[-1] + 1
    receipt["feature_set_id"] = fsid
    receipt[f"byc_{name}"] = record
    view = dict(receipt["admission_view"])
    view["immutable_input_roots"] = [*view["immutable_input_roots"], str(root.resolve()), str(side.resolve())]
    receipt["admission_view"] = view
    (real / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    frozen = json.loads((out / "wide/frozen.json").read_text())
    frozen["admission_view"] = view
    frozen["wide_receipts"] = {"main/real": sha(real / "receipt.json")}
    frozen["auxiliary_files"] = {rel: sha(out / rel) for rel in frozen["auxiliary_files"] if not any(h in rel for h in HDR)}
    frozen[f"byc_{name}"] = record
    (out / "wide/frozen.json").write_text(json.dumps(frozen, indent=1) + "\n")
    log(f"{name} root written: {out}")
    return 0


GATED_FIRST = GLOBAFF_WIDTH
# Gated channels (XYB order, cell k = 3 * scale + channel) -> (gated difference count, feature set id, declaration
# registration text). `byc_fsid basic+peaks+v2 <first + count> <era> 0-227,372-719,1867-<first + count - 1>` (hash from
# zensim::feature_set_id).
GATED = {
    "XYB": ((0, 1, 2), 410, "basic+peaks+v2@w2301/rev5_localwin_ringhpm_globaffgate#45e0e84c",
            "exploration only: benchmarks/byconstruction_2026-10-22.md (fx1 plus gated damage at research ids f1891-f2300: "
            "each difference x (1 - byc_globaff share of its scale and channel cell); the descriptor itself is not read)"),
    "Y": ((1,), 164, "basic+peaks+v2@w2055/rev5_localwin_ringhpm_globaffgatey#7b645e91",
          "exploration only: benchmarks/byconstruction_2026-10-22.md (fx1 plus gated luminance damage at research ids "
          "f1891-f2054: each Y-cell difference x (1 - byc_globaff share of its scale's Y cell); the descriptor itself is "
          "not read)"),
}
GATED_WIDTH = GATED_FIRST + GATED["XYB"][1]
GATED_FSID = GATED["XYB"][2]


def gated_plan(decl: Path, channels: str = "XYB") -> tuple[list[int], list[int]]:
    """fx1's differences in a gated channel, in ascending order, and the share id of each one's (scale, channel) cell:
    the declaration pairs each difference with the `pjnd_fragility` slot of its own cell, `372 + 29k + 21` (E33
    section 5.1, k = 3 * scale + channel), and the descriptor's share of cell k sits at `1867 + 2k`."""
    gated_channels, count, _, _ = GATED[channels]
    pairs = json.loads(decl.read_text())["products"]
    if [a for a, _ in pairs] != sorted(a for a, _ in pairs) or len(pairs) != 410:
        raise ValueError(f"{decl}: expected 410 differences in ascending order")
    diffs, cells = [], []
    for a, f in pairs:
        k, rem = divmod(f - 372 - 21, 29)
        if rem or not 0 <= k < 12:
            raise ValueError(f"{decl}: {f} is not a pjnd_fragility slot")
        if k % 3 in gated_channels:
            diffs.append(a)
            cells.append(GLOBAFF_IDS[0] + 2 * k)
    if len(diffs) != count:
        raise ValueError(f"{decl}: {len(diffs)} differences in channels {channels}, expected {count}")
    return diffs, cells


def gate_table(table: pa.Table, diffs: list[int], shares: list[int], what: str) -> pa.Table:
    width = sum(1 for c in table.column_names if c.startswith("f") and c[1:].isdigit())
    if width != GATED_FIRST:
        raise ValueError(f"{what}: width {width}, expected the descriptor root's {GATED_FIRST}")
    for j, (d, sh) in enumerate(zip(diffs, shares)):
        dv = table.column(f"f{d}").to_numpy(zero_copy_only=False).astype(np.float32)
        u = np.float32(1) - table.column(f"f{sh}").to_numpy(zero_copy_only=False).astype(np.float32)
        g = dv * u
        if not np.isfinite(g).all() or (g < 0).any():
            raise ValueError(f"{what}: f{d} x (1 - f{sh}) is negative or nonfinite")
        table = table.append_column(f"f{GATED_FIRST + j}", pa.array(g))
    return table


def cmd_gate_table(a) -> int:
    """One table; a mixture table's `table.json` beside it is carried over with the gating recorded."""
    src, out = Path(a.table), Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists")
    diffs, shares = gated_plan(Path(a.decl), a.channels)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(gate_table(pq.read_table(src), diffs, shares, src.name), out)
    meta = src.parent / "table.json"
    if src.name == "table.parquet" and out.name == "table.parquet" and meta.is_file():
        m = json.loads(meta.read_text())
        m["gated"] = {"from_table_sha256": sha(src), "ids": [GATED_FIRST, GATED_FIRST + len(diffs) - 1],
                      **({} if a.channels == "XYB" else {"channels": a.channels}), "builder": "ring_root.py gate-table"}
        (out.parent / "table.json").write_text(json.dumps(m, indent=1) + "\n")
    log(f"gated ({a.channels}) {src} -> {out} ({sha(out)[:12]})")
    return 0


def cmd_gated_decl(a) -> int:
    """The declaration a gated root trains with (`zensim-fx1-v1`): fx1's differences and the gated ids direct (the
    gated ids also unpaired), fx1's fragility products; the descriptor itself is not read."""
    out = Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists")
    src = json.loads(Path(a.decl).read_text())
    diffs, _ = gated_plan(Path(a.decl), a.channels)
    ids = list(range(GATED_FIRST, GATED_FIRST + len(diffs)))
    decl = {"schema": "zensim-fx1-v1", "registration": GATED[a.channels][3],
            "direct": [i for i in src["direct"] if i not in GLOBAFF_IDS] + ids, "products": src["products"],
            "unpaired": ids}
    out.write_text(json.dumps(decl, indent=1) + "\n")
    log(f"declaration ({a.channels}) {out} ({sha(out)})")
    return 0


def cmd_gated(a) -> int:
    root, out = Path(a.root), Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists")
    diffs, shares = gated_plan(Path(a.decl), a.channels)
    width, fsid = GATED_FIRST + len(diffs), GATED[a.channels][2]
    shutil.copytree(root, out)
    real = out / "wide/main/real"
    record = {"source_root": str(root), "source_frozen_sha256": sha(root / "wide/frozen.json"),
              "declaration_sha256": sha(Path(a.decl)), "ids": list(range(GATED_FIRST, width)),
              "differences": diffs, "share_ids": shares, "tables": {}}
    if a.channels != "XYB":  # the all-channel root's records stay as they were first written
        record["channels"] = a.channels
    tables = [t for t in sorted(real.glob("*.parquet")) if not t.name.endswith(".keys.parquet")]
    for t in [*tables, out / "e15/coverage_pool.parquet"]:
        write(gate_table(pq.read_table(t), diffs, shares, t.stem), t)
        record["tables"][t.name if t.parent == real else "e15/coverage_pool.parquet"] = sha(t)
        log(f"gated {t.name}")
    note = (f"; gated damage at f{GATED_FIRST}-f{width - 1} (each fx1 difference"
            + ("" if a.channels == "XYB" else f" in channels {a.channels}") + " x (1 - byc_globaff share of its cell))")
    ids = list(range(GLOBAFF_IDS[0], width))
    for name in list(record["tables"]):
        path = out / name if name.startswith("e15/") else real / name
        mp = Path(f"{path}.manifest.json")
        m = json.loads(mp.read_text())
        m["byc_gated_derived_from"] = {"table_sha256": m.get("table_sha256"), "manifest_sha256": sha(mp)}
        m["table_sha256"] = sha(path)
        if "source_table_sha256" in m:
            m["source_table_sha256"] = m["table_sha256"]
        m["byc_globaff_ids"] = ids
        m["feature_set_id"] = fsid
        m["source_bank_feature_set_id"] = fsid
        m["composite"] = m.get("composite", "") + note
        m["immutable_input_roots"] = [*m.get("immutable_input_roots", []), str(root.resolve())]
        mp.write_text(json.dumps(m, indent=1) + "\n")
    receipt = json.loads((real / "receipt.json").read_text())
    for leg in receipt["legs"].values():
        for part in leg.values():
            if isinstance(part, dict) and "rel" in part:
                q = out / part["rel"]
                part["sha256"] = sha(q)
                part["manifest_sha256"] = sha(Path(f"{q}.manifest.json"))
    receipt["width"] = width
    receipt["feature_set_id"] = fsid
    receipt["byc_gated"] = record
    view = dict(receipt["admission_view"])
    view["immutable_input_roots"] = [*view["immutable_input_roots"], str(root.resolve())]
    receipt["admission_view"] = view
    (real / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    frozen = json.loads((out / "wide/frozen.json").read_text())
    frozen["admission_view"] = view
    frozen["wide_receipts"] = {"main/real": sha(real / "receipt.json")}
    frozen["auxiliary_files"] = {rel: sha(out / rel) for rel in frozen["auxiliary_files"]}
    frozen["byc_gated"] = record
    (out / "wide/frozen.json").write_text(json.dumps(frozen, indent=1) + "\n")
    log(f"gated root written: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    dg = sub.add_parser("dataset-guard")
    dg.add_argument("--admission", required=True)
    dg.add_argument("--root", required=True)
    dg.add_argument("--out", required=True)
    dg.set_defaults(fn=cmd_dataset_guard)
    ds = sub.add_parser("datasets")
    for flag in ("root", "out", "admission", "features", "library", "columns"):
        ds.add_argument("--"+flag, required=True)
    ds.set_defaults(fn=cmd_datasets)
    dr = sub.add_parser("dataset-report")
    dr.add_argument("--root", required=True)
    dr.add_argument("--out", required=True)
    dr.set_defaults(fn=cmd_dataset_report)
    p = sub.add_parser("pool")
    p.add_argument("--arm", choices=sorted(ARMS), default="hpm")
    p.add_argument("--bin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=24)
    p.set_defaults(fn=cmd_pool)
    b = sub.add_parser("build")
    b.add_argument("--arm", choices=sorted(ARMS), default="hpm")
    b.add_argument("--bank", required=True)
    b.add_argument("--pool", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)
    g = sub.add_parser("globaff-pairs")
    g.add_argument("--bank", required=True)
    g.add_argument("--pool", required=True)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_globaff_pairs)
    w = sub.add_parser("globaff")
    w.add_argument("--root", required=True)
    w.add_argument("--bank", required=True)
    w.add_argument("--sidecars", required=True)
    w.add_argument("--out", required=True)
    w.set_defaults(fn=cmd_globaff)
    o = sub.add_parser("orient")
    o.add_argument("--root", required=True)
    o.add_argument("--bank", required=True)
    o.add_argument("--sidecars", required=True)
    o.add_argument("--out", required=True)
    o.set_defaults(fn=cmd_orient)
    t = sub.add_parser("gate-table")
    t.add_argument("--table", required=True)
    t.add_argument("--decl", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--channels", choices=sorted(GATED), default="XYB")
    t.set_defaults(fn=cmd_gate_table)
    q = sub.add_parser("gated")
    q.add_argument("--root", required=True)
    q.add_argument("--decl", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--channels", choices=sorted(GATED), default="XYB")
    q.set_defaults(fn=cmd_gated)
    gd = sub.add_parser("gated-decl")
    gd.add_argument("--decl", required=True)
    gd.add_argument("--out", required=True)
    gd.add_argument("--channels", choices=sorted(GATED), default="XYB")
    gd.set_defaults(fn=cmd_gated_decl)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
