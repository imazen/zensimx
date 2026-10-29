"""Rev5 feature bank (spec `benchmarks/rev5_spec_2026-10-04.md` §6): extract one set's basic + peaks + v2 slots and assemble
features.parquet + keys.parquet + _MANIFEST.json in the Rev4 bank's layout.

  python3 rev5_bank.py <set> --bin EXTRACTOR --build-commit SHA --revision 5 --era ERA [--out DIR] [--chunk N] [--limit N]

The pair list comes from the OLD bank's keys.parquet (ref_path, dist_path in row order, human_score 0 placeholder,
row_id = row position), exactly as the Rev4 re-extraction did (`/var/tmp/reextract/assemble_set.py`, which this generalises).
No labels file and nothing under `_sealed/` is opened.

The extractor is asked only for the Rev5 families (`--restore-cuts basic,peaks,v2` at the research layout width). The
research path emits unrequested slots as structural zeros; this assembler writes them as NaN, so an absent feature can never
pass for a measured zero. Every requested slot must be finite. The manifest records the requested slot ranges.

Identity gate (`--revision 4`): the same request at Rev4 must reproduce the Rev4 bank's requested columns bit for bit
(`verify-against`), which shows that requesting a subset does not change the subset's values.
"""

import argparse
import csv
import collections
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

OLD_BANK = Path("/var/tmp/rev4-featbank/bank")
TOKENS = "basic,peaks,v2"
ASSESSMENT_KEY_COLUMNS = {"pair_key", "row_id", "ref_path", "dist_path", "ref_group", "pixels_identical",
    "entry", "image_id", "codec", "q", "codec_param", "param_kind", "case", "panel", "block",
    "source_reference", "source_distorted", "source_role", "delivery_receipt", "source_row_id", "origin",
    "ref_pixels_sha256", "dist_pixels_sha256", "knob", "n_stimuli", "width", "height"}
SLOT_RANGES = ((0, 228), (372, 720))  # Basic f0..155 + Peaks f156..227, V2 f372..719 (feature_set_id::ComputeToken)
SCHEMA = "rev5-featbank-v1"
HDR_TEACHER_SHA = {
    "train": "deb70e775b043a578c77e0c3ff27960ebfa936e9d901497f9d74389e6ce9fbce",
    "val": "4b0f39dea0255659232f248070c8683dd7edf5916c63621cbb099b4afb6c9057",
}
HDR_TRANSFORM = "score=10*q_jod;no-clipping"
# Sealed lane chromahdr admission: 10,422 retained chroma rungs, 26 TRAIN / 12 EVAL consumer origins.
LADDER_EXPECT = dict(rungs=10422, train=26, eval=12)


def _hdr_ladder_plan(a, executor_sha, binary_sha):
    """Native Rev7 views for the sealed HDR chroma-ladder rungs (kept_chroma only).

    Reads the admission cells' keys and hashes, never a quality target. Reference
    and candidate PQ16 PNGs are read in place under --hdr-ladder-root; the
    executor re-hashes every file before the kernel runs.
    """
    from collections import defaultdict

    root = Path(a.hdr_ladder_root).resolve(strict=True)
    roles = json.loads(Path(a.hdr_ladder_roles).read_text())
    if roles.get("schema") != "hdr-chroma-consumer-origin-roles-v1":
        raise ValueError("unregistered consumer role file")
    role_of = {r["source"]: r["consumer_role"] for r in roles["sources"]}
    if sorted(collections.Counter(role_of.values()).items()) != [("eval", LADDER_EXPECT["eval"]), ("train", LADDER_EXPECT["train"])]:
        raise ValueError("consumer roles are not the registered 26 TRAIN / 12 EVAL origins")
    cols = ["view_id", "source", "kind", "encoded_sha256", "rendition_sha256", "reference_sha256", "chroma_q", "quality", "kept_chroma"]
    rows = [r for r in pq.read_table(a.hdr_ladder_cells, columns=cols).to_pylist() if r["kept_chroma"]]
    if len(rows) != LADDER_EXPECT["rungs"]:
        raise ValueError("kept chroma rung count differs from the sealed admission")
    groups = defaultdict(list)
    for r in rows:
        stem = r["view_id"].split("_", 1)[1]
        groups[(str(root / "views" / r["source"] / f"{stem}.png"), r["reference_sha256"])].append(r)
    contract = "hdr-common-primaries-v2-cicp-pq10000"
    views, mapping = [], []
    for (reference, reference_sha), members in sorted(groups.items()):
        if sha256_file(reference) != reference_sha:
            raise ValueError(f"ladder reference changed: {reference}")
        unique = {}
        for r in members:
            unique.setdefault(r["rendition_sha256"], str(root / "artifacts" / "zenavif" / f"{r['encoded_sha256']}.pq.png"))
        ordered = sorted(unique)
        for lo in range(0, len(ordered), 128):
            selected = ordered[lo:lo + 128]
            controls = dict(reference_sha256=reference_sha, candidates=selected,
                            candidate_paths=[unique[d] for d in selected], input_contract=contract,
                            extractor_sha256=binary_sha, extractor_build_commit=a.build_commit,
                            executor_sha256=executor_sha)
            canonical = json.dumps(controls, sort_keys=True, separators=(",", ":"))
            inputs = sorted({reference_sha, *selected, binary_sha, executor_sha,
                             hashlib.sha256(canonical.encode()).hexdigest(),
                             hashlib.sha256(a.build_commit.encode()).hexdigest()})
            vi = len(views)
            views.append(dict(view="hdr-native-rev7-features-v1", inputs=inputs,
                              cell=dict(image_path=reference, codec="hdr:features", q=-1, knob_tuple_json=canonical),
                              requires=["hdr-native-rev7-features-v1"],
                              hint=dict(peak_mem_bytes=2 * 1024**3, threads=1, vram_bytes=None)))
            pos = {d: i for i, d in enumerate(selected)}
            for r in members:
                if r["rendition_sha256"] in pos:
                    mapping.append(dict(view_index=vi, feature_row=pos[r["rendition_sha256"]], view_id=r["view_id"],
                                        source=r["source"], consumer_role=role_of[r["source"]], encoded_sha256=r["encoded_sha256"],
                                        rendition_sha256=r["rendition_sha256"], chroma_q=[int(x) for x in r["chroma_q"]],
                                        quality=int(r["quality"])))
    if len(mapping) != LADDER_EXPECT["rungs"]:
        raise ValueError("ladder rung mapping incomplete")
    out = Path(a.out) / a.set
    out.mkdir(parents=True, exist_ok=False)
    write = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + "\n")
    write("views.spec.json", dict(schema=1, views=views))
    write("membership-feature-map.json", sorted(mapping, key=lambda m: (m["view_index"], m["feature_row"])))
    write("_MANIFEST.json", dict(schema="hdr-ladder-native-rev7-fleet-plan-v1", build_commit=a.build_commit,
          producer_sha256=sha256_file(__file__), cells_sha256=sha256_file(a.hdr_ladder_cells),
          roles_sha256=sha256_file(a.hdr_ladder_roles), library_root=str(root), target_values_read=False, pixels_read=True,
          views=len(views), rows=len(mapping), extractor_sha256=binary_sha, executor_sha256=executor_sha,
          spec_sha256=sha256_file(out / "views.spec.json"), mapping_sha256=sha256_file(out / "membership-feature-map.json"),
          state="DECLARATION_INPUT_ONLY; use zenfleet-ctl declare-views; no extraction"))
    return 0


def cmd_hdr_fleet_plan(a):
    """Declare native Rev7 views through zenfleet's existing declare-views owner.

    Target values are not projected. Historical Rev5 extraction stays frozen.
    """
    from collections import defaultdict

    if a.revision != 7 or a.era != "rev7_rowlanes" or a.limit:
        raise ValueError("complete explicit native Rev7 feature planning required")
    executor = Path(a.hdr_fleet_executor)
    executor_sha, binary_sha = sha256_file(executor), sha256_file(a.bin)
    if a.upiq_native_admission:
        source = Path(a.upiq_native_admission)
        admission = json.loads(source.read_text())
        parent = json.loads((source.parent / "_MANIFEST.json").read_text())
        if (parent.get("schema") != "upiq380-native-rev7-relocation-v1"
                or parent.get("admission_sha256") != sha256_file(source)
                or admission.get("schema") != "upiq380-extraction-admission-rev7-v1"
                or admission.get("requested_ids") != requested_ids()
                or admission.get("role") != "train" or admission.get("tier") != "T2"
                or admission.get("authority") != "D3-2026-10-07"
                or len(admission.get("rows", [])) != 380):
            raise ValueError("unreviewed native UPIQ metadata")
        controls = dict(admission_sha256=sha256_file(source), extractor_sha256=binary_sha,
                        extractor_build_commit=a.build_commit, executor_sha256=executor_sha)
        if a.upiq_native_recovery:
            recovery = json.loads(Path(a.upiq_native_recovery).read_text())
            if set(recovery) != {"root", "sha256", "failed_job_id"} or set(recovery["sha256"]) != {"features.tsv", "features.tsv.manifest.json", "extract.log"}:
                raise ValueError("incomplete native extraction recovery receipt")
            for name, digest in recovery["sha256"].items():
                if sha256_file(Path(recovery["root"]) / name) != digest:
                    raise ValueError("retained native bytes changed")
            producer = json.loads((Path(recovery["root"]) / "features.tsv.manifest.json").read_text())
            if producer["allowlist_sha256"] != controls["admission_sha256"] or producer["research"]["zensim_build_commit"] != a.build_commit:
                raise ValueError("retained native extraction does not bind this admission/build")
            controls["stored_extraction"] = recovery
        canonical = json.dumps(controls, sort_keys=True, separators=(",", ":"))
        inputs = {sha256_file(source), binary_sha, executor_sha, hashlib.sha256(canonical.encode()).hexdigest(),
                  hashlib.sha256(a.build_commit.encode()).hexdigest()}
        if a.upiq_native_recovery:
            inputs.update(recovery["sha256"].values())
            inputs.add(recovery["failed_job_id"])
        for row in admission["rows"]:
            inputs.update([row["reference_sha256"], row["distorted_sha256"]])
        out = Path(a.out) / a.set
        out.mkdir(parents=True, exist_ok=False)
        spec = dict(schema=1, views=[dict(view="hdr-upiq380-rev7-features-v1", inputs=sorted(inputs),
                    cell=dict(image_path=str(source), codec="hdr:upiq380-features", q=-1, knob_tuple_json=canonical),
                    requires=["hdr-upiq380-rev7-features-v1"],
                    hint=dict(peak_mem_bytes=4 * 1024**3, threads=1, vram_bytes=None))])
        (out / "views.spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        (out / "_MANIFEST.json").write_text(json.dumps(dict(schema="upiq380-native-rev7-fleet-plan-v1",
            build_commit=a.build_commit, producer_sha256=sha256_file(__file__),
            admission_sha256=sha256_file(source), spec_sha256=sha256_file(out / "views.spec.json"),
            target_values_read=False, pixels_read=False, rows=380, state="DECLARATION_INPUT_ONLY"), indent=2) + "\n")
        return 0
    if getattr(a, "hdr_ladder_cells", None):
        return _hdr_ladder_plan(a, executor_sha, binary_sha)
    if a.hdr_teacher:
        source = Path(a.hdr_teacher)
        columns = ["row_id", "role", "agree", "ref_path", "dist_path", "ref_sha256", "dist_sha256"]
        rows = pq.read_table(source, columns=columns).to_pylist()
        roles = {r["role"] for r in rows}
        if len(roles) != 1:
            raise ValueError("mixed teacher roles")
        role = next(iter(roles))
        if role not in HDR_TEACHER_SHA or sha256_file(source) != HDR_TEACHER_SHA[role]:
            raise ValueError("unregistered teacher bytes")
        if len(rows) != {"train": 7425, "val": 3900}[role]:
            raise ValueError("incomplete teacher role")
        rows = [r for r in rows if role == "val" or r["agree"]]
        if len(rows) != {"train": 7390, "val": 3900}[role]:
            raise ValueError("teacher agreement membership differs")
        contract = "hdr-common-primaries-v2-cicp-pq10000"
        if getattr(a, "hdr_path_map", None):
            src, dst = a.hdr_path_map.split("=", 1)
            rows = [dict(r, ref_path=dst + r["ref_path"][len(src):] if r["ref_path"].startswith(src) else r["ref_path"],
                         dist_path=dst + r["dist_path"][len(src):] if r["dist_path"].startswith(src) else r["dist_path"]) for r in rows]
    else:
        source = Path(a.hdr_fleet_keys)
        manifest = json.loads((source.parent / "_MANIFEST.json").read_text())
        if (manifest.get("schema") != "hdr-human-source-membership-v1"
                or manifest.get("memberships_sha256") != sha256_file(source)):
            raise ValueError("human source membership receipt differs")
        rows = [r for r in json.loads(source.read_text()) if r["dataset_id"] == "rousselot-4kdtb"]
        if len(rows) != 96 or {r["role"] for r in rows} != {"TRAIN", "EVAL"}:
            raise ValueError("frozen complete 4Kdtb role metadata required")
        contract = "rousselot-registered-k179-as-is"
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["ref_path"], row["ref_sha256"])].append((index, row))
    views, mapping = [], []
    for (reference, reference_sha), members in sorted(groups.items()):
        unique = {}
        for _, row in members:
            unique.setdefault(row["dist_sha256"], row["dist_path"])
        ordered = sorted(unique)
        for lo in range(0, len(ordered), 128):
            selected = ordered[lo:lo + 128]
            controls = dict(reference_sha256=reference_sha, candidates=selected,
                            candidate_paths=[unique[d] for d in selected], input_contract=contract,
                            extractor_sha256=binary_sha, extractor_build_commit=a.build_commit,
                            executor_sha256=executor_sha)
            canonical = json.dumps(controls, sort_keys=True, separators=(",", ":"))
            inputs = sorted({reference_sha, *selected, binary_sha, executor_sha,
                             hashlib.sha256(canonical.encode()).hexdigest(),
                             hashlib.sha256(a.build_commit.encode()).hexdigest()})
            view_index = len(views)
            views.append(dict(view="hdr-native-rev7-features-v1", inputs=inputs,
                              cell=dict(image_path=reference, codec="hdr:features", q=-1, knob_tuple_json=canonical),
                              requires=["hdr-native-rev7-features-v1"],
                              hint=dict(peak_mem_bytes=2 * 1024**3, threads=1, vram_bytes=None)))
            positions = {digest: i for i, digest in enumerate(selected)}
            for index, row in members:
                if row["dist_sha256"] in positions:
                    mapping.append(dict(source_index=index, view_index=view_index,
                                        feature_row=positions[row["dist_sha256"]], metadata=row))
    out = Path(a.out) / a.set
    out.mkdir(parents=True, exist_ok=False)
    write = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + "\n")
    write("views.spec.json", dict(schema=1, views=views))
    write("membership-feature-map.json", sorted(mapping, key=lambda r: r["source_index"]))
    write("_MANIFEST.json", dict(schema="hdr-native-rev7-fleet-plan-v1", build_commit=a.build_commit,
          producer_sha256=sha256_file(__file__), source_path=str(source), source_sha256=sha256_file(source),
          target_values_read=False, agreement_flags_read=bool(a.hdr_teacher), pixels_read=False,
          views=len(views), rows=len(rows), extractor_sha256=binary_sha, executor_sha256=executor_sha,
          spec_sha256=sha256_file(out / "views.spec.json"), mapping_sha256=sha256_file(out / "membership-feature-map.json"),
          state="DECLARATION_INPUT_ONLY; use zenfleet-ctl declare-views; no extraction or TRAIN admission"))
    return 0


def cmd_hdr_extract(a) -> int:
    """The same bank owner, explicit native HDR research mode and original teacher order."""
    import csv
    import e21_cheap_recipe as e21

    if a.revision != 5 or a.limit:
        raise ValueError("E26 requires a complete Rev5 role; partial tables are inadmissible")
    teacher = Path(a.hdr_teacher)
    tab = pq.read_table(teacher)
    role = (tab.schema.metadata or {}).get(b"zensim.hdrteach.role", b"").decode()
    if role not in HDR_TEACHER_SHA or sha256_file(teacher) != HDR_TEACHER_SHA[role]:
        raise ValueError("teacher is not the registered immutable HDRTEACH role")
    rows = tab.to_pylist()
    if any(r["role"] != role for r in rows) or len(rows) != {"train": 7425, "val": 3900}[role]:
        raise ValueError("teacher role/coverage mismatch")
    rows = [r for r in rows if role == "val" or r["agree"]]
    if len(rows) != {"train": 7390, "val": 3900}[role]:
        raise ValueError("registered agreement population mismatch")
    ids = e21.columns("by_v2fy")
    want = set(ids)
    hashes = {}
    for r in rows:
        for prefix in ("ref", "dist"):
            path, digest = r[f"{prefix}_path"], r[f"{prefix}_sha256"]
            if path in hashes and hashes[path] != digest:
                raise ValueError("conflicting native input hash")
            hashes[path] = digest
    for path, expected in hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"native input changed: {path}")
    outdir = Path(a.out) / a.set
    outdir.mkdir(parents=True, exist_ok=False)
    wd = outdir / "_work"
    wd.mkdir()
    keys = pa.Table.from_pylist(rows)
    pq.write_table(keys, outdir / "keys.parquet", compression="zstd")
    keys_sha = sha256_file(outdir / "keys.parquet")
    width, writer, producer, chunks = None, None, None, []
    env = dict(os.environ, ZENSIM_FORMULA_REV="5", ZENSIM_ROOT_FORM="sqrt", RAYON_NUM_THREADS=str(a.threads))
    start = time.time()
    try:
        for ci, lo in enumerate(range(0, len(rows), a.chunk)):
            group = rows[lo:lo + a.chunk]
            pairs, result = wd / f"pairs_{ci:03d}.tsv", wd / f"features_{ci:03d}.tsv"
            with pairs.open("w") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["q", "ref_path", "dist_path"])
                w.writerows((r["q"], r["ref_path"], r["dist_path"]) for r in group)
            cmd = [a.bin, "--pairs", str(pairs), "--ref-root", "/", "--enc-root", "/", "--absolute-pairs",
                   "--requested-ids", ",".join(map(str, ids)), "--input-contract", "hdr-common-primaries-v2-cicp-pq10000",
                   "--threads", str(a.threads), "--out", str(result)]
            t0 = time.time()
            with (wd / f"chunk_{ci:03d}.log").open("w") as lg:
                subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT, check=True)
            man = json.loads(result.with_suffix(".manifest.json").read_text())
            if man["formula_revision"] != "Rev5" or man["requested_ids"] != ids or man["rows"] != len(group):
                raise ValueError("native extractor revision/read-set/coverage mismatch")
            if producer is not None and producer != man["research"]:
                raise ValueError("research provenance changed between chunks")
            producer = man["research"]
            width = man["feature_count"]
            features = pacsv.read_csv(result, parse_options=pacsv.ParseOptions(delimiter="\t"),
                convert_options=pacsv.ConvertOptions(column_types={"q": pa.string(), "dist_basename": pa.string(),
                    **{f"f{i}": pa.float64() for i in range(width)}}))
            if (features["dist_basename"].to_pylist() != [Path(r["dist_path"]).name for r in group]
                    or features["q"].to_pylist() != [r["q"] for r in group]):
                raise ValueError("native row order/key join failed")
            for i in range(width):
                values = features[f"f{i}"].to_numpy()
                if not (np.isfinite(values).all() if i in want else np.isnan(values).all()):
                    raise ValueError(f"measured/absent contract failed f{i}")
            features = features.select([f"f{i}" for i in range(width)])
            features = features.add_column(0, "row_id", pa.array([r["row_id"] for r in group]))
            if writer is None:
                writer = pq.ParquetWriter(outdir / "features.parquet", features.schema, compression="zstd")
            writer.write_table(features)
            chunks.append(dict(rows=len(group), pairs_sha256=sha256_file(pairs), extractor_manifest=man,
                               wall_s=time.time()-t0))
            log(f"{a.set} chunk {ci}: {len(group)} native rows in {time.time()-t0:.1f}s")
    finally:
        if writer is not None:
            writer.close()
    manifest = dict(schema=SCHEMA, study="E26", role=role, rows=len(rows), feature_width=width,
                    formula_revision="Rev5", root_form="sqrt", input_contract="hdr-common-primaries-v2-cicp-pq10000",
                    requested_ids=ids, absent_slots="NaN", target_transform=HDR_TRANSFORM,
                    teacher_sha256=HDR_TEACHER_SHA[role], keys_sha256=keys_sha,
                    features_parquet_sha256=sha256_file(outdir / "features.parquet"),
                    build_commit=a.build_commit, binary_sha256=sha256_file(a.bin), research=producer,
                    native_inputs=hashes, row_order_sha256=hashlib.sha256(json.dumps([r["row_id"] for r in rows]).encode()).hexdigest(),
                    chunks=chunks, wall_s=time.time()-start, assembler_sha256=sha256_file(__file__))
    (outdir / "_MANIFEST.json").write_text(json.dumps(manifest, indent=1)+"\n")
    return 0


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def requested_ids() -> list[int]:
    return [i for lo, hi in SLOT_RANGES for i in range(lo, hi)]


def read_feature_csv(csv_path, expected_row_ids, width):
    """Canonical bank ingress: exact f64 values and NaN for unrequested slots."""
    conv = pacsv.ConvertOptions(column_types={"row_id": pa.int64(), **{f"f{i}": pa.float64() for i in range(width)}},
                               include_columns=["row_id"] + [f"f{i}" for i in range(width)])
    tab = pacsv.read_csv(csv_path, convert_options=conv)
    rid = tab.column("row_id").to_numpy()
    order = np.argsort(rid, kind="stable")
    expected = np.asarray(expected_row_ids, dtype=np.int64)
    if tab.num_rows != len(expected) or not (rid[order] == expected).all():
        raise ValueError("row_id join failed")
    tab = tab.take(pa.array(order)); want = set(requested_ids()); cols = {}
    for i in range(width):
        v = tab.column(f"f{i}").to_numpy()
        if i in want:
            if not np.isfinite(v).all():
                raise ValueError(f"non-finite value in requested slot f{i}")
            cols[f"f{i}"] = pa.array(v)
        else:
            cols[f"f{i}"] = pa.array(np.full(len(v), np.nan))
    return tab, cols


def cmd_extract(a) -> int:
    s = a.set
    assessment = getattr(a, "instrument_manifest", None)
    spec = None
    key_path = OLD_BANK / s / "keys.parquet"
    if assessment:
        from v2c_wide import assessment_path as safe_path
        from v2_common import refuse_immutable_output
        if not s or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in s):
            raise ValueError("assessment set token required")
        spec = json.loads(safe_path(assessment).read_text())
        if a.limit or spec.get("schema") != "rev5-assessment-keys-v1" or spec.get("labels_read") is not False:
            raise ValueError("explicit label-free assessment keys required")
        key_path = safe_path(spec["keys"]["path"])
        roots = [Path(v) for v in spec["immutable_roots"]] + [key_path.parent, Path(assessment).parent]
        for dest in [Path(a.out) / s, Path(a.out) / "_work" / s]:
            refuse_immutable_output(dest, roots)
            if dest.exists():
                raise ValueError("fresh assessment bank/scratch required")
        if set(pq.ParquetFile(key_path).schema_arrow.names) - ASSESSMENT_KEY_COLUMNS:
            raise ValueError("label-bearing assessment keys forbidden before payload read")
        # Revision 7: the by-construction re-extraction of the same instruments (benchmarks/byconstruction_2026-10-22.md).
        if a.revision not in (5, 7) or sha256_file(key_path) != spec["keys"]["sha256"] or sha256_file(a.bin) != spec["extractor"]["sha256"]:
            raise ValueError("changed assessment keys/extractor")
    keys = pq.read_table(key_path)
    if spec:
        if set(keys.column_names) - ASSESSMENT_KEY_COLUMNS or keys.num_rows != spec["rows"] or keys.num_rows == 0:
            raise ValueError("assessment key columns/rows are not label-free")
        for path in keys.column("ref_path").to_pylist() + keys.column("dist_path").to_pylist():
            safe_path(path)
        pixel_columns = [c for c in ("ref_path", "dist_path", "source_reference", "source_distorted") if c in keys.column_names]
        roots += sorted({safe_path(p).resolve().parent for c in pixel_columns for p in keys.column(c).to_pylist()})
        for dest in [Path(a.out) / s, Path(a.out) / "_work" / s]:
            refuse_immutable_output(dest, roots)
        if keys.column("row_id").to_pylist() != list(range(keys.num_rows)) or len(set(keys.column("pair_key").to_pylist())) != keys.num_rows:
            raise ValueError("assessment ordinal/key order required")
    n = a.limit or keys.num_rows
    ref, dist = keys.column("ref_path").to_pylist(), keys.column("dist_path").to_pylist()
    pk = keys.column("pair_key").to_pylist()
    outdir = Path(a.out) / (f"_partial/{s}" if a.limit else s)
    outdir.mkdir(parents=True, exist_ok=True)
    wd = Path(a.out) / "_work" / s
    wd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ZENSIM_FORMULA_REV=str(a.revision), ZENSIM_ROOT_FORM="sqrt", RAYON_NUM_THREADS=str(a.threads))
    want = set(requested_ids())
    audit_records = []
    writer, chunk_meta, tsv_hashes, fsids, eras, width, t_all = None, [], [], set(), None, None, time.time()
    for ci, lo in enumerate(range(0, n, a.chunk)):
        hi = min(lo + a.chunk, n)
        tsv = wd / f"pairs_{ci:03d}.tsv"
        with open(tsv, "w") as f:
            f.write("ref_path\tdist_path\thuman_score\trow_id\n")
            for i in range(lo, hi):
                f.write(f"{ref[i]}\t{dist[i]}\t{i if spec else 0}\t{i}\n")
        tsv_hashes.append(sha256_file(tsv))
        csvp = wd / f"chunk_{ci:03d}.csv"
        cmd = [a.bin, "--corpus", "pairs-tsv", "--path", str(tsv), "--out", str(csvp), "--restore-cuts", TOKENS,
               "--input-contract", "legacy-rgb8", "--era-label", a.era, "--force-tier", a.tier]
        audit_path = wd / f"chunk_{ci:03d}.audit.jsonl"
        if spec:
            cmd += ["--audit-jsonl", str(audit_path)]
        t0 = time.time()
        with open(wd / f"chunk_{ci:03d}.log", "w") as lg:
            rc = subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
        t_ext = time.time() - t0
        if rc != 0:
            sys.exit(f"extractor failed rc={rc} chunk {ci}: see {wd}/chunk_{ci:03d}.log")
        man = json.loads(Path(f"{csvp}.manifest.json").read_text())
        rm = json.loads(Path(f"{csvp}.research_manifest.json").read_text())
        fsids.add(man["feature_set_id"])
        eras = eras or rm["formula_revision_eras"]
        if eras != rm["formula_revision_eras"] or man["era_label"] != a.era or str(man["formula_revision"]) != str(a.revision):
            sys.exit(f"chunk {ci}: extractor manifest disagrees (era {man['era_label']!r}, revision {man['formula_revision']!r})")
        names = pacsv.read_csv(csvp, read_options=pacsv.ReadOptions(skip_rows_after_names=10**9)).column_names
        w = sum(1 for c in names if c.startswith("f") and c[1:].isdigit())
        width = width or w
        if w != width or not want <= set(range(w)):
            sys.exit(f"chunk {ci}: width {w} (first chunk {width}) does not cover the requested slots")
        try:
            tab, cols = read_feature_csv(csvp, np.arange(lo, hi), w)
        except ValueError as error:
            sys.exit(f"chunk {ci}: {error}")
        out = pa.table({"pair_key": pa.array(pk[lo:hi]), "row_id": tab.column("row_id"), **cols})
        if writer is None:
            writer = pq.ParquetWriter(outdir / "features.parquet", out.schema, compression="zstd", compression_level=3)
        writer.write_table(out, row_group_size=5000)
        audits = None
        if spec:
            audits = [json.loads(line) for line in audit_path.read_text().splitlines()]
            audits.sort(key=lambda r: r["human_score"])
            if len(audits) != hi-lo or [r["human_score"] for r in audits] != list(range(lo, hi)):
                raise ValueError("assessment audit key coverage")
            for j, audit in enumerate(audits):
                expected = lo+j
                if audit["model_inputs"] or audit["reference"] != ref[expected] or audit["distorted"] != dist[expected]:
                    raise ValueError("features-only audit must have no model and exact paths")
                if "canonical_features_f32_le_sha256" in audit and audit["canonical_features_f32_le_sha256"] != hashlib.sha256(np.array(
                        [tab.column(f"f{i}")[j].as_py() for i in range(w)], dtype="<f4").tobytes()).hexdigest():
                    raise ValueError("extractor/CSV f32 byte parity failed")
                if "delivery_receipt" in keys.column_names:
                    delivery_path = safe_path(keys.column("delivery_receipt")[expected].as_py())
                    delivery = json.loads(delivery_path.read_text())
                    if any(delivery[k] != audit[k] for k in ("width", "height", "reference_pixels_sha256", "distorted_pixels_sha256")):
                        raise ValueError("canonical steering decoded pixel parity failed")
            audit_records.extend(audits)
        csv_bytes = csvp.stat().st_size
        csvp.unlink()
        chunk_meta.append({"chunk": ci, "rows": hi - lo, "pairs_tsv_sha256": tsv_hashes[-1], "extract_wall_s": round(t_ext, 2),
                           "csv_bytes_deleted": csv_bytes, "extractor_manifest": man, **({"audit_path":str(audit_path),"audit_sha256":sha256_file(audit_path)} if spec else {})})
        log(f"{s} chunk {ci} rows {lo}-{hi} extract {t_ext:.1f}s")
    writer.close()
    if len(fsids) != 1:
        sys.exit(f"feature_set_id changed between chunks: {fsids}")
    if not a.limit:
        if spec:
            extra = {k: pa.array([v[k] for v in audit_records]) for k in (
                "pixels_identical", "width", "height", "reference_file_sha256", "distorted_file_sha256",
                "reference_pixels_sha256", "distorted_pixels_sha256")}
            delivered = pa.table({**{k: keys.column(k) for k in keys.column_names if k not in extra}, **extra})
            pq.write_table(delivered, outdir / "keys.parquet", compression="zstd")
        else:
            (outdir / "keys.parquet").write_bytes(key_path.read_bytes())
    manifest = {
        "set": s, "schema": SCHEMA, "rows": n, "feature_width": width, "requested_slot_ranges": [list(r) for r in SLOT_RANGES],
        "absent_slots": "NaN", "restore_cuts": TOKENS, "build_commit": a.build_commit, "binary_sha256": sha256_file(a.bin),
        "formula_revision": f"Rev{a.revision}", "root_form": "sqrt", "input_contract": "legacy-rgb8", "tier_request": a.tier,
        "era_label": a.era, "feature_set_id": next(iter(fsids)), "formula_revision_eras": eras,
        "pairs_tsv_sha256": tsv_hashes, "pairs_origin": f"{key_path} (ref_path,dist_path in row order)",
        "keys_sha256": sha256_file(outdir / "keys.parquet") if spec else sha256_file(key_path),
        "features_parquet_sha256": sha256_file(outdir / "features.parquet"), "dtype": "float64",
        "chunks": chunk_meta, "wall_s_total": round(time.time() - t_all, 1),
        "assembler": {"path": "scripts/rev4_featpot/rev5_bank.py", "sha256": sha256_file(__file__)},
    }
    if spec:
        manifest["assessment"] = {"features_only": True, "labels_read": False,
            "input_keys": spec["keys"], "input_manifest_sha256": sha256_file(assessment),
            "instrument": spec["instrument"], "exposure_freeze": spec["exposure_freeze"],
            "human_score_semantics":"ordinal row key only; no target", "input_manifest":str(assessment),
            "all_decoded_pixels_bound":True, "immutable_roots":[str(r.resolve()) for r in roots],
            "extractor_feature_digest_available": all("canonical_features_f32_le_sha256" in r for r in audit_records)}
    (outdir / "_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    log(f"{s} DONE rows {n} width {width} wall {manifest['wall_s_total']}s fsid {manifest['feature_set_id']}")
    return 0


def cmd_verify_against(a) -> int:
    """Requested columns of a subset-request bank equal the reference bank's columns bit for bit (row-limited ok)."""
    base = Path(a.out) / a.set
    if not base.is_dir():
        base = Path(a.out) / "_partial" / a.set
    mine = pq.read_table(base / "features.parquet")
    ref = pq.read_table(Path(a.reference) / a.set / "features.parquet")
    n = mine.num_rows
    if mine.column("pair_key").to_pylist() != ref.column("pair_key").to_pylist()[:n]:
        sys.exit("pair_key order differs")
    bad = []
    for i in requested_ids():
        x, y = mine.column(f"f{i}").to_numpy(), ref.column(f"f{i}").to_numpy()[:n]
        if not np.array_equal(x.view(np.uint64), y.view(np.uint64)):
            bad.append(i)
    absent_ok = all(np.isnan(mine.column(c).to_numpy()).all() for c in mine.column_names
                    if c.startswith("f") and c[1:].isdigit() and int(c[1:]) not in set(requested_ids()))
    print(json.dumps({"set": a.set, "rows": n, "requested": len(requested_ids()), "differing_slots": bad[:20],
                      "n_differing": len(bad), "absent_all_nan": absent_ok}))
    return 0 if not bad and absent_ok else 1


# PALETTE uses the existing feature-bank owner; inputs are projected keys,
# never the target-bearing instrument tables or labels__* payloads.
PALETTE_MEMBERS = {"kadid_train", "kadid_select", "tid2013", "konfig_train", "konfig_val", "cid22_a25", "safesyn", "cid22_train"}
PALETTE_TABLES = ("kadid", "tid2013", "konfig", "cid22_a25", "safesyn_fit", "safesyn_dev", "cid22_fit", "cid22_dev")
PALETTE_IDS = list(range(1825, 1867))


def palette_extract_group(a, name, keys, source_receipts):
    """Extract thin keyed sidecars with pixel-bound receipts and strict row coverage."""
    out = Path(a.out) / name
    if out.exists():
        manifest_path = out / '_MANIFEST.json'
        if not manifest_path.exists():
            raise ValueError(f'incomplete existing palette group: {out}; preserve before retry')
        manifest = json.loads(manifest_path.read_text())
        if (manifest['build_commit'] != a.build_commit or manifest['binary_sha256'] != sha256_file(a.bin)
                or manifest['assembler_sha256'] != sha256_file(__file__) or manifest['sources'] != source_receipts
                or manifest['rows'] != len(keys) or manifest['feature_ids'] != PALETTE_IDS
                or manifest['keys_sha256'] != sha256_file(out/'keys.parquet')
                or manifest['features_sha256'] != sha256_file(out/'features__palette.parquet')):
            raise ValueError('existing palette receipt mismatch')
        stored = pq.read_table(out/'keys.parquet', columns=['pair_key'])
        if stored['pair_key'].to_pylist() != keys['pair_key'].to_pylist():
            raise ValueError('existing palette key order mismatch')
        log(f'PALETTE verified completed group {name}: {len(keys)} rows')
        return manifest
    out.mkdir(parents=True, exist_ok=False)
    wd = out / "_work"
    wd.mkdir()
    pq.write_table(keys, out / "keys.parquet", compression="zstd")
    records = keys.to_pylist()
    writer, producer, chunks = None, None, []
    env = dict(os.environ, TMPDIR=str(Path.home()/"tmp/palette"), ZENSIM_FORMULA_REV="5", ZENSIM_ROOT_FORM="sqrt", RAYON_NUM_THREADS=str(a.threads))
    start = time.time()
    for ci, lo in enumerate(range(0, len(records), a.chunk)):
        group = records[lo:lo+a.chunk]
        pairs, csvp, audit = (wd/f"{ci:04d}.{ext}" for ext in ("tsv", "csv", "audit.jsonl"))
        with pairs.open("w") as f:
            f.write("ref_path\tdist_path\thuman_score\trow_id\n")
            for i,r in enumerate(group,lo):
                f.write(f"{r['ref_path']}\t{r['dist_path']}\t{i}\t{i}\n")
        cmd = [a.bin,"--corpus","pairs-tsv","--path",str(pairs),"--out",str(csvp),"--palette-only",
               "--input-contract","legacy-rgb8","--era-label","palette_v2","--audit-jsonl",str(audit)]
        with (wd/f"{ci:04d}.log").open("w") as lg:
            subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT, check=True)
        man=json.loads(Path(str(csvp)+".manifest.json").read_text())
        if man['formula_revision']!='5' or man['populated_feature_ids']!=PALETTE_IDS:
            raise ValueError("palette extractor revision/coverage mismatch")
        research_path=Path(str(csvp)+'.research_manifest.json')
        research=json.loads(research_path.read_text())
        if (research['zensim_build_commit']!=a.build_commit or research['emitted_slot_count']!=42
                or research['emitted_slots']!='1825-1866'
                or man['producer_binary_sha256']!=sha256_file(a.bin)):
            raise ValueError('palette producer receipt mismatch')
        fsid=man['feature_set_id']
        if not fsid.startswith("palette@w1867/palette_v2#") or (producer is not None and producer!=fsid):
            raise ValueError("palette feature identity mismatch")
        producer=fsid
        tab=pacsv.read_csv(csvp,convert_options=pacsv.ConvertOptions(include_columns=['row_id']+[f'f{i}' for i in PALETTE_IDS],
            column_types={'row_id':pa.int64(),**{f'f{i}':pa.float64() for i in PALETTE_IDS}}))
        order=np.argsort(tab['row_id'].to_numpy(),kind='stable');tab=tab.take(pa.array(order))
        if tab['row_id'].to_pylist()!=list(range(lo,lo+len(group))):raise ValueError('palette row join mismatch')
        if not all(np.isfinite(tab[f'f{i}'].to_numpy()).all() for i in PALETTE_IDS):raise ValueError('nonfinite palette features')
        audits=sorted((json.loads(l) for l in audit.read_text().splitlines()),key=lambda r:r['human_score'])
        if len(audits)!=len(group):raise ValueError('palette audit coverage')
        for i,(r,v) in enumerate(zip(group,audits),lo):
            if v['model_inputs'] or v['human_score']!=i or v['reference']!=r['ref_path'] or v['distorted']!=r['dist_path']:
                raise ValueError('palette audit model/path/order mismatch')
            for key,old in [('reference_pixels_sha256','ref_pixels_sha256'),('distorted_pixels_sha256','dist_pixels_sha256')]:
                if old in r and r[old]!=v[key]:raise ValueError(f"pixel-era mismatch {name} row {i} {key}")
        result=tab.add_column(0,'pair_key',pa.array([r['pair_key'] for r in group]))
        for field in ('reference_pixels_sha256','distorted_pixels_sha256'):
            result=result.append_column(field,pa.array([v[field] for v in audits]))
        if writer is None:writer=pq.ParquetWriter(out/'features__palette.parquet',result.schema,compression='zstd')
        writer.write_table(result)
        chunks.append({'rows':len(group),'pairs_sha256':sha256_file(pairs),'audit_sha256':sha256_file(audit),
                       'csv_sha256':sha256_file(csvp),'research_manifest_sha256':sha256_file(research_path),'extractor_manifest':man})
        log(f"PALETTE {name} {lo+len(group)}/{len(records)}")
    if writer is None:raise ValueError('empty palette input')
    writer.close()
    manifest={'schema':'palette-sidecar-v1','set':name,'rows':len(records),'labels_read':False,'serving_allowed':False,
              'formula_revision':5,'palette_revision':'palette_v2','feature_set_id':producer,'feature_ids':PALETTE_IDS,
              'dtype':'float64','input_contract':'legacy-rgb8','build_commit':a.build_commit,'binary_sha256':sha256_file(a.bin),
              'assembler_sha256':sha256_file(__file__),'keys_sha256':sha256_file(out/'keys.parquet'),
              'features_sha256':sha256_file(out/'features__palette.parquet'),'sources':source_receipts,'chunks':chunks,
              'wall_s':time.time()-start,'instrument_auxiliary_ids':'Do not overwrite instrument f1825+; join explicitly by registered ID mapping.'}
    (out/'_MANIFEST.json').write_text(json.dumps(manifest,indent=1)+'\n')
    return manifest


def cmd_palette(a):
    if a.revision!=5 or a.limit or a.era!='palette_v2' or a.tier!='native':
        raise ValueError('palette requires complete Rev5 palette_v2 native extraction')
    root=Path(a.palette_instrument)
    if root.resolve()!=Path('/var/tmp/rev4-featpot/v2c5').resolve():raise ValueError('explicit authorized v2c5 root required')
    selected={};receipts={}
    for name in PALETTE_TABLES:
        path=root/'wide/main/real'/f'{name}.keys.parquet'
        # Parquet projection prevents target payload reads.
        tab=pq.read_table(path,columns=['pair_key','member_set'])
        receipts[name]={'path':str(path),'sha256':sha256_file(path),'columns_read':tab.column_names,'rows':len(tab)}
        for key,member in zip(tab['pair_key'].to_pylist(),tab['member_set'].to_pylist()):
            if member not in PALETTE_MEMBERS:raise ValueError('protected member refused before bank access')
            selected.setdefault(member,set()).add(key)
    done={}
    columns=['pair_key','ref_path','dist_path','ref_pixels_sha256','dist_pixels_sha256']
    for member,want in selected.items():
        path=OLD_BANK/member/'keys.parquet'
        tab=pq.read_table(path,columns=columns)
        tab=tab.filter(pa.array([k in want for k in tab['pair_key'].to_pylist()]))
        if set(tab['pair_key'].to_pylist())!=want:raise ValueError('palette member key coverage mismatch')
        done[member]=palette_extract_group(a,member,tab,[{'path':str(path),'sha256':sha256_file(path),'columns_read':columns},receipts])
    for name in ('nits','live','mciqa'):
        path=root/'external'/f'{name}.keys.parquet'
        tab=pq.read_table(path,columns=columns)
        done[name]=palette_extract_group(a,name,tab,[{'path':str(path),'sha256':sha256_file(path),'columns_read':columns,'role':'external-features-only'}])
    # Existing e15 owner records the ordered selection indices in label-free keys.
    keypath=root/'e15/coverage_pool.keys.parquet'
    path=root/'e15/selection.parquet'
    cols=['source_filename','type','severity_level','ref_path','dist_path','family']
    indices=pq.read_table(keypath,columns=['__index_level_0__'])['__index_level_0__'].to_pylist()
    tab=pq.read_table(path,columns=cols).take(pa.array(indices))
    if len(indices)!=len(set(indices)):raise ValueError('coverage ordinal index duplication')
    tab=tab.add_column(0,'pair_key',pa.array([hashlib.sha256((r['ref_path']+'\0'+r['dist_path']).encode()).hexdigest() for r in tab.to_pylist()]))
    done['coverage_pool']=palette_extract_group(a,'coverage_pool',tab,[{'path':str(path),'sha256':sha256_file(path),'columns_read':cols,'role':'TRAIN-ordinal'},
        {'path':str(keypath),'sha256':sha256_file(keypath),'columns_read':['__index_level_0__'],'join':'original selection row index'}])
    (Path(a.out)/'_MANIFEST.json').write_text(json.dumps({'schema':'palette-bank-v1','build_commit':a.build_commit,
        'labels_read':False,'sets':{k:{'rows':v['rows'],'manifest_sha256':sha256_file(Path(a.out)/k/'_MANIFEST.json')} for k,v in done.items()},'instrument_keys':receipts},indent=1)+'\n')
    return 0


def cmd_palette_views(a):
    """Join the measured family into ordered instrument views, preserving repeats."""
    root=Path(a.palette_instrument)
    if root.resolve()!=Path('/var/tmp/rev4-featpot/v2c5').resolve() or a.limit:
        raise ValueError('complete authorized instrument required')
    bank=Path(a.palette_views)
    top=json.loads((bank/'_MANIFEST.json').read_text())
    if top['build_commit']!=a.build_commit or top['labels_read']:
        raise ValueError('palette bank binding mismatch')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    features={};positions={};receipts={}
    for member in PALETTE_MEMBERS:
        path=bank/member/'_MANIFEST.json'
        man=json.loads(path.read_text())
        feature_path=bank/member/'features__palette.parquet'
        if (man['feature_ids']!=PALETTE_IDS or man['palette_revision']!='palette_v2' or man['labels_read'] or man['build_commit']!=a.build_commit
                or man['binary_sha256']!=sha256_file(a.bin)
                or man['features_sha256']!=sha256_file(feature_path)
                or top['sets'][member]['manifest_sha256']!=sha256_file(path)):
            raise ValueError('palette group provenance mismatch')
        tab=pq.read_table(feature_path)
        keys=tab['pair_key'].to_pylist()
        if len(keys)!=len(set(keys)):raise ValueError('duplicate bank key')
        features[member]=tab
        positions[member]={k:i for i,k in enumerate(keys)}
        receipts[member]={'path':str(path),'sha256':sha256_file(path)}
    views={}
    for name in PALETTE_TABLES:
        keypath=root/'wide/main/real'/f'{name}.keys.parquet'
        if top['instrument_keys'][name]['sha256']!=sha256_file(keypath):
            raise ValueError('instrument source changed after extraction')
        keys=pq.read_table(keypath,columns=['pair_key','member_set'])
        members=keys['member_set'].to_pylist();pairs=keys['pair_key'].to_pylist()
        if not set(members)<=PALETTE_MEMBERS:raise ValueError('protected member refused')
        ordered_members=sorted(set(members));offsets={};chunks=[];offset=0
        for member in ordered_members:
            offsets[member]=offset;chunks.append(features[member]);offset+=len(features[member])
        joined=pa.concat_tables(chunks).take(pa.array([offsets[m]+positions[m][k] for m,k in zip(members,pairs)]))
        if joined['pair_key'].to_pylist()!=pairs:raise ValueError('instrument join order mismatch')
        view=keys.append_column('row_id',pa.array(range(len(keys)),type=pa.int64()))
        for i in PALETTE_IDS:view=view.append_column(f'palette_f{i}',joined[f'f{i}'])
        for field in ('reference_pixels_sha256','distorted_pixels_sha256'):view=view.append_column(field,joined[field])
        path=out/f'{name}.parquet';pq.write_table(view,path,compression='zstd')
        views[name]={'rows':len(view),'unique_member_pair_keys':len(set(zip(members,pairs))),
                     'sha256':sha256_file(path),'instrument_keys_path':str(keypath),'instrument_keys_sha256':sha256_file(keypath),
                     'columns_read':['pair_key','member_set'],'groups':{m:receipts[m] for m in ordered_members}}
        log(f'PALETTE view {name}: {len(view)} ordered rows')
    manifest={'schema':'palette-instrument-views-v1','build_commit':a.build_commit,'assembler_sha256':sha256_file(__file__),
        'bank_manifest_sha256':sha256_file(bank/'_MANIFEST.json'),'labels_read':False,'serving_allowed':False,
        'feature_set_id':'palette@w1867/palette_v2#30b09cd1','feature_ids':PALETTE_IDS,
        'column_map':{str(i):f'palette_f{i}' for i in PALETTE_IDS},'views':views,
        'coverage_pool':'../bank/coverage_pool/features__palette.parquet; original selection ordinal key receipt',
        'external':'../bank/{nits,live,mciqa}/features__palette.parquet; external features only',
        'collision_policy':'Named palette_f columns never overwrite instrument f1825+ auxiliary columns.'}
    (out/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return 0


def cmd_palette_chromaq(a):
    import importlib.util
    owner=Path(a.palette_chromaq)/'analyze.py'
    spec=importlib.util.spec_from_file_location('chromaq_owner',owner)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fields=['image_path','knob_tuple_json','encoded_filename']
    rows=[];sources=[]
    for name in ('sweep444.tsv','sweep420.tsv'):
        path=Path(a.palette_chromaq)/'out'/name
        sources.append({'path':str(path),'sha256':sha256_file(path),'columns_read':fields})
        with path.open() as f:
            for raw in csv.DictReader(f,delimiter='\t'):
                r={k:raw[k] for k in fields}
                family,multiplier=module.family(json.loads(r['knob_tuple_json']))
                ref=str(Path(a.palette_chromaq)/'src'/Path(r['image_path']).name)
                dist=str(Path(a.palette_chromaq)/'dist'/(Path(r['encoded_filename']).stem+'.png'))
                rows.append({'pair_key':hashlib.sha256((ref+'\0'+dist).encode()).hexdigest(),
                             'ref_path':ref,'dist_path':dist,'ladder':family,'multiplier':float(multiplier)})
    sources.append({'classification_owner':str(owner),'sha256':sha256_file(owner),'labels_read':False})
    manifest=palette_extract_group(a,'chromaq',pa.Table.from_pylist(rows),sources)
    features=pq.read_table(Path(a.out)/'chromaq/features__palette.parquet',columns=[f'f{i}' for i in PALETTE_IDS])
    groups={}
    for i,r in enumerate(rows):groups.setdefault((r['ladder'],r['multiplier']),[]).append(i)
    signals=['mean_shift','lightness_signed','chroma_signed','hue_signed','weight_emd','largest_shift']
    with (Path(a.out)/'chromaq/curves.csv').open('x') as f:
        writer=csv.writer(f);writer.writerow(['ladder','multiplier','N','pairs']+[x+'_mean' for x in signals]+[x+'_median' for x in signals])
        for (ladder,multiplier),ix in sorted(groups.items()):
            for n in range(2,9):
                columns=[features[f'f{1825+(n-2)*6+j}'].to_numpy()[ix] for j in range(6)]
                writer.writerow([ladder,multiplier,n,len(ix)]+[float(np.mean(c)) for c in columns]+[float(np.median(c)) for c in columns])
    log(f'CHROMAQ diagnostic complete: {len(rows)} pairs; {len(groups)*7} curve cells')
    return 0


def validate_palette_instrument_manifest(path, build_commit, bank_sha256, pinned_sha256):
    """Admit exact pinned bytes and the registered research-only ID/column map.

    The pin is supplied by the consumer's frozen registration, never inferred
    from the file being admitted. Semantics are checked even when its byte pin
    matches, so a consistently rehashed wrong-era/map receipt is still refused.
    """
    raw=Path(path).read_bytes()
    if (not isinstance(pinned_sha256,str) or len(pinned_sha256)!=64
            or hashlib.sha256(raw).hexdigest()!=pinned_sha256):
        raise ValueError('pinned instrument manifest bytes required')
    im=json.loads(raw)
    if (im.get('schema')!='palette-instrument-views-v1'
            or im.get('labels_read') is not False or im.get('serving_allowed') is not False
            or im.get('build_commit')!=build_commit
            or im.get('feature_set_id')!='palette@w1867/palette_v2#30b09cd1'
            or im.get('feature_ids')!=PALETTE_IDS
            or any(type(i) is not int for i in im.get('feature_ids',[]))
            or im.get('column_map')!={str(i):f'palette_f{i}' for i in PALETTE_IDS}
            or set(im.get('views',{}))!=set(PALETTE_TABLES)
            or im.get('bank_manifest_sha256')!=bank_sha256):
        raise ValueError('instrument semantic identity/map mismatch')
    return im


def cmd_palette_verify(a):
    """Validate artifact schema, hashes, row identity and ordered joins independently."""
    root=Path(a.palette_verify)
    bank=root/'bank';top=json.loads((bank/'_MANIFEST.json').read_text())
    expected=PALETTE_MEMBERS|{'nits','live','mciqa','coverage_pool'}
    if set(top['sets'])!=expected or top['build_commit']!=a.build_commit or top['labels_read']:
        raise ValueError('complete authorized bank required')
    instrument_path=root/'instrument/_MANIFEST.json'
    im=validate_palette_instrument_manifest(instrument_path, a.build_commit,
        sha256_file(bank/'_MANIFEST.json'), a.palette_instrument_manifest_sha256)
    verified={};tables={}
    for name,entry in top['sets'].items():
        folder=bank/name;mp=folder/'_MANIFEST.json';m=json.loads(mp.read_text())
        if (sha256_file(mp)!=entry['manifest_sha256'] or m['build_commit']!=a.build_commit
                or m['palette_revision']!='palette_v2' or m['labels_read'] or m['serving_allowed']
                or m['binary_sha256']!=sha256_file(a.bin) or m['feature_ids']!=PALETTE_IDS):
            raise ValueError('palette receipt mismatch')
        for file,key in [('keys.parquet','keys_sha256'),('features__palette.parquet','features_sha256')]:
            if sha256_file(folder/file)!=m[key]:raise ValueError('palette file changed')
        keys=pq.read_table(folder/'keys.parquet',columns=['pair_key'])
        t=pq.read_table(folder/'features__palette.parquet')
        names={'pair_key','row_id','reference_pixels_sha256','distorted_pixels_sha256'}|{f'f{i}' for i in PALETTE_IDS}
        if set(t.column_names)!=names or len(t)!=m['rows'] or len(t)!=entry['rows']:
            raise ValueError('palette schema/row mismatch')
        if t['pair_key'].to_pylist()!=keys['pair_key'].to_pylist() or t['row_id'].to_pylist()!=list(range(len(t))):
            raise ValueError('palette key/order mismatch')
        for i in PALETTE_IDS:
            if t[f'f{i}'].type!=pa.float64() or not np.isfinite(t[f'f{i}'].to_numpy()).all():raise ValueError('invalid palette measured values')
        if sum(c['rows'] for c in m['chunks'])!=len(t):raise ValueError('chunk row coverage')
        for c in m['chunks']:
            em=c['extractor_manifest']
            if em['feature_set_id']!='palette@w1867/palette_v2#30b09cd1' or em['populated_feature_ids']!=PALETTE_IDS:
                raise ValueError('chunk producer identity')
        verified[name]={'rows':len(t),'features_sha256':m['features_sha256']}
        tables[name]=t
    views={}
    for name,m in im['views'].items():
        path=root/'instrument'/f'{name}.parquet'
        if sha256_file(path)!=m['sha256'] or sha256_file(m['instrument_keys_path'])!=m['instrument_keys_sha256']:
            raise ValueError('instrument file/source changed')
        v=pq.read_table(path);keys=pq.read_table(m['instrument_keys_path'],columns=['pair_key','member_set'])
        if len(v)!=m['rows'] or v['row_id'].to_pylist()!=list(range(len(v))):raise ValueError('instrument rows')
        for col in ('pair_key','member_set'):
            if v[col].to_pylist()!=keys[col].to_pylist():raise ValueError('instrument key order')
        # Compare each measured column against a key lookup in the bank,
        # independent of the projection's concatenation/offset algorithm.
        members=v['member_set'].to_pylist();pairs=v['pair_key'].to_pylist()
        if not set(members)<=PALETTE_MEMBERS:raise ValueError('unauthorized instrument member')
        lookups={member:{k:j for j,k in enumerate(tables[member]['pair_key'].to_pylist())} for member in set(members)}
        for member in set(members):
            ix=np.array([j for j,memb in enumerate(members) if memb==member],dtype=np.int64)
            bank_ix=np.array([lookups[member][pairs[j]] for j in ix],dtype=np.int64)
            for i in PALETTE_IDS:
                if not np.array_equal(v[f'palette_f{i}'].to_numpy()[ix],tables[member][f'f{i}'].to_numpy()[bank_ix]):
                    raise ValueError('instrument measured value join mismatch')
        views[name]={'rows':len(v),'sha256':m['sha256'],'exact_column_join':True}
    receipt={'schema':'palette-final-verification-v2','build_commit':a.build_commit,'verifier_sha256':sha256_file(__file__),
        'instrument_manifest_sha256':a.palette_instrument_manifest_sha256,
        'instrument_feature_set_id':im['feature_set_id'],'instrument_feature_ids':im['feature_ids'],
        'instrument_column_map':im['column_map'],'instrument_build_commit':im['build_commit'],
        'instrument_serving_allowed':im['serving_allowed'],'semantic_identity_checked':True,
        'bank':verified,'instrument':views,'bank_rows':sum(x['rows'] for x in verified.values()),
        'instrument_rows':sum(x['rows'] for x in views.values()),'labels_read':False,'serving_allowed':False}
    with (root/'_VERIFIED.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    log(f"PALETTE verified bank {receipt['bank_rows']} rows; ordered views {receipt['instrument_rows']} rows")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "verify-against"])
    ap.add_argument("set")
    ap.add_argument("--bin")
    ap.add_argument("--palette-instrument-manifest-sha256", help="frozen exact-byte instrument manifest pin, required for palette verification")
    ap.add_argument("--palette-verify", help="independent final palette artifact and join verification")
    ap.add_argument("--palette-views", help="complete pinned palette bank to project into ordered TRAIN instrument views")
    ap.add_argument("--palette-chromaq", help="existing CHROMAQ ladder owner directory, diagnostic only")
    ap.add_argument("--palette-instrument", help="authorized v2c5 TRAIN/external features-only sidecars")
    ap.add_argument("--instrument-manifest", type=Path, help="explicit label-free assessment keys; historical bank default unchanged")
    ap.add_argument("--upiq-native-recovery", type=Path, help="explicit hash-bound stored-extraction publication recovery; no recomputation")
    ap.add_argument("--upiq-native-admission", type=Path, help="reviewed tower UPIQ-380 Rev7 admission")
    ap.add_argument("--hdr-fleet-executor", type=Path, help="emit native Rev7 declare-views inputs; no quality target values")
    ap.add_argument("--hdr-fleet-keys", type=Path, help="frozen human HDR source memberships")
    ap.add_argument("--hdr-teacher", help="immutable HDRTEACH role; native E26 extraction")
    ap.add_argument("--hdr-path-map", help="FROM=TO prefix rewrite for teacher inputs staged on tower; hashes still bind every file")
    ap.add_argument("--hdr-ladder-cells", type=Path, help="sealed HDR chroma-ladder admission-r1/CELLS.parquet; native Rev7 views for kept chroma rungs")
    ap.add_argument("--hdr-ladder-roles", type=Path, help="consumer-roles-r1/ORIGIN_ROLES.json for the ladder cells")
    ap.add_argument("--hdr-ladder-root", type=Path, help="sealed hdr-full root holding views/ and artifacts/zenavif/; read in place")
    ap.add_argument("--build-commit", default="")
    ap.add_argument("--revision", type=int, choices=[4, 5, 7], default=5)
    ap.add_argument("--era", default="")
    ap.add_argument("--out", default="/var/tmp/rev5-featbank")
    ap.add_argument("--reference", default="/var/tmp/rev4-featbank-r4")
    ap.add_argument("--chunk", type=int, default=25000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tier", default="native")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "extract":
        if not (a.bin and a.build_commit and a.era):
            ap.error("extract needs --bin, --build-commit and --era")
        if a.palette_verify:
            if not a.palette_instrument_manifest_sha256:
                ap.error("palette verification needs a frozen --palette-instrument-manifest-sha256")
            return cmd_palette_verify(a)
        if a.palette_views:
            return cmd_palette_views(a)
        if a.palette_chromaq:
            return cmd_palette_chromaq(a)
        if a.palette_instrument:
            return cmd_palette(a)
        if a.hdr_fleet_executor:
            if sum(bool(v) for v in (a.hdr_teacher, a.hdr_fleet_keys, a.upiq_native_admission, a.hdr_ladder_cells)) != 1:
                ap.error("native HDR planning requires exactly one teacher or human metadata source")
            return cmd_hdr_fleet_plan(a)
        return cmd_hdr_extract(a) if a.hdr_teacher else cmd_extract(a)
    return cmd_verify_against(a)


if __name__ == "__main__":
    sys.exit(main())
