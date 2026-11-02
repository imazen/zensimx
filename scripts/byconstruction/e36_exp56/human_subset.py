"""Experiment 5 (E36 record, MOS transfer): a strict-admissible subset of the design-released human table.

Keeps the rows of the named production sources (member_set -> source) in their original order, writes the table, its
keys and a manifest derived from the parent's with the subset's own hashes, human_sources and row selection, so every
strict admission check (decision, keys, table and row-key pins) still runs against the subset."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from v2_human_role import MEMBER_SOURCE, PRODUCTION_SOURCES
from v2_teacher import row_keys_sha


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def subset(table: Path, sources, out_dir: Path, rows=None, label=None) -> Path:
    """`rows` (experiment 6): keep exactly these parent row indices (ascending) among the named sources' rows;
    `label` names the output."""
    sources = list(sources)
    if not sources or len(set(sources)) != len(sources) or not set(sources) <= set(PRODUCTION_SOURCES):
        raise ValueError(f"human subset needs distinct production sources, got {sources}")
    keys_path = table.with_suffix(".keys.parquet")
    parent = json.loads(Path(f"{table}.manifest.json").read_text())
    if parent["table_sha256"] != sha(table) or parent["keys_sha256"] != sha(keys_path):
        raise ValueError("parent human table or keys changed")
    t, keys = pq.read_table(table), pq.read_table(keys_path)
    if t.num_rows != keys.num_rows or t["ref_basename"].to_pylist() != keys["ref_basename"].to_pylist():
        raise ValueError("parent table and keys disagree")
    member = keys["member_set"].to_pylist()
    take = [i for i, m in enumerate(member) if MEMBER_SOURCE[m] in sources]
    if rows is not None:
        rows = [int(i) for i in rows]
        if rows != sorted(set(rows)) or not set(rows) <= set(take):
            raise ValueError("row list must be ascending, distinct and within the named sources")
        take = rows
    if not take:
        raise ValueError("empty human subset")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "human_" + (label or "_".join(sorted(sources))) + "_fit"
    out, out_keys = out_dir / f"{name}.parquet", out_dir / f"{name}.keys.parquet"
    pq.write_table(t.take(take), out, compression="zstd")
    sub_keys = keys.take(take)
    pq.write_table(sub_keys, out_keys, compression="zstd")
    m = dict(parent)
    # Curated binding (group_admission.rs role_from_receipts): the subset names its parent, which the recipe
    # receipt binds directly, by table and manifest hash.
    m.update(source_table_sha256=parent["table_sha256"], source_manifest_sha256=sha(f"{table}.manifest.json"),
             table_sha256=sha(out), keys_sha256=sha(out_keys), row_keys_sha256=row_keys_sha(sub_keys),
             human_sources=[s for s in PRODUCTION_SOURCES if s in sources], rows_kept=len(take),
             row_selection_sha256=hashlib.sha256(json.dumps(take).encode()).hexdigest(),
             experiment5_subset_of=dict(table_sha256=parent["table_sha256"], keys_sha256=parent["keys_sha256"],
                                        parent_source_table_sha256=parent.get("source_table_sha256"),
                                        parent_source_manifest_sha256=parent.get("source_manifest_sha256"),
                                        sources=sorted(sources), rows=len(take)))
    Path(f"{out}.manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    return out
