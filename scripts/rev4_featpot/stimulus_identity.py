"""Metadata-only native stimulus identities, including pixel-identical stimuli.

The pinned bank converter preserves source_row_id separately from pair_key.
Its source TSV puts image paths first and row_id last; response text between
them is never parsed. A representative bank filename cannot identify every
observation sharing its decoded pixels.
"""
from pathlib import Path
import re

from subjective_training_leg import document, safe, sha

PAIRS_SHA = "caf94cfc8b9bfad6ff2bb527dc07e49733a296c27c5a1e0d9d543ed513cce430"


def source_metadata(path, pin):
    path = safe(path)
    if sha(path) != pin:
        raise ValueError("stimulus source pin changed")
    rows = {}
    with path.open() as stream:
        if next(stream).rstrip("\r\n") != "ref_path\tdist_path\thuman_score\trow_id":
            raise ValueError("stimulus source schema changed")
        for line in stream:
            line = line.rstrip("\r\n")
            if line.count("\t") != 3:
                raise ValueError("stimulus source row shape changed")
            reference, distorted, _ = line.split("\t", 2)
            text = line.rsplit("\t", 1)[1]
            if not text.isascii() or not text.isdecimal() or str(int(text)) != text:
                raise ValueError("invalid stimulus source row ID")
            row_id = int(text)
            if row_id in rows or not reference or not distorted:
                raise ValueError("duplicate or empty stimulus identity")
            rows[row_id] = dict(ref_path=reference, dist_path=distorted)
    return rows


def promoted_stimuli(members):
    import pyarrow.parquet as pq
    from admit_bank import BANK, PINNED_MANIFEST_SHA
    banks, inputs, sources = {}, {}, {}
    for member in members:
        mp = BANK/member/"_MANIFEST.json"
        manifest = document(mp, PINNED_MANIFEST_SHA[member])
        inputs[str(mp)] = PINNED_MANIFEST_SHA[member]
        path = safe(manifest["source_files"]["pairs"])
        if manifest["source_sha256"]["pairs"] != PAIRS_SHA:
            raise ValueError("promoted stimulus provenance changed")
        if not sources:
            sources = source_metadata(path, PAIRS_SHA)
        inputs[str(path)] = PAIRS_SHA
        kp = BANK/member/"keys.parquet"
        pin = manifest["files"]["keys.parquet"]["sha256"]
        if sha(kp) != pin:
            raise ValueError("promoted stimulus keys changed")
        inputs[str(kp)] = pin
        columns = ["pair_key", "ref_group", "ref_path", "dist_path", "ref_pixels_sha256", "dist_pixels_sha256"]
        rows = pq.read_table(kp, columns=columns).to_pylist()
        banks[member] = {r["pair_key"]: r for r in rows}
        if len(banks[member]) != len(rows):
            raise ValueError("duplicate promoted pixel key")
    return banks, sources, inputs


def resolve(key, banks, sources):
    row_id = key["source_row_id"]
    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 0:
        raise ValueError("invalid fitting stimulus source row ID")
    bank = banks[key["member_set"]][key["pair_key"]]
    source = sources[row_id]
    reference = Path(source["ref_path"]).stem.lower()
    if (bank["ref_group"] != key["ref_basename"]
            or reference != Path(bank["ref_path"]).stem.lower()
            or reference != key["ref_basename"].split(":")[-1].lower()):
        raise ValueError("native/promoted/fitting reference identity differs")
    result = dict(source)
    if key["member_set"].startswith("kadid_"):
        match = re.fullmatch(r"(i[0-9]{2})_([0-9]{2})_([0-9]{2})", Path(source["dist_path"]).stem.lower())
        if (not match or match[1] != reference or not 1 <= int(match[2]) <= 25
                or not 1 <= int(match[3]) <= 5):
            raise ValueError("native KADID stimulus identity differs")
        result["codec"] = "kadid_" + match[2]
    return result
