"""D1's receipt-bound production population; historical SOURCE_ORDER stays intact."""
import json
from pathlib import Path

from v2_common import SOURCES, sha

PRODUCTION_SOURCES = ("kadid", "tid2013", "konfig", "cid22_a25")
LEDGER_COMMIT = "1d3bf35a78a149f0925029d71014c6df8535f550"
MEMBER_SOURCE = {member: source for source in PRODUCTION_SOURCES for member in SOURCES[source]}

# D6 admits one named split and extraction, independently of D1's four-source
# receipt. These are content bindings, never permission for the AIC family.
D6_INPUT_PINS = {
    "source_table_sha256": "a2969c70f2e1c96f18348b0b8e66a28e41ee87ad839636e8329e63408b138b71",
    "pairs_sha256": "8ce5a1e913d3c4a9c10848560125b3449542301921d5e9316bfe14b1bb854d66",
    "assignment_sha256": "d67be7b4a5d09dac07ca9ba30ae2acbcfc13bac9369ccd61290f509ab869f267",
}


def aic_public_train_declaration(declaration, keys, *, fit=True):
    """D6's exact public_train leg; metadata and label-free keys only.

    Callers still hash the table after every group's metadata passes. D1's
    decision, human and bank allowlists remain unchanged.
    """
    d = declaration
    if (d.get("schema") != "aic2026-public-train-rev7-v1" or d.get("ruling") != "D6"
            or d.get("split") != "public_train" or d.get("role") != "TRAIN"
            or d.get("use") != ("fit" if fit else "out-of-fold-score")
            or d.get("formula_revision") != 7
            or d.get("feature_set_id") != "basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af"
            or d.get("target_column") != "human_score" or d.get("target_transform") != "negate-JND_score"
            or any(d.get(k) != v for k, v in D6_INPUT_PINS.items())):
        raise ValueError("D6 admits exactly revision-7 AIC2026 public_train")
    omitted = d.get("omitted_source_fold")
    if omitted is not None and (type(omitted) is not int or not 0 <= omitted < 7):
        raise ValueError("D6 source fold differs")
    from lib.assessment_identity import safe_path
    assignment_path = safe_path(d["assignment_path"])
    if sha(assignment_path) != D6_INPUT_PINS["assignment_sha256"]:
        raise ValueError("D6 assignment pin differs")
    assignment = json.loads(assignment_path.read_text())["assignment"]
    if not keys or len(keys) != d.get("rows"):
        raise ValueError("D6 ordered source-key coverage differs")
    seen, sources = set(), set()
    for i, k in enumerate(keys):
        if (set(k) != {"row", "key", "source", "split", "source_fold"}
                or k["row"] != i or k["split"] != "public_train"
                or not isinstance(k["key"], str) or not k["key"]
                or not isinstance(k["source"], str) or not k["source"]
                or k["key"] in seen or type(k["source_fold"]) is not int
                or not 0 <= k["source_fold"] < 7
                or assignment.get(k["source"]) != k["source_fold"]
                or (omitted is not None and (k["source_fold"] == omitted) == fit)):
            raise ValueError("D6 ordered source-key role/fold differs")
        seen.add(k["key"])
        sources.add(k["source"])
    expected_rows = 3920 if omitted is None else (3360 if fit else 560)
    expected_sources = 28 if omitted is None else (24 if fit else 4)
    if (len(keys) != expected_rows or len(sources) != expected_sources
            or sorted(sources) != d.get("sources")
            or any(sum(k["source"] == s for k in keys) != 140 for s in sources)):
        raise ValueError("D6 source-fold population differs")


def decision_record(path, receipt_sha):
    from lib.assessment_identity import safe_path
    if path is None:
        raise ValueError("PENDING SHIPPATH-human-production-role: receipt-bound D1 decision required")
    record = json.loads(safe_path(path).read_text())
    if (record.get("schema") != "shippath-human-role-decision-v1"
            or record.get("decision_id") != "SHIPPATH-human-production-role"
            or record.get("state") != "approved" or not record.get("decided_by")
            or record.get("allowed_use") != "qualified-recipe-training"
            or record.get("sources") != list(PRODUCTION_SOURCES)
            or record.get("ledger_commit") != LEDGER_COMMIT
            or not isinstance(receipt_sha, str) or len(receipt_sha) != 64
            or record.get("source_receipt_sha256") != receipt_sha):
        raise ValueError("PENDING SHIPPATH-human-production-role: D1 requires exactly four receipt-bound sources; AIC family forbidden")
    return record


def human_declaration(declaration, decision):
    sources = declaration.get("human_sources", [])
    if (decision is None or not sources or len(sources) != len(set(sources))
            or declaration.get("data_role_decision_required") != decision["decision_id"]
            or any(s not in PRODUCTION_SOURCES for s in sources)
            or declaration.get("source_receipt_sha256") != decision["source_receipt_sha256"]):
        raise ValueError("PENDING SHIPPATH-human-production-role: unapproved/AIC human table refused before payload access")


def human_keys(keys, declaration):
    if "member_set" not in keys.column_names:
        raise ValueError("strict human admission needs label-free member identity")
    members = set(keys["member_set"].to_pylist())
    if not members or not members <= MEMBER_SOURCE.keys():
        raise ValueError("strict human keys include an unapproved/AIC-family source")
    if {MEMBER_SOURCE[m] for m in members} != set(declaration["human_sources"]):
        raise ValueError("strict human source declaration differs from row keys")


def preflight_recipe(root, decision_path, heldout=None):
    """Validate D1 and every human declaration before wrapper payload checks."""
    from v2_common import load_frozen
    from lib.assessment_identity import safe_path
    if heldout is not None and heldout not in PRODUCTION_SOURCES:
        raise ValueError("D1/E30 forbid an AIC-family held-out fold")
    load_frozen(root, training_only=True, metadata_only=True)
    receipt = json.loads(safe_path(root / "wide/main/real/receipt.json").read_text())
    if "research_palette" in receipt:
        from e32_palette import CONTRACT
        if (receipt["research_palette"] != CONTRACT
                or receipt["research_palette"].get("serving_allowed") is not False):
            raise ValueError("E32 frozen research transport identity required before payload access")
    decision = decision_record(decision_path, receipt["admission_view"]["source_receipt_sha256"])
    view = receipt["admission_view"]
    if (view.get("source_frozen_sha256") != decision.get("source_frozen_sha256")
            or view.get("human_role_decision_sha256") != sha(decision_path)):
        raise ValueError("D1 decision/frozen receipt is not bound in this admission view")
    name = f"human_without_{heldout}" if heldout else "human_all"
    expected = [s for s in PRODUCTION_SOURCES if s != heldout]
    for split in ("fit", "dev"):
        rec = receipt["legs"][name][split]
        path = safe_path(root / rec["rel"])
        d = json.loads(safe_path(Path(f"{path}.manifest.json")).read_text())
        if sha(Path(f"{path}.manifest.json")) != rec["manifest_sha256"]:
            raise ValueError("human declaration changed after freeze")
        human_declaration(d, decision)
        bank_members(d)
        if ("research_palette" in d) != ("research_palette" in receipt):
            raise ValueError("E32 cannot mix projected and legacy legs")
        if "research_palette" in d:
            from e32_palette import admit_declaration
            projection = admit_declaration(d)
            if projection["role"] != {"fit":"D1-fit", "dev":"D1-development"}[split]:
                raise ValueError("E32 human fit/development split differs from its role")
        import pyarrow.parquet as pq
        from v2_teacher import key_path, row_keys_sha
        kp = safe_path(key_path(path))
        if sha(kp) != d["keys_sha256"]:
            raise ValueError("D1 human row-key pin changed")
        keys = pq.read_table(kp)
        if "research_palette" in d:
            from e32_palette import admit_keys
            admit_keys(path, d, keys)
        human_keys(keys, d)
        if row_keys_sha(keys) != d["row_keys_sha256"]:
            raise ValueError("D1 human row-key order changed")
        if d["human_sources"] != expected:
            raise ValueError("D1 human leg must omit AIC and the held-out source exactly")
    # Include teacher and coverage declarations/keys in the same payload-free phase.
    from v2_common import TEACHERS
    other = [(receipt["legs"][t][split], "TRAIN-oracle-fit" if split == "fit" else "TRAIN-oracle-development")
             for t in TEACHERS for split in ("fit", "dev")]
    other.append(({"rel": "e15/coverage_pool.parquet"}, "TRAIN-ordinal"))
    for rec, expected_role in other:
        path = safe_path(root / rec["rel"])
        d = json.loads(safe_path(Path(f"{path}.manifest.json")).read_text())
        bank_members(d)
        if ("research_palette" in d) != ("research_palette" in receipt):
            raise ValueError("E32 cannot mix projected and legacy legs")
        if "research_palette" in d:
            from e32_palette import admit_declaration, admit_keys
            projection = admit_declaration(d)
            if projection["role"] != expected_role:
                raise ValueError("E32 teacher fit/development/ordinal split differs from its role")
            admit_keys(path, d, pq.read_table(safe_path(key_path(path))))
        if d.get("human_sources") or d.get("data_role_decision_required"):
            human_declaration(d, decision)
            human_keys(pq.read_table(safe_path(key_path(path))), d)
    if "hdr_consensus" in receipt["legs"]:
        from e29_consensus import admit_metadata
        for arm, rec in receipt["legs"]["hdr_consensus"].items():
            if arm not in ("hb4", "hc4"):
                raise ValueError("unregistered HDR consensus arm")
            path = safe_path(root / rec["fit"]["rel"])
            manifest = Path(f"{path}.manifest.json")
            if sha(manifest) != rec["fit"]["manifest_sha256"]:
                raise ValueError("E29 HDR manifest changed")
            admit_metadata(path, json.loads(manifest.read_text()))
    return decision


def bank_members(declaration):
    """Population allowlist shared by preparation and lower table admission."""
    from v2_common import TEACHERS
    allowed = {*MEMBER_SOURCE, *(t[0] for t in TEACHERS.values())}
    if not set(declaration.get("bank_manifest_sha256", {})) <= allowed:
        raise ValueError("D1 declaration includes an unapproved/AIC bank member")
