"""E36 CLIC sets (record, 2026-11-03): CLIC 2021 validation as the confirmation set, CLIC 2024 as R7's training leg.

  clic.py keys SET ROOT                 label-free assessment keys for rev5_bank.py extract (SET: clic21_valid, clic24)
  clic.py leg ROOT OUT --bank BANK      R7: ROOT copied to OUT with CLIC 2024 appended as a hard-pair-preference leg
  clic.py confirm EVAL --bank BANK --predict BIN --model NAME=BAKE ... --vs BASE --arm LABEL=NAME@0,NAME@1,... ...
                                        confirmation read: two-alternative agreement per model and seed-mean paired
                                        differences against BASE (whole-reference cluster bootstrap)

`keys` reads only the `ref`, `a`, `b` and `split` columns of the library's pairs.tsv and writes one row per distinct
(reference, distorted) pair, in first-seen order, plus the instrument manifest that rev5_bank.py's assessment route
requires. `confirm` is the registered label read: it ledgers the rows it parses in EVAL/LABEL_READS.json before parsing
them, merges rows asking the same question (either order) by summing votes, as the M2 screen did, and scores each
question with the soft-label agreement p_a·[A ahead] + (1 - p_a)·[B ahead] + ½·[tie] (dataset_eval.pair_agreement,
the canonical owner, checked against this script's own sum).
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LIB = Path("/mnt/tower/input/iqa/datasets")
EXTRACTOR = Path("/var/tmp/byconstruction-bin/b77883be18db-rev7/extract_features_372col")
EXTRACTOR_SHA = "ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8"
# token -> (library id, split filter or None)
SETS = {"clic21_valid": ("clic-2021-perceptual", "valid"), "clic24": ("clic-2024-perceptual-image", None)}
KEY_COLUMNS = ("ref", "a", "b", "split")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pair_key(token, ref, dist):
    return f"{token}:{hashlib.sha256(f'{ref}\t{dist}'.encode()).hexdigest()}"


def question_rows(dataset, split):
    """(ref, a, b) per table row, reading only the key columns; rows outside the split are skipped unread."""
    with open(LIB / dataset / "pairs.tsv", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        head = next(reader)
        col = {c: head.index(c) for c in KEY_COLUMNS if c in head}
        for i, r in enumerate(reader):
            if split is not None and r[col["split"]] != split:
                continue
            yield i, r[col["ref"]], r[col["a"]], r[col["b"]]


def cmd_keys(a):
    dataset, split = SETS[a.set]
    root = Path(a.root) / "inputs" / a.set
    root.mkdir(parents=True, exist_ok=False)
    content = LIB / dataset / "content"
    seen = {}
    for _, ref, x, y in question_rows(dataset, split):
        for d in (x, y):
            seen.setdefault((ref, d), None)
    rows = list(seen)
    missing = [p for r in rows for p in r if not (content / p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} missing images, e.g. {missing[:3]}")
    keys = pa.table({"row_id": pa.array(range(len(rows)), pa.int64()),
                     "pair_key": [pair_key(a.set, r, d) for r, d in rows],
                     "ref_path": [str(content / r) for r, _ in rows],
                     "dist_path": [str(content / d) for _, d in rows],
                     "origin": [r for r, _ in rows]})
    assert len(set(keys.column("pair_key").to_pylist())) == keys.num_rows
    kp = root / "keys.parquet"
    pq.write_table(keys, kp, compression="zstd")
    if sha(EXTRACTOR) != EXTRACTOR_SHA:
        raise SystemExit("extractor changed")
    inst = dict(schema="rev5-assessment-keys-v1", labels_read=False, rows=keys.num_rows,
                keys=dict(path=str(kp), sha256=sha(kp)), extractor=dict(path=str(EXTRACTOR), sha256=EXTRACTOR_SHA),
                immutable_roots=[str(content), str(root)],
                instrument=dict(name=a.set, source=dict(dataset=dataset, table_sha256=sha(LIB / dataset / "pairs.tsv"),
                                                        columns_read=list(KEY_COLUMNS), split=split),
                                selection="distinct (reference, distorted) pairs of the registered rows; no label "
                                          "column read"),
                exposure_freeze=dict(state="features-only", labels_read=False,
                                     scope="zensimx record 2026-11-03: confirmation read 1 / R7 registrations"))
    (root / "instrument.json").write_text(json.dumps(inst, indent=2) + "\n")
    print(json.dumps(dict(set=a.set, pairs=keys.num_rows, references=len({r for r, _ in rows}))))


def ledger(path, entry):
    v = json.loads(path.read_text()) if path.exists() else dict(schema="e36-clic-label-reads-v1", reads=[])
    v["reads"].append(entry)
    path.write_text(json.dumps(v, indent=1) + "\n")


def cmd_confirm(a):
    import dataset_eval as de
    out = Path(a.eval)
    out.mkdir(parents=True, exist_ok=True)
    dataset, split = SETS["clic21_valid"]
    table = LIB / dataset / "pairs.tsv"
    reads = out / "LABEL_READS.json"
    if not reads.exists():
        ledger(reads, dict(table=str(table), sha256=sha(table), rows=f"split == {split}",
                           columns=["ref", "a", "b", "n_a", "n_b", "n_tie", "split"], state="authorized-before-read",
                           registration="zensimx benchmarks/e36_subset_test_2026-11-01.md, confirmation read 1"))
        merged = {}
        with open(table, newline="") as f:
            for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
                if r["split"] != split:
                    continue
                na, nb, nt = (float(r[c] or 0) for c in ("n_a", "n_b", "n_tie"))
                x, y = r["a"], r["b"]
                if x > y:
                    x, y, na, nb = y, x, nb, na
                m = merged.setdefault((r["ref"], x, y), dict(n_a=0.0, n_b=0.0, n_tie=0.0, rows=[]))
                m["n_a"] += na; m["n_b"] += nb; m["n_tie"] += nt; m["rows"].append(i)
        rows = []
        for (ref, x, y), m in merged.items():
            n = m["n_a"] + m["n_b"] + m["n_tie"]
            if n > 0:
                rows.append(dict(collection="clic-2021-valid", kind="pair", group=ref,
                                 a=pair_key("clic21_valid", ref, x), b=pair_key("clic21_valid", ref, y),
                                 target=(m["n_a"] + 0.5 * m["n_tie"]) / n, n=n, source_rows=m["rows"]))
        (out / "eval_rows.json").write_text(json.dumps(rows) + "\n")
        ledger(reads, dict(state="read", questions=len(rows), references=len({r["group"] for r in rows})))
        f = pq.read_table(Path(a.bank) / "features.parquet")
        f = f.drop(["row_id"]).rename_columns(["ref_basename" if c == "pair_key" else c for c in f.drop(["row_id"]).column_names])
        f = f.append_column("human_score", pa.array(np.zeros(f.num_rows)))
        pq.write_table(f, out / "features.parquet", compression="zstd")
        (out / "judge_pairs.tsv").write_text("ref_path\tdist_path\tpair_key\n")
    rows = json.loads((out / "eval_rows.json").read_text())
    for spec in a.model:
        name, bake = spec.split("=", 1)
        if not (out / f"pred_{name}.tsv").exists():
            subprocess.run([a.predict, "predict", "--bake", bake, "--corpus", str(out / "features.parquet"),
                            "--score-units", "--out", str(out / f"pred_{name}.tsv")], check=True)
    scores = de.load_scores(out)
    groups = sorted({r["group"] for r in rows})
    gi = {g: i for i, g in enumerate(groups)}
    g = np.array([gi[r["group"]] for r in rows])
    t = np.array([r["target"] for r in rows])
    per = {}
    res = dict(questions=len(rows), references=len(groups), models={}, arms={})
    for name, s in scores.items():
        sa = np.array([s[r["a"]] for r in rows]); sb = np.array([s[r["b"]] for r in rows])
        agree = np.where(sa == sb, 0.5, t * (sa > sb) + (1 - t) * (sa < sb))
        per[name] = np.bincount(g, agree, len(groups))
        acc, lo, hi, n = de.pair_agreement(rows, s)
        if abs(acc - agree.mean()) > 1e-9:
            raise ValueError(f"{name}: agreement {agree.mean()} disagrees with the canonical owner's {acc}")
        res["models"][name] = dict(agreement=acc, lo=lo, hi=hi, pairs=n)
    den = np.bincount(g, minlength=len(groups)).astype(float)
    rng = np.random.default_rng(20261103)
    draws = np.stack([np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
                      for _ in range(a.draws)])
    for spec in a.arm:
        spec, _, vs = spec.partition(";vs=")
        label, members = spec.split("=", 1)
        pairs = []
        for m in members.split(","):
            arm, seed = m.split("@")
            pairs.append((arm, (vs or a.vs).replace("@", seed)))
        diff = np.mean([(per[x] - per[y]) for x, y in pairs], axis=0)
        point = diff.sum() / den.sum()
        boots = (draws @ diff) / (draws @ den)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        reading = "supported" if lo > 0 else "contradicted" if hi < 0 else "neutral"
        res["arms"][label] = dict(diff=float(point), lo=float(lo), hi=float(hi), reading=reading,
                                  pairs=[list(p) for p in pairs])
    (out / "CONFIRM.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"{res['questions']} questions, {res['references']} references")
    for name, v in sorted(res["models"].items()):
        print(f"{name:40s} {v['agreement']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]")
    for label, v in res["arms"].items():
        print(f"{label:40s} {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] {v['reading']}")


D4_PARENTS = Path("/mnt/v/zensim/d4-roles-private/followup/clic-2024-perceptual-image-source-disposition.tsv")


def cmd_leg(a):
    """R7: a copy of a frozen training root with CLIC 2024 appended as a hard-pair-preference leg (the BAPPS leg's
    form), then re-frozen. Inherited legs and tables are copied unchanged; only datasets/receipt.json, the new leg's
    files and wide/frozen.json's bindings change."""
    import collections
    import shutil
    from ring_root import bank_transport, write
    source, out, bank = Path(a.root), Path(a.out), Path(a.bank)
    if out.exists():
        raise SystemExit("fresh root destination required")
    dataset = "clic-2024-perceptual-image"
    table = LIB / dataset / "pairs.tsv"
    manifest = json.loads((bank / "_MANIFEST.json").read_text())
    if (manifest["feature_set_id"] != "basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af"
            or manifest["build_commit"] != "b77883be18db5933a9db4b9e2670a2ba88a9729b"
            or sha(bank / "features.parquet") != manifest["features_parquet_sha256"]):
        raise SystemExit("CLIC 2024 bank producer/hash differs")
    parents = {}
    with open(D4_PARENTS, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            parents[r["source_ids"]] = r["original_parents"]
    shutil.copytree(source, out)
    legdir = out / "datasets"
    ledger = out.parent / f"{out.name}.LABEL_READS.json"
    ledger_entry = dict(table=str(table), sha256=sha(table), rows="all", columns=["ref", "a", "b", "n_a", "n_b", "n_tie"],
                        state="authorized-before-read", role="TRAIN",
                        registration="zensimx benchmarks/e36_subset_test_2026-11-01.md, R7 registration")
    v = dict(schema="e36-clic-label-reads-v1", reads=[ledger_entry])
    ledger.write_text(json.dumps(v, indent=1) + "\n")
    merged = {}
    with open(table, newline="") as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
            na, nb, nt = (float(r[c] or 0) for c in ("n_a", "n_b", "n_tie"))
            x, y = r["a"], r["b"]
            if x > y:
                x, y, na, nb = y, x, nb, na
            m = merged.setdefault((r["ref"], x, y), dict(n_a=0.0, n_b=0.0, n_tie=0.0, rows=[]))
            m["n_a"] += na; m["n_b"] += nb; m["n_tie"] += nt; m["rows"].append(i)
    v["reads"][0]["state"] = "read"
    v["reads"][0]["questions_after_merge"] = len(merged)
    ledger.write_text(json.dumps(v, indent=1) + "\n")
    obs, dropped = [], collections.Counter()
    for (ref, x, y), m in merged.items():
        n = m["n_a"] + m["n_b"] + m["n_tie"]
        if n <= 0:
            dropped["no_votes"] += 1
            continue
        p = (m["n_a"] + 0.5 * m["n_tie"]) / n
        if p == 0.5:
            dropped["exact_probability_tie"] += 1
            continue
        identity = f"{dataset}:pairs.tsv:{m['rows'][0]}"
        family = "clic24:" + parents[Path(ref).name]
        for side, d in (("a", x), ("b", y)):
            obs.append(dict(pair_key=pair_key("clic24", ref, d), ref_basename=identity,
                            human_score=float((p > 0.5) == (side == "a")), observation=f"{identity}:{side}", family=family))
    guard = json.loads((legdir / "FAMILY_GUARD.json").read_text())
    if {o["family"] for o in obs} & set(guard["protected_families"]):
        raise SystemExit("a CLIC 2024 family is protected")
    tab = pq.read_table(bank / "features.parquet")
    index = {k: i for i, k in enumerate(tab["pair_key"].to_pylist())}
    tab = tab.take(pa.array([index[o["pair_key"]] for o in obs]))
    width = json.loads((out / "wide/main/real/receipt.json").read_text())["width"]
    tab = bank_transport(tab, [o["ref_basename"] for o in obs], [o["human_score"] for o in obs], width)
    path = legdir / "clic-2024-perceptual-image.parquet"
    write(tab, path)
    counts = dict(sorted(collections.Counter(o["ref_basename"] for o in obs).items()))
    keys = [dict(row=i, observation=o["observation"], source=o["ref_basename"], reference=o["ref_basename"],
                 family=o["family"], role="fit", tier="T2") for i, o in enumerate(obs)]
    kp = legdir / "clic-2024-perceptual-image.keys.json"
    kp.write_text(json.dumps(keys, separators=(",", ":")) + "\n")
    packet = json.loads((legdir / "receipt.json").read_text())
    selected = packet["feature_ids"]
    decl = dict(schema="subjective-training-leg-v2", loss="hard-pair-preference", nominal_weight=0.5,
                target_orientation="quality", target_transform="orient-native-quality", target_column="human_score",
                selection="fixed-final-epoch", formula_revision=7, tier="v3", feature_ids=selected,
                producer_commit=manifest["build_commit"], feature_set_id=manifest["feature_set_id"], native_role="TRAIN",
                rows=len(obs), table_sha256=sha(path), admission_root=str(out), training_width=width,
                membership_sha256=sha(Path(a.bank).parents[1] / "inputs/clic24/keys.parquet"),
                feature_manifest_sha256=sha(bank / "_MANIFEST.json"),
                label_reads_path=str(ledger), family_source=str(D4_PARENTS), family_source_sha256=sha(D4_PARENTS))
    dp = Path(str(path) + ".manifest.json")
    dp.write_text(json.dumps(decl, indent=1) + "\n")
    weight = 0.5 / (sum(1 - 1 / c for c in counts.values()) / len(counts))
    packet["groups"].append(dict(
        name="subjective_rank_clic_2024_perceptual_image", path=str(path), train_weight=weight, val_weight=0.0,
        mode="withinref,rank", subjective=True, table_sha256=sha(path), manifest_sha256=sha(dp), keys_path=str(kp),
        keys_sha256=sha(kp), rows=len(obs), source_counts=counts, collection=dataset, membership_rows=len(merged),
        dropped=dict(dropped), groups=len(counts), pairs=sum(c * (c - 1) // 2 for c in counts.values())))
    packet["r7"] = dict(parent_frozen_sha256=sha(source / "wide/frozen.json"), parent_receipt_sha256=sha(source / "datasets/receipt.json"),
                        label_reads_path=str(ledger), registration="zensimx benchmarks/e36_subset_test_2026-11-01.md, R7")
    packet["family_guard_path"] = str(legdir / "FAMILY_GUARD.json")
    for g in packet["groups"][:-1]:  # inherited legs now live in the copy
        for k in ("path", "keys_path"):
            if k in g:
                g[k] = str(out / Path(g[k]).relative_to(source))
    (legdir / "receipt.json").write_text(json.dumps(packet, indent=1) + "\n")
    frozen = json.loads((out / "wide/frozen.json").read_text())
    frozen["dataset_legs_sha256"] = sha(legdir / "receipt.json")
    frozen["auxiliary_files"] = {rel: sha(out / rel) for rel in frozen["auxiliary_files"]}
    frozen["auxiliary_files"].update({str(p.relative_to(out)): sha(p) for p in legdir.iterdir()})
    (out / "wide/frozen.json").write_text(json.dumps(frozen, indent=1) + "\n")
    print(json.dumps(dict(root=str(out), questions=len(merged), comparisons=len(counts), rows=len(obs),
                          dropped=dict(dropped), families=len({o["family"] for o in obs}), train_weight=weight)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keys"); k.add_argument("set", choices=sorted(SETS)); k.add_argument("root")
    lg = sub.add_parser("leg"); lg.add_argument("root"); lg.add_argument("out"); lg.add_argument("--bank", required=True)
    c = sub.add_parser("confirm"); c.add_argument("eval"); c.add_argument("--bank", required=True)
    c.add_argument("--predict", required=True); c.add_argument("--model", action="append", default=[])
    c.add_argument("--vs", required=True, help="baseline name with @ for the seed, e.g. p30L_s@")
    c.add_argument("--arm", action="append", default=[],
                   help="LABEL=NAME@SEED,...[;vs=BASE_s@] (each seed against --vs, or the arm's own baseline)")
    c.add_argument("--draws", type=int, default=10000)
    a = ap.parse_args()
    {"keys": cmd_keys, "confirm": cmd_confirm, "leg": cmd_leg}[a.cmd](a)


if __name__ == "__main__":
    main()
