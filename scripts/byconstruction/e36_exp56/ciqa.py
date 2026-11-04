"""E36 CIQA confirmation read (record, R10 amendment 3): CIQA's pairwise JPEG comparisons on its private recreation.

  ciqa.py keys ROOT                 label-free assessment keys for rev5_bank.py extract
  ciqa.py confirm EVAL --bank BANK --predict BIN --model NAME=BAKE ... --vs BASE --arm LABEL=NAME@0,NAME@1,... ...
                                    the registered read: two-alternative agreement per model and seed-mean paired
                                    differences against BASE (whole-reference cluster bootstrap), all rows and test split

Pixels: the references are TensorFlow's INTEGER_FAST decodes of the AVA sources and the stimuli its decodes of the
recreated JPEGs, rebuilt by zenjpeg's coefficient decode with libjpeg-turbo's fast-integer reconstruction and kept
only where each image's SHA-256 equals the one the TensorFlow recreation recorded (`references-zenjpeg-ifast/`,
`stimuli-zenjpeg-ifast/` under the private CIQA root). `keys` reads only the library pairs.tsv's `ref`, `a`, `b` and
`split` columns. `confirm` ledgers its read in EVAL/LABEL_READS.json before parsing `p_a`, keeps every published row
as its own question (the library already dropped the 16 exact repeats; questions asked in both orders stay, as the
library keeps them), and scores each with clic.py's soft-label agreement.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clic import EXTRACTOR, EXTRACTOR_SHA, ledger, pair_key, sha  # noqa: E402

TABLE = Path("/mnt/tower/input/iqa/datasets/ciqa-google/pairs.tsv")
PRIVATE = Path("/mnt/v/zensim/ciqa-private")
REFS, STIMULI = PRIVATE / "references-zenjpeg-ifast", PRIVATE / "stimuli-zenjpeg-ifast"
TOKEN = "ciqa"
KEY_COLUMNS = ("ref", "a", "b", "split")


def png(name):
    """The rebuilt image for a library name: <id>/ref.jpg -> references/<id>.png, <id>/<q>.jpg -> stimuli/<id>_<q>.png."""
    ref_id, leaf = name.split("/")
    return REFS / f"{ref_id}.png" if leaf == "ref.jpg" else STIMULI / f"{ref_id}_{leaf.removesuffix('.jpg')}.png"


def question_rows():
    with open(TABLE, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        head = next(reader)
        col = {c: head.index(c) for c in KEY_COLUMNS}
        for i, r in enumerate(reader):
            yield i, r[col["ref"]], r[col["a"]], r[col["b"]], r[col["split"]]


def cmd_keys(a):
    root = Path(a.root) / "inputs" / TOKEN
    root.mkdir(parents=True, exist_ok=False)
    seen = {}
    for _, ref, x, y, _ in question_rows():
        for d in (x, y):
            seen.setdefault((ref, d), None)
    rows = list(seen)
    missing = [str(png(p)) for r in rows for p in r if not png(p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} missing images, e.g. {missing[:3]}")
    keys = pa.table({"row_id": pa.array(range(len(rows)), pa.int64()),
                     "pair_key": [pair_key(TOKEN, r, d) for r, d in rows],
                     "ref_path": [str(png(r)) for r, _ in rows],
                     "dist_path": [str(png(d)) for _, d in rows],
                     "origin": [r.split("/")[0] for r, _ in rows]})
    assert len(set(keys.column("pair_key").to_pylist())) == keys.num_rows
    kp = root / "keys.parquet"
    pq.write_table(keys, kp, compression="zstd")
    if sha(EXTRACTOR) != EXTRACTOR_SHA:
        raise SystemExit("extractor changed")
    inst = dict(schema="rev5-assessment-keys-v1", labels_read=False, rows=keys.num_rows,
                keys=dict(path=str(kp), sha256=sha(kp)), extractor=dict(path=str(EXTRACTOR), sha256=EXTRACTOR_SHA),
                immutable_roots=[str(REFS), str(STIMULI), str(root)],
                instrument=dict(name=TOKEN, source=dict(dataset="ciqa-google", table_sha256=sha(TABLE),
                                                        columns_read=list(KEY_COLUMNS), split=None),
                                selection="distinct (reference, distorted) pairs of every published row; no label "
                                          "column read"),
                exposure_freeze=dict(state="features-only", labels_read=False,
                                     scope="zensimx record 2026-11-03: R10 amendment 3 (CIQA confirmation read)"))
    (root / "instrument.json").write_text(json.dumps(inst, indent=2) + "\n")
    print(json.dumps(dict(set=TOKEN, pairs=keys.num_rows, references=len({r for r, _ in rows}))))


def cmd_confirm(a):
    import subprocess
    import dataset_eval as de
    out = Path(a.eval)
    out.mkdir(parents=True, exist_ok=True)
    reads = out / "LABEL_READS.json"
    if not reads.exists():
        ledger(reads, dict(table=str(TABLE), sha256=sha(TABLE), rows="every row",
                           columns=["ref", "a", "b", "p_a", "split"], state="authorized-before-read",
                           registration=f"zensimx benchmarks/e36_subset_test_2026-11-01.md, {a.registration}"))
        rows = []
        with open(TABLE, newline="") as f:
            for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
                rows.append(dict(collection="ciqa", kind="pair", group=r["ref"].split("/")[0],
                                 a=pair_key(TOKEN, r["ref"], r["a"]), b=pair_key(TOKEN, r["ref"], r["b"]),
                                 target=float(r["p_a"]), split=r["split"], source_rows=[i]))
        (out / "eval_rows.json").write_text(json.dumps(rows) + "\n")
        ledger(reads, dict(state="read", questions=len(rows), references=len({r["group"] for r in rows})))
        f = pq.read_table(Path(a.bank) / "features.parquet")
        f = f.drop(["row_id"]).rename_columns(["ref_basename" if c == "pair_key" else c for c in f.drop(["row_id"]).column_names])
        f = f.append_column("human_score", pa.array(np.zeros(f.num_rows)))
        pq.write_table(f, out / "features.parquet", compression="zstd")
        (out / "judge_pairs.tsv").write_text("ref_path\tdist_path\tpair_key\n")
    rows_all = json.loads((out / "eval_rows.json").read_text())
    for spec in a.model:
        name, bake = spec.split("=", 1)
        if not (out / f"pred_{name}.tsv").exists():
            subprocess.run([a.predict, "predict", "--bake", bake, "--corpus", str(out / "features.parquet"),
                            "--score-units", "--out", str(out / f"pred_{name}.tsv")], check=True)
    scores = de.load_scores(out)
    res = {}
    for subset in ("all", "test"):
        rows = [r for r in rows_all if subset == "all" or r["split"] == subset]
        groups = sorted({r["group"] for r in rows})
        gi = {g: i for i, g in enumerate(groups)}
        g = np.array([gi[r["group"]] for r in rows])
        t = np.array([r["target"] for r in rows])
        per = {}
        part = dict(questions=len(rows), references=len(groups), models={}, arms={})
        for name, s in scores.items():
            sa = np.array([s[r["a"]] for r in rows]); sb = np.array([s[r["b"]] for r in rows])
            agree = np.where(sa == sb, 0.5, t * (sa > sb) + (1 - t) * (sa < sb))
            per[name] = np.bincount(g, agree, len(groups))
            acc, lo, hi, n = de.pair_agreement(rows, s)
            if abs(acc - agree.mean()) > 1e-9:
                raise ValueError(f"{name}: agreement {agree.mean()} disagrees with the canonical owner's {acc}")
            part["models"][name] = dict(agreement=acc, lo=lo, hi=hi, pairs=n)
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
            part["arms"][label] = dict(diff=float(point), lo=float(lo), hi=float(hi), reading=reading,
                                       pairs=[list(p) for p in pairs])
        res[subset] = part
        print(f"== {subset}: {part['questions']} questions, {part['references']} references")
        for name, v in sorted(part["models"].items()):
            print(f"{name:40s} {v['agreement']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]")
        for label, v in part["arms"].items():
            print(f"{label:40s} {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] {v['reading']}")
    (out / "CONFIRM.json").write_text(json.dumps(res, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keys"); k.add_argument("root")
    c = sub.add_parser("confirm"); c.add_argument("eval"); c.add_argument("--bank", required=True)
    c.add_argument("--predict", required=True); c.add_argument("--model", action="append", default=[])
    c.add_argument("--vs", required=True, help="baseline name with @ for the seed, e.g. control_s@")
    c.add_argument("--arm", action="append", default=[], help="LABEL=NAME@SEED,...[;vs=BASE_s@]")
    c.add_argument("--draws", type=int, default=10000)
    c.add_argument("--registration", required=True)
    a = ap.parse_args()
    {"keys": cmd_keys, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    main()
