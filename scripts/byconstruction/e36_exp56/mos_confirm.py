"""E36 confirmation read of a library MOS set (record, R10 amendment 4): MMSPG JPEG XR, the one compression-centric
human set in the tower library that is in no training leg, not in EVAL, not reserved and shares no source with training.

  mos_confirm.py keys SET ROOT --labels LABELS.tsv --decode-map MAP.tsv --png-dir DIR
                                    label-free assessment keys: every (ref, item) of labels.tsv, pixels = the M2 trust
                                    screen's zenbitmaps decodes (MAP.tsv: input, png, width, height, rgb8_sha256), found
                                    by basename under DIR
  mos_confirm.py confirm EVAL --labels LABELS.tsv --bank BANK --predict BIN --model NAME=BAKE ... --vs BASE_s@
                 --arm LABEL=NAME@0,NAME@1,... --registration TEXT
                                    the registered read: SROCC with a whole-reference interval, within-reference SROCC,
                                    error (median absolute residual around the 4PL fit over the target's 1-99% span, as
                                    the EVAL readout), and seed-mean paired differences against BASE with a
                                    whole-reference bootstrap (few references: the interval is reported, not trusted)

`keys` reads only `item` and `ref`. `confirm` ledgers its read in EVAL/LABEL_READS.json before parsing `score`.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clic import EXTRACTOR, EXTRACTOR_SHA, ledger, pair_key, sha  # noqa: E402

KEY_COLUMNS = ("item", "ref")


def key_rows(labels):
    with open(labels, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        head = next(reader)
        col = {c: head.index(c) for c in KEY_COLUMNS}
        for i, r in enumerate(reader):
            yield i, r[col["ref"]], r[col["item"]]


def png_map(decode_map, png_dir):
    out = {}
    with open(decode_map, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            p = Path(png_dir) / Path(r["png"]).name
            out[Path(r["input"]).as_posix()] = p
    return out


def cmd_keys(a):
    root = Path(a.root) / "inputs" / a.set
    root.mkdir(parents=True, exist_ok=False)
    pngs = png_map(a.decode_map, a.png_dir)
    content = Path(a.labels).parent / "content"

    def png(name):
        p = pngs[(content / name).as_posix()]
        if not p.is_file():
            raise SystemExit(f"missing decode {p}")
        return p

    rows = [(ref, item) for _, ref, item in key_rows(a.labels)]
    assert len(set(rows)) == len(rows)
    keys = pa.table({"row_id": pa.array(range(len(rows)), pa.int64()),
                     "pair_key": [pair_key(a.set, r, d) for r, d in rows],
                     "ref_path": [str(png(r)) for r, _ in rows],
                     "dist_path": [str(png(d)) for _, d in rows],
                     "origin": [r for r, _ in rows]})
    kp = root / "keys.parquet"
    pq.write_table(keys, kp, compression="zstd")
    if sha(EXTRACTOR) != EXTRACTOR_SHA:
        raise SystemExit("extractor changed")
    inst = dict(schema="rev5-assessment-keys-v1", labels_read=False, rows=keys.num_rows,
                keys=dict(path=str(kp), sha256=sha(kp)), extractor=dict(path=str(EXTRACTOR), sha256=EXTRACTOR_SHA),
                immutable_roots=[str(Path(a.png_dir)), str(root)],
                instrument=dict(name=a.set, source=dict(table=str(a.labels), table_sha256=sha(a.labels),
                                                        columns_read=list(KEY_COLUMNS), split=None,
                                                        pixels=dict(decode_map=str(a.decode_map), sha256=sha(a.decode_map))),
                                selection="every row of the label table (reference rows included, as the M2 screen "
                                          "scored them); no label column read"),
                exposure_freeze=dict(state="features-only", labels_read=False,
                                     scope="zensimx record 2026-11-03: R10 amendment 4 (library MOS confirmation read)"))
    (root / "instrument.json").write_text(json.dumps(inst, indent=2) + "\n")
    print(json.dumps(dict(set=a.set, pairs=keys.num_rows, references=len({r for r, _ in rows}))))


def error_stat(pred, y):
    from aic_panel import logistic_fit
    span = float(np.quantile(y, 0.99) - np.quantile(y, 0.01))
    res = np.abs(np.asarray(logistic_fit(pred, y)) - y)
    return float(np.median(res) / span)


def srocc(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def cmd_confirm(a):
    import subprocess
    import dataset_eval as de
    out = Path(a.eval)
    out.mkdir(parents=True, exist_ok=True)
    reads = out / "LABEL_READS.json"
    if not reads.exists():
        ledger(reads, dict(table=str(a.labels), sha256=sha(a.labels), rows="every row",
                           columns=["item", "ref", "score", "higher_is_better"], state="authorized-before-read",
                           registration=f"zensimx benchmarks/e36_subset_test_2026-11-01.md, {a.registration}"))
        rows = []
        with open(a.labels, newline="") as f:
            for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
                sign = 1.0 if r["higher_is_better"] in ("1", "true", "True") else -1.0
                rows.append(dict(collection=a.set, kind="mos", group=r["ref"], a=pair_key(a.set, r["ref"], r["item"]),
                                 target=sign * float(r["score"]), source_rows=[i]))
        (out / "eval_rows.json").write_text(json.dumps(rows) + "\n")
        ledger(reads, dict(state="read", rows=len(rows), references=len({r["group"] for r in rows})))
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
    y = np.array([r["target"] for r in rows])
    res = dict(rows=len(rows), references=len(groups), models={}, arms={})
    P = {}
    for name, s in scores.items():
        p = np.array([s[r["a"]] for r in rows])
        P[name] = p
        (iv, n) = de.mos_interval(rows, s)
        res["models"][name] = dict(srocc=iv["srocc"] if isinstance(iv, dict) and "srocc" in iv else iv,
                                   within=de.within(rows, s), error=error_stat(p, y), rows=n)
    rng = np.random.default_rng(20261103)
    draws = [np.concatenate([np.flatnonzero(g == k) for k in rng.integers(0, len(groups), len(groups))])
             for _ in range(a.draws)]
    for spec in a.arm:
        label, members = spec.split("=", 1)
        pairs = []
        for m in members.split(","):
            arm, seed = m.split("@")
            pairs.append((arm, a.vs.replace("@", seed)))
        d_s = np.mean([srocc(P[x], y) - srocc(P[b], y) for x, b in pairs])
        d_e = np.mean([error_stat(P[x], y) - error_stat(P[b], y) for x, b in pairs])
        boots = np.array([np.mean([srocc(P[x][d], y[d]) - srocc(P[b][d], y[d]) for x, b in pairs]) for d in draws])
        lo, hi = np.quantile(boots, [0.025, 0.975])
        reading = "supported" if lo > 0 else "contradicted" if hi < 0 else "neutral"
        res["arms"][label] = dict(srocc_diff=float(d_s), srocc_lo=float(lo), srocc_hi=float(hi), error_diff=float(d_e),
                                  reading=reading, pairs=[list(p) for p in pairs],
                                  note="whole-reference bootstrap over few references; reported, not trusted")
    (out / "CONFIRM.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"== {a.set}: {res['rows']} rows, {res['references']} references")
    for name, v in sorted(res["models"].items()):
        print(f"{name:40s} SROCC {v['srocc']} within {v['within']:.4f} error {v['error'] * 100:.2f}%")
    for label, v in res["arms"].items():
        print(f"{label:40s} dSROCC {v['srocc_diff']:+.4f} [{v['srocc_lo']:+.4f}, {v['srocc_hi']:+.4f}] "
              f"derror {v['error_diff'] * 100:+.2f} points {v['reading']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keys"); k.add_argument("set"); k.add_argument("root")
    k.add_argument("--labels", required=True); k.add_argument("--decode-map", required=True); k.add_argument("--png-dir", required=True)
    c = sub.add_parser("confirm"); c.add_argument("eval"); c.add_argument("--set", required=True)
    c.add_argument("--labels", required=True); c.add_argument("--bank", required=True)
    c.add_argument("--predict", required=True); c.add_argument("--model", action="append", default=[])
    c.add_argument("--vs", required=True); c.add_argument("--arm", action="append", default=[])
    c.add_argument("--draws", type=int, default=10000); c.add_argument("--registration", required=True)
    a = ap.parse_args()
    {"keys": cmd_keys, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    main()
