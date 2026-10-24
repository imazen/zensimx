"""Table route for the label-free damage-mixture test: score bakes that read research columns the pixel scorer does not
compute (the byc_globaff descriptor at f1867-f1890, the byc_orient orientation damage at f2301-f2336).

Exploration record: benchmarks/byconstruction_2026-10-22.md. For every image of the mixture manifest, the computed Rev5
vector comes from `score_pair_with_bake --dump-features-to` (B's served bake reads every id the fx1 family reads; the
ring form is whatever ZENSIM_RING_FORM selects) and the descriptor from `byc_globaff`. One wide table holds them in
manifest image order; `bake_dial_refit predict --score-units` scores it. The route is checked against the pixel route on
a bake that reads no research column (`check`), which must agree before any table score is reported. With `--orient`
the table also carries the orientation damage (width 2337; `--globaff` is then optional).

  build   --mixture DIR --bin-dir BIN --bake B.bin [--globaff BIN] [--orient BIN] --out DIR [--jobs 16]
  build-pairs --pairs PAIRS.tsv --bin-dir BIN --bake B.bin [--globaff BIN] [--orient BIN] --out DIR [--jobs 16]
          (any labelled pairs list: columns key, group, label, ref_path, dist_path; the table keeps them in file order
          as `key`, `ref_basename` = group and `human_score` = label)
  score   --mixture DIR --table DIR --bin-dir BIN --bake NAME=PATH [--bake ...] [--decl DECL.json]
  check   --mixture DIR --table DIR --pixel-scores scores_X.json --table-scores scores_Y.json
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

GLOBAFF = list(range(1867, 1891))
ORIENT = list(range(2301, 2337))


def images(mix: Path) -> list[tuple[str, str]]:
    man = json.loads((mix / "manifest.json").read_text())
    return sorted({(r["ref"], r["base"]) for r in man["rows"]} | {(r["ref"], r["more"]) for r in man["rows"]})


def build_table(pairs: list[tuple[str, str]], out: Path, a, groups: list[str], labels, keys: list[str]) -> None:
    """Dump the computed vector of every (ref, dist) pair, add the descriptor, write `table.parquet` + `table.json`."""
    out.mkdir(parents=True, exist_ok=False)
    dumps = out / "dumps"
    dumps.mkdir()

    def one(i):
        ref, dist = pairs[i]
        dest = dumps / f"{i:05d}.f32"
        subprocess.run([str(Path(a.bin_dir) / "score_pair_with_bake"), "--bake", a.bake, "--bake-post", "raw", "--ref",
                        ref, "--dist", dist, "--dump-features-to", str(dest)], check=True, capture_output=True)
        b = dest.read_bytes()
        n = int.from_bytes(b[:4], "little")
        return np.frombuffer(b[8:8 + 4 * n], dtype="<f4")

    with cf.ThreadPoolExecutor(a.jobs) as pool:
        vecs = list(pool.map(one, range(len(pairs))))
    n = {len(v) for v in vecs}
    if len(n) != 1 or next(iter(n)) < 720:
        raise ValueError(f"unexpected dump widths {n}")
    tsv = out / "pairs.tsv"
    tsv.write_text("ref_path\tdist_path\thuman_score\trow_id\n" +
                   "".join(f"{r}\t{d}\t0\t{i}\n" for i, (r, d) in enumerate(pairs)))
    if not (a.globaff or a.orient):
        raise SystemExit("give --globaff, --orient or both")
    sidecars = []  # (binary, csv name, column prefix, ids)
    if a.globaff:
        sidecars.append((a.globaff, "globaff.csv", "g", GLOBAFF))
    if a.orient:
        sidecars.append((a.orient, "orient.csv", "o", ORIENT))
    width = ORIENT[-1] + 1 if a.orient else GLOBAFF[-1] + 1
    X = np.full((len(pairs), width), np.nan, np.float32)
    k = next(iter(n))
    X[:, :min(k, 720)] = np.stack(vecs)[:, :min(k, 720)]
    for binary, name, prefix, ids in sidecars:
        dest = out / name
        subprocess.run([binary, "--pairs", str(tsv), "--out", str(dest)], check=True,
                       env=dict(os.environ, RAYON_NUM_THREADS=str(a.jobs)))
        g = pacsv.read_csv(dest)
        order = np.argsort(g.column("row_id").to_numpy())
        X[:, ids] = np.stack([g.column(f"{prefix}{i}").to_numpy()[order] for i in range(len(ids))],
                             axis=1).astype(np.float32)
    cols = {"key": pa.array(keys), "ref_basename": pa.array(groups),
            "human_score": pa.array(np.asarray(labels, dtype=np.float64))}
    cols.update({f"f{i}": pa.array(X[:, i]) for i in range(width)})
    pq.write_table(pa.table(cols), out / "table.parquet", compression="zstd")
    meta = {"images": keys, "ring_form": os.environ.get("ZENSIM_RING_FORM", "raw"), "dump_bake": a.bake,
            "dump_bake_sha256": hashlib.sha256(Path(a.bake).read_bytes()).hexdigest(), "dump_width": k}
    for binary, name, _, _ in sidecars:
        meta[f"{name.removesuffix('.csv')}_sha256"] = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    (out / "table.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(pairs)} images, dump width {k}")


def cmd_build(a) -> int:
    mix = Path(a.mixture)
    pairs = images(mix)
    build_table([(str(mix / "img" / r), str(mix / "img" / d)) for r, d in pairs], Path(a.out), a,
                [r for r, _ in pairs], np.zeros(len(pairs)), [d for _, d in pairs])
    return 0


def cmd_build_pairs(a) -> int:
    rows = pacsv.read_csv(a.pairs, parse_options=pacsv.ParseOptions(delimiter="\t")).to_pylist()
    build_table([(r["ref_path"], r["dist_path"]) for r in rows], Path(a.out), a, [str(r["group"]) for r in rows],
                [float(r["label"]) for r in rows], [str(r["key"]) for r in rows])
    return 0


def cmd_score(a) -> int:
    mix, table = Path(a.mixture), Path(a.table)
    meta = json.loads((table / "table.json").read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
    os.environ.setdefault("REV4_V2_BIN_DIR", a.bin_dir)
    from v2_lodo_mlp import predict  # noqa: E402
    for spec in a.bake:
        name, path = spec.split("=", 1)
        dest = mix / f"scores_{name}.json"
        if dest.exists():
            print(f"exists: {dest}")
            continue
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as td:
            if a.decl:  # research bake: zenpredict forward (byc_predict), same row layout as the trainer
                subprocess.run([str(Path(a.bin_dir) / "byc_predict"), "--bake", path, "--decl", a.decl, "--corpus",
                                str(table / "table.parquet"), "--out", str(Path(td) / "pred.tsv")],
                               check=True, capture_output=True)
                import pandas as pd
                pred = pd.read_csv(Path(td) / "pred.tsv", sep="\t").pred.to_numpy(dtype=np.float64)
            else:
                pred = predict(Path(path), table / "table.parquet", Path(td) / "pred.tsv")
        scores = dict(zip(meta["images"], (float(round(p, 6)) for p in pred)))
        dest.write_text(json.dumps({"bake": path, "post": "table-score-units", "route": "table",
                                    "ring_form": meta["ring_form"], "table": str(table),
                                    "bake_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                    "scores": scores}, indent=1) + "\n")
        print(f"{name}: {len(scores)} images scored (table route)")
    return 0


def cmd_check(a) -> int:
    p = json.loads(Path(a.pixel_scores).read_text())["scores"]
    t = json.loads(Path(a.table_scores).read_text())["scores"]
    d = np.array([abs(p[k] - t[k]) for k in p])
    print(json.dumps({"images": len(d), "max_abs_diff": float(d.max()), "identical": int((d == 0).sum())}))
    return 0 if d.max() <= a.tol else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--mixture", required=True)
    b.add_argument("--bin-dir", required=True)
    b.add_argument("--bake", required=True)
    b.add_argument("--globaff")
    b.add_argument("--orient")
    b.add_argument("--out", required=True)
    b.add_argument("--jobs", type=int, default=16)
    b.set_defaults(fn=cmd_build)
    bp = sub.add_parser("build-pairs")
    bp.add_argument("--pairs", required=True)
    bp.add_argument("--bin-dir", required=True)
    bp.add_argument("--bake", required=True)
    bp.add_argument("--globaff")
    bp.add_argument("--orient")
    bp.add_argument("--out", required=True)
    bp.add_argument("--jobs", type=int, default=16)
    bp.set_defaults(fn=cmd_build_pairs)
    s = sub.add_parser("score")
    s.add_argument("--mixture", required=True)
    s.add_argument("--table", required=True)
    s.add_argument("--bin-dir", required=True)
    s.add_argument("--bake", action="append", required=True)
    s.add_argument("--decl", help="fx1-family declaration: forward compact research bakes through byc_predict")
    s.set_defaults(fn=cmd_score)
    c = sub.add_parser("check")
    c.add_argument("--mixture", required=True)
    c.add_argument("--table", required=True)
    c.add_argument("--pixel-scores", required=True)
    c.add_argument("--table-scores", required=True)
    c.add_argument("--tol", type=float, default=1e-4)
    c.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
