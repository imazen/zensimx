"""DVIFM 0.3 (dvifmish preset `port-lap-v0.3`) on the four E21 populations and the label-free mixture set.

Exploration record: benchmarks/byconstruction_2026-10-22.md. The scorer is the frozen runtime of the 2026-10-20 AIC2026
run (`jnd-reconstruction-2026-10-20-dvifm03/dvifm-runtime`, dvifmish 31c4e256); its binary is SHA-256 checked against
that run's FROZEN.json before any score. The preset's constants were trained on AIC2026 by DVIFM's author, so the E21
sets (KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL, CID22-A 25 references) and the mixture set are unseen by it.

  pairs     OUT/pairs.tsv (unique ref/dist pairs, no labels) and OUT/rows.parquet (E21 row order with type and level)
  score     OUT/dvifm.tsv through the frozen binary (run under run-heavy)
  mixture   --mixture DIR: scores the mixture images and writes DIR/scores_DVIFM03.json as quality = -JND
  judge     --rd DIR: scores an RD judge panel's pairs (DIR/judge_pairs.tsv, scripts/v_next/rd_probe_2026-07-18.sh) into
            DIR/judge_dvifm.tsv (`jnd`, a distance), the panel's DVIFM 0.3 judge

E21 row order is the fold tables' (`wide/main/real/<ds>.keys.parquet`), which is E34 panel-input order. Paths,
distortion type and level come from the promoted feature bank's keys of each member set; no label is read here.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

RUNTIME = Path("/mnt/v/output/zensim/jnd-reconstruction-2026-10-20-dvifm03/dvifm-runtime")
PRESET = "port-lap-v0.3"
E21 = Path("/mnt/v/output/zensim/v40r4-2026-10-08/v2e29/wide/main/real")
BANK = Path("/var/tmp/rev4-featbank/bank")
MEMBERS = {"kadid": ("kadid_train", "kadid_select"), "tid2013": ("tid2013",), "konfig": ("konfig_train", "konfig_val"),
           "cid22_a25": ("cid22_a25",)}
OUT = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/dvifm03")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def frozen_binary() -> Path:
    frozen = json.loads((RUNTIME / "FROZEN.json").read_text())
    binary = RUNTIME / "dvifmish"
    if sha(binary) != frozen["binary_sha256"] or frozen["preset"] != PRESET:
        raise SystemExit(f"{binary}: not the frozen DVIFM runtime")
    return binary


def batch(pairs: pd.DataFrame, dest: Path, threads: int) -> pd.DataFrame:
    """Score unique (ref, dist) pairs; returns them with the preset's `jnd` output (a distance, lower is better)."""
    src = dest.with_suffix(".pairs.tsv")
    pairs[["ref_path", "dist_path"]].to_csv(src, sep="\t", index=False)
    subprocess.run([str(frozen_binary()), "batch", "--pairs", str(src), "--out", str(dest), "--preset", PRESET,
                    "--threads", str(threads)], check=True)
    got = pd.read_csv(dest, sep="\t")
    if len(got) != len(pairs) or not (got.dist_path.to_numpy() == pairs.dist_path.to_numpy()).all():
        raise SystemExit(f"{dest}: rows do not match the pairs list")
    return got


def cmd_pairs(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, sets in MEMBERS.items():
        keys = pd.read_parquet(E21 / f"{ds}.keys.parquet")
        bank = pd.concat([pd.read_parquet(BANK / s / "keys.parquet") for s in sets]).drop_duplicates("pair_key")
        bank = bank.set_index("pair_key")
        miss = set(keys.pair_key) - set(bank.index)
        if miss:
            raise SystemExit(f"{ds}: {len(miss)} E21 keys have no bank paths")
        b = bank.loc[keys.pair_key]
        rows.append(pd.DataFrame({"dataset": ds, "e21_row": range(len(keys)), "pair_key": keys.pair_key.to_numpy(),
                                  "ref_group": keys.ref_basename.to_numpy(), "type": b.codec.to_numpy(),
                                  "level": b.knob.astype(str).to_numpy(), "ref_path": b.ref_path.to_numpy(),
                                  "dist_path": b.dist_path.to_numpy()}))
    rows = pd.concat(rows, ignore_index=True)
    rows.to_parquet(OUT / "rows.parquet", index=False)
    uniq = rows[["ref_path", "dist_path"]].drop_duplicates().reset_index(drop=True)
    uniq.to_csv(OUT / "pairs.tsv", sep="\t", index=False)
    print(json.dumps({"rows": {d: int((rows.dataset == d).sum()) for d in MEMBERS}, "unique_pairs": len(uniq),
                      "rows_sha256": sha(OUT / "rows.parquet"), "pairs_sha256": sha(OUT / "pairs.tsv")}))
    return 0


def cmd_score(a) -> int:
    uniq = pd.read_csv(OUT / "pairs.tsv", sep="\t")
    got = batch(uniq, OUT / "dvifm.tsv", a.threads)
    print(json.dumps({"pairs": len(got), "dvifm_sha256": sha(OUT / "dvifm.tsv"), "binary": sha(frozen_binary())}))
    return 0


def cmd_mixture(a) -> int:
    mix = Path(a.mixture)
    dest = mix / "scores_DVIFM03.json"
    if dest.exists():
        raise SystemExit(f"exists: {dest}")
    man = json.loads((mix / "manifest.json").read_text())
    images = sorted({(r["ref"], r["base"]) for r in man["rows"]} | {(r["ref"], r["more"]) for r in man["rows"]})
    pairs = pd.DataFrame([(str(mix / "img" / r), str(mix / "img" / d)) for r, d in images],
                         columns=["ref_path", "dist_path"])
    got = batch(pairs, OUT / "mixture_dvifm.tsv", a.threads)
    scores = {d: -float(j) for (_, d), j in zip(images, got.jnd)}
    dest.write_text(json.dumps({"bake": f"dvifmish {PRESET}", "post": "negated DVIFM jnd (quality = -distance)",
                                "binary_sha256": sha(frozen_binary()), "scores": scores}, indent=1) + "\n")
    print(f"DVIFM03: {len(scores)} mixture images scored")
    return 0


def cmd_judge(a) -> int:
    rd = Path(a.rd)
    dest = rd / "judge_dvifm.tsv"
    if dest.exists():
        raise SystemExit(f"exists: {dest}")
    pairs = pd.read_csv(rd / "judge_pairs.tsv", sep="\t")[["ref_path", "dist_path"]].drop_duplicates()
    got = batch(pairs.reset_index(drop=True), dest, a.threads)
    print(json.dumps({"pairs": len(got), "judge_dvifm_sha256": sha(dest), "binary": sha(frozen_binary())}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pairs")
    s = sub.add_parser("score")
    s.add_argument("--threads", type=int, default=16)
    m = sub.add_parser("mixture")
    m.add_argument("--mixture", required=True)
    m.add_argument("--threads", type=int, default=16)
    j = sub.add_parser("judge")
    j.add_argument("--rd", required=True)
    j.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    return {"pairs": cmd_pairs, "score": cmd_score, "mixture": cmd_mixture, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
