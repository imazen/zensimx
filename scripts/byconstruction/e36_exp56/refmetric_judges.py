"""Reference metrics on the E36 confirmation sets (record, R10 amendment 5): turn `jnd-ladder-score` and
`zenmetrics batch` outputs into the `judge_*.tsv` files `dataset_eval.load_scores` reads beside the candidates'
`pred_*.tsv`, so clic.py / ciqa.py / mos_confirm.py `confirm`, exp5/readout.py and ceval.py report the metrics with the
same statistics as the models. One value column per file, oriented higher = better (`load_scores` negates only
`butteraugli*` files, so JND-style distances are stored negated here).

  refmetric_judges.py EVAL_DIR PAIRS.tsv [--haar SCORES.tsv] [--fsim SCORES.tsv] [--haarpsi SCORES.tsv] [--keep-ssim2]

PAIRS.tsv is the `id ref dist` list the scorers ran on (id = the eval rows' pair key). `--haar` is jnd-ladder-score's
output (ssim2, ba_max, ba_p3, psnr_y, <config>_jnd / _raw); `--fsim` and `--haarpsi` are `zenmetrics batch` outputs
(ref_path, dist_path, pair_key, metric columns). judge_pairs.tsv is written from PAIRS.tsv when it holds no rows.
Without --keep-ssim2 the jnd-ladder ssim2 / ba_p3 columns are written as judge_ssim2 / judge_butteraugli only if those
files do not exist yet (EVAL keeps the pinned judge build's).
"""
import argparse
import csv
from pathlib import Path


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_judge(out, name, pairs, values, overwrite=True):
    dest = out / f"judge_{name}.tsv"
    if dest.exists() and not overwrite:
        print(f"keep {dest.name}")
        return
    with open(dest, "w") as g:
        g.write(f"ref_path\tdist_path\tpair_key\t{name}\n")
        n = 0
        for pid, (ref, dist) in pairs.items():
            v = values.get(pid)
            if v is None:
                continue
            g.write(f"{ref}\t{dist}\t{pid}\t{v!r}\n")
            n += 1
    print(f"wrote {dest.name}: {n} of {len(pairs)} pairs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("pairs")
    ap.add_argument("--haar")
    ap.add_argument("--fsim")
    ap.add_argument("--haarpsi")
    ap.add_argument("--keep-ssim2", action="store_true", help="never write judge_ssim2 / judge_butteraugli")
    a = ap.parse_args()
    out = Path(a.eval_dir)
    pairs = {r["id"]: (r["ref"], r["dist"]) for r in read_tsv(a.pairs)}
    jp = out / "judge_pairs.tsv"
    if not jp.exists() or len(read_tsv(jp)) == 0:
        with open(jp, "w") as g:
            g.write("ref_path\tdist_path\tpair_key\n")
            for pid, (ref, dist) in pairs.items():
                g.write(f"{ref}\t{dist}\t{pid}\n")
        print(f"wrote judge_pairs.tsv: {len(pairs)} pairs")
    else:
        have = {r["pair_key"]: (r["ref_path"], r["dist_path"]) for r in read_tsv(jp)}
        missing = [k for k in pairs if k not in have]
        if missing:
            raise SystemExit(f"judge_pairs.tsv lacks {len(missing)} of the pair list's keys, e.g. {missing[:2]}")
    if a.haar:
        rows = {r["id"]: r for r in read_tsv(a.haar)}
        for r in rows.values():
            if pairs.get(r["id"]) != (r["ref"], r["dist"]):
                raise SystemExit(f"{a.haar}: row {r['id']} does not match the pair list")
        cols = [c for c in next(iter(rows.values())) if c.endswith("_jnd")]
        for c in cols:
            short = {"HaarXYB-SSIM-deltaL1248_jnd": "haar_ssim", "HaarXYB-SSIM-mask-deltaL1248_jnd": "haar_mask",
                     "HaarXYB-SSIM-deltaL1248-fit_jnd": "haar_ssim_fit",
                     "HaarXYB-SSIM-mask-deltaL1248-fit_jnd": "haar_mask_fit"}.get(c, c.replace("_jnd", "").replace("-", "_"))
            write_judge(out, short, pairs, {k: -float(r[c]) for k, r in rows.items() if r[c] not in ("", "nan")})
        if not a.keep_ssim2:
            write_judge(out, "ssim2", pairs, {k: float(r["ssim2"]) for k, r in rows.items()}, overwrite=False)
            # positive values; load_scores negates every `butteraugli*` file
            write_judge(out, "butteraugli", pairs, {k: float(r["ba_p3"]) for k, r in rows.items()}, overwrite=False)
    for opt, names in ((a.fsim, {"fsimc_imazen_v0_1_0": "fsimc", "fsim_imazen_v0_1_0": "fsim_y"}),
                       (a.haarpsi, {"haarpsi_imazen_v0_1_0": "haarpsi"})):
        if not opt:
            continue
        rows = read_tsv(opt)
        for r in rows:
            if pairs.get(r["pair_key"]) != (r["ref_path"], r["dist_path"]):
                raise SystemExit(f"{opt}: row {r['pair_key']} does not match the pair list")
        for col, short in names.items():
            if col in rows[0]:
                write_judge(out, short, pairs, {r["pair_key"]: float(r[col]) for r in rows if r[col] not in ("", "nan")})


if __name__ == "__main__":
    main()
