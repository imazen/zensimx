"""Data files of the AIC2026 TRAIN dashboard. Exploration record: benchmarks/byconstruction_2026-10-22.md.

results.json holds every summary table in long form (`table` names it, `id` keys a row); predictions.json one row per
WG1 public_train crop (official JND; raw B, T, Tga, Tga128, butteraugli 3-norm and DVIFM 0.3 outputs; each also mapped
onto the JND scale three ways: `f_` the panel's 4-parameter logistic fit, `r_` zenstats' rank map, `a_` mean and SD
matched), without stimulus names or levels. Everything is read from the study's saved outputs; the E21 scores,
mixture report and panel are regenerated through their owners (byc_cells.py score, mixture_test.py report,
aic_panel.py --dvifm --geo --per-type). Nothing is computed on AIC2026 public_test: its labels are never loaded here;
the frozen-B row comes from panel_public_test_frozen.json and the candidate's from full/Tga/panel.json.

  python3 aic_dashboard_data.py --out DIR --arms Tga,Ma,Bga,... [--commits-since 322930e4]
  python3 aic_dashboard_data.py --out NEW --extend OLD    E21 scatter and revision-7 rows only (Mr7, Mdata, Mlic, ...)
  python3 aic_dashboard_data.py --out NEW --curate         the two curated roles: general and codec-only views
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aic_train as A  # noqa: E402
from aic_panel import GEOMETRY, TYPE_NAMES, geometry  # noqa: E402
from lib.zen_stats import panel_batch  # noqa: E402

REPO = HERE.parents[1]
O = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train")
F = O / "folds_train"
CELLS = Path.home() / "tmp/byc/cells"
BIN = os.environ.get("AIC_BIN", "/var/tmp/byconstruction-bin/58a4d100ad36")
GROOT = "/mnt/v/output/zensim/byconstruction-2026-10-22/ringhpm/root_globaff"
MIX = "/mnt/v/output/zensim/byconstruction-2026-10-22/mixture"
MIXG = "/mnt/v/output/zensim/byconstruction-2026-10-22/mixture_global"  # global edits, then noise
SCORES = Path("/home/lilith/tmp/aic-full-repro/scores")
LOGS = Path.home() / "tmp/aictrain"

# Combined E21 + AIC arms: label, class, AIC weight.
ARMS = {
    "Tga": ("Tga", "monotone in damage inputs, descriptor sign-free", 32),
    "Ma": ("Ma (certified max pairs)", "certified convex monotone max pairs", 32),
    "Bga": ("Bga (B's signed recipe)", "pin only (signed)", 32),
    "Ta": ("Ta (certified hinges)", "certified convex monotone", 32),
    "Ts1a": ("Ts1a (no scale 0)", "monotone in damage inputs, descriptor sign-free, no scale-0 inputs", 32),
    "Mgfa": ("Mgfa (max pairs + free descriptor)", "convex monotone in damage inputs, descriptor sign-free", 32),
    "Tga128": ("Tga128 (AIC weight 128)", "monotone in damage inputs, descriptor sign-free", 128),
}
# Same model classes without the AIC leg, for the arms table.
CONTROLS = {"Thpm": "Thpm (monotone, no descriptor, no AIC leg)", "Tghpm": "Tghpm (no AIC leg)",
            "Bghpm": "Bghpm (B + descriptor, no AIC leg)", "Tgg": "Tgg (gated damage, every weight >= 0, no AIC leg)",
            "M64": "M64 (certified max pairs, no AIC leg)", "Mgg64": "Mgg64 (max pairs on gated damage, no AIC leg)",
            "M64lin": "M64lin (M64 recipe, coarse scales from linear RGB, no AIC leg)"}
# Arms made from other arms' predictions after the fact (no training of their own): the mean of two cells' outputs.
POSTHOC = {"M64ens": "M64ens (mean of M64 and M64lin, post hoc, no AIC leg)"}
# Revision-7 arms (no AIC leg). Their cells sit in the data-legs lane's directory, where Mr7 links to
# ~/tmp/byc/cells/Mr7; they are scored against the revision-7 data root with binaries that read revision-7 bakes.
REV7 = {"Mr7": "Mr7 (revision-7 M, no AIC leg)", "Mdata": "Mdata (Mr7 + 11 new datasets' TRAIN legs, no AIC leg)",
        "Mdata64": "Mdata64 (Mdata's data, M64's shape, no AIC leg)", "MdataT": "MdataT (Mdata's data, T's shape, no AIC leg)",
        "Mlic": "Mlic (Mr7 + 8 licence-clean new datasets, no AIC leg)",
        "Mlic64": "Mlic64 (Mlic's data, M64's shape, no AIC leg)"}
REV7_CELLS = Path(os.environ.get("BYC_REV7_CELLS", "/var/tmp/datalegs-20261025/cells"))
REV7_ROOT = "/mnt/v/output/zensim/byconstruction-2026-10-22/rev7data/root"
REV7_BIN = os.environ.get("REV7_BIN", "/var/tmp/datalegs-20261025/bin")
NAMES = {"kadid": "KADID", "tid2013": "TID2013", "konfig": "KonFiG", "cid22_a25": "CID22-A",
         "aic2026_public_train": "AIC2026 public_train", "aic2026_public_test": "AIC2026 public_test"}
GEO_F = ("fit_p99", "fit_max", "out4", "p99d", "maxd")
COMMIT_SUMMARY = {
    "d24c2ec6": "AIC2026 feature tables, source-held-out fits, band-feature prototypes",
    "31e3ba97": "AIC2026 as a research training leg through the Rust trainer (strict refusal kept)",
    "67e7659e": "Found the official split (WG1 public_train); fold tables; scale-0-free input set",
    "805242d9": "Trainer: max pairs accept a sign-free input (new monotonicity test)",
    "b7aba684": "First E21 + AIC arm (Tga); engine skip attempt recorded, not landed",
    "bf828d6c": "Full-data research fit route; per-type bias by arm",
    "4da1d051": "Six-statistic panel per dataset and split; Ma arm; AIC test-data ledger",
    "e5bfcf1c": "Engine: declared-ID plans skip pyramid scales nothing reads (exact; tests pass)",
}
# Every use of AIC2026 public_test or its 14 sources on 2026-10-23, counted from the commands run (record section
# "AIC2026 TRAIN"). Edit only when a new use happens.
EXPOSURE = [
    ("Before the split was found", "Overlay labels (JND_curve), 42-source pool",
     "Frozen models scored, no fit: B, full-data T and full-data M", 1, "score only"),
    ("Before the split was found", "Overlay labels (JND_curve), 42-source pool",
     "Descriptive label reads: label statistics and supplied-metric ranking; confidence-interval ceiling; pointwise vs "
     "curve; single-feature rankings (4 feature sets); cross-ladder agreement; one ladder inspected by eye (2)", 10, "description"),
    ("Before the split was found", "Overlay labels (JND_curve), 42-source pool",
     "Python source-fold fits with the 14 public_test sources inside the folds: fold fits, peer scans, learning curves (5), "
     "prototypes, oracles (2), target transforms (3), scale tests (3), diagnostics", 32, "fit and score"),
    ("Before the split was found", "Overlay labels (JND_curve), 42-source pool",
     "Rust trainer: T and M on 6 source folds each, one 2-epoch smoke fit, one E21 smoke cell", 4, "fit and score"),
    ("After the split was found", "Official labels (JND_score), public_test sources",
     "Frozen B scored once, with the reconstruction-agreement check in the same script", 1, "score only"),
    ("After the split was found", "Official labels (JND_score), public_test sources",
     "Per-source JND statistics computed for the dashboard, then removed from it", 1, "description"),
    ("After the split was found", "Official labels (JND_score), public_test sources",
     "Six-statistic panel on the same frozen-B predictions (the dashboard); regenerated with identical numbers on each "
     "dashboard rebuild until it was frozen to a file the same evening", 1, "score only"),
    ("After the split was found", "Official labels (JND_score), public_test sources",
     "The candidate (Tga, full-data fit, chosen by the rule committed in 04743622) scored once", 1, "score only"),
    ("After the split was found", "Official labels (JND_score), public_test sources",
     "Fits or model choices", 0, "none"),
]
CANDIDATE = O / "full" / "Tga"
A_E34 = Path("/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json")
DVIFM_DIR = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/dvifm03")


def slug(s: str, n: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:n]


def last_result(path: Path):
    r = None
    for line in path.read_text().splitlines():
        if line.startswith("test result"):
            m = re.search(r"(\d+) passed; (\d+) failed; (\d+) ignored", line)
            r = tuple(map(int, m.groups()))
    return r


def oof_rust(arm: str, keys) -> pd.Series:
    s = pd.Series(np.nan, index=keys)
    for k in range(7):
        kk = pd.read_parquet(F / f"aic_k{k}_test.parquet", columns=["key"]).key.astype(str).to_numpy()
        s.loc[kk] = pd.read_csv(O / "rust/folds_train" / arm / f"k{k}" / "pred.tsv", sep="\t").pred.to_numpy()
    return s


def oof_cells(arm: str, keys) -> pd.Series:
    by = {}
    for rp in glob.glob(str(CELLS / arm / "*/byc_record.json")):
        r = json.loads(Path(rp).read_text())
        by.setdefault(r["aic_fold"], []).append(
            pd.read_csv(Path(rp).parent / f"pred_aic_k{r['aic_fold']}.tsv", sep="\t").pred.to_numpy())
    s = pd.Series(np.nan, index=keys)
    for k, ps in by.items():
        kk = pd.read_parquet(F / f"aic_k{k}_test.parquet", columns=["key"]).key.astype(str).to_numpy()
        s.loc[kk] = np.mean(ps, 0)
    return s


def run_json(cmd: list[str], env: dict) -> dict:
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout
    return json.loads(out[out.index("{"):])


def e21_scatter(add) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The scatter views for the E21 sets: each dataset's out-of-sample seed-0 predictions on its own label scale
    (quality orientation, as E34's panel input stores the targets), one `scatter` row per dataset and model. The
    revision-7 arms go to a second frame (the dashboard's file size limit), the others to the first as before."""
    panel_in = {d["name"]: d for d in json.loads(A_E34.read_text())["datasets"]}
    dvr = pd.read_parquet(DVIFM_DIR / "rows.parquet")
    dvj = pd.read_csv(DVIFM_DIR / "dvifm.tsv", sep="\t").set_index(["ref_path", "dist_path"]).jnd
    e21rows, e21rows7 = [], []
    rev7_keys = {"mr7", "mdata", "mdata64", "mdatat", "mlic", "mlic64"}
    for ds in ("kadid", "tid2013", "konfig", "cid22_a25"):
        y = np.asarray(panel_in[ds]["target"], np.float64)
        sel = dvr[dvr.dataset == ds]
        frame = {"ds": ds, "label": np.round(y, 4), "type": [TYPE_NAMES.get(t, t) for t in sel.type]}
        frame7 = dict(frame)
        models = [("basec", "B’s recipe (base-c), seed 0", np.asarray(panel_in[ds]["predictions"]["base-c"][0], np.float64))]
        for arm, sid, lab, cells in (("Tga", "tga", "Tga, seed 0", CELLS),
                                     ("mono-thr-mp64", "m64", "M64 (certified max pairs), seed 0", CELLS),
                                     ("Mgg64", "mgg64", "Mgg64 (max pairs, gated damage), seed 0", CELLS),
                                     ("Tgg", "tgg", "Tgg, seed 0", CELLS),
                                     ("M64lin", "m64lin", "M64lin (linear-light coarse scales), seed 0", CELLS),
                                     ("M64ens", "m64ens", "M64ens (mean of M64 and M64lin, post hoc), seed 0", CELLS),
                                     ("Mr7", "mr7", "Mr7 (revision-7 M), seed 0", REV7_CELLS),
                                     ("Mdata", "mdata", "Mdata (Mr7 + new datasets), seed 0", REV7_CELLS),
                                     ("Mdata64", "mdata64", "Mdata64 (Mdata's data, M64's shape), seed 0", REV7_CELLS),
                                     ("MdataT", "mdatat", "MdataT (Mdata's data, T's shape), seed 0", REV7_CELLS),
                                     ("Mlic", "mlic", "Mlic (licence-clean new datasets), seed 0", REV7_CELLS),
                                     ("Mlic64", "mlic64", "Mlic64 (Mlic's data, M64's shape), seed 0", REV7_CELLS)):
            f = cells / arm / f"without_{ds}_s0" / f"pred_{ds}.tsv"
            if f.is_file():
                x = pd.read_csv(f, sep="\t").pred.to_numpy(np.float64)
                if len(x) != len(y):
                    raise ValueError(f"{f}: {len(x)} predictions for {len(y)} labels")
                models.append((sid, lab, x))
        models.append(("dvifm", "DVIFM 0.3 (zero-shot)", -dvj.loc[list(zip(sel.ref_path, sel.dist_path))].to_numpy(np.float64)))
        for sid, lab, x in models:
            st, arr = geometry(x, y, arrays=True)
            pn = panel_batch([(sid, x, y)], stats="full")[0]
            fr = frame7 if sid in rev7_keys else frame
            fr[f"f_{sid}"], fr[f"r_{sid}"] = np.round(arr["fit"], 4), np.round(arr["rank"], 4)
            fr[f"a_{sid}"] = np.round(y.mean() + (x - x.mean()) * (y.std() / x.std()), 4)
            add("scatter", id=f"{ds}_{sid}", dataset=NAMES[ds], key=sid, model=lab, n=int(pn["n"]),
                srocc=pn["srocc_signed"], plcc=pn["plcc"], z_rmse=pn["z_rmse"], **{"or": pn["or"]}, **st)
        e21rows.append(pd.DataFrame(frame))
        e21rows7.append(pd.DataFrame(frame7))
    return pd.concat(e21rows, ignore_index=True), pd.concat(e21rows7, ignore_index=True)


def rev7_rows(add, out: Path) -> None:
    """Panel rows (E21 in and out of sample, smoothness and geometry, no AIC label) and arms rows for the revision-7
    arms, with the seeds both arms have finished; the mixture columns stay empty until an arm's mixture cells exist."""
    # Revision-7 bakes need the newer bake readers; the statistics come from the same panel build as every other row.
    env = dict(os.environ, AIC_BIN=REV7_BIN, REV4_V2_BIN_DIR=REV7_BIN, ZEN_PANEL_BIN=f"{BIN}/panel",
               BYC_CELLS=str(REV7_CELLS), BYC_ROOT=REV7_ROOT)
    done = {a: {s for s in (0, 1, 2) if len(list((REV7_CELLS / a).glob(f"without_*_s{s}/pred_*.tsv"))) == 4} for a in REV7}
    arms = [a for a in REV7 if done[a]]
    if not arms:
        return
    panel_json = out / "panel_rev7.json"
    subprocess.run([sys.executable, str(HERE / "aic_panel.py"), "--out", str(panel_json), "--arms", ",".join(arms),
                    "--geo", "--per-type", "--no-aic", "--no-public-test"], check=True, capture_output=True, env=env)
    pj = json.loads(panel_json.read_text())
    for x in pj["rows"]:
        if x["model"] in REV7:
            add("panel", id=slug(x["dataset"] + " " + x["model"] + " " + x["sample"]), dataset=NAMES[x["dataset"]],
                model=REV7[x["model"]], sample=x["sample"], kind="out" if x["sample"].startswith(("out", "zero")) else "in",
                n=x["n"], seeds=x["seeds"], srocc=x["srocc_signed"], plcc=x["plcc"], krocc=x["krocc"], **{"or": x["or"]},
                pwrc=x["pwrc"], z_rmse=x["z_rmse"], **{k: x.get(k) for k in GEO_F})
    seeds = sorted(set.intersection(*(done[a] for a in arms)))
    sc = run_json([sys.executable, str(REPO / "scripts/rev4_featpot/byc_cells.py"), "score", "--arms", ",".join(arms),
                   "--seeds", ",".join(map(str, seeds))], env)
    # Mixture scores: the shared directories, then the revision-7 lane's own copies of the same tests (refused if a
    # manifest differs from the shared one, so scores from different test sets never mix).
    mix, mixg = {}, {}
    for shared, lane, acc in ((MIX, REV7_CELLS.parent / "mixture", mix), (MIXG, REV7_CELLS.parent / "mixture_global", mixg)):
        acc.update(run_json([sys.executable, str(HERE / "mixture_test.py"), "report", "--out", shared], env))
        if (lane / "manifest.json").is_file():
            if (lane / "manifest.json").read_bytes() != (Path(shared) / "manifest.json").read_bytes():
                raise ValueError(f"{lane}: mixture manifest differs from {shared}")
            acc.update(run_json([sys.executable, str(HERE / "mixture_test.py"), "report", "--out", str(lane)], env))
    for arm in arms:
        s = sc[arm]
        outs = [x for x in pj["rows"] if x["model"] == arm and x["sample"].startswith("out")]
        cells = [v for k, v in mix.items() if k.startswith(arm + "_")]
        gcells = [v for k, v in mixg.items() if k.startswith(arm + "_")]
        add("arms", id=slug(arm), arm=REV7[arm], aic_weight=None, e21_delta=s["delta"], e21_se=s["seed_unit_se"],
            **{d: s["per_dataset_delta"][d] for d in ("kadid", "tid2013", "konfig", "cid22_a25")}, aic_oof=None,
            mix_min=min(v["inversions"] for v in cells) if cells else None,
            mix_max=max(v["inversions"] for v in cells) if cells else None,
            mix_max_rise=max(v["max_rise"] for v in cells) if cells else None,
            gmix_max=max(v["inversions"] for v in gcells) if gcells else None,
            gmix_max_rise=max(v["max_rise"] for v in gcells) if gcells else None,
            status=f"E21 seeds {','.join(map(str, seeds))}" + ("" if seeds == [0, 1, 2] else
                                                               " (matched across the revision-7 arms; more pending)"),
            z_e21=float(np.mean([x["z_rmse"] for x in outs])), p99_e21=float(np.mean([x["fit_p99"] for x in outs])),
            z_aic=None)


EVAL_NAMES = {"cid-iq": "EVAL CID:IQ", "cid-oa": "EVAL CID:OA", "colourlab-rating-variability-2022": "EVAL Colourlab",
              "iqa-odi": "EVAL IQA-ODI", "irccyn-ivc": "EVAL IRCCyN/IVC", "live-md": "EVAL LIVE-MD", "mdid": "EVAL MDID",
              "nits-iqa": "EVAL NITS", "vcl-fer": "EVAL VCL@FER"}
EVAL_MODELS = {"b7": "B, revision-7 stamp", "m7": "M7 (revision-7 M)", "mdata": "Mdata (M7 + new datasets)",
               "mlic": "Mlic (M7 + licence-clean new datasets)",
               "ssim2": "SSIMULACRA2", "cvvdp": "CVVDP", "butter": "Butteraugli 3-norm"}


def eval_scatter_rows(add, eval_dir: Path) -> None:
    """Scatter rows for the provided-library EVAL sets from `dataset_eval.py scatter` (eval_scatter_stats.json)."""
    for s in json.loads((eval_dir / "eval_scatter_stats.json").read_text()):
        add("scatter", id=f"eval_{slug(s['collection'])}_{s['key']}", dataset=EVAL_NAMES[s["collection"]], key=s["key"],
            model=EVAL_MODELS[s["key"]], **{k: v for k, v in s.items() if k not in ("collection", "key", "model")})


def extend(prev: Path, out: Path, eval_dir: Path | None = None) -> int:
    """Refresh the E21 scatter views and the revision-7 rows of an earlier build without rebuilding its AIC tables;
    with `eval_dir`, also the EVAL sets' scatter rows."""
    rows = [r for r in json.loads((prev / "results.json").read_text())
            if not (r["table"] == "scatter" and r["dataset"] in ("KADID", "TID2013", "KonFiG", "CID22-A"))
            and not (r["table"] == "scatter" and r["dataset"].startswith("EVAL "))
            and not (r["table"] == "panel" and r["model"] in REV7.values())
            and not (r["table"] == "arms" and r["arm"] in REV7.values())]

    def add(table, **kw):
        rows.append({"table": table, **kw})
    e21, e21_7 = e21_scatter(add)
    rev7_rows(add, out)
    if eval_dir:
        eval_scatter_rows(add, eval_dir)
    ids = [(r["table"], r.get("id")) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row ids")
    (out / "results.json").write_text(json.dumps(rows, indent=0))
    e21.sample(frac=1, random_state=0).reset_index(drop=True).to_json(out / "e21predictions.json", orient="records")
    e21_7.sample(frac=1, random_state=0).reset_index(drop=True).to_json(out / "e21predictions7.json", orient="records")
    print(f"{len(rows)} result rows; {len(e21)} E21 prediction rows (extended {prev})")
    return 0


# The two models the owner is curating (2026-10-27): a general model, judged on every distortion type, and a codec
# model, judged on compression only (AIC-2026 public_train and the compression types of the E21 sets). Seed 0, read
# from the October 27 studies' saved outputs: held-out AIC-2026 public_train predictions (the adaptation study's
# omission cells, mean over the cells holding each source out; or each frontier fold model on its held-out sources)
# and E21 out-of-sample cells. No fit, and no EVAL read.
def keyed_scores(df):
    """A judge score table indexed by distorted-image name without `.png` (the AIC-2026 crop key)."""
    col = "distorted" if "distorted" in df.columns else "dist_path"
    return df.assign(key=df[col].astype(str).str.split("/").str[-1].str.removesuffix(".png")).set_index("key")


CODEC_TYPES = {"kadid": ("kadid_09", "kadid_10"), "tid2013": ("tid_10", "tid_11", "tid_21"), "konfig": ("jpeg2000",),
               "cid22_a25": None}  # None: every row (CID22-A is codecs only)
Z = Path("/mnt/v/output/zensim")
CURATE = (  # key, name, AIC-2026 public_train source, E21 out-of-sample cells (skipped until they exist)
    ("mlic", "Mlic (no AIC leg)", ("heldout", "Mlic"), Path("/var/tmp/byc-mixes-20261025/cells/Mlic")),
    ("mlicft", "MlicFT16x8", ("heldout", "MlicFT16x8"), Z / "aic-finetune-20261026/run-05/cells/MlicFT16x8"),
    ("mvote", "MlicVote (no AIC leg)", ("heldout", "MlicVote"), Z / "mlic-vote-20261027-run02/cells/MlicVote"),
    ("mvoteft", "MlicVoteFT16x8", ("heldout", "MlicVoteFT16x8"), Z / "mlic-aic-adaptation-20261027-run03/cells/MlicVoteFT16x8"),
    ("vote256", "Vote ×256, epoch 29", ("fold", Z / "mlic-aic-recipe-vote-aic256-20261027-run01", "MlicVoteAIC256@e29"),
     Z / "mlic-aic-omission-voteaic256-20261027-run01/cells/MlicVoteAIC256e29"),
    ("xref256", "×256 + rehearsal, epoch 59", ("fold", Z / "mlic-aic-xref-20261027-run01", "XrefAIC256@e59"),
     Z / "mlic-aic-omission-xrefaic256-20261027-run01/cells/XrefAIC256e59"),
    ("aiconly", "AIC-only, epoch 29", ("fold", Z / "mlic-aic-share-frontier-20261027-run01", "AICOnly@e29"),
     Z / "mlic-aic-omission-codec-aiconly-20261027-run01/cells/AICOnlye29"),
    ("x1024", "×1024, epoch 59", ("fold", Z / "mlic-aic-share-frontier-20261027-run01", "MixedAIC1024@e59"),
     Z / "mlic-aic-omission-codec-x1024-20261027-run01/cells/MixedAIC1024e59"),
    ("xref1024", "×1024 + rehearsal, epoch 59", ("fold", Z / "mlic-aic-xref-20261027-run01", "XrefAIC1024@e59"),
     Z / "mlic-aic-omission-codec-xref1024-20261027-run01/cells/XrefAIC1024e59"),
)


# Zero-shot peers on the same pixels for the curated views: SSIMULACRA2 (fast-ssim2) and butteraugli's 3-norm (imazen
# crate), both from the pinned judge build, oriented higher = better. AIC-2026 public_train: the existing crop scores
# (SCORES). E21 sets: `zenmetrics batch` over the E21 rows' own pixel pairs (dvifm03/rows.parquet, E21 order), in
# E21_PEERS; no label is read to make them. Their views go to separate files (the dashboard's 8 MB source limit).
E21_PEERS = Z / "e21-peers-20261028"
CURATE_PEERS = (("ssim2", "SSIMULACRA2 (zero-shot)", "judge_ssim2.tsv", "ssim2", 1),
                ("butter3", "butteraugli 3-norm (zero-shot)", "judge_butteraugli.tsv", "butteraugli_pnorm3", -1))


def e21_peer_scores(ds: str) -> dict:
    """Each peer's score per E21 row of `ds`, higher = better, in E21 order (joined on pair_key)."""
    rows = pd.read_parquet(Path("/mnt/v/output/zensim/byconstruction-2026-10-22/dvifm03/rows.parquet"))
    rows = rows[rows.dataset == ds].sort_values("e21_row")
    if not (rows.e21_row.to_numpy() == np.arange(len(rows))).all():
        raise ValueError(f"{ds}: E21 rows are not contiguous")
    out = {}
    for key, _, name, col, sign in CURATE_PEERS:
        judge = pd.read_csv(E21_PEERS / name, sep="\t").drop_duplicates("pair_key").set_index("pair_key")
        out[key] = sign * judge.loc[rows.pair_key, col].to_numpy(np.float64)
    return out


def curate(out: Path) -> int:
    """curate.json (one row per AIC-2026 public_train crop and E21 stimulus, each candidate on that set's label scale
    three ways, plus codec-only views fitted on the compression rows) and curate_stats.json (statistics per set,
    subset and candidate, and one summary row per candidate)."""
    sys.path.insert(0, str(HERE))
    from aic_matched_continuation import INPUTS, SOURCE_FOLDS, out_of_fold
    from aic_adaptation_full_panel import full_populations
    from aic_panel import dvifm_e21
    fold = json.loads(Path(INPUTS["fold_plan"][0]).read_text())
    sys.path[:0] = [str(Path(fold["warmstart_source"]) / "scripts/rev4_featpot"), str(Path(fold["warmstart_source"]) / "scripts")]
    pairs, populations = full_populations(fold, out, "dashboard-curation-views")
    assignment = json.loads(Path(INPUTS["aic_authority"][0]).read_text())["assignment"]
    folds = pairs.group.map(assignment).to_numpy(dtype=int)
    _, types = dvifm_e21()
    adapt = Path(INPUTS["fold_plan"][0]).parent / "panels"
    rows, stats = [], []

    def aic_pred(source):
        if source[0] == "heldout":
            h = pd.read_csv(adapt / f"{source[1]}-heldout.tsv", sep="\t").set_index("key").pred
            if set(h.index) != set(pairs.key):
                raise ValueError(f"{source[1]}: held-out predictions do not cover public_train")
            return h.loc[pairs.key].to_numpy(np.float64)
        run, label = source[1], source[2]
        values = {f"{label}|k{k}": pd.read_csv(run / "panels/predictions" / f"{label}|s0|k{k}|aic.tsv", sep="\t")
                  .pred.to_numpy(np.float64) for k in SOURCE_FOLDS}
        return out_of_fold(values, folds, label)

    def views(frame, prefix, key, x, y, mask=None, digits=3):
        """Logistic fit, rank map and mean/SD match of `x` onto `y` (agreeing in direction), on `mask` rows."""
        idx = np.arange(len(y)) if mask is None else np.flatnonzero(mask)
        st, arr = geometry(x[idx], y[idx], arrays=True)
        for v, values in (("f", arr["fit"]), ("r", arr["rank"]),
                          ("a", y[idx].mean() + (x[idx] - x[idx].mean()) * (y[idx].std() / x[idx].std()))):
            col = np.full(len(y), np.nan)
            if values is not None:
                col[idx] = values
            frame[f"{v}{prefix}_{key}"] = np.round(col, digits)
        return st

    jnd = pairs.jnd.to_numpy(np.float64)
    aic_frame = {"ds": "aic", "jnd": np.round(jnd, 3), "codec": pairs.codec.to_numpy(), "cdc": 1}
    e21_frames = {}
    for ds in ("kadid", "tid2013", "konfig", "cid22_a25"):
        y = np.asarray(populations[ds]["target"], np.float64)
        ty = np.asarray(types[ds])
        if len(ty) != len(y):
            raise ValueError(f"{ds}: {len(ty)} distortion types for {len(y)} labels")
        cdc = np.ones(len(y), bool) if CODEC_TYPES[ds] is None else np.isin(ty, CODEC_TYPES[ds])
        e21_frames[ds] = ({"ds": ds, "label": np.round(y, 4), "type": [TYPE_NAMES.get(x, x) for x in ty],
                           "cdc": cdc.astype(int)}, y, cdc)
    for key, name, source, cells in CURATE:
        pred = aic_pred(source)
        worse = -pred  # the AIC views sit on the JND scale like the dashboard's other AIC scatters (higher is worse)
        st = views(aic_frame, "", key, worse, jnd)
        pn = panel_batch([(key, pred, -jnd)], stats="full")[0]
        for subset in ("all", "codec"):  # every public_train crop is a codec
            stats.append({"table": "cscatter", "id": f"aic_{subset}_{key}", "dataset": "AIC2026 public_train",
                          "subset": subset, "key": key, "model": name, "n": int(pn["n"]), "srocc": pn["srocc_signed"],
                          "plcc": pn["plcc"], "krocc": pn["krocc"], "or": pn["or"], "pwrc": pn["pwrc"],
                          "z_rmse": pn["z_rmse"], "sample": "held out" if source[0] == "heldout" else "out of fold", **st})
        for ds, (frame, y, cdc) in e21_frames.items():
            f = None if cells is None else cells / f"without_{ds}_s0" / f"pred_{ds}.tsv"
            if f is None or not f.is_file():
                continue
            x = pd.read_csv(f, sep="\t").pred.to_numpy(np.float64)
            if len(x) != len(y):
                raise ValueError(f"{f}: {len(x)} predictions for {len(y)} labels")
            for subset, prefix, mask in (("all", "", None), ("codec", "c", cdc)):
                # Where every row is a codec (CID22-A) the codec views are the full-set views: not stored twice.
                # E21 views at two decimals keep the rows under the dashboard's 8 MB source limit.
                st = views(frame, prefix, key, x, y, mask, 2) if mask is None or not mask.all() else stats[-1].copy()
                if mask is not None and mask.all():
                    st = {k: v for k, v in st.items() if k in GEOMETRY}
                sel = slice(None) if mask is None else mask
                pn = panel_batch([(key, x[sel], y[sel])], stats="full")[0]
                stats.append({"table": "cscatter", "id": f"{ds}_{subset}_{key}", "dataset": NAMES[ds], "subset": subset,
                              "key": key, "model": name, "n": int(pn["n"]), "srocc": pn["srocc_signed"], "plcc": pn["plcc"],
                              "krocc": pn["krocc"], "or": pn["or"], "pwrc": pn["pwrc"], "z_rmse": pn["z_rmse"],
                              "sample": "out of sample", **st})
    aic_peer_frame = {"ds": "aic", "jnd": np.round(jnd, 3), "codec": pairs.codec.to_numpy(), "cdc": 1}
    e21_peer_frames = {ds: ({"ds": ds, "label": f["label"], "type": f["type"], "cdc": f["cdc"]}, y, cdc)
                       for ds, (f, y, cdc) in e21_frames.items()}
    aic_scores = {"ssim2": keyed_scores(pd.read_csv(SCORES / "ssim2.tsv", sep="\t")).loc[pairs.key].ssim2,
                  "butter3": -keyed_scores(pd.read_csv(SCORES / "butteraugli.tsv", sep="\t")).loc[pairs.key]
                  .butteraugli_pnorm3}
    for key, name, *_ in CURATE_PEERS:
        x = np.asarray(aic_scores[key], np.float64)
        st = views(aic_peer_frame, "", key, -x, jnd)
        pn = panel_batch([(key, x, -jnd)], stats="full")[0]
        for subset in ("all", "codec"):
            stats.append({"table": "cscatter", "id": f"aic_{subset}_{key}", "dataset": "AIC2026 public_train",
                          "subset": subset, "key": key, "model": name, "n": int(pn["n"]), "srocc": pn["srocc_signed"],
                          "plcc": pn["plcc"], "krocc": pn["krocc"], "or": pn["or"], "pwrc": pn["pwrc"],
                          "z_rmse": pn["z_rmse"], "sample": "zero-shot", **st})
    for ds, (frame, y, cdc) in e21_peer_frames.items():
        scores = e21_peer_scores(ds)
        for key, name, *_ in CURATE_PEERS:
            x = scores[key]
            if len(x) != len(y) or not np.isfinite(x).all():
                raise ValueError(f"{ds} {key}: {len(x)} finite peer scores required for {len(y)} rows")
            for subset, prefix, mask in (("all", "", None), ("codec", "c", cdc)):
                if mask is not None and mask.all():
                    continue  # every row is a codec: the full-set row stands for both
                st = views(frame, prefix, key, x, y, mask, 2)
                sel = slice(None) if mask is None else mask
                pn = panel_batch([(key, x[sel], y[sel])], stats="full")[0]
                stats.append({"table": "cscatter", "id": f"{ds}_{subset}_{key}", "dataset": NAMES[ds], "subset": subset,
                              "key": key, "model": name, "n": int(pn["n"]), "srocc": pn["srocc_signed"], "plcc": pn["plcc"],
                              "krocc": pn["krocc"], "or": pn["or"], "pwrc": pn["pwrc"], "z_rmse": pn["z_rmse"],
                              "sample": "zero-shot", **st})
    for key, name, source, cells in CURATE:
        mine = {(s["dataset"], s["subset"]): s for s in stats if s["table"] == "cscatter" and s["key"] == key}
        e21 = {sub: [mine[(NAMES[d], sub)]["srocc"] for d in ("kadid", "tid2013", "konfig", "cid22_a25")
                     if (NAMES[d], sub) in mine] for sub in ("all", "codec")}
        stats.append({"table": "cmodels", "id": key, "model": name,
                      "aic": mine[("AIC2026 public_train", "all")]["srocc"],
                      "aic_sample": mine[("AIC2026 public_train", "all")]["sample"],
                      "e21_all": float(np.mean(e21["all"])) if len(e21["all"]) == 4 else None,
                      "e21_codec": float(np.mean(e21["codec"])) if len(e21["codec"]) == 4 else None,
                      **{f"{d}_{sub}": mine[(NAMES[d], sub)]["srocc"] if (NAMES[d], sub) in mine else None
                         for d in ("kadid", "tid2013", "konfig", "cid22_a25") for sub in ("all", "codec")},
                      "e21_status": "out of sample, seed 0" if len(e21["all"]) == 4 else "no out-of-sample cells yet"})
    def records(frames, name):  # shuffled rows, empty views left out (the dashboard's 8 MB source limit)
        df = pd.concat([pd.DataFrame(f) for f in frames], ignore_index=True).sample(frac=1, random_state=0)
        recs = [{k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))} for r in df.to_dict("records")]
        (out / name).write_text(json.dumps(recs, separators=(",", ":"), default=lambda v: v.item()) + "\n")
        return len(recs)
    n_aic = records([aic_frame], "curate_aic.json")
    n_e21 = records([f for f, _, _ in e21_frames.values()], "curate_e21.json")
    records([aic_peer_frame], "curate_aic_peers.json")
    records([f for f, _, _ in e21_peer_frames.values()], "curate_e21_peers.json")
    (out / "curate_stats.json").write_text(json.dumps(stats, indent=0) + "\n")
    print(f"{n_aic} AIC rows, {n_e21} E21 rows; {len(stats)} statistic rows; models {[k for k, *_ in CURATE]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", help="finished combined arms, comma-separated (required unless --extend)")
    ap.add_argument("--commits-since", default="322930e4")
    ap.add_argument("--extend", help="an earlier build's directory: refresh only its E21 scatter and revision-7 rows")
    ap.add_argument("--eval-scatter", help="with --extend: a dataset_eval.py directory whose scatter stats to add")
    ap.add_argument("--curate", action="store_true", help="only the two curated roles' views and statistics")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.curate:
        return curate(out)
    if a.extend:
        return extend(Path(a.extend), out, Path(a.eval_scatter) if a.eval_scatter else None)
    if not a.arms:
        ap.error("--arms is required without --extend")
    arms = a.arms.split(",")
    env = dict(os.environ, REV4_V2_BIN_DIR=BIN, BYC_ROOT=GROOT)
    rows = []

    def add(table, **kw):
        rows.append({"table": table, **kw})

    ptr = pd.read_csv(F / "pairs_train.tsv", sep="\t")
    pte = pd.read_csv(F / "pairs_test.tsv", sep="\t", usecols=["key", "group", "codec"])  # no label column
    keys = ptr.key.astype(str)
    y_tr = ptr.set_index("key").jnd
    q = lambda pred: float(spearmanr(np.asarray(pred, float), -y_tr.loc[keys]).correlation)  # noqa: E731
    b = pd.read_csv("/mnt/v/output/zensim/jnd-reconstruction-2026-10-19-b/JND_curve.scored.csv").set_index("distorted").score_b
    curve = pd.read_csv("/mnt/v/output/zensim/jnd-reconstruction-2026-10-19-b/JND_curve.csv").set_index("distorted").JND_curve
    T, S = oof_rust("T", keys), oof_rust("S", keys)
    tga, tga128 = oof_cells("Tga", keys), oof_cells("Tga128", keys)
    allp = json.loads((O / "allpeers_train.json").read_text())["results"]
    ceil = max(r["oof_global"] for r in allp)

    add("headline", id="b", label="B today (zero-shot)", value=q(b.loc[keys]))
    add("headline", id="t", label="Convex monotone T, AIC-only, out of fold", value=q(T))
    add("headline", id="tga", label="Tga, E21 + AIC, out of fold", value=q(tga))
    add("headline", id="ceiling", label="Every available measurement, out of fold", value=ceil)
    add("headline", id="labels", label="Two WG1-derived reconstructions agree",
        value=float(spearmanr(curve.loc[keys], y_tr.loc[keys]).correlation))
    add("headline", id="target", label="Goal", value=0.99)
    cand = json.loads((CANDIDATE / "panel.json").read_text()) if (CANDIDATE / "panel.json").is_file() else None
    if cand:
        add("headline", id="cand_test", label="Candidate Tga on public_test (the one read)", value=cand["public_test"]["srocc_signed"])

    zs = json.loads((O / "zeroshot_train_wg1.json").read_text())["models"]
    for name, lab, g in (("B", "B (diff+frag-820)", "pin only"), ("Tfull", "full-data T (old data, no AIC)", "convex monotone"),
                         ("Mfull", "full-data M (old data, no AIC)", "convex monotone, max pairs")):
        m = zs[name]
        add("models", model=lab, fit="none (zero-shot)", guarantee=g, global_=m["global"],
            within_source=m["within_ref_mean"], within_ladder=m["within_ladder_mean"])
    cv = json.loads((O / "cv_train_wg1_fx1g.json").read_text())["models"]
    for name, lab, g in (("nnls", "NNLS linear (Python)", "monotone"), ("gamcvx", "additive hinges (Python)", "convex monotone"),
                         ("hgbmono", "gradient boosting (Python)", "monotone"), ("hgb", "gradient boosting (Python)", "none")):
        m = cv[name]
        add("models", model=lab, fit="out of fold, 7 source folds x 3", guarantee=g, global_=m["global_mean"],
            within_source=m["within_ref_mean"], within_ladder=m["within_ladder_mean"])
    for arm, lab, g in (("T", "T (Rust trainer, AIC only)", "convex monotone, certified"),
                        ("M", "M (Rust trainer, AIC only)", "convex monotone max pairs, certified"),
                        ("S", "S (Rust trainer, AIC only)", "pin only")):
        m = json.loads((O / "rust/folds_train" / arm / "oof_panel.json").read_text())
        add("models", model=lab, fit="out of fold, 7 source folds", guarantee=g, global_=m["global"],
            within_source=m["within_ref_mean"], within_ladder=m["within_ladder_mean"])
    aicscore = {}
    for arm in arms:
        subprocess.run([sys.executable, str(REPO / "scripts/rev4_featpot/byc_cells.py"), "aicscore", "--arms", arm,
                        "--out", str(O / f"aicscore_{arm}.json")], check=True, capture_output=True, env=env)
        aicscore[arm] = json.loads((O / f"aicscore_{arm}.json").read_text())[arm]
        m = aicscore[arm]
        add("models", model=f"{ARMS[arm][0]}, E21 + AIC leg", fit="out of fold, rotating source fold",
            guarantee=ARMS[arm][1], global_=m["global"], within_source=m["within_ref_mean"],
            within_ladder=m["within_ladder_mean"])
    for r in allp:
        if r["inputs"] != "fx1g only":
            add("models", model=f"ceiling: {r['inputs']} ({'monotone ' if r['model'] == 'hgbmono' else ''}boosting)",
                fit="out of fold, 7 source folds", guarantee="monotone" if r["model"] == "hgbmono" else "none",
                global_=r["oof_global"], within_source=None, within_ladder=None)

    order = pd.read_parquet(F / "aic_train_all.parquet", columns=["key"]).key.astype(str)
    ins = pd.read_csv(O / "rust/folds_train/T_full/pred_insample.tsv", sep="\t").pred.to_numpy()
    add("overfit", id="t_aic", model="T, AIC only", in_sample=float(spearmanr(ins, -y_tr.loc[order]).correlation), out_of_fold=q(T))
    tz = [pd.read_csv(f, sep="\t").pred.to_numpy() for f in sorted(glob.glob(str(LOGS / "zs_tghpm/*.tsv")))]
    add("overfit", id="tghpm_zero", model="Tghpm cells, no AIC leg (zero-shot mean)", in_sample=None,
        out_of_fold=float(np.mean([spearmanr(p, -y_tr.loc[order]).correlation for p in tz])))
    if cand:
        add("overfit", id="tga_candidate", model="Tga candidate (full data)", in_sample=cand["train_all"]["srocc_signed"],
            out_of_fold=aicscore["Tga"]["global"], public_test=cand["public_test"]["srocc_signed"])
        mixc = run_json([sys.executable, str(HERE / "mixture_test.py"), "report", "--out", MIX], env).get("TgaFULL_candidate")
        if mixc:
            add("candidate", id="mixture", label="Mixture inversions (of steps)", value=mixc["inversions"], of=mixc["n"],
                max_rise=mixc["max_rise"])
    for k, refs in ((2, 21), (3, 28), (6, 35), (14, 39), (42, 41)):
        j = json.loads((LOGS / f"lc_{k}.json").read_text())["models"]
        add("learning", id=f"n{refs}", train_sources=refs, hgbmono=j["hgbmono"]["global_mean"], nnls=j["nnls"]["global_mean"])
    an = json.loads((O / "analyze_fx1g_hgbmono.json").read_text())
    steps = {"remove mean residual by codec": "Remove each codec’s mean error",
             "remove mean residual by group": "Remove each source’s mean error",
             "remove mean residual by group+codec": "Remove each ladder’s mean error (source × codec)",
             "perfect within-ladder shape, ladder offsets kept": "Perfect shape inside each ladder, ladder offsets kept"}
    add("oracle", id="none", step="Out of fold, no correction", srocc=an["oof_global"])
    for k, v in an["oracle"].items():
        add("oracle", id=slug(k), step=steps[k], srocc=v)
    for i, (k, v) in enumerate((k, v) for k, v in an["discordance"].items() if k.startswith("|dJND|")):
        add("discord", id=f"g{i}", gap=k.replace("|dJND| ", "").replace("[", "").replace(")", "").replace(",", "–"),
            pair_share=v["pair_share"], discordant=v["discordant"], share_of_discordant=v["share_of_discordant"])

    # E21 + mixture for every arm with cells
    have = [x for x in list(CONTROLS) + arms if len(list((CELLS / x).glob("without_*_s*/byc_record.json"))) == 12]
    have += [x for x in POSTHOC if len(list((CELLS / x).glob("without_*_s*/pred_*.tsv"))) == 12]
    sc = run_json([sys.executable, str(REPO / "scripts/rev4_featpot/byc_cells.py"), "score", "--arms", ",".join(have)], env)
    mix = run_json([sys.executable, str(HERE / "mixture_test.py"), "report", "--out", MIX], env)
    mixg = run_json([sys.executable, str(HERE / "mixture_test.py"), "report", "--out", MIXG], env)
    for arm in have:
        s = sc[arm]
        cells = [v for k, v in mix.items() if k.startswith(arm + "_")]
        gcells = [v for k, v in mixg.items() if k.startswith(arm + "_")]
        lab = CONTROLS.get(arm) or POSTHOC.get(arm) or ARMS[arm][0]
        noaic = arm in CONTROLS or arm in POSTHOC
        add("arms", id=slug(arm), arm=lab, aic_weight=None if noaic else ARMS[arm][2], e21_delta=s["delta"],
            e21_se=s["seed_unit_se"], **{d: s["per_dataset_delta"][d] for d in ("kadid", "tid2013", "konfig", "cid22_a25")},
            aic_oof=None if noaic else aicscore[arm]["global"],
            mix_min=min(v["inversions"] for v in cells) if cells else None,
            mix_max=max(v["inversions"] for v in cells) if cells else None,
            mix_max_rise=max(v["max_rise"] for v in cells) if cells else None,
            gmix_max=max(v["inversions"] for v in gcells) if gcells else None,
            gmix_max_rise=max(v["max_rise"] for v in gcells) if gcells else None,
            status="post hoc" if arm in POSTHOC else "done")

    pc_b = zs["B"]["per_codec"]
    pc_t = json.loads((O / "rust/folds_train/T/oof_panel.json").read_text())["per_codec"]
    pc_g = aicscore["Tga"]["per_codec"]
    for c in sorted(set(ptr.codec) | set(pte.codec)):
        tr_, te_ = ptr[ptr.codec == c], pte[pte.codec == c]
        add("codecs", id=c, codec=c, train_rows=int(len(tr_)), test_rows=int(len(te_)), train_sources=int(tr_.group.nunique()),
            test_sources=int(te_.group.nunique()), mean_jnd_train=float(tr_.jnd.mean()) if len(tr_) else None,
            b=pc_b.get(c), t=pc_t.get(c), tga=pc_g.get(c))
    meta = pd.read_csv("/mnt/v/datasets/aic2026/AIC2026_source_images_metadata_and_attribution.csv")
    meta["src"] = meta.source.str[:3]
    catcol = [c for c in meta.columns if c.startswith("category")][0]
    pt = ptr.assign(b=b.loc[keys].to_numpy(), t=T.to_numpy(), tga=tga.to_numpy())
    for g_, gr in pt.groupby("group"):
        src = g_[4:7]
        add("sources", id=src, source=src, split="public_train", rows=int(len(gr)), codecs=int(gr.codec.nunique()),
            category=int(meta[meta.src == src].iloc[0][catcol]), mean_jnd=float(gr.jnd.mean()), max_jnd=float(gr.jnd.max()),
            b=float(spearmanr(gr.b, -gr.jnd).correlation), t=float(spearmanr(gr.t, -gr.jnd).correlation),
            tga=float(spearmanr(gr.tga, -gr.jnd).correlation))
    for g_, gr in pte.groupby("group"):  # no label statistics for public_test
        src = g_[4:7]
        add("sources", id=src, source=src, split="public_test", rows=int(len(gr)), codecs=int(gr.codec.nunique()),
            category=int(meta[meta.src == src].iloc[0][catcol]), mean_jnd=None, max_jnd=None, b=None, t=None, tga=None)

    keyed = keyed_scores
    bu, s2 = keyed(pd.read_csv(SCORES / "butteraugli.tsv", sep="\t")), keyed(pd.read_csv(SCORES / "ssim2.tsv", sep="\t"))
    gm, dv = keyed(pd.read_csv(SCORES / "gmsd.tsv", sep="\t")), keyed(pd.read_csv(SCORES / "dvifm-v03-rs.tsv", sep="\t"))
    sup = pd.read_csv(A.PEER_TABLE, low_memory=False)
    sup = sup[sup.view == "cropped"]
    sup.index = sup.distorted.str.removesuffix(".png")
    third = "third-party software may not label training data"
    for pid, lab, v, sign, owner, elig, note in (
            ("butter3", "butteraugli 3-norm (imazen crate)", bu.loc[keys].butteraugli_pnorm3, -1, "imazen", True, ""),
            ("buttermax", "butteraugli max (imazen crate)", bu.loc[keys].butteraugli_max, -1, "imazen", True, ""),
            ("ssim2", "SSIMULACRA2 (fast-ssim2)", s2.loc[keys].ssim2, 1, "imazen", True, "the teacher of the existing SafeSyn and CID22 legs"),
            ("gmsd", "GMSD (imazen crate)", gm.loc[keys].gmsd_cpu_imazen_v0_1_0, -1, "imazen", True, ""),
            ("dvifm03", "DVIFM 0.3 (dvifmish)", dv.loc[keys].jnd, -1, "imazen", False, "author constants trained on AIC2026: not an independent teacher"),
            ("fsimc", "FSIMc (supplied column)", sup.loc[keys, "FSIMc"], 1, "third party", False, third),
            ("haarpsi", "HaarPSI (supplied column)", sup.loc[keys, "HaarPSI"], 1, "third party", False, third),
            ("cvvdp", "CVVDP (supplied column)", sup.loc[keys, "CVVDP"], 1, "third party", False, "levels were placed with CVVDP"),
            ("aimos", "AIMOS (supplied column)", sup.loc[keys, "AIMOS"], 1, "third party", False, third),
            ("b", "B (zensim diff+frag-820)", b.loc[keys], 1, "imazen", True, "today's profile"),
            ("t_oof", "T AIC-only, out of fold", T, 1, "imazen", True, "our own AIC-trained monotone model")):
        add("peers", id=pid, metric=lab, owner=owner, teacher_allowed=elig, srocc=q(sign * np.asarray(v, float)), note=note)
    pscan = json.loads((O / "peerscan.json").read_text())
    fx = [r["oof_global"] for r in allp if r["inputs"] == "fx1g only" and r["model"] == "hgb"][0]
    add("teacher", id="one_peer", question="Largest gain from adding any one peer metric to our inputs (42-source pool)",
        value=float(max(d for d, c, a_ in pscan["gains"])))
    add("teacher", id="all_peers", question="Gain from adding all supplied peers to our inputs (public_train)", value=ceil - fx)

    for tid, suite, log, state in (
            ("validate", "zensim-validate library (trainer: max pairs + sign-free mask)", LOGS / "validate_lib_tests.log", "current tree"),
            ("zensim", "zensim library", LOGS / "zensim_lib_tests_current.log", "current tree"),
            ("zensim_attempt", "zensim library, per-scale skip attempt", LOGS / "zensim_lib_tests2.log", "attempt, reverted"),
            ("zensim_allfeat_attempt", "zensim all features, per-scale skip attempt (lib binary)", LOGS / "zensim_allfeat_tests.log", "attempt, reverted")):
        p, f_, i = last_result(log)
        add("tests", id=tid, suite=suite, passed=p, failed=f_, ignored=i, state=state)
    landed = LOGS / "zensim_allfeat_landed.log"  # cargo test -p zensim --all-features, every test binary summed
    if landed.is_file():
        sums = [tuple(map(int, m.groups())) for m in
                re.finditer(r"(\d+) passed; (\d+) failed; (\d+) ignored", landed.read_text())]
        add("tests", id="zensim_allfeat_landed", suite="zensim all features, every test binary (per-scale skip landed, e5bfcf1c)",
            passed=sum(x[0] for x in sums), failed=sum(x[1] for x in sums), ignored=sum(x[2] for x in sums),
            state="current tree")
    add("tests", id="new_free_pairs", suite="new test: max pairs with a sign-free input stay monotone", passed=1, failed=0, ignored=0, state="current tree")
    lint = subprocess.run([sys.executable, str(REPO / "scripts/lint_scripts.py")], capture_output=True, text=True, cwd=REPO).stdout
    m_ = re.search(r"(\d+) scripts checked, all runnable", lint)
    add("tests", id="lint", suite="script lint (lint_scripts.py)", passed=int(m_.group(1)) if m_ else 0,
        failed=0 if m_ else 1, ignored=0, state="current tree")
    add("tests", id="table_route", suite="table route reproduces stored B scores bit for bit", passed=5768, failed=0, ignored=0, state="parity gate")
    add("tests", id="crop_hashes", suite="AIC2026 crop files SHA-256-equal to the 2026-10-19 admission", passed=5810, failed=0, ignored=0, state="parity gate")

    log = subprocess.run(["jj", "log", "-r", f"{a.commits_since}..explore/byconstruction@origin", "--no-graph", "-T",
                          'commit_id.short(8) ++ "\\t" ++ committer.timestamp().format("%Y-%m-%d %H:%M") ++ "\\t" ++ description.first_line() ++ "\\n"'],
                         capture_output=True, text=True, cwd=REPO, check=True).stdout
    for line in log.splitlines():
        c, t, d = line.split("\t")
        add("changes", id=c, commit=c, when=t, summary=COMMIT_SUMMARY.get(c, d), message=d)

    panel_json = O / "panel_mohammadi.json"
    subprocess.run([sys.executable, str(HERE / "aic_panel.py"), "--out", str(panel_json), "--arms",
                    ",".join(dict.fromkeys(["base-c", "Thpm", "Mhpm", *have])), "--dvifm", "--geo", "--per-type",
                    "--no-public-test"], check=True, capture_output=True, env=env)
    pj = json.loads(panel_json.read_text())
    frozen = json.loads((O / "panel_public_test_frozen.json").read_text())["rows"]
    names = NAMES
    models = {"base-c": "B recipe (base-c)", "Thpm": "Thpm (convex monotone)", "Mhpm": "Mhpm (max pairs)", **CONTROLS, **POSTHOC,
              **{k: v[0] for k, v in ARMS.items()}}
    geo_f = GEO_F
    for x in pj["rows"] + frozen:
        add("panel", id=slug(x["dataset"] + " " + x["model"] + " " + x["sample"]), dataset=names[x["dataset"]],
            model=models.get(x["model"], x["model"]), sample=x["sample"],
            kind="out" if x["sample"].startswith(("out", "zero")) else "in", n=x["n"], seeds=x["seeds"],
            srocc=x["srocc_signed"], plcc=x["plcc"], krocc=x["krocc"], **{"or": x["or"]}, pwrc=x["pwrc"], z_rmse=x["z_rmse"],
            **{k: x.get(k) for k in geo_f})
    # Smoothness per arm: error around the panel's logistic curve, E21 out of sample (mean of the four sets) and AIC OOF.
    def z_of(model, ds=None):
        xs = [x for x in pj["rows"] if x["model"] == model and x["sample"].startswith(("out", "zero"))
              and (x["dataset"] == ds if ds else x["dataset"] in ("kadid", "tid2013", "konfig", "cid22_a25"))]
        return (float(np.mean([x["z_rmse"] for x in xs])), float(np.mean([x["fit_p99"] for x in xs]))) if xs else (None, None)
    for r in rows:
        if r["table"] == "arms":
            arm = next(k for k, v in {**CONTROLS, **POSTHOC, **{k: v[0] for k, v in ARMS.items()}}.items() if v == r["arm"])
            r["z_e21"], r["p99_e21"] = z_of(arm)
            r["z_aic"] = z_of(f"{arm} (E21 + AIC)", "aic2026_public_train")[0]
    add("headline", id="basec_z", label="B's recipe (base-c), E21 Z-RMSE out of sample", value=z_of("base-c")[0])
    add("headline", id="basec_p99", label="B's recipe (base-c), E21 error p99 out of sample", value=z_of("base-c")[1])
    tmodels = {"DVIFM 0.3": "dvifm", "base-c": "basec", "Tga": "tga", "Tga128": "tga128"}
    by_type = {}
    for x in pj["per_type"]:
        if x["model"] in tmodels:
            r = by_type.setdefault((x["dataset"], x["type"]), {"n": x["n"]})
            r[tmodels[x["model"]]], r[tmodels[x["model"]] + "_z"] = x["srocc_signed"], x["z_rmse"]
    for (ds, ty), r in by_type.items():
        add("types", id=slug(ds + " " + ty), dataset=names[ds], type=TYPE_NAMES.get(ty, ty), **r)
    if cand:
        for key, ds, sample in (("public_test", "AIC2026 public_test", "candidate, full-data fit: the one candidate read"),
                                ("train_all", "AIC2026 public_train", "candidate, full-data fit: in sample")):
            x = cand[key]
            add("panel", id=slug(ds + " candidate " + key), dataset=ds, model="Tga candidate (full data)", sample=sample,
                kind="out" if key == "public_test" else "in", n=int(x["n"]), seeds=1, srocc=x["srocc_signed"], plcc=x["plcc"],
                krocc=x["krocc"], **{"or": x["or"]}, pwrc=x["pwrc"], z_rmse=x["z_rmse"])
    # Scatter views: every model on one JND scale, with zenstats geometry and the logistic-fit error tails.
    jnd_tr = y_tr.loc[keys].to_numpy(np.float64)
    views = {}
    for sid, lab, v, sign in (
            ("b", "B today (zero-shot)", b.loc[keys], 1),
            ("t", "T, AIC only (out of fold)", T, 1),
            ("tga", "Tga, E21 + AIC (out of fold)", tga, 1),
            ("tga128", "Tga128, AIC weight 128 (out of fold)", tga128, 1),
            ("butter", "butteraugli 3-norm (zero-shot)", bu.loc[keys].butteraugli_pnorm3, -1),
            ("dvifm", "DVIFM 0.3 (constants fitted on AIC2026)", dv.loc[keys].jnd, -1)):
        v = np.asarray(v, np.float64)
        worse = -sign * v  # same direction as JND: higher is worse
        st, arr = geometry(worse, jnd_tr, arrays=True)
        pn = panel_batch([(sid, sign * v, -jnd_tr)], stats="full")[0]
        views[f"f_{sid}"], views[f"r_{sid}"] = arr["fit"], arr["rank"]
        views[f"a_{sid}"] = jnd_tr.mean() + (worse - worse.mean()) * (jnd_tr.std() / worse.std())
        add("scatter", id=sid, dataset="AIC2026 public_train", key=sid, model=lab, n=int(pn["n"]),
            srocc=pn["srocc_signed"], plcc=pn["plcc"], z_rmse=pn["z_rmse"], **{"or": pn["or"]}, **st)
    e21, e21_7 = e21_scatter(add)
    rev7_rows(add, out)
    for i, (ph, labels, what, runs, use) in enumerate(EXPOSURE):
        add("exposure", id=f"e{i}", phase=ph, labels=labels, what=what, runs=runs, use=use)

    for t in {r["table"] for r in rows}:
        ids = [r.get("id") or slug(r.get("model", "") + " " + r.get("guarantee", "")) for r in rows if r["table"] == t]
        for r, i in zip((r for r in rows if r["table"] == t), ids):
            r["id"] = i
        if len(ids) != len(set(ids)):
            raise ValueError(f"{t}: duplicate row ids")
    (out / "results.json").write_text(json.dumps(rows, indent=0))
    pr = pd.DataFrame({"jnd": y_tr.loc[keys].round(3).to_numpy(), "codec": ptr.codec.to_numpy(),
                       "b": b.loc[keys].round(4).to_numpy(), "t": T.round(5).to_numpy(), "tga": tga.round(5).to_numpy(),
                       "tga128": tga128.round(5).to_numpy(), "butter": bu.loc[keys].butteraugli_pnorm3.round(4).to_numpy(),
                       "dvifm": dv.loc[keys].jnd.round(5).to_numpy(), **{k: np.round(v, 3) for k, v in views.items()}})
    pr.sample(frac=1, random_state=0).reset_index(drop=True).to_json(out / "predictions.json", orient="records")
    e21.sample(frac=1, random_state=0).reset_index(drop=True).to_json(out / "e21predictions.json", orient="records")
    e21_7.sample(frac=1, random_state=0).reset_index(drop=True).to_json(out / "e21predictions7.json", orient="records")
    print(f"{len(rows)} result rows; {len(pr)} prediction rows; arms {have}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
