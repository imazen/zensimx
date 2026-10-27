"""One Instrument v2 cell (R915 sampling): (arm spec, head, held-out human source, seed index).

Governing record: benchmarks/rev4_featpot_v2_amendment_2026-09-30.md, revision R1. One training run:
SafeSyn and CID22-train teacher legs (MSE + rank, within-reference pairs) and the human leg of the four
non-held-out sources (rank only, within-reference), each with its reference-disjoint dev group; the trainer
exports the best epoch by the weighted mean dev geomean3 (R915's --val-policy mean). The arm is the
--keep-features subset of the shared wide table of its variant. Predict and score the held-out source.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch  # noqa: E402
from v2_common import (EPOCH_RULE, EPOCHS, FITBIN, HEADS, HIDDEN, HUMAN_VAL_WEIGHT, N_SEEDS, NOMINAL_WEIGHT, PAIRS_PER_EPOCH, arm_columns, block_spec, recipe_of, split_weight,
                       PANEL, REPLAY, REPO, SOURCE_ORDER, TEACHERS, TRAINER, V2, WIDTH, acceptance_weight,
                       parse_spec, seeds, selection_id, sha, table_path)

WIDE_SCHEMAS = ("rev4-featpot-v2-wide-v2", "rev4-featpot-v2c-wide-v1")
# Under the final-epoch rule no per-epoch dev score selects anything, so the dev panels run every 17th epoch (17 divides
# EPOCHS - 1 = 119: epoch 0, every 17th, and the final epoch, which is also where --dump-checkpoints-every EPOCHS-1
# fires). Evaluation is pure (no RNG, no state, LR depends on the epoch index only), so the trajectory and the final
# weights do not depend on this. best_dev still needs every epoch.
LOG_EVERY = 17 if EPOCH_RULE == "last" else 1
if (EPOCHS - 1) % LOG_EVERY:
    raise ValueError(f"LOG_EVERY={LOG_EVERY} must divide EPOCHS-1={EPOCHS - 1}")
EPOCH_RE = re.compile(r"epoch\s+(\d+)\s+\|.*?val\(geomean3\)=([+-]?\d+\.\d+)")


def run(cmd: list[str], log: Path) -> None:
    with log.open("w") as stream:
        stream.write("$ " + " ".join(cmd) + "\n")
        stream.flush()
        proc = subprocess.run(cmd, text=True, stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"{cmd[0]} rc={proc.returncode}; see {log}")


def checked(record: dict) -> Path:
    path = table_path(record)
    if sha(path) != record["sha256"] or sha(Path(f"{path}.manifest.json")) != record["manifest_sha256"]:
        raise ValueError(f"{path}: table changed after its receipt")
    return path


def refs_of(path: Path) -> list[str]:
    return pq.read_table(path, columns=["ref_basename"])["ref_basename"].to_pylist()


def predict(bake: Path, table: Path, out: Path, *, research_palette_cached=False) -> np.ndarray:
    from v2_common import dense_bake, table_revision
    if table_revision(table) >= 5:  # Rev5 tables carry NaN absent slots: score with the dense bake (bit-identical by gate)
        bake = dense_bake(bake, out.parent, research_palette_cached=research_palette_cached)
    run([str(FITBIN), "predict", "--bake", str(bake), "--corpus", str(table), "--score-units",
         "--out", str(out), *(["--research-palette-cached"] if research_palette_cached else [])], out.with_suffix(".log"))
    result = pd.read_csv(out, sep="\t")
    if result.columns.tolist() != ["row_idx", "pred"] or not np.array_equal(
            result.row_idx.to_numpy(), np.arange(len(result))):
        raise ValueError(f"{out}: unexpected prediction row order")
    pred = result.pred.to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError(f"{out}: nonfinite prediction")
    return pred


def admit_coverage_keep(core_spec: str, keep: list[int], strict_admission: bool) -> None:
    """The coverage pool carries ids below the ordinal width; a keep list reaching past it needs a pool that declares
    those ids: the by-construction derived pool (ring_root.py globaff declares its research columns) or the E32
    palette extension. Shared by the LODO cell and the full-data fit (v2_confirm_fit.py)."""
    import v2_teacher
    if max(keep) < v2_teacher.ORDINAL_WIDTH:
        return
    if not strict_admission:
        raise ValueError(f"{core_spec}: the coverage pool has no f{v2_teacher.ORDINAL_WIDTH}+ (NaN); refusing this keep list")
    pool_decl = json.loads((V2 / "e15" / f"{Path(v2_teacher.POOL_NAME).name}.manifest.json").read_text())
    extra = {k for k in keep if k >= v2_teacher.ORDINAL_WIDTH}
    if not ("byc_derived_from" in pool_decl and extra <= set(pool_decl.get("byc_globaff_ids", []))):
        from e32_palette import coverage_extension
        coverage_extension(V2 / "e15" / Path(v2_teacher.POOL_NAME).name, keep)


def strict_group_input_roots(groups: list) -> tuple[Path, ...]:
    from v2_common import table_input_roots
    roots = []
    for name, path, _, _, _ in groups:
        bound = table_input_roots(path)
        if name == "upiq380":
            bound = (*bound, Path(path).resolve().parent)
        if name == "aic_public_train":
            bound = (*bound, Path(path).resolve().parent)
        if not bound:
            raise ValueError(f"{name}: strict outputs require immutable input root bindings; regenerate a fresh admission view")
        roots.extend(bound)
    return tuple(dict.fromkeys(roots))


def train_command(groups: list, init_seed: int, sample_seed: int, width: int, keep_file: Path, head: str,
                  out: Path, recipe: dict | None = None, *, strict_admission: bool = False,
                  data_role_decision: Path | None = None,
                  upiq_label_disposition: Path | None = None,
                  subjective_authority: tuple | None = None) -> list[str]:
    """The trainer argv of one v2 cell; `groups` = (name, path, train_weight, val_weight, mode). Shared with
    v2_confirm_fit.py, which trains on the same recipe over the full-data legs."""
    if subjective_authority is not None:
        from subjective_training_leg import command_binding
        authority, pin, cell, _ = subjective_authority
        command_binding(authority,pin,cell,init_seed,sample_seed,width,head,recipe)
    if strict_admission:
        strict_training_groups(groups, data_role_decision, upiq_label_disposition, subjective_authority)
        from v2_common import refuse_immutable_output
        roots = strict_group_input_roots(groups)
        for path in (out, keep_file):
            refuse_immutable_output(path, roots)
    # Strict defaults use this checkout's trainer, not the pinned historical
    # fit-cell binary. An explicit existing binary-directory override wins.
    trainer = (Path(os.environ.get("REV4_V2_BIN_DIR", str(REPO / "target/debug"))) / "zensim_mlp_train"
               if strict_admission else TRAINER)
    cmd = [str(trainer)]
    if upiq_label_disposition is not None:
        if not strict_admission:
            raise ValueError("E31 requires strict admission")
        cmd += ["--upiq-label-disposition", str(upiq_label_disposition)]
    if subjective_authority is not None:
        if not strict_admission:
            raise ValueError("unnamed subjective leg requires strict admission")
        authority, pin, cell, _ = subjective_authority
        cmd += ["--subjective-authority", str(authority), "--subjective-authority-sha256", pin,
                "--subjective-cell", cell]
        continuation = json.loads(Path(authority).read_text())["cells"][cell].get("continuation")
        if continuation:
            cmd += ["--init-from", continuation["checkpoint"], "--lr", str(continuation["lr"]),
                    "--lr-schedule", "constant"]
    aic = [g for g in groups if g[0] == "aic_public_train"]
    if aic:
        authority = Path(aic[0][1]).parent / "_MANIFEST.json"
        cmd += ["--aic-public-train-authority", str(authority), "--aic-public-train-authority-sha256", sha(authority)]
    for name, path, tw, vw, mode in groups:
        cmd += ["--group", f"{name}:{path}:{tw!r}:{vw!r}:{mode}"]
    recipe = recipe or {}
    if recipe.get("smooth_hinge"):
        if head != "N":
            raise ValueError("sh1 requires the plain N head")
        cmd += ["--smooth-hinge"]
    cmd += ["--target-column", "human_score", "--target-scale", "1", "--hidden", str(recipe.get("hidden", HIDDEN)),
            "--epochs", str(EPOCHS), "--pairs-per-epoch", str(PAIRS_PER_EPOCH),
            "--init-seed", str(init_seed), "--sample-seed", str(sample_seed),
            "--pair-sampling", "uniform", "--max-features", str(width), "--keep-features", str(keep_file),
            "--mse-weight", "1", "--early-stop-patience", "0", "--val-policy", "mean",
            "--val-aggregate", "geomean3", "--out-dtype", "f32", "--log-every", str(LOG_EVERY), "--no-auto-eval"]
    if not strict_admission:
        cmd += ["--historical-replay", REPLAY]
    cmd += ["--out", str(out)]
    if "group_l1" in recipe:  # design log E8: proximal group lasso on layer-1 input rows
        cmd += ["--group-l1", repr(recipe["group_l1"])]
    if "ssim2_recipe" in recipe:
        from e28_recipe import read_pin
        pin = read_pin()
        active = [name for name, _, tw, _, _ in groups if tw > 0 and name in pin["arms"][recipe["ssim2_recipe"]]["pooled_legs"]]
        cmd += ["--pooled-rank-share", str(pin["pooled_rank_share"]), "--pooled-pearson-weight", str(pin["pooled_pearson_weight"])]
        for name in active:
            cmd += ["--pooled-leg", name]
    if "hdr_consensus" in recipe:
        from e29_consensus import trainer_options
        cmd += trainer_options(groups, recipe["hdr_consensus"])
    if recipe.get("derived_inputs") in ("fx1", "fx2", "fx1nr", "fx1g", "fx1gs1", "fx1gg", "fx1gy", "fx1o", "fx1ns0"):
        from v2_common import derived_declaration
        declaration_path, _, decl = derived_declaration(recipe["derived_inputs"])
        if [int(x) for x in keep_file.read_text().split()] != decl["direct"]:
            raise ValueError("fx1: the kept columns must be exactly the declaration's direct ids")
        cmd += ["--derived-inputs", str(declaration_path)]
    if head == "N":
        cmd.append("--nonneg-distance")
    if recipe.get("mono_all") or recipe.get("thresholds") or recipe.get("ramps") or recipe.get("free_ramps"):
        if head != "N" or recipe.get("monotone_cbc") or recipe.get("smooth_hinge") or recipe.get("steer_pairs"):
            raise ValueError("by-construction tokens require the plain N head without E34 L/K or sh1")
        if (recipe.get("ramps") or recipe.get("free_ramps")) and not (recipe.get("mono_all") and recipe.get("thresholds")):
            raise ValueError("rp<N>/fr<N> require mono and thr")
        if recipe.get("mono_all"):
            cmd.append("--monotone-cbc")
        if recipe.get("thresholds"):
            cmd.append("--nonneg-thresholds")
        if recipe.get("ramps"):
            cmd += ["--monotone-ramps", str(recipe["ramps"])]
        if recipe.get("free_ramps"):
            cmd += ["--free-ramps", str(recipe["free_ramps"]), "--free-budget", repr(recipe["free_budget"])]
    if recipe.get("free_mask"):
        if not recipe.get("mono_all") or recipe.get("ramps") or recipe.get("free_ramps") or recipe.get("gated_products"):
            raise ValueError("a free mask applies to the plain mono recipe, with or without max pairs")
        cmd += ["--monotone-feature-mask", str(REPO / "scripts/byconstruction/masks" / f"{recipe['free_mask']}.tsv")]
        if recipe.get("drop_masked"):
            cmd.append("--monotone-strict")
    if recipe.get("max_pairs"):
        if not (recipe.get("mono_all") and recipe.get("thresholds")) or recipe.get("gated_products"):
            raise ValueError("mp<N> requires mono and thr, without gate")
        cmd += ["--max-pairs", str(recipe["max_pairs"])]
    if recipe.get("gated_products"):
        if not (recipe.get("mono_all") and recipe.get("derived_inputs") == "fx1"):
            raise ValueError("gate requires mono and fx1")
        cmd.append("--monotone-gated-products")
    if recipe.get("monotone_cbc"):
        if head != "N" or recipe.get("derived_inputs") or recipe.get("hidden") != 1:
            raise ValueError("E34 L requires A's plain N H1 recipe")
        cmd.append("--monotone-cbc")
    if recipe.get("steer_pairs") is not None or recipe.get("steer_lambda", 0) != 0:
        import math
        weight = recipe.get("steer_lambda", 0)
        if (head != "N" or recipe.get("monotone_cbc") or not recipe.get("steer_pairs")
                or not math.isfinite(weight) or weight <= 0):
            raise ValueError("E34 K requires N, pairs and positive finite lambda; no L projection")
        cmd += ["--steer-pairs", str(recipe["steer_pairs"]), "--steer-lambda", repr(weight)]
    from e36_training import command_flags
    cmd += command_flags(recipe,trainer)
    from matched_codec import command_flags as codec_flags
    cmd += codec_flags(recipe, strict_admission=strict_admission, head=head)
    return cmd


def e34_options(args, recipe):
    """Explicit E34 options leave every legacy recipe dictionary unchanged."""
    recipe = dict(recipe)
    if args.monotone_cbc:
        recipe["monotone_cbc"] = True
    if args.steer_pairs is not None or args.steer_lambda != 0:
        recipe.update(steer_pairs=str(args.steer_pairs) if args.steer_pairs else None,
                      steer_lambda=args.steer_lambda)
    from e36_training import options
    return options(args,recipe)


# Formula revision 7 (raw ringing; owner 2026-10-25): every measured slot re-extracted at revision 7
# (scripts/byconstruction/ring_root.py --arm rev7); its tables declare formula_revision 7.
REV7_FSID = "basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af"
BYC_DERIVED_FSIDS = {"basic+peaks+v2@w1825/rev5_localwin_ringhpm#36c3f3af",
                     "basic+peaks+v2@w1825/rev5_localwin_linpyr#36c3f3af",
                     "basic+peaks+v2@w2337/rev5_localwin_orient#24a93918",
                     "basic+peaks+v2@w1891/rev5_localwin_ringhpm_globaff#d56ad477",
                     "basic+peaks+v2@w2301/rev5_localwin_ringhpm_globaffgate#45e0e84c",
                     "basic+peaks+v2@w2055/rev5_localwin_ringhpm_globaffgatey#7b645e91",
                     REV7_FSID}


def strict_training_groups(groups: list, data_role_decision: Path | None = None,
                  upiq_label_disposition: Path | None = None,
                  subjective_authority: tuple | None = None) -> list[dict]:
    """Require declared full provenance and a separately supplied human-role decision.

    This checks metadata/label-free keys before returning trainer argv. The
    Rust trainer remains the authoritative registered-slot/revision admission
    owner and refuses incompatibilities without historical replay.
    """
    from v2c_wide import safe_path
    from v2_teacher import key_path, row_keys_sha
    from v2_human_role import decision_record, human_declaration, human_keys, bank_members
    decision = None
    if data_role_decision is not None:
        raw = json.loads(safe_path(data_role_decision).read_text())
        decision = decision_record(data_role_decision, raw.get("source_receipt_sha256"))
    subjective = {}
    if subjective_authority is not None:
        from subjective_training_leg import admit
        authority, pin, cell, selected = subjective_authority
        subjective = admit(authority, pin, cell, groups, selected)
    records = []
    aic_pending = []
    checked_metadata = []
    for name, path, train_w, val_w, mode in groups:
        path = safe_path(path)
        sp = Path(f"{path}.manifest.json")
        d = json.loads(sp.read_text())
        if d.get("schema") == "aic2026-public-train-rev7-v1":
            from v2_human_role import aic_public_train_declaration
            if name != "aic_public_train" or train_w <= 0 or val_w != 0 or mode != "rank":
                raise ValueError("D6 requires its fit-only pooled-rank group")
            kp = safe_path(d["keys_path"])
            if sha(kp) != d["keys_sha256"]:
                raise ValueError("D6 key pin changed")
            aic_public_train_declaration(d, json.loads(kp.read_text()))
            aic_pending.append((path, d))
            records.append(dict(name=name, manifest=d, manifest_sha256=sha(sp)))
            continue
        if d.get("schema") in ("subjective-training-leg-v1", "subjective-training-leg-v2",
                               "subjective-training-leg-v3"):
            if subjective.get(name) != d:
                raise ValueError("unnamed leg requires independently pinned authority")
            records.append(dict(name=name, manifest=d, manifest_sha256=sha(sp)))
            continue
        if name == "upiq380":
            from e31_training import fit_metadata
            from e21_cheap_recipe import columns
            if upiq_label_disposition is None:
                raise ValueError("E31 requires a label-provenance disposition")
            records.append(fit_metadata(path, upiq_label_disposition, columns("by_v2fy")))
            continue
        if d.get("source") == "UPIQ-380":
            raise ValueError("native UPIQ must use the registered E31 group")
        if d.get("data_role_decision_required") or d.get("human_sources"):
            human_declaration(d, decision)
        if d.get("study") == "E29":
            from e29_consensus import admit_metadata
            admit_metadata(path, d)
            checked_metadata.append((name, path, sp, d, pq.read_table(key_path(path))))
            continue
        bank_members(d)
        palette = "research_palette" in d
        if palette:
            from e32_palette import admit_declaration
            projection = admit_declaration(d)
            if ((train_w > 0 and projection["role"].endswith("development"))
                    or (train_w == 0 and val_w > 0 and projection["role"].endswith("fit"))):
                raise ValueError("E32 fit/development role disagrees with group weights")
        # By-construction exploration: a derived root (scripts/byconstruction/ring_root.py) re-extracts named columns
        # under a measurement arm and declares its own era; every other measured column was checked bit for bit.
        derived = "byc_derived_from" in d and d.get("feature_set_id") in BYC_DERIVED_FSIDS
        if ((not palette and not derived and d.get("feature_set_id") != "basic+peaks+v2@w1825/rev5_localwin#36c3f3af")
                or d.get("formula_revision") != (7 if d.get("feature_set_id") == REV7_FSID else 5)
                or not d.get("decoder_era")
                or not d.get("table_sha256") or not d.get("row_selection_sha256")):
            raise ValueError(f"{name}: strict admission requires bound Rev5 table provenance")
        if not d.get("data_role_decision_required") and d.get("data_role") not in ("TRAIN oracle teacher", "TRAIN ordinal KADIS source_id%10<8; no human labels"):
            raise ValueError(f"{name}: strict admission needs an explicit permitted data role")
        kp = key_path(path)
        if sha(kp) != d.get("keys_sha256"):
            raise ValueError(f"{name}: admitted row key file changed")
        keys = pq.read_table(kp)
        if palette:
            from e32_palette import admit_keys
            admit_keys(path, d, keys)
        if not palette:
            from v2_teacher import admit_ordered_keys
            admit_ordered_keys(path, d, keys)
        if d.get("data_role_decision_required"):
            human_keys(keys, d)
        checked_metadata.append((name, path, sp, d, keys))
    if upiq_label_disposition is not None and not any(g[0] == "upiq380" for g in groups):
        raise ValueError("E31 disposition supplied without UPIQ fit")
    for name, path, tw, vw, mode in groups:
        if name == "upiq380":
            from e31_training import TABLE_SHA
            if tw != 4.34410740924913 or vw != 0 or mode != "rank" or sha(path) != TABLE_SHA:
                raise ValueError("E31 requires the registered pooled rank-only nominal-4 fit group")
    # Every population is admitted before the first table payload is hashed/read.
    for path, d in aic_pending:
        if sha(path) != d["table_sha256"]:
            raise ValueError("D6 payload pin changed")
    for name, path, sp, d, keys in checked_metadata:
        if d["table_sha256"] != sha(path):
            raise ValueError(f"{name}: strict admission requires bound Rev5 table provenance")
        refs = refs_of(path)
        ref_col = "ladder" if "ladder" in keys.column_names else "ref_basename"
        if row_keys_sha(keys) != d.get("row_keys_sha256") or refs != keys[ref_col].to_pylist():
            raise ValueError(f"{name}: admitted row key identity/order differs")
        records.append({"name": name, "table_sha256": d["table_sha256"], "manifest_sha256": sha(sp),
                        "keys_sha256": d["keys_sha256"], "row_keys_sha256": d["row_keys_sha256"],
                        "row_selection_sha256": d["row_selection_sha256"],
                        "data_role_decision_sha256": sha(data_role_decision) if d.get("data_role_decision_required") else None})
    if subjective:
        from subjective_training_leg import payloads
        payloads(subjective, groups)
    if not records:
        raise ValueError("strict admission needs at least one training/development table")
    return records


def train_and_select(groups: list, init_seed: int, sample_seed: int, width: int, keep_file: Path, head: str,
                     dest: Path, recipe: dict | None = None, *, strict_admission: bool = False,
                     data_role_decision: Path | None = None,
                  upiq_label_disposition: Path | None = None,
                     subjective_authority: tuple | None = None,
                     research_extra: tuple[list[str], str] | None = None) -> tuple[Path, dict[int, float], dict]:
    """Train one cell and return (selected bake, dev curve, selection record) under EPOCH_RULE.

    best_dev: the trainer's own best-validation bake (refit/best.bin); the recorded epoch is the argmax of the log's
    4-decimal curve, which can differ from the trainer's full-precision pick on ties (label only; the bake is the
    trainer's). last: the trainer also dumps the final epoch's weights (--dump-checkpoints-every EPOCHS-1 fires at epoch 0
    and EPOCHS-1) and that checkpoint is the selected bake (refit/last.bin)."""
    # Guard derived paths even when this owner is called without either CLI.
    if strict_admission:
        from v2_common import refuse_immutable_output
        roots = strict_group_input_roots(groups)
        for path in (dest, dest / "refit", dest / "refit/best.bin", dest / "refit/last.bin",
                     dest / "ckpt", dest / "train.log", keep_file):
            refuse_immutable_output(path, roots)
    # Admission precedes output creation and the Rust trainer's payload reads.
    e36_filter_record = None
    if recipe and recipe.get('e36'):
        if not strict_admission:
            raise ValueError('E36 requires strict original-parent admission')
        strict_training_groups(groups,data_role_decision,upiq_label_disposition,subjective_authority)
        from e36_training import filter_groups
        e36_filter_record=filter_groups(groups,recipe,Path(os.environ.get('TMPDIR') or dest))
        if e36_filter_record is not None:
            (dest/'e36-base-filter.json').write_text(json.dumps(e36_filter_record,indent=2)+'\n')
    cmd = train_command(groups, init_seed, sample_seed, width, keep_file, head, dest / "refit" / "best.bin", recipe,
                        strict_admission=strict_admission, data_role_decision=data_role_decision,
                        upiq_label_disposition=upiq_label_disposition,
                        subjective_authority=subjective_authority)
    if research_extra is not None:
        # By-construction research (benchmarks/byconstruction_2026-10-22.md): extra trainer groups the strict route does
        # not admit (AIC2026 TRAIN). The registered groups above passed their strict checks; the trainer runs in replay
        # mode, which records that the fit cannot qualify a model.
        extra, reason = research_extra
        if not extra or not reason or subjective_authority is not None or upiq_label_disposition is not None:
            raise ValueError("research extra groups need a replay reason and no E31/subjective leg")
        at = cmd.index("--target-column")
        cmd[at:at] = [x for g in extra for x in ("--group", g)]
        cmd += ["--historical-replay", reason]
    (dest / "refit").mkdir(exist_ok=True)
    ckpt = dest / "ckpt"
    if EPOCH_RULE == "last" and EPOCHS > 0:
        if strict_admission and ckpt.exists() and any(ckpt.iterdir()):
            raise ValueError("strict route requires a fresh checkpoint directory")
        ckpt.mkdir(exist_ok=True)
        every = max(1, EPOCHS - 1)
        if subjective_authority is not None:
            authority, _, cell, _ = subjective_authority
            # The registered diagnostic binds every-epoch dumps; the native owner refuses any other schedule.
            every = json.loads(Path(authority).read_text())["cells"][cell].get(
                "continuation", {}).get("dump_checkpoints_every", every)
        cmd += ["--dump-checkpoints-every", str(every), "--dump-checkpoints-dir", str(ckpt)]
    elif EPOCH_RULE not in ("last", "best_dev"):
        raise ValueError(f"unknown EPOCH_RULE {EPOCH_RULE!r}")
    run(cmd, dest / "train.log")
    curve = read_curve(dest / "train.log")
    best = max(curve, key=curve.get) if curve else -1
    if EPOCHS == 0:
        bake, selected = dest / "refit" / "best.bin", -1
    elif EPOCH_RULE == "last":
        final = ckpt / f"ckpt_epoch{EPOCHS - 1:03d}.bin"
        if not final.is_file():
            raise ValueError(f"{final}: final-epoch checkpoint missing")
        bake = dest / "refit" / "last.bin"
        shutil.copyfile(final, bake)
        # Checkpoints are experiment artifacts, including the epoch-zero control.
        # Retain them for provenance and prospective convergence diagnostics;
        # selecting the final epoch does not make the other outputs disposable.
        selected = EPOCHS - 1
    else:
        bake, selected = dest / "refit" / "best.bin", best
    selection = {"epoch_rule": EPOCH_RULE, "selected_epoch": selected, "best_epoch_by_curve": best}
    if e36_filter_record is not None:
        selection['e36_base_filter']=e36_filter_record
    if strict_admission:
        selection["strict_table_admission"] = strict_training_groups(groups, data_role_decision, upiq_label_disposition, subjective_authority)
    return bake, curve, selection


def read_curve(log: Path) -> dict[int, float]:
    curve = {int(e): float(v) for e, v in EPOCH_RE.findall(log.read_text())}
    expected = sorted(set(range(0, EPOCHS, LOG_EVERY)) | ({EPOCHS - 1} if EPOCHS else set()))
    if sorted(curve) != expected:
        raise ValueError(f"validation curve incomplete: {len(curve)} of {len(expected)} evaluated epochs")
    return curve


def resolve_keep(core_spec: str, columns: str | None, lists: dict) -> tuple[str, str, list[int]]:
    """(family, variant, kept wide columns) of a cell. Registered arms come from the pinned keep lists, E9 block specs
    from the registered block ranges, and a `sel:<id>` subset from --columns, which must hash to <id>."""
    if core_spec.startswith("sel:"):
        if not columns:
            raise ValueError(f"{core_spec}: a sel: spec needs --columns")
        cols = [int(x) for x in columns.split(",")]
        if cols != sorted(set(cols)) or selection_id(cols) != core_spec[4:]:
            raise ValueError(f"{core_spec}: --columns are not the sorted unique subset this spec names")
        return "main", "real", cols
    if columns:
        raise ValueError("--columns is only for sel: specs")
    if core_spec in lists["specs"]:
        entry = lists["specs"][core_spec]
        return entry["family"], entry["variant"], entry["keep"]
    if block_spec(core_spec):  # design log E9 block specs are derived from the registered block ranges, not the pinned lists
        return arm_columns(core_spec)
    raise ValueError(f"{core_spec}: not in the keep lists")


def hdr_training_group(record: dict, keep: list[int], recipe: dict) -> tuple[tuple, dict]:
    """Admit the unchanged E26 TRAIN authority, then apply the registered loss form."""
    import v2_teacher
    if "hdr_consensus" in recipe:
        from e29_consensus import hdr_group
        return hdr_group(record, keep, recipe)
    path, admitted = v2_teacher.hdr_leg(record, keep)
    mode = recipe.get("hdr_mode", "withinref,rank")
    if mode not in ("withinref,rank", "rank", "withinref,both"):
        raise ValueError("unregistered HDR loss form")
    weight = acceptance_weight(recipe["hdr_weight"], refs_of(path))
    return ("hdr", path, weight, 0, mode), admitted


def strict_output_preflight(root: Path, dest: Path) -> None:
    """Protect every immutable ancestor before CLI destination/scratch writes."""
    from v2_common import admission_input_roots, refuse_immutable_output
    roots = admission_input_roots(root)
    for path in (dest, dest / "keep_features.txt", dest / "refit", dest / "ckpt",
                 Path(os.environ.get("TMPDIR") or dest)):
        refuse_immutable_output(path, roots)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("strict route requires a fresh destination; cannot reuse a historical result")


def main() -> None:
    os.environ.setdefault("ZEN_PANEL_BIN", str(PANEL))  # the fit-cell executor sets it; local runs may not
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--head", choices=HEADS, required=True)
    ap.add_argument("--heldout", choices=SOURCE_ORDER, required=True)
    ap.add_argument("--seed-index", type=int, choices=range(N_SEEDS), required=True)
    ap.add_argument("--root", help="instrument root (default: the Rev3 v2 root); read by v2_common from argv")
    ap.add_argument("--strict-admission", action="store_true", help="strict Rev5 table admission; no historical replay")
    ap.add_argument("--data-role-decision", type=Path, help="coordinator's bound human-role decision JSON")
    ap.add_argument("--upiq380-fit", type=Path, help="pinned native HDR fit table for uh4")
    ap.add_argument("--upiq-label-disposition", type=Path, help="owner decision on the legacy label producer gap")
    ap.add_argument("--dest", type=Path, help="strict route output directory outside frozen inputs")
    ap.add_argument("--train-only", action="store_true", help="stop after selected bake, without human assessment")
    ap.add_argument("--exclude-datasets", default="", help="omit frozen dataset legs, including linked subset protocols")
    ap.add_argument("--columns", help="comma-separated sorted wide columns of a sel:<id> spec (E9′ method 2 refits)")
    ap.add_argument("--steer-pairs", type=Path, help="E34 K label-free repair table")
    ap.add_argument("--steer-lambda", type=float, default=0.0, help="E34 K residual weight")
    ap.add_argument("--monotone-cbc", action="store_true", help="E34 L: project plain N H1 inputs nonnegative")
    ap.add_argument("--local-smoke-budget", help="explicit non-installable local smoke epochs:pairs")
    ap.add_argument("--research-extra-group", action="append", default=[],
                    help="by-construction research only: extra trainer group NAME:PATH:TW:VW:FLAGS (needs --research-replay)")
    ap.add_argument("--research-replay", help="replay reason recorded by the trainer for a research-extra fit")
    ap.add_argument("--init-from", type=Path)
    ap.add_argument("--fine-tune-epochs", type=int)
    ap.add_argument("--fine-tune-lr", type=float, default=0.00001)
    ap.add_argument("--aic-public-train", type=Path)
    from e36_training import add_args
    add_args(ap)
    from matched_codec import add_args as codec_args
    codec_args(ap)
    args = ap.parse_args()
    if args.init_from is not None or args.fine_tune_epochs is not None:
        if not args.init_from or args.fine_tune_epochs not in (0, 4, 8) or not args.strict_admission or not args.train_only:
            ap.error("continuation requires its checkpoint, registered budget and strict TRAIN-only route")
        globals().update(EPOCHS=args.fine_tune_epochs, LOG_EVERY=1)
    elif args.aic_public_train is not None:
        ap.error("D6 leg here requires a registered continuation")
    from e36_training import bootstrap
    bootstrap(args)
    if args.local_smoke_budget:
        if not (args.strict_admission and args.train_only):
            ap.error("local smoke requires strict training-only")
        from v2_smoke_contract import apply_smoke_budget
        apply_smoke_budget(args.local_smoke_budget, globals())
    if args.strict_admission and (args.dest is None or not args.train_only):
        ap.error("strict route requires --dest and --train-only; assessment is a separately registered read")
    if args.exclude_datasets and not (args.strict_admission and args.train_only):
        ap.error("dataset exclusions require strict training-only route")
    native_recipe = recipe_of(args.spec).get("upiq380", False)
    if native_recipe != bool(args.upiq380_fit) or native_recipe != bool(args.upiq_label_disposition):
        ap.error("uh4 requires exactly --upiq380-fit and --upiq-label-disposition")
    if native_recipe:
        if args.spec != "sel:59f0bbc2f290@h32:H128:cv16:cf98:uh4":
            ap.error("E31 requires the registered control recipe plus uh4 only")
        if not args.strict_admission or args.head != "N":
            ap.error("E31 requires the registered strict N head")
        from e31_training import fit_group
        from e21_cheap_recipe import columns
        upiq_group, upiq_record = fit_group(args.upiq380_fit, args.upiq_label_disposition, columns("by_v2fy"))
        from v2_common import refuse_immutable_output
        for output in (args.dest, Path(os.environ.get("TMPDIR") or args.dest)):
            refuse_immutable_output(output, (args.upiq380_fit.resolve().parent,))
    if args.strict_admission:
        strict_output_preflight(V2, args.dest)
        from v2_human_role import preflight_recipe
        role_decision = preflight_recipe(V2, args.data_role_decision, args.heldout)
    if "hdr_consensus" in recipe_of(args.spec):
        if not args.strict_admission:
            raise ValueError("E29 requires strict four-source SDR admission")
        from e29_consensus import preflight
        preflight(V2, args.heldout, recipe_of(args.spec)["hdr_consensus"])
    parse_spec(args.spec)
    core_spec, human_w = split_weight(args.spec)
    lists = json.loads((V2 / "wide" / "keep_lists.json").read_text())
    if lists["schema"] != "rev4-featpot-v2-keeplists-v2":
        raise ValueError("keep-list schema mismatch")
    family, variant, keep = resolve_keep(core_spec, args.columns, lists)
    if native_recipe:
        from e21_cheap_recipe import columns
        if keep != columns("by_v2fy"):
            raise ValueError("E31 requires the registered 420 kept slots")
    vdir = V2 / "wide" / family / variant
    receipt_path = vdir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    # v2c (CANONTAB): the canon receipt's width may exceed 1825 when sidecar families are appended after f1824.
    width = receipt["width"]
    if (receipt["schema"] not in WIDE_SCHEMAS or receipt["family"] != family or receipt["variant"] != variant
            or width < WIDTH or (receipt["schema"] == WIDE_SCHEMAS[0] and width != WIDTH)):
        raise ValueError("wide receipt identity mismatch")
    if args.strict_admission:
        from v2_common import load_frozen
        frozen, frozen_sha = load_frozen(V2, training_only=True)
        if frozen["wide_receipts"].get(f"{family}/{variant}") != sha(receipt_path):
            raise ValueError("wide receipt is not the admitted frozen one")
    if any(not 0 <= c < width for c in keep):
        raise ValueError(f"{core_spec}: kept columns outside 0..{width}")
    recipe = e34_options(args, recipe_of(args.spec))
    if "research_palette" in receipt:
        from e32_palette import admit_recipe
        admit_recipe(receipt, recipe, keep, args.head, args.strict_admission, args.train_only)
    if "ssim2_recipe" in recipe:
        from e28_recipe import admit_humans, admit_receipt
        admit_receipt(V2)
        admit_humans(recipe["ssim2_recipe"], args.heldout, receipt["legs"])
    # Two-part cell path under v2/cells (the fit-cell executor's destination contract).
    dest = args.dest or V2 / "cells" / f"{args.spec}__{args.head}" / f"without_{args.heldout}_s{args.seed_index}"
    if (dest / "result.json").is_file():
        print(json.dumps({"skip": str(dest / "result.json")}))
        return
    dest.mkdir(parents=True, exist_ok=True)
    keep_file = dest / "keep_features.txt"
    keep_file.write_text("\n".join(map(str, keep)) + "\n")
    legs = receipt["legs"]
    recipe = e34_options(args, recipe_of(args.spec))
    groups = []  # (name, path, train_weight, val_weight, mode)
    weights = {}
    curated, teacher_record, curated_extra = None, None, None
    for leg, (_, _, val_w) in TEACHERS.items():
        if recipe.get("ssim2_recipe") == "s2m" and leg == "safesyn":
            continue
        fit, dev = checked(legs[leg]["fit"]), checked(legs[leg]["dev"])
        subset = recipe.get("teacher_subset") if leg == "safesyn" else None
        if subset == "none":  # design log E13: no SafeSyn fit leg (its dev leg still logs)
            teacher_record = {"rule": "none", "rows_in": len(refs_of(fit)), "rows_kept": 0}
            groups.append((f"{leg}_development", dev, 0, val_w, "withinref,both"))
            continue
        if subset:  # design log E13: curated SafeSyn fit rows (v2_teacher), written to scratch, deleted after training
            import v2_teacher
            scratch = Path(os.environ.get("TMPDIR") or dest)
            fit, teacher_record = v2_teacher.curated_leg(subset, fit, legs[leg]["fit"]["keys_sha256"], legs[leg]["bounds"][0],
                                                         scratch)
            curated = fit
        weights[leg] = acceptance_weight(NOMINAL_WEIGHT[leg], refs_of(fit))
        groups += [(leg, fit, weights[leg], 0, "withinref,both"),
                   (f"{leg}_development", dev, 0, val_w, "withinref,both")]
    ordinal_record = None
    if "kadis_ordinal" in recipe:  # design log E14: rank-only KADIS ladders, pairs drawn within a ladder
        import v2_teacher
        if max(keep) >= v2_teacher.ORDINAL_WIDTH:
            raise ValueError(f"{core_spec}: the KADIS ordinal leg has no f{v2_teacher.ORDINAL_WIDTH}+ (NaN); refusing this keep list")
        opath, ordinal_record = v2_teacher.ordinal_leg()
        weights["kadis_ordinal"] = acceptance_weight(recipe["kadis_ordinal"], refs_of(opath))
        groups.append(("kadis_ordinal", opath, weights["kadis_ordinal"], 0, "withinref,rank"))
    coverage_record = None
    if "coverage_weight" in recipe and recipe.get("ssim2_recipe") != "s2m":  # design log E15: chosen families of the ordinal coverage pool, rank-only within ladders
        import v2_teacher
        admit_coverage_keep(core_spec, keep, args.strict_admission)
        cpath, coverage_record = v2_teacher.coverage_leg(recipe["coverage_mask"], Path(os.environ.get("TMPDIR") or dest),
                                                       admitted_root=V2 if args.strict_admission else None)
        curated_extra = cpath
        weights["coverage"] = acceptance_weight(recipe["coverage_weight"], refs_of(cpath))
        groups.append(("coverage", cpath, weights["coverage"], 0, "withinref,rank"))
    hdr_record = None
    if "hdr_weight" in recipe:
        if int(receipt.get("formula_revision", 4)) != 5:
            raise ValueError("HDR teacher leg requires the registered Rev5 SDR root")
        if native_recipe:
            group, hdr_record = upiq_group, upiq_record
        else:
            group, hdr_record = hdr_training_group(legs["hdr_consensus"] if "hdr_consensus" in recipe else legs["hdr"], keep, recipe)
        weights["hdr"] = group[2]
        groups.append(group)
    hfit = checked(legs[f"human_without_{args.heldout}"]["fit"])
    hdev = checked(legs[f"human_without_{args.heldout}"]["dev"])
    weights["human"] = acceptance_weight(NOMINAL_WEIGHT["human"] if human_w is None else human_w, refs_of(hfit))
    groups += [("human", hfit, weights["human"], 0, "withinref,rank"),
               ("human_development", hdev, 0, HUMAN_VAL_WEIGHT, "withinref,rank")]
    e28_record = None
    if "ssim2_recipe" in recipe:
        from e28_recipe import training_groups
        groups, e28_record = training_groups(recipe["ssim2_recipe"], args.heldout, legs, groups,
                                             NOMINAL_WEIGHT["human"] if human_w is None else human_w)
        weights = {name: tw for name, _, tw, _, _ in groups if tw > 0}
    init_seed, sample_seed = seeds(args.heldout, args.seed_index)
    from matched_codec import attach as attach_codec
    groups, recipe = attach_codec(args, groups, V2, init_seed, sample_seed, recipe)
    dataset_authority = None
    if args.strict_admission:
        from subjective_training_leg import dataset_groups
        groups, dataset_authority = dataset_groups(V2, groups, keep, args.heldout, args.seed_index,
                                                   init_seed, sample_seed, dest,
                                                   exclude=tuple(filter(None, args.exclude_datasets.split(","))))
        if dataset_authority:
            weights.update({name:tw for name,_,tw,_,_ in groups if tw > 0})
    if args.init_from is not None:
        if dataset_authority is None:
            raise ValueError("Mlic continuation requires frozen dataset authority")
        from subjective_training_leg import continuation_authority
        dataset_authority = continuation_authority(dataset_authority, args.init_from, EPOCHS, args.fine_tune_lr)
    if args.aic_public_train is not None:
        if not args.research_extra_group:
            raise ValueError("D6 continuation needs its explicitly weighted group")
        aic_spec = args.research_extra_group
        if len(aic_spec) != 1:
            raise ValueError("one D6 continuation leg required")
        name, path, tw, vw, mode = aic_spec[0].split(":")
        if name != "aic_public_train" or Path(path).resolve() != args.aic_public_train.resolve() or vw != "0" or mode != "rank":
            raise ValueError("D6 continuation leg identity differs")
        groups.append((name, Path(path), float(tw), 0, mode))
        args.research_extra_group = []
    try:
        from v2_common import refuse_nonfinite_kept
        if args.strict_admission:
            strict_training_groups(groups, args.data_role_decision, args.upiq_label_disposition, dataset_authority)
        read = keep
        if recipe.get("derived_inputs") in ("fx1", "fx2", "fx1nr", "fx1g", "fx1gs1", "fx1gg", "fx1gy", "fx1o", "fx1ns0"):
            from v2_common import derived_declaration
            read = sorted(set(keep) | {b for _, b in derived_declaration(recipe["derived_inputs"])[2]["products"]})
        refuse_nonfinite_kept([g[1] for g in groups], read)  # Rev5 tables mark absent slots NaN; a kept one is refused here
        if bool(args.research_extra_group) != bool(args.research_replay):
            raise ValueError("--research-extra-group and --research-replay go together")
        if args.research_extra_group and not (args.strict_admission and args.train_only):
            raise ValueError("research extra groups ride on the strict training-only route")
        bake, curve, selection = train_and_select(groups, init_seed, sample_seed, width, keep_file, args.head, dest, recipe,
                                                  strict_admission=args.strict_admission,
                                                  data_role_decision=args.data_role_decision,
                                                  upiq_label_disposition=args.upiq_label_disposition,
                                                  subjective_authority=dataset_authority,
                                                  research_extra=(args.research_extra_group, args.research_replay)
                                                  if args.research_extra_group else None)
    finally:
        for tmp in (curated, curated_extra):
            if tmp is not None:
                tmp.unlink(missing_ok=True)
                Path(f"{tmp}.manifest.json").unlink(missing_ok=True)
                tmp.with_suffix(".keys.parquet").unlink(missing_ok=True)
    if native_recipe:
        selection["upiq380_fit_admission"] = hdr_record
    best_epoch = selection["selected_epoch"]
    if args.train_only:
        schema = ("e31-native-hdr-research-training-cell-v1" if native_recipe else
                  "e29-research-training-cell-v1" if hdr_record and "hdr_consensus" in recipe else
                  "byc-research-replay-training-cell-v1" if args.research_extra_group else
                  "rev5-qualified-training-cell-v1" if args.strict_admission else "historical-training-only-v1")
        (dest / "result.json").write_text(json.dumps({"schema": schema,
            "hdr_leg": hdr_record,
            "training_only": True, "execution_contract": "local-smoke" if args.local_smoke_budget else "registered-fit", "selection": selection,
            "research_extra_groups": args.research_extra_group or None, "research_replay": args.research_replay, "heldout": args.heldout, "epochs": EPOCHS, "pairs_per_epoch": PAIRS_PER_EPOCH,
            "seed_index": args.seed_index, "width": width, "kept_features": len(keep),
            "wide_receipt_sha256": sha(receipt_path), "frozen_sha256": frozen_sha if args.strict_admission else None,
            "human_sources": role_decision["sources"] if args.strict_admission else None,
            "data_role_decision_sha256": sha(args.data_role_decision) if args.strict_admission else None,
            "selected_bake": str(bake), "selected_bake_sha256": sha(bake), "dev_curve": curve,
            "spec": args.spec, "head": args.head, "init_seed": init_seed, "sample_seed": sample_seed,
            "train_weights": weights, "coverage_leg": coverage_record,
            **({"derived_inputs": {"token": recipe["derived_inputs"], "declaration_sha256": __import__("v2_common").derived_declaration(recipe["derived_inputs"])[1]}}
               if recipe.get("derived_inputs") else {})}) + "\n")
        from e36_training import model_binding
        model_binding(dest,recipe,args,init_seed,sample_seed,bake,selection=selection)
        return
    heldout = legs[args.heldout]
    table = checked(heldout["full"])
    pred = predict(bake, table, dest / "eval_preds.tsv")
    keys = pq.read_table(vdir / f"{args.heldout}.keys.parquet").to_pandas()
    if sha(vdir / f"{args.heldout}.keys.parquet") != heldout["keys_sha256"] or len(keys) != len(pred):
        raise ValueError("held-out keys changed or length mismatch")
    y = keys.target.to_numpy(dtype=np.float64)
    score = panel_batch([(args.heldout, pred, y)], stats="full")[0]
    out = {"schema": "rev4-featpot-v2-cell-v2", "label": "POTENTIAL — ceiling, not a model score",
           "recipe": "R915 sampling (amendment revision R1, layout R1.1)", "spec": args.spec,
           "human_nominal_weight": NOMINAL_WEIGHT["human"] if human_w is None else human_w,
           "family": family, "variant": variant,
           "kept_features": len(keep), "head": args.head, "heldout": args.heldout,
           "seed_index": args.seed_index, "init_seed": init_seed, "sample_seed": sample_seed,
           "train_weights": weights, "hidden": recipe.get("hidden", HIDDEN), "epochs": EPOCHS, "pairs_per_epoch": PAIRS_PER_EPOCH,
           **({"recipe_tokens": recipe} if recipe else {}),
           **({"teacher_subset": teacher_record} if teacher_record else {}),
           **({"ordinal_leg": ordinal_record} if ordinal_record else {}),
           **({"coverage_leg": coverage_record} if coverage_record else {}),
           **({"hdr_leg": hdr_record} if hdr_record else {}),
           **({"ssim2_recipe": e28_record} if e28_record else {}),
           "wide_receipt_sha256": sha(receipt_path), "table_receipt_sha256": sha(receipt_path),
           "keep_lists_sha256": sha(V2 / "wide" / "keep_lists.json"), "binaries": {p.name: sha(p) for p in (TRAINER, FITBIN, PANEL)},
           "dev_geomean3_by_epoch": curve, **selection,
           "selected_bake": str(bake), "selected_bake_sha256": sha(bake), "rows": len(y),
           "references": int(keys.ref_basename.nunique()), "keys_sha256": heldout["keys_sha256"],
           "prediction": pred.tolist(), "score": score}
    (dest / "result.json").write_text(json.dumps(out) + "\n")
    print(json.dumps({"result": str(dest / "result.json"), "best_epoch": best_epoch,
                      "srocc": score.get("srocc")}), flush=True)


if __name__ == "__main__":
    main()
