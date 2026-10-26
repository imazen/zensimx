"""One full-data confirmatory fit cell (amendment R2): (arm spec, head, seed index) -> predictions, no labels.

Governing record: benchmarks/rev4_featpot_v2_amendment_2026-09-30.md, revision R2. Trains exactly as v2_lodo_mlp.py
(R915 teacher legs + the human leg, shared `train_command`; heads N and F; H = 32; 120 epochs x 50,000 pairs; best epoch
by the weighted dev geomean3) but the human leg is `human_all` (the union of the five exploratory sources; no source is
held out), then predicts every confirmatory feature table of the spec's family with the selected bake through
`bake_dial_refit predict --score-units`. `result.json` holds per-set predictions, the keys hashes they align to and the
receipts. It never reads a label: a confirmatory set has none in these tables.

  python v2_confirm_fit.py --spec c1 --head N --seed-index 0 [--root /var/tmp/canontab/v2c]

The null is MATCHED (coordinator decision 2026-10-01): a spec's variant selects its confirmatory tables, so `<arm>~pk` is
evaluated on the within-reference permuted confirmatory columns (built label-free by `v2c_wide.py confirm`), as v2 did.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from v2_common import (load_frozen, recipe_of, EPOCHS, FITBIN, HEADS, HIDDEN, HUMAN_VAL_WEIGHT, NOMINAL_WEIGHT, PANEL, PAIRS_PER_EPOCH, TEACHERS,
                       TRAINER, V2, WIDTH, acceptance_weight, confirm_seeds, parse_spec, sha, split_weight, table_path)
from v2_lodo_mlp import e34_options, WIDE_SCHEMAS, checked, predict, refs_of, resolve_keep, train_and_select, strict_training_groups, strict_output_preflight

CONFIRM_SCHEMA = "rev4-featpot-v2c-confirm-v1"
RESULT_SCHEMA = "rev4-featpot-v2c-confirm-cell-v1"


def confirm_tables(family: str, variant: str, receipt: dict) -> dict[str, dict]:
    """{set: table record} for the spec's family; the table record carries its path, hashes and keys hash."""
    want = variant
    out = {}
    for name, rec in receipt["sets"].items():
        by_variant = rec["tables"].get(family, {})
        if want not in by_variant:
            raise ValueError(f"{name}: no {family}/{want} confirmatory table (build it with v2c_wide.py confirm)")
        out[name] = by_variant[want]
    return out


def cell_dir(spec: str, head: str, seed_index: int) -> Path:
    return V2 / "confirm" / "cells" / f"{spec}__{head}" / f"full_s{seed_index}"


def main() -> None:
    os.environ.setdefault("ZEN_PANEL_BIN", str(PANEL))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude-datasets", default="", help="TRAIN-only dataset-group holdout, comma-separated collections")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--head", choices=HEADS, required=True)
    ap.add_argument("--seed-index", type=int, choices=range(10), required=True)
    ap.add_argument("--dest", type=Path, help="write here instead of <root>/confirm/cells/... (determinism checks)")
    ap.add_argument("--strict-admission", action="store_true", help="strict Rev5 table admission; no historical replay")
    ap.add_argument("--data-role-decision", type=Path, help="coordinator's bound human-role decision JSON")
    ap.add_argument("--pack-production", action="store_true", help="densify and f16 pack, then calibrate on TRAIN cid22 oracle")
    ap.add_argument("--e33-output-stage", action="store_true",
                    help="E33 registration section 8: identity knot (pin, 100) and collinear tail (requires --pack-production)")
    ap.add_argument("--train-only", action="store_true", help="stop after selected bake; no confirmatory predictions")
    ap.add_argument("--root", help="instrument root (default: the Rev3 v2 root); read by v2_common from argv")
    ap.add_argument("--columns", help="comma-separated sorted wide columns of a sel:<id> spec (as v2_lodo_mlp)")
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
    ap.add_argument("--aic-weight", type=float)
    from e36_training import add_args
    add_args(ap)
    args = ap.parse_args()
    if args.init_from is not None or args.fine_tune_epochs is not None:
        if (not args.init_from or not args.strict_admission or not args.train_only
                or args.exclude_datasets or args.local_smoke_budget or args.research_extra_group
                or args.research_replay or args.aic_public_train is None or args.fine_tune_lr != 0.00001
                or (args.fine_tune_epochs, args.aic_weight) not in ((4, 8.0), (8, 16.0))):
            ap.error("full continuation requires a registered D6 strength, original legs and strict TRAIN-only route")
        import v2_lodo_mlp
        globals().update(EPOCHS=args.fine_tune_epochs)
        v2_lodo_mlp.EPOCHS = args.fine_tune_epochs
        v2_lodo_mlp.LOG_EVERY = 1
    elif args.aic_public_train is not None or args.aic_weight is not None:
        ap.error("D6 full-data leg requires its registered continuation")
    from e36_training import bootstrap
    bootstrap(args)
    if args.local_smoke_budget:
        if not (args.strict_admission and args.train_only):
            ap.error("local smoke requires strict training-only")
        from v2_smoke_contract import apply_smoke_budget
        apply_smoke_budget(args.local_smoke_budget, globals())
    if args.e33_output_stage and not args.pack_production:
        ap.error("--e33-output-stage requires --pack-production")
    if args.pack_production and not (args.strict_admission and args.train_only):
        ap.error("production packing requires strict admission and training-only")
    if args.strict_admission and (args.dest is None or not args.train_only):
        ap.error("strict route requires --dest and --train-only; assessment is a separately registered read")
    if args.exclude_datasets and not (args.strict_admission and args.train_only):
        ap.error("dataset-group holdout requires strict training-only route")
    if args.strict_admission:
        strict_output_preflight(V2, args.dest)
        from v2_human_role import preflight_recipe
        role_decision = preflight_recipe(V2, args.data_role_decision)
    parse_spec(args.spec)
    core_spec, human_w = split_weight(args.spec)
    lists = json.loads((V2 / "wide" / "keep_lists.json").read_text())
    if lists["schema"] != "rev4-featpot-v2-keeplists-v2":
        raise ValueError("keep-list schema mismatch")
    # 2026-10-04 (set-compare confirmatory read): registered arms, E9 block/set specs and sel: subsets resolve exactly as in the
    # LODO cells (v2_lodo_mlp.resolve_keep); a registered arm's keep list is unchanged.
    family, variant, keep = resolve_keep(core_spec, args.columns, lists)
    vdir = V2 / "wide" / family / variant
    receipt_path = vdir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    width = receipt["width"]
    if (receipt["schema"] != WIDE_SCHEMAS[1] or receipt["family"] != family or receipt["variant"] != variant
            or width < WIDTH or not receipt.get("complete")):
        raise ValueError("wide receipt identity mismatch or incomplete (confirmatory fits need the canon tables, all legs)")
    frozen, frozen_sha = load_frozen(V2, training_only=args.strict_admission and args.train_only)  # refuses an unfrozen root or any receipt changed since the freeze
    if frozen["wide_receipts"].get(f"{family}/{variant}") != sha(receipt_path):
        raise ValueError("wide receipt is not the frozen one")
    if not args.train_only:
        confirm_path = V2 / "wide" / "confirm" / "receipt.json"
        confirm = json.loads(confirm_path.read_text())
        if confirm["schema"] != CONFIRM_SCHEMA or confirm["width"] != width or confirm["feature_set_id"] != receipt["feature_set_id"]:
            raise ValueError("confirmatory receipt does not match the wide receipt (schema, width or feature_set_id)")
        tables = confirm_tables(family, variant, confirm)
    dest = args.dest or cell_dir(args.spec, args.head, args.seed_index)
    if (dest / "result.json").is_file():
        print(json.dumps({"skip": str(dest / "result.json")}))
        return
    dest.mkdir(parents=True, exist_ok=True)
    keep_file = dest / "keep_features.txt"
    keep_file.write_text("\n".join(map(str, keep)) + "\n")
    legs = receipt["legs"]
    groups, weights = [], {}
    for leg, (_, _, val_w) in TEACHERS.items():
        fit, dev = checked(legs[leg]["fit"]), checked(legs[leg]["dev"])
        weights[leg] = acceptance_weight(NOMINAL_WEIGHT[leg], refs_of(fit))
        groups += [(leg, fit, weights[leg], 0, "withinref,both"), (f"{leg}_development", dev, 0, val_w, "withinref,both")]
    recipe = e34_options(args, recipe_of(args.spec))
    if "hdr_weight" in recipe:
        raise ValueError("E26 is registered for LODO cells only; no HDR confirmation fit")
    coverage_record, curated_extra = None, None
    if "coverage_weight" in recipe:  # design log E15/E17: the coverage leg exactly as v2_lodo_mlp builds it (label-free ordinal pool)
        import v2_teacher
        from v2_lodo_mlp import admit_coverage_keep
        admit_coverage_keep(core_spec, keep, args.strict_admission)
        cpath, coverage_record = v2_teacher.coverage_leg(recipe["coverage_mask"], Path(os.environ.get("TMPDIR") or dest),
                                                       admitted_root=V2 if args.strict_admission else None)
        curated_extra = cpath
        weights["coverage"] = acceptance_weight(recipe["coverage_weight"], refs_of(cpath))
        groups.append(("coverage", cpath, weights["coverage"], 0, "withinref,rank"))
    if "kadis_ordinal" in recipe or recipe.get("teacher_subset"):
        raise ValueError(f"{args.spec}: the confirm fitter implements the coverage leg only (no ko/ts recipe tokens)")
    hfit, hdev = checked(legs["human_all"]["fit"]), checked(legs["human_all"]["dev"])
    weights["human"] = acceptance_weight(NOMINAL_WEIGHT["human"] if human_w is None else human_w, refs_of(hfit))
    groups += [("human", hfit, weights["human"], 0, "withinref,rank"),
               ("human_development", hdev, 0, HUMAN_VAL_WEIGHT, "withinref,rank")]
    init_seed, sample_seed = confirm_seeds(args.seed_index)
    dataset_authority = None
    if args.strict_admission:
        from subjective_training_leg import dataset_groups
        groups, dataset_authority = dataset_groups(V2, groups, keep, "full", args.seed_index,
                                                   init_seed, sample_seed, dest,
                                                   exclude=tuple(filter(None, args.exclude_datasets.split(","))))
        if dataset_authority:
            weights.update({name:tw for name,_,tw,_,_ in groups if tw > 0})
    if args.init_from is not None:
        if dataset_authority is None:
            raise ValueError("full continuation requires frozen Mlic dataset authority")
        from subjective_training_leg import continuation_authority
        dataset_authority = continuation_authority(dataset_authority, args.init_from,
            args.fine_tune_epochs, args.fine_tune_lr)
        groups.append(("aic_public_train", args.aic_public_train, args.aic_weight, 0, "rank"))
        weights["aic_public_train"] = args.aic_weight
    try:
        from v2_common import refuse_nonfinite_kept
        if args.strict_admission:
            strict_training_groups(groups, args.data_role_decision, subjective_authority=dataset_authority)
        read = keep
        if recipe.get("derived_inputs") in ("fx1", "fx2", "fx1nr", "fx1g", "fx1gs1", "fx1gg", "fx1gy", "fx1o", "fx1ns0"):
            from v2_common import derived_declaration
            read = sorted(set(keep) | {b for _, b in derived_declaration(recipe["derived_inputs"])[2]["products"]})
        refuse_nonfinite_kept([g[1] for g in groups], read)  # Rev5 tables mark absent slots NaN
        if bool(args.research_extra_group) != bool(args.research_replay):
            raise ValueError("--research-extra-group and --research-replay go together")
        if args.research_extra_group and not (args.strict_admission and args.train_only) or (
                args.research_extra_group and args.pack_production):
            raise ValueError("research extra groups ride on the strict training-only route, unpacked")
        bake, curve, selection = train_and_select(groups, init_seed, sample_seed, width, keep_file, args.head, dest, recipe,
                                                  strict_admission=args.strict_admission,
                                                  data_role_decision=args.data_role_decision,
                                                  subjective_authority=dataset_authority,
                                                  research_extra=(args.research_extra_group, args.research_replay)
                                                  if args.research_extra_group else None)
    finally:
        if curated_extra is not None:
            curated_extra.unlink(missing_ok=True)
            Path(f"{curated_extra}.manifest.json").unlink(missing_ok=True)
            curated_extra.with_suffix(".keys.parquet").unlink(missing_ok=True)
    best_epoch = selection["selected_epoch"]
    if args.train_only:
        packed = {}
        if args.pack_production:
            from v2_production_pack import pack_production
            from e36_training import calibration_anchor
            try:
                packed = pack_production(bake, dest, calibration_anchor(checked(legs["cid22"]["fit"]),recipe),
                                         e33_output_stage=args.e33_output_stage)
            except RuntimeError as refusal:
                if not args.e33_output_stage:
                    raise
                # E33 registration section 8: a K1-K4 refusal is an ineligibility RESULT. Keep the trained model,
                # its dense form and the pack log in the cell, and report deterministically (fit_cell_exec does not
                # retry this marker).
                pack_log = dest / "pack.log"
                tail = pack_log.read_text(errors="replace").splitlines()[-12:] if pack_log.is_file() else []
                (dest / "E33_OUTPUT_STAGE_REFUSAL.json").write_text(json.dumps({
                    "schema": "e33-output-stage-refusal-v1", "selected_bake": str(bake),
                    "selected_bake_sha256": sha(bake), "pack_log_tail": tail, "error": str(refusal)}) + "\n")
                print("E33 output-stage refusal: " + " | ".join(tail[-3:]), flush=True)
                raise SystemExit(3)
        (dest / "result.json").write_text(json.dumps({"schema": "byc-research-replay-training-cell-v1" if args.research_extra_group else
            "rev5-qualified-training-cell-v1" if args.strict_admission else "historical-training-only-v1",
            "research_extra_groups": args.research_extra_group or None, "research_replay": args.research_replay,
            "training_only": True, "execution_contract": "local-smoke" if args.local_smoke_budget else "registered-fit", "selection": selection, "epochs": EPOCHS, "pairs_per_epoch": PAIRS_PER_EPOCH,
            "seed_index": args.seed_index, "width": width, "kept_features": len(keep),
            "wide_receipt_sha256": sha(receipt_path), "frozen_sha256": frozen_sha if args.strict_admission else None,
            "human_sources": role_decision["sources"] if args.strict_admission else None,
            "data_role_decision_sha256": sha(args.data_role_decision) if args.strict_admission else None,
            "selected_bake": str(bake), "selected_bake_sha256": sha(bake), "dev_curve": curve,
            "spec": args.spec, "head": args.head, "init_seed": init_seed, "sample_seed": sample_seed,
            "train_weights": weights, "coverage_leg": coverage_record, **packed,
            **({"continuation": dict(initial_checkpoint=str(args.init_from),
                initial_checkpoint_sha256=sha(args.init_from), epochs=args.fine_tune_epochs,
                lr=args.fine_tune_lr, lr_schedule="constant", optimizer_state="fresh")}
                if args.init_from is not None else {}),
            **({"derived_inputs": {"token": recipe["derived_inputs"], "declaration_sha256": __import__("v2_common").derived_declaration(recipe["derived_inputs"])[1]}}
               if recipe.get("derived_inputs") else {})}) + "\n")
        from e36_training import model_binding
        model_binding(dest,recipe,args,init_seed,sample_seed,bake,packed,selection=selection)
        return
    predictions = {}
    for name, table in tables.items():
        path = table_path(table)
        if sha(path) != table["sha256"] or sha(Path(f"{path}.manifest.json")) != table["manifest_sha256"]:
            raise ValueError(f"{name}: confirmatory table changed after its receipt")
        pred = predict(bake, path, dest / f"preds_{name}.tsv")
        rows = confirm["sets"][name]["rows"]
        if len(pred) != rows:
            raise ValueError(f"{name}: {len(pred)} predictions for {rows} rows")
        predictions[name] = {"rows": rows, "keys_sha256": table["keys_sha256"], "table_sha256": table["sha256"],
                             "pred": pred.tolist()}
    out = {"schema": RESULT_SCHEMA, "label": "POTENTIAL — ceiling, not a model score; features only, no label read",
           "recipe": "R915 sampling (amendment R1/R1.1) + full-data human_all leg (amendment R2)", "spec": args.spec,
           "human_nominal_weight": NOMINAL_WEIGHT["human"] if human_w is None else human_w, "family": family,
           "variant": variant, "eval_variant": variant, "kept_features": len(keep), "head": args.head,
           "seed_index": args.seed_index, "init_seed": init_seed, "sample_seed": sample_seed, "train_weights": weights,
           "hidden": recipe.get("hidden", HIDDEN), "epochs": EPOCHS, "recipe_tokens": recipe,
           **({"coverage_leg": coverage_record} if coverage_record else {}), "pairs_per_epoch": PAIRS_PER_EPOCH, "width": width,
           "wide_receipt_sha256": sha(receipt_path), "frozen_sha256": frozen_sha, "confirm_receipt_sha256": sha(confirm_path),
           "keep_lists_sha256": sha(V2 / "wide" / "keep_lists.json"),
           "binaries": {p.name: sha(p) for p in (TRAINER, FITBIN, PANEL)},
           "dev_geomean3_by_epoch": curve, **selection,
           "selected_bake": str(bake), "selected_bake_sha256": sha(bake),
           "predictions": predictions}
    (dest / "result.json").write_text(json.dumps(out) + "\n")
    print(json.dumps({"result": str(dest / "result.json"), "best_epoch": best_epoch,
                      "sets": {k: v["rows"] for k, v in predictions.items()}}), flush=True)


if __name__ == "__main__":
    main()
