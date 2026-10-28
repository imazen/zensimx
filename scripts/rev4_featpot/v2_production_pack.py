"""Canonical deployment order: selected epoch -> densify -> quantize -> TRAIN dial."""
from pathlib import Path
from v2_common import FITBIN, dense_bake, sha
from v2_lodo_mlp import run, strict_training_groups


# E33 registration section 8 (owner-approved values): identity knot at the --nonneg-distance pin, tail factor 2.
E33_IDENTITY_KNOT = "100"
E33_TAIL_FACTOR = "2"


def pack_production(bake, dest, anchor, e33_output_stage=False, no_prune=False):
    # Teacher role, revision, ordered keys and admission hashes precede its read.
    strict_training_groups([("TRAIN calibration", anchor, 0., 0., "withinref,both")])
    dense = dense_bake(bake, dest)
    packed = Path(dest) / "refit/production-f16.bin"
    stage = ["--identity-knot", E33_IDENTITY_KNOT, "--tail-extend", E33_TAIL_FACTOR] if e33_output_stage else []
    pruning = ["--no-prune"] if no_prune else []
    run([str(FITBIN), "pack", "--in", str(dense), "--out", str(packed),
         "--dtype", "f16", "--zerobias-bulk", "0", "--protect-last", "--neg-tail", *stage, *pruning,
         "--anchor", str(anchor), "--target-col", "human_score", "--verify", "none"],
        Path(dest) / "pack.log")
    result = {"packed_model": str(packed), "packed_model_sha256": sha(packed),
            "dense_model_sha256": sha(dense), "TRAIN_calibration_table_sha256": sha(anchor),
            "output_stage": "e33-identity-knot-tail2" if e33_output_stage else "legacy-neg-tail",
            "deployment_order": "final epoch; canonical densify; f16 quantization; TRAIN calibration"}
    if no_prune:
        result["input_pruning"] = "disabled; preserve derived-input positions"
    return result
