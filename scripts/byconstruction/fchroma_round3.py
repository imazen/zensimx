#!/usr/bin/env python3
"""Collect profile F round 3 (F3 / F3codec) into one JSON for the coordinator's plots.

Reads finished artifacts only. Writes outside the repository: it holds AIC-2026 public_train statistics, which are not committed.
Missing artifacts are recorded as "not_run". Reuses the round-1 collector's readers.

  fchroma_round3.py --out ~/tmp/handoff/f-chroma/results_round3.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fchroma_results as r1  # noqa: E402

S = Path("/var/tmp/byc-fchroma-20261029")
R3 = S / "r3"
ST = R3 / "stage2"
G = ST / "gates"
OLD = Path("/var/tmp/byc-fx2-20261028/stage2")
BAKES = Path.home() / "tmp/fchroma-bakes"
MODELS = ("E", "ECodec", "F", "Fcodec", "F3", "F3codec")


def fold_panels(path, labels):
    """out-of-fold aggregate (mean of 3 seeds): codec rows, and the mean of the per-source rows."""
    d = r1.load(path)
    if d == "not_run":
        return d
    out = {}
    for label in labels:
        rows = [x for x in d if x["model"] == label]
        src = [x["statistics"] for x in rows if x["category"] == "source"]
        keys = ("srocc", "plcc", "krocc")
        out[label] = dict(
            per_codec={x["group"].strip("'"): {k: x["statistics"].get(k) for k in ("n", *keys)} for x in rows if x["category"] == "codec"},
            mean_over_sources={k: sum(s[k] for s in src) / len(src) for k in keys} | dict(n_sources=len(src)))
    return out


def e21(arm):
    d = r1.load(R3 / arm / "e21/e21.json")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = {
        "E": "E.bin", "ECodec": "ECodec.bin", "F": "F.bin", "Fcodec": "Fcodec.bin", "F3": "F3.bin", "F3codec": "F3codec.bin"}
    roles = {
        "F3": "fx2 + chroma teacher leg without the <=64 px views, general continuation e59, seed 0 (weight 1)",
        "F3codec": "same, codec continuation e29 (no-prune pack), seed 0"}
    out = dict(
        schema="fchroma-results-round3-v1", note="private: contains AIC-2026 public_train statistics; do not commit",
        models={m: dict(path=str(BAKES / p), sha256=r1.sha(BAKES / p), role=roles.get(m)) for m, p in paths.items()},
        in_sample_full_data_panels=dict(
            F3=r1.panel(ST / "fx2-full-general-e59/panels/MODEL_PANELS.json", ["XrefAIC256e59", "Initial", "Mlic"]),
            F3codec=r1.panel(ST / "fx2-full-codec-e29-repack/panels/MODEL_PANELS.json", ["AICOnlye29", "Initial", "Mlic"]),
            E=r1.panel(r1.V / "mlic-aic-fulldata-xrefaic256e59-20261027-run01/panels/MODEL_PANELS.json", ["XrefAIC256e59", "Mlic"]),
            ECodec=r1.panel(r1.V / "mlic-aic-fulldata-aiconlye29-20261027-run01/panels/MODEL_PANELS.json", ["AICOnlye29", "Mlic"]),
            scope="in sample for full-data models; aic = AIC-2026 public_train (local only)"),
        continuation_source_fold_study_out_of_fold=dict(
            note="held-out source folds, mean of 3 seeds; F3 = filtered-root fx2 arm, E = revision-7 Mlic root (fx2cont lane fold study, fx2 declaration, no chroma leg)",
            F3=fold_panels(ST / "fx2-folds-r2/panels/OUT_OF_FOLD_AGGREGATE_PANELS.json",
                           ["XrefAIC256@e59|mean3", "AICOnly@e29|mean3", "Initial|mean3"]),
            fx2_no_chroma_leg=fold_panels(OLD / "fx2-folds-run01/panels/OUT_OF_FOLD_AGGREGATE_PANELS.json",
                                          ["XrefAIC256@e59|mean3", "AICOnly@e29|mean3", "Initial|mean3"])),
        e21_arms_vs_stored_mlic={arm: e21(arm) for arm in ("w100", "w50", "w25", "w12")},
        gates={
            "F3": dict(contracts=r1.gate_block(G / "f3general.summary.json", "f3general"), gsteer=r1.gsteer(G / "f3general/steer135/f3general/result.json"),
                       nearid=r1.nearid(ST / "nearid/COMPLETE.json", "F3General"), identity_dev_12_permutations=r1.identity(G / "f3general/identity-s0.jsonl"),
                       certify_monotone=[r1.load(ST / f"fx2-full-general-e59/full/certify_s{s}.json") for s in range(3)]),
            "F3codec": dict(contracts=r1.gate_block(G / "f3codec.summary.json", "f3codec"), gsteer=r1.gsteer(G / "f3codec/steer135/f3codec/result.json"),
                            nearid=r1.nearid(ST / "nearid/COMPLETE.json", "F3Codec"), identity_dev_12_permutations=r1.identity(G / "f3codec/identity-s0.jsonl"),
                            certify_monotone=[r1.load(ST / f"fx2-full-codec-e29-repack/full/certify_s{s}.json") for s in range(3)])},
        g_chroma_dev_over_64px=dict(rank=r1.load(ST / "gchroma-dev/G_CHROMA.json"), level={c: r1.load(ST / f"gchroma-dev/G_CHROMA_L_{c}.json") for c in ("F3", "F3codec")}),
        g_chroma_calibration_over_64px=dict(note="read once, at the end", rank=r1.load(ST / "gchroma-cal/G_CHROMA.json"),
                                            level={c: r1.load(ST / f"gchroma-cal/G_CHROMA_L_{c}.json") for c in ("F3", "F3codec")}),
        failed_attempts_kept=[str(p) for p in sorted(ST.glob("*failed*")) + sorted(G.glob("*failed*")) + [ST / "fx2-folds", ST / "fx2-full-codec-e29"]],
    )
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
