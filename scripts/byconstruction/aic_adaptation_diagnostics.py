"""Explicit denominators and ladder diagnostics for the registered AIC comparison."""
import itertools
import math

import numpy as np


def complete_mean(values):
    """Preserve the declared estimand when a fold or seed is undefined."""
    values = np.asarray(values, dtype=float)
    defined = np.isfinite(values)
    return dict(value=float(values.mean()) if defined.all() and len(values) else None,
                required=len(values), defined=int(defined.sum()), undefined=int((~defined).sum()))


def paired_seeds(candidate, baseline, folds, metric):
    seeds = [complete_mean([candidate[f, s]["statistics"][metric]
                            - baseline[f, s]["statistics"][metric] for f in folds]) for s in range(3)]
    values = [s["value"] if s["value"] is not None else float("nan") for s in seeds]
    mean = complete_mean(values)
    return dict(per_seed_delta=values, per_seed_fold_counts=seeds, mean=mean["value"],
                seed_counts={k: v for k, v in mean.items() if k != "value"},
                seed_unit_se=float(np.std(values, ddof=1) / np.sqrt(3)) if mean["defined"] == 3 else None,
                estimand="mean of four matched cell-metric deltas within each of three paired seeds")


def checked_statistics(jobs, statistics):
    """The native owner must return every requested label and row denominator."""
    if len(jobs) != len(statistics):
        raise ValueError("native statistics coverage differs")
    for (label, x, y), row in zip(jobs, statistics, strict=True):
        if row["label"] != label or row["n"] + row["n_dropped"] != len(x) or len(x) != len(y):
            raise ValueError("native statistics label or row denominator differs")


def cohort(continued, role):
    return dict(aic_exposure="continued" if continued else "zero-shot initialization",
                cohort_semantics=("continued model, in sample" if role == "fitted" else
                                  "continued model, source held out") if continued else
                f"zero-shot initialization on the continuation-{role} cohort")


def ladder_diagnostics(pairs, prediction, target, mask, logistic_fit):
    """AIC's quality field is its distortion-level index: lower is better.

    This is the dataset's CVVDP-spaced level, not a codec-native encoder knob.
    Require its exact filename binding; never infer direction from responses.
    Calibration is diagnostic and fitted only on the requested cohort.
    """
    p = pairs.loc[mask].reset_index(drop=True)
    x, y = np.asarray(prediction, float)[mask], np.asarray(target, float)[mask]
    q = p.quality.to_numpy(dtype=float)
    if any(str(key) != f"{str(group).removesuffix('_Ref_00')}_{codec}_{int(level):02d}"
           or level != int(level) or not 1 <= level <= 20
           for key, group, codec, level in zip(p.key, p.group, p.codec, q, strict=True)):
        raise ValueError("AIC distortion-level filename binding differs")
    mapped = np.asarray(logistic_fit(x, y), dtype=float)
    if mapped.shape != x.shape:
        raise ValueError("diagnostic map coverage differs")
    span = float(np.quantile(y, .99, method="linear") - np.quantile(y, .01, method="linear"))
    bins = {name: [] for name in ("concurring", "subjective_only")}
    counts = dict(total_pairs=0, nonfinite_pairs=0, level_ties=0, subjective_ties=0)
    residuals = []
    ladders = p.groupby(["group", "codec"], sort=True).indices
    for idx in ladders.values():
        residual = mapped[idx] - y[idx]
        residuals.extend(residual - residual.mean())
        for i, j in itertools.combinations(idx, 2):
            counts["total_pairs"] += 1
            if not np.isfinite([q[i], q[j], y[i], y[j], x[i], x[j], mapped[i], mapped[j]]).all():
                counts["nonfinite_pairs"] += 1
            elif q[i] == q[j]:
                counts["level_ties"] += 1
            elif y[i] == y[j]:
                counts["subjective_ties"] += 1
            else:
                good, bad = (i, j) if q[i] < q[j] else (j, i)
                name = "concurring" if y[good] > y[bad] else "subjective_only"
                bins[name].append((good, bad))
    result = dict(rows=len(p), sources=int(p.group.nunique()), ladders=len(ladders), **counts,
                  level_meaning="CVVDP-spaced distortion index; lower is better; not codec-native quality",
                  diagnostic_map="canonical logistic_fit on this cohort only",
                  magnitude_scale="mapped difference / cohort target p99-p1", target_span=span)
    for name, indices in bins.items():
        delta = np.array([mapped[g] - mapped[b] for g, b in indices])
        raw = np.array([x[g] - x[b] for g, b in indices])
        sizes = -delta[delta < 0] / span if math.isfinite(span) and span > 0 else None
        result[name] = dict(eligible_pairs=len(indices), inversions=int((delta < 0).sum()),
            raw_inversions=int((raw < 0).sum()), prediction_ties=int((delta == 0).sum()),
            follows_level=int((delta > 0).sum()), inversion_rate=float((delta < 0).mean()) if len(delta) else None,
            size_mean=float(sizes.mean()) if sizes is not None and len(sizes) else (0.0 if sizes is not None else None),
            size_p90=float(np.quantile(sizes, .9, method="linear")) if sizes is not None and len(sizes) else (0.0 if sizes is not None else None),
            size_max=float(sizes.max()) if sizes is not None and len(sizes) else (0.0 if sizes is not None else None))
    result["ladder_rms"] = float(np.sqrt(np.mean(np.square(residuals))) / span) if span > 0 and residuals else None
    return result
