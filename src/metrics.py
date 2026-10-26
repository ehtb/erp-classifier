"""Small numpy-only metric helpers shared by evaluation and baseline scripts."""

import math
from collections.abc import Sequence

import numpy as np

# A step is only required to be correct outside this relative band. The band is
# excluded from the tolerance check because no model can resolve 800.00 EUR
# exactly from data that never contains that point.
DEFAULT_STEP_TOL_REL = 0.02


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_ci(
    correct: Sequence[float],
    groups: Sequence,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
):
    """Percentile CI for accuracy, resampling whole groups rather than items."""
    correct = np.asarray(correct, dtype=float)
    groups = np.asarray(groups)
    if len(correct) == 0:
        return (0.0, 0.0)
    unique = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        vals.append(correct[idx].mean())
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def cross_threshold(amounts, p_hi, threshold: float = 0.5) -> dict:
    """First upward crossing of `threshold`, interpolated in log-amount.

    Returns the crossing amount (or None), whether the curve is censored (never
    crosses), and whether it starts above the threshold. "Starts above" is
    counted as a crossing at the left edge of the grid so an all-CAPEX curve is
    penalised instead of silently censored.
    """
    amounts = np.asarray(amounts, dtype=float)
    p = np.asarray(p_hi, dtype=float)
    if len(p) == 0:
        return {"crossing": None, "censored": True, "starts_above": False, "index": None}

    above = p >= threshold
    if above[0]:
        return {"crossing": float(amounts[0]), "censored": False, "starts_above": True, "index": 0}

    upward = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    if len(upward) == 0:
        return {"crossing": None, "censored": True, "starts_above": False, "index": None}

    i = int(upward[0])
    x0, x1 = math.log(amounts[i - 1]), math.log(amounts[i])
    p0, p1 = float(p[i - 1]), float(p[i])
    t = 0.0 if p1 == p0 else (threshold - p0) / (p1 - p0)
    t = min(max(t, 0.0), 1.0)
    crossing = math.exp(x0 + t * (x1 - x0))
    return {
        "crossing": float(crossing),
        "censored": False,
        "starts_above": False,
        "index": i,
    }


def sweep_case_metrics(
    amounts,
    p_hi,
    boundary: float,
    *,
    tight_rel: float = 0.0625,
    step_tol_rel: float = DEFAULT_STEP_TOL_REL,
    mono_tol: float = 0.05,
) -> dict:
    """Metrics for one fixed-text amount sweep.

    `p_hi` is the probability renormalised over the two accounts either side of
    the boundary. `tolerance_step` is the old "perfect step" without the
    meaningless +/-0.001% grid points: it requires every amount outside a
    +/-2% band (`step_tol_rel`) to be classified correctly.
    """
    amounts = np.asarray(amounts, dtype=float)
    p = np.asarray(p_hi, dtype=float)
    pred = p >= 0.5
    want = amounts >= boundary
    tight = np.abs(amounts - boundary) <= tight_rel * boundary
    outside = np.abs(amounts - boundary) > step_tol_rel * boundary
    diffs = np.diff(p) if len(p) > 1 else np.array([], dtype=float)
    flips = int(np.sum(np.diff(pred.astype(np.int8)) != 0)) if len(pred) > 1 else 0
    crossing = cross_threshold(amounts, p)

    return {
        "points": int(len(amounts)),
        "point_correct": int((pred == want).sum()),
        "tight_points": int(tight.sum()),
        "tight_correct": int((pred[tight] == want[tight]).sum()),
        "tolerance_points": int(outside.sum()),
        "tolerance_correct": int((pred[outside] == want[outside]).sum()),
        "tolerance_step": int(bool((pred[outside] == want[outside]).all())),
        "flips": flips,
        "one_flip": int(flips == 1),
        "non_monotone": int(bool((diffs < -mono_tol).any())),
        "max_negative_step": float(max(0.0, -float(diffs.min()))) if len(diffs) else 0.0,
        "crossing": crossing["crossing"],
        "crossing_censored": int(crossing["censored"]),
        "crossing_starts_above": int(crossing["starts_above"]),
    }


def aggregate_sweep(
    cases: list[dict],
    *,
    mono_tol: float = 0.05,
    step_tol_rel: float = DEFAULT_STEP_TOL_REL,
    tight_rel: float = 0.0625,
) -> dict:
    """Aggregate per-case sweep metrics without conditioning on flip count.

    Crossing error is computed for every case that has a crossing; the coverage
    (`crossing_coverage`) and censored fraction are always reported, so a config
    with 1% coverage cannot be compared to one with 100% as if they were equal.
    """
    if not cases:
        return {
            "cases": 0, "point_accuracy": None, "tight_band_point_accuracy": None,
            "strict_point_accuracy": None, "tolerance_band_point_accuracy": None,
            "tolerance_step_fraction": None, "exactly_one_flip_fraction": None,
            "non_monotone_fraction": None, "non_monotone_tolerance": mono_tol,
            "step_tolerance_rel": step_tol_rel, "tight_band_rel": tight_rel,
            "max_negative_step": None, "crossing_coverage": 0.0,
            "crossing_censored_fraction": None, "crossing_starts_above_fraction": None,
            "median_crossing_rel_error": None, "crossing_iqr_rel_error": None,
            "crossing_rel_errors": [],
        }

    n = len(cases)
    points = sum(c["points"] for c in cases)
    tight_points = sum(c["tight_points"] for c in cases)
    tol_points = sum(c["tolerance_points"] for c in cases)
    crossing_errs = []
    censored = 0
    starts_above = 0
    for c in cases:
        b = c["boundary"]
        if c["crossing"] is None:
            censored += 1
        else:
            crossing_errs.append(abs(c["crossing"] - b) / b)
            starts_above += c["crossing_starts_above"]

    arr = np.asarray(crossing_errs, dtype=float)
    strict_vals = [c["strict_correct"] for c in cases if "strict_correct" in c]
    strict_points = sum(c["points"] for c in cases if "strict_correct" in c)
    return {
        "cases": n,
        "point_accuracy": sum(c["point_correct"] for c in cases) / points if points else None,
        "tight_band_point_accuracy": sum(c["tight_correct"] for c in cases) / tight_points if tight_points else None,
        "strict_point_accuracy": sum(strict_vals) / strict_points if strict_points else None,
        "tolerance_band_point_accuracy": sum(c["tolerance_correct"] for c in cases) / tol_points if tol_points else None,
        "tolerance_step_fraction": sum(c["tolerance_step"] for c in cases) / n,
        "exactly_one_flip_fraction": sum(c["one_flip"] for c in cases) / n,
        "non_monotone_fraction": sum(c["non_monotone"] for c in cases) / n,
        "non_monotone_tolerance": mono_tol,
        "step_tolerance_rel": step_tol_rel,
        "tight_band_rel": tight_rel,
        "max_negative_step": max(c["max_negative_step"] for c in cases),
        "crossing_coverage": len(arr) / n,
        "crossing_censored_fraction": censored / n,
        "crossing_starts_above_fraction": starts_above / n,
        "median_crossing_rel_error": float(np.median(arr)) if len(arr) else None,
        "crossing_iqr_rel_error": (
            [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))] if len(arr) else None
        ),
        "crossing_rel_errors": [float(x) for x in arr],
    }


def expected_calibration_error(conf, correct, n_bins: int = 10) -> float:
    """Expected calibration error (ECE) over equal-width confidence bins."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def risk_coverage(conf, correct, target: float = 0.99):
    """Accuracy/coverage curve across confidence cut-offs, plus the first cut-off meeting `target`."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    curve = []
    for t in np.arange(0.40, 1.001, 0.05):
        mask = conf >= t
        if mask.any():
            curve.append({
                "cutoff": round(float(t), 2),
                "coverage": float(mask.mean()),
                "accuracy": float(correct[mask].mean()),
                "n_accepted": int(mask.sum()),
            })
    chosen = next((c for c in curve if c["accuracy"] >= target), None)
    return curve, chosen
