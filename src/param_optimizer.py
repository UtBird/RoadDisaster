"""
Parameter optimizer for Detection Threshold and Damage Booster.

Supports:
  - Otsu: Unsupervised threshold optimization (no ground truth)
  - Grid search: Try (booster, threshold) combinations
  - Scipy minimize: Black-box optimization
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


def apply_threshold_logic(
    p_bg: np.ndarray,
    p_road: np.ndarray,
    p_partial: np.ndarray,
    p_total: np.ndarray,
    damage_booster: float,
    sensitivity_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply booster and threshold logic to get final pred_labels and damage_ratio.
    Returns (pred_labels, damage_ratio).
    """
    p_partial_boosted = p_partial * damage_booster
    p_total_boosted = p_total * damage_booster

    scores = np.stack([p_bg, p_road, p_partial_boosted, p_total_boosted])
    pred_labels = np.argmax(scores, axis=0)

    sum_road_mass = p_road + p_partial_boosted + p_total_boosted + 1e-6
    damage_ratio = (p_partial_boosted + p_total_boosted) / sum_road_mass

    h, w = pred_labels.shape
    for r in range(h):
        for c in range(w):
            if pred_labels[r, c] != 0:
                if damage_ratio[r, c] > sensitivity_threshold:
                    pred_labels[r, c] = 3 if p_total_boosted[r, c] > p_partial_boosted[r, c] else 2

    return pred_labels, damage_ratio


def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """
    Otsu's method: find threshold that maximizes between-class variance.
    values: 1D array (e.g. damage_ratio for road pixels)
    """
    values = np.asarray(values).ravel().astype(np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.5

    hist, bin_edges = np.histogram(values, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return 0.5

    sum_total = np.dot(hist, bin_centers)
    sum_b = 0
    w_b = 0
    best_var = 0
    best_threshold = 0.5

    for i in range(n_bins - 1):
        w_b += hist[i]
        w_f = total - w_b
        if w_b == 0 or w_f == 0:
            continue
        sum_b += hist[i] * bin_centers[i]
        mu_b = sum_b / w_b
        mu_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (mu_b - mu_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_threshold = bin_centers[i]

    return float(best_threshold)


def optimize_threshold_otsu(
    p_road: np.ndarray,
    p_partial: np.ndarray,
    p_total: np.ndarray,
    damage_booster: float = 1.5,
    road_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Find optimal sensitivity_threshold using Otsu on damage_ratio distribution.
    Only considers pixels that are on roads (or all non-background if no mask).

    Args:
        p_road, p_partial, p_total: Probability maps (H, W)
        damage_booster: Booster value (affects damage_ratio)
        road_mask: Optional (H, W) bool - only use these pixels. If None, use p_road > 0.1

    Returns:
        Optimal threshold in [0, 1]
    """
    p_partial_boosted = p_partial * damage_booster
    p_total_boosted = p_total * damage_booster
    sum_road_mass = p_road + p_partial_boosted + p_total_boosted + 1e-6
    damage_ratio = (p_partial_boosted + p_total_boosted) / sum_road_mass

    if road_mask is not None:
        mask = road_mask.astype(bool)
    else:
        mask = (p_road + p_partial + p_total) > 0.1

    values = damage_ratio[mask]
    if values.size < 10:
        return 0.25

    return _otsu_threshold(values)


def optimize_booster_and_threshold_grid(
    p_road: np.ndarray,
    p_partial: np.ndarray,
    p_total: np.ndarray,
    booster_range: Tuple[float, float, float] = (1.0, 3.0, 0.5),
    road_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Grid search over booster values; for each booster, find Otsu threshold.
    Pick (booster, threshold) that maximizes Otsu's between-class variance.

    Returns:
        (best_booster, best_threshold, best_score)
    """
    boosters = np.arange(*booster_range)
    best_booster = 1.5
    best_threshold = 0.25
    best_score = -1

    for booster in boosters:
        p_partial_b = p_partial * booster
        p_total_b = p_total * booster
        sum_road_mass = p_road + p_partial_b + p_total_b + 1e-6
        damage_ratio = (p_partial_b + p_total_b) / sum_road_mass

        if road_mask is not None:
            mask = road_mask.astype(bool)
        else:
            mask = (p_road + p_partial + p_total) > 0.1

        values = damage_ratio[mask]
        if values.size < 10:
            continue

        thresh = _otsu_threshold(values)
        p_low = values[values <= thresh]
        p_high = values[values > thresh]
        if p_low.size > 0 and p_high.size > 0:
            var_between = p_low.size * p_high.size * (p_low.mean() - p_high.mean()) ** 2
            if var_between > best_score:
                best_score = var_between
                best_booster = float(booster)
                best_threshold = float(thresh)

    return best_booster, best_threshold, best_score


@dataclass
class OptimizerResult:
    booster: float
    threshold: float
    method: str
    score: Optional[float] = None


def auto_optimize(
    p_road: np.ndarray,
    p_partial: np.ndarray,
    p_total: np.ndarray,
    road_mask: Optional[np.ndarray] = None,
    method: str = "otsu_threshold_only",
) -> OptimizerResult:
    """
    Main entry: auto-optimize threshold and optionally booster.

    method:
      - "otsu_threshold_only": Fix booster=1.5, optimize only threshold (fast)
      - "otsu_full": Optimize both booster and threshold via grid search
    """
    if method == "otsu_threshold_only":
        threshold = optimize_threshold_otsu(
            p_road, p_partial, p_total, damage_booster=1.5, road_mask=road_mask
        )
        return OptimizerResult(booster=1.5, threshold=threshold, method="otsu_threshold")

    if method == "otsu_full":
        booster, threshold, score = optimize_booster_and_threshold_grid(
            p_road, p_partial, p_total, road_mask=road_mask
        )
        return OptimizerResult(booster=booster, threshold=threshold, method="otsu_full", score=score)

    raise ValueError(f"Unknown method: {method}")
