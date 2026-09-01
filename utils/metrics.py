"""Binary anomaly-detection metrics and low-FPR operating points."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def _as_optional(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Tie-aware interpolated EER and its adjacent score bracket.

    The bracket follows the Gotham baseline convention.  For mechanism audits,
    ``threshold_upper`` is used with ``score >= threshold``.  With large score
    ties, the empirical FPR/FNR at that realizable threshold can differ from
    the interpolated EER and must therefore be reported separately.
    """

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    normal_total = int(np.sum(labels == 0))
    attack_total = int(np.sum(labels == 1))
    if normal_total == 0 or attack_total == 0:
        raise ValueError("EER requires both normal and attack samples")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    thresholds: list[float] = [float("inf")]
    fprs: list[float] = [0.0]
    fnrs: list[float] = [1.0]
    normal_above = 0
    attack_above = 0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        thresholds.append(float(sorted_scores[start]))
        fprs.append(normal_above / normal_total)
        fnrs.append((attack_total - attack_above) / attack_total)
        group = sorted_labels[start:end]
        attack_above += int(np.sum(group == 1))
        normal_above += int(np.sum(group == 0))
        start = end
    thresholds.append(float("-inf"))
    fprs.append(1.0)
    fnrs.append(0.0)
    fprs_array = np.asarray(fprs, dtype=np.float64)
    fnrs_array = np.asarray(fnrs, dtype=np.float64)
    differences = fprs_array - fnrs_array
    for index in range(len(thresholds) - 1):
        left = float(differences[index])
        right = float(differences[index + 1])
        if left == 0.0:
            return {
                "value": float(fprs_array[index]),
                "threshold_lower": float(thresholds[index]),
                "threshold_upper": float(thresholds[index]),
            }
        if left < 0.0 < right:
            ratio = -left / (right - left)
            value = fprs_array[index] + ratio * (
                fprs_array[index + 1] - fprs_array[index]
            )
            return {
                "value": float(value),
                "threshold_lower": float(thresholds[index + 1]),
                "threshold_upper": float(thresholds[index]),
            }
    index = int(np.argmin(np.abs(differences)))
    return {
        "value": float((fprs_array[index] + fnrs_array[index]) / 2.0),
        "threshold_lower": float(thresholds[index]),
        "threshold_upper": float(thresholds[index]),
    }


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    # A quantile is a boundary; only strict exceedances are anomalous. This is
    # important for small calibration sets whose 99th percentile is the maximum.
    predictions = scores > threshold
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & negatives))
    tn = int(np.sum(~predictions & negatives))
    fn = int(np.sum(~predictions & positives))
    tpr_at_threshold = tp / max(1, tp + fn)
    fpr_at_threshold = fp / max(1, fp + tn)
    result: dict[str, Any] = {
        "num_samples": int(len(labels)),
        "num_normal": int(negatives.sum()),
        "num_attack": int(positives.sum()),
        "threshold": float(threshold),
        "FPR": float(fpr_at_threshold),
        "TPR": float(tpr_at_threshold),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if not positives.any() or not negatives.any():
        result.update({"AUROC": None, "AUPRC": None, "average_precision": None, "EER": None, "eer": None, "TPR@FPR<=1%": None, "TPR@FPR<=0.1%": None})
        return result
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    eer = equal_error_rate(labels, scores)
    average_precision = _as_optional(average_precision_score(labels, scores))
    trapezoid_auprc = _as_optional(auc(recall[::-1], precision[::-1]))
    result.update(
        {
            "AUROC": _as_optional(roc_auc_score(labels, scores)),
            # Match the TFusion/Kitsune baseline convention: PR-AUC is
            # tie-aware Average Precision.  Keep trapezoidal integration as a
            # separate diagnostic because large score ties can make it
            # pathologically different from AP.
            "AUPRC": average_precision,
            "average_precision": average_precision,
            "AUPRC_trapezoid": trapezoid_auprc,
            "EER": float(eer["value"]),
            "eer": eer,
            "TPR@FPR<=1%": float(tpr[fpr <= 0.01].max(initial=0.0)),
            "TPR@FPR<=0.1%": float(tpr[fpr <= 0.001].max(initial=0.0)),
        }
    )
    return result
