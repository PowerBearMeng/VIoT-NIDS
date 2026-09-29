"""Label-isolated evaluation of frozen continuous scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Scores and binary labels must align")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    result = {"samples": len(labels), "normal": int((labels == 0).sum()),
              "attack": int((labels == 1).sum()),
              "AUROC": None, "AUPRC": None, "EER": None}
    if len(np.unique(labels)) < 2:
        return result, pd.DataFrame(), pd.DataFrame()
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    differences = np.abs(fpr - (1 - tpr))
    eer_index = int(np.argmin(differences))
    result.update({"AUROC": float(roc_auc_score(labels, scores)),
                   "AUPRC": float(average_precision_score(labels, scores)),
                   "EER": float((fpr[eer_index] + 1 - tpr[eer_index]) / 2),
                   "EER_FPR": float(fpr[eer_index]),
                   "EER_FNR": float(1 - tpr[eer_index])})
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    pr = pd.DataFrame({"precision": precision, "recall": recall,
                       "threshold": np.r_[pr_thresholds, np.nan]})
    return result, roc, pr


def evaluate(scores_path: Path, labels_path: Path, out: Path,
             attack_name: str | None = None) -> dict:
    score = pd.read_csv(scores_path)
    labels = pd.read_csv(labels_path)
    if "row_id" not in score or "row_id" not in labels or "binary_label" not in labels:
        raise ValueError("Expected row_id in both inputs and binary_label in labels")
    if score.row_id.duplicated().any() or labels.row_id.duplicated().any():
        raise ValueError("Duplicate row_id")
    merged = score.merge(labels[["row_id", "binary_label"]], on="row_id", how="left", validate="one_to_one")
    if merged.binary_label.isna().any() or len(merged) != len(labels):
        raise ValueError("Scores and labels have different rows")
    result, roc, pr = metrics(merged.binary_label.to_numpy(), merged.final_score.to_numpy())
    if attack_name:
        result["attack_name"] = attack_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    roc.to_csv(out / "roc.csv", index=False)
    pr.to_csv(out / "pr.csv", index=False)
    return result
