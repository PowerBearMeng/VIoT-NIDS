"""Offline metrics only; capture labels never enter the detector."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
from .io import write_json


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

def interpolated_eer(roc):
    delta=roc.fpr.to_numpy()+roc.tpr.to_numpy()-1
    i=int(np.flatnonzero(delta>=0)[0])
    if delta[i]==0:return float(roc.fpr.iloc[i])
    ratio=-delta[i-1]/(delta[i]-delta[i-1])
    return float(roc.fpr.iloc[i-1]+ratio*(roc.fpr.iloc[i]-roc.fpr.iloc[i-1]))


def evaluate(scores_path, labels_path, output):
    def frame(path):
        path=Path(path)
        if path.suffix=='.npz':
            with np.load(path,allow_pickle=False) as f:return pd.DataFrame({k:f[k] for k in f.files})
        return pd.read_csv(path)
    scores=frame(scores_path);labels=frame(labels_path)
    if not {'row_id','binary_label'}.issubset(labels.columns) or 'row_id' not in scores:
        raise ValueError('Expected independent row_id/binary_label labels')
    if labels.row_id.duplicated().any() or scores.row_id.duplicated().any():
        raise ValueError('Duplicate row_id')
    aligned=scores.merge(labels[['row_id','binary_label']],on='row_id',how='outer',validate='one_to_one',indicator=True)
    if not aligned['_merge'].eq('both').all():raise ValueError('Score and label populations differ')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for branch in ('prototype_raw','mass_raw','final_score'):
        result,roc,pr=metrics(aligned.binary_label.to_numpy(),aligned[branch].to_numpy())
        result['branch']=branch;result['EER_nearest']=result['EER']
        if len(roc):result['EER']=interpolated_eer(roc)
        result['EER_definition']='linear ROC crossing FPR=FNR; EER_nearest is empirical diagnostic'
        rows.append(result)
        roc.to_csv(root/(branch+'_roc.csv.gz'),index=False);pr.to_csv(root/(branch+'_pr.csv.gz'),index=False)
    pd.DataFrame(rows).to_csv(root/'metrics.csv',index=False)
    write_json(root/'metrics.json',rows)
