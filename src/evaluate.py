"""
Clinical Metrics and Model Evaluation Utilities
Computes:
- ROC-AUC and PR-AUC (Average Precision)
- Brier Score (Probability Calibration)
- Top-Decile Lift & Recall@Top-K% (Operational Clinical Capacity)
- Precision, Recall, Specificity, F1 at optimal threshold
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    top_k_percent: float = 0.10,
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of clinical and statistical metrics.
    """
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)

    # Discrimination metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Operational metric: Recall at Top-K% of highest risk patients
    k_cutoff = int(len(y_prob) * top_k_percent)
    top_k_indices = np.argsort(y_prob)[-k_cutoff:]
    total_positives = np.sum(y_true)
    positives_in_top_k = np.sum(y_true[top_k_indices])
    recall_at_top_k = positives_in_top_k / (total_positives + 1e-9)
    baseline_positives_top_k = total_positives * top_k_percent
    top_decile_lift = positives_in_top_k / (baseline_positives_top_k + 1e-9)

    # Binary threshold metrics
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sensitivity = tp / (tp + fn + 1e-9)  # Recall
    specificity = tn / (tn + fp + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Compute optimal threshold based on Youden's J statistic (Sensitivity + Specificity - 1)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold_youden = float(roc_thresholds[optimal_idx]) if optimal_idx < len(roc_thresholds) else 0.5

    # Compute optimal threshold for F1
    prec_arr, rec_arr, f1_thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (prec_arr * rec_arr) / (prec_arr + rec_arr + 1e-9)
    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold_f1 = float(f1_thresholds[best_f1_idx]) if best_f1_idx < len(f1_thresholds) else 0.5

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "recall_at_top_10pct": float(recall_at_top_k),
        "top_decile_lift": float(top_decile_lift),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "threshold_used": float(threshold),
        "optimal_threshold_youden": float(optimal_threshold_youden),
        "optimal_threshold_f1": float(optimal_threshold_f1),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
