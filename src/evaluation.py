"""Evaluation of anomaly predictions against the (held-out) true labels.

``Class`` is used for evaluation only, never for training or parameter choice.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, score=None) -> dict:
    """Return precision/recall/f1/AP plus TP/TN/FP/FN counts.

    ``score`` is the continuous anomaly score (higher = more anomalous). AP is
    ``None`` when no such score exists or the sample has a single class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    ap = None
    if score is not None and len(np.unique(y_true)) > 1:
        ap = float(average_precision_score(y_true, score))

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ap": ap,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def classify_outcomes(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Map each sample to 'TP'/'TN'/'FP'/'FN' (or None when no true label)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = np.full(len(y_true), None, dtype=object)
    out[(y_true == 1) & (y_pred == 1)] = "TP"
    out[(y_true == 0) & (y_pred == 0)] = "TN"
    out[(y_true == 0) & (y_pred == 1)] = "FP"
    out[(y_true == 1) & (y_pred == 0)] = "FN"
    return out
