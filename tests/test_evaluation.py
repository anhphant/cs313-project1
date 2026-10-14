import numpy as np

from src.evaluation import classify_outcomes, evaluate


def test_evaluate_known_case():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    m = evaluate(y_true, y_pred, score=None)
    assert m["tp"] == 1 and m["tn"] == 1 and m["fp"] == 1 and m["fn"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["ap"] is None  # no continuous score -> AP unavailable


def test_ap_with_continuous_score():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.4])  # perfect ranking
    m = evaluate(y_true, y_pred, score=score)
    assert m["ap"] == 1.0


def test_ap_none_for_single_class():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    score = np.array([0.1, 0.2, 0.3, 0.4])
    m = evaluate(y_true, y_pred, score=score)
    assert m["ap"] is None


def test_classify_outcomes():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    out = classify_outcomes(y_true, y_pred)
    assert list(out) == ["TN", "FP", "FN", "TP"]


def test_zero_division_returns_zero():
    y_true = np.array([1, 1])
    y_pred = np.array([0, 0])
    m = evaluate(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
