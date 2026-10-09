"""App-level logic tests: group building, no-label handling, expand/search."""
import os
import tempfile

import numpy as np
import pandas as pd

import app
from src.evaluation import classify_outcomes, evaluate
from src.utils import DEFAULT_FEATURES


def _synth_csv(has_label=True):
    rng = np.random.default_rng(2)
    n = 400
    cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in cols})
    if has_label:
        df["Class"] = 0
        df.loc[[5, 50, 200], "Class"] = 1
    path = os.path.join(tempfile.gettempdir(), "test_app_cc.csv")
    df.to_csv(path, index=False)
    return path


def _prepped_state(path, has_label=True):
    state = app._empty_state()
    state, *_ = app.on_preprocess(path, 300, 42, False, True, list(DEFAULT_FEATURES), "PCA", state)
    return state


def test_preprocess_missing_label_sets_has_label_false():
    state = _prepped_state(_synth_csv(has_label=False), has_label=False)
    assert state["has_label"] is False
    assert state["prep"]["y"] is None


def test_run_without_label_produces_no_metrics():
    state = _prepped_state(_synth_csv(has_label=False), has_label=False)
    state, *_ = app.on_run("DBSCAN", 0.5, 5, 15, 20, 0.01, 100, 42, state)
    assert state["results"]["DBSCAN"]["metrics"] is None
    assert "không có nhãn" in app._g_message(state)


def test_group_frame_splits_tp_tn_fp_fn_correctly():
    state = app._empty_state()
    state["prep"] = dict(
        idx=np.array([0, 1, 2, 3]), amount=np.array([10.0, 20.0, 30.0, 40.0]),
        time=np.array([1.0, 2.0, 3.0, 4.0]), y=np.array([0, 0, 1, 1]),
    )
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 0, 1])
    state["results"]["DBSCAN"] = dict(
        pred=pred, score=None, runtime=0.0, n_anomaly=2,
        metrics=evaluate(y, pred, None), outcome=classify_outcomes(y, pred),
    )
    state["current"] = "DBSCAN"

    assert list(app._group_frame(state, "TP")["Transaction ID"]) == [3]
    assert list(app._group_frame(state, "TN")["Transaction ID"]) == [0]
    assert list(app._group_frame(state, "FP")["Transaction ID"]) == [1]
    assert list(app._group_frame(state, "FN")["Transaction ID"]) == [2]


def test_group_search_filters_by_transaction_id():
    state = app._empty_state()
    state["prep"] = dict(
        idx=np.array([10, 20, 30]), amount=np.array([1.0, 2.0, 3.0]),
        time=np.array([1.0, 2.0, 3.0]), y=np.array([0, 1, 1]),
    )
    pred = np.array([1, 1, 1])
    state["results"]["DBSCAN"] = dict(
        pred=pred, score=None, runtime=0.0, n_anomaly=3,
        metrics=evaluate(np.array([0, 1, 1]), pred, None),
        outcome=classify_outcomes(np.array([0, 1, 1]), pred),
    )
    state["current"] = "DBSCAN"
    frame = app._group_frame(state, "TP", search="20", rows=10)
    assert list(frame["Transaction ID"]) == [20]


def test_expand_toggles():
    new, label, vis = app.on_expand(False)
    assert new is True and vis.get("visible") is True
    new2, _, vis2 = app.on_expand(True)
    assert new2 is False and vis2.get("visible") is False


def test_download_group_writes_csv():
    state = app._empty_state()
    state["prep"] = dict(
        idx=np.array([0]), amount=np.array([1.0]), time=np.array([1.0]),
        y=np.array([1]),
    )
    state["results"]["DBSCAN"] = dict(
        pred=np.array([1]), score=None, runtime=0.0, n_anomaly=1,
        metrics=evaluate(np.array([1]), np.array([1]), None),
        outcome=classify_outcomes(np.array([1]), np.array([1])),
    )
    state["current"] = "DBSCAN"
    path = app.on_download_group(state, "TP")
    assert os.path.exists(path) and os.path.getsize(path) > 0
