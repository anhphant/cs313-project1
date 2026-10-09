"""Unsupervised anomaly detectors with a uniform output convention.

Output convention: ``pred`` is an int array where **1 = anomaly, 0 = normal**.
For DBSCAN/HDBSCAN the cluster label ``-1`` (noise) is mapped to anomaly=1 as a
*noise* signal, not as a fraud verdict. ``score`` is a continuous ranking score
(higher = more anomalous) where the algorithm provides one; ``None`` otherwise.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

DEFAULT_PARAMS = {
    "DBSCAN": {"eps": 5.0, "min_samples": 5},
    "HDBSCAN": {"min_cluster_size": 15, "min_samples": 5},
    "Local Outlier Factor": {"n_neighbors": 20, "contamination": 0.01},
    "Isolation Forest": {"n_estimators": 100, "contamination": 0.01, "random_state": 42},
}


def run_detector(name: str, X: np.ndarray, params: dict) -> dict:
    """Run one detector and return pred, score, runtime, n_anomaly."""
    t0 = time.perf_counter()

    if name == "DBSCAN":
        model = DBSCAN(eps=float(params["eps"]), min_samples=int(params["min_samples"]))
        labels = model.fit_predict(X)
        pred = (labels == -1).astype(int)
        score = None  # DBSCAN gives no continuous ranking score -> AP unavailable.

    elif name == "HDBSCAN":
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=int(params["min_cluster_size"]),
            min_samples=int(params["min_samples"]),
        )
        labels = model.fit_predict(X)
        pred = (labels == -1).astype(int)
        try:
            score = np.asarray(model.outlier_scores_, dtype=float)  # higher = outlier
        except AttributeError:
            score = None

    elif name == "Local Outlier Factor":
        model = LocalOutlierFactor(
            n_neighbors=int(params["n_neighbors"]),
            contamination=float(params["contamination"]),
            novelty=False,
        )
        labels = model.fit_predict(X)  # 1 inlier / -1 outlier
        pred = (labels == -1).astype(int)
        score = -np.asarray(model.negative_outlier_factor_, dtype=float)

    elif name == "Isolation Forest":
        model = IsolationForest(
            n_estimators=int(params["n_estimators"]),
            contamination=float(params["contamination"]),
            random_state=int(params["random_state"]),
        )
        labels = model.fit_predict(X)  # 1 inlier / -1 anomaly
        pred = (labels == -1).astype(int)
        score = -model.decision_function(X)

    else:
        raise ValueError(f"Unknown detector: {name}")

    runtime = time.perf_counter() - t0
    return {
        "name": name,
        "pred": pred.astype(int),
        "score": score,
        "runtime": runtime,
        "n_anomaly": int(pred.sum()),
    }
