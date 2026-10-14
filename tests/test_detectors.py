import numpy as np
import pytest

from src.detectors import DEFAULT_PARAMS, run_detector
from src.utils import DETECTORS


def _make_data(n_normal=300, n_outlier=10):
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 0.3, size=(n_normal, 4))
    # Scattered, isolated outliers (each alone) so noise-based detectors flag them.
    outlier = rng.uniform(4, 8, size=(n_outlier, 4))
    return np.vstack([normal, outlier])


@pytest.mark.parametrize("name", DETECTORS)
def test_run_detector_output_contract(name):
    X = _make_data()
    res = run_detector(name, X, DEFAULT_PARAMS[name])
    assert res["name"] == name
    assert res["pred"].shape == (len(X),)
    assert set(np.unique(res["pred"])).issubset({0, 1})
    assert res["n_anomaly"] == int(res["pred"].sum())
    assert res["runtime"] >= 0
    if res["score"] is not None:
        assert res["score"].shape == (len(X),)


def test_dbscan_has_no_continuous_score():
    X = _make_data()
    res = run_detector("DBSCAN", X, DEFAULT_PARAMS["DBSCAN"])
    assert res["score"] is None


def test_dbscan_flags_isolated_outliers():
    X = _make_data()
    # Pin eps to the synthetic data's scale (features ~0.3, outliers at 4-8);
    # the app default (5.0) targets the scaled 30-feature credit-card data.
    res = run_detector("DBSCAN", X, {"eps": 0.5, "min_samples": 5})
    # 10 scattered, isolated points must be noise (-1 -> anomaly=1) under DBSCAN.
    assert res["pred"][-10:].sum() == 10


def test_ranking_scores_rank_outliers_above_normals():
    X = _make_data()
    for name in ["Local Outlier Factor", "Isolation Forest"]:
        res = run_detector(name, X, DEFAULT_PARAMS[name])
        assert res["score"] is not None
        assert res["score"][-10:].mean() > res["score"][:300].mean(), \
            f"{name} score did not rank outliers above normals"
