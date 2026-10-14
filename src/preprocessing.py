"""Preprocessing: sampling, dedup, amount scaling, feature selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .utils import LABEL_COL


def preprocess(
    df: pd.DataFrame,
    n_samples: int,
    seed: int,
    drop_duplicates: bool,
    scale_features: bool,
    feature_cols: list[str],
) -> dict:
    """Build the feature matrix for unsupervised detectors.

    Returns a dict with ``X`` (numpy float array) plus the metadata needed to
    map rows back to the original table (``idx``, ``amount``, ``time``, ``y``).
    ``Class`` is never part of ``X``.

    When ``scale_features`` is true, every feature is robust-scaled so the
    distance-based detectors (DBSCAN/HDBSCAN/LOF) are not dominated by
    ``Time``/``Amount``, whose raw scale dwarfs the ``V*`` features.
    """
    d = df.copy()

    if drop_duplicates:
        d = d.drop_duplicates()

    if n_samples and n_samples < len(d):
        d = d.sample(n=n_samples, random_state=seed)

    idx = d.index.to_numpy()
    amount = d["Amount"].to_numpy(dtype=float)
    time = d["Time"].to_numpy(dtype=float)
    y = d[LABEL_COL].to_numpy(dtype=int) if LABEL_COL in d.columns else None

    feats = [c for c in feature_cols if c in d.columns and c != LABEL_COL]
    X = np.nan_to_num(d[feats].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    if scale_features:
        X = RobustScaler().fit_transform(X)

    return {"X": X, "idx": idx, "amount": amount, "time": time, "y": y, "feats": feats}
