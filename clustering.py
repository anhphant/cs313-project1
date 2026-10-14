"""Clustering wrappers (DBSCAN / HDBSCAN / KMeans) + metrics + stream detection.

Uses scikit-learn's HDBSCAN (available since 1.3) so the standalone `hdbscan`
package is not required.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusterResult:
    labels: np.ndarray
    algorithm: str
    runtime_ms: float
    n_clusters: int
    n_noise: int
    noise_frac: float
    silhouette: float | None
    sizes: dict[int, int]
    is_real: dict[int, bool]
    main_cluster: int | None
    elongation: dict[int, float]


def _finish(X: np.ndarray, df: pd.DataFrame, labels: np.ndarray,
            algorithm: str, t0: float) -> ClusterResult:
    clusters = [c for c in np.unique(labels) if c != -1]
    n_noise = int((labels == -1).sum())
    sizes = {int(c): int((labels == c).sum()) for c in clusters}

    sil = None
    if len(clusters) >= 2:
        idx = np.where(labels != -1)[0]
        if len(idx) > 4000:  # silhouette is O(n^2) — subsample
            idx = np.random.default_rng(0).choice(idx, 4000, replace=False)
        sil = float(silhouette_score(X[idx], labels[idx]))

    is_real = _classify(df, labels)
    main, elong = _main_cluster(df, labels, is_real, sizes)
    return ClusterResult(
        labels=labels, algorithm=algorithm,
        runtime_ms=(time.perf_counter() - t0) * 1000,
        n_clusters=len(clusters), n_noise=n_noise, noise_frac=n_noise / len(labels),
        silhouette=sil, sizes=sizes, is_real=is_real,
        main_cluster=main, elongation=elong,
    )


def _classify(df: pd.DataFrame, labels: np.ndarray) -> dict[int, bool]:
    """A cluster is 'real' if kinematically cold (tight proper motion).

    Proper motion, not parallax, is the stream signature: for distant streams
    (GD-1, Pal 5) parallax ~ 0.1 mas with comparable errors, so parallax.std is
    noise. A tight pm clump (< 1 mas/yr scatter) separates a cold stream from
    the broad field.
    """
    out: dict[int, bool] = {}
    for c in np.unique(labels):
        if c == -1:
            continue
        s = df[labels == c]
        out[int(c)] = bool(
            s["pmra"].std() < 1.0
            and s["pmdec"].std() < 1.0
            and len(s) >= 10
        )
    return out


def _main_cluster(df: pd.DataFrame, labels: np.ndarray, is_real: dict[int, bool],
                  sizes: dict[int, int]) -> tuple[int | None, dict[int, float]]:
    """Main = largest 'real' cluster (fallback: largest overall).

    Elongation = major/minor axis ratio of on-sky positions (PCA via SVD);
    a stream is long and narrow, so this is a proxy for 'streamness'.
    """
    elong: dict[int, float] = {}
    for c in sizes:
        pts = df[labels == c][["ra", "dec"]].to_numpy(float)
        pts = pts - pts.mean(0)
        if len(pts) < 3:
            elong[c] = 1.0
            continue
        s = np.linalg.svd(pts, compute_uv=False)
        elong[c] = float((s[0] + 1e-9) / (s[1] + 1e-9))

    real = {c for c, ok in is_real.items() if ok}
    pool = [c for c in sizes if c in real] or list(sizes)
    main = max(pool, key=lambda c: sizes[c]) if pool else None
    return main, elong


def run_dbscan(X: np.ndarray, df: pd.DataFrame, *, eps: float = 0.5,
               min_samples: int = 12) -> ClusterResult:
    t0 = time.perf_counter()
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X)
    return _finish(X, df, labels, "DBSCAN", t0)


def run_hdbscan(X: np.ndarray, df: pd.DataFrame, *, min_cluster_size: int = 12,
                min_samples: int = 1, cluster_selection_epsilon: float = 0.0) -> ClusterResult:
    t0 = time.perf_counter()
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    ).fit_predict(X)
    return _finish(X, df, labels, "HDBSCAN", t0)


def run_kmeans(X: np.ndarray, df: pd.DataFrame, *, n_clusters: int = 5,
               seed: int = 0) -> ClusterResult:
    t0 = time.perf_counter()
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)
    return _finish(X, df, labels, "KMeans", t0)
