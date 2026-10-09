"""2D projection (PCA / UMAP) for visualization only."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

PROJECTION_METHODS = ["PCA", "UMAP"]


def project(X: np.ndarray, method: str, seed: int) -> np.ndarray:
    """Project ``X`` to 2D. Used for the scatter plot, never for detection.

    ``X`` is assumed already scaled (see ``preprocessing.preprocess``).
    """
    if method == "PCA":
        return PCA(n_components=2, random_state=seed).fit_transform(X)

    if method == "UMAP":
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
            verbose=False,
        )
        return reducer.fit_transform(X)

    raise ValueError(f"Unknown projection method: {method}")
