"""Grid-search DBSCAN / HDBSCAN hyperparameters for the earthquake app.

Reuses the exact scoring logic from app.py (haversine distances, silhouette on
a bounded sample) and dumps every evaluated config to `tune_results.json`.
"""
from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import hdbscan

DATA_FILE = Path(__file__).with_name("earthquakes.csv")
EARTH_RADIUS_KM = 6371.0
MAX_SILHOUETTE_POINTS = 2000


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["time"])
    df = df.dropna(subset=["latitude", "longitude", "magnitude"]).copy()
    df = df[(df["latitude"] != 0) | (df["longitude"] != 0)]
    return df.reset_index(drop=True)


def coords_rad(df: pd.DataFrame) -> np.ndarray:
    return np.radians(df[["latitude", "longitude"]].to_numpy())


def silhouette(df: pd.DataFrame, labels: np.ndarray) -> float | None:
    clustered = df[labels != -1].reset_index(drop=True)
    cl = labels[labels != -1]
    if len(np.unique(cl)) < 2:
        return None
    if len(clustered) > MAX_SILHOUETTE_POINTS:
        pos = np.random.default_rng(0).choice(
            len(clustered), MAX_SILHOUETTE_POINTS, replace=False
        )
        coords = np.radians(clustered.iloc[pos][["latitude", "longitude"]].to_numpy())
        sl = cl[pos]
    else:
        coords = np.radians(clustered[["latitude", "longitude"]].to_numpy())
        sl = cl
    return float(silhouette_score(coords, sl, metric="haversine"))


def score(df, labels):
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    noise_pct = n_noise / len(df) * 100.0
    sil = silhouette(df, labels)
    return {"n_clusters": n_clusters, "noise_pct": noise_pct, "silhouette": sil}


def main():
    df = load_clean()
    X = coords_rad(df)
    results = []

    # ---- DBSCAN -------------------------------------------------------------
    for eps_km, min_samples in itertools.product(
        [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150, 200, 300, 500],
        [3, 5, 10, 15, 20, 30, 40, 50, 75, 100],
    ):
        db = DBSCAN(
            eps=eps_km / EARTH_RADIUS_KM,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree",
        )
        labels = db.fit_predict(X)
        m = score(df, labels)
        results.append(
            {"algo": "DBSCAN", "eps_km": eps_km, "min_samples": min_samples, **m}
        )

    # ---- HDBSCAN ------------------------------------------------------------
    for mcs, ms, eps_km in itertools.product(
        [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500],
        [None, 3, 5, 15, 50],
        [0, 10, 20, 30, 50, 100],
    ):
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_epsilon=(eps_km / EARTH_RADIUS_KM) if eps_km else 0.0,
            metric="haversine",
        )
        labels = hdb.fit_predict(X)
        m = score(df, labels)
        results.append(
            {
                "algo": "HDBSCAN",
                "min_cluster_size": mcs,
                "min_samples": ms,
                "eps_km": eps_km,
                **m,
            }
        )

    with open("tune_results.json", "w") as f:
        json.dump(results, f, indent=2)

    for algo in ("DBSCAN", "HDBSCAN"):
        rows = [r for r in results if r["algo"] == algo]
        # usable: at least 2 clusters, not everything noise, not 1 giant blob
        usable = [r for r in rows if r["n_clusters"] >= 2 and 0 < r["noise_pct"] < 99]
        usable.sort(key=lambda r: r["silhouette"] or -1, reverse=True)
        print(f"\n===== {algo}: {len(rows)} configs, {len(usable)} usable =====")
        for r in usable[:15]:
            print(
                f"  sil={r['silhouette']:.3f}  n={r['n_clusters']:>4}  "
                f"noise={r['noise_pct']:5.1f}%  {r}"
            )


if __name__ == "__main__":
    main()
