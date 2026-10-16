"""
Earthquake Spatial Clustering Dashboard
=======================================
Unsupervised spatial clustering (DBSCAN / HDBSCAN) of earthquake data.

Data source:
  * Bundled dataset — `earthquakes.csv` (real USGS "all_month" snapshot),
    works fully offline.

Install dependencies
--------------------
    pip install streamlit pandas numpy scikit-learn plotly hdbscan

Run locally
-----------
    streamlit run app.py
"""

from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).with_name("earthquakes.csv")
EARTH_RADIUS_KM = 6371.0  # mean Earth radius; converts eps (km) <-> radians
MAX_SILHOUETTE_POINTS = 2000  # silhouette is O(n^2); sample above this
MAP_MAX_POINTS = 4000  # browser map gets heavy past this; thin points for it only

# Categorical color theme for clusters. Fixed order, never re-sorted, so a
# cluster keeps its color when the user changes a slider. Noise gets grey.
# ponytail: > 24 clusters reuses hues; fold excess clusters into "Other" when
# that matters for real analysis.
CLUSTER_PALETTE = px.colors.qualitative.Light24
NOISE_COLOR = "#9e9e9e"


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_bundled_dataset() -> pd.DataFrame:
    """Load the bundled CSV snapshot (offline-friendly)."""
    return pd.read_csv(DATA_FILE, parse_dates=["time"])


def clean_earthquake_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and obvious coordinate artifacts."""
    df = df.dropna(subset=["latitude", "longitude", "magnitude"]).copy()
    # USGS emits (0, 0) for some events with an unknown fix location.
    df = df[(df["latitude"] != 0) | (df["longitude"] != 0)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Machine learning engine (DBSCAN / HDBSCAN)
# ---------------------------------------------------------------------------
def _coords_rad(df: pd.DataFrame) -> np.ndarray:
    return np.radians(df[["latitude", "longitude"]].to_numpy())


def run_dbscan(df: pd.DataFrame, eps_km: float, min_samples: int) -> np.ndarray:
    """Cluster (lat, lon) with DBSCAN using haversine distance."""
    if df.empty:
        return np.array([], dtype=int)
    db = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    return db.fit_predict(_coords_rad(df))


def run_hdbscan(
    df: pd.DataFrame,
    min_cluster_size: int,
    min_samples: int | None,
    eps_km: float,
) -> np.ndarray:
    """Cluster (lat, lon) with HDBSCAN using haversine distance.

    `eps_km` maps to `cluster_selection_epsilon` (HDBSCAN's analogue of DBSCAN
    eps), converted to radians; 0 disables it.
    """
    if df.empty:
        return np.array([], dtype=int)
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=(eps_km / EARTH_RADIUS_KM) if eps_km else 0.0,
        metric="haversine",
    )
    return hdb.fit_predict(_coords_rad(df))


def compute_silhouette(df: pd.DataFrame, labels: np.ndarray) -> float | None:
    """Mean silhouette over clustered points (noise excluded).

    Returns None when the result can't be scored (all noise / one cluster).
    Computed on a bounded sample because the metric is O(n^2) in points.
    """
    clustered = df[labels != -1].reset_index(drop=True)
    clustered_labels = labels[labels != -1]
    if len(np.unique(clustered_labels)) < 2:
        return None

    if len(clustered) > MAX_SILHOUETTE_POINTS:
        pos = np.random.default_rng(0).choice(
            len(clustered), MAX_SILHOUETTE_POINTS, replace=False
        )
        coords = np.radians(clustered.iloc[pos][["latitude", "longitude"]].to_numpy())
        sample_labels = clustered_labels[pos]
    else:
        coords = np.radians(clustered[["latitude", "longitude"]].to_numpy())
        sample_labels = clustered_labels

    return float(silhouette_score(coords, sample_labels, metric="haversine"))


def downsample_for_map(df: pd.DataFrame) -> pd.DataFrame:
    """Thin points for the browser map only (metrics and table use the full df).

    Stratified per cluster with a floor so small fault zones stay visible, and
    deterministic (seed 0) so the map doesn't flicker between reruns.
    """
    if len(df) <= MAP_MAX_POINTS:
        return df
    frac = MAP_MAX_POINTS / len(df)
    rng = np.random.default_rng(0)
    parts = []
    for _, g in df.groupby("cluster", observed=True, sort=False):
        keep = max(20, int(len(g) * frac))
        keep = min(keep, len(g))
        if keep < len(g):
            g = g.sample(keep, random_state=int(rng.integers(2**31)))
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Earthquake Spatial Clustering", layout="wide")

st.title("🌍 Earthquake Spatial Clustering")
st.caption(
    "Unsupervised detection of seismic fault zones by clustering (lat, lon) with "
    "a haversine-distance DBSCAN or HDBSCAN."
)

# ---- Sidebar controls ------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    algorithm = st.radio(
        "Clustering algorithm",
        ["DBSCAN", "HDBSCAN"],
        help="DBSCAN separates by density with a fixed radius; HDBSCAN finds "
        "clusters of varying density and needs no eps.",
    )

    if algorithm == "DBSCAN":
        eps_km = st.number_input(
            "eps (km)",
            min_value=1.0,
            max_value=10000.0,
            value=25.0,
            step=10.0,
            help="Maximum neighbour distance. Larger eps merges more quakes "
            "into the same fault zone. Type any value within range.",
        )
        min_samples = st.number_input(
            "min_samples",
            min_value=1,
            max_value=500,
            value=40,
            step=1,
            help="Minimum quakes to form a dense region. Higher = fewer, "
            "tighter clusters and more noise.",
        )
        min_cluster_size = None
    else:
        min_cluster_size = st.number_input(
            "min_cluster_size",
            min_value=2,
            max_value=2000,
            value=100,
            step=1,
            help="Smallest cluster to keep (HDBSCAN's main knob). Higher = "
            "fewer, larger, more conservative clusters.",
        )
        auto_min_samples = st.checkbox(
            "min_samples = min_cluster_size (default)", value=True
        )
        min_samples = None if auto_min_samples else st.number_input(
            "min_samples",
            min_value=1,
            max_value=2000,
            value=5,
            step=1,
            help="Neighbourhood size for density estimation. Lower = more "
            "noise-tolerant; higher = more points required nearby.",
        )
        eps_km = st.number_input(
            "cluster_selection_epsilon (km, 0 = off)",
            min_value=0.0,
            max_value=10000.0,
            value=20.0,
            step=10.0,
            help="HDBSCAN analogue of DBSCAN eps: merges clusters whose points "
            "are within this distance. 0 disables it.",
        )

# ---- Load + clean ----------------------------------------------------------
raw = load_bundled_dataset()
df = clean_earthquake_data(raw)

if df.empty:
    st.warning("No earthquakes to analyze.")
    st.stop()

labels = (
    run_dbscan(df, eps_km, min_samples)
    if algorithm == "DBSCAN"
    else run_hdbscan(df, min_cluster_size, min_samples, eps_km)
)

df["cluster"] = labels
# Readable legend label; noise is kept separate from the cluster groups.
df["group"] = np.where(labels == -1, "Noise", [f"Cluster {c}" for c in labels])
# Display-only size, floored so negative magnitudes don't produce bad markers.
df["size"] = df["magnitude"].clip(lower=0.5)

cluster_ids = sorted(set(labels) - {-1})
n_clusters = len(cluster_ids)
n_noise = int((labels == -1).sum())
n_total = len(df)
noise_pct = (n_noise / n_total * 100.0) if n_total else 0.0
silhouette = compute_silhouette(df, labels)

# ---- Metrics row -----------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total earthquakes analyzed", f"{n_total:,}")
col2.metric("Distinct clusters (fault zones)", f"{n_clusters:,}")
col3.metric("Noise (isolated quakes)", f"{noise_pct:.1f}%")
col4.metric(
    "Silhouette score",
    "—" if silhouette is None else f"{silhouette:.3f}",
    help="Internal: how well-separated the clusters are. Near +1 = compact and "
    "apart; near 0 = overlapping. Not shown when there is one cluster or none.",
)

# ---- Map -------------------------------------------------------------------
# Fixed color assignment: cluster id -> palette position, noise -> grey.
color_map = {"Noise": NOISE_COLOR}
for i, cid in enumerate(cluster_ids):
    color_map[f"Cluster {cid}"] = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]

group_order = ["Noise"] + [f"Cluster {c}" for c in cluster_ids]

# scatter_geo needs no Mapbox token and renders offline (bundled coastlines).
map_df = downsample_for_map(df)
fig = px.scatter_geo(
    map_df,
    lat="latitude",
    lon="longitude",
    color="group",
    color_discrete_map=color_map,
    size="size",
    size_max=12,
    hover_name="place",
    hover_data={
        "magnitude": ":.1f",
        "depth_km": ":.1f",
        "time": True,
        "latitude": ":.3f",
        "longitude": ":.3f",
        "group": False,
        "size": False,
    },
    projection="natural earth",
    category_orders={"group": group_order},
    labels={"group": "Seismic zone"},
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600)
fig.update_traces(marker=dict(sizemin=3))
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Map shows a stratified sample ({len(map_df):,} of {len(df):,} quakes) to keep "
    "the view responsive; metrics and table below use the full dataset."
)

# ---- Cluster size distribution ---------------------------------------------
st.subheader("Cluster sizes")
sizes = (
    df.groupby("group", observed=True)
    .size()
    .reindex(group_order)
    .dropna()
    .astype(int)
    .reset_index(name="count")
)
fig2 = px.bar(
    sizes,
    x="group",
    y="count",
    color="group",
    color_discrete_map=color_map,
    labels={"group": "Seismic zone", "count": "Earthquakes"},
    category_orders={"group": group_order},
)
fig2.update_layout(showlegend=False, height=320, margin={"t": 20})
st.plotly_chart(fig2, use_container_width=True)

# ---- Data table ------------------------------------------------------------
st.subheader("Sample records")
st.caption("Strongest events first; `cluster = -1` marks isolated noise points.")
table_cols = [
    "place",
    "magnitude",
    "depth_km",
    "latitude",
    "longitude",
    "time",
    "cluster",
]
st.dataframe(
    df[table_cols]
    .rename(columns={"cluster": "cluster_id"})
    .sort_values("magnitude", ascending=False)
    .head(100),
    use_container_width=True,
)
