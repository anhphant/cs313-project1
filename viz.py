"""Plotly figures for GAIA STREAM HUNTER.

Colour roles (dark-mode reference palette from the dataviz skill):
  HERO  gold   — the revealed stream (documented status-warning step #fab219)
  OTHER fixed-order categorical ramp for secondary clusters
  NOISE muted gray, dimmed via opacity for field stars
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

HERO = "#fab219"
OTHER = ["#3987e5", "#199e70", "#9085e9", "#d55181"]  # blue, aqua, violet, magenta
NOISE = "#898781"
SCAN = "#f0efec"

SURFACE = "#1a1a19"
GRID = "#2c2c2a"
INK = "#f0efec"
MUTED = "#898781"

AXIS = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#383835",
            title_font=dict(color=INK), tickfont=dict(color=MUTED))
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
    font=dict(family="system-ui, Segoe UI, sans-serif", color=INK),
    margin=dict(l=40, r=16, t=40, b=40),
    legend=dict(font=dict(color=INK), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="#2c2c2a", font=dict(color=INK)),
)

_HOVER = ("RA %{customdata[0]:.3f}°, Dec %{customdata[1]:.3f}°<br>"
          "pm_ra %{customdata[2]:.2f} · pm_dec %{customdata[3]:.2f} mas/yr<br>"
          "G %{customdata[4]:.2f} mag<extra></extra>")


def cluster_palette(labels: np.ndarray, main: int | None) -> dict[int, str]:
    """Stable colours keyed by cluster id; the stream keeps HERO everywhere."""
    others = [c for c in sorted(np.unique(labels)) if c != -1 and c != main]
    pal: dict[int, str] = {}
    if main is not None:
        pal[int(main)] = HERO
    for i, c in enumerate(others):
        pal[int(c)] = OTHER[i % len(OTHER)]
    return pal


def _sizes(g: np.ndarray) -> np.ndarray:
    return np.clip(2.5 + (19 - g) * 1.5, 2, 8)


def _customdata(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([df["ra"], df["dec"], df["pmra"], df["pmdec"],
                            df["phot_g_mean_mag"]])


def _scatter(df: pd.DataFrame, x, y, z, *, color, alpha, name, legendgroup=None, size=None):
    cd = _customdata(df)
    s = np.full(len(df), size, float) if size is not None else _sizes(df["phot_g_mean_mag"].to_numpy())
    kw = dict(
        x=x, y=y, mode="markers", name=name, legendgroup=legendgroup,
        showlegend=True,
        marker=dict(size=s, color=color, opacity=alpha, line=dict(width=0)),
        customdata=cd, hovertemplate=_HOVER,
    )
    if z is not None:
        kw["z"] = z
        return go.Scatter3d(**kw)
    return go.Scatter(**kw)


def _galactic(df: pd.DataFrame):
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        c = SkyCoord(ra=df["ra"].to_numpy() * u.deg, dec=df["dec"].to_numpy() * u.deg,
                     frame="icrs")
        g = c.galactic
        return pd.Series(g.l.deg), pd.Series(g.b.deg)
    except Exception:
        return df["ra"], df["dec"]  # astropy unavailable -> fall back to equatorial


def _axes(df: pd.DataFrame, space: str):
    if space == "sky":
        return df["ra"], df["dec"], None, "RA [deg]", "Dec [deg]", None
    if space == "pm":
        return df["pmra"], df["pmdec"], None, "pm_ra [mas/yr]", "pm_dec [mas/yr]", None
    if space == "phase_ra":
        return df["ra"], df["dec"], df["pm_total"], "RA [deg]", "Dec [deg]", "pm_total [mas/yr]"
    if space == "phase_pm":
        return df["pmra"], df["pmdec"], -df["phot_g_mean_mag"], "pm_ra [mas/yr]", "pm_dec [mas/yr]", "−G [mag]"
    if space == "galactic":
        l, b = _galactic(df)
        return l, b, df["pm_total"], "l [deg]", "b [deg]", "pm_total [mas/yr]"
    if space == "cmd":
        return df["bp_rp"], df["phot_g_mean_mag"], None, "BP−RP [mag]", "G [mag]", None
    raise ValueError(f"unknown space {space!r}")


def build_figure(df: pd.DataFrame, labels: np.ndarray, *, pal: dict[int, str],
                 main: int | None, space: str = "sky", mode: str = "final",
                 scan: np.ndarray | None = None, height: int = 520) -> go.Figure:
    """One figure per space. `mode` walks the reveal: raw -> scan -> main -> all -> final."""
    x, y, z, xl, yl, zl = _axes(df, space)
    traces: list = []

    def add(mask, color, alpha, name, lg=None, size=None):
        traces.append(_scatter(df[mask], x[mask], y[mask], z[mask] if z is not None else None,
                               color=color, alpha=alpha, name=name, legendgroup=lg, size=size))

    noise = labels == -1
    dim = {"raw": 0.85, "scan": 0.85, "main": 0.85, "all": 0.85, "final": 0.12}[mode]

    if mode in ("raw", "scan"):
        add(np.ones(len(df), bool), NOISE, 0.85, f"Raw Gaia ({len(df):,})", size=3)
        if mode == "scan" and scan is not None:
            add(scan, SCAN, 0.9, "Scanning…", "scan", size=6)
    else:
        add(noise, NOISE, dim, f"Field / noise ({int(noise.sum()):,})", size=3)
        if scan is not None:
            add(scan, SCAN, 0.9, "Scanning…", "scan", size=6)
        for c in sorted(pal):
            if c == main:
                continue
            m = labels == c
            add(m, pal[c] if mode != "main" else NOISE, 0.9 if mode != "main" else 0.85,
                f"Cluster {c}", size=5)
        if main is not None:
            m = labels == main
            add(m, HERO, 0.95, f"Stream ({int(m.sum()):,})", "main", size=9)

    fig = go.Figure(data=traces)
    fig.update_layout(height=height, **_BASE)
    if z is not None:
        fig.update_layout(scene=dict(
            xaxis=dict(title=xl, **AXIS), yaxis=dict(title=yl, **AXIS),
            zaxis=dict(title=zl, **AXIS), bgcolor=SURFACE,
        ))
    else:
        fig.update_layout(xaxis=dict(title=xl, **AXIS), yaxis=dict(title=yl, **AXIS))
    if space == "cmd":
        fig.update_yaxes(autorange="reversed")
    return fig


def cmd_figure(df: pd.DataFrame, labels: np.ndarray, pal: dict[int, str],
               main: int | None, height: int = 520) -> go.Figure:
    m = df["bp_rp"].notna().to_numpy()
    return build_figure(df[m].reset_index(drop=True), labels[m], pal=pal, main=main,
                        space="cmd", mode="final", height=height)


def thin(df: pd.DataFrame, labels: np.ndarray, *, max_noise: int = 2000,
         max_points: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Downsample for a clear plot: keep all cluster members, cap the field.

    Noise (labels == -1) is capped at `max_noise`; the total is then capped at
    `max_points` (uniform subsample). The stream stays fully visible against a
    thin background.
    """
    rng = np.random.default_rng(seed)
    keep = np.ones(len(labels), bool)

    noise = np.where(labels == -1)[0]
    if len(noise) > max_noise:
        keep[rng.choice(noise, len(noise) - max_noise, replace=False)] = False

    idx = np.where(keep)[0]
    if len(idx) > max_points:
        keep[rng.choice(idx, len(idx) - max_points, replace=False)] = False

    return df[keep].reset_index(drop=True), labels[keep]


def reveal_stages(df: pd.DataFrame, labels: np.ndarray, pal: dict[int, str],
                  main: int | None, space: str = "pm", seed: int = 0) -> list[tuple[str, go.Figure]]:
    """The five staged figures behind the REVEAL button."""
    rng = np.random.default_rng(seed)
    n = len(df)
    scan = np.zeros(n, bool)
    scan[rng.choice(n, max(1, n // 25), replace=False)] = True
    return [(m, build_figure(df, labels, pal=pal, main=main, space=space,
                             mode=m, scan=scan)) for m in ("raw", "scan", "main", "all", "final")]
