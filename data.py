"""Gaia DR3 data access with parquet caching + a deterministic synthetic fallback.

Data path:
  1. Query Gaia DR3 (direct TAP sync, fast) -> cache to ./cache/*.parquet.
  2. On any failure (no network, empty result), fall back to a synthetic
     GD-1-like stream so the reveal demo still works offline.

ponytail: 6D kinematics are not computed here — distance = 1/parallax is
meaningless for distant streams (GD-1 ~ 8 kpc), so position + proper motion is
the signal. Add `gala` if a physics-accurate 6D phase space is ever required.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

CACHE_DIR = Path(__file__).resolve().parent / "cache"


@dataclass(frozen=True)
class Region:
    """A cone-search region on the sky (ICRS degrees)."""

    name: str
    ra: float
    dec: float
    radius: float
    mag_limit: float = 19.0


# Approximate stream centres. GD-1 is the headline preset and the most reliable.
REGIONS: dict[str, Region] = {
    "GD-1": Region("GD-1", 150.0, 40.0, 8.0),
    "Palomar 5": Region("Palomar 5", 229.02, -0.11, 1.5, mag_limit=20.5),
    "Sagittarius": Region("Sagittarius", 283.0, -30.0, 10.0),
    "Orphan": Region("Orphan", 168.0, 48.0, 10.0),
    "Helmi": Region("Helmi", 320.0, -45.0, 15.0),
}

_COLUMNS = (
    "source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec, "
    "phot_g_mean_mag, bp_rp, radial_velocity, ruwe"
)


def _query(region: Region, mag_limit: float = 19.0, max_rows: int = 50000) -> str:
    # NOTE: no `parallax_over_error > 5` filter. Distant streams (GD-1, Sgr, ...)
    # have parallax ~ 0.1 mas with comparable errors, so that cut would delete the
    # stream. Proper motion + position are the signal. `parallax < 2` drops nearby
    # foreground stars (< 500 pc) whose huge proper motion (±100s mas/yr) would
    # otherwise dominate the scaling and swamp the stream.
    # `TOP max_rows ORDER BY random_index` bounds the download to a uniform random
    # sample of the cone; 50k field stars is more than enough noise to hide a stream.
    # (A magnitude sort would force a full cone sort and take minutes.)
    return f"""
        SELECT TOP {max_rows} {_COLUMNS}
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {region.ra}, {region.dec}, {region.radius}))
          AND phot_g_mean_mag < {mag_limit}
          AND ruwe < 1.4
          AND parallax > -1 AND parallax < 2.0
        ORDER BY random_index
    """


# TAP sync endpoints — the ESAC archive queues async jobs slowly; the sync endpoint
# returns a bounded CSV in seconds. Heidelberg mirror is used first (fast), ESAC as fallback.
_TAP_URLS = (
    "https://gaia.ari.uni-heidelberg.de/tap/sync",
    "https://gea.esac.esa.int/tap-server/tap/sync",
)


def query_gaia(region: Region, mag_limit: float = 19.0, max_rows: int = 50000,
               timeout: float = 240.0) -> pd.DataFrame:
    """Run a Gaia DR3 cone search via direct TAP sync and return a DataFrame."""
    adql = _query(region, mag_limit, max_rows)
    last: Exception | None = None
    for url in _TAP_URLS:
        try:
            r = requests.post(
                url,
                data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql},
                timeout=timeout,
            )
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:  # noqa: BLE001 — try the next mirror
            last = e
    raise RuntimeError(f"Gaia TAP query failed for {region.name}: {last}")


def cache_path(region_name: str) -> Path:
    return CACHE_DIR / f"{region_name.lower().replace(' ', '_')}.parquet"


def load_region(region: Region, *, use_cache: bool = True, mag_limit: float | None = None,
                max_rows: int = 50000) -> pd.DataFrame:
    """Return cached Gaia data, querying (and caching) on first use.

    Raises on query failure so callers can fall back to `synthetic_stream`.
    """
    if mag_limit is None:
        mag_limit = region.mag_limit
    p = cache_path(region.name)
    if use_cache and p.exists():
        return pd.read_parquet(p)
    df = query_gaia(region, mag_limit=mag_limit, max_rows=max_rows)
    if df.empty:
        raise RuntimeError(f"Gaia returned no rows for {region.name}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return df


def clear_cache() -> None:
    """Delete all cached parquet files (used by the 'query again' button)."""
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.parquet"):
            p.unlink()


def _stable01(name: str) -> float:
    """Deterministic float in [0, 1) from a name (stable across runs)."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") / 2**32


def synthetic_stream(region: Region | None = None, n_total: int = 9000,
                     n_stream: int = 550, seed: int = 7) -> pd.DataFrame:
    """Deterministic simulated stream embedded in field-star noise.

    The stream is a curved track in (ra, dec) with a tight clump in (pmra, pmdec),
    centred on `region` (default GD-1). Each region gets its own on-sky position
    and proper-motion vector, so switching regions produces visibly different maps
    even offline. Clearly a stand-in, not real Gaia data.
    """
    if region is None:
        region = REGIONS["GD-1"]
    rng = np.random.default_rng(seed)
    n_field = n_total - n_stream

    ra0, dec0, rad = region.ra, region.dec, region.radius
    pm_ra0 = -11 + 20 * _stable01(region.name)
    pm_dec0 = -9 + 18 * _stable01(region.name + ":dec")

    ra = np.concatenate([rng.uniform(ra0 - rad, ra0 + rad, n_field), np.zeros(n_stream)])
    dec = np.concatenate([rng.uniform(dec0 - rad, dec0 + rad, n_field), np.zeros(n_stream)])
    pmra = np.concatenate([rng.normal(0, 12, n_field), np.zeros(n_stream)])
    pmdec = np.concatenate([rng.normal(0, 12, n_field), np.zeros(n_stream)])

    t = rng.uniform(-1, 1, n_stream)
    ra[n_field:] = ra0 + 0.8 * rad * t + rng.normal(0, 0.02 * rad, n_stream)
    dec[n_field:] = dec0 + 0.6 * rad * t - 0.3 * rad * t**2 + rng.normal(0, 0.02 * rad, n_stream)
    pmra[n_field:] = pm_ra0 + 0.6 * t + rng.normal(0, 0.3, n_stream)
    pmdec[n_field:] = pm_dec0 - 0.5 * t + rng.normal(0, 0.3, n_stream)

    parallax = np.concatenate([
        np.abs(rng.normal(1.2, 0.5, n_field)),
        0.125 + rng.normal(0, 0.02, n_stream),
    ])
    g = rng.uniform(16.5, 19.0, n_total)
    return pd.DataFrame({
        "source_id": np.arange(n_total),
        "ra": ra,
        "dec": dec,
        "parallax": parallax,
        "parallax_over_error": np.abs(rng.normal(3, 1, n_total)),
        "pmra": pmra,
        "pmdec": pmdec,
        "phot_g_mean_mag": g,
        "bp_rp": 0.5 + 0.6 * (g - 16.5) / 2.5 + rng.normal(0, 0.1, n_total),
        "radial_velocity": np.nan,
        "ruwe": rng.uniform(0.9, 1.3, n_total),
    })


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing core features and add derived columns."""
    df = df.dropna(subset=["ra", "dec", "pmra", "pmdec", "parallax"]).reset_index(drop=True)
    df["pm_total"] = np.hypot(df["pmra"], df["pmdec"])
    return df


def build_features(
    df: pd.DataFrame,
    *,
    pm_weight: float = 2.0,
    include_parallax: bool = False,
) -> tuple[np.ndarray, StandardScaler]:
    """Return (X, scaler): position scaled to unit variance, proper motion
    amplified by `pm_weight` — the demo's core lever."""
    cols = ["ra", "dec", "pmra", "pmdec"]
    if include_parallax:
        cols.append("parallax")
    scaler = StandardScaler().fit(df[cols])
    X = scaler.transform(df[cols])
    X[:, 2:4] *= pm_weight  # pmra, pmdec
    if include_parallax:
        X[:, 4] *= 1.5
    return X, scaler
