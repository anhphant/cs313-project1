"""Shared constants and small helpers for the fraud-detection demo."""
from __future__ import annotations

import time

# Dataset schema.
LABEL_COL = "Class"
DEFAULT_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
REQUIRED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

# Detector names (also used as UI labels).
DETECTORS = ["DBSCAN", "HDBSCAN", "Local Outlier Factor", "Isolation Forest"]

# Consistent, colorblind-aware color mapping (dataviz default palette).
COLOR_NORMAL = "#2a78d6"   # blue
COLOR_FRAUD = "#e34948"    # red
COLOR_UNKNOWN = "#898781"  # muted gray (no label / no prediction)

# TP / TN / FP / FN -> status palette. Text labels are always shown alongside.
OUTCOME_ORDER = ["TP", "TN", "FP", "FN"]
OUTCOME_COLORS = {
    "TP": "#0ca30c",  # good (green)
    "TN": "#2a78d6",  # neutral/blue
    "FP": "#eb6834",  # warning (orange)
    "FN": "#d03b3b",  # critical (red)
}


class Timer:
    """Context manager / callable for wall-clock timing."""

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self.start


def fmt_runtime(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def fmt_count(n: int) -> str:
    return f"{n:,}"
