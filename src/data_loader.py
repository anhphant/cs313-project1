"""Loading and validating the ULB credit-card fraud dataset."""
from __future__ import annotations

import pandas as pd

from .utils import LABEL_COL, REQUIRED_COLUMNS


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Raise a clear error if required feature columns are missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset thiếu các cột bắt buộc: " + ", ".join(missing)
        )
    return df


def load_credit_card_data(path: str) -> pd.DataFrame:
    """Read a CSV from disk and validate its schema."""
    df = pd.read_csv(path)
    validate_dataframe(df)
    return df


def has_label(df: pd.DataFrame) -> bool:
    return LABEL_COL in df.columns
