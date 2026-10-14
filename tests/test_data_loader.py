import numpy as np
import pandas as pd
import pytest

from src.data_loader import has_label, load_credit_card_data, validate_dataframe


def _sample_df(n=50):
    rng = np.random.default_rng(0)
    cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    return pd.DataFrame(
        {c: rng.random(n) for c in cols},
    )


def test_has_label():
    df = _sample_df()
    assert has_label(df) is True
    assert has_label(df.drop(columns=["Class"])) is False


def test_validate_ok():
    df = _sample_df()
    assert validate_dataframe(df) is df


def test_validate_missing_raises():
    df = _sample_df().drop(columns=["V17", "Amount"])
    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_load_from_csv(tmp_path):
    df = _sample_df()
    path = tmp_path / "creditcard.csv"
    df.to_csv(path, index=False)
    loaded = load_credit_card_data(str(path))
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == len(df)
