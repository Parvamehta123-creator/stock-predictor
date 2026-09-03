"""
Tests for pipeline/features.py.

The most important test here isn't "does it run" -- it's proving that features
computed for a given row don't change if future rows are removed. If they did,
that would mean a feature is peeking into data that shouldn't be available yet
(lookahead bias) -- the single most common bug that makes a backtest look great
and a live model fail.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.features import add_technical_indicators, add_target
from pipeline.generate_sample_data import generate_ohlcv


def test_indicators_do_not_use_future_data():
    """Truncating the dataframe to the past should not change earlier feature values."""
    full_df = generate_ohlcv("TEST", days=200, seed=1)
    full_features = add_technical_indicators(full_df)

    cutoff = 150
    truncated_df = full_df.iloc[:cutoff].copy()
    truncated_features = add_technical_indicators(truncated_df)

    # Every row up to the cutoff should be identical whether or not future
    # rows existed when the indicator was computed.
    compare_cols = [c for c in full_features.columns if c not in ("Date",)]
    pd.testing.assert_frame_equal(
        full_features.loc[: cutoff - 1, compare_cols].reset_index(drop=True),
        truncated_features.loc[:, compare_cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_target_matches_future_close():
    """target[t] should be exactly (Close[t+1] > Close[t]), by construction."""
    df = generate_ohlcv("TEST", days=50, seed=2)
    df = add_target(df, horizon=1)

    for t in range(len(df) - 1):
        expected = int(df["Close"].iloc[t + 1] > df["Close"].iloc[t])
        assert df["target"].iloc[t] == expected, f"Mismatch at row {t}"


def test_last_rows_have_nan_target():
    """The final `horizon` rows have no future price -- their target must be NaN, not guessed."""
    df = generate_ohlcv("TEST", days=50, seed=3)
    df = add_target(df, horizon=1)
    assert np.isnan(df["target"].iloc[-1])


def test_dropna_removes_warmup_and_final_rows():
    """After dropna, no row should have an unresolved (NaN) target or indicator."""
    df = generate_ohlcv("TEST", days=200, seed=4)
    df = add_technical_indicators(df)
    df = add_target(df, horizon=1)
    clean = df.dropna()
    assert clean.isna().sum().sum() == 0
    assert len(clean) < len(df)  # confirms warmup/tail rows were actually dropped
