"""
Tests for pipeline/merge_sentiment.py.

test_weekend_note_rolls_to_next_trading_day is a regression test for the
bug found while building this: a note dated on a non-trading day (the
sample note is dated a Sunday) matched nothing in a plain merge on Date
and vanished with no error. That's a worse failure mode than a crash --
the pipeline runs "successfully" and just quietly drops the signal.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.merge_sentiment import align_to_trading_days, merge_with_features


def _trading_days(dates):
    return pd.Series(pd.to_datetime(dates))


def test_weekend_note_rolls_to_next_trading_day():
    trading_days = _trading_days(["2026-08-28", "2026-08-31", "2026-09-01"])  # Fri, Mon, Tue
    note_dates = _trading_days(["2026-08-30"])  # Sunday
    aligned = align_to_trading_days(note_dates, trading_days)
    assert aligned.iloc[0] == pd.Timestamp("2026-08-31")


def test_note_on_trading_day_is_unchanged():
    trading_days = _trading_days(["2026-08-28", "2026-08-31"])
    note_dates = _trading_days(["2026-08-28"])
    aligned = align_to_trading_days(note_dates, trading_days)
    assert aligned.iloc[0] == pd.Timestamp("2026-08-28")


def test_note_after_last_trading_day_is_dropped_not_guessed():
    trading_days = _trading_days(["2026-08-28", "2026-08-31"])
    note_dates = _trading_days(["2026-09-15"])  # far beyond available price data
    aligned = align_to_trading_days(note_dates, trading_days)
    assert pd.isna(aligned.iloc[0])


def test_mention_does_not_forward_fill_but_macro_does():
    feature_df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-28", "2026-08-31", "2026-09-01"]),
        "Close": [100.0, 101.0, 102.0],
    })
    sentiment_log = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-30"]),  # Sunday -> rolls to 08-31
        "ticker": ["CYIENT"],
        "mention_sentiment": [-1.0],
        "macro_sentiment": [-0.647],
    })
    merged = merge_with_features(feature_df, sentiment_log, "CYIENT")

    row_31 = merged[merged["Date"] == "2026-08-31"].iloc[0]
    row_09_01 = merged[merged["Date"] == "2026-09-01"].iloc[0]

    assert row_31["mentioned"] == 1
    assert row_31["mention_sentiment"] == -1.0
    # No new mention on 09-01 -> mention resets to 0 (silence is information)...
    assert row_09_01["mentioned"] == 0
    assert row_09_01["mention_sentiment"] == 0.0
    # ...but macro tone persists until a newer note overrides it.
    assert row_09_01["macro_sentiment"] == -0.647
