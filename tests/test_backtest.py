"""
Tests for pipeline/backtest.py.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.backtest import max_drawdown, sharpe_ratio, simulate_strategy


def test_sharpe_ratio_zero_std_returns_zero():
    """A constant return series has zero volatility -- dividing by std=0
    must not raise or return NaN/inf."""
    constant_returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert sharpe_ratio(constant_returns) == 0.0


def test_sharpe_ratio_positive_for_positive_mean_return():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    assert sharpe_ratio(returns) > 0


def test_max_drawdown_known_case():
    """Equity path 1.0 -> 1.2 -> 1.1 -> 0.9 -> 1.3: the worst drop is from
    the peak of 1.2 down to 0.9, a -25% drawdown. Computed by hand so the
    test isn't just re-deriving the function it's checking."""
    equity = pd.Series([1.0, 1.2, 1.1, 0.9, 1.3])
    expected = (0.9 - 1.2) / 1.2  # -0.25
    assert abs(max_drawdown(equity) - expected) < 1e-9


def test_strategy_stays_flat_when_never_predicting_up():
    """If the model predicts 0 (down/flat) every day, the strategy should
    never take a position -- equity must stay exactly at 1.0 throughout."""
    df = pd.DataFrame({
        "return_1d": [0.01, -0.02, 0.03, 0.01, -0.01],
        "prediction": [0, 0, 0, 0, 0],
    })
    result = simulate_strategy(df)
    assert (result["strategy_return"] == 0.0).all()
    assert (result["strategy_equity"] == 1.0).all()


def test_strategy_captures_next_day_return_when_predicting_up():
    df = pd.DataFrame({
        "return_1d": [0.01, 0.02, -0.01, 0.03, 0.01],
        "prediction": [1, 1, 1, 1, 1],
    })
    result = simulate_strategy(df)
    # strategy_return[t] should equal return_1d[t+1] (shift(-1)) when always predicting up
    expected = df["return_1d"].shift(-1).iloc[:-1]
    pd.testing.assert_series_equal(result["strategy_return"].reset_index(drop=True), expected.reset_index(drop=True), check_names=False)
