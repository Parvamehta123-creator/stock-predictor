"""
Backtests a trading strategy driven by the model's predictions, evaluated
ONLY on out-of-sample (walk-forward test-fold) predictions.

Why not just train on everything and backtest on the same data? Because
that's in-sample evaluation -- the model has already seen those exact
outcomes during training, so the backtest would be measuring memorization,
not prediction. This module reuses the same TimeSeriesSplit folds from
modeling.py and concatenates ONLY the test-fold predictions, so every
predicted day was genuinely unseen by the model that predicted it.

One consequence worth knowing: the first fold's training window (the
initial ~15-20% of the dataset) has no out-of-sample prediction at all --
there's nothing before it to train on. The backtest necessarily starts
partway through the dataset, not at day 1.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from pipeline.modeling import TARGET_COLUMN

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

TRADING_DAYS_PER_YEAR = 252


def out_of_sample_predictions(df: pd.DataFrame, feature_cols: list[str], model_fn, n_splits: int = 5) -> pd.DataFrame:
    """Concatenates predictions across all walk-forward test folds. Rows
    before the first fold's test set (used only for initial training) are
    excluded -- there is no out-of-sample prediction for them."""
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].astype(int).values

    rows = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        for idx, pred, actual in zip(test_idx, preds, y[test_idx]):
            rows.append({"index": idx, "prediction": pred, "actual": actual})

    result = pd.DataFrame(rows).set_index("index").sort_index()
    return df.join(result, how="inner")


def simulate_strategy(df_with_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Long/flat strategy: hold the stock (capture that day's return) when the
    model predicts UP; sit in cash (zero return) when it predicts DOWN. No
    shorting, no leverage, no transaction costs -- deliberately the simplest
    possible strategy, because the point is evaluating the model's signal,
    not strategy sophistication.
    """
    df = df_with_predictions.copy()
    df["strategy_return"] = np.where(df["prediction"] == 1, df["return_1d"].shift(-1), 0.0)
    df["buy_hold_return"] = df["return_1d"].shift(-1)

    # The last row has no next-day return to realize -- drop it rather than
    # silently treating a NaN return as zero.
    df = df.iloc[:-1].copy()

    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    df["buy_hold_equity"] = (1 + df["buy_hold_return"]).cumprod()
    return df


def sharpe_ratio(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def summarize_backtest(df: pd.DataFrame) -> dict:
    return {
        "strategy_total_return": df["strategy_equity"].iloc[-1] - 1,
        "buy_hold_total_return": df["buy_hold_equity"].iloc[-1] - 1,
        "strategy_sharpe": sharpe_ratio(df["strategy_return"]),
        "buy_hold_sharpe": sharpe_ratio(df["buy_hold_return"]),
        "strategy_max_drawdown": max_drawdown(df["strategy_equity"]),
        "buy_hold_max_drawdown": max_drawdown(df["buy_hold_equity"]),
        "days_in_market": int((df["prediction"] == 1).sum()),
        "total_days": len(df),
    }


if __name__ == "__main__":
    import sys
    from pipeline.modeling import load_dataset, get_baseline_model

    ticker = sys.argv[1] if len(sys.argv) > 1 else "DEMO"
    df, feature_cols = load_dataset(ticker)

    preds_df = out_of_sample_predictions(df, feature_cols, get_baseline_model)
    backtest_df = simulate_strategy(preds_df)
    summary = summarize_backtest(backtest_df)

    print(f"Backtest window: {backtest_df['Date'].iloc[0].date()} to {backtest_df['Date'].iloc[-1].date()} ({len(backtest_df)} trading days)\n")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_path = PROCESSED_DIR / f"{ticker.upper()}_backtest.csv"
    backtest_df[["Date", "prediction", "strategy_equity", "buy_hold_equity"]].to_csv(out_path, index=False)
    print(f"\nSaved equity curves to {out_path}")
