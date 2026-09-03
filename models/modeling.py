"""
Model training and validation for the stock direction classifier.

THE CENTRAL LESSON OF THIS FILE:
Shuffled k-fold cross-validation on time-series data reports an accuracy
that is systematically too optimistic, and it does this WITHOUT crashing,
warning, or looking wrong. See compare_validation_strategies() -- it trains
the exact same model on the exact same data with only the split strategy
changed, and prints both numbers side by side.

Why shuffling leaks: adjacent trading days share overlapping rolling-window
inputs (a 10-day SMA on day t and day t+1 share 9 of the same 10 prices).
When shuffled, a training fold can contain day t+1 while the test fold
contains day t -- the model effectively "sees" test-adjacent information
during training, even though it never technically sees the target. Walk-
forward validation (train on the past, test on the future, always) is the
only split that matches how the model will actually be used.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, TimeSeriesSplit

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Engineered, roughly-stationary features only -- deliberately excluding raw
# price/volume levels (Close, sma_10, bb_upper, Volume, ...). A raw price of
# $150 in the training window and $50 in a later regime carry no comparable
# meaning to the model; ratios and oscillators (RSI, MACD, price-vs-SMA%) do.
FEATURE_COLUMNS = [
    "return_1d", "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "price_vs_sma10", "price_vs_sma50", "macd", "macd_signal", "rsi_14",
    "bb_position", "volume_ratio", "day_of_week", "month",
]
OPTIONAL_SENTIMENT_COLUMNS = ["mention_sentiment", "macro_sentiment", "mentioned"]
TARGET_COLUMN = "target"


def load_dataset(ticker: str, with_sentiment: bool = False) -> tuple[pd.DataFrame, list[str]]:
    suffix = "_features_with_sentiment" if with_sentiment else "_features"
    path = PROCESSED_DIR / f"{ticker.upper()}{suffix}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    feature_cols = list(FEATURE_COLUMNS)
    if with_sentiment:
        feature_cols += [c for c in OPTIONAL_SENTIMENT_COLUMNS if c in df.columns]

    return df, feature_cols


def get_baseline_model() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def get_boosted_model():
    """Prefers XGBoost; falls back to sklearn's GradientBoostingClassifier
    if xgboost isn't installed (e.g. this dev sandbox has no internet to
    install it). Swapping back to XGBoost locally requires no code changes --
    just having the package available."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("[get_boosted_model] xgboost not available -- using GradientBoostingClassifier instead.")
        return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)


def _fold_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "majority_baseline": max(y_true.mean(), 1 - y_true.mean()),  # accuracy of "always predict the majority class"
    }


def walk_forward_evaluate(df: pd.DataFrame, feature_cols: list[str], model_fn, n_splits: int = 5) -> pd.DataFrame:
    """
    TimeSeriesSplit with default settings uses an EXPANDING window: fold k's
    training set is everything before fold k's test set, growing each time.
    Every test fold is strictly later in time than everything it was trained
    on -- this is the walk-forward property that keeps validation honest.
    """
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].astype(int).values
    dates = df["Date"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        metrics = _fold_metrics(y[test_idx], preds)
        metrics.update({
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "test_start": pd.Timestamp(dates[test_idx[0]]).date(),
            "test_end": pd.Timestamp(dates[test_idx[-1]]).date(),
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def compare_validation_strategies(df: pd.DataFrame, feature_cols: list[str], model_fn=get_baseline_model, n_splits: int = 5) -> dict:
    """
    Trains the SAME model on the SAME data with only the CV strategy changed.
    Prints both mean accuracies side by side -- this is the concrete evidence
    for why walk-forward validation matters, not just an assertion.
    """
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].astype(int).values

    # Walk-forward (correct)
    wf_accuracies = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        wf_accuracies.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))

    # Shuffled k-fold (the common mistake)
    kf_accuracies = []
    for train_idx, test_idx in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        kf_accuracies.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))

    result = {
        "walk_forward_mean_accuracy": float(np.mean(wf_accuracies)),
        "shuffled_kfold_mean_accuracy": float(np.mean(kf_accuracies)),
        "inflation": float(np.mean(kf_accuracies) - np.mean(wf_accuracies)),
    }
    return result


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "DEMO"

    df, feature_cols = load_dataset(ticker)
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features for {ticker.upper()}\n")

    print("=== Walk-forward validation (baseline: logistic regression) ===")
    results = walk_forward_evaluate(df, feature_cols, get_baseline_model, n_splits=5)
    print(results.to_string(index=False))
    print(f"\nMean accuracy: {results['accuracy'].mean():.4f}  |  Mean majority-class baseline: {results['majority_baseline'].mean():.4f}")

    print("\n=== Shuffled k-fold vs. walk-forward: does the split strategy matter? ===")
    comparison = compare_validation_strategies(df, feature_cols)
    for k, v in comparison.items():
        print(f"  {k}: {v:.4f}")
