"""
Feature engineering: turns raw OHLCV data into a model-ready dataset.

THE ONE RULE THAT MATTERS MOST IN THIS FILE:
Every feature at row t must only use information available AT OR BEFORE t.
The target column is the only thing allowed to look forward -- and only
because that's literally what we're trying to predict.

We implement indicators by hand with pandas rolling/ewm windows instead of
pulling in a library. It's more code, but you can see exactly what data
each feature touches, which matters a lot when the whole point is proving
you didn't leak the future into your training set.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_raw(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No raw data at {path}. Run fetch_data.py or generate_sample_data.py first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Returns ---
    # pct_change() at row t compares Close[t] to Close[t-1] -- backward-looking. Safe.
    df["return_1d"] = df["Close"].pct_change(1)

    # --- Lagged returns (explicitly shifted so it's obvious they're historical) ---
    for lag in (1, 2, 3, 5):
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag - 1)

    # --- Moving averages ---
    # rolling(window) at row t uses rows [t-window+1, t] -- backward-looking. Safe.
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["price_vs_sma10"] = df["Close"] / df["sma_10"] - 1
    df["price_vs_sma50"] = df["Close"] / df["sma_50"] - 1

    # --- EMA / MACD ---
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- RSI (14-day) ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands ---
    rolling_std_20 = df["Close"].rolling(20).std()
    sma_20 = df["Close"].rolling(20).mean()
    df["bb_upper"] = sma_20 + 2 * rolling_std_20
    df["bb_lower"] = sma_20 - 2 * rolling_std_20
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # --- Volume features ---
    df["volume_sma_10"] = df["Volume"].rolling(10).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma_10"]

    # --- Calendar features (no leakage risk -- these are just the date) ---
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    return df


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Adds the prediction target: did the price go UP over the next `horizon` days?

    This is the ONE place we look forward: target[t] = 1 if Close[t+horizon] > Close[t].
    shift(-horizon) pulls a FUTURE value back to row t -- that's intentional here,
    but it's exactly the mistake to avoid in add_technical_indicators() above.
    """
    df = df.copy()
    future_close = df["Close"].shift(-horizon)
    df["target"] = (future_close > df["Close"]).astype(int)
    # The last `horizon` rows have no future price to compare against -- their
    # target is meaningless (NaN before casting), so they must be dropped, not
    # imputed. Imputing here would inject fake labels into your training set.
    df.loc[df.index[-horizon:], "target"] = np.nan
    return df


def build_feature_dataset(ticker: str, horizon: int = 1) -> pd.DataFrame:
    df = load_raw(ticker)
    df = add_technical_indicators(df)
    df = add_target(df, horizon=horizon)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows with NaNs (warm-up period for rolling windows + final target rows).")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{ticker.upper()}_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} feature rows to {out_path}")
    return df


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "DEMO"
    build_feature_dataset(ticker)
