"""
Generates synthetic OHLCV data with the EXACT same schema fetch_data.py produces.

Why this exists: this sandbox has no internet access, so we can't call yfinance
here. Real projects hit this problem too -- rate limits, API keys not yet set up,
CI environments with no network -- and the standard fix is the same one we're
using: a fake data source that satisfies the same contract as the real one, so
you can develop and test everything downstream before the real dependency is
available.

The price series follows geometric Brownian motion (the same model behind
Black-Scholes) -- it's not a realistic trading signal, but it has the right
*statistical shape* (random walk with drift and volatility clustering) to
exercise the feature engineering and modeling code honestly.

Usage:
    python pipeline/generate_sample_data.py DEMO --days 750
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def generate_ohlcv(ticker: str, days: int = 750, start_price: float = 150.0, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)  # business days only

    # Geometric Brownian motion for the daily closes: drift + volatility shock,
    # plus a slow-moving regime shift in volatility so the series isn't perfectly stationary
    # (real markets have calm periods and turbulent periods -- this fakes that).
    daily_drift = 0.0003
    base_vol = 0.015
    vol_regime = base_vol * (1 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, days)))
    shocks = rng.normal(loc=daily_drift, scale=vol_regime)
    log_returns = shocks
    close = start_price * np.exp(np.cumsum(log_returns))

    # Derive open/high/low from close with small intraday noise
    intraday_noise = rng.normal(scale=base_vol / 2, size=days)
    open_ = close * (1 + intraday_noise)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(scale=base_vol / 3, size=days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(scale=base_vol / 3, size=days)))

    # Volume: baseline with noise, spiking on big price moves (mimics real volume behavior)
    base_volume = 5_000_000
    move_size = np.abs(log_returns)
    volume = base_volume * (1 + 5 * move_size) * rng.lognormal(mean=0, sigma=0.2, size=days)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_.round(2),
        "High": high.round(2),
        "Low": low.round(2),
        "Close": close.round(2),
        "Volume": volume.astype(int),
    })
    return df


def save_raw(df: pd.DataFrame, ticker: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ticker.upper()}.csv"
    df.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OHLCV data for development/testing.")
    parser.add_argument("ticker", nargs="?", default="DEMO", help="Fake ticker name (default: DEMO)")
    parser.add_argument("--days", type=int, default=750, help="Number of trading days to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    df = generate_ohlcv(args.ticker, days=args.days, seed=args.seed)
    path = save_raw(df, args.ticker)
    print(f"Generated {len(df)} synthetic rows for {args.ticker.upper()} -> {path}")
    print(df.head())
    print("...")
    print(df.tail())


if __name__ == "__main__":
    main()
