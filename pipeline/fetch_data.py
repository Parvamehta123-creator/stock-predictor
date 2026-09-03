"""
Fetches historical daily OHLCV data for a ticker and saves it to data/raw/.

This is the ONLY file in the whole project that talks to an external API.
Every other module reads from data/raw/{ticker}.csv and doesn't care whether
that file came from yfinance, a different vendor, or a synthetic generator.
That separation is deliberate: it means you can swap data sources later
without touching feature engineering, modeling, or serving code.

Usage (run locally, where you have internet access):
    python pipeline/fetch_data.py AAPL --start 2018-01-01
"""
import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_ticker(ticker: str, start: str = "2018-01-01", end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data for `ticker` and return a clean DataFrame."""
    import yfinance as yf  # imported lazily so this file can be read/tested without the package

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and date range.")

    # yfinance can return MultiIndex columns when auto_adjust/group_by settings vary
    # across versions -- flatten defensively so downstream code always sees a flat schema.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def save_raw(df: pd.DataFrame, ticker: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ticker.upper()}.csv"
    df.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Fetch historical stock data.")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    df = fetch_ticker(args.ticker, start=args.start, end=args.end)
    path = save_raw(df, args.ticker)
    print(f"Saved {len(df)} rows for {args.ticker.upper()} to {path}")


if __name__ == "__main__":
    main()
