"""
Merges per-ticker sentiment features (from commentary notes) into the
price/technical feature dataset built by features.py.

TIMING ASSUMPTION -- read this before using in production:
A note dated 2026-08-30 is assumed to be available and readable BEFORE
that trading day's target is determined (i.e. before the 2026-08-30 ->
2026-08-31 price move it might inform). If your real notes are published
AFTER market close, they describe information the market already
absorbed for that same day -- using them as a same-day feature would be
leakage, and the fix is to shift the note's effective date forward by
one trading day. This module does NOT shift same-day-vs-next-day for you, because the right
answer depends on the note's actual publish time, which you know and we
don't. Decide this explicitly; don't let it default silently.

It DOES roll a note's date forward to the next actual trading day (see
align_to_trading_days below) -- a note dated on a weekend or holiday has
to land somewhere, and "the next day the market is open" is the only
non-arbitrary choice.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.notes_loader import load_notes_folder
from pipeline.sentiment import process_note

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def align_to_trading_days(dates: pd.Series, trading_days: pd.Series) -> pd.Series:
    """
    Snaps each date to the next date that actually exists in `trading_days`.

    Without this, a note dated Sunday (this project's sample note is dated
    2026-08-30, a Sunday) matches NOTHING in a business-day price series --
    a plain merge on Date silently drops it with no error, no warning, and
    a merged dataframe that looks completely normal. That's the trap: the
    bug produces zero rows changed, zero exceptions, and a plausible-looking
    output. Found this exact way, by noticing the sentiment columns were
    suspiciously all-zero after a merge that "worked."

    Dates after the last available trading day map to NaT and get dropped
    by the caller -- there's no valid trading day to attribute them to yet.
    """
    sorted_days = np.sort(trading_days.dropna().unique())
    positions = np.searchsorted(sorted_days, dates.values.astype("datetime64[ns]"))
    in_range = positions < len(sorted_days)
    safe_positions = np.clip(positions, 0, len(sorted_days) - 1)
    aligned = np.where(in_range, sorted_days[safe_positions], np.datetime64("NaT"))
    return pd.Series(aligned, index=dates.index)


def build_sentiment_log(notes_folder=None) -> pd.DataFrame:
    """One row per (date, ticker) mention, plus that date's macro sentiment."""
    notes = load_notes_folder(notes_folder) if notes_folder else load_notes_folder()
    rows = []
    for note in notes:
        result = process_note(note)
        for m in result["mentions"]:
            rows.append({
                "Date": pd.Timestamp(result["date"]),
                "ticker": m["ticker"],
                "mention_sentiment": m["local_sentiment"],
                "macro_sentiment": result["macro_sentiment"],
            })
    return pd.DataFrame(rows)


def merge_with_features(feature_df: pd.DataFrame, sentiment_log: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Left-joins sentiment features onto feature_df for one ticker.

    Fill strategy (explicit, not automatic):
    - mentioned / mention_sentiment: 0 on days with no note mentioning the
      ticker. A missing mention is real information ("nobody was talking
      about this stock today"), not a gap to interpolate.
    - macro_sentiment: forward-filled. The market-wide view expressed in
      Tuesday's note is a reasonable stand-in for Wednesday's if no new
      note exists yet -- unlike a stock-specific mention, macro tone
      doesn't reset to "unknown" just because nobody wrote a fresh note.
    """
    df = feature_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    sentiment_log = sentiment_log.copy()
    sentiment_log["Date"] = align_to_trading_days(sentiment_log["Date"], df["Date"])
    sentiment_log = sentiment_log.dropna(subset=["Date"])

    ticker_sentiment = sentiment_log[sentiment_log["ticker"] == ticker][["Date", "mention_sentiment"]]
    # keep="last": if two notes both roll forward onto the same trading day
    # (e.g. Saturday and Sunday notes both landing on Monday), the later
    # original note wins. Reasonable default; revisit if that ever matters.
    macro = sentiment_log[["Date", "macro_sentiment"]].drop_duplicates(subset="Date", keep="last")

    df = df.merge(ticker_sentiment, on="Date", how="left")
    df = df.merge(macro, on="Date", how="left")

    df["mentioned"] = df["mention_sentiment"].notna().astype(int)
    df["mention_sentiment"] = df["mention_sentiment"].fillna(0.0)
    df["macro_sentiment"] = df["macro_sentiment"].ffill().fillna(0.0)

    return df


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "CYIENT"

    sentiment_log = build_sentiment_log()
    print("Sentiment log:")
    print(sentiment_log.to_string(index=False))

    feature_path = PROCESSED_DIR / f"{ticker.upper()}_features.csv"
    if not feature_path.exists():
        print(f"\nNo feature file at {feature_path} -- run generate_sample_data.py and features.py for {ticker} first.")
        sys.exit(0)

    feature_df = pd.read_csv(feature_path)
    merged = merge_with_features(feature_df, sentiment_log, ticker.upper())

    out_path = PROCESSED_DIR / f"{ticker.upper()}_features_with_sentiment.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nMerged dataset saved to {out_path}")
    print(merged[["Date", "Close", "mentioned", "mention_sentiment", "macro_sentiment"]].tail(10).to_string(index=False))
