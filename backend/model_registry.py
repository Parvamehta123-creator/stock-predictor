"""
Loads trained models and their metadata for serving.

Deliberately separate from main.py's route definitions, for two reasons:
1. This logic (file loading, caching, error handling) can be unit-tested
   without FastAPI running at all -- it's pure Python.
2. main.py stays a thin routing layer that delegates here, rather than
   reimplementing "how do we find and load a model" inline in a route
   handler where it's harder to test in isolation.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# In-memory cache so repeated requests for the same ticker don't re-read the
# joblib file from disk every time. Cleared explicitly (see clear_cache) --
# there is no TTL, because a model file only changes when you retrain it,
# which is a deliberate action, not something that happens on a timer.
_model_cache: dict[str, tuple] = {}


def available_tickers() -> list[str]:
    """Tickers with a trained model on disk, discovered from metadata files
    rather than a hardcoded list -- so training a new ticker makes it
    servable automatically, with no code change here."""
    return sorted(p.stem.replace("_boosted_metadata", "") for p in MODELS_DIR.glob("*_boosted_metadata.json"))


def load_model_and_metadata(ticker: str) -> tuple:
    ticker = ticker.upper()
    if ticker in _model_cache:
        return _model_cache[ticker]

    model_path = MODELS_DIR / f"{ticker}_boosted.joblib"
    metadata_path = MODELS_DIR / f"{ticker}_boosted_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"No trained model for {ticker}. Run: python models/train_final_model.py {ticker}"
        )

    model = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    _model_cache[ticker] = (model, metadata)
    return model, metadata


def get_latest_feature_row(ticker: str, feature_columns: list[str]) -> tuple[pd.Series, pd.Timestamp]:
    """
    Returns the most recent engineered-feature row for a ticker, ready to
    feed to model.predict(). Raises rather than silently substituting a
    default if the feature file or a required column is missing -- a wrong
    silent value here would produce a plausible-looking but meaningless
    prediction, which is worse than an obvious error.
    """
    ticker = ticker.upper()
    path = PROCESSED_DIR / f"{ticker}_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No feature data for {ticker}. Run: python pipeline/features.py {ticker}"
        )

    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Feature file for {ticker} is missing expected columns: {missing}")

    latest = df.iloc[-1]
    return latest[feature_columns], latest["Date"]


def clear_cache() -> None:
    """Used by tests, and available for a future /reload endpoint if a
    model gets retrained while the server is already running."""
    _model_cache.clear()
