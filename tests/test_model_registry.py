"""
Tests for backend/model_registry.py.

These test the loading/caching/error-handling logic WITHOUT needing FastAPI
running at all -- the registry is deliberately independent of the web
framework so it can be verified this way.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.model_registry import (
    available_tickers,
    clear_cache,
    get_latest_feature_row,
    load_model_and_metadata,
)


def test_available_tickers_includes_demo():
    """Assumes DEMO has been trained (models/train_final_model.py DEMO).
    If this fails, train that model first -- it's not this test's job to."""
    assert "DEMO" in available_tickers()


def test_load_model_and_metadata_missing_ticker_raises_with_helpful_message():
    clear_cache()
    with pytest.raises(FileNotFoundError, match="train_final_model"):
        load_model_and_metadata("NOTATICKER")


def test_load_model_and_metadata_uses_cache_on_second_call():
    """The second call should return the exact same object, not a freshly
    reloaded one -- proves the cache is actually being hit, not just present."""
    clear_cache()
    model_a, meta_a = load_model_and_metadata("DEMO")
    model_b, meta_b = load_model_and_metadata("DEMO")
    assert model_a is model_b
    assert meta_a is meta_b


def test_get_latest_feature_row_missing_ticker_raises():
    with pytest.raises(FileNotFoundError, match="pipeline/features.py"):
        get_latest_feature_row("NOTATICKER", ["return_1d"])


def test_get_latest_feature_row_missing_column_raises_value_error():
    """If the metadata says a model expects a column that isn't in the
    feature file (e.g. sentiment columns for a ticker that has no notes),
    this must fail loudly -- not silently drop the column or fill zeros,
    either of which would feed the model different inputs than it was
    trained on."""
    with pytest.raises(ValueError, match="missing expected columns"):
        get_latest_feature_row("DEMO", ["return_1d", "this_column_does_not_exist"])


def test_get_latest_feature_row_returns_most_recent_date():
    _, metadata = load_model_and_metadata("DEMO")
    row, date = get_latest_feature_row("DEMO", metadata["feature_columns"])
    assert len(row) == len(metadata["feature_columns"])
    assert date is not None
