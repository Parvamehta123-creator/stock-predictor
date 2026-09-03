"""
Tests for pipeline/modeling.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.modeling import _fold_metrics


def test_walk_forward_splits_never_train_on_future():
    """Every training index in a TimeSeriesSplit fold must be smaller (earlier)
    than every test index in that same fold -- this is the property that
    makes walk-forward validation valid at all."""
    X = np.arange(200).reshape(-1, 1)
    for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
        assert train_idx.max() < test_idx.min(), "Found a fold where training data is not strictly before test data"


def test_majority_baseline_matches_class_balance():
    y_true = pd.Series([1, 1, 1, 0])  # 75% class 1
    y_pred = pd.Series([1, 1, 1, 1])  # predicts all 1s -> 75% accuracy
    metrics = _fold_metrics(y_true, y_pred)
    assert metrics["majority_baseline"] == 0.75
    assert metrics["accuracy"] == 0.75


def test_majority_baseline_uses_larger_class():
    """If the minority class were used instead, this would silently
    understate how hard the baseline is to beat."""
    y_true = pd.Series([0, 0, 0, 0, 1])  # 80% class 0
    y_pred = pd.Series([0, 0, 0, 0, 0])
    metrics = _fold_metrics(y_true, y_pred)
    assert metrics["majority_baseline"] == 0.8
