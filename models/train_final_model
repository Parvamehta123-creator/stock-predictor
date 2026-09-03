"""
Trains the final model on the FULL available dataset and saves it alongside
its walk-forward metrics, so anything serving this model later (the FastAPI
backend, in the next phase) has an honest performance estimate attached.

Ordering matters here: walk-forward evaluation happens BEFORE the final fit,
on folds that never touch the full dataset. The final model -- fit on
everything, to use all available signal for live predictions -- is never
itself evaluated. If you evaluated the final model on the data it was fit on,
you'd get a number that measures memorization, not generalization, and it
would look great for entirely the wrong reason.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.modeling import TARGET_COLUMN, get_boosted_model, load_dataset, walk_forward_evaluate

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_and_save(ticker: str, model_fn=get_boosted_model, model_name: str = "boosted") -> tuple:
    df, feature_cols = load_dataset(ticker)

    wf_results = walk_forward_evaluate(df, feature_cols, model_fn)

    X = df[feature_cols].values
    y = df[TARGET_COLUMN].astype(int).values
    final_model = model_fn()
    final_model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{ticker.upper()}_{model_name}.joblib"
    joblib.dump(final_model, model_path)

    metadata = {
        "ticker": ticker.upper(),
        "model_name": model_name,
        "feature_columns": feature_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(df),
        "walk_forward_mean_accuracy": float(wf_results["accuracy"].mean()),
        "walk_forward_mean_majority_baseline": float(wf_results["majority_baseline"].mean()),
        "walk_forward_folds": wf_results.to_dict(orient="records"),
    }
    metadata_path = MODELS_DIR / f"{ticker.upper()}_{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")
    print(
        f"Walk-forward mean accuracy: {metadata['walk_forward_mean_accuracy']:.4f} "
        f"(majority baseline: {metadata['walk_forward_mean_majority_baseline']:.4f})"
    )
    return final_model, metadata


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "DEMO"
    train_and_save(ticker)