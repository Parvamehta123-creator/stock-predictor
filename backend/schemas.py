"""
Pydantic response models for the API.

PredictionResponse deliberately always includes walk_forward_accuracy and
majority_baseline alongside the prediction itself. A bare {"direction": "up"}
response looks more confident than the underlying model deserves -- forcing
every consumer of this API (the Streamlit dashboard, a curl request, anyone
reading /docs) to see the honest accuracy context in the same payload is a
design choice, not an afterthought.
"""
from typing import Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    ticker: str
    as_of_date: str = Field(description="Date of the most recent feature row used to make this prediction")
    prediction: int = Field(description="1 = predicted up, 0 = predicted down, for the next trading day")
    direction: str
    confidence: Optional[float] = Field(default=None, description="Model's predicted probability of the predicted class, if the model supports predict_proba")
    walk_forward_accuracy: float = Field(description="Honest out-of-sample accuracy from walk-forward validation -- this is NOT the same as this single prediction's confidence")
    majority_baseline: float = Field(description="Accuracy of always predicting the majority class. Compare walk_forward_accuracy against THIS, not against 100%, to judge whether the model has real edge")


class ModelInfo(BaseModel):
    ticker: str
    trained_at: str
    training_rows: int
    walk_forward_accuracy: float
    majority_baseline: float
