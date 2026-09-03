"""
Integration tests for backend/main.py using FastAPI's TestClient.

Unlike test_model_registry.py, these actually need fastapi (and httpx,
which TestClient uses under the hood) installed -- run with:
    pytest tests/test_backend_api.py -v
after `pip install -r requirements.txt` and after training at least the
DEMO model: `python models/train_final_model.py DEMO`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models_includes_demo():
    response = client.get("/models")
    assert response.status_code == 200
    tickers = [m["ticker"] for m in response.json()]
    assert "DEMO" in tickers


def test_predict_demo_returns_honest_context():
    """The response must carry BOTH the prediction and the accuracy context
    needed to judge whether to trust it -- never just a bare direction."""
    response = client.post("/predict/DEMO")
    assert response.status_code == 200
    body = response.json()
    assert body["direction"] in ("up", "down")
    assert "walk_forward_accuracy" in body
    assert "majority_baseline" in body


def test_predict_unknown_ticker_returns_404_with_runnable_fix():
    response = client.post("/predict/NOTATICKER")
    assert response.status_code == 404
    assert "train_final_model" in response.json()["detail"]
