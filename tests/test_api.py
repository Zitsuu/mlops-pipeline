import pytest
from fastapi.testclient import TestClient

# Run from project root so relative paths (mlflow.db, data/) resolve correctly
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.main import app

VALID_WINE = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.70,
    "citric_acid": 0.00,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────
def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_body(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "wine-quality-classifier"
    assert body["stage"] == "Production"


# ── /predict ──────────────────────────────────────────────────────────────────
def test_predict_valid_input(client):
    response = client.post("/predict", json=VALID_WINE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("good", "not good")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_label_matches_prediction(client):
    body = client.post("/predict", json=VALID_WINE).json()
    expected_label = "good" if body["prediction"] == 1 else "not good"
    assert body["label"] == expected_label


def test_predict_missing_fields_returns_422(client):
    response = client.post("/predict", json={"fixed_acidity": 7.4})
    assert response.status_code == 422


def test_predict_empty_body_returns_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_wrong_type_returns_422(client):
    bad = {**VALID_WINE, "alcohol": "not-a-float"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
