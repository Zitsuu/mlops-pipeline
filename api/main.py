# Sample curl command for /predict:
#
# curl -X POST "http://localhost:8000/predict" \
#      -H "Content-Type: application/json" \
#      -d '{
#            "fixed_acidity": 7.4,
#            "volatile_acidity": 0.70,
#            "citric_acid": 0.00,
#            "residual_sugar": 1.9,
#            "chlorides": 0.076,
#            "free_sulfur_dioxide": 11.0,
#            "total_sulfur_dioxide": 34.0,
#            "density": 0.9978,
#            "pH": 3.51,
#            "sulphates": 0.56,
#            "alcohol": 9.4
#          }'

import os
import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Column order must match training features exactly ─────────────────────────
FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]

# ── Shared state ──────────────────────────────────────────────────────────────
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler once at startup; release on shutdown."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    try:
        state["model"] = mlflow.pyfunc.load_model("models:/wine-quality-classifier/Production")
    except Exception as exc:
        raise RuntimeError(f"Failed to load MLflow model: {exc}") from exc

    try:
        state["scaler"] = joblib.load("data/scaler.pkl")
    except Exception as exc:
        raise RuntimeError(f"Failed to load scaler: {exc}") from exc

    yield

    state.clear()


app = FastAPI(
    title="Wine Quality Classifier",
    description="Predicts whether a red wine is good (1) or not good (0).",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Input schema ──────────────────────────────────────────────────────────────
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": "wine-quality-classifier", "stage": "Production"}


@app.post("/predict")
def predict(features: WineFeatures):
    if "model" not in state or "scaler" not in state:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")

    # Map underscore field names back to original space-separated column names
    raw = {
        "fixed acidity":         features.fixed_acidity,
        "volatile acidity":      features.volatile_acidity,
        "citric acid":           features.citric_acid,
        "residual sugar":        features.residual_sugar,
        "chlorides":             features.chlorides,
        "free sulfur dioxide":   features.free_sulfur_dioxide,
        "total sulfur dioxide":  features.total_sulfur_dioxide,
        "density":               features.density,
        "pH":                    features.pH,
        "sulphates":             features.sulphates,
        "alcohol":               features.alcohol,
    }

    df = pd.DataFrame([raw], columns=FEATURE_COLUMNS)
    scaled = state["scaler"].transform(df)
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS)

    prediction = int(state["model"].predict(scaled_df)[0])

    # Retrieve probability for the predicted class
    try:
        proba = state["model"].predict_proba(scaled_df)[0]
        confidence = float(proba[prediction])
    except AttributeError:
        # pyfunc wrapper may not expose predict_proba; fall back gracefully
        confidence = 1.0

    return {
        "prediction": prediction,
        "label": "good" if prediction == 1 else "not good",
        "confidence": round(confidence, 4),
    }
