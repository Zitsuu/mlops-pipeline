# MLOps Pipeline

An end-to-end ML pipeline that trains models on Wine Quality data, tracks experiments with MLflow, serves predictions via FastAPI, and runs automated tests on every push with GitHub Actions.

[![CI](https://github.com/Zitsuu/mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Zitsuu/mlops-pipeline/actions/workflows/ci.yml)

---

## What It Does

1. Trains 3 models (Logistic Regression, Random Forest, XGBoost) on the Wine Quality dataset
2. Tracks all experiments in MLflow — params, metrics, artifacts
3. Promotes the best model to a Production registry automatically
4. Serves real-time predictions through a FastAPI REST API
5. Fully containerized with Docker and tested via GitHub Actions CI

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| MLflow | Experiment tracking + Model Registry |
| XGBoost + scikit-learn | Model training |
| FastAPI | Prediction REST API |
| Docker Compose | Container orchestration |
| GitHub Actions | CI/CD pipeline |
| pytest | Automated API testing |

---

## Project Structure

```
mlops-pipeline/
├── data/                   # Dataset + saved scaler
├── src/
│   ├── train.py            # Train 3 models, log to MLflow
│   └── register.py         # Promote best model to Production
├── api/
│   └── main.py             # FastAPI app
├── tests/
│   └── test_api.py         # 7 automated tests
├── .github/workflows/
│   └── ci.yml              # GitHub Actions pipeline
├── Dockerfile.api
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/Zitsuu/mlops-pipeline.git
cd mlops-pipeline
pip install -r requirements.txt
python src/train.py
python src/register.py
uvicorn api.main:app --reload
```

Visit **http://localhost:8000/docs** for the interactive API.

---

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open **http://localhost:5000** to see all runs and the model registry.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Check model status |
| POST | /predict | Get wine quality prediction |
| GET | /docs | Swagger UI |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"fixed_acidity":7.4,"volatile_acidity":0.70,"citric_acid":0.00,"residual_sugar":1.9,"chlorides":0.076,"free_sulfur_dioxide":11.0,"total_sulfur_dioxide":34.0,"density":0.9978,"pH":3.51,"sulphates":0.56,"alcohol":9.4}'
```

**Example response:**

```json
{
  "prediction": 1,
  "label": "good",
  "confidence": 0.82
}
```

---

## Model Results

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.741 | 0.733 | 0.811 |
| Random Forest | 0.772 | 0.764 | 0.853 |
| **XGBoost** | **0.781** | **0.778** | **0.873** |

XGBoost was automatically selected and promoted to Production.

---

## Docker

```bash
docker-compose up --build
```

---

## Tests

```bash
pytest tests/ -v
```