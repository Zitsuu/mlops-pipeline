import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/winequality-red.csv", sep=";")

# ── 2. Binary target ──────────────────────────────────────────────────────────
df["target"] = (df["quality"] >= 6).astype(int)

# ── 3. Features / target split ────────────────────────────────────────────────
X = df.drop(columns=["quality", "target"])
y = df["target"]

# ── 4. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── 5. StandardScaler ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ── 6. MLflow setup ───────────────────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-quality-classification")


def log_metrics(y_true, y_pred, y_prob):
    """Compute and return the five evaluation metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_score":  f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall":    recall_score(y_true, y_pred, average="weighted"),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


results = []

# ── 7a. Logistic Regression ───────────────────────────────────────────────────
with mlflow.start_run(run_name="logistic_regression") as run:
    mlflow.set_tags({"model_type": "logistic_regression"})

    params = {"C": 1.0, "max_iter": 1000}
    mlflow.log_params(params)

    model = LogisticRegression(**params, random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    metrics = log_metrics(y_test, y_pred, y_prob)
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(model, artifact_path="model")

    results.append({"run_id": run.info.run_id, "model_type": "logistic_regression", **metrics})
    print("[done] Logistic Regression run complete")

# ── 7b. Random Forest ─────────────────────────────────────────────────────────
with mlflow.start_run(run_name="random_forest") as run:
    mlflow.set_tags({"model_type": "random_forest"})

    params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    metrics = log_metrics(y_test, y_pred, y_prob)
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(model, artifact_path="model")

    results.append({"run_id": run.info.run_id, "model_type": "random_forest", **metrics})
    print("[done] Random Forest run complete")

# ── 7c. XGBoost ───────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="xgboost") as run:
    mlflow.set_tags({"model_type": "xgboost"})

    params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "random_state": 42}
    mlflow.log_params(params)

    model = XGBClassifier(**params, eval_metric="logloss", verbosity=0)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    metrics = log_metrics(y_test, y_pred, y_prob)
    mlflow.log_metrics(metrics)

    mlflow.xgboost.log_model(model, artifact_path="model")

    results.append({"run_id": run.info.run_id, "model_type": "xgboost", **metrics})
    print("[done] XGBoost run complete")

# ── 8. Summary table ──────────────────────────────────────────────────────────
summary = pd.DataFrame(results).sort_values("f1_score", ascending=False)
summary = summary.reset_index(drop=True)

metric_cols = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
summary[metric_cols] = summary[metric_cols].round(4)

print("\n" + "=" * 90)
print("EXPERIMENT SUMMARY  (sorted by f1_score desc)")
print("=" * 90)
print(summary[["run_id", "model_type"] + metric_cols].to_string(index=False))
print("=" * 90)

# ── 9. Save scaler ────────────────────────────────────────────────────────────
joblib.dump(scaler, "data/scaler.pkl")
print("\nScaler saved to data/scaler.pkl")
