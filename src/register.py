import time
import mlflow
from mlflow.tracking import MlflowClient

# ── 1. Connect to MLflow ──────────────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# ── 2. Get all runs from the experiment ───────────────────────────────────────
experiment = client.get_experiment_by_name("wine-quality-classification")
if experiment is None:
    raise RuntimeError("Experiment 'wine-quality-classification' not found. Run train.py first.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"],
)

if not runs:
    raise RuntimeError("No runs found in the experiment.")

# ── 3. Find the run with the highest f1_score ─────────────────────────────────
best_run = runs[0]
best_run_id = best_run.info.run_id
model_type = best_run.data.tags.get("model_type", "unknown")
f1 = best_run.data.metrics["f1_score"]

# ── 4. Print best model info ──────────────────────────────────────────────────
print(f"Best model: {model_type} | Run ID: {best_run_id} | F1: {f1:.4f}")

# ── 5. Register the model ─────────────────────────────────────────────────────
model_name = "wine-quality-classifier"
artifact_uri = f"runs:/{best_run_id}/model"

print(f"\nRegistering model '{model_name}' from artifact URI: {artifact_uri}")
mv = mlflow.register_model(model_uri=artifact_uri, name=model_name)

# Poll until registration leaves PENDING_REGISTRATION state
timeout = 60
start = time.time()
while True:
    mv = client.get_model_version(name=model_name, version=mv.version)
    if mv.status not in ("PENDING_REGISTRATION", "REGISTRY_PENDING_REGISTRATION"):
        break
    if time.time() - start > timeout:
        raise TimeoutError("Model registration timed out.")
    time.sleep(1)

if mv.status != "READY":
    raise RuntimeError(f"Registration failed with status: {mv.status}")

print(f"Registration complete  ->  version={mv.version}  status={mv.status}")

# ── 6. Transition to Production ───────────────────────────────────────────────
client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Production",
    archive_existing_versions=True,
)

# ── 7. Confirm ────────────────────────────────────────────────────────────────
mv = client.get_model_version(name=model_name, version=mv.version)
print(f"Model v{mv.version} promoted to Production")
