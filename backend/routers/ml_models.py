from fastapi import APIRouter, BackgroundTasks, HTTPException

from config import settings
from services.data_generator import load_dataset
from services.feature_engineering import engineer_features
from services.segmentation import run_segmentation
from services.prediction_engine import train_all_models
from utils.logger import logger

router = APIRouter(prefix="/train-models", tags=["Training"])

_job_state: dict = {"status": "idle", "message": "No training job running"}


@router.post("")
async def trigger_training(background_tasks: BackgroundTasks, n_clusters: int = None, method: str = "kmeans"):
    if _job_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")

    k = n_clusters or settings.n_clusters
    _job_state.update({"status": "running", "message": "Feature engineering started..."})
    background_tasks.add_task(_run_training, k, method)
    return {"status": "running", "message": f"Training pipeline started ({method}, {k} clusters)"}


async def _run_training(n_clusters: int, method: str):
    global _job_state
    try:
        _job_state["message"] = "Loading dataset..."
        df_raw = load_dataset()

        _job_state["message"] = "Running feature engineering..."
        df_enriched, X_scaled, scaler, encoders = engineer_features(df_raw)

        _job_state["message"] = f"Running {method} segmentation with {n_clusters} clusters..."
        seg_result = run_segmentation(X_scaled, df_enriched, n_clusters, method)
        logger.info(f"Segmentation complete — silhouette={seg_result['silhouette_score']}")

        _job_state["message"] = "Training prediction models..."
        import pandas as pd
        df_seg = pd.read_csv(settings.data_path / "consumers_segmented.csv")
        pred_result = train_all_models(df_seg)

        _job_state.update({
            "status": "complete",
            "message": "All models trained successfully",
            "segmentation": {
                "n_clusters": seg_result["n_clusters"],
                "silhouette_score": seg_result["silhouette_score"],
            },
            "predictions": {
                "churn_accuracy": pred_result["churn"].get("accuracy"),
                "ltv_r2": pred_result["ltv"].get("r2_score"),
                "conversion_accuracy": pred_result["conversion"].get("accuracy"),
            },
        })
    except FileNotFoundError as e:
        logger.error(f"Training failed — data not found: {e}")
        _job_state.update({"status": "failed", "message": f"Data not found: {e}"})
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        _job_state.update({"status": "failed", "message": str(e)})


@router.get("/status")
async def get_training_status():
    return _job_state
