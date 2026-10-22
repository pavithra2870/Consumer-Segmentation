from fastapi import APIRouter, BackgroundTasks, HTTPException

from config import settings, current_dataset
from services.datasets import get_spec, prepare_features
from services.segmentation import run_segmentation
from services.prediction_engine import train_all_models
from utils.logger import logger

router = APIRouter(prefix="/train-models", tags=["Training"])

# One job state per dataset
_job_states: dict[str, dict] = {}
_IDLE = {"status": "idle", "message": "No training job running"}


def _state() -> dict:
    return _job_states.setdefault(current_dataset.get(), dict(_IDLE))


@router.post("")
async def trigger_training(background_tasks: BackgroundTasks, n_clusters: int = None, method: str = "kmeans"):
    state = _state()
    if state["status"] == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")

    k = n_clusters or settings.n_clusters
    state.clear()
    state.update({"status": "running", "message": "Feature engineering started..."})
    background_tasks.add_task(_run_training, current_dataset.get(), k, method)
    return {"status": "running", "message": f"Training pipeline started ({method}, {k} clusters)"}


async def _run_training(dataset_id: str, n_clusters: int, method: str):
    current_dataset.set(dataset_id)
    state = _job_states[dataset_id]
    try:
        spec = get_spec()

        state["message"] = "Loading dataset and running feature engineering..."
        df_enriched, X_scaled = prepare_features(spec)

        state["message"] = f"Running {method} segmentation with {n_clusters} clusters..."
        seg_result = run_segmentation(X_scaled, df_enriched, n_clusters, method)
        logger.info(f"Segmentation complete — silhouette={seg_result['silhouette_score']}")

        state["message"] = "Training prediction models..."
        import pandas as pd
        df_seg = pd.read_csv(settings.segmented_path)
        pred_result = train_all_models(df_seg)

        state.update({
            "status": "complete",
            "message": "All models trained successfully",
            "segmentation": {
                "n_clusters": seg_result["n_clusters"],
                "silhouette_score": seg_result["silhouette_score"],
            },
            "predictions": {
                task: pred_result[task].get("auc", pred_result[task].get("r2_score"))
                for task in spec.tasks
            },
        })
    except FileNotFoundError as e:
        logger.error(f"Training failed — data not found: {e}")
        state.update({"status": "failed", "message": f"Data not found: {e}"})
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        state.update({"status": "failed", "message": str(e)})


@router.get("/status")
async def get_training_status():
    return _state()
