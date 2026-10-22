from fastapi import APIRouter, BackgroundTasks, HTTPException
from pathlib import Path

from config import settings, current_dataset
from services.data_generator import generate_dataset, load_dataset
from utils.logger import logger

router = APIRouter(prefix="/generate-data", tags=["Data"])

_job_state: dict = {"status": "idle", "message": "No job running"}


@router.post("")
async def trigger_data_generation(background_tasks: BackgroundTasks, n_rows: int = None):
    """Trigger async synthetic dataset generation."""
    if current_dataset.get() != "synthetic":
        raise HTTPException(status_code=400, detail="Only the synthetic dataset is generated. Other datasets are loaded from data/ during training.")
    if _job_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Data generation already in progress")

    n = n_rows or settings.dataset_size
    _job_state.update({"status": "running", "message": f"Generating {n:,} rows..."})
    background_tasks.add_task(_run_generation, n)
    return {"status": "running", "message": f"Generating {n:,} rows in background", "n_rows": n}


async def _run_generation(n: int):
    global _job_state
    try:
        df = await generate_dataset(n)
        _job_state.update({
            "status": "complete",
            "message": f"Dataset generated: {df.shape[0]:,} rows × {df.shape[1]} columns",
            "rows": df.shape[0],
            "columns": df.shape[1],
        })
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        _job_state.update({"status": "failed", "message": str(e)})


@router.get("/status")
async def get_generation_status():
    return _job_state


@router.get("/info")
async def get_dataset_info():
    """Return metadata about the existing dataset."""
    try:
        df = load_dataset()
        path = settings.table_path
        size_mb = round(path.stat().st_size / (1024 * 1024), 2)
        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "size_mb": size_mb,
            "features": df.columns.tolist(),
            "generated_at": str(path.stat().st_mtime),
            # Blank review_text parses as NaN, which is not valid JSON
            "sample": df.head(5).astype(object).where(df.head(5).notna(), None).to_dict(orient="records"),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
