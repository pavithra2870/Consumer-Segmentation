import json

from fastapi import APIRouter, BackgroundTasks, HTTPException

from config import settings
from services.benchmark_datasets import LOADERS
from services.benchmark_runner import results_dir, run_dataset
from utils.logger import logger

router = APIRouter(prefix="/benchmarks", tags=["Benchmarks"])

LABELS = {
    "synthetic": "Synthetic Consumers (ours)",
    "telco": "Telco Customer Churn (IBM)",
    "bank": "Bank Marketing (UCI)",
    "retail": "Online Retail II (UCI)",
}

_job_state: dict = {"status": "idle", "message": "No benchmark running"}


@router.get("/datasets")
async def list_datasets():
    out = []
    for ds_id, (_, files) in LOADERS.items():
        has_files = all((settings.data_path / f).exists() for f in files)
        res_file = results_dir() / f"{ds_id}.json"
        out.append({
            "id": ds_id,
            "label": LABELS[ds_id],
            "files_present": has_files,
            "has_results": res_file.exists(),
        })
    return out


@router.post("/run")
async def run_benchmarks(background_tasks: BackgroundTasks, dataset: str = "all"):
    if _job_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Benchmarks already running")
    ids = list(LOADERS) if dataset == "all" else [dataset]
    unknown = [d for d in ids if d not in LOADERS]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {unknown[0]}")
    _job_state.update({"status": "running", "message": f"Starting {', '.join(ids)}..."})
    background_tasks.add_task(_run, ids)
    return {"status": "running", "datasets": ids}


def _run(ids: list[str]):
    try:
        for ds_id in ids:
            _job_state["message"] = f"Benchmarking {ds_id}..."
            run_dataset(ds_id)
        _job_state.update({"status": "complete", "message": "Benchmarks complete"})
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        _job_state.update({"status": "failed", "message": str(e)})


@router.get("/status")
async def benchmark_status():
    return _job_state


@router.get("/{dataset_id}")
async def get_benchmark(dataset_id: str):
    if dataset_id not in LOADERS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset_id}")
    path = results_dir() / f"{dataset_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No results yet. POST /benchmarks/run first.")
    return json.loads(path.read_text())
