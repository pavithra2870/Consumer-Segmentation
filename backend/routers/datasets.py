from fastapi import APIRouter

from services.datasets import REGISTRY, dataset_status

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("")
async def list_datasets():
    """Datasets the dashboard can switch between, with their capabilities and readiness."""
    return [dataset_status(spec) for spec in REGISTRY.values()]
