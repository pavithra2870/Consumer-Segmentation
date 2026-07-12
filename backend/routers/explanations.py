from fastapi import APIRouter, HTTPException
import pandas as pd

from config import settings
from services.explainability import compute_explanations
from utils.logger import logger

router = APIRouter(prefix="/explanations", tags=["Explainability"])


@router.get("")
async def get_explanations():
    """Return SHAP-based feature importance and segment reasoning for all models."""
    seg_path = settings.data_path / "consumers_segmented.csv"
    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="Models not trained. Call POST /train-models first.")

    df = pd.read_csv(seg_path)
    explanations = compute_explanations(df)
    return explanations
