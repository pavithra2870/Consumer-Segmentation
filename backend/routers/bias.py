from fastapi import APIRouter, HTTPException
import pandas as pd

from config import settings
from services.bias_fairness import generate_bias_report
from services.datasets import get_spec

router = APIRouter(prefix="/bias-report", tags=["Fairness"])


@router.get("")
async def get_bias_report():
    """Return demographic fairness analysis across model predictions."""
    if not get_spec().protected:
        raise HTTPException(status_code=404, detail="This dataset has no demographic attributes, so a fairness audit does not apply.")
    seg_path = settings.segmented_path
    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="Models not trained. Open Data Pipeline and train this dataset first.")

    df = pd.read_csv(seg_path)
    return generate_bias_report(df)
