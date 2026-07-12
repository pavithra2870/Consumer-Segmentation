from fastapi import APIRouter, HTTPException
import pandas as pd

from config import settings
from services.bias_fairness import generate_bias_report

router = APIRouter(prefix="/bias-report", tags=["Fairness"])


@router.get("")
async def get_bias_report():
    """Return demographic fairness analysis across model predictions."""
    seg_path = settings.data_path / "consumers_segmented.csv"
    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="Models not trained. Call POST /train-models first.")

    df = pd.read_csv(seg_path)
    report = generate_bias_report(df)
    return report
