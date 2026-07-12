from fastapi import APIRouter, HTTPException
import pandas as pd

from config import settings
from services.persona_generator import generate_personas
from utils.logger import logger

router = APIRouter(prefix="/personas", tags=["Personas"])


@router.get("")
async def get_personas():
    """Generate and return persona profiles for all segments."""
    seg_path = settings.data_path / "consumers_segmented.csv"
    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="Segmentation not run. Call POST /train-models first.")

    df = pd.read_csv(seg_path)
    import joblib
    from services.segmentation import _label_for_cluster

    n_clusters = int(df["segment"].nunique())
    labels = df["segment"].values
    segments = []
    for sid in range(n_clusters):
        mask = labels == sid
        subset = df[mask]
        segments.append({
            "segment_id": sid,
            "label": _label_for_cluster(sid, subset.mean(numeric_only=True)),
            "size": int(mask.sum()),
            "percentage": round(float(mask.sum()) / len(labels) * 100, 2),
            "avg_ltv": round(float(subset["ltv_actual"].mean()), 2) if "ltv_actual" in subset else 0,
            "avg_churn_risk": round(float(subset["churn_probability"].mean()), 4) if "churn_probability" in subset else 0,
            "top_features": {},
        })

    segmentation_result = {"segments": segments}
    personas = await generate_personas(segmentation_result, df)
    return {"personas": personas}
