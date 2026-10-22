from fastapi import APIRouter, HTTPException
import pandas as pd

from config import settings
from services.datasets import get_spec, segment_cards
from services.persona_generator import generate_personas

router = APIRouter(prefix="/personas", tags=["Personas"])


@router.get("")
async def get_personas():
    """Generate and return persona profiles for all segments."""
    seg_path = settings.segmented_path
    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="Segmentation not run. Open Data Pipeline and train this dataset first.")

    df = pd.read_csv(seg_path)
    personas = await generate_personas({"segments": segment_cards(df)}, df)
    return {"meta": get_spec().meta(), "personas": personas}
