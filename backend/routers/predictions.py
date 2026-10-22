from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
import json
import numpy as np
import pandas as pd
import joblib

from config import settings
from services.datasets import get_spec
from services.prediction_engine import sample_predictions
from utils.logger import logger

router = APIRouter(prefix="/predictions", tags=["Predictions"])


def _read_json(name: str) -> dict:
    path = settings.models_path / name
    return json.loads(path.read_text()) if path.exists() else {}


@router.get("")
async def get_predictions():
    """Return prediction model results and sample predictions for the active dataset."""
    seg_path = settings.segmented_path
    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="Models not trained. Open Data Pipeline and train this dataset first.")

    spec = get_spec()
    df = pd.read_csv(seg_path)
    models_path = settings.models_path
    saved_metrics = _read_json("metrics.json")
    targets = {"churn": "is_churned", "conversion": "converted", "ltv": "ltv_actual"}

    result = {"meta": spec.meta()}
    for name in spec.tasks:
        try:
            model = joblib.load(models_path / f"{name}_model.pkl")
            features = joblib.load(models_path / f"{name}_features.pkl")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Model not found: {name}")

        avail = [f for f in features if f in df.columns]
        X = df[avail].fillna(0)
        fi = dict(zip(avail, model.feature_importances_.tolist()))
        top_fi = {k: round(v, 5) for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]}

        if name == "ltv":
            y_log = np.log1p(df[targets[name]])
            pred_log = model.predict(X)
            from sklearn.metrics import r2_score
            hist, edges = np.histogram(np.expm1(pred_log), bins=20)
            dist = [{"range": f"{edges[i]:.0f}-{edges[i+1]:.0f}", "count": int(hist[i])} for i in range(len(hist))]
            result[name] = {
                "model": type(model).__name__,
                "r2_score": round(float(r2_score(y_log, pred_log)), 4),
                "feature_importance": top_fi,
                "distribution": dist,
            }
        else:
            probs = model.predict_proba(X)[:, 1]
            hist, edges = np.histogram(probs, bins=10, range=(0, 1))
            dist = [{"range": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "count": int(hist[i])} for i in range(len(hist))]
            result[name] = {
                "model": type(model).__name__,
                **saved_metrics.get(name, {}),
                "feature_importance": top_fi,
                "distribution": dist,
            }

    try:
        result["sample_predictions"] = sample_predictions(df, list(spec.tasks))
    except Exception as e:
        logger.warning(f"Sample prediction error: {e}")
        result["sample_predictions"] = []
    return result


@router.get("/what-if/schema")
async def what_if_schema():
    """Slider configuration (most influential features, data-driven ranges) for the active dataset."""
    cfg = _read_json("whatif.json")
    if not cfg:
        raise HTTPException(status_code=404, detail="Models not trained. Open Data Pipeline and train this dataset first.")
    return {"meta": get_spec().meta(), "sliders": cfg["sliders"]}


class WhatIfRequest(BaseModel):
    features: dict[str, Any]


@router.post("/what-if")
async def what_if_prediction(req: WhatIfRequest):
    """Predict outcomes for a hypothetical customer. Unspecified features default to dataset medians."""
    spec = get_spec()
    models_path = settings.models_path
    cfg = _read_json("whatif.json")
    if not cfg:
        raise HTTPException(status_code=404, detail="Models not trained yet")
    medians, metrics = cfg["medians"], _read_json("metrics.json")

    def row(feats):
        return pd.DataFrame([{f: req.features.get(f, medians.get(f, 0)) for f in feats}])

    out, recs = {}, []
    for task in spec.tasks:
        model = joblib.load(models_path / f"{task}_model.pkl")
        feats = joblib.load(models_path / f"{task}_features.pkl")
        if task == "ltv":
            out["ltv_estimate"] = round(float(np.expm1(model.predict(row(feats))[0])), 2)
        else:
            key = "churn_probability" if task == "churn" else "conversion_probability"
            out[key] = round(float(model.predict_proba(row(feats))[0, 1]), 4)

    # Thresholds are relative to each dataset's own base rates
    churn_base = metrics.get("churn", {}).get("base_rate", 0.2)
    conv_base = metrics.get("conversion", {}).get("base_rate", 0.2)
    ltv_median = metrics.get("ltv", {}).get("ltv_median", 0)
    if "churn_probability" in out and out["churn_probability"] > 2 * churn_base:
        recs.append("High churn risk (over 2x the average) — trigger retention campaign immediately")
    if "ltv_estimate" in out and out["ltv_estimate"] < ltv_median:
        recs.append("Below-median value — consider upsell or subscription upgrade offer")
    if "conversion_probability" in out and out["conversion_probability"] < conv_base:
        recs.append("Below-average conversion likelihood — improve targeting and engagement before spending on this customer")
    if not recs:
        recs.append("Customer profile is healthy — maintain current engagement strategy")

    return {**out, "base_rates": {"churn": churn_base, "conversion": conv_base}, "recommendations": recs}
