from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
import numpy as np
import pandas as pd
import joblib

from config import settings
from utils.logger import logger

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("")
async def get_predictions():
    """Return prediction model results and sample predictions."""
    seg_path = settings.data_path / "consumers_segmented.csv"
    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="Models not trained. Call POST /train-models first.")

    df = pd.read_csv(seg_path)
    models_path = settings.models_path

    result = {}
    for name, model_file, feat_file, target_col, model_type in [
        ("churn", "churn_model.pkl", "churn_features.pkl", "is_churned", "classifier"),
        ("ltv", "ltv_model.pkl", "ltv_features.pkl", "ltv_actual", "regressor"),
        ("conversion", "conversion_model.pkl", "conversion_features.pkl", "converted", "classifier"),
    ]:
        try:
            model = joblib.load(models_path / model_file)
            features = joblib.load(models_path / feat_file)
            avail = [f for f in features if f in df.columns]
            X = df[avail].fillna(0)

            if model_type == "classifier":
                probs = model.predict_proba(X)[:, 1]
                import numpy as np
                hist, edges = np.histogram(probs, bins=10, range=(0, 1))
                dist = [{"range": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "count": int(hist[i])} for i in range(len(hist))]
                fi = dict(zip(avail, model.feature_importances_.tolist() if hasattr(model, "feature_importances_") else np.abs(model.coef_[0]).tolist()))
                result[name] = {
                    "model": type(model).__name__,
                    "feature_importance": {k: round(v, 5) for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]},
                    "distribution": dist,
                }
            else:
                y_log = np.log1p(df[target_col]) if target_col in df.columns else np.zeros(len(df))
                y_pred_log = model.predict(X)
                from sklearn.metrics import r2_score
                r2 = float(r2_score(y_log, y_pred_log))
                ltv_pred = np.expm1(y_pred_log)
                hist, edges = np.histogram(ltv_pred, bins=20)
                dist = [{"range": f"{edges[i]:.0f}-{edges[i+1]:.0f}", "count": int(hist[i])} for i in range(len(hist))]
                fi = dict(zip(avail, model.feature_importances_.tolist()))
                result[name] = {
                    "model": type(model).__name__,
                    "r2_score": round(r2, 4),
                    "feature_importance": {k: round(v, 5) for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]},
                    "distribution": dist,
                }
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_file}")

    # Sample predictions
    sample = df.sample(min(50, len(df)), random_state=42)
    sample_preds = []
    try:
        churn_m = joblib.load(models_path / "churn_model.pkl")
        churn_f = joblib.load(models_path / "churn_features.pkl")
        ltv_m = joblib.load(models_path / "ltv_model.pkl")
        ltv_f = joblib.load(models_path / "ltv_features.pkl")
        conv_m = joblib.load(models_path / "conversion_model.pkl")
        conv_f = joblib.load(models_path / "conversion_features.pkl")

        for _, row in sample.iterrows():
            churn_p = float(churn_m.predict_proba(row[[f for f in churn_f if f in row.index]].fillna(0).values.reshape(1, -1))[0, 1])
            ltv_p = float(np.expm1(ltv_m.predict(row[[f for f in ltv_f if f in row.index]].fillna(0).values.reshape(1, -1))[0]))
            conv_p = float(conv_m.predict_proba(row[[f for f in conv_f if f in row.index]].fillna(0).values.reshape(1, -1))[0, 1])
            sample_preds.append({
                "customer_id": row.get("customer_id", ""),
                "churn_probability": round(churn_p, 4),
                "ltv_estimate": round(ltv_p, 2),
                "conversion_score": round(conv_p, 4),
                "segment": int(row.get("segment", -1)),
            })
    except Exception as e:
        logger.warning(f"Sample prediction error: {e}")

    result["sample_predictions"] = sample_preds
    return result


class WhatIfRequest(BaseModel):
    features: dict[str, Any]


@router.post("/what-if")
async def what_if_prediction(req: WhatIfRequest):
    """Predict outcomes for a hypothetical customer feature set."""
    models_path = settings.models_path
    df = pd.DataFrame([req.features])

    try:
        churn_m = joblib.load(models_path / "churn_model.pkl")
        churn_f = joblib.load(models_path / "churn_features.pkl")
        ltv_m = joblib.load(models_path / "ltv_model.pkl")
        ltv_f = joblib.load(models_path / "ltv_features.pkl")
        conv_m = joblib.load(models_path / "conversion_model.pkl")
        conv_f = joblib.load(models_path / "conversion_features.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Models not trained yet")

    def get_row(feats):
        return pd.DataFrame([{f: req.features.get(f, 0) for f in feats}])

    churn_p = float(churn_m.predict_proba(get_row(churn_f).fillna(0))[0, 1])
    ltv_p = float(np.expm1(ltv_m.predict(get_row(ltv_f).fillna(0))[0]))
    conv_p = float(conv_m.predict_proba(get_row(conv_f).fillna(0))[0, 1])

    recs = []
    if churn_p > 0.6:
        recs.append("High churn risk — trigger retention campaign immediately")
    if ltv_p < 500:
        recs.append("Low LTV — consider upsell or subscription upgrade offer")
    if conv_p < 0.3:
        recs.append("Low conversion score — improve onboarding and email engagement")
    if not recs:
        recs.append("Customer profile is healthy — maintain current engagement strategy")

    return {
        "churn_probability": round(churn_p, 4),
        "ltv_estimate": round(ltv_p, 2),
        "conversion_probability": round(conv_p, 4),
        "recommendations": recs,
    }
