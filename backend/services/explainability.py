"""
Explainability module: SHAP-based feature importance and segment reasoning.
Falls back to permutation importance if SHAP is unavailable.
"""

import numpy as np
import pandas as pd
import joblib

from config import settings
from utils.logger import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed — using permutation importance fallback")


def _permutation_importance_fallback(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> dict:
    """Simple permutation-based importance when SHAP is unavailable."""
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    importance = dict(zip(X.columns, result.importances_mean))
    return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:15])


def _shap_importance(model, X: pd.DataFrame, model_type: str = "tree") -> tuple[dict, list]:
    """Compute SHAP values and return feature importance + sample explanations."""
    sample_size = min(500, len(X))
    X_sample = X.sample(sample_size, random_state=42)

    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            # Binary classification — use positive class
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        mean_abs = np.abs(sv).mean(axis=0)
        importance = dict(zip(X.columns, mean_abs.tolist()))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])

        # Sample explanations (top 5 customers)
        sample_rows = []
        for i in range(min(5, len(X_sample))):
            row_shap = sv[i]
            row_feat_vals = X_sample.iloc[i]
            top_features = sorted(
                zip(X.columns, row_shap, row_feat_vals),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
            sample_rows.append({
                "customer_idx": int(X_sample.index[i]),
                "top_contributors": [
                    {"feature": f, "shap_value": round(float(s), 5), "feature_value": round(float(v), 4)}
                    for f, s, v in top_features
                ],
            })

        return importance, sample_rows

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return {}, []


def compute_explanations(df: pd.DataFrame) -> dict:
    """
    Compute SHAP/importance explanations for all trained models.
    Returns structured explanation data for the API.
    """
    logger.info("Computing model explanations...")
    models_path = settings.models_path
    results = {}

    model_specs = [
        ("churn", "churn_model.pkl", "churn_features.pkl", "is_churned", "tree"),
        ("ltv", "ltv_model.pkl", "ltv_features.pkl", "ltv_actual", "tree"),
        ("conversion", "conversion_model.pkl", "conversion_features.pkl", "converted", "linear"),
    ]

    for name, model_file, feat_file, target_col, model_type in model_specs:
        try:
            model = joblib.load(models_path / model_file)
            features = joblib.load(models_path / feat_file)
            avail_features = [f for f in features if f in df.columns]
            X = df[avail_features].fillna(0)

            if SHAP_AVAILABLE:
                importance, sample_explanations = _shap_importance(model, X, model_type)
            else:
                y = df[target_col] if target_col in df.columns else pd.Series(np.zeros(len(df)))
                importance = _permutation_importance_fallback(model, X, y)
                sample_explanations = []

            top_features = [
                {
                    "feature": feat,
                    "mean_abs_shap": round(float(val), 6),
                    "direction": "positive" if float(val) >= 0 else "negative",
                }
                for feat, val in list(importance.items())[:10]
            ]

            global_importance = [
                {"feature": feat, "importance": round(float(val), 6)}
                for feat, val in importance.items()
            ]

            results[name] = {
                "model": name,
                "top_features": top_features,
                "global_importance": global_importance,
                "segment_reasoning": sample_explanations,
            }
            logger.info(f"Explanations computed for {name} model")

        except FileNotFoundError:
            logger.warning(f"Model {model_file} not found — skipping {name} explanations")
            results[name] = {"model": name, "top_features": [], "global_importance": [], "segment_reasoning": []}
        except Exception as e:
            logger.error(f"Explanation error for {name}: {e}")
            results[name] = {"model": name, "top_features": [], "global_importance": [], "segment_reasoning": []}

    return results
