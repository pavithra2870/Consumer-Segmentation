"""
Prediction engine: churn classification, LTV regression, conversion scoring.
Which tasks run, and on which features, comes from the active dataset (services/datasets.py).
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, r2_score, roc_auc_score
import joblib

from config import settings
from utils.logger import logger

CHURN_FEATURES = [
    "rfm_recency", "rfm_frequency", "rfm_monetary", "rfm_score",
    "engagement_score", "support_burden", "session_quality_score",
    "last_purchase_days_ago", "nps_score", "email_open_rate",
    "app_opens_30d", "support_tickets",
]

LTV_FEATURES = [
    "rfm_score", "total_spend", "avg_order_value", "purchase_frequency_monthly",
    "account_age_days", "engagement_score", "digital_activity_score",
    "session_count_30d", "subscription_type_enc",
]

CONVERSION_FEATURES = [
    "engagement_score", "email_open_rate", "email_click_rate",
    "session_count_30d", "pages_per_session", "rfm_score",
    "digital_activity_score", "income_bracket_enc",
]

TARGETS = {"churn": "is_churned", "conversion": "converted", "ltv": "ltv_actual"}
WHATIF_SLIDERS = 10


def _task_features(task: str, df: pd.DataFrame) -> list[str]:
    from services.datasets import task_features
    return [f for f in task_features(task) if f in df.columns]


def _train_classifier(task: str, df: pd.DataFrame, max_depth: int) -> dict:
    features = _task_features(task, df)
    X = df[features].fillna(0)
    y = df[TARGETS[task]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.random_state, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=max_depth,
        random_state=settings.random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    ap = float(average_precision_score(y_test, y_prob))
    baseline = float(max(y_test.mean(), 1 - y_test.mean()))
    logger.info(f"{task} model — accuracy={acc:.4f}, AUC={auc:.4f}")

    importance = dict(zip(features, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

    joblib.dump(model, settings.models_path / f"{task}_model.pkl")
    joblib.dump(features, settings.models_path / f"{task}_features.pkl")

    return {
        "model": "GradientBoostingClassifier",
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "baseline_accuracy": round(baseline, 4),
        "base_rate": round(float(y.mean()), 4),
        "feature_importance": {k: round(v, 5) for k, v in importance.items()},
        "distribution": _build_distribution(y_prob, f"{task}_probability"),
    }


def train_churn_model(df: pd.DataFrame) -> dict:
    return _train_classifier("churn", df, max_depth=4)


def train_conversion_model(df: pd.DataFrame) -> dict:
    return _train_classifier("conversion", df, max_depth=3)


def train_ltv_model(df: pd.DataFrame) -> dict:
    features = _task_features("ltv", df)
    X = df[features].fillna(0)
    y = np.log1p(df["ltv_actual"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.random_state
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        random_state=settings.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    logger.info(f"LTV model — R²={r2:.4f}")

    importance = dict(zip(features, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

    joblib.dump(model, settings.models_path / "ltv_model.pkl")
    joblib.dump(features, settings.models_path / "ltv_features.pkl")

    return {
        "model": "RandomForestRegressor",
        "r2_score": round(r2, 4),
        "ltv_median": round(float(np.expm1(y).median()), 2),
        "feature_importance": {k: round(v, 5) for k, v in importance.items()},
        "distribution": _build_distribution_continuous(np.expm1(y), "ltv", n_bins=20),
    }


def _build_distribution(probs: np.ndarray, name: str, n_bins: int = 10) -> list[dict]:
    hist, edges = np.histogram(probs, bins=n_bins, range=(0, 1))
    return [
        {"range": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "count": int(hist[i])}
        for i in range(len(hist))
    ]


def _build_distribution_continuous(values: np.ndarray, name: str, n_bins: int = 20) -> list[dict]:
    hist, edges = np.histogram(values, bins=n_bins)
    return [
        {"range": f"{edges[i]:.0f}-{edges[i+1]:.0f}", "count": int(hist[i])}
        for i in range(len(hist))
    ]


def _whatif_config(df: pd.DataFrame, results: dict) -> dict:
    """Medians for every model feature plus slider ranges for the most important ones."""
    all_feats, weight = set(), {}
    for task in results:
        feats = joblib.load(settings.models_path / f"{task}_features.pkl")
        all_feats.update(feats)
        for f, w in results[task]["feature_importance"].items():
            weight[f] = weight.get(f, 0.0) + w

    medians = {f: float(df[f].fillna(0).median()) for f in all_feats if f in df.columns}
    sliders = []
    for f in sorted(weight, key=weight.get, reverse=True)[:WHATIF_SLIDERS]:
        col = df[f].fillna(0)
        lo, hi = float(col.quantile(0.01)), float(col.quantile(0.99))
        if hi <= lo:
            hi = lo + 1.0
        is_int = bool(np.allclose(col, col.round()))
        step = 1.0 if is_int else (hi - lo) / 100
        sliders.append({
            "key": f, "label": f.replace("_", " "),
            "min": round(lo, 4), "max": round(hi, 4), "step": round(step, 4),
            "default": round(medians.get(f, lo), 4),
        })
    return {"medians": medians, "sliders": sliders}


def train_all_models(df: pd.DataFrame) -> dict:
    from services.datasets import get_spec
    spec = get_spec()
    logger.info(f"Training prediction models for {spec.id}: {', '.join(spec.tasks)}")

    trainers = {"churn": train_churn_model, "ltv": train_ltv_model, "conversion": train_conversion_model}
    results = {task: trainers[task](df) for task in spec.tasks}

    metric_keys = ("accuracy", "auc", "avg_precision", "baseline_accuracy", "base_rate", "ltv_median", "r2_score")
    metrics = {
        name: {k: results[name][k] for k in metric_keys if k in results[name]}
        for name in results
    }
    with open(settings.models_path / "metrics.json", "w") as f:
        json.dump(metrics, f)
    with open(settings.models_path / "whatif.json", "w") as f:
        json.dump(_whatif_config(df, results), f)

    results["sample_predictions"] = sample_predictions(df, list(spec.tasks))
    logger.info("All prediction models trained successfully")
    return results


def sample_predictions(df: pd.DataFrame, tasks: list[str], n: int = 50) -> list[dict]:
    """Model outputs for a random sample of customers, predicted in one batch per model."""
    sample = df.sample(min(n, len(df)), random_state=42)
    cols = {"churn": "churn_probability", "conversion": "conversion_score", "ltv": "ltv_estimate"}
    preds = {}
    for task in tasks:
        model = joblib.load(settings.models_path / f"{task}_model.pkl")
        feats = joblib.load(settings.models_path / f"{task}_features.pkl")
        X = sample[[f for f in feats if f in sample.columns]].fillna(0)
        if task == "ltv":
            preds[task] = np.expm1(model.predict(X))
        else:
            preds[task] = model.predict_proba(X)[:, 1]

    rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        out = {"customer_id": str(row.get("customer_id", "")), "segment": int(row.get("segment", -1))}
        for task in tasks:
            out[cols[task]] = round(float(preds[task][i]), 4 if task != "ltv" else 2)
        rows.append(out)
    return rows
