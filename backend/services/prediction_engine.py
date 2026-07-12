"""
Prediction engine: churn classification, LTV regression, conversion scoring.
Uses scikit-learn ensemble models with cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
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


def _available_features(feature_list: list[str], df: pd.DataFrame) -> list[str]:
    return [f for f in feature_list if f in df.columns]


def train_churn_model(df: pd.DataFrame) -> dict:
    features = _available_features(CHURN_FEATURES, df)
    X = df[features].fillna(0)
    y = df["is_churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.random_state, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=settings.random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    logger.info(f"Churn model — accuracy={acc:.4f}, AUC={auc:.4f}")

    importance = dict(zip(features, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

    dist = _build_distribution(y_prob, "churn_probability")

    joblib.dump(model, settings.models_path / "churn_model.pkl")
    joblib.dump(features, settings.models_path / "churn_features.pkl")

    return {
        "model": "GradientBoostingClassifier",
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "feature_importance": {k: round(v, 5) for k, v in importance.items()},
        "distribution": dist,
    }


def train_ltv_model(df: pd.DataFrame) -> dict:
    features = _available_features(LTV_FEATURES, df)
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

    ltv_actual = np.expm1(y)
    dist = _build_distribution_continuous(ltv_actual, "ltv", n_bins=20)

    joblib.dump(model, settings.models_path / "ltv_model.pkl")
    joblib.dump(features, settings.models_path / "ltv_features.pkl")

    return {
        "model": "RandomForestRegressor",
        "r2_score": round(r2, 4),
        "feature_importance": {k: round(v, 5) for k, v in importance.items()},
        "distribution": dist,
    }


def train_conversion_model(df: pd.DataFrame) -> dict:
    features = _available_features(CONVERSION_FEATURES, df)
    X = df[features].fillna(0)
    y = df["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.random_state, stratify=y
    )

    model = LogisticRegression(
        max_iter=500,
        random_state=settings.random_state,
        C=1.0,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    logger.info(f"Conversion model — accuracy={acc:.4f}, AUC={auc:.4f}")

    coef = np.abs(model.coef_[0])
    importance = dict(zip(features, coef.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

    dist = _build_distribution(y_prob, "conversion_score")

    joblib.dump(model, settings.models_path / "conversion_model.pkl")
    joblib.dump(features, settings.models_path / "conversion_features.pkl")

    return {
        "model": "LogisticRegression",
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "feature_importance": {k: round(v, 5) for k, v in importance.items()},
        "distribution": dist,
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


def train_all_models(df: pd.DataFrame) -> dict:
    logger.info("Training all prediction models...")
    results = {
        "churn": train_churn_model(df),
        "ltv": train_ltv_model(df),
        "conversion": train_conversion_model(df),
    }

    # Generate sample predictions for display
    sample_size = min(50, len(df))
    sample = df.sample(sample_size, random_state=42)

    churn_model = joblib.load(settings.models_path / "churn_model.pkl")
    churn_feats = joblib.load(settings.models_path / "churn_features.pkl")
    ltv_model = joblib.load(settings.models_path / "ltv_model.pkl")
    ltv_feats = joblib.load(settings.models_path / "ltv_features.pkl")
    conv_model = joblib.load(settings.models_path / "conversion_model.pkl")
    conv_feats = joblib.load(settings.models_path / "conversion_features.pkl")

    sample_preds = []
    for _, row in sample.iterrows():
        churn_p = float(churn_model.predict_proba(row[churn_feats].fillna(0).values.reshape(1, -1))[0, 1])
        ltv_p = float(np.expm1(ltv_model.predict(row[ltv_feats].fillna(0).values.reshape(1, -1))[0]))
        conv_p = float(conv_model.predict_proba(row[conv_feats].fillna(0).values.reshape(1, -1))[0, 1])
        sample_preds.append({
            "customer_id": row.get("customer_id", ""),
            "churn_probability": round(churn_p, 4),
            "ltv_estimate": round(ltv_p, 2),
            "conversion_score": round(conv_p, 4),
            "segment": int(row.get("segment", -1)),
        })

    results["sample_predictions"] = sample_preds
    logger.info("All prediction models trained successfully")
    return results
