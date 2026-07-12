"""
Feature engineering pipeline.
Computes RFM metrics, session statistics, derived behavioral scores,
and prepares scaled feature matrices for ML models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

from config import settings
from utils.logger import logger

NUMERIC_FEATURES = [
    "age",
    "account_age_days",
    "total_purchases",
    "total_spend",
    "avg_order_value",
    "purchase_frequency_monthly",
    "last_purchase_days_ago",
    "session_count_30d",
    "avg_session_duration_min",
    "pages_per_session",
    "email_open_rate",
    "email_click_rate",
    "app_opens_30d",
    "support_tickets",
    "nps_score",
    # Derived
    "rfm_recency",
    "rfm_frequency",
    "rfm_monetary",
    "rfm_score",
    "engagement_score",
    "digital_activity_score",
    "support_burden",
    "spend_per_day",
    "session_quality_score",
]

CATEGORICAL_FEATURES = [
    "gender",
    "income_bracket",
    "education_level",
    "occupation",
    "city_tier",
    "subscription_type",
    "preferred_device",
    "product_category",
]


def _rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    max_recency = df["last_purchase_days_ago"].max()
    df["rfm_recency"] = (max_recency - df["last_purchase_days_ago"]) / max_recency
    df["rfm_frequency"] = np.log1p(df["purchase_frequency_monthly"])
    df["rfm_monetary"] = np.log1p(df["total_spend"])
    df["rfm_score"] = (df["rfm_recency"] + df["rfm_frequency"] / df["rfm_frequency"].max() +
                       df["rfm_monetary"] / df["rfm_monetary"].max()) / 3.0
    return df


def _behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df["engagement_score"] = (
        df["email_open_rate"] * 0.25
        + df["email_click_rate"] * 0.25
        + (df["session_count_30d"] / (df["session_count_30d"].max() + 1e-9)) * 0.25
        + (df["app_opens_30d"] / (df["app_opens_30d"].max() + 1e-9)) * 0.25
    )
    df["digital_activity_score"] = np.log1p(
        df["session_count_30d"] * df["avg_session_duration_min"] * df["pages_per_session"]
    )
    df["support_burden"] = df["support_tickets"] / (df["account_age_days"] / 30.0 + 1e-9)
    df["spend_per_day"] = df["total_spend"] / (df["account_age_days"] + 1e-9)
    df["session_quality_score"] = (
        df["avg_session_duration_min"] / (df["avg_session_duration_min"].max() + 1e-9) * 0.5
        + df["pages_per_session"] / (df["pages_per_session"].max() + 1e-9) * 0.5
    )
    return df


def _encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, dict]:
    """
    Returns:
        df_enriched   — original DataFrame with derived columns
        X_scaled      — scaled numeric + encoded feature matrix
        scaler        — fitted StandardScaler
        encoders      — LabelEncoders keyed by column name
    """
    logger.info("Starting feature engineering...")
    df = _rfm_features(df.copy())
    df = _behavioral_features(df)
    df, encoders = _encode_categoricals(df)

    encoded_cat_cols = [c + "_enc" for c in CATEGORICAL_FEATURES if c in df.columns]
    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + encoded_cat_cols

    X = df[feature_cols].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    models_path = settings.models_path
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_path / "scaler.pkl")
    joblib.dump(encoders, models_path / "encoders.pkl")
    joblib.dump(imputer, models_path / "imputer.pkl")
    joblib.dump(feature_cols, models_path / "feature_cols.pkl")

    logger.info(f"Feature engineering complete — {X_scaled_df.shape[1]} features")
    return df, X_scaled_df, scaler, encoders


def load_feature_matrix() -> tuple[pd.DataFrame, list[str]]:
    """Load saved feature matrix and column names."""
    scaler: StandardScaler = joblib.load(settings.models_path / "scaler.pkl")
    feature_cols: list[str] = joblib.load(settings.models_path / "feature_cols.pkl")
    return scaler, feature_cols
