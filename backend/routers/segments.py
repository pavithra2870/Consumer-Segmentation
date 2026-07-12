from fastapi import APIRouter, HTTPException
import pandas as pd

from config import settings
from utils.logger import logger

router = APIRouter(prefix="/segments", tags=["Segments"])


def _load_segmented_df() -> pd.DataFrame:
    path = settings.data_path / "consumers_segmented.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Segmentation not run. Call POST /train-models first.")
    return pd.read_csv(path)


@router.get("")
async def get_segments():
    """Return segment summaries, scatter plot data, and cluster metadata."""
    import joblib
    df = _load_segmented_df()

    try:
        model = joblib.load(settings.models_path / "segmentation_model.pkl")
        pca = joblib.load(settings.models_path / "pca.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Segmentation model not found")

    import numpy as np
    from services.segmentation import _label_for_cluster

    n_clusters = int(df["segment"].nunique())
    labels = df["segment"].values

    # Rebuild scatter data from PCA
    feat_cols = joblib.load(settings.models_path / "feature_cols.pkl")
    imputer = joblib.load(settings.models_path / "imputer.pkl")
    scaler = joblib.load(settings.models_path / "scaler.pkl")

    avail = [c for c in feat_cols if c in df.columns]
    X = df[avail].fillna(0).values
    X_imp = imputer.transform(X)
    X_sc = scaler.transform(X_imp)
    X_2d = pca.transform(X_sc)

    rng = np.random.default_rng(42)
    sample_size = min(3000, len(X_2d))
    idx = rng.choice(len(X_2d), size=sample_size, replace=False)
    scatter_data = [
        {"x": round(float(X_2d[i, 0]), 4), "y": round(float(X_2d[i, 1]), 4), "segment": int(labels[i])}
        for i in idx
    ]

    # Silhouette score
    from sklearn.metrics import silhouette_score
    sil = float(silhouette_score(X_sc[:5000], labels[:5000]))

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
            "top_features": {
                col: round(float(subset[col].mean()), 4)
                for col in ["rfm_score", "engagement_score", "total_spend"]
                if col in subset.columns
            },
        })

    return {
        "n_clusters": n_clusters,
        "method": "kmeans",
        "silhouette_score": round(sil, 4),
        "segments": segments,
        "scatter_data": scatter_data,
    }


@router.get("/{segment_id}")
async def get_segment_detail(segment_id: int):
    """Return detailed statistics for a single segment."""
    df = _load_segmented_df()
    if segment_id not in df["segment"].values:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")

    subset = df[df["segment"] == segment_id]
    numeric_cols = subset.select_dtypes(include="number").columns.tolist()
    for col in ["customer_id", "segment"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    return {
        "segment_id": segment_id,
        "size": len(subset),
        "statistics": subset[numeric_cols].describe().round(4).to_dict(),
        "top_categories": subset["product_category"].value_counts().head(5).to_dict() if "product_category" in subset else {},
        "subscription_mix": subset["subscription_type"].value_counts().to_dict() if "subscription_type" in subset else {},
        "income_mix": subset["income_bracket"].value_counts().to_dict() if "income_bracket" in subset else {},
        "sample_customers": subset.head(10).to_dict(orient="records"),
    }
