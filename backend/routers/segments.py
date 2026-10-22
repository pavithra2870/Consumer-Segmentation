from fastapi import APIRouter, HTTPException
import pandas as pd

from config import settings
from services.datasets import get_spec, segment_cards
from utils.logger import logger

router = APIRouter(prefix="/segments", tags=["Segments"])


def _load_segmented_df() -> pd.DataFrame:
    path = settings.segmented_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Segmentation not run. Open Data Pipeline and train this dataset first.")
    return pd.read_csv(path)


@router.get("")
async def get_segments():
    """Return segment summaries, scatter plot data, and cluster metadata."""
    import joblib
    import numpy as np

    df = _load_segmented_df()
    spec = get_spec()

    try:
        pca = joblib.load(settings.models_path / "pca.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Segmentation model not found")

    n_clusters = int(df["segment"].nunique())
    labels = df["segment"].values

    # Rebuild scatter data from PCA
    feat_cols = joblib.load(settings.models_path / "feature_cols.pkl")
    imputer = joblib.load(settings.models_path / "imputer.pkl")
    scaler = joblib.load(settings.models_path / "scaler.pkl")

    avail = [c for c in feat_cols if c in df.columns]
    X_sc = scaler.transform(imputer.transform(df[avail].fillna(0).values))
    X_2d = pca.transform(X_sc)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_2d), size=min(3000, len(X_2d)), replace=False)
    scatter_data = [
        {"x": round(float(X_2d[i, 0]), 4), "y": round(float(X_2d[i, 1]), 4), "segment": int(labels[i])}
        for i in idx
    ]

    from sklearn.metrics import silhouette_score
    sil = float(silhouette_score(X_sc[:5000], labels[:5000]))

    return {
        "meta": spec.meta(),
        "n_clusters": n_clusters,
        "method": "kmeans",
        "silhouette_score": round(sil, 4),
        "segments": segment_cards(df),
        "scatter_data": scatter_data,
    }


@router.get("/{segment_id}")
async def get_segment_detail(segment_id: int):
    """Return detailed statistics for a single segment."""
    df = _load_segmented_df()
    spec = get_spec()
    if segment_id not in df["segment"].values:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")

    subset = df[df["segment"] == segment_id]
    numeric_cols = [c for c in subset.select_dtypes(include="number").columns if c not in ("customer_id", "segment")]

    mixes = [
        {"title": title, "data": {str(k): int(v) for k, v in subset[col].value_counts().head(8).items()}}
        for col, title in spec.mix_cols if col in subset.columns
    ]
    sample = subset.head(10)
    sample = sample.astype(object).where(sample.notna(), None)

    return {
        "meta": spec.meta(),
        "segment_id": segment_id,
        "size": len(subset),
        "statistics": subset[numeric_cols].describe().round(4).to_dict(),
        "mixes": mixes,
        "sample_columns": [{"key": c, "label": l, "kind": k} for c, l, k in spec.sample_cols],
        "sample_customers": sample[[c for c, _, _ in spec.sample_cols if c in sample.columns]].to_dict(orient="records"),
    }
