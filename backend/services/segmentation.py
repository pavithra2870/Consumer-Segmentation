"""
Segmentation engine: KMeans + GMM clustering with PCA dimensionality reduction
for 2D visualization and silhouette scoring.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

from config import settings
from utils.logger import logger

CLUSTER_LABELS = {
    0: "High-Value Loyalists",
    1: "At-Risk Customers",
    2: "Occasional Shoppers",
    3: "New Explorers",
    4: "Deal Seekers",
    5: "Dormant Accounts",
    6: "Brand Advocates",
    7: "Budget Conscious",
}


def _label_for_cluster(cluster_id: int, stats: pd.Series) -> str:
    """Assign a descriptive label based on cluster statistics."""
    label = CLUSTER_LABELS.get(cluster_id, f"Segment {cluster_id}")
    return label


def run_segmentation(
    X_scaled: pd.DataFrame,
    df_enriched: pd.DataFrame,
    n_clusters: int = None,
    method: str = "kmeans",
) -> dict:
    """
    Run clustering on scaled feature matrix.
    Returns cluster assignments, statistics, scatter data, and model artifacts.
    """
    n_clusters = n_clusters or settings.n_clusters
    logger.info(f"Running {method} segmentation with {n_clusters} clusters...")

    X = X_scaled.values

    if method == "gmm":
        model = GaussianMixture(
            n_components=n_clusters,
            random_state=settings.random_state,
            max_iter=200,
            covariance_type="full",
        )
        model.fit(X)
        labels = model.predict(X)
        probabilities = model.predict_proba(X)
    else:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=settings.random_state,
            n_init=10,
            max_iter=300,
        )
        labels = model.fit_predict(X)
        probabilities = None

    sil_score = float(silhouette_score(X[:5000], labels[:5000], sample_size=None))
    logger.info(f"Silhouette score: {sil_score:.4f}")

    pca = PCA(n_components=2, random_state=settings.random_state)
    X_2d = pca.fit_transform(X)

    df_enriched = df_enriched.copy()
    df_enriched["segment"] = labels

    from services.datasets import segment_cards
    segments = segment_cards(df_enriched)

    sample_size = min(3000, len(X_2d))
    sample_idx = np.random.default_rng(42).choice(len(X_2d), size=sample_size, replace=False)
    scatter_data = [
        {
            "x": round(float(X_2d[i, 0]), 4),
            "y": round(float(X_2d[i, 1]), 4),
            "segment": int(labels[i]),
            "customer_id": df_enriched.iloc[i]["customer_id"] if "customer_id" in df_enriched.columns else str(i),
        }
        for i in sample_idx
    ]

    models_path = settings.models_path
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_path / "segmentation_model.pkl")
    joblib.dump(pca, models_path / "pca.pkl")

    df_enriched.to_csv(settings.segmented_path, index=False)

    return {
        "n_clusters": n_clusters,
        "method": method,
        "silhouette_score": round(sil_score, 4),
        "segments": segments,
        "scatter_data": scatter_data,
    }


def load_segmentation_results() -> dict:
    path = settings.segmented_path
    if not path.exists():
        raise FileNotFoundError("Segmentation not run yet. Call /train-models first.")
    df = pd.read_csv(path)
    model = joblib.load(settings.models_path / "segmentation_model.pkl")
    return {"df": df, "model": model}
