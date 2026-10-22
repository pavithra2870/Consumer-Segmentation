"""
Benchmark runner: trains this project's model families against standard baselines
on each benchmark dataset and stores the results as JSON.

Every model in a task sees the same stratified 80/20 split.
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, davies_bouldin_score, f1_score,
    mean_absolute_error, r2_score, roc_auc_score, silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import settings
from utils.logger import logger
from services.benchmark_datasets import LOADERS

SEED = settings.random_state
N_BOOT = 200
SIL_SAMPLE = 10000
AGGLO_SAMPLE = 6000
K_RANGE = range(2, 9)

TASK_LABELS = {
    "churn": "Churn Prediction",
    "conversion": "Conversion Scoring",
    "ltv": "LTV Prediction",
    "segmentation": "Customer Segmentation",
}


def results_dir():
    d = settings.models_root / "benchmarks"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- classification
def _classifiers(task: str) -> list[tuple[str, bool, object]]:
    """(name, is_ours, estimator). 'Ours' mirrors the model family used in prediction_engine."""
    if task == "churn":
        ours = ("Gradient Boosting (ours)", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4, random_state=SEED))
    else:
        ours = ("Gradient Boosting (ours)", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=SEED))
    return [
        ("Majority class", False, DummyClassifier(strategy="prior")),
        ("Logistic Regression", False, make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))),
        ("Random Forest", False, RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        ("LightGBM", False, LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1)),
        (ours[0], True, ours[1]),
    ]


def _lift_top_decile(y_true: np.ndarray, prob: np.ndarray) -> float:
    if np.unique(prob).size == 1:
        return 1.0  # constant scores carry no ranking information
    n = max(1, len(y_true) // 10)
    top = np.argsort(-prob)[:n]
    base = y_true.mean()
    return float(y_true[top].mean() / base) if base > 0 else 0.0


def _auc_ci(y_true: np.ndarray, prob: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    scores = []
    n = len(y_true)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        scores.append(roc_auc_score(y_true[idx], prob[idx]))
    return [round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)]


def _feature_importance(model, cols: list[str]) -> dict:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        vals = np.abs(est.coef_[0])
    else:
        return {}
    vals = vals / (vals.sum() or 1.0)
    top = sorted(zip(cols, vals), key=lambda kv: kv[1], reverse=True)[:10]
    return {k: round(float(v), 4) for k, v in top}


def _fit_eval(estimator, X_tr, y_tr, X_te, y_te) -> dict:
    t0 = time.time()
    estimator.fit(X_tr, y_tr)
    secs = time.time() - t0
    prob = estimator.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    yt = y_te.to_numpy()
    return {
        "auc": round(float(roc_auc_score(yt, prob)), 4),
        "auc_ci": _auc_ci(yt, prob),
        "pr_auc": round(float(average_precision_score(yt, prob)), 4),
        "f1": round(float(f1_score(yt, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(yt, pred)), 4),
        "lift_top_decile": round(_lift_top_decile(yt, prob), 3),
        "train_seconds": round(secs, 1),
    }


def run_classification(task_name: str, task: dict) -> dict:
    X, y = task["X"].fillna(0), task["y"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    rate = float(y.mean())

    models, fitted_ours = [], None
    for name, is_ours, est in _classifiers(task_name):
        logger.info(f"  [{task_name}] {name}")
        row = {"name": name, "is_ours": is_ours, **_fit_eval(est, X_tr, y_tr, X_te, y_te)}
        models.append(row)
        if is_ours:
            fitted_ours = est

    out = {
        "label": TASK_LABELS[task_name], "kind": "classification", "target": task["target"],
        "positive_rate": round(rate, 4), "baseline_accuracy": round(max(rate, 1 - rate), 4),
        "n_train": len(X_tr), "n_test": len(X_te), "n_features": X.shape[1],
        "models": models,
        "feature_importance": _feature_importance(fitted_ours, list(X.columns)),
        "notes": list(task.get("notes", [])),
    }

    if "leaky_X" in task:
        Xl = task["leaky_X"].fillna(0)
        Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(Xl, y, test_size=0.2, random_state=SEED, stratify=y)
        leaky = _fit_eval(LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1),
                          Xl_tr, yl_tr, Xl_te, yl_te)
        out["leakage_check"] = {
            "label": "LightGBM with `duration` included",
            "auc": leaky["auc"],
            "note": "Call duration is only known after the call, so this score is unrealistically high.",
        }
    return out


# --------------------------------------------------------------------------- regression
def _regressors() -> list[tuple[str, bool, object]]:
    return [
        ("Mean predictor", False, DummyRegressor(strategy="mean")),
        ("Ridge", False, make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("LightGBM", False, LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1)),
        ("Random Forest (ours)", True, RandomForestRegressor(n_estimators=150, max_depth=8, n_jobs=-1, random_state=SEED)),
    ]


def run_regression(task: dict) -> dict:
    X, y = task["X"].fillna(0), task["y"].clip(lower=0)
    y_log = np.log1p(y)
    X_tr, X_te, ylog_tr, ylog_te = train_test_split(X, y_log, test_size=0.2, random_state=SEED)
    y_te_raw = np.expm1(ylog_te)

    models, fitted_ours = [], None
    for name, is_ours, est in _regressors():
        logger.info(f"  [ltv] {name}")
        t0 = time.time()
        est.fit(X_tr, ylog_tr)
        secs = time.time() - t0
        pred_log = est.predict(X_te)
        pred = np.expm1(np.clip(pred_log, 0, None))
        models.append({
            "name": name, "is_ours": is_ours,
            "r2_log": round(float(r2_score(ylog_te, pred_log)), 4),
            "mae": round(float(mean_absolute_error(y_te_raw, pred)), 2),
            "train_seconds": round(secs, 1),
        })
        if is_ours:
            fitted_ours = est

    return {
        "label": TASK_LABELS["ltv"], "kind": "regression", "target": task["target"],
        "target_mean": round(float(y.mean()), 2), "zero_share": round(float((y == 0).mean()), 4),
        "n_train": len(X_tr), "n_test": len(X_te), "n_features": X.shape[1],
        "models": models,
        "feature_importance": _feature_importance(fitted_ours, list(X.columns)),
        "notes": list(task.get("notes", [])),
    }


# --------------------------------------------------------------------------- segmentation
def _sil_db(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(SIL_SAMPLE, len(X)), replace=False)
    return (float(silhouette_score(X[idx], labels[idx])),
            float(davies_bouldin_score(X[idx], labels[idx])))


def _stability(X: np.ndarray, k: int) -> float:
    """Mean adjusted Rand index between KMeans fits on different seeds."""
    from sklearn.metrics import adjusted_rand_score
    fits = [KMeans(n_clusters=k, n_init=3, random_state=s).fit_predict(X) for s in range(3)]
    pairs = [adjusted_rand_score(fits[i], fits[j]) for i in range(3) for j in range(i + 1, 3)]
    return float(np.mean(pairs))


def run_segmentation(task: dict) -> dict:
    X = StandardScaler().fit_transform(task["X"].fillna(0).to_numpy(dtype=float))
    k_ours = settings.n_clusters

    sweep = []
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
        sil, db = _sil_db(X, labels)
        sweep.append({"k": k, "silhouette": round(sil, 4), "davies_bouldin": round(db, 4)})

    methods = []
    rng = np.random.default_rng(SEED)

    def add(name, is_ours, labels, Xm=None, **extra):
        sil, db = _sil_db(X if Xm is None else Xm, labels)
        methods.append({"name": name, "is_ours": is_ours, "silhouette": round(sil, 4),
                        "davies_bouldin": round(db, 4), **extra})

    add("Random assignment", False, rng.integers(0, k_ours, len(X)))
    km_labels = KMeans(n_clusters=k_ours, n_init=10, random_state=SEED).fit_predict(X)
    add(f"KMeans k={k_ours} (ours)", True, km_labels, stability_ari=round(_stability(X, k_ours), 4))
    add(f"Gaussian Mixture k={k_ours}", False,
        GaussianMixture(n_components=k_ours, random_state=SEED).fit(X).predict(X))
    sub = rng.choice(len(X), size=min(AGGLO_SAMPLE, len(X)), replace=False)
    add(f"Agglomerative (Ward) k={k_ours}", False,
        AgglomerativeClustering(n_clusters=k_ours).fit_predict(X[sub]), Xm=X[sub])

    best = max(sweep, key=lambda r: r["silhouette"])
    return {
        "label": TASK_LABELS["segmentation"], "kind": "segmentation",
        "n_samples": len(X), "n_features": X.shape[1], "k_ours": k_ours,
        "methods": methods, "k_sweep": sweep, "best_k_by_silhouette": best["k"],
        "feature_names": list(task["X"].columns),
        "notes": list(task.get("notes", [])) + ["Silhouette scored on a 10k sample; Agglomerative runs on a 6k subsample."],
    }


# --------------------------------------------------------------------------- orchestration
def run_dataset(dataset_id: str) -> dict:
    loader, _ = LOADERS[dataset_id]
    logger.info(f"Benchmarking dataset: {dataset_id}")
    data = loader()
    tasks = {}
    for name, task in data["tasks"].items():
        if task["kind"] == "classification":
            tasks[name] = run_classification(name, task)
        elif task["kind"] == "regression":
            tasks[name] = run_regression(task)
        else:
            tasks[name] = run_segmentation(task)

    result = {
        **data["meta"],
        "tasks": tasks,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(results_dir() / f"{dataset_id}.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Benchmark results saved for {dataset_id}")
    return result
