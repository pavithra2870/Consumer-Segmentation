"""
Bias and fairness module.
Checks model predictions for demographic disparities and computes
standard fairness metrics (demographic parity, adverse impact ratio).
"""

import numpy as np
import pandas as pd
import joblib

from config import settings
from utils.logger import logger

ADVERSE_IMPACT_THRESHOLD = 0.80  # 80% rule (EEOC guideline)
PARITY_THRESHOLD = 0.10  # 10pp demographic parity gap


def _positive_rate_by_group(df: pd.DataFrame, attribute: str, prediction_col: str) -> dict:
    rates = {}
    for group, group_df in df.groupby(attribute):
        rate = float(group_df[prediction_col].mean())
        rates[str(group)] = round(rate, 4)
    return rates


def _adverse_impact_ratio(rates: dict) -> dict:
    if not rates:
        return {}
    max_rate = max(rates.values())
    return {group: round(rate / max_rate, 4) if max_rate > 0 else 1.0 for group, rate in rates.items()}


def _top_rate_flag(scores: pd.Series, rate: float) -> pd.Series:
    """Flag the highest-scoring `rate` share of customers, so imbalanced tasks get a usable binary decision."""
    return (scores >= scores.quantile(1 - rate)).astype(int)


def _compute_fairness_metrics(df: pd.DataFrame, prediction_col: str, label: str, attributes: list[str]) -> list[dict]:
    metrics = []
    for attr in attributes:
        if attr not in df.columns:
            continue
        rates = _positive_rate_by_group(df, attr, prediction_col)
        airs = _adverse_impact_ratio(rates)

        for group, rate in rates.items():
            air = airs.get(group, 1.0)
            max_rate = max(rates.values())
            parity_gap = abs(rate - max_rate)
            flagged = air < ADVERSE_IMPACT_THRESHOLD or parity_gap > PARITY_THRESHOLD

            metrics.append({
                "group": group,
                "attribute": attr,
                "prediction": label,
                "positive_rate": rate,
                "adverse_impact_ratio": air,
                "parity_gap": round(parity_gap, 4),
                "flagged": flagged,
            })

    return metrics


def _distribution_by_group(df: pd.DataFrame, attributes: list[str]) -> dict:
    dist = {}
    for attr in attributes:
        if attr not in df.columns:
            continue
        counts = df[attr].value_counts().to_dict()
        dist[attr] = {str(k): int(v) for k, v in counts.items()}
    return dist


def generate_bias_report(df: pd.DataFrame) -> dict:
    """
    Compute fairness metrics for churn and conversion predictions.
    Returns a structured bias report with recommendations.
    """
    logger.info("Generating bias and fairness report...")

    from services.datasets import get_spec
    spec = get_spec()
    attributes = spec.protected
    models_path = settings.models_path
    df = df.copy()

    # Flag the top-scoring share of customers (matching each task's base rate) as positive
    all_metrics = []
    for task, target, col in (("churn", "is_churned", "pred_churn_binary"),
                              ("conversion", "converted", "pred_converted_binary")):
        if task not in spec.tasks:
            continue
        model = joblib.load(models_path / f"{task}_model.pkl")
        feats = joblib.load(models_path / f"{task}_features.pkl")
        avail = [f for f in feats if f in df.columns]
        scores = pd.Series(model.predict_proba(df[avail].fillna(0))[:, 1], index=df.index)
        df[col] = _top_rate_flag(scores, float(df[target].mean()))
        all_metrics += _compute_fairness_metrics(df, col, task, attributes)

    flagged_count = sum(1 for m in all_metrics if m["flagged"])
    overall_flag = flagged_count > 0

    summary_lines = []
    if overall_flag:
        summary_lines.append(f"⚠ {flagged_count} fairness issue(s) detected across demographic groups.")
        flagged = [m for m in all_metrics if m["flagged"]]
        for m in flagged[:3]:
            summary_lines.append(
                f"  • {m['prediction'].capitalize()} prediction: group '{m['group']}' "
                f"(by {m['attribute']}) has AIR={m['adverse_impact_ratio']:.2f}, "
                f"parity gap={m['parity_gap']:.2%}"
            )
    else:
        summary_lines.append("All demographic groups pass the 80% adverse impact rule and parity threshold.")

    recommendations = _build_recommendations(all_metrics)
    dist = _distribution_by_group(df, attributes)

    return {
        "overall_flag": overall_flag,
        "summary": " ".join(summary_lines),
        "metrics": all_metrics,
        "recommendations": recommendations,
        "distribution_by_group": dist,
    }


def _build_recommendations(metrics: list[dict]) -> list[str]:
    recs = []
    flagged = [m for m in metrics if m["flagged"]]
    if not flagged:
        return ["No immediate bias mitigation required. Continue monitoring quarterly."]

    attrs_flagged = set(m["attribute"] for m in flagged)
    preds_flagged = set(m["prediction"] for m in flagged)

    if "income_bracket" in attrs_flagged:
        recs.append(
            "Income-based disparity detected. Consider reweighting training samples "
            "or introducing income-aware calibration to equalize positive prediction rates."
        )
    if "gender" in attrs_flagged:
        recs.append(
            "Gender-based disparity detected. Audit feature correlates (e.g., spending patterns) "
            "that may proxy gender and apply fairness constraints during model retraining."
        )
    if "city_tier" in attrs_flagged:
        recs.append(
            "Geographic tier disparity detected. Review whether city_tier features introduce "
            "systemic disadvantage — consider geographic stratification in training splits."
        )
    other = attrs_flagged - {"income_bracket", "gender", "city_tier"}
    if other:
        recs.append(
            f"Disparity detected across {', '.join(sorted(other))}. Check whether correlated features act as "
            "proxies for these groups and consider fairness constraints or reweighting during retraining."
        )
    if "churn" in preds_flagged:
        recs.append(
            "Churn model shows unequal false-positive rates. High false-positive churn flags "
            "may drive over-incentivization for one group while under-serving another."
        )
    if "conversion" in preds_flagged:
        recs.append(
            "Conversion model shows targeting disparity. Ensure marketing budget allocation "
            "is not systematically excluding lower-ranked demographic groups."
        )

    recs.append(
        "Implement ongoing fairness monitoring with quarterly audits using updated data "
        "and document all mitigation steps for compliance reporting."
    )
    return recs
