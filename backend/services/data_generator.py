"""
Synthetic consumer dataset generator.
Combines rule-based numpy distributions with optional Ollama text generation.
Produces 100k+ realistic consumer records with demographics, behavioral metrics,
transaction history, NPS scores, and time-series activity.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from config import settings
from utils.logger import logger
from utils.ollama_client import ollama

fake = Faker()
RNG = np.random.default_rng(settings.random_state)

# --- Distributions ---
INCOME_BRACKETS = ["low", "medium", "high", "very_high"]
INCOME_PROBS = [0.25, 0.40, 0.25, 0.10]
INCOME_SPEND_SCALE = {"low": 0.2, "medium": 0.5, "high": 0.8, "very_high": 1.0}

EDUCATION_LEVELS = ["high_school", "some_college", "bachelor", "master", "phd"]
EDUCATION_PROBS = [0.20, 0.20, 0.35, 0.18, 0.07]

OCCUPATIONS = [
    "professional", "manager", "technician", "sales", "service",
    "administrative", "student", "retired", "self_employed", "other",
]
OCCUPATION_PROBS = [0.15, 0.12, 0.14, 0.10, 0.10, 0.10, 0.08, 0.07, 0.09, 0.05]

SUBSCRIPTION_TYPES = ["free", "basic", "premium", "enterprise"]
SUBSCRIPTION_PROBS = [0.35, 0.30, 0.25, 0.10]
SUBSCRIPTION_SPEND_MULT = {"free": 0.1, "basic": 0.4, "premium": 0.8, "enterprise": 1.5}

PREFERRED_DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_PROBS = [0.55, 0.35, 0.10]

PRODUCT_CATEGORIES = [
    "electronics", "apparel", "home_garden", "sports", "beauty",
    "books", "toys", "food_beverage", "health", "automotive",
]

CITY_TIERS = [1, 2, 3]
CITY_TIER_PROBS = [0.30, 0.40, 0.30]


def _sample(arr: list, probs: list, size: int) -> np.ndarray:
    return RNG.choice(arr, size=size, p=probs)


def _clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def _income_factor(income_brackets: np.ndarray) -> np.ndarray:
    return np.vectorize(INCOME_SPEND_SCALE.get)(income_brackets)


def _subscription_mult(subs: np.ndarray) -> np.ndarray:
    return np.vectorize(SUBSCRIPTION_SPEND_MULT.get)(subs)


def _generate_demographics(n: int) -> dict:
    income_brackets = _sample(INCOME_BRACKETS, INCOME_PROBS, n)
    subscription_types = _sample(SUBSCRIPTION_TYPES, SUBSCRIPTION_PROBS, n)
    return {
        "customer_id": [f"C{i+1:07d}" for i in range(n)],
        "age": _clamp(RNG.integers(18, 81, size=n), 18, 80).astype(int),
        "gender": _sample(["M", "F", "Non-binary"], [0.48, 0.48, 0.04], n),
        "income_bracket": income_brackets,
        "education_level": _sample(EDUCATION_LEVELS, EDUCATION_PROBS, n),
        "occupation": _sample(OCCUPATIONS, OCCUPATION_PROBS, n),
        "city_tier": _sample(CITY_TIERS, CITY_TIER_PROBS, n),
        "subscription_type": subscription_types,
        "preferred_device": _sample(PREFERRED_DEVICES, DEVICE_PROBS, n),
        "product_category": _sample(PRODUCT_CATEGORIES, [0.1] * 10, n),
        "_income_factor": _income_factor(income_brackets),
        "_sub_mult": _subscription_mult(subscription_types),
    }


def _generate_behavioral(n: int, demo: dict) -> dict:
    inf = demo["_income_factor"]
    smult = demo["_sub_mult"]
    combined = (inf + smult) / 2

    account_age_days = _clamp(
        RNG.integers(30, 1826, size=n), 30, 1825
    ).astype(int)

    purchase_count = _clamp(
        (combined * 80 + RNG.poisson(15, size=n)).astype(int), 1, 500
    )
    total_spend = _clamp(
        combined * 8000 + RNG.exponential(2000, size=n), 5.0, 80000.0
    )
    avg_order_value = _clamp(total_spend / purchase_count, 5.0, 1000.0)

    last_purchase_days_ago = _clamp(
        RNG.integers(1, 366, size=n) * (1.5 - combined), 1, 365
    ).astype(int)

    session_count_30d = _clamp(
        (combined * 60 + RNG.poisson(10, size=n)).astype(int), 0, 200
    )
    avg_session_duration_min = _clamp(
        combined * 30 + RNG.exponential(8, size=n), 0.5, 90.0
    )
    pages_per_session = _clamp(
        combined * 12 + RNG.exponential(3, size=n), 1.0, 40.0
    )

    email_open_rate = _clamp(combined * 0.4 + RNG.beta(2, 5, size=n) * 0.3, 0.0, 1.0)
    email_click_rate = _clamp(email_open_rate * 0.3 + RNG.beta(1, 8, size=n) * 0.1, 0.0, 1.0)

    app_opens_30d = _clamp(
        (combined * 100 + RNG.poisson(20, size=n)).astype(int), 0, 300
    )

    support_tickets = _clamp(
        RNG.poisson(1.5, size=n) * (1.2 - combined), 0, 25
    ).astype(int)

    nps_score = _clamp(
        (combined * 6 + 4 + RNG.normal(0, 1.5, size=n)).astype(int), 0, 10
    ).astype(int)

    purchase_frequency_monthly = _clamp(
        purchase_count / (account_age_days / 30.0), 0.01, 30.0
    )

    return {
        "account_age_days": account_age_days,
        "total_purchases": purchase_count,
        "total_spend": np.round(total_spend, 2),
        "avg_order_value": np.round(avg_order_value, 2),
        "purchase_frequency_monthly": np.round(purchase_frequency_monthly, 3),
        "last_purchase_days_ago": last_purchase_days_ago,
        "session_count_30d": session_count_30d,
        "avg_session_duration_min": np.round(avg_session_duration_min, 2),
        "pages_per_session": np.round(pages_per_session, 2),
        "email_open_rate": np.round(email_open_rate, 4),
        "email_click_rate": np.round(email_click_rate, 4),
        "app_opens_30d": app_opens_30d,
        "support_tickets": support_tickets,
        "nps_score": nps_score,
    }


def _generate_labels(n: int, demo: dict, behavioral: dict) -> dict:
    inf = demo["_income_factor"]
    recency_score = 1.0 - behavioral["last_purchase_days_ago"] / 365.0
    frequency_score = _clamp(behavioral["purchase_frequency_monthly"] / 10.0, 0, 1)
    engagement_score = (
        behavioral["email_open_rate"] * 0.3
        + _clamp(behavioral["session_count_30d"] / 100.0, 0, 1) * 0.3
        + _clamp(behavioral["app_opens_30d"] / 200.0, 0, 1) * 0.2
        + (1 - _clamp(behavioral["support_tickets"] / 10.0, 0, 1)) * 0.2
    )

    churn_score = (
        (1 - recency_score) * 0.4
        + (1 - frequency_score) * 0.3
        + (1 - engagement_score) * 0.2
        + (1 - inf) * 0.1
        + RNG.normal(0, 0.05, size=n)
    )
    churn_prob = _clamp(churn_score, 0.0, 1.0)
    is_churned = (RNG.uniform(0, 1, size=n) < churn_prob).astype(int)

    ltv = _clamp(
        behavioral["total_spend"] * (recency_score + frequency_score)
        * (1 + inf)
        + RNG.exponential(500, size=n),
        10.0,
        200000.0,
    )

    conversion_score = (
        engagement_score * 0.5
        + frequency_score * 0.3
        + inf * 0.2
        + RNG.normal(0, 0.05, size=n)
    )
    converted = (RNG.uniform(0, 1, size=n) < _clamp(conversion_score, 0, 1)).astype(int)

    return {
        "churn_probability": np.round(churn_prob, 4),
        "is_churned": is_churned,
        "ltv_actual": np.round(ltv, 2),
        "converted": converted,
        "created_at": [
            (datetime.now() - timedelta(days=int(d))).strftime("%Y-%m-%d")
            for d in behavioral["account_age_days"]
        ],
    }


def _generate_monthly_activity(n: int, behavioral: dict) -> list[str]:
    """Generate 12-month activity JSON per customer (compressed)."""
    results = []
    sessions = behavioral["session_count_30d"]
    for i in range(n):
        base = max(1, int(sessions[i] / 4))
        months = {}
        for m in range(1, 13):
            noise = int(RNG.poisson(base))
            months[f"m{m:02d}"] = max(0, noise)
        results.append(json.dumps(months))
    return results


async def _fetch_ollama_reviews(n_samples: int = 50) -> list[str]:
    """Fetch a small batch of realistic review snippets from Ollama."""
    available = await ollama.check_availability()
    if not available:
        logger.warning("Ollama unavailable — using static review templates")
        return []

    system = "You are generating realistic e-commerce customer reviews. Be concise (1-2 sentences)."
    prompts = [
        "Write a short customer review for an online shopping platform. Vary sentiment.",
    ] * n_samples

    logger.info(f"Fetching {n_samples} reviews from Ollama...")
    reviews = await ollama.generate_batch(prompts[:10], system=system)
    return [r.strip() for r in reviews if r.strip()]


STATIC_REVIEWS = [
    "Great experience overall, fast shipping and quality products.",
    "Delivery was delayed but customer service resolved it quickly.",
    "Product exactly as described. Will buy again.",
    "Disappointed with the quality for the price paid.",
    "Excellent value. Highly recommend to anyone looking for deals.",
    "App is easy to use. Checkout process could be smoother.",
    "Had issues with the return process but eventually got refunded.",
    "Amazing selection. Found exactly what I needed.",
    "Prices are competitive but the website is slow.",
    "Perfect gift. Recipient loved it. Smooth transaction.",
    "Product arrived damaged. Support was helpful in resolving.",
    "Subscription is worth every penny. Premium tier unlocks great deals.",
    "Not happy with recent price increases. Considering alternatives.",
    "The loyalty rewards program is fantastic.",
    "Regular customer for 3 years — quality has been consistent.",
]


async def generate_dataset(n: int = None) -> pd.DataFrame:
    """
    Generate the full synthetic consumer dataset.
    Returns a DataFrame and saves it to the data directory.
    """
    n = n or settings.dataset_size
    logger.info(f"Generating synthetic dataset with {n:,} rows...")

    demo = _generate_demographics(n)
    logger.info("Demographics generated")

    behav = _generate_behavioral(n, demo)
    logger.info("Behavioral features generated")

    labels = _generate_labels(n, demo, behav)
    logger.info("Labels generated")

    monthly = _generate_monthly_activity(n, behav)
    logger.info("Time-series activity generated")

    llm_reviews = await _fetch_ollama_reviews(50)
    all_reviews = llm_reviews + STATIC_REVIEWS
    review_pool = all_reviews if all_reviews else STATIC_REVIEWS
    review_text = [
        review_pool[RNG.integers(0, len(review_pool))] if RNG.random() < 0.6 else ""
        for _ in range(n)
    ]

    df = pd.DataFrame({
        **{k: v for k, v in demo.items() if not k.startswith("_")},
        **behav,
        **labels,
        "monthly_activity": monthly,
        "review_text": review_text,
    })

    data_path = settings.data_path
    data_path.mkdir(parents=True, exist_ok=True)
    out_file = data_path / "consumers.csv"
    df.to_csv(out_file, index=False)
    logger.info(f"Dataset saved → {out_file} ({df.shape[0]:,} rows, {df.shape[1]} cols)")
    return df


def load_dataset() -> pd.DataFrame:
    path = settings.data_path / "consumers.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run /generate-data first.")
    return pd.read_csv(path)
