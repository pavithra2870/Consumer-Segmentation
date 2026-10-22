"""
Dataset registry for the dashboard.

Each DatasetSpec describes how one dataset plugs into the shared pipeline:
which prediction tasks it supports, which columns are model features, which column
means "customer value" and "risk", and what to show in segment / persona views.

The active dataset for a request comes from config.current_dataset. Services call
get_spec() and never hard-code dataset columns.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import settings, current_dataset
from utils.logger import logger

TELCO_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
BANK_FILE = "bank-full.csv"
RETAIL_XLSX = "online_retail_II.xlsx"


@dataclass
class DatasetSpec:
    id: str
    label: str
    description: str
    source: str
    license: str
    tasks: tuple
    source_files: list
    value_col: str
    value_label: str
    value_prefix: str
    risk_col: str
    risk_label: str
    risk_kind: str                    # "churn" or "conversion"
    protected: list = field(default_factory=list)
    stat_cols: list = field(default_factory=list)      # (column, label) averaged per segment
    cat_cols: list = field(default_factory=list)       # (column, label) most common per segment
    mix_cols: list = field(default_factory=list)       # (column, title) shown in segment detail
    sample_cols: list = field(default_factory=list)    # (column, label, kind)
    features: dict = field(default_factory=dict)       # task -> feature list (filled by build)
    seg_cols: Optional[list] = None
    build: Optional[Callable[[], pd.DataFrame]] = None  # None: synthetic generator flow

    def meta(self) -> dict:
        return {
            "dataset": self.id,
            "value_label": self.value_label,
            "value_prefix": self.value_prefix,
            "risk_label": self.risk_label,
            "risk_kind": self.risk_kind,
            "tasks": list(self.tasks),
            "has_fairness": bool(self.protected),
        }


# --------------------------------------------------------------------------- builders
def _build_telco() -> pd.DataFrame:
    df = pd.read_csv(settings.data_path / TELCO_FILE)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce").fillna(0.0)

    feats = pd.get_dummies(df.drop(columns=["customerID", "Churn"]), drop_first=True, dtype=float)

    service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    extra = pd.DataFrame({
        "customer_id": df["customerID"],
        "is_churned": (df["Churn"] == "Yes").astype(int),
        "senior_citizen": df["SeniorCitizen"].map({0: "No", 1: "Yes"}),
        "partner": df["Partner"],
        "contract": df["Contract"],
        "internet_service": df["InternetService"],
        "payment_method": df["PaymentMethod"],
        "n_services": sum((df[c] == "Yes").astype(int) for c in service_cols),
        "has_internet": (df["InternetService"] != "No").astype(int),
        "fiber": (df["InternetService"] == "Fiber optic").astype(int),
        "contract_months": df["Contract"].map({"Month-to-month": 1, "One year": 12, "Two year": 24}),
        "autopay": df["PaymentMethod"].str.contains("automatic").astype(int),
    })
    out = pd.concat([extra, df[["gender"]], feats], axis=1)
    TELCO.features["churn"] = list(feats.columns)
    return out


def _build_bank() -> pd.DataFrame:
    df = pd.read_csv(settings.data_path / BANK_FILE, sep=";")
    base = df.drop(columns=["y", "duration"])   # duration is only known after the call: leaks the outcome
    base["previously_contacted"] = (base["pdays"] != -1).astype(int)
    base.loc[base["pdays"] == -1, "pdays"] = 0
    feats = pd.get_dummies(base, drop_first=True, dtype=float)

    extra = pd.DataFrame({
        "customer_id": [f"B{i + 1:06d}" for i in range(len(df))],
        "converted": (df["y"] == "yes").astype(int),
        "age_group": pd.cut(df["age"], [0, 29, 44, 59, 120], labels=["<30", "30-44", "45-59", "60+"]).astype(str),
        "has_housing_loan": (df["housing"] == "yes").astype(int),
        "has_personal_loan": (df["loan"] == "yes").astype(int),
        "in_default": (df["default"] == "yes").astype(int),
        "education_level": df["education"].map({"primary": 1, "secondary": 2, "tertiary": 3}).fillna(2),
        "married": (df["marital"] == "married").astype(int),
    })
    raw_display = df[["job", "marital", "education"]]
    out = pd.concat([extra, raw_display, feats], axis=1)
    BANK.features["conversion"] = list(feats.columns)
    return out


def _build_retail() -> pd.DataFrame:
    from services.benchmark_datasets import _load_retail_transactions, build_retail_customers
    cust = build_retail_customers(_load_retail_transactions())
    feat_cols = [c for c in cust.columns if c not in ("future_spend", "churned")]

    out = cust[feat_cols].copy()
    out.insert(0, "customer_id", cust.index.astype(str))
    out["is_churned"] = cust["churned"].astype(int)
    out["ltv_actual"] = cust["future_spend"].astype(float)
    out["market"] = np.where(cust["is_uk"] == 1, "UK", "International")
    out["order_band"] = pd.cut(cust["frequency"], [0, 1, 3, 9, 10_000],
                               labels=["1 order", "2-3 orders", "4-9 orders", "10+ orders"]).astype(str)
    out["log_frequency"] = np.log1p(cust["frequency"])
    out["log_monetary"] = np.log1p(cust["monetary"].clip(lower=0))
    RETAIL.features["churn"] = feat_cols
    RETAIL.features["ltv"] = feat_cols
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- registry
def _synthetic_features() -> dict:
    from services.prediction_engine import CHURN_FEATURES, LTV_FEATURES, CONVERSION_FEATURES
    return {"churn": CHURN_FEATURES, "ltv": LTV_FEATURES, "conversion": CONVERSION_FEATURES}


SYNTHETIC = DatasetSpec(
    id="synthetic", label="Synthetic Consumers",
    description="100k simulated consumers with latent archetypes and calibrated class rates.",
    source="Generated by this project", license="n/a",
    tasks=("churn", "conversion", "ltv"), source_files=["consumers.csv"],
    value_col="ltv_actual", value_label="Avg LTV", value_prefix="$",
    risk_col="churn_probability", risk_label="Churn Risk", risk_kind="churn",
    protected=["gender", "income_bracket", "city_tier"],
    stat_cols=[("total_spend", "Avg total spend ($)"), ("purchase_frequency_monthly", "Avg purchases / month"),
               ("nps_score", "Avg NPS"), ("session_count_30d", "Avg sessions (30d)"), ("age", "Avg age")],
    cat_cols=[("income_bracket", "Top income bracket"), ("subscription_type", "Top subscription")],
    mix_cols=[("subscription_type", "Subscription Mix"), ("income_bracket", "Income Distribution")],
    sample_cols=[("customer_id", "Customer ID", "id"), ("age", "Age", "num"), ("total_spend", "Spend", "money"),
                 ("ltv_actual", "LTV", "money"), ("churn_probability", "Churn Prob", "pct"), ("nps_score", "NPS", "num")],
)

TELCO = DatasetSpec(
    id="telco", label="Telco Churn (IBM)",
    description="7,043 telecom customers with contract, services and billing data and a real churn flag.",
    source="IBM sample dataset (Kaggle: blastchar/telco-customer-churn)",
    license="See the Kaggle dataset page before redistributing",
    tasks=("churn",), source_files=[TELCO_FILE],
    value_col="TotalCharges", value_label="Avg billed to date", value_prefix="$",
    risk_col="is_churned", risk_label="Churn Rate", risk_kind="churn",
    protected=["gender", "senior_citizen", "partner"],
    stat_cols=[("tenure", "Avg tenure (months)"), ("MonthlyCharges", "Avg monthly charge ($)"),
               ("n_services", "Avg services"), ("contract_months", "Avg contract length (months)")],
    cat_cols=[("contract", "Top contract"), ("internet_service", "Top internet service"), ("payment_method", "Top payment")],
    mix_cols=[("contract", "Contract Type"), ("internet_service", "Internet Service")],
    sample_cols=[("customer_id", "Customer ID", "id"), ("tenure", "Tenure (mo)", "num"),
                 ("MonthlyCharges", "Monthly", "money"), ("TotalCharges", "Billed", "money"),
                 ("contract", "Contract", "text"), ("is_churned", "Churned", "flag")],
    seg_cols=["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "n_services",
              "has_internet", "fiber", "contract_months", "autopay"],
    build=_build_telco,
)

BANK = DatasetSpec(
    id="bank", label="Bank Marketing (UCI)",
    description="45,211 outbound marketing calls from a Portuguese bank. Target: term-deposit subscription.",
    source="UCI ML Repository #222 (Moro, Rita & Cortez, 2014)", license="CC BY 4.0",
    tasks=("conversion",), source_files=[BANK_FILE],
    value_col="balance", value_label="Avg balance", value_prefix="€",
    risk_col="converted", risk_label="Subscription Rate", risk_kind="conversion",
    protected=["age_group", "marital", "education"],
    stat_cols=[("age", "Avg age"), ("balance", "Avg balance (€)"), ("campaign", "Avg contacts this campaign"),
               ("previous", "Avg previous contacts")],
    cat_cols=[("job", "Top job"), ("education", "Top education"), ("marital", "Top marital status")],
    mix_cols=[("job", "Job"), ("education", "Education")],
    sample_cols=[("customer_id", "Customer ID", "id"), ("age", "Age", "num"), ("job", "Job", "text"),
                 ("balance", "Balance", "money"), ("campaign", "Contacts", "num"), ("converted", "Subscribed", "flag")],
    seg_cols=["age", "balance", "has_housing_loan", "has_personal_loan", "in_default",
              "education_level", "married", "campaign", "previous"],
    build=_build_bank,
)

RETAIL = DatasetSpec(
    id="retail", label="Online Retail II (UCI)",
    description="824k transactions from a UK online retailer (2009-2011) aggregated to customers; "
                "churn and spend targets come from the 90 days after a cutoff date.",
    source="UCI ML Repository #502 (Chen, 2012)", license="CC BY 4.0",
    tasks=("churn", "ltv"), source_files=[RETAIL_XLSX],
    value_col="ltv_actual", value_label="Avg 90-day spend", value_prefix="£",
    risk_col="is_churned", risk_label="Churn Rate", risk_kind="churn",
    protected=[],
    stat_cols=[("recency_days", "Avg days since last order"), ("frequency", "Avg orders"),
               ("monetary", "Avg total spend (£)"), ("n_products", "Avg distinct products")],
    cat_cols=[("market", "Top market"), ("order_band", "Top order band")],
    mix_cols=[("market", "Market"), ("order_band", "Order Frequency")],
    sample_cols=[("customer_id", "Customer ID", "id"), ("recency_days", "Recency (d)", "num"),
                 ("frequency", "Orders", "num"), ("monetary", "Spend", "money"),
                 ("ltv_actual", "Next 90d", "money"), ("is_churned", "Churned", "flag")],
    seg_cols=["recency_days", "log_frequency", "log_monetary", "avg_order_value", "n_products", "tenure_days"],
    build=_build_retail,
)

REGISTRY = {s.id: s for s in (SYNTHETIC, TELCO, BANK, RETAIL)}


def get_spec(dataset_id: Optional[str] = None) -> DatasetSpec:
    spec = REGISTRY[dataset_id or current_dataset.get()]
    if spec.id == "synthetic" and not spec.features:
        spec.features = _synthetic_features()
    return spec


def task_features(task: str) -> list:
    """Model features for a task. Non-synthetic specs fill these when their table is built."""
    spec = get_spec()
    if not spec.features:
        load_feature_lists(spec)
    return spec.features.get(task, [])


def load_feature_lists(spec: DatasetSpec) -> None:
    """Recover feature lists from saved model artifacts when the table has not been rebuilt this process."""
    for task in spec.tasks:
        path = settings.models_path / f"{task}_features.pkl"
        if path.exists():
            spec.features[task] = joblib.load(path)


# --------------------------------------------------------------------------- pipeline helpers
def prepare_features(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the customer table and the scaled matrix used for clustering.
    Returns (df_enriched, X_scaled).
    """
    if spec.build is None:
        from services.data_generator import load_dataset
        from services.feature_engineering import engineer_features
        df_enriched, X_scaled, _, _ = engineer_features(load_dataset())
        return df_enriched, X_scaled

    logger.info(f"Building customer table for {spec.id}...")
    df = spec.build()
    df.to_csv(settings.table_path, index=False)

    X = df[spec.seg_cols].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    models_path = settings.models_path
    joblib.dump(scaler, models_path / "scaler.pkl")
    joblib.dump(imputer, models_path / "imputer.pkl")
    joblib.dump(list(spec.seg_cols), models_path / "feature_cols.pkl")
    return df, pd.DataFrame(X_scaled, columns=spec.seg_cols, index=df.index)


def _generic_label(subset: pd.DataFrame, df: pd.DataFrame, spec: DatasetSpec) -> str:
    def rel(col):
        overall = df[col].mean()
        return subset[col].mean() / overall if overall else 1.0

    v, r = rel(spec.value_col), rel(spec.risk_col)
    value_word = "High-value" if v >= 1.25 else "Low-value" if v <= 0.75 else "Mid-value"
    if spec.risk_kind == "churn":
        risk_word = "high churn" if r >= 1.25 else "low churn" if r <= 0.75 else "avg churn"
    else:
        risk_word = "high propensity" if r >= 1.25 else "low propensity" if r <= 0.75 else "avg propensity"
    return f"{value_word}, {risk_word}"


def segment_cards(df: pd.DataFrame) -> list[dict]:
    """Per-segment summary used by the segmentation, overview and persona views."""
    spec = get_spec()
    labels = df["segment"].values
    out = []
    for sid in sorted(df["segment"].unique()):
        mask = labels == sid
        subset = df[mask]
        if spec.id == "synthetic":
            from services.segmentation import _label_for_cluster
            label = _label_for_cluster(int(sid), subset.mean(numeric_only=True))
        else:
            label = _generic_label(subset, df, spec)
        top = {col: round(float(subset[col].mean()), 4)
               for col, _ in spec.stat_cols[:3] if col in subset.columns}
        out.append({
            "segment_id": int(sid),
            "label": label,
            "size": int(mask.sum()),
            "percentage": round(float(mask.sum()) / len(labels) * 100, 2),
            "avg_ltv": round(float(subset[spec.value_col].mean()), 2) if spec.value_col in subset else 0,
            "avg_churn_risk": round(float(subset[spec.risk_col].mean()), 4) if spec.risk_col in subset else 0,
            "top_features": top,
        })

    # Generic labels can repeat across segments; suffix duplicates with the segment id
    seen = {}
    for c in out:
        seen[c["label"]] = seen.get(c["label"], 0) + 1
    for c in out:
        if seen[c["label"]] > 1:
            c["label"] = f"{c['label']} ({c['segment_id']})"
    return out


def dataset_status(spec: DatasetSpec) -> dict:
    """Capabilities and readiness, evaluated for the given spec regardless of the current request."""
    token = current_dataset.set(spec.id)
    try:
        files_present = all((settings.data_path / f).exists() for f in spec.source_files
                            if spec.id != "synthetic")
        return {
            **spec.meta(),
            "id": spec.id,
            "label": spec.label,
            "description": spec.description,
            "source": spec.source,
            "license": spec.license,
            "files_present": files_present or spec.id == "synthetic",
            "data_ready": settings.table_path.exists(),
            "models_ready": settings.segmented_path.exists() and (settings.models_path / "segmentation_model.pkl").exists(),
        }
    finally:
        current_dataset.reset(token)
