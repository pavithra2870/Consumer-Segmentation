from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    result: Optional[Any] = None


class DatasetInfo(BaseModel):
    rows: int
    columns: int
    size_mb: float
    features: list[str]
    generated_at: str


class SegmentSummary(BaseModel):
    segment_id: int
    label: str
    size: int
    percentage: float
    avg_ltv: float
    avg_churn_risk: float
    top_features: dict[str, float]


class SegmentsResponse(BaseModel):
    n_clusters: int
    method: str
    silhouette_score: float
    segments: list[SegmentSummary]
    scatter_data: list[dict]


class PersonaProfile(BaseModel):
    segment_id: int
    name: str
    tagline: str
    demographics: dict[str, Any]
    behaviors: dict[str, Any]
    motivations: list[str]
    pain_points: list[str]
    recommended_actions: list[str]
    narrative: str


class PersonasResponse(BaseModel):
    personas: list[PersonaProfile]


class PredictionSummary(BaseModel):
    model: str
    accuracy: Optional[float] = None
    r2_score: Optional[float] = None
    feature_importance: dict[str, float]
    distribution: list[dict]


class PredictionsResponse(BaseModel):
    churn: PredictionSummary
    ltv: PredictionSummary
    conversion: PredictionSummary
    sample_predictions: list[dict]


class ShapFeature(BaseModel):
    feature: str
    mean_abs_shap: float
    direction: str


class ExplanationResponse(BaseModel):
    model: str
    top_features: list[ShapFeature]
    segment_reasoning: list[dict]
    global_importance: list[dict]


class FairnessMetric(BaseModel):
    group: str
    attribute: str
    positive_rate: float
    adverse_impact_ratio: float
    flagged: bool


class BiasReportResponse(BaseModel):
    overall_flag: bool
    summary: str
    metrics: list[FairnessMetric]
    recommendations: list[str]
    distribution_by_group: dict[str, Any]


class WhatIfRequest(BaseModel):
    customer_id: Optional[str] = None
    features: dict[str, Any]


class WhatIfResponse(BaseModel):
    churn_probability: float
    ltv_estimate: float
    conversion_probability: float
    segment_assignment: int
    recommendations: list[str]
