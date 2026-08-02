from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    dataset_path: str
    total_customers: int
    total_rows: int
    feature_count: int
    class_distribution: Dict[str, int]
    missing_values: Dict[str, int]
    zero_consumption_pct: float


class FeatureDistributionResponse(BaseModel):
    feature: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    bins: List[float]
    counts: List[int]


class CorrelationMatrixResponse(BaseModel):
    features: List[str]
    matrix: List[List[float]]


class TimeSeriesPoint(BaseModel):
    day_index: int
    consumption_kwh: Optional[float] = None
    sudden_drop: bool = False
    anomaly_score: float = 0.0


class CustomerTimeseriesResponse(BaseModel):
    customer_id: str
    label: Optional[int] = None
    points: List[TimeSeriesPoint]
    summary: Dict[str, Any]


class PredictionReason(BaseModel):
    feature: str
    value: float
    shap_value: float


class SinglePredictionRequest(BaseModel):
    customer_id: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class SinglePredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    prediction: int
    probability: float
    threshold: float
    top_reasons: List[PredictionReason] = Field(default_factory=list)


class ThresholdPreviewResponse(BaseModel):
    threshold: float
    metrics: Dict[str, float]
    confusion_matrix: Dict[str, int]


class CompareResponse(BaseModel):
    xgboost: Dict[str, Any]
    baselines: Dict[str, Any]


class MonitorResponse(BaseModel):
    timestamp: Optional[str] = None
    data_drift: Dict[str, Any] = Field(default_factory=dict)
    concept_drift: Dict[str, Any] = Field(default_factory=dict)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class TrainingJobCreateRequest(BaseModel):
    mode: str = Field(default="quick", pattern="^(quick|full|custom)$")
    config_overrides: Optional[Dict[str, Any]] = None


class TrainingJobStatusResponse(BaseModel):
    job_id: str
    mode: str
    status: str
    current_step: str
    best_score: Optional[float] = None
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
