"""Request and response models for every API endpoint."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Tier = Literal["high", "medium", "low"]
CaseStatus = Literal["new", "reviewing", "dispatched", "confirmed", "cleared"]


class Reason(BaseModel):
    feature: str
    label: str
    value: Optional[float] = None
    display_value: str
    shap_value: float


# --- Health ------------------------------------------------------------------

class ReportsStatus(BaseModel):
    available: bool
    fpdf_version: str
    detail: Optional[str] = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    reports: ReportsStatus


# --- Model -------------------------------------------------------------------

class ConfusionMatrix(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int


class ModelMetrics(BaseModel):
    threshold: float
    trained_threshold: float
    model_version: str
    trained_at: Optional[str] = None
    metrics: Dict[str, float]
    confusion_matrix: ConfusionMatrix
    customers_monitored: int
    flagged: int
    base_rate: float
    risk_tier_distribution: Dict[Tier, int]


class ThresholdUpdate(BaseModel):
    # None returns to the threshold chosen during training.
    threshold: Optional[float] = Field(default=None, gt=0.0, lt=1.0)


class OperatingPoint(ConfusionMatrix):
    threshold: float
    precision: float
    recall: float


class ScoreDistribution(BaseModel):
    edges: List[float]
    honest: List[int]
    theft: List[int]
    threshold: float


class ComparisonRow(BaseModel):
    model: str
    label: str
    threshold: float
    auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float


class Driver(BaseModel):
    feature: str
    label: str
    mean_abs_shap: float
    risk_when: Literal["higher", "lower", "unclear"]


class GlobalDrivers(BaseModel):
    sample_size: int
    drivers: List[Driver]


class TrainingStage(BaseModel):
    name: str
    seconds: float


class TrainingSummary(BaseModel):
    model_version: Optional[str] = None
    trained_at: Optional[str] = None
    quick_mode: Optional[bool] = None
    n_trials: Optional[int] = None
    cv_metric: Optional[str] = None
    cv_best_score: Optional[float] = None
    train_customers: Optional[int] = None
    test_customers: Optional[int] = None
    n_features: Optional[int] = None
    auc: Optional[float] = None
    pr_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    threshold: Optional[float] = None
    stages: List[TrainingStage]
    best_params: Dict[str, float]


# --- Customers and predictions ----------------------------------------------

class CustomerRow(BaseModel):
    customer_id: str
    rank: int
    risk_score: float
    risk_tier: Tier


class CustomerList(BaseModel):
    items: List[CustomerRow]
    total: int
    page: int
    page_size: int
    threshold: float


class Reading(BaseModel):
    date: str
    kwh: Optional[float] = None


class TimeSeries(BaseModel):
    customer_id: str
    label: int
    points: List[Reading]


class Explanation(BaseModel):
    customer_id: str
    probability: float
    base_value: float
    contributions: List[Reason]


class PredictionRequest(BaseModel):
    customer_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    features: Optional[Dict[str, float]] = Field(default=None, max_length=500)
    # Defaults to the threshold in service.
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Prediction(BaseModel):
    customer_id: Optional[str] = None
    probability: float
    prediction: int
    threshold: float
    risk_tier: Tier
    reasons: List[Reason]


class ThresholdPreview(BaseModel):
    threshold: float
    metrics: Dict[str, float]
    confusion_matrix: ConfusionMatrix


# --- Cases -------------------------------------------------------------------

class CaseRow(BaseModel):
    customer_id: str
    rank: int
    risk_score: float
    risk_tier: Tier
    flagged: bool
    status: CaseStatus
    note: str
    updated_at: Optional[str] = None
    top_driver: Optional[Reason] = None


class CaseList(BaseModel):
    items: List[CaseRow]
    total: int
    page: int
    page_size: int
    status_counts: Dict[CaseStatus, int]
    threshold: float


class CaseEvent(BaseModel):
    at: str
    event: str


class CaseDetail(CaseRow):
    history: List[CaseEvent]
    population: int


class CaseUpdate(BaseModel):
    status: Optional[CaseStatus] = None
    note: Optional[str] = Field(default=None, max_length=4000)


# --- Pipeline ----------------------------------------------------------------

class PipelineStage(BaseModel):
    key: str
    name: str
    seconds: float
    detail: str


class RunSummary(BaseModel):
    customers: int
    flagged: int
    tiers: Dict[Tier, int]
    threshold: float
    model_version: str


class PipelineRun(BaseModel):
    run_id: str
    trigger: str
    started_at: str
    finished_at: Optional[str] = None
    seconds: Optional[float] = None
    status: Literal["running", "succeeded", "failed"]
    error: Optional[str] = None
    stages: List[PipelineStage]
    summary: Optional[RunSummary] = None


class PipelineConfig(BaseModel):
    run_on_startup: bool
    scoring_interval_minutes: float
    environment: str


# --- Datasets and reports ----------------------------------------------------

class LabelMetrics(BaseModel):
    theft: int
    caught: int
    precision: Optional[float] = None
    recall: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None


class ScoredRow(BaseModel):
    customer_id: str
    probability: float
    risk_tier: Tier
    label: Optional[int] = None


class DatasetSummary(BaseModel):
    format: Literal["consumption", "features"]
    customers: int
    days: Optional[int] = None
    features_found: int
    features_expected: int
    threshold: float
    flagged: int
    tiers: Dict[Tier, int]
    mean_probability: float
    labelled: bool
    label_metrics: Optional[LabelMetrics] = None
    top: List[ScoredRow]


class Dataset(BaseModel):
    dataset_id: str
    filename: str
    uploaded_at: str
    summary: DatasetSummary


class ReportRequest(BaseModel):
    kind: Literal["portfolio", "dataset", "case"]
    dataset_id: Optional[str] = Field(default=None, max_length=64)
    customer_id: Optional[str] = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def subject_given(self) -> "ReportRequest":
        if self.kind == "dataset" and not self.dataset_id:
            raise ValueError("dataset_id is required for a dataset report")
        if self.kind == "case" and not self.customer_id:
            raise ValueError("customer_id is required for a case report")
        return self


class Report(BaseModel):
    report_id: str
    kind: Literal["portfolio", "dataset", "case"]
    title: str
    subject: str
    created_at: str
    bytes: int
