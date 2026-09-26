"""Request and response models for every API endpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Tier = Literal["high", "medium", "low"]
CaseStatus = Literal["new", "reviewing", "dispatched", "confirmed", "cleared"]


class Reason(BaseModel):
    feature: str
    label: str
    value: Optional[float] = None
    display_value: str
    shap_value: float


# --- Health and identity -----------------------------------------------------

class ReportsStatus(BaseModel):
    available: bool
    fpdf_version: str
    detail: Optional[str] = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    problems: List[str]
    database: str
    blob_store: str
    reports: ReportsStatus


class Me(BaseModel):
    name: str
    role: Literal["analyst", "supervisor"]
    auth_required: bool


class AuditEntry(BaseModel):
    at: str
    actor: str
    role: str
    action: str
    target: Optional[str] = None
    detail: Optional[str] = None


# --- Operations: model in service ---------------------------------------------

class ConfusionMatrix(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int


class Population(BaseModel):
    id: str
    source: Literal["sample", "upload"]
    filename: str
    promoted_at: Optional[str] = None
    promoted_by: Optional[str] = None


class PopulationRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=64)


class ModelMetrics(BaseModel):
    threshold: float
    trained_threshold: float
    model_version: str
    trained_at: Optional[str] = None
    pipeline: str
    pipeline_label: str
    customers_monitored: int
    flagged: int
    expected_thefts_flagged: float
    expected_thefts_total: float
    risk_tier_distribution: Dict[Tier, int]
    population: Population


class ThresholdUpdate(BaseModel):
    # None returns to the threshold chosen during training.
    threshold: Optional[float] = Field(default=None, gt=0.0, lt=1.0)


class ValidationPoint(ConfusionMatrix):
    precision: Optional[float] = None
    recall: float
    f1: float


class ValidationSummary(ValidationPoint):
    customers: int
    theft: int


class ThresholdPreview(BaseModel):
    threshold: float
    validation: ValidationSummary
    population_flagged: int
    population_expected_thefts: float


class OperatingPoint(ValidationPoint):
    threshold: float
    population_flagged: int
    visits: int
    expected_thefts_found: float
    net_value: float


class OperatingCurve(BaseModel):
    validation_customers: int
    validation_theft: int
    population_customers: int
    capacity: Optional[int] = None
    cost_per_visit: float
    value_per_theft: float
    threshold: float
    trained_threshold: float
    points: List[OperatingPoint]


class ScoreDistribution(BaseModel):
    edges: List[float]
    counts: List[int]
    threshold: float


class Driver(BaseModel):
    feature: str
    label: str
    mean_abs_shap: float
    risk_when: Literal["higher", "lower", "unclear"]


class GlobalDrivers(BaseModel):
    sample_size: int
    drivers: List[Driver]


# --- Research: the test-set record ---------------------------------------------

class TestPopulation(BaseModel):
    split: Literal["test"]
    customers: int
    theft: int
    description: str
    limitation: str


class CalibrationStats(BaseModel):
    brier: float
    log_loss: float
    ece: float
    mean_predicted: float
    observed_rate: float


class ResearchEvaluation(BaseModel):
    population: TestPopulation
    pipeline: str
    pipeline_label: str
    model_version: str
    threshold: float
    metrics: Dict[str, float]
    confusion_matrix: ConfusionMatrix
    calibration: Dict[str, Optional[CalibrationStats]]


class ComparisonRow(BaseModel):
    model: str
    label: str
    served: bool
    preprocessing: str  # "raw", "clean" or, for the hybrid, "raw+sequence"
    treatment: Literal["none", "smote", "smote_enn"]
    threshold: float
    auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    gmean: float
    mcc: float
    training_time: float
    inference_ms_per_customer: float
    model_size_mb: float
    pr_auc_ci: Optional[List[float]] = None
    f1_ci: Optional[List[float]] = None
    p_value_pr_auc: Optional[float] = None
    p_value_f1: Optional[float] = None
    p_value_mcnemar: Optional[float] = None


class CurvePoint(ConfusionMatrix):
    threshold: float
    precision: float
    recall: float


class ResearchCurve(BaseModel):
    population: TestPopulation
    threshold: float
    points: List[CurvePoint]


class ResearchDistribution(BaseModel):
    population: TestPopulation
    edges: List[float]
    honest: List[int]
    theft: List[int]
    threshold: float


class ReliabilityBin(BaseModel):
    low: float
    high: float
    count: int
    mean_predicted: float
    observed_rate: float


class CalibrationRow(BaseModel):
    model: str
    label: str
    validation_raw: CalibrationStats
    validation_platt: CalibrationStats
    validation_isotonic: CalibrationStats
    test_raw: CalibrationStats
    test_platt: CalibrationStats
    test_isotonic: CalibrationStats


class CalibrationReport(BaseModel):
    population: TestPopulation
    method: Optional[str] = None
    fitted_on: Optional[str] = None
    served: Optional[str] = None
    pipelines: List[CalibrationRow]
    reliability: Dict[str, List[ReliabilityBin]]


class ClassCounts(BaseModel):
    honest: int
    theft: int


class SeparabilityStats(BaseModel):
    rows: int
    theft_share: float
    silhouette: float
    fisher_ratio_mean: float
    fisher_ratio_max: float
    boundary_noise: float
    boundary_noise_theft: float


class TreatmentCounts(BaseModel):
    before: ClassCounts
    after: ClassCounts
    synthetic_created: int
    synthetic_removed_by_enn: Optional[int] = None
    honest_removed_by_enn: Optional[int] = None
    theft_removed_by_enn: Optional[int] = None


class TreatmentEffect(BaseModel):
    counts: TreatmentCounts
    diagnostics: SeparabilityStats


class ResamplingEffect(BaseModel):
    config: Dict[str, float]
    before: SeparabilityStats
    smote: TreatmentEffect
    smote_enn: TreatmentEffect


class TrainingStage(BaseModel):
    name: str
    seconds: float


class TrainingSummary(BaseModel):
    pipeline: Optional[str] = None
    pipeline_label: Optional[str] = None
    model_version: Optional[str] = None
    trained_at: Optional[str] = None
    quick_mode: Optional[bool] = None
    device: Optional[str] = None
    n_trials: Optional[int] = None
    cv_metric: Optional[str] = None
    cv_best_score: Optional[float] = None
    cv_fold_scores: Optional[List[float]] = None
    train_customers: Optional[int] = None
    validation_customers: Optional[int] = None
    test_customers: Optional[int] = None
    n_features: Optional[int] = None
    stages: List[TrainingStage]
    best_params: Dict[str, float]
    provenance: Dict[str, Any]
    manifest: Dict[str, Any]


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
    points: List[Reading]


class ModelPart(BaseModel):
    """One part of the hybrid: its calibrated probability and its weight in the blend."""
    name: str
    label: str
    probability: float
    weight: float


class Explanation(BaseModel):
    customer_id: str
    probability: float
    # The SHAP contributions explain the XGBoost model's raw score; for the hybrid that model is one part.
    raw_score: float
    tree_probability: Optional[float] = None
    base_value: float
    contributions: List[Reason]
    parts: Optional[List[ModelPart]] = None


class WeekEffect(BaseModel):
    week: int
    start: Optional[str] = None
    end: Optional[str] = None
    effect: float


class SequenceExplanation(BaseModel):
    customer_id: str
    available: bool
    label: Optional[str] = None
    probability: Optional[float] = None
    raw_score: Optional[float] = None
    weight: Optional[float] = None
    weeks: List[WeekEffect]


class Attribution(BaseModel):
    feature: str
    label: str
    weight: float


class ExplanationCheck(BaseModel):
    customer_id: str
    top_n: int
    shap: List[Attribution]
    lime: List[Attribution]
    shared: List[str]
    consistent: bool
    message: str
    note: str


class PredictionRequest(BaseModel):
    customer_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    # Every model feature; null marks a value that could not be computed (a missing reading).
    features: Optional[Dict[str, Optional[float]]] = Field(default=None, max_length=500)
    # Defaults to the threshold in service.
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Prediction(BaseModel):
    customer_id: Optional[str] = None
    probability: float
    raw_score: float
    prediction: int
    threshold: float
    risk_tier: Tier
    reasons: List[Reason]
    parts: Optional[List[ModelPart]] = None


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
    updated_by: Optional[str] = None
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
    actor: str
    event: str


class Resolution(BaseModel):
    outcome: Literal["confirmed", "cleared"]
    reason: str
    evidence: str
    by: str
    at: str


class CaseDetail(CaseRow):
    history: List[CaseEvent]
    population: int
    allowed_transitions: List[CaseStatus]
    resolution: Optional[Resolution] = None


class CaseUpdate(BaseModel):
    status: Optional[CaseStatus] = None
    note: Optional[str] = Field(default=None, max_length=4000)
    # Resolving (confirmed / cleared) needs both; reopening needs a reason.
    reason: Optional[str] = Field(default=None, max_length=1000)
    evidence: Optional[str] = Field(default=None, max_length=500)


# --- Pipeline ----------------------------------------------------------------

class PipelineStage(BaseModel):
    key: str
    name: str
    seconds: float
    detail: str


class RunSummary(BaseModel):
    customers: int
    flagged: int
    expected_thefts_flagged: Optional[float] = None
    tiers: Dict[Tier, int]
    threshold: float
    model_version: str
    population: Optional[str] = None


class PipelineRun(BaseModel):
    run_id: str
    trigger: str
    actor: Optional[str] = None
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
    uploaded_by: str
    summary: DatasetSummary


ReportKind = Literal["portfolio", "dataset", "case", "research"]


class ReportRequest(BaseModel):
    kind: ReportKind
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
    kind: ReportKind
    title: str
    subject: str
    created_at: str
    created_by: str
    bytes: int
