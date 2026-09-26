import axios, { AxiosError } from "axios";

const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL ?? "/api" });

const API_KEY_STORAGE_KEY = "sgcc.apiKey";

// The key is entered on the Settings page and kept for this browser tab only
// (sessionStorage): it is gone when the tab closes, and never ships in the build.
export function getStoredApiKey(): string {
  try {
    return window.sessionStorage.getItem(API_KEY_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setStoredApiKey(value: string): void {
  try {
    if (value) window.sessionStorage.setItem(API_KEY_STORAGE_KEY, value);
    else window.sessionStorage.removeItem(API_KEY_STORAGE_KEY);
  } catch {
    // Storage unavailable: the key lasts until reload.
  }
}

api.interceptors.request.use((config) => {
  const apiKey = getStoredApiKey();
  if (apiKey) config.headers.set("X-API-Key", apiKey);
  return config;
});

type Params = Record<string, string | number | undefined>;

async function get<T>(url: string, params?: Params): Promise<T> {
  return (await api.get<T>(url, { params })).data;
}

// ---------------------------------------------------------------------------
// Types (mirror backend/schemas.py)
// ---------------------------------------------------------------------------

export type Tier = "high" | "medium" | "low";
export type CaseStatus = "new" | "reviewing" | "dispatched" | "confirmed" | "cleared";

export interface Reason {
  feature: string;
  label: string;
  value: number | null;
  display_value: string;
  shap_value: number;
}

interface Health {
  status: "ok" | "degraded";
  model_loaded: boolean;
  model_version: string;
  problems: string[];
  database: string;
  blob_store: string;
  reports: { available: boolean; fpdf_version: string; detail: string | null };
}

type Role = "analyst" | "supervisor";

interface Me {
  name: string;
  role: Role;
  auth_required: boolean;
}

interface AuditEntry {
  at: string;
  actor: string;
  role: string;
  action: string;
  target: string | null;
  detail: string | null;
}

interface ConfusionMatrix {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

interface Population {
  id: string;
  source: "sample" | "upload";
  filename: string;
  promoted_at: string | null;
  promoted_by: string | null;
}

interface ModelMetrics {
  threshold: number;
  trained_threshold: number;
  model_version: string;
  trained_at: string | null;
  pipeline: string;
  pipeline_label: string;
  customers_monitored: number;
  flagged: number;
  expected_thefts_flagged: number;
  expected_thefts_total: number;
  risk_tier_distribution: Record<Tier, number>;
  population: Population;
}

interface ValidationPoint extends ConfusionMatrix {
  precision: number | null;
  recall: number;
  f1: number;
}

interface ThresholdPreview {
  threshold: number;
  validation: ValidationPoint & { customers: number; theft: number };
  population_flagged: number;
  population_expected_thefts: number;
}

export interface OperatingPoint extends ValidationPoint {
  threshold: number;
  population_flagged: number;
  visits: number;
  expected_thefts_found: number;
  net_value: number;
}

interface OperatingCurve {
  validation_customers: number;
  validation_theft: number;
  population_customers: number;
  capacity: number | null;
  cost_per_visit: number;
  value_per_theft: number;
  threshold: number;
  trained_threshold: number;
  points: OperatingPoint[];
}

interface PopulationDistribution {
  edges: number[];
  counts: number[];
  threshold: number;
}

// --- Research (test set) ---

interface TestPopulation {
  split: "test";
  customers: number;
  theft: number;
  description: string;
  limitation: string;
}

interface CalibrationStats {
  brier: number;
  log_loss: number;
  ece: number;
  mean_predicted: number;
  observed_rate: number;
}

interface ResearchEvaluation {
  population: TestPopulation;
  pipeline: string;
  pipeline_label: string;
  model_version: string;
  threshold: number;
  metrics: Record<"auc" | "pr_auc" | "precision" | "recall" | "f1" | "mcc" | "gmean" | "specificity" | "accuracy", number>;
  confusion_matrix: ConfusionMatrix;
  calibration: { raw: CalibrationStats | null; platt: CalibrationStats | null };
}

interface ResearchCurvePoint extends ConfusionMatrix {
  threshold: number;
  precision: number;
  recall: number;
}

interface ResearchCurve {
  population: TestPopulation;
  threshold: number;
  points: ResearchCurvePoint[];
}

interface ScoreDistribution {
  edges: number[];
  honest: number[];
  theft: number[];
  threshold: number;
}

interface ResearchDistribution extends ScoreDistribution {
  population: TestPopulation;
}

interface ComparisonRow {
  model: string;
  label: string;
  served: boolean;
  preprocessing: "raw" | "clean";
  treatment: "none" | "smote" | "smote_enn";
  threshold: number;
  auc: number;
  pr_auc: number;
  precision: number;
  recall: number;
  f1: number;
  gmean: number;
  mcc: number;
  training_time: number;
  inference_ms_per_customer: number;
  model_size_mb: number;
  pr_auc_ci: [number, number] | null;
  f1_ci: [number, number] | null;
  p_value_pr_auc: number | null;
  p_value_f1: number | null;
  p_value_mcnemar: number | null;
}

export interface ReliabilityBin {
  low: number;
  high: number;
  count: number;
  mean_predicted: number;
  observed_rate: number;
}

interface CalibrationReport {
  population: TestPopulation;
  method: string | null;
  fitted_on: string | null;
  served: string | null;
  pipelines: Array<{
    model: string;
    label: string;
    validation_raw: CalibrationStats;
    validation_platt: CalibrationStats;
    validation_isotonic: CalibrationStats;
    test_raw: CalibrationStats;
    test_platt: CalibrationStats;
    test_isotonic: CalibrationStats;
  }>;
  reliability: { raw?: ReliabilityBin[]; platt?: ReliabilityBin[] };
}

interface SeparabilityStats {
  rows: number;
  theft_share: number;
  silhouette: number;
  fisher_ratio_mean: number;
  fisher_ratio_max: number;
  boundary_noise: number;
  boundary_noise_theft: number;
}

interface TreatmentEffect {
  counts: {
    before: { honest: number; theft: number };
    after: { honest: number; theft: number };
    synthetic_created: number;
    synthetic_removed_by_enn: number | null;
    honest_removed_by_enn: number | null;
    theft_removed_by_enn: number | null;
  };
  diagnostics: SeparabilityStats;
}

interface ResamplingEffect {
  config: Record<string, number>;
  before: SeparabilityStats;
  smote: TreatmentEffect;
  smote_enn: TreatmentEffect;
}

interface GlobalDrivers {
  sample_size: number;
  drivers: Array<{ feature: string; label: string; mean_abs_shap: number; risk_when: "higher" | "lower" | "unclear" }>;
}

interface TrainingSummary {
  pipeline: string | null;
  pipeline_label: string | null;
  model_version: string | null;
  trained_at: string | null;
  device: string | null;
  n_trials: number | null;
  cv_metric: string | null;
  cv_best_score: number | null;
  cv_fold_scores: number[] | null;
  train_customers: number | null;
  validation_customers: number | null;
  test_customers: number | null;
  n_features: number | null;
  stages: Array<{ name: string; seconds: number }>;
  best_params: Record<string, number>;
  provenance: {
    code?: { commit: string | null; dirty: boolean | null };
    data_sha256?: string;
    libraries?: Record<string, string>;
  };
  manifest: { published_at?: string };
}

export interface Reading {
  date: string;
  kwh: number | null;
}

interface TimeSeries {
  customer_id: string;
  points: Reading[];
}

interface Explanation {
  customer_id: string;
  probability: number;
  raw_score: number;
  base_value: number;
  contributions: Reason[];
}

interface ExplanationCheck {
  customer_id: string;
  top_n: number;
  shap: Array<{ feature: string; label: string; weight: number }>;
  lime: Array<{ feature: string; label: string; weight: number }>;
  shared: string[];
  consistent: boolean;
  message: string;
  note: string;
}

interface Prediction {
  customer_id: string | null;
  probability: number;
  raw_score: number;
  prediction: number;
  threshold: number;
  risk_tier: Tier;
  reasons: Reason[];
}

interface CaseRow {
  customer_id: string;
  rank: number;
  risk_score: number;
  risk_tier: Tier;
  flagged: boolean;
  status: CaseStatus;
  note: string;
  updated_at: string | null;
  updated_by: string | null;
  top_driver: Reason | null;
}

interface CaseList {
  items: CaseRow[];
  total: number;
  page: number;
  page_size: number;
  status_counts: Record<CaseStatus, number>;
  threshold: number;
}

export interface CaseDetail extends CaseRow {
  history: Array<{ at: string; actor: string; event: string }>;
  population: number;
  allowed_transitions: CaseStatus[];
  resolution: { outcome: "confirmed" | "cleared"; reason: string; evidence: string; by: string; at: string } | null;
}

export interface CaseUpdate {
  status?: CaseStatus;
  note?: string;
  reason?: string;
  evidence?: string;
}

interface PipelineRun {
  run_id: string;
  trigger: string;
  actor: string | null;
  started_at: string;
  finished_at: string | null;
  seconds: number | null;
  status: "running" | "succeeded" | "failed";
  error: string | null;
  stages: Array<{ key: string; name: string; seconds: number; detail: string }>;
  summary: {
    customers: number;
    flagged: number;
    expected_thefts_flagged: number | null;
    threshold: number;
    model_version: string;
    population: string | null;
  } | null;
}

interface PipelineConfig {
  run_on_startup: boolean;
  scoring_interval_minutes: number;
  environment: string;
}

export interface Dataset {
  dataset_id: string;
  filename: string;
  uploaded_at: string;
  uploaded_by: string;
  summary: {
    format: "consumption" | "features";
    customers: number;
    days: number | null;
    features_found: number;
    features_expected: number;
    threshold: number;
    flagged: number;
    tiers: Record<Tier, number>;
    mean_probability: number;
    labelled: boolean;
    label_metrics: { theft: number; caught: number; precision: number | null; recall: number | null; roc_auc: number | null; pr_auc: number | null } | null;
    top: Array<{ customer_id: string; probability: number; risk_tier: Tier; label: number | null }>;
  };
}

export type ReportKind = "portfolio" | "dataset" | "case" | "research";

interface Report {
  report_id: string;
  kind: ReportKind;
  title: string;
  subject: string;
  created_at: string;
  created_by: string;
  bytes: number;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const getHealth = () => get<Health>("/health");
export const getMe = () => get<Me>("/me");
export const getAudit = (limit = 50) => get<AuditEntry[]>("/audit", { limit });

export async function promotePopulation(datasetId: string): Promise<Population> {
  return (await api.post<Population>("/population", { dataset_id: datasetId })).data;
}

export async function resetPopulation(): Promise<Population> {
  return (await api.delete<Population>("/population")).data;
}

// Operations: the population in service (no labels); threshold studio on validation customers.
export const getModelMetrics = () => get<ModelMetrics>("/model/metrics");
export const getOperatingCurve = (params: { capacity?: number; cost_per_visit?: number; value_per_theft?: number } = {}) =>
  get<OperatingCurve>("/model/operating-curve", params);
export const getPopulationDistribution = () => get<PopulationDistribution>("/model/score-distribution");
export const getThresholdPreview = (threshold: number) => get<ThresholdPreview>("/model/threshold-preview", { threshold });
export const getModelDrivers = () => get<GlobalDrivers>("/model/drivers");

export async function publishThreshold(threshold: number | null): Promise<ModelMetrics> {
  return (await api.put<ModelMetrics>("/model/threshold", { threshold })).data;
}

// Research: the test customers, scored once.
export const getResearchEvaluation = () => get<ResearchEvaluation>("/research/evaluation");
export const getModelComparison = () => get<ComparisonRow[]>("/research/comparison");
export const getResearchCurve = () => get<ResearchCurve>("/research/operating-curve");
export const getResearchDistribution = () => get<ResearchDistribution>("/research/score-distribution");
export const getCalibration = () => get<CalibrationReport>("/research/calibration");
export const getTrainingSummary = () => get<TrainingSummary>("/research/training");
export const getResamplingEffect = () => get<ResamplingEffect>("/research/resampling");

export const getTimeseries = (customerId: string) => get<TimeSeries>(`/customers/${encodeURIComponent(customerId)}/timeseries`);
export const getExplanation = (customerId: string) => get<Explanation>(`/customers/${encodeURIComponent(customerId)}/explanation`);
export const getExplanationCheck = (customerId: string) =>
  get<ExplanationCheck>(`/customers/${encodeURIComponent(customerId)}/explanation-check`);

export async function predictCustomer(customerId: string): Promise<Prediction> {
  return (await api.post<Prediction>("/predict/single", { customer_id: customerId })).data;
}

export const getCases = (params: { status?: CaseStatus; tier?: Tier; search?: string; page?: number; page_size?: number } = {}) =>
  get<CaseList>("/cases", params);
export const getCase = (customerId: string) => get<CaseDetail>(`/cases/${encodeURIComponent(customerId)}`);

export async function updateCase(customerId: string, payload: CaseUpdate): Promise<CaseDetail> {
  return (await api.patch<CaseDetail>(`/cases/${encodeURIComponent(customerId)}`, payload)).data;
}

export const getPipelineRuns = (limit = 20) => get<PipelineRun[]>("/pipeline/runs", { limit });
export const getPipelineConfig = () => get<PipelineConfig>("/pipeline/config");

export async function startPipelineRun(): Promise<PipelineRun> {
  return (await api.post<PipelineRun>("/pipeline/runs")).data;
}

export const getDatasets = () => get<Dataset[]>("/datasets");

export async function uploadDataset(file: File): Promise<Dataset> {
  const form = new FormData();
  form.append("file", file);
  return (await api.post<Dataset>("/datasets", form)).data;
}

export async function deleteDataset(datasetId: string): Promise<void> {
  await api.delete(`/datasets/${encodeURIComponent(datasetId)}`);
}

export const getReports = () => get<Report[]>("/reports");

export async function deleteReport(reportId: string): Promise<void> {
  await api.delete(`/reports/${encodeURIComponent(reportId)}`);
}

async function createReport(payload: { kind: ReportKind; dataset_id?: string; customer_id?: string }): Promise<Report> {
  return (await api.post<Report>("/reports", payload)).data;
}

function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Downloads a report PDF with the API key header (a plain link cannot send it). */
export async function downloadReport(reportId: string): Promise<void> {
  const response = await api.get<Blob>(`/reports/${encodeURIComponent(reportId)}/pdf`, { responseType: "blob" });
  saveBlob(response.data, `${reportId}.pdf`);
}

/** Generates a report and downloads it straight away. */
export async function createAndDownloadReport(payload: { kind: ReportKind; dataset_id?: string; customer_id?: string }): Promise<Report> {
  const report = await createReport(payload);
  await downloadReport(report.report_id);
  return report;
}

/** Scores a CSV (SGCC meter data or model features) and saves the returned scores. */
export async function downloadBatchScores(file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  const response = await api.post<Blob>("/predict/batch", form, { responseType: "blob" });
  saveBlob(response.data, `scores-${file.name.replace(/\.csv$/i, "")}.csv`);
}

/** A readable message for any failed request, including blob responses and network failures. */
export async function apiErrorMessage(error: unknown, fallback = "Request failed"): Promise<string> {
  if (!(error instanceof AxiosError)) return fallback;
  const response = error.response;
  if (!response) return "The API could not be reached. Is the backend running?";
  if (response.status === 401) return "The API rejected the key. Set it on the Settings page.";
  if (response.status === 403) return "Your role cannot do this; ask a supervisor.";
  if (response.status === 429) return "Too many requests; wait a moment and try again.";
  let data: unknown = response.data;
  if (data instanceof Blob) {
    try {
      data = JSON.parse(await data.text());
    } catch {
      data = null;
    }
  }
  const detail = (data as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) return String((detail[0] as { msg?: string }).msg ?? fallback);
  return `${fallback} (HTTP ${response.status})`;
}
