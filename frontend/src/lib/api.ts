import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "/api",
});

export const apiClient = api;

const API_KEY_STORAGE_KEY = "sgcc.apiKey";

// The key is entered on the Settings page and kept in this browser only, so it
// never ships inside the built JavaScript. VITE_API_KEY remains a local-dev fallback.
export function getStoredApiKey(): string {
  try {
    return window.localStorage.getItem(API_KEY_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setStoredApiKey(value: string): void {
  try {
    if (value) {
      window.localStorage.setItem(API_KEY_STORAGE_KEY, value);
    } else {
      window.localStorage.removeItem(API_KEY_STORAGE_KEY);
    }
  } catch {
    // Storage unavailable (private mode); the key lasts until reload.
  }
}

api.interceptors.request.use((config) => {
  const apiKey = getStoredApiKey() || import.meta.env.VITE_API_KEY;
  if (apiKey) {
    config.headers = config.headers ?? {};
    config.headers["X-API-Key"] = apiKey;
  }
  return config;
});

function toParams(params: Record<string, string | number | boolean | null | undefined>) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      query.set(key, String(value));
    }
  });
  return query;
}

async function request<T>(method: "get" | "post", url: string, data?: unknown, params?: Record<string, string | number | boolean | null | undefined>): Promise<T> {
  const response = await api.request<T>({ method, url, data, params: params ? toParams(params) : undefined });
  return response.data;
}

export interface HealthResponse {
  status: string;
  service: string;
  model_loaded?: boolean;
  model_version?: string;
}

export interface DatasetSummary {
  dataset_path: string;
  total_customers: number;
  total_rows: number;
  feature_count: number;
  class_distribution: Record<string, number>;
  missing_values: Record<string, number>;
  zero_consumption_pct: number;
}

export interface FeatureDistributionResponse {
  feature: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  bins: number[];
  counts: number[];
}

export interface CorrelationMatrixResponse {
  features: string[];
  matrix: number[][];
}

export interface TimeSeriesPoint {
  day_index: number;
  date?: string | null;
  consumption_kwh: number | null;
  sudden_drop: boolean;
  anomaly_score: number;
}

export interface CustomerTimeseriesResponse {
  customer_id: string;
  label: number | null;
  points: TimeSeriesPoint[];
  summary: Record<string, unknown>;
}

export interface PredictionReason {
  feature: string;
  value: number | null;
  shap_value: number;
}

export interface PredictRequest {
  customer_id?: string;
  features?: Record<string, number>;
  threshold?: number;
}

export interface PredictResponse {
  customer_id?: string | null;
  prediction: number;
  probability: number;
  threshold: number;
  top_reasons: PredictionReason[];
}

export interface ThresholdPreviewResponse {
  threshold: number;
  metrics: Record<string, number>;
  confusion_matrix: Record<string, number>;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface GlobalShapResponse {
  feature_names: string[];
  shap_values: number[][];
  feature_values: Array<Array<number | null>>;
  sample_count: number;
}

export interface LocalShapResponse {
  customer_id: string;
  feature_names: string[];
  feature_values: Array<number | null>;
  shap_values: number[];
  base_value: number;
  probability: number;
  top_reasons: PredictionReason[];
}

export interface CompareResponse {
  xgboost: Record<string, unknown>;
  baselines: Record<string, unknown>;
}

export interface MonitorResponse {
  timestamp?: string | null;
  data_drift: Record<string, unknown>;
  concept_drift: Record<string, unknown>;
  alerts: Array<Record<string, unknown>>;
  recommendations: string[];
}

export interface CustomerItem {
  customer_id: string;
  risk_score: number;
  risk_tier: "high" | "medium" | "low" | string;
  predicted_label: number;
  threshold: number;
  rank: number;
}

export interface CustomersResponse {
  items: CustomerItem[];
  total: number;
  page: number;
  page_size: number;
  search?: string | null;
  sort_by: string;
  sort_dir: string;
  risk_tier?: string | null;
}

export interface ModelMetricsResponse {
  threshold: number;
  trained_threshold?: number | null;
  model_version?: string | null;
  trained_at?: string | null;
  metrics: Record<string, number>;
  support: Record<string, number>;
  confusion_matrix: Record<string, number>;
  customers_monitored: number;
  flagged_today: number;
  current_mean_probability: number;
  base_rate: number;
  risk_tier_distribution: Record<string, number>;
}

export interface ModelConfigResponse {
  feature_groups: Record<string, string[]>;
  feature_parameters: Record<string, unknown>;
  model: Record<string, unknown>;
  preprocessing: Record<string, unknown>;
  evaluation: Record<string, unknown>;
}

export interface SinglePredictionRequest {
  customer_id?: string;
  features?: Record<string, number>;
  threshold?: number;
}

export interface SinglePredictionResponse {
  customer_id?: string | null;
  prediction: number;
  probability: number;
  threshold: number;
  risk_tier?: string | null;
  top_reasons: PredictionReason[];
}

export interface DatasetCatalogItem {
  dataset_id: string;
  original_filename: string;
  stored_filename: string;
  stored_path: string;
  uploaded_at: string;
  rows: number;
  columns: number;
  status: string;
  summary: Record<string, unknown>;
  top_risk_rows: Record<string, unknown>[];
}

export interface DatasetUploadResponse {
  item: DatasetCatalogItem;
}

export interface ReportRequest {
  dataset_id?: string;
  country_code?: string;
  latitude?: number;
  longitude?: number;
}

export interface ReportResponse {
  report_id: string;
  dataset_id?: string | null;
  dataset_label: string;
  generated_at: string;
  model_metrics: Record<string, unknown>;
  dataset_summary: Record<string, unknown>;
  top_risk_rows: Record<string, unknown>[];
  context: Record<string, unknown>;
  pdf_path: string;
  download_url: string;
}

export interface AnalyticsDashboardResponse {
  dataset_summary: DatasetSummary;
  model_metrics: ModelMetricsResponse;
  uploads: DatasetCatalogItem[];
  reports: Record<string, unknown>[];
  context: Record<string, unknown>;
}

export interface TrainingJobCreateRequest {
  mode: "quick" | "full";
  config_overrides?: { n_trials?: number; cv_folds?: number; timeout_seconds?: number; test_size?: number };
}

export interface TrainingJobStatusResponse {
  job_id: string;
  mode: string;
  status: string;
  current_step: string;
  best_score: number | null;
  message: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("get", "/health");
}

export async function getDatasetSummary(): Promise<DatasetSummary> {
  return request<DatasetSummary>("get", "/eda/summary");
}

export async function getFeatureDistribution(feature: string): Promise<FeatureDistributionResponse> {
  return request<FeatureDistributionResponse>("get", "/eda/feature-distributions", undefined, { feature });
}

export async function getCorrelationMatrix(): Promise<CorrelationMatrixResponse> {
  return request<CorrelationMatrixResponse>("get", "/eda/correlation-matrix");
}

export async function getCustomerTimeseries(customerId: string): Promise<CustomerTimeseriesResponse> {
  return request<CustomerTimeseriesResponse>("get", `/customers/${encodeURIComponent(customerId)}/timeseries`);
}

export async function listCustomers(params: {
  search?: string;
  risk_tier?: string;
  sort_by?: string;
  sort_dir?: string;
  page?: number;
  page_size?: number;
}): Promise<CustomersResponse> {
  return request<CustomersResponse>("get", "/customers", undefined, params);
}

export async function getCustomerShap(customerId: string): Promise<LocalShapResponse> {
  return request<LocalShapResponse>("get", `/customers/${encodeURIComponent(customerId)}/shap`);
}

export async function getModelMetrics(): Promise<ModelMetricsResponse> {
  return request<ModelMetricsResponse>("get", "/model/metrics");
}

export async function getModelConfig(): Promise<ModelConfigResponse> {
  return request<ModelConfigResponse>("get", "/model/config");
}

export async function getFeatureImportance(limit = 15): Promise<FeatureImportanceItem[]> {
  return request<FeatureImportanceItem[]>("get", "/model/feature-importance", undefined, { limit });
}

export async function getGlobalShap(sample_count = 200): Promise<GlobalShapResponse> {
  return request<GlobalShapResponse>("get", "/explain/global-shap", undefined, { sample_count });
}

export async function getLocalShap(customerId: string): Promise<LocalShapResponse> {
  return request<LocalShapResponse>("get", `/explain/local-shap/${encodeURIComponent(customerId)}`);
}

export async function getCompareBaselines(): Promise<CompareResponse> {
  return request<CompareResponse>("get", "/compare/baselines");
}

export async function getMonitorDrift(): Promise<MonitorResponse> {
  return request<MonitorResponse>("get", "/monitor/drift");
}

export async function predictSingle(payload: SinglePredictionRequest): Promise<SinglePredictionResponse> {
  return request<SinglePredictionResponse>("post", "/predict/single", payload);
}

export async function thresholdPreview(threshold: number): Promise<ThresholdPreviewResponse> {
  return request<ThresholdPreviewResponse>("get", "/predict/threshold-preview", undefined, { threshold });
}

export async function getAnalyticsDashboard(): Promise<AnalyticsDashboardResponse> {
  return request<AnalyticsDashboardResponse>("get", "/analytics/dashboard");
}

export async function uploadDataset(file: File): Promise<DatasetUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post<DatasetUploadResponse>("/datasets/upload", formData);
  return response.data;
}

export async function generateReport(payload: ReportRequest = {}): Promise<ReportResponse> {
  return request<ReportResponse>("post", "/reports/generate", payload);
}

export async function createTrainingJob(payload: TrainingJobCreateRequest): Promise<TrainingJobStatusResponse> {
  return request<TrainingJobStatusResponse>("post", "/train/jobs", payload);
}

export async function getTrainingJob(jobId: string): Promise<TrainingJobStatusResponse> {
  return request<TrainingJobStatusResponse>("get", `/train/jobs/${encodeURIComponent(jobId)}`);
}

export async function streamTrainingJob(jobId: string): Promise<TrainingJobStatusResponse> {
  return request<TrainingJobStatusResponse>("get", `/train/jobs/${encodeURIComponent(jobId)}/stream`);
}


// ---------------------------------------------------------------------------
// Operations: scoring pipeline, cases, operating threshold
// ---------------------------------------------------------------------------

export interface PipelineStage {
  key: string;
  name: string;
  seconds: number;
  detail: string;
}

export interface PipelineRun {
  run_id: string;
  trigger: "startup" | "manual" | "schedule" | string;
  started_at: string;
  finished_at?: string;
  seconds?: number;
  status: "running" | "succeeded" | "failed" | string;
  error?: string;
  stages: PipelineStage[];
  summary?: {
    customers: number;
    flagged: number;
    tiers: Record<string, number>;
    threshold: number;
    model_version: string;
  };
}

export interface TrainingSummary {
  model_version?: string;
  trained_at?: string;
  quick_mode?: boolean;
  n_trials?: number;
  cv_metric?: string;
  cv_best_score?: number;
  train_customers?: number;
  test_customers?: number;
  n_features?: number;
  auc?: number;
  pr_auc?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  threshold?: number;
  stages: Array<{ name: string; seconds: number }>;
  best_params: Record<string, number>;
}

export type CaseStatus = "new" | "reviewing" | "dispatched" | "confirmed" | "cleared";

export interface CaseItem {
  customer_id: string;
  rank: number;
  risk_score: number;
  risk_tier: "high" | "medium" | "low";
  status: CaseStatus;
  note: string;
  updated_at?: string | null;
  top_driver?: PredictionReason | null;
}

export interface CasesResponse {
  items: CaseItem[];
  total: number;
  page: number;
  page_size: number;
  status_counts: Record<CaseStatus, number>;
  threshold: number;
}

export interface CaseDetail extends CaseItem {
  flagged: boolean;
  population: number;
  history: Array<{ at: string; event: string }>;
}

export interface OperatingPoint {
  threshold: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
}

export interface ScoreDistribution {
  edges: number[];
  honest: number[];
  theft: number[];
  threshold: number;
}

export interface BaselineMetrics {
  auc: number;
  pr_auc: number;
  precision: number;
  recall: number;
  f1: number;
  training_time?: number;
}

export async function getPipelineRuns(limit = 20): Promise<PipelineRun[]> {
  return request<PipelineRun[]>("get", "/pipeline/runs", undefined, { limit });
}

export async function startPipelineRun(): Promise<PipelineRun> {
  return request<PipelineRun>("post", "/pipeline/runs");
}

export async function getTrainingSummary(): Promise<TrainingSummary> {
  return request<TrainingSummary>("get", "/pipeline/training");
}

export async function getCases(params: {
  status?: CaseStatus;
  tier?: "high" | "medium";
  search?: string;
  page?: number;
  page_size?: number;
} = {}): Promise<CasesResponse> {
  return request<CasesResponse>("get", "/cases", undefined, params);
}

export async function getCase(customerId: string): Promise<CaseDetail> {
  return request<CaseDetail>("get", `/cases/${encodeURIComponent(customerId)}`);
}

export async function updateCase(customerId: string, payload: { status?: CaseStatus; note?: string }): Promise<CaseDetail> {
  const response = await api.patch<CaseDetail>(`/cases/${encodeURIComponent(customerId)}`, payload);
  return response.data;
}

export async function getOperatingCurve(): Promise<OperatingPoint[]> {
  return request<OperatingPoint[]>("get", "/model/operating-curve");
}

export async function getScoreDistribution(): Promise<ScoreDistribution> {
  return request<ScoreDistribution>("get", "/model/score-distribution");
}

export async function publishThreshold(threshold: number | null): Promise<ModelMetricsResponse> {
  const response = await api.put<ModelMetricsResponse>("/model/threshold", { threshold });
  return response.data;
}

/** Downloads a report PDF with the API key header (a plain link cannot send it). */
export async function downloadReport(reportId: string): Promise<void> {
  const response = await api.get<Blob>(`/reports/${encodeURIComponent(reportId)}/download`, { responseType: "blob" });
  const url = URL.createObjectURL(response.data);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${reportId}.pdf`;
  link.click();
  URL.revokeObjectURL(url);
}

/** Scores a CSV of feature columns and saves the returned predictions CSV. */
export async function downloadBatchPredictions(file: File): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post<Blob>("/predict/batch", formData, { responseType: "blob" });
  const url = URL.createObjectURL(response.data);
  const link = document.createElement("a");
  link.href = url;
  link.download = "predictions.csv";
  link.click();
  URL.revokeObjectURL(url);
}

export function apiErrorMessage(error: unknown, fallback = "Request failed"): string {
  const response = (error as { response?: { status?: number; data?: { detail?: unknown } } })?.response;
  if (response?.status === 401) return "The API rejected the key. Set it on the Settings page.";
  const detail = response?.data?.detail;
  if (typeof detail === "string") return detail;
  return fallback;
}

export interface PipelineConfig {
  run_on_startup: boolean;
  scoring_interval_minutes: number;
  training_api_enabled: boolean;
  environment: string;
}

export async function getPipelineConfig(): Promise<PipelineConfig> {
  return request<PipelineConfig>("get", "/pipeline/config");
}

export interface ReportIndexEntry {
  report_id: string;
  dataset_id?: string | null;
  dataset_label: string;
  generated_at: string;
}

export async function getLatestReports(): Promise<ReportIndexEntry[]> {
  return request<ReportIndexEntry[]>("get", "/reports/latest");
}

export async function getDatasetCatalog(): Promise<{ items: DatasetCatalogItem[] }> {
  return request<{ items: DatasetCatalogItem[] }>("get", "/datasets/catalog");
}
