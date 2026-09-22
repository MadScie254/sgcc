import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "/api",
});

export const apiClient = api;

api.interceptors.request.use((config) => {
  const apiKey = import.meta.env.VITE_API_KEY;
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
  value: number;
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
  feature_values: number[][];
  sample_count: number;
}

export interface LocalShapResponse {
  customer_id: string;
  feature_names: string[];
  feature_values: number[];
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
  mode: "quick" | "full" | "custom";
  config_overrides?: Record<string, unknown>;
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
