// Plain-language names and value formatting for the model's 85 features
// (see src/features.py for definitions).

type Kind = "ratio" | "percent" | "days" | "kwh" | "slope" | "count" | "plain" | "position";

const FEATURES: Record<string, { label: string; kind: Kind }> = {
  mean: { label: "Average daily use", kind: "kwh" },
  median: { label: "Median daily use", kind: "kwh" },
  std: { label: "Consumption volatility", kind: "kwh" },
  min: { label: "Lowest daily reading", kind: "kwh" },
  max: { label: "Highest daily reading", kind: "kwh" },
  range: { label: "Range of daily readings", kind: "kwh" },
  coef_var: { label: "Relative volatility", kind: "ratio" },
  skewness: { label: "Skew of readings", kind: "plain" },
  kurtosis: { label: "Spikiness of readings", kind: "plain" },
  q10_rel: { label: "Low-use days vs average", kind: "ratio" },
  q25_rel: { label: "Lower-quartile use vs average", kind: "ratio" },
  q75_rel: { label: "Upper-quartile use vs average", kind: "ratio" },
  q90_rel: { label: "High-use days vs average", kind: "ratio" },
  missing_ratio: { label: "Missing reads", kind: "percent" },
  missing_ratio_first_third: { label: "Missing reads, first third", kind: "percent" },
  missing_ratio_middle_third: { label: "Missing reads, middle third", kind: "percent" },
  missing_ratio_last_third: { label: "Missing reads, last third", kind: "percent" },
  zero_day_count: { label: "Zero-consumption days", kind: "count" },
  zero_ratio: { label: "Share of zero readings", kind: "percent" },
  longest_zero_run: { label: "Longest zero streak", kind: "days" },
  longest_missing_run: { label: "Longest reporting gap", kind: "days" },
  missing_sequences_count: { label: "Reporting gaps over 3 days", kind: "count" },
  first_obs_frac: { label: "First reading position", kind: "position" },
  last_obs_frac: { label: "Last reading position", kind: "position" },
  sudden_drop_count: { label: "Sudden drops (>50%)", kind: "count" },
  sudden_drop_rate: { label: "Sudden-drop rate", kind: "percent" },
  diff_abs_mean_rel: { label: "Day-to-day swing", kind: "ratio" },
  autocorr_lag1: { label: "Day-to-day consistency", kind: "plain" },
  autocorr_lag7: { label: "Weekly consistency", kind: "plain" },
  slope_full: { label: "Overall trend", kind: "slope" },
  slope_last_30d: { label: "Trend, last 30 days", kind: "slope" },
  slope_last_90d: { label: "Trend, last 90 days", kind: "slope" },
  slope_full_rel: { label: "Overall trend vs average", kind: "ratio" },
  last30_vs_mean: { label: "Last 30 days vs average", kind: "ratio" },
  last90_vs_mean: { label: "Last 90 days vs average", kind: "ratio" },
  last180_vs_mean: { label: "Last 180 days vs average", kind: "ratio" },
  first30_vs_mean: { label: "First 30 days vs average", kind: "ratio" },
  first90_vs_mean: { label: "First 90 days vs average", kind: "ratio" },
  first180_vs_mean: { label: "First 180 days vs average", kind: "ratio" },
  weekday_vs_weekend_ratio: { label: "Weekday vs weekend use", kind: "ratio" },
  peak_day_ratio: { label: "Share of peak days", kind: "percent" },
  monthly_cv: { label: "Month-to-month volatility", kind: "ratio" },
  monthly_min_rel: { label: "Lowest month vs average", kind: "ratio" },
  monthly_max_rel: { label: "Highest month vs average", kind: "ratio" },
  low_months: { label: "Months below 20% of average", kind: "count" },
  max_monthly_drop: { label: "Largest monthly drop", kind: "percent" },
  monthly_drop_count: { label: "Monthly drops over 50%", kind: "count" },
  changepoint_min_ratio: { label: "Use after vs before change point", kind: "ratio" },
  changepoint_pos: { label: "Change point position", kind: "position" },
  yoy_last_12m: { label: "Last 12 months vs year before", kind: "ratio" },
  yoy_prev_12m: { label: "Year-over-year, previous year", kind: "ratio" },
};

const SERIES_DAYS = 1034;
const SERIES_START = Date.UTC(2014, 0, 1);

export function featureLabel(name: string): string {
  const known = FEATURES[name];
  if (known) return known.label;
  const lag = /^month_lag_(\d+)$/.exec(name);
  if (lag) {
    const months = Number(lag[1]);
    return months === 0 ? "Latest month vs average" : `Use ${months} month${months === 1 ? "" : "s"} ago vs average`;
  }
  return name.replace(/_/g, " ");
}

export function featureValue(name: string, value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "no readings";
  const kind = FEATURES[name]?.kind ?? (name.startsWith("month_lag_") ? "ratio" : "plain");
  switch (kind) {
    case "percent":
      return `${(value * 100).toFixed(value < 0.01 && value > 0 ? 1 : 0)}%`;
    case "ratio":
      return `${value.toFixed(2)}×`;
    case "days":
      return `${Math.round(value).toLocaleString("en-US")} days`;
    case "count":
      return Math.round(value).toLocaleString("en-US");
    case "kwh":
      return `${value.toFixed(2)} kWh`;
    case "slope":
      return `${value >= 0 ? "+" : ""}${value.toFixed(3)} kWh/day`;
    case "position": {
      const day = Math.round(value * SERIES_DAYS);
      const date = new Date(SERIES_START + day * 86_400_000);
      return `day ${day.toLocaleString("en-US")} · ${date.toLocaleDateString("en-GB", { month: "short", year: "numeric", timeZone: "UTC" })}`;
    }
    default:
      return value.toFixed(3);
  }
}
