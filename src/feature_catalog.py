"""
Plain-language names, proposal groups and value formatting for the model's features.

The single source of truth for how a feature is described to people: the API
attaches these labels to explanations, and the thesis figures use them too.

Every feature belongs to one of the proposal's four groups (section 3.8):
statistical, temporal, trend and anomaly. ``CORE_FEATURES`` is the proposal's
25-feature set; the model may use the extended set built by
``src.features.build_features_wide`` (the ablation compares the two).
"""

import math
from datetime import date, timedelta
from typing import Optional

_FEATURES = {
    "mean": ("Average daily use", "kwh", "statistical"),
    "median": ("Median daily use", "kwh", "statistical"),
    "std": ("Consumption volatility", "kwh", "statistical"),
    "min": ("Lowest daily reading", "kwh", "statistical"),
    "max": ("Highest daily reading", "kwh", "statistical"),
    "range": ("Range of daily readings", "kwh", "statistical"),
    "coef_var": ("Relative volatility", "ratio", "statistical"),
    "skewness": ("Skew of readings", "plain", "statistical"),
    "kurtosis": ("Spikiness of readings", "plain", "statistical"),
    "q10_rel": ("Low-use days vs average", "ratio", "statistical"),
    "q25_rel": ("Lower-quartile use vs average", "ratio", "statistical"),
    "q75_rel": ("Upper-quartile use vs average", "ratio", "statistical"),
    "q90_rel": ("High-use days vs average", "ratio", "statistical"),
    "missing_ratio": ("Missing reads", "percent", "anomaly"),
    "missing_ratio_first_third": ("Missing reads, first third", "percent", "anomaly"),
    "missing_ratio_middle_third": ("Missing reads, middle third", "percent", "anomaly"),
    "missing_ratio_last_third": ("Missing reads, last third", "percent", "anomaly"),
    "zero_day_count": ("Zero-consumption days", "count", "anomaly"),
    "zero_ratio": ("Share of zero readings", "percent", "anomaly"),
    "longest_zero_run": ("Longest zero streak", "days", "anomaly"),
    "longest_missing_run": ("Longest reporting gap", "days", "anomaly"),
    "missing_sequences_count": ("Reporting gaps over 3 days", "count", "anomaly"),
    "first_obs_frac": ("First reading position", "position", "anomaly"),
    "last_obs_frac": ("Last reading position", "position", "anomaly"),
    "sudden_drop_count": ("Sudden drops (>50%)", "count", "anomaly"),
    "sudden_drop_rate": ("Sudden-drop rate", "percent", "anomaly"),
    "diff_abs_mean_rel": ("Day-to-day swing", "ratio", "temporal"),
    "autocorr_lag1": ("Day-to-day consistency", "plain", "temporal"),
    "autocorr_lag7": ("Weekly consistency", "plain", "temporal"),
    "slope_full": ("Overall trend", "slope", "trend"),
    "slope_last_30d": ("Trend, last 30 days", "slope", "trend"),
    "slope_last_90d": ("Trend, last 90 days", "slope", "trend"),
    "slope_last_180d": ("Trend, last 180 days", "slope", "trend"),
    "trend_direction_changes": ("Trend reversals (30-day windows)", "count", "trend"),
    "slope_full_rel": ("Overall trend vs average", "ratio", "trend"),
    "last30_vs_mean": ("Last 30 days vs average", "ratio", "trend"),
    "last90_vs_mean": ("Last 90 days vs average", "ratio", "trend"),
    "last180_vs_mean": ("Last 180 days vs average", "ratio", "trend"),
    "first30_vs_mean": ("First 30 days vs average", "ratio", "trend"),
    "first90_vs_mean": ("First 90 days vs average", "ratio", "trend"),
    "first180_vs_mean": ("First 180 days vs average", "ratio", "trend"),
    "weekday_vs_weekend_ratio": ("Weekday vs weekend use", "ratio", "temporal"),
    "peak_day_ratio": ("Share of peak days", "percent", "temporal"),
    "monthly_cv": ("Month-to-month volatility", "ratio", "temporal"),
    "monthly_min_rel": ("Lowest month vs average", "ratio", "temporal"),
    "monthly_max_rel": ("Highest month vs average", "ratio", "temporal"),
    "low_months": ("Months below 20% of average", "count", "anomaly"),
    "max_monthly_drop": ("Largest monthly drop", "percent", "anomaly"),
    "monthly_drop_count": ("Monthly drops over 50%", "count", "anomaly"),
    "changepoint_min_ratio": ("Use after vs before change point", "ratio", "trend"),
    "changepoint_pos": ("Change point position", "position", "trend"),
    "yoy_last_12m": ("Last 12 months vs year before", "ratio", "temporal"),
    "yoy_prev_12m": ("Year-over-year, previous year", "ratio", "temporal"),
}


FEATURE_GROUPS = ("statistical", "temporal", "trend", "anomaly")

CORE_FEATURES = (
    # statistical (6)
    "mean", "std", "coef_var", "max", "skewness", "kurtosis",
    # temporal (6)
    "weekday_vs_weekend_ratio", "autocorr_lag1", "autocorr_lag7", "peak_day_ratio", "monthly_cv", "yoy_last_12m",
    # trend (6), including Appendix A.2: overall, recent 180-day, and direction changes in 30-day windows
    "slope_full", "slope_last_90d", "slope_last_180d", "trend_direction_changes", "last90_vs_mean", "changepoint_min_ratio",
    # anomaly (7)
    "missing_ratio", "longest_missing_run", "zero_day_count", "longest_zero_run", "sudden_drop_count", "low_months",
    "max_monthly_drop",
)


def _lag_months(name: str) -> Optional[int]:
    prefix = "month_lag_"
    if name.startswith(prefix) and name[len(prefix):].isdigit():
        return int(name[len(prefix):])
    return None


def feature_label(name: str) -> str:
    """Human-readable name of a feature."""
    if name in _FEATURES:
        return _FEATURES[name][0]
    months = _lag_months(name)
    if months is not None:
        return "Latest month vs average" if months == 0 else f"Use {months} month{'s' if months != 1 else ''} ago vs average"
    return name.replace("_", " ")


def feature_group(name: str) -> str:
    """The proposal group a feature belongs to; the monthly profile is temporal."""
    if name in _FEATURES:
        return _FEATURES[name][2]
    return "temporal" if _lag_months(name) is not None else "statistical"


def format_feature_value(name: str, value: Optional[float], start: date = date(2014, 1, 1), n_days: int = 1034) -> str:
    """Human-readable value; ``start``/``n_days`` place position features on the calendar."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "no readings"
    kind = _FEATURES.get(name, (None, "ratio" if _lag_months(name) is not None else "plain"))[1]
    if kind == "percent":
        return f"{value * 100:.{1 if 0 < value < 0.01 else 0}f}%"
    if kind == "ratio":
        return f"{value:.2f}×"
    if kind == "days":
        return f"{round(value):,} days"
    if kind == "count":
        return f"{round(value):,}"
    if kind == "kwh":
        return f"{value:.2f} kWh"
    if kind == "slope":
        return f"{value:+.3f} kWh/day"
    if kind == "position":
        day = round(value * n_days)
        return f"day {day:,} · {(start + timedelta(days=day)).strftime('%b %Y')}"
    return f"{value:.3f}"
