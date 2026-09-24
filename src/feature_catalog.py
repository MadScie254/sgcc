"""
Plain-language names and value formatting for the model's features.

The single source of truth for how a feature is described to people: the API
attaches these labels to explanations, and the thesis figures use them too.
"""

import math
from datetime import date, timedelta
from typing import Optional

_FEATURES = {
    "mean": ("Average daily use", "kwh"),
    "median": ("Median daily use", "kwh"),
    "std": ("Consumption volatility", "kwh"),
    "min": ("Lowest daily reading", "kwh"),
    "max": ("Highest daily reading", "kwh"),
    "range": ("Range of daily readings", "kwh"),
    "coef_var": ("Relative volatility", "ratio"),
    "skewness": ("Skew of readings", "plain"),
    "kurtosis": ("Spikiness of readings", "plain"),
    "q10_rel": ("Low-use days vs average", "ratio"),
    "q25_rel": ("Lower-quartile use vs average", "ratio"),
    "q75_rel": ("Upper-quartile use vs average", "ratio"),
    "q90_rel": ("High-use days vs average", "ratio"),
    "missing_ratio": ("Missing reads", "percent"),
    "missing_ratio_first_third": ("Missing reads, first third", "percent"),
    "missing_ratio_middle_third": ("Missing reads, middle third", "percent"),
    "missing_ratio_last_third": ("Missing reads, last third", "percent"),
    "zero_day_count": ("Zero-consumption days", "count"),
    "zero_ratio": ("Share of zero readings", "percent"),
    "longest_zero_run": ("Longest zero streak", "days"),
    "longest_missing_run": ("Longest reporting gap", "days"),
    "missing_sequences_count": ("Reporting gaps over 3 days", "count"),
    "first_obs_frac": ("First reading position", "position"),
    "last_obs_frac": ("Last reading position", "position"),
    "sudden_drop_count": ("Sudden drops (>50%)", "count"),
    "sudden_drop_rate": ("Sudden-drop rate", "percent"),
    "diff_abs_mean_rel": ("Day-to-day swing", "ratio"),
    "autocorr_lag1": ("Day-to-day consistency", "plain"),
    "autocorr_lag7": ("Weekly consistency", "plain"),
    "slope_full": ("Overall trend", "slope"),
    "slope_last_30d": ("Trend, last 30 days", "slope"),
    "slope_last_90d": ("Trend, last 90 days", "slope"),
    "slope_full_rel": ("Overall trend vs average", "ratio"),
    "last30_vs_mean": ("Last 30 days vs average", "ratio"),
    "last90_vs_mean": ("Last 90 days vs average", "ratio"),
    "last180_vs_mean": ("Last 180 days vs average", "ratio"),
    "first30_vs_mean": ("First 30 days vs average", "ratio"),
    "first90_vs_mean": ("First 90 days vs average", "ratio"),
    "first180_vs_mean": ("First 180 days vs average", "ratio"),
    "weekday_vs_weekend_ratio": ("Weekday vs weekend use", "ratio"),
    "peak_day_ratio": ("Share of peak days", "percent"),
    "monthly_cv": ("Month-to-month volatility", "ratio"),
    "monthly_min_rel": ("Lowest month vs average", "ratio"),
    "monthly_max_rel": ("Highest month vs average", "ratio"),
    "low_months": ("Months below 20% of average", "count"),
    "max_monthly_drop": ("Largest monthly drop", "percent"),
    "monthly_drop_count": ("Monthly drops over 50%", "count"),
    "changepoint_min_ratio": ("Use after vs before change point", "ratio"),
    "changepoint_pos": ("Change point position", "position"),
    "yoy_last_12m": ("Last 12 months vs year before", "ratio"),
    "yoy_prev_12m": ("Year-over-year, previous year", "ratio"),
}


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
