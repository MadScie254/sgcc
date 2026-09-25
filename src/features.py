"""
SGCC Theft Detector - Feature Engineering Module

Turns each customer's daily consumption series into a fixed-length feature
vector. ``build_features_wide`` works on the whole customer-by-day matrix at
once with NumPy.

Missing readings are theft signal in SGCC. When the series has been cleaned
(gaps imputed), pass the raw ``missing_mask`` so the missingness features still
describe the customer's real reporting gaps.
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_CONFIG = {
    "sudden_drop_threshold": 0.5,
    "peak_day_percentile": 0.9,
    "missing_sequence_threshold": 3,
    "low_month_ratio": 0.2,
    "monthly_profile_months": 34,
    # Used to place day columns on a calendar when the data has no date header.
    "start_date": "2014-01-01",
}


def _safe_div(num, den, fill=0.0):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, fill, dtype=float)
    np.divide(num, den, out=out, where=(den != 0) & ~np.isnan(den))
    return out


def _nanmean(a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _run_lengths(mask: np.ndarray, min_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per row: the longest run of True, and the number of runs longer than ``min_length``."""
    n = mask.shape[0]
    padded = np.zeros((n, mask.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    edges = np.diff(padded, axis=1)
    rows_s, cols_s = np.nonzero(edges == 1)
    _, cols_e = np.nonzero(edges == -1)
    lengths = cols_e - cols_s  # starts and ends pair up row by row in C order
    longest = np.zeros(n)
    np.maximum.at(longest, rows_s, lengths)
    count = np.bincount(rows_s[lengths > min_length], minlength=n).astype(float)
    return longest, count


def _slopes(values: np.ndarray) -> np.ndarray:
    """Row-wise least-squares slope over column position, ignoring NaN."""
    t = np.arange(values.shape[1], dtype=float)[None, :]
    ok = ~np.isnan(values)
    count = ok.sum(axis=1)
    t_mean = _safe_div(np.where(ok, t, 0).sum(axis=1), count)
    v_mean = _safe_div(np.where(ok, values, 0).sum(axis=1), count)
    dt = np.where(ok, t - t_mean[:, None], 0.0)
    dv = np.where(ok, values - v_mean[:, None], 0.0)
    return _safe_div((dt * dv).sum(axis=1), (dt ** 2).sum(axis=1))


def _autocorr(centered: np.ndarray, lag: int) -> np.ndarray:
    a, b = centered[:, :-lag], centered[:, lag:]
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    return _safe_div((a * b).sum(axis=1), np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)))


def _calendar(columns, start_date: str) -> pd.DatetimeIndex:
    if isinstance(columns, pd.DatetimeIndex):
        return columns
    return pd.date_range(start_date, periods=len(columns), freq="D")


def build_features_wide(wide: pd.DataFrame, config: Optional[dict] = None,
                        missing_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Build the feature matrix from a customer-by-day matrix.

    Args:
        wide: One row per customer, one column per day in chronological order.
            Columns may be a DatetimeIndex; otherwise days are placed on a
            calendar starting at ``config['start_date']``.
        config: Feature parameters; missing keys use DEFAULT_FEATURE_CONFIG.
        missing_mask: Boolean matrix shaped like ``wide`` marking readings that were
            missing before cleaning. Defaults to the NaN cells of ``wide``.

    Returns:
        DataFrame indexed like ``wide``. Undefined values are NaN.
    """
    cfg = {**DEFAULT_FEATURE_CONFIG, **(config or {})}
    W = wide.to_numpy(dtype=np.float64)
    n, d = W.shape
    dates = _calendar(wide.columns, cfg["start_date"])
    present = ~np.isnan(W)
    n_obs = present.sum(axis=1)
    observed = ~missing_mask if missing_mask is not None else present
    f = {}

    # Statistics of observed readings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(W, axis=1)
        std = np.nanstd(W, axis=1)
        f["mean"] = mean
        f["median"] = np.nanmedian(W, axis=1)
        f["std"] = std
        f["min"] = np.nanmin(W, axis=1)
        f["max"] = np.nanmax(W, axis=1)
        quantiles = np.nanpercentile(W, [10, 25, 75, 90, 100 * cfg["peak_day_percentile"]], axis=1)
    f["coef_var"] = _safe_div(std, mean)
    f["range"] = f["max"] - f["min"]
    centered = W - mean[:, None]
    f["skewness"] = _safe_div(_nanmean(centered ** 3), std ** 3)
    f["kurtosis"] = _safe_div(_nanmean(centered ** 4), std ** 4)
    for q, values in zip((10, 25, 75, 90), quantiles[:4]):
        f[f"q{q}_rel"] = _safe_div(values, mean)

    # Missing and zero patterns
    zero = W == 0
    f["missing_ratio"] = 1.0 - observed.sum(axis=1) / d
    thirds = np.array_split(np.arange(d), 3)
    for name, cols in zip(("first", "middle", "last"), thirds):
        f[f"missing_ratio_{name}_third"] = 1.0 - observed[:, cols].mean(axis=1)
    f["zero_day_count"] = zero.sum(axis=1).astype(float)
    f["zero_ratio"] = _safe_div(f["zero_day_count"], n_obs)
    f["longest_zero_run"], _ = _run_lengths(zero, cfg["missing_sequence_threshold"])
    f["longest_missing_run"], f["missing_sequences_count"] = _run_lengths(~observed, cfg["missing_sequence_threshold"])
    any_obs = observed.any(axis=1)
    first_obs = np.where(any_obs, observed.argmax(axis=1), d)
    last_obs = np.where(any_obs, d - 1 - observed[:, ::-1].argmax(axis=1), 0)
    f["first_obs_frac"] = first_obs / d
    f["last_obs_frac"] = last_obs / d

    # Day-over-day behaviour
    prev, cur = W[:, :-1], W[:, 1:]
    comparable = ~np.isnan(prev) & ~np.isnan(cur) & (prev > 0)
    drop = np.where(comparable, _safe_div(prev - cur, prev), 0.0)
    f["sudden_drop_count"] = (drop > cfg["sudden_drop_threshold"]).sum(axis=1).astype(float)
    f["sudden_drop_rate"] = _safe_div(f["sudden_drop_count"], comparable.sum(axis=1))
    f["diff_abs_mean_rel"] = _safe_div(_nanmean(np.abs(cur - prev)), mean)
    f["autocorr_lag1"] = _autocorr(centered, 1)
    f["autocorr_lag7"] = _autocorr(centered, 7)

    # Trends
    f["slope_full"] = _slopes(W)
    f["slope_last_30d"] = _slopes(W[:, -30:])
    f["slope_last_90d"] = _slopes(W[:, -90:])
    f["slope_last_180d"] = _slopes(W[:, -180:])
    # Sign changes between the trends of consecutive 30-day windows (proposal Appendix A.2)
    window_slopes = np.sign(np.column_stack([_slopes(W[:, i:i + 30]) for i in range(0, d - 29, 30)]))
    signed = np.where(np.isnan(window_slopes) | (window_slopes == 0), np.nan, window_slopes)
    signed = pd.DataFrame(signed).ffill(axis=1).to_numpy()  # skip windows without a trend
    f["trend_direction_changes"] = ((signed[:, 1:] != signed[:, :-1]) & ~np.isnan(signed[:, :-1])).sum(axis=1).astype(float)
    f["slope_full_rel"] = _safe_div(f["slope_full"] * d, mean)
    for k in (30, 90, 180):
        f[f"last{k}_vs_mean"] = _safe_div(_nanmean(W[:, -k:]), mean, fill=np.nan)
        f[f"first{k}_vs_mean"] = _safe_div(_nanmean(W[:, :k]), mean, fill=np.nan)

    # Weekly and peak patterns
    weekend = dates.dayofweek.to_numpy() >= 5
    f["weekday_vs_weekend_ratio"] = _safe_div(_nanmean(W[:, ~weekend]), _nanmean(W[:, weekend]), fill=1.0)
    f["peak_day_ratio"] = _safe_div((W >= quantiles[4][:, None]).sum(axis=1), n_obs)

    # Monthly profile
    periods = dates.to_period("M")
    months = periods.unique()
    M = np.column_stack([_nanmean(W[:, periods == p]) for p in months])
    month_mean = _nanmean(M)
    M_rel = M / np.where(month_mean > 0, month_mean, np.nan)[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["monthly_cv"] = _safe_div(np.nanstd(M, axis=1), month_mean)
        f["monthly_min_rel"] = np.nanmin(M_rel, axis=1)
        f["monthly_max_rel"] = np.nanmax(M_rel, axis=1)
    f["low_months"] = (M_rel < cfg["low_month_ratio"]).sum(axis=1).astype(float)
    m_prev, m_cur = M[:, :-1], M[:, 1:]
    m_drop = np.where((m_prev > 0) & ~np.isnan(m_cur), _safe_div(m_prev - m_cur, m_prev), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["max_monthly_drop"] = np.nanmax(m_drop, axis=1)
    f["monthly_drop_count"] = (m_drop > cfg["sudden_drop_threshold"]).sum(axis=1).astype(float)

    # Best single change point in the monthly series: mean after / mean before
    month_obs = ~np.isnan(M)
    csum = np.nancumsum(M, axis=1)
    ccount = np.cumsum(month_obs, axis=1)
    before = _safe_div(csum[:, :-1], ccount[:, :-1], fill=np.nan)
    after = _safe_div(csum[:, -1:] - csum[:, :-1], ccount[:, -1:] - ccount[:, :-1], fill=np.nan)
    ratio = np.where(before > 0, _safe_div(after, before, fill=np.nan), np.nan)
    has_ratio = ~np.isnan(ratio).all(axis=1)
    safe_ratio = np.where(np.isnan(ratio), np.inf, ratio)
    f["changepoint_min_ratio"] = np.where(has_ratio, safe_ratio.min(axis=1), np.nan)
    f["changepoint_pos"] = np.where(has_ratio, safe_ratio.argmin(axis=1) / max(ratio.shape[1], 1), np.nan)

    # Year-over-year on trailing 12-month windows
    last12, prev12, prev24 = (_nanmean(M[:, max(len(months) - e, 0):len(months) - s]) if len(months) > s else np.full(n, np.nan)
                              for s, e in ((0, 12), (12, 24), (24, 36)))
    f["yoy_last_12m"] = _safe_div(last12, prev12, fill=np.nan)
    f["yoy_prev_12m"] = _safe_div(prev12, prev24, fill=np.nan)

    # Relative monthly profile, most recent month first
    profile = int(cfg["monthly_profile_months"])
    for lag in range(profile):
        idx = len(months) - 1 - lag
        f[f"month_lag_{lag:02d}"] = M_rel[:, idx] if idx >= 0 else np.full(n, np.nan)

    X = pd.DataFrame(f, index=wide.index)
    return X.replace([np.inf, -np.inf], np.nan).astype("float32")

