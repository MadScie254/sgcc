"""
SGCC Theft Detector - Series cleaning (proposal section 3.7)

Cleans each customer's daily series without looking at any other customer or at
the labels, so the same function serves training and scoring:

1. Negative readings are replaced by the customer's median reading.
2. Readings above ``cap_kwh`` (physically implausible for a household) are capped.
3. Readings more than ``outlier_sigma`` standard deviations from the customer's
   mean are replaced by the customer's median.
4. Gaps of at most ``short_gap_days`` days between two readings are linearly
   interpolated; every other gap is forward-filled, then back-filled.
5. Customers with no readings at all are set to zero.

Zero runs longer than ``zero_run_days`` are counted for review, not changed:
a long run of zeros can be the footprint of tampering.

Missingness itself is theft signal in SGCC, so the feature builder takes the
raw missing mask separately (see ``src.pipeline.model_input``).
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_CLEANING_CONFIG = {
    "short_gap_days": 3,
    "cap_kwh": 10_000.0,
    "outlier_sigma": 3.0,
    "zero_run_days": 30,
}


def _runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row, start and end (exclusive) of every run of True in each row."""
    n, d = mask.shape
    padded = np.zeros((n, d + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    edges = np.diff(padded, axis=1)
    rows, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)  # pairs with starts row by row in C order
    return rows, starts, ends


def clean_series(wide: pd.DataFrame, config: Optional[dict] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean a customer-by-day matrix.

    Returns:
        (cleaned matrix with no NaN, log of how many cells or customers each rule touched)
    """
    cfg = {**DEFAULT_CLEANING_CONFIG, **(config or {})}
    W = wide.to_numpy(dtype=np.float64, copy=True)
    n, d = W.shape
    missing = np.isnan(W)
    log: Dict[str, int] = {"customers": n, "days": d, "cells": n * d, "missing_cells": int(missing.sum())}

    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows: handled in step 5
        median = np.nan_to_num(np.nanmedian(np.where(W >= 0, W, np.nan), axis=1))

    negative = W < 0
    W[negative] = np.broadcast_to(median[:, None], W.shape)[negative]
    log["negative_replaced"] = int(negative.sum())

    above = W > cfg["cap_kwh"]
    W[above] = cfg["cap_kwh"]
    log["capped_above_limit"] = int(above.sum())

    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(W, axis=1, keepdims=True)
        std = np.nanstd(W, axis=1, keepdims=True)
        outlier = (np.abs(W - mean) > cfg["outlier_sigma"] * std) & (std > 0)
    W[outlier] = np.broadcast_to(median[:, None], W.shape)[outlier]
    log["outliers_replaced"] = int(outlier.sum())

    rows, starts, ends = _runs(missing)
    lengths = ends - starts
    short = (lengths <= cfg["short_gap_days"]) & (starts > 0) & (ends < d)
    r, s, e, length = rows[short], starts[short], ends[short], lengths[short]
    if len(r):
        cell_run = np.repeat(np.arange(len(r)), length)
        offset = np.arange(len(cell_run)) - np.repeat(np.cumsum(length) - length, length)
        cols = s[cell_run] + offset
        before, after = W[r, s - 1], W[r, e]
        step = (offset + 1) / (length[cell_run] + 1)
        W[r[cell_run], cols] = before[cell_run] + (after[cell_run] - before[cell_run]) * step
    log["short_gap_cells_interpolated"] = int(length.sum())

    remaining = np.isnan(W)
    empty = remaining.all(axis=1)
    filled = pd.DataFrame(W).ffill(axis=1).bfill(axis=1).to_numpy()
    filled[empty] = 0.0
    log["long_gap_cells_filled"] = int(remaining[~empty].sum())
    log["customers_without_readings"] = int(empty.sum())

    zero_rows, zero_starts, zero_ends = _runs(filled == 0)
    long_zero = zero_ends - zero_starts > cfg["zero_run_days"]
    log["customers_with_long_zero_runs"] = int(len(np.unique(zero_rows[long_zero])))

    return pd.DataFrame(filled, index=wide.index, columns=wide.columns), log
