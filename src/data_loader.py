"""
SGCC Theft Detector - Data Loader Module

Reads smart-meter data in the SGCC layout (one row per customer, one column per
day) into a chronologically ordered customer-by-day matrix.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ID_COLUMNS = ("CONS_NO", "CUSTOMER_ID", "customer_id")
LABEL_COLUMNS = ("FLAG", "label")
MIN_DAYS = 30

_DATE_LIKE = re.compile(r"^\d{1,4}[/-]\d{1,2}[/-]\d{1,4}$")


def _find_column(columns, candidates) -> Optional[str]:
    return next((name for name in candidates if name in columns), None)


def _parse_dates(columns) -> Optional[pd.DatetimeIndex]:
    """Parse day columns as dates, or return None if they are not all dates."""
    names = [str(c).strip() for c in columns]
    if not names or not all(_DATE_LIKE.match(n) for n in names):
        return None
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        parsed = pd.to_datetime(names, format=fmt, errors="coerce")
        if not parsed.isna().any():
            return pd.DatetimeIndex(parsed)
    return None


def is_consumption_frame(df: pd.DataFrame) -> bool:
    """True if the frame looks like SGCC-layout meter data: an id column and date-named day columns."""
    id_col = _find_column(df.columns, ID_COLUMNS)
    label_col = _find_column(df.columns, LABEL_COLUMNS)
    day_columns = [c for c in df.columns if c not in {id_col, label_col}]
    return id_col is not None and len(day_columns) >= MIN_DAYS and _parse_dates(day_columns) is not None


def frame_to_wide(df: pd.DataFrame, require_labels: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Convert an SGCC-layout frame to a customer-by-day matrix.

    Expects a customer id column (``CONS_NO``, ``CUSTOMER_ID`` or ``customer_id``),
    an optional 0/1 label column (``FLAG`` or ``label``), and one column per day.
    Date-named day columns are sorted chronologically: the raw SGCC file stores
    them lexicographically ("2014/1/1", "2014/1/10", ...).

    Returns:
        (wide, labels): ``wide`` is float32, indexed by customer id (str), one
        column per day (a DatetimeIndex when the header holds dates). ``labels``
        is an int Series named ``label`` on the same index, or None when the
        frame has no label column and ``require_labels`` is False.
    """
    id_col = _find_column(df.columns, ID_COLUMNS)
    if id_col is None:
        raise ValueError(f"No customer id column; expected one of {', '.join(ID_COLUMNS)}")
    label_col = _find_column(df.columns, LABEL_COLUMNS)
    if label_col is None and require_labels:
        raise ValueError(f"No label column; expected one of {', '.join(LABEL_COLUMNS)}")

    day_columns = [c for c in df.columns if c not in {id_col, label_col}]
    if len(day_columns) < MIN_DAYS:
        raise ValueError(f"Expected at least {MIN_DAYS} daily reading columns, found {len(day_columns)}")

    values = df[day_columns].apply(pd.to_numeric, errors="coerce")
    dates = _parse_dates(day_columns)
    if dates is not None:
        order = np.argsort(dates.values, kind="stable")
        values = values.iloc[:, order]
        values.columns = dates[order]
    else:
        values.columns = range(len(day_columns))

    wide = values.astype("float32")
    wide.index = pd.Index(df[id_col].astype(str).str.strip().to_numpy(), name="customer_id")

    # The public SGCC dump contains a handful of duplicated customer rows.
    keep = ~wide.index.duplicated(keep="first")
    if not keep.all():
        logger.warning("Dropping %d duplicated customer rows", int((~keep).sum()))
    wide = wide.loc[keep]

    labels = None
    if label_col is not None:
        raw = pd.to_numeric(df[label_col], errors="coerce").to_numpy()[keep]
        if np.isnan(raw).any() or not set(np.unique(raw)).issubset({0, 1}):
            raise ValueError(f"Label column {label_col!r} must contain only 0 and 1")
        labels = pd.Series(raw.astype(int), index=wide.index, name="label")
    return wide, labels


def load_wide(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a labelled SGCC-layout CSV (plain or gzip) as (wide, labels); see ``frame_to_wide``."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    logger.info("Loading data from %s...", path)
    wide, labels = frame_to_wide(pd.read_csv(file_path, low_memory=False))
    logger.info(
        "Loaded %d customers x %d days (%.1f%% missing, %.1f%% theft)",
        wide.shape[0], wide.shape[1], 100 * float(wide.isna().to_numpy().mean()), 100 * float(labels.mean()),
    )
    return wide, labels
