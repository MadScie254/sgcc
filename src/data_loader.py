"""
SGCC Theft Detector - Data Loader Module

Loads the SGCC smart-meter dataset (one row per customer, one column per day)
into a chronologically ordered wide matrix, and converts it to long format for
the API's per-customer time-series views.
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

# Extra metadata columns written by older augmentation scripts.
_METADATA_COLUMNS = {
    "IS_SYNTHETIC", "CUSTOMER_TYPE", "THEFT_TYPE", "MEAN_MONTHLY_CONSUMPTION",
    "STD_MONTHLY_CONSUMPTION", "MAX_MONTHLY_CONSUMPTION", "MIN_MONTHLY_CONSUMPTION",
    "MEDIAN_MONTHLY_CONSUMPTION", "CONSUMPTION_TREND", "COEFFICIENT_OF_VARIATION",
    "MAX_CONSUMPTION_DROP", "MONTHS_WITH_ZERO", "MONTHS_WITH_LOW_CONSUMPTION",
    "RECENT_VS_HISTORICAL_RATIO", "QUARTERLY_STD",
}

_DATE_LIKE = re.compile(r"^\d{1,4}[/-]\d{1,2}[/-]\d{1,4}$")


def _find_column(columns, candidates) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    return None


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


def load_wide(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the SGCC dataset as a wide customer-by-day matrix.

    Supported layouts:
    - SGCC original: ``CONS_NO, FLAG, <date columns>`` (date columns in any order)
    - Named id/label columns anywhere (``CUSTOMER_ID``/``customer_id``, ``FLAG``/``label``)
    - Positional: day columns first, then customer id, then label

    Date-named day columns are sorted chronologically; the raw SGCC file stores
    them in lexicographic order ("2014/1/1", "2014/1/10", ...), which scrambles
    any order-dependent feature if left as is.

    Returns:
        (wide, labels): ``wide`` is float32, indexed by customer id (str), with
        one column per day (DatetimeIndex when the header holds dates).
        ``labels`` is an int Series named ``label`` on the same index.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading data from %s...", path)
    df = pd.read_csv(file_path, low_memory=False)

    id_col = _find_column(df.columns, ID_COLUMNS)
    label_col = _find_column(df.columns, LABEL_COLUMNS)
    if id_col is None or label_col is None:
        if df.shape[1] < 3:
            raise ValueError("Expected day columns plus customer id and label columns")
        id_col, label_col = df.columns[-2], df.columns[-1]

    customer_ids = df[id_col].astype(str).str.strip()
    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"Label column {label_col!r} must be binary 0/1")

    day_columns = [c for c in df.columns if c not in {id_col, label_col} and c not in _METADATA_COLUMNS]
    values = df[day_columns].apply(pd.to_numeric, errors="coerce")

    dates = _parse_dates(day_columns)
    if dates is not None:
        order = np.argsort(dates.values, kind="stable")
        values = values.iloc[:, order]
        values.columns = dates[order]
    else:
        values.columns = range(len(day_columns))

    wide = values.astype("float32")
    wide.index = pd.Index(customer_ids.values, name="customer_id")

    # The public SGCC dump contains a handful of duplicated customer rows.
    duplicated = wide.index.duplicated(keep="first")
    if duplicated.any():
        logger.warning("Dropping %d duplicated customer rows", int(duplicated.sum()))
    wide = wide.loc[~duplicated]
    label_series = pd.Series(labels.values[~duplicated], index=wide.index, name="label")

    logger.info(
        "Loaded %d customers x %d days (%.1f%% missing, %.1f%% theft)",
        wide.shape[0], wide.shape[1], 100 * float(wide.isna().to_numpy().mean()),
        100 * float(label_series.mean()),
    )
    return wide, label_series


def wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide matrix to long format [customer_id, day_index, consumption_kwh]."""
    n_customers, n_days = wide.shape
    return pd.DataFrame({
        "customer_id": np.repeat(wide.index.astype(str).to_numpy(), n_days),
        "day_index": np.tile(np.arange(n_days, dtype=np.int32), n_customers),
        "consumption_kwh": wide.to_numpy(dtype=np.float64).ravel(),
    })


def long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long format back to a wide matrix ordered by day_index, keeping customer order."""
    customer_order = pd.unique(df_long["customer_id"])
    wide = df_long.pivot_table(
        index="customer_id", columns="day_index", values="consumption_kwh",
        aggfunc="first", dropna=False,
    )
    wide = wide.reindex(index=customer_order).sort_index(axis=1)
    return wide.astype("float32")


def load_raw(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the dataset in long format.

    Returns:
        (df_long, labels): ``df_long`` has columns [customer_id, day_index,
        consumption_kwh] with day_index in chronological order; ``labels`` is
        indexed by customer id.
    """
    wide, labels = load_wide(path)
    return wide_to_long(wide), labels


def load_processed_features(path: str = "artifacts/features.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """Load pre-computed features (index = customer id) and labels from CSV."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Features file not found: {path}")

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.index.name = None
    if "label" not in df.columns:
        raise ValueError("Features file must contain 'label' column")

    y = df["label"].astype("int32")
    X = df.drop(columns="label")
    return X, y


def save_processed_features(X: pd.DataFrame, y: pd.Series, path: str = "artifacts/features.csv") -> None:
    """Save features and labels to CSV, keeping the customer id index."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = X.copy()
    df["label"] = y
    df.to_csv(path, index=True)
    logger.info("Saved features to %s", path)
