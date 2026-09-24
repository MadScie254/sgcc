from __future__ import annotations

import math
from datetime import date
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.data_loader import load_wide
from src.features import build_features_wide

from .config import get_config, get_paths
from .errors import NotFoundError


def finite_or_none(value: Any) -> Optional[float]:
    """JSON-safe float: NaN/inf (e.g. undefined features) become None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@lru_cache(maxsize=1)
def get_wide_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Customer-by-day consumption of the served (held-out) customers, with labels."""
    return load_wide(str(get_paths()["serving_data"]))


@lru_cache(maxsize=1)
def get_feature_matrix() -> Tuple[pd.DataFrame, pd.Series]:
    wide, labels = get_wide_data()
    X = build_features_wide(wide, get_config().get("features"))
    return X, labels.reindex(X.index).astype(int)


def series_calendar() -> Tuple[date, int]:
    """First day and length of the served series, to place position features on the calendar."""
    wide, _ = get_wide_data()
    first = wide.columns[0]
    start = first.date() if isinstance(first, pd.Timestamp) else date.fromisoformat(get_config()["features"]["start_date"])
    return start, wide.shape[1]


def get_customer_timeseries(customer_id: str) -> Dict[str, Any]:
    wide, labels = get_wide_data()
    if customer_id not in wide.index:
        raise NotFoundError(f"Unknown customer_id: {customer_id}")
    values = wide.loc[customer_id]
    return {
        "customer_id": customer_id,
        "label": int(labels.loc[customer_id]),
        "points": [
            {"date": day.strftime("%Y-%m-%d"), "kwh": finite_or_none(value)}
            for day, value in zip(pd.DatetimeIndex(values.index), values.to_numpy())
        ],
    }
