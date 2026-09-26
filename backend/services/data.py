"""Meter readings the API scores, and how the served model sees a customer.

The operational population is the committed, unlabelled sample written by
training (``data/sgcc_demo.csv.gz``) until a supervisor promotes an uploaded
meter-data file (``settings.population``). It never carries labels: the only
labelled data the API reads are the saved validation and test predictions
(``artifacts/predictions/``), which the research views and the threshold studio use.
"""

from __future__ import annotations

import io
import json
import math
from datetime import date
from functools import lru_cache
from typing import Any, Dict, Optional

import pandas as pd

from src.data_loader import frame_to_wide, load_wide
from src.pipeline import model_input

from . import db
from .blobstore import get_blobstore
from .config import get_paths
from .errors import NotFoundError


class ModelUnavailable(RuntimeError):
    """The published model files are missing or do not match their manifest; scoring is refused."""


def finite_or_none(value: Any) -> Optional[float]:
    """JSON-safe float: NaN/inf (e.g. undefined features) become None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@lru_cache(maxsize=1)
def get_pipeline_spec() -> Dict[str, Any]:
    """How the served model was trained to see a customer: frozen by training in artifacts/pipeline.json."""
    path = get_paths()["artifacts"] / "pipeline.json"
    if not path.is_file():
        raise ModelUnavailable("artifacts/pipeline.json is missing: run python -m src.train")
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_input(wide: pd.DataFrame) -> pd.DataFrame:
    """Readings to model rows with the feature settings saved at training (not today's config.yaml)."""
    spec = get_pipeline_spec()
    return model_input(wide, spec, spec["feature_config"])


# ---------------------------------------------------------------------------
# Operational population
# ---------------------------------------------------------------------------

SAMPLE_POPULATION = "sample"


def active_population() -> Dict[str, Any]:
    """Which readings are scored now: the committed sample, or a promoted upload."""
    setting = db.get_setting("population")
    if setting is None:
        return {"id": SAMPLE_POPULATION, "source": "sample", "filename": get_paths()["serving_data"].name,
                "blob_key": None, "promoted_at": None, "promoted_by": None}
    value = setting["value"]
    return {"id": value["dataset_id"], "source": "upload", "filename": value["filename"], "blob_key": value["blob_key"],
            "promoted_at": setting["updated_at"], "promoted_by": setting["updated_by"]}


@lru_cache(maxsize=2)
def _population_readings(population_id: str, blob_key: Optional[str]) -> pd.DataFrame:
    if blob_key is None:
        wide, _ = load_wide(str(get_paths()["serving_data"]), require_labels=False)
    else:
        # Any label column in an upload is ignored: operations never sees labels.
        wide, _ = frame_to_wide(pd.read_csv(io.BytesIO(get_blobstore().get(blob_key)), low_memory=False), require_labels=False)
    return wide


@lru_cache(maxsize=2)
def _population_features(population_id: str, blob_key: Optional[str]) -> pd.DataFrame:
    return build_model_input(_population_readings(population_id, blob_key))


def get_wide_data() -> pd.DataFrame:
    """Customer-by-day readings of the operational population."""
    population = active_population()
    return _population_readings(population["id"], population["blob_key"])


def get_feature_matrix() -> pd.DataFrame:
    population = active_population()
    return _population_features(population["id"], population["blob_key"])


def series_calendar() -> tuple[date, int]:
    """First day and length of the served series, to place position features on the calendar."""
    wide = get_wide_data()
    first = wide.columns[0]
    start = first.date() if isinstance(first, pd.Timestamp) else date.fromisoformat(get_pipeline_spec()["feature_config"]["start_date"])
    return start, wide.shape[1]


def get_customer_timeseries(customer_id: str) -> Dict[str, Any]:
    wide = get_wide_data()
    if customer_id not in wide.index:
        raise NotFoundError(f"Unknown customer_id: {customer_id}")
    values = wide.loc[customer_id]
    return {
        "customer_id": customer_id,
        "points": [
            {"date": day.strftime("%Y-%m-%d"), "kwh": finite_or_none(value)}
            for day, value in zip(pd.DatetimeIndex(values.index), values.to_numpy())
        ],
    }


# ---------------------------------------------------------------------------
# Saved evaluation predictions (labelled; research and threshold choice only)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def load_predictions(split: str) -> pd.DataFrame:
    """Every pipeline's raw score and calibrated probability for the validation or test customers."""
    if split not in ("validation", "test"):
        raise ValueError(f"Unknown split {split!r}")
    path = get_paths()["artifacts"] / "predictions" / f"{split}.csv.gz"
    if not path.is_file():
        raise ModelUnavailable(f"artifacts/predictions/{split}.csv.gz is missing: run python -m src.train")
    return pd.read_csv(path, dtype={"customer_id": str})


CACHED = (get_pipeline_spec, _population_readings, _population_features, load_predictions)
