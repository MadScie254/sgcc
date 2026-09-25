"""
SGCC Theft Detector - From meter readings to model input

``model_input`` is the one path from a customer-by-day matrix to the rows the
model scores. Training, the API and the scripts all call it with the pipeline
spec that training saved (``artifacts/pipeline.json``), so a customer is always
scored exactly the way the model was trained:

    {"preprocessing": "raw" | "clean",   # clean: src.preprocessing.clean_series first
     "cleaning": {...},                  # clean_series settings
     "impute": {feature: median} | null} # training medians, when the model was trained
                                         # on resampled (hence gap-free) rows
"""

from typing import Any, Dict, Optional

import pandas as pd

from .features import build_features_wide
from .preprocessing import clean_series

RAW_SPEC: Dict[str, Any] = {"preprocessing": "raw", "cleaning": None, "impute": None}


def features_for(wide: pd.DataFrame, preprocessing: str, cleaning: Optional[dict] = None,
                 feature_config: Optional[dict] = None) -> pd.DataFrame:
    """Feature matrix on raw or cleaned readings; missingness features always use the raw gaps."""
    if preprocessing == "raw":
        return build_features_wide(wide, feature_config)
    if preprocessing == "clean":
        cleaned, _ = clean_series(wide, cleaning)
        return build_features_wide(cleaned, feature_config, missing_mask=wide.isna().to_numpy())
    raise ValueError(f"Unknown preprocessing {preprocessing!r}")


def model_input(wide: pd.DataFrame, spec: Optional[Dict[str, Any]] = None,
                feature_config: Optional[dict] = None) -> pd.DataFrame:
    """Rows ready for the model, built the way ``spec`` says the model was trained."""
    spec = {**RAW_SPEC, **(spec or {})}
    X = features_for(wide, spec["preprocessing"], spec["cleaning"], feature_config)
    if spec["impute"]:
        X = X.fillna(pd.Series(spec["impute"], dtype="float32"))
    return X
