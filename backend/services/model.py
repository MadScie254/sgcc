"""The served model in operations: scores, thresholds, explanations.

Nothing here reads a label. Scores are Platt-calibrated probabilities (fitted on
validation customers, frozen in artifacts/pipeline.json). A single-model pipeline is
explained by SHAP on its raw score (log-odds), of which the calibrated probability is a
monotone rescaling. The hybrid pipeline blends two calibrated parts, XGBoost on the
engineered features and the sequence model (a CNN on the daily readings, served as ONNX),
and recalibrates the blend; SHAP explains the XGBoost part, and the sequence part is
explained week by week (``sequence_explanation``).
The threshold studio's precision and recall come from the saved validation
predictions, the customers thresholds are chosen on, never the test set.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from src import sequence
from src.calibration import apply_platt
from src.feature_catalog import feature_label, format_feature_value
from src.modeling import load_model
from src.publish import verify_manifest

from . import data as data_service
from . import db
from .config import BASE_DIR, get_paths
from .data import (
    ModelUnavailable, active_population, finite_or_none, get_feature_matrix, get_pipeline_spec, get_wide_data, series_calendar,
)
from .errors import NotFoundError

# Calibrated probabilities at or above this are "high" risk (theft more likely than not); between
# the decision threshold and this they are "medium"; below the threshold "low".
HIGH_RISK_PROBABILITY = 0.5


class FeatureInputError(ValueError):
    """Feature rows do not carry every feature the model was trained on, or the model needs the readings."""


NEEDS_READINGS = ("The model in service (the hybrid) also reads each customer's daily readings, so rows of "
                  "features alone cannot be scored. Upload SGCC meter data, or score a customer of the population by id.")
SEQUENCE = "wide_deep_cnn"


# ---------------------------------------------------------------------------
# Model, integrity and threshold
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def integrity_problems() -> List[str]:
    """Published files that are missing or changed since training, or a model/spec feature mismatch."""
    problems = verify_manifest(BASE_DIR)
    if problems:
        return problems
    spec = get_pipeline_spec()
    names = list(load_model(str(get_paths()["model_file"])).get_booster().feature_names or [])
    if names != list(spec["features"]):
        problems.append("The model's features differ from the feature list in artifacts/pipeline.json")
    if is_hybrid():
        try:
            sequence.OnnxNetwork(BASE_DIR / spec["components"][SEQUENCE]["file"])
        except Exception as exc:  # noqa: BLE001 - any failure to load disables scoring
            problems.append(f"The sequence model cannot be loaded: {exc}")
    return problems


def is_hybrid() -> bool:
    """Whether the served pipeline blends XGBoost with the sequence model."""
    return "components" in get_pipeline_spec()


@lru_cache(maxsize=1)
def get_sequence_network() -> sequence.OnnxNetwork:
    problems = integrity_problems()
    if problems:
        raise ModelUnavailable("Scoring is disabled: " + "; ".join(problems))
    return sequence.OnnxNetwork(BASE_DIR / get_pipeline_spec()["components"][SEQUENCE]["file"])


@lru_cache(maxsize=1)
def get_trained_model():
    problems = integrity_problems()
    if problems:
        raise ModelUnavailable("Scoring is disabled: " + "; ".join(problems))
    return load_model(str(get_paths()["model_file"]))


def get_feature_names() -> List[str]:
    return list(get_pipeline_spec()["features"])


def get_trained_threshold() -> float:
    """Threshold chosen during training: max F1 on calibrated validation probabilities."""
    return float(get_pipeline_spec()["threshold"])


def get_decision_threshold() -> float:
    """Threshold in service: the published operating threshold, else the trained one."""
    setting = db.get_setting("threshold")
    return float(setting["value"]["threshold"]) if setting else get_trained_threshold()


def set_operating_threshold(threshold: Optional[float], actor: str) -> None:
    """Publish an operating threshold, or None to return to the trained one."""
    db.set_setting("threshold", None if threshold is None else {"threshold": float(threshold)}, actor)


def risk_tier(probability: float, threshold: float) -> str:
    if probability >= max(HIGH_RISK_PROBABILITY, threshold):
        return "high"
    if probability >= threshold:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _aligned(features: pd.DataFrame) -> pd.DataFrame:
    aligned = features.reindex(columns=get_feature_names()).apply(pd.to_numeric, errors="coerce")
    impute = get_pipeline_spec().get("impute")
    if impute:  # the model was trained on gap-free (resampled) rows
        aligned = aligned.fillna(pd.Series(impute, dtype=float))
    return aligned


def raw_scores(features: pd.DataFrame) -> np.ndarray:
    """The XGBoost model's own output (what SHAP explains)."""
    return np.asarray(get_trained_model().predict_proba(_aligned(features))[:, 1], dtype=float)


def tree_probability(raw: np.ndarray) -> np.ndarray:
    """Calibrated probability of the XGBoost model alone (the whole model when it is not a hybrid)."""
    spec = get_pipeline_spec()
    return apply_platt(raw, spec["components"]["xgboost"]["calibration"] if is_hybrid() else spec["calibration"])


def sequence_probability(wide: pd.DataFrame) -> tuple:
    """(raw score, calibrated probability) of the sequence model for customers' daily readings."""
    part = get_pipeline_spec()["components"][SEQUENCE]
    values, mask = sequence.prepare(wide, part.get("cleaning"))
    raw = get_sequence_network().predict(values, mask)
    return raw, apply_platt(raw, part["calibration"])


def score(features: pd.DataFrame, wide: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calibrated theft probability of each row, with its parts: ``tree`` (XGBoost) and, for the hybrid,
    ``sequence``. The hybrid needs the customers' daily readings (``wide``, same index as ``features``).
    """
    raw = raw_scores(features)
    tree = tree_probability(raw)
    frame = pd.DataFrame({"raw_score": raw, "tree": tree}, index=features.index)
    if not is_hybrid():
        frame["probability"] = tree
        return frame
    if wide is None:
        raise FeatureInputError(NEEDS_READINGS)
    spec = get_pipeline_spec()
    _, frame["sequence"] = sequence_probability(wide.loc[features.index])
    weight = float(spec["blend"]["weight_sequence"])
    frame["probability"] = apply_platt(weight * frame["sequence"].to_numpy() + (1 - weight) * tree, spec["calibration"])
    return frame


def predict_proba(features: pd.DataFrame, wide: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Calibrated theft probabilities; non-numeric feature values count as missing."""
    return score(features, wide)["probability"].to_numpy(dtype=float)


def blend_parts(scored: pd.Series) -> Optional[List[Dict[str, Any]]]:
    """The hybrid's two parts for one scored customer: label, calibrated probability and weight."""
    if not is_hybrid():
        return None
    spec = get_pipeline_spec()
    weight = float(spec["blend"]["weight_sequence"])
    parts = spec["components"]
    return [
        {"name": "xgboost", "label": parts["xgboost"]["label"], "probability": float(scored["tree"]), "weight": 1 - weight},
        {"name": SEQUENCE, "label": parts[SEQUENCE]["label"], "probability": float(scored["sequence"]), "weight": weight},
    ]


@lru_cache(maxsize=2)
def _population_scores(population_id: str) -> pd.DataFrame:
    X = get_feature_matrix()
    frame = score(X, get_wide_data() if is_hybrid() else None)
    frame.index = frame.index.astype(str)
    return frame.sort_values("probability", ascending=False, kind="mergesort")


def get_population_scores() -> pd.DataFrame:
    """Each customer's calibrated probability (and the hybrid's parts), highest first."""
    return _population_scores(active_population()["id"])


def get_population_probabilities() -> pd.Series:
    """Calibrated theft probability of every customer in the operational population, highest first."""
    return get_population_scores()["probability"]


def get_model_metrics() -> Dict[str, Any]:
    """The operational picture: how many customers are flagged, and how many thefts that should find."""
    spec = get_pipeline_spec()
    probabilities = get_population_probabilities().to_numpy()
    threshold = get_decision_threshold()
    flagged = probabilities >= threshold
    tiers = pd.Series([risk_tier(p, threshold) for p in probabilities])
    population = active_population()
    return {
        "threshold": threshold,
        "trained_threshold": get_trained_threshold(),
        "model_version": spec["model_version"],
        "trained_at": spec["trained_at"],
        "pipeline": spec["name"],
        "pipeline_label": spec["label"],
        "customers_monitored": int(len(probabilities)),
        "flagged": int(flagged.sum()),
        # Sums of calibrated probabilities: estimates, not observed outcomes.
        "expected_thefts_flagged": float(probabilities[flagged].sum()),
        "expected_thefts_total": float(probabilities.sum()),
        "risk_tier_distribution": {tier: int((tiers == tier).sum()) for tier in ("high", "medium", "low")},
        "population": {key: population[key] for key in ("id", "source", "filename", "promoted_at", "promoted_by")},
    }


def list_customers(search: Optional[str], tier: Optional[str], page: int, page_size: int) -> Dict[str, Any]:
    """Customers in the operational population ranked by theft probability."""
    probabilities = get_population_probabilities()
    threshold = get_decision_threshold()
    rows = [
        {"customer_id": cid, "rank": rank, "risk_score": float(p), "risk_tier": risk_tier(p, threshold)}
        for rank, (cid, p) in enumerate(probabilities.items(), start=1)
    ]
    if search:
        rows = [r for r in rows if search.lower() in r["customer_id"].lower()]
    if tier:
        rows = [r for r in rows if r["risk_tier"] == tier]
    start = (page - 1) * page_size
    return {"items": rows[start:start + page_size], "total": len(rows), "page": page, "page_size": page_size, "threshold": threshold}


@lru_cache(maxsize=1)
def _validation_arrays() -> tuple:
    frame = data_service.load_predictions("validation")
    return frame["label"].to_numpy(dtype=int), frame[get_pipeline_spec()["name"]].to_numpy(dtype=float)


def _validation_point(threshold: float) -> Dict[str, Any]:
    y, p = _validation_arrays()
    flagged = p >= threshold
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    fn, tn = int((~flagged & (y == 1)).sum()), int((~flagged & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def threshold_preview(threshold: float) -> Dict[str, Any]:
    """What a threshold means: validation precision and recall, and the workload in the population."""
    probabilities = get_population_probabilities().to_numpy()
    flagged = probabilities >= threshold
    y, _ = _validation_arrays()
    return {
        "threshold": threshold,
        "validation": {**_validation_point(threshold), "customers": int(len(y)), "theft": int(y.sum())},
        "population_flagged": int(flagged.sum()),
        "population_expected_thefts": float(probabilities[flagged].sum()),
    }


def operating_curve(capacity: Optional[int] = None, cost_per_visit: float = 0.0, value_per_theft: float = 0.0) -> Dict[str, Any]:
    """
    For thresholds 0.01-0.99: validation precision and recall, and in the operational population the
    customers flagged, the visits possible within ``capacity`` (the highest-ranked first), the thefts
    those visits should find (sum of their calibrated probabilities) and the net value.
    """
    y, _ = _validation_arrays()
    ranked = get_population_probabilities().to_numpy()  # descending
    expected = np.r_[0.0, np.cumsum(ranked)]
    points = []
    for threshold in np.round(np.arange(0.01, 1.0, 0.01), 2):
        flagged = int((ranked >= threshold).sum())
        visits = flagged if capacity is None else min(flagged, int(capacity))
        caught = float(expected[visits])
        points.append({
            "threshold": float(threshold), **_validation_point(float(threshold)),
            "population_flagged": flagged, "visits": visits, "expected_thefts_found": caught,
            "net_value": caught * value_per_theft - visits * cost_per_visit,
        })
    return {
        "validation_customers": int(len(y)), "validation_theft": int(y.sum()),
        "population_customers": int(len(ranked)), "capacity": capacity,
        "cost_per_visit": cost_per_visit, "value_per_theft": value_per_theft,
        "threshold": get_decision_threshold(), "trained_threshold": get_trained_threshold(),
        "points": points,
    }


def score_distribution() -> Dict[str, Any]:
    """20-bin histogram of calibrated probabilities over the operational population (no labels)."""
    probabilities = get_population_probabilities().to_numpy()
    edges = np.linspace(0.0, 1.0, 21)
    return {"edges": edges.round(4).tolist(), "counts": np.histogram(probabilities, bins=edges)[0].astype(int).tolist(),
            "threshold": get_decision_threshold()}


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_shap_explainer():
    return shap.TreeExplainer(get_trained_model())


def _shap_matrix(features: pd.DataFrame) -> np.ndarray:
    values = get_shap_explainer().shap_values(_aligned(features))
    if isinstance(values, list):
        values = values[-1]
    return np.asarray(values, dtype=float).reshape(len(features), -1)


def _base_value() -> float:
    base = get_shap_explainer().expected_value
    return float(np.asarray(base).ravel()[-1])


def reason(name: str, value: Any, shap_value: float) -> Dict[str, Any]:
    start, n_days = series_calendar()
    value = finite_or_none(value)
    return {
        "feature": name,
        "label": feature_label(name),
        "value": value,
        "display_value": format_feature_value(name, value, start, n_days),
        "shap_value": float(shap_value),
    }


def _reasons(features: pd.Series, contributions: np.ndarray, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return [reason(features.index[i], features.iloc[i], contributions[i]) for i in order]


def _customer_features(customer_id: str) -> pd.DataFrame:
    X = get_feature_matrix()
    if customer_id not in X.index:
        raise NotFoundError(f"Unknown customer_id: {customer_id}")
    return X.loc[[customer_id], get_feature_names()]


def explain_customer(customer_id: str) -> Dict[str, Any]:
    """
    Every feature's SHAP contribution to the XGBoost raw score (log-odds): base + sum = logit(raw
    score). ``probability`` is the served (for the hybrid: blended) probability, ``parts`` its parts.
    """
    features = _customer_features(customer_id)
    scored = get_population_scores().loc[customer_id]
    return {
        "customer_id": customer_id,
        "probability": float(scored["probability"]),
        "raw_score": float(scored["raw_score"]),
        "tree_probability": float(scored["tree"]),
        "base_value": _base_value(),
        "contributions": _reasons(features.iloc[0], _shap_matrix(features)[0]),
        "parts": blend_parts(scored),
    }


def sequence_explanation(customer_id: str) -> Dict[str, Any]:
    """
    Which weeks of the customer's readings raised the sequence model's score: for each week, the logit
    minus the logit with that week replaced by the customer's typical day (src.sequence).
    """
    if not is_hybrid():
        return {"customer_id": customer_id, "available": False, "weeks": []}
    wide = get_wide_data()
    if customer_id not in wide.index:
        raise NotFoundError(f"Unknown customer_id: {customer_id}")
    part = get_pipeline_spec()["components"][SEQUENCE]
    row = wide.loc[[customer_id]]
    values, mask = sequence.prepare(row, part.get("cleaning"))
    effects = sequence.week_effects(get_sequence_network(), values[0], mask[0])
    days = sequence.day_index(row.shape[1])
    dates = pd.DatetimeIndex(row.columns) if isinstance(row.columns, pd.DatetimeIndex) else None
    weeks = []
    for item in effects:
        positions = [int(d) for d in days[item["week"] * 7:(item["week"] + 1) * 7] if d >= 0]
        if not positions:
            continue
        weeks.append({
            "week": item["week"], "effect": item["effect"],
            "start": dates[positions[0]].strftime("%Y-%m-%d") if dates is not None else None,
            "end": dates[positions[-1]].strftime("%Y-%m-%d") if dates is not None else None,
        })
    scored = get_population_scores().loc[customer_id]
    return {"customer_id": customer_id, "available": True, "label": part["label"],
            "probability": float(scored["sequence"]), "raw_score": float(sequence_probability(row)[0][0]),
            "weight": float(get_pipeline_spec()["blend"]["weight_sequence"]), "weeks": weeks}


def _prediction(features: pd.DataFrame, threshold: Optional[float], wide: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    threshold = get_decision_threshold() if threshold is None else threshold
    scored = score(features, wide).iloc[0]
    probability = float(scored["probability"])
    return {
        "probability": probability,
        "raw_score": float(scored["raw_score"]),
        "prediction": int(probability >= threshold),
        "threshold": threshold,
        "risk_tier": risk_tier(probability, threshold),
        "reasons": _reasons(features.iloc[0], _shap_matrix(features)[0], top_n=3),
        "parts": blend_parts(scored),
    }


def predict_customer(customer_id: str, threshold: Optional[float] = None) -> Dict[str, Any]:
    features = _customer_features(customer_id)
    wide = get_wide_data().loc[[customer_id]] if is_hybrid() else None
    return {"customer_id": customer_id, **_prediction(features, threshold, wide)}


def require_features(columns) -> None:
    """Every model feature must be present; a missing column would silently count as a missing reading."""
    missing = [name for name in get_feature_names() if name not in set(columns)]
    if missing:
        shown = ", ".join(missing[:12]) + (f" and {len(missing) - 12} more" if len(missing) > 12 else "")
        raise FeatureInputError(f"{len(missing)} of the {len(get_feature_names())} model features are missing: {shown}")


def predict_features(features: Dict[str, Optional[float]], threshold: Optional[float] = None) -> Dict[str, Any]:
    if is_hybrid():
        raise FeatureInputError(NEEDS_READINGS)
    require_features(features)
    names = get_feature_names()
    frame = pd.DataFrame([[features[name] for name in names]], columns=names, dtype=float)
    return {"customer_id": None, **_prediction(frame, threshold)}


@lru_cache(maxsize=4)
def _flagged_drivers(population_id: str, threshold: float) -> Dict[str, Dict[str, Any]]:
    probabilities = get_population_probabilities()
    flagged = probabilities[probabilities >= threshold].index
    if len(flagged) == 0:
        return {}
    frame = get_feature_matrix().loc[flagged, get_feature_names()]
    values = _shap_matrix(frame)
    drivers = {}
    for row, customer_id in enumerate(flagged):
        i = int(np.argmax(values[row]))
        drivers[customer_id] = reason(frame.columns[i], frame.iat[row, i], values[row, i])
    return drivers


def get_flagged_drivers() -> Dict[str, Dict[str, Any]]:
    """Strongest SHAP driver towards theft for every customer at or above the threshold."""
    return _flagged_drivers(active_population()["id"], get_decision_threshold())


@lru_cache(maxsize=2)
def _global_drivers(population_id: str, sample_size: int = 500, top_n: int = 15) -> Dict[str, Any]:
    X = get_feature_matrix()
    sample = X[get_feature_names()].sample(n=min(sample_size, len(X)), random_state=0)
    values = _shap_matrix(sample)
    drivers = []
    for j, name in enumerate(sample.columns):
        column = sample[name].to_numpy(dtype=float)
        observed = ~np.isnan(column)
        # Sign of the covariance between feature value and contribution: does more of it push towards theft?
        if observed.sum() > 1 and np.std(column[observed]) > 0:
            direction = "higher" if np.cov(column[observed], values[observed, j])[0, 1] >= 0 else "lower"
        else:
            direction = "unclear"
        drivers.append({"feature": name, "label": feature_label(name), "mean_abs_shap": float(np.abs(values[:, j]).mean()), "risk_when": direction})
    drivers.sort(key=lambda d: d["mean_abs_shap"], reverse=True)
    return {"sample_size": int(len(sample)), "drivers": drivers[:top_n]}


def global_drivers() -> Dict[str, Any]:
    """Mean |SHAP| per feature over a fixed sample of the population, with the direction of effect."""
    return _global_drivers(active_population()["id"])


# ---------------------------------------------------------------------------
# Explanation consistency: LIME (proposal section 3.12)
# ---------------------------------------------------------------------------

EXPLANATION_CHECK_TOP_N = 5
CONSISTENCY_NOTE = ("SHAP and LIME both describe what this model associates with theft. Agreement between them "
                    "says the explanation is stable, not that it is causal or that theft occurred.")


@lru_cache(maxsize=2)
def _lime_explainer(population_id: str):
    from lime.lime_tabular import LimeTabularExplainer

    background = get_feature_matrix()[get_feature_names()]
    medians = background.median().fillna(0.0)
    # Decile bins: with LIME's default quartiles its surrogate leans on magnitude features and agrees
    # with SHAP on fewer of the highest-risk customers.
    explainer = LimeTabularExplainer(background.fillna(medians).to_numpy(), feature_names=get_feature_names(),
                                     class_names=["honest", "theft"], mode="classification", discretizer="decile",
                                     random_state=0)
    return explainer, medians


@lru_cache(maxsize=256)
def _explanation_check(population_id: str, customer_id: str) -> Dict[str, Any]:
    names = get_feature_names()
    shap_top = explain_customer(customer_id)["contributions"][:EXPLANATION_CHECK_TOP_N]
    explainer, medians = _lime_explainer(population_id)
    row = _customer_features(customer_id).iloc[0].fillna(medians)

    def class_probabilities(rows: np.ndarray) -> np.ndarray:
        theft = raw_scores(pd.DataFrame(rows, columns=names))
        return np.column_stack([1 - theft, theft])

    lime = explainer.explain_instance(row.to_numpy(dtype=float), class_probabilities,
                                      num_features=EXPLANATION_CHECK_TOP_N, num_samples=5000)
    lime_weights = {names[i]: float(w) for i, w in lime.as_map()[1]}
    lime_top = sorted(lime_weights, key=lambda f: -abs(lime_weights[f]))
    shared = [c["feature"] for c in shap_top if c["feature"] in lime_weights]
    strongest = shap_top[0]
    same_direction = strongest["feature"] in lime_weights and np.sign(lime_weights[strongest["feature"]]) == np.sign(strongest["shap_value"])
    consistent = len(shared) >= 3 and bool(same_direction)
    return {
        "customer_id": customer_id,
        "top_n": EXPLANATION_CHECK_TOP_N,
        "shap": [{"feature": c["feature"], "label": c["label"], "weight": c["shap_value"]} for c in shap_top],
        "lime": [{"feature": f, "label": feature_label(f), "weight": lime_weights[f]} for f in lime_top],
        "shared": shared,
        "consistent": consistent,
        "message": (f"SHAP and LIME agree on {len(shared)} of the top {EXPLANATION_CHECK_TOP_N} signals."
                    if consistent else
                    f"SHAP and LIME share only {len(shared)} of the top {EXPLANATION_CHECK_TOP_N} signals"
                    f"{'' if same_direction else ' and disagree on the strongest one'}: treat the explanation with care."),
        "note": CONSISTENCY_NOTE,
    }


def explanation_check(customer_id: str) -> Dict[str, Any]:
    """
    Compare SHAP's and LIME's strongest features for one customer (both on the raw score). They are
    consistent when at least 3 of the top 5 coincide and SHAP's strongest feature is in LIME's top 5
    with the same direction.
    """
    return _explanation_check(active_population()["id"], customer_id)


_CACHED = (
    integrity_problems, get_trained_model, get_sequence_network, _population_scores, _validation_arrays, get_shap_explainer,
    _flagged_drivers, _global_drivers, _lime_explainer, _explanation_check, *data_service.CACHED,
)


def clear_caches() -> None:
    """Reload model, data and every derived result on next use."""
    for function in _CACHED:
        function.cache_clear()
