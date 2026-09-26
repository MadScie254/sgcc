"""Tests for the pieces the extension studies add: treatments, top-of-list metrics, windows, the CNN."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.eval import net_value
from src.modeling import make_folds
from src.resampling import Treatment
from src.stats import map_at_k, paired_comparison, top_share

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="module")
def imbalanced():
    X, y = make_classification(n_samples=600, n_features=5, n_informative=3, weights=[0.9, 0.1], random_state=1)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    X.iloc[::11, 0] = np.nan
    return X, pd.Series(y)


@pytest.mark.parametrize("kind", ["smote_tomek", "borderline_smote", "adasyn", "undersample"])
def test_new_treatments_rebalance_training_rows_only(imbalanced, kind):
    X, y = imbalanced
    treatment = Treatment(kind, {"sampling_strategy": 0.5}, random_state=0)
    X_res, y_res = treatment.fit_resample(X, y)
    assert list(X_res.columns) == list(X.columns)
    assert not X_res.isna().any().any()
    assert y_res.mean() > y.mean()  # more balanced than before
    if kind == "undersample":
        assert len(y_res) < len(y) and (y_res == 1).sum() == (y == 1).sum()
    else:
        assert (y_res == 1).sum() > (y == 1).sum()
    # Scoring rows only get the training-median fill, never resampled.
    held_out = X.iloc[:40]
    assert treatment.transform(held_out).shape == held_out.shape
    assert treatment.transform(held_out).index.equals(held_out.index)


def test_new_treatments_stay_inside_folds(imbalanced):
    X, y = imbalanced
    for X_tr, y_tr, X_val, val_idx in make_folds(X, y, cv=3, random_state=0, treatment="smote_tomek"):
        assert len(X_val) == len(val_idx)
        assert X_val.index.equals(X.index[val_idx])


def test_map_at_k_matches_hand_computation():
    y = np.array([1, 0, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert map_at_k(y, score, 3) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert map_at_k(y, score, 2) == pytest.approx(1.0)
    assert map_at_k(np.zeros(5), score, 3) == 0.0


def test_top_share_and_net_value():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    score = np.linspace(1, 0, 10)
    top = top_share(y, score, 0.2)
    assert top["inspected"] == 2 and top["precision"] == pytest.approx(0.5) and top["recall"] == pytest.approx(1 / 3)
    flags = score >= 0.75  # the top three: two thieves
    assert net_value(y, flags, cost_per_visit=30, value_per_theft=400) == pytest.approx(2 * 400 - 3 * 30)


def test_paired_comparison_of_identical_pipelines_finds_no_difference():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.15).astype(int)
    score = np.clip(0.4 * y + rng.normal(0.3, 0.2, 300), 0, 1)
    scores = {"a": score, "b": score.copy()}
    flags = {"a": score >= 0.5, "b": score >= 0.5}
    pipelines, comparisons = paired_comparison(y, scores, flags, "a", resamples=200)
    assert comparisons["b"]["pr_auc"]["difference"] == 0.0
    assert comparisons["b"]["pr_auc"]["p_holm"] == pytest.approx(1.0)
    assert comparisons["b"]["mcnemar"]["p_value"] == 1.0
    assert pipelines["a"]["pr_auc"]["ci_low"] <= pipelines["a"]["pr_auc"]["estimate"] <= pipelines["a"]["pr_auc"]["ci_high"]


def test_history_window_keeps_the_last_months():
    from history_length import window

    days = pd.date_range("2014-01-01", "2016-10-31", freq="D")
    wide = pd.DataFrame(np.ones((2, len(days))), columns=days)
    last3 = window(wide, 3)
    assert last3.columns.min() == pd.Timestamp("2016-08-01") and last3.columns.max() == pd.Timestamp("2016-10-31")
    assert window(wide, 34).shape[1] == len(days)


def test_wide_deep_cnn_scores_one_logit_per_customer():
    torch = pytest.importorskip("torch")
    from src.sequence import DAYS, build_network, prepare

    days = pd.date_range("2014-01-01", periods=1034, freq="D")
    wide = pd.DataFrame(np.random.default_rng(0).random((3, 1034)) * 5, columns=days)
    wide.iloc[0, 10:20] = np.nan
    values, mask = prepare(wide, None)
    assert values.shape == mask.shape == (3, DAYS)
    assert values.min() >= 0 and values.max() <= 1 and mask[0, 10:20].all() and mask[:, -2:].all()
    out = build_network()(torch.from_numpy(values), torch.from_numpy(mask))
    assert tuple(out.shape) == (3,)


def test_day_index_maps_input_positions_to_days():
    from src.sequence import DAYS, day_index

    sgcc = day_index(1034)  # padded at the end to whole weeks
    assert len(sgcc) == DAYS and sgcc[0] == 0 and sgcc[1033] == 1033 and (sgcc[-2:] == -1).all()
    short = day_index(365)  # a shorter history is padded at the start
    assert (short[:DAYS - 365] == -1).all() and short[-1] == 364
    long = day_index(1200)  # a longer one keeps the most recent days
    assert long[0] == 1200 - DAYS and long[-1] == 1199


def test_sequence_model_exports_to_onnx_and_scores_the_same(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from src.sequence import WEEKS, OnnxNetwork, build_network, export_onnx, predict_network, prepare, week_effects

    torch.manual_seed(0)
    days = pd.date_range("2014-01-01", periods=1034, freq="D")
    wide = pd.DataFrame(np.random.default_rng(1).random((4, 1034)) * 5, columns=days)
    wide.iloc[1, 100:140] = np.nan
    values, mask = prepare(wide, None)
    network = build_network()
    export_onnx(network, tmp_path / "sequence.onnx")
    onnx = OnnxNetwork(tmp_path / "sequence.onnx")
    assert np.abs(onnx.predict(values, mask) - predict_network(network, values, mask)).max() < 1e-5
    effects = week_effects(onnx, values[1], mask[1])
    assert len(effects) == WEEKS and all(np.isfinite(e["effect"]) for e in effects)
