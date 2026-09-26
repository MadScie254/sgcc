"""Tests for src.stats: the weighted metrics match scikit-learn, and the tests behave."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

from src.stats import all_metrics, bootstrap_p_value, holm, mcnemar_exact, stratified_counts


@pytest.fixture(scope="module")
def scored():
    rng = np.random.default_rng(1)
    y = (rng.random(600) < 0.12).astype(int)
    score = np.round(np.clip(0.3 * y + rng.normal(0.3, 0.2, len(y)), 0, 1), 2)  # rounding makes ties
    return y, score, score >= 0.45


def test_weighted_metrics_match_sklearn(scored):
    y, score, flags = scored
    weights = stratified_counts(y, 5, np.random.default_rng(0))
    got = all_metrics(y, score, flags, weights)
    for row, w in enumerate(weights):
        assert got["pr_auc"][row] == pytest.approx(average_precision_score(y, score, sample_weight=w))
        assert got["auc"][row] == pytest.approx(roc_auc_score(y, score, sample_weight=w))
        assert got["f1"][row] == pytest.approx(f1_score(y, flags, sample_weight=w))
        assert got["mcc"][row] == pytest.approx(matthews_corrcoef(y, flags, sample_weight=w))


def test_stratified_counts_keep_the_class_sizes(scored):
    y, _, _ = scored
    counts = stratified_counts(y, 50, np.random.default_rng(0))
    assert (counts[:, y == 1].sum(axis=1) == y.sum()).all()
    assert (counts.sum(axis=1) == len(y)).all()


def test_p_values():
    assert bootstrap_p_value(np.full(999, 0.1)) == pytest.approx(2 / 1000)
    assert bootstrap_p_value(np.linspace(-1, 1, 1001)) == pytest.approx(1.0)
    assert list(holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    result = mcnemar_exact(y, y, 1 - y)  # A always right, B always wrong
    assert (result["only_reference_right"], result["only_other_right"]) == (8, 0)
    assert result["p_value"] == pytest.approx(2 * 0.5 ** 8)
