"""Tests for evaluation helpers."""

import numpy as np
import pytest

from src.eval import ranking_metrics


def test_ranking_metrics_counts_thefts_in_the_top_fraction() -> None:
    # 100 customers, 10 thefts; the 5 highest scores are thefts, then 5 more are spread out
    y = np.zeros(100, dtype=int)
    y[[0, 1, 2, 3, 4, 20, 40, 60, 80, 99]] = 1
    scores = np.linspace(1.0, 0.0, 100)  # customer 0 scores highest

    rows = {row["fraction"]: row for row in ranking_metrics(y, scores, fractions=(0.05, 0.10, 1.0))}

    assert rows[0.05]["inspections"] == 5
    assert rows[0.05]["thefts_found"] == 5
    assert rows[0.05]["precision"] == pytest.approx(1.0)
    assert rows[0.05]["recall"] == pytest.approx(0.5)
    assert rows[0.05]["lift"] == pytest.approx(10.0)
    assert rows[0.10]["thefts_found"] == 5
    assert rows[1.0]["recall"] == pytest.approx(1.0)
    assert rows[1.0]["lift"] == pytest.approx(1.0)
