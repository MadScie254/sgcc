"""
SGCC Theft Detector - Paired comparison of pipelines on the test customers

Every pipeline scored the same test customers once, so the comparison is
paired: the bootstrap resamples customers (with replacement, thieves and honest
customers separately so each resample keeps the class ratio) and recomputes every
pipeline's metrics on the same resample. A resample is represented by how many
times each customer is drawn, which makes the ranking metrics exact
(ties grouped as scikit-learn does) and fast to compute for many resamples at once.

McNemar's exact test compares two pipelines' flag decisions customer by customer.
"""

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.stats import binomtest

METRICS = ("pr_auc", "auc", "f1", "recall", "precision", "mcc")


def stratified_counts(y: np.ndarray, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """(resamples, n) draw counts; each row keeps the number of thieves and honest customers."""
    y = np.asarray(y).astype(int)
    counts = np.zeros((resamples, len(y)), dtype=np.int32)
    for label in (0, 1):
        members = np.flatnonzero(y == label)
        counts[:, members] = rng.multinomial(len(members), np.full(len(members), 1 / len(members)), size=resamples)
    return counts


def _groups(score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Customers ordered by descending score, and the last position of every run of tied scores."""
    order = np.argsort(-score, kind="mergesort")
    ordered = score[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered) != 0), len(score) - 1]
    return order, ends


def ranking_metrics(y: np.ndarray, score: np.ndarray, weights: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Weighted PR-AUC (average precision) and ROC-AUC, one value per row of ``weights``;
    identical to scikit-learn's with ``sample_weight``.
    """
    y = np.asarray(y).astype(int)
    weights = np.atleast_2d(weights).astype(float)
    order, ends = _groups(np.asarray(score, dtype=float))
    w = weights[:, order]
    pos = w * y[order]
    neg = w - pos
    tp = np.cumsum(pos, axis=1)[:, ends]
    fp = np.cumsum(neg, axis=1)[:, ends]
    total_pos, total_neg = tp[:, -1:], fp[:, -1:]
    recall_step = np.diff(np.c_[np.zeros(len(w)), tp], axis=1) / total_pos
    precision = tp / np.maximum(tp + fp, 1e-12)
    ap = (recall_step * precision).sum(axis=1)
    # ROC-AUC: a thief outranks each honest customer scored lower, and half-counts ties.
    group_pos = np.diff(np.c_[np.zeros(len(w)), tp], axis=1)
    group_neg = np.diff(np.c_[np.zeros(len(w)), fp], axis=1)
    neg_below = total_neg - fp  # honest customers scored strictly lower than this group
    auc = (group_pos * (neg_below + 0.5 * group_neg)).sum(axis=1) / (total_pos[:, 0] * total_neg[:, 0])
    return {"pr_auc": ap, "auc": auc}


def threshold_metrics(y: np.ndarray, flags: np.ndarray, weights: np.ndarray) -> Dict[str, np.ndarray]:
    """F1, recall, precision and MCC of fixed flag decisions, one value per row of ``weights``."""
    y, flags = np.asarray(y).astype(bool), np.asarray(flags).astype(bool)
    weights = np.atleast_2d(weights).astype(float)
    tp = weights @ (flags & y)
    fp = weights @ (flags & ~y)
    fn = weights @ (~flags & y)
    tn = weights @ (~flags & ~y)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return {"f1": f1, "recall": recall, "precision": precision, "mcc": mcc}


def all_metrics(y, score, flags, weights) -> Dict[str, np.ndarray]:
    return {**ranking_metrics(y, score, weights), **threshold_metrics(y, flags, weights)}


def bootstrap_p_value(differences: np.ndarray) -> float:
    """Two-sided p-value for "no difference": twice the smaller share of resamples on either side of 0."""
    b = len(differences)
    below = (np.count_nonzero(differences <= 0) + 1) / (b + 1)
    above = (np.count_nonzero(differences >= 0) + 1) / (b + 1)
    return float(min(1.0, 2 * min(below, above)))


def mcnemar_exact(y, flags_a, flags_b) -> Dict[str, float]:
    """Exact McNemar test: customers only A classifies correctly (b) against only B (c)."""
    y = np.asarray(y).astype(bool)
    right_a, right_b = np.asarray(flags_a).astype(bool) == y, np.asarray(flags_b).astype(bool) == y
    b, c = int(np.count_nonzero(right_a & ~right_b)), int(np.count_nonzero(~right_a & right_b))
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"only_reference_right": b, "only_other_right": c, "p_value": p}


def holm(p_values: Iterable[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted
