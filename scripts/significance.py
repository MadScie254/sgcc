"""
Statistical comparison of the pipelines on the test customers (proposal section 3.11).

    python scripts/significance.py                   # after python -m src.train; about a minute
    python scripts/significance.py --resamples 2000  # quicker, coarser p-values

Reads the test predictions training saved (artifacts/predictions/test.csv.gz): every
pipeline scored the same 6,356 test customers once, after all tuning, thresholds and
calibration were fixed on training and validation customers. Nothing is refitted, so
no GPU is needed here; ``--device`` does not apply.

- Paired stratified bootstrap (10,000 resamples of the test customers; thieves and honest
  customers resampled separately): a 95% percentile interval for every pipeline's
  PR-AUC, ROC-AUC, F1, recall, precision and MCC, and for the difference between the
  proposed pipeline and each other pipeline, with a two-sided bootstrap p-value and
  Holm's correction across the four comparisons of each metric.
- McNemar's exact test on the paired flag decisions (each pipeline at its own
  validation-chosen threshold), Holm-corrected across the four comparisons.

Writes artifacts/significance.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import CANDIDATES  # noqa: E402
from src.stats import METRICS, all_metrics, bootstrap_p_value, holm, mcnemar_exact, stratified_counts  # noqa: E402
from src.train import load_config  # noqa: E402

REFERENCE = "proposed"
ALPHA = 0.05
CHUNK = 500


def interval(values: np.ndarray) -> dict:
    low, high = np.percentile(values, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap and McNemar tests on the saved test predictions")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    artifacts = ROOT / config["paths"]["artifacts"]
    comparison = json.loads((ROOT / config["paths"]["models"] / "baselines" / "comparison_results.json").read_text())
    test = pd.read_csv(artifacts / "predictions" / "test.csv.gz", dtype={"customer_id": str})
    y = test["label"].to_numpy(dtype=int)
    names = list(CANDIDATES)
    others = [n for n in names if n != REFERENCE]
    scores = {n: test[n].to_numpy(dtype=float) for n in names}
    flags = {n: scores[n] >= comparison[n]["threshold"] for n in names}
    ones = np.ones((1, len(y)))
    point = {n: {m: float(v[0]) for m, v in all_metrics(y, scores[n], flags[n], ones).items()} for n in names}

    # Resampled metrics for every pipeline on the same resamples (paired), in chunks to bound memory.
    rng = np.random.default_rng(args.seed)
    draws = {n: {m: [] for m in METRICS} for n in names}
    for start in range(0, args.resamples, CHUNK):
        weights = stratified_counts(y, min(CHUNK, args.resamples - start), rng)
        for n in names:
            for m, values in all_metrics(y, scores[n], flags[n], weights).items():
                draws[n][m].append(values)
        print(f"{min(start + CHUNK, args.resamples):,}/{args.resamples:,} resamples", flush=True)
    draws = {n: {m: np.concatenate(v) for m, v in d.items()} for n, d in draws.items()}

    pipelines = {n: {"label": CANDIDATES[n]["label"], "threshold": float(comparison[n]["threshold"]),
                     "metrics": {m: {"estimate": point[n][m], **interval(draws[n][m])} for m in METRICS}}
                 for n in names}
    comparisons = {n: {} for n in others}
    for m in METRICS:
        raw_p = []
        for n in others:
            diff = draws[REFERENCE][m] - draws[n][m]
            comparisons[n][m] = {"difference": point[REFERENCE][m] - point[n][m], **interval(diff),
                                 "p_value": bootstrap_p_value(diff)}
            raw_p.append(comparisons[n][m]["p_value"])
        for n, adjusted in zip(others, holm(raw_p)):
            comparisons[n][m]["p_holm"] = float(adjusted)
            comparisons[n][m]["significant"] = bool(adjusted < ALPHA)
    mcnemar = {n: mcnemar_exact(y, flags[REFERENCE], flags[n]) for n in others}
    for n, adjusted in zip(others, holm([mcnemar[n]["p_value"] for n in others])):
        comparisons[n]["mcnemar"] = {**mcnemar[n], "p_holm": float(adjusted), "significant": bool(adjusted < ALPHA)}

    result = {
        "method": "paired stratified bootstrap and exact McNemar test on the test customers",
        "population": {"split": "test", "customers": int(len(y)), "theft": int(y.sum())},
        "resamples": args.resamples, "seed": args.seed, "alpha": ALPHA, "reference": REFERENCE,
        "metrics": list(METRICS),
        "pipelines": pipelines,
        "comparisons": comparisons,
    }
    out = artifacts / "significance.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    for n in others:
        row = comparisons[n]
        print(f"{REFERENCE} - {n}: " + ", ".join(
            f"{m} {row[m]['difference']:+.3f} [{row[m]['ci_low']:+.3f}, {row[m]['ci_high']:+.3f}] p_holm={row[m]['p_holm']:.4f}"
            for m in ("pr_auc", "f1")) + f"; McNemar p_holm={row['mcnemar']['p_holm']:.4g}")


if __name__ == "__main__":
    main()
