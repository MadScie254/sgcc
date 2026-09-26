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

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import CANDIDATES  # noqa: E402
from src.stats import METRICS, paired_comparison  # noqa: E402
from src.train import load_config  # noqa: E402

REFERENCE = "proposed"
ALPHA = 0.05


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
    metrics, comparisons = paired_comparison(y, scores, flags, REFERENCE, args.resamples, args.seed, ALPHA,
                                             progress=True)
    pipelines = {n: {"label": CANDIDATES[n]["label"], "threshold": float(comparison[n]["threshold"]),
                     "metrics": metrics[n]} for n in names}

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
