"""
Metrics the SGCC literature reports, and what each pipeline is worth to an inspection budget.

    python scripts/literature_metrics.py        # seconds; reads saved predictions, refits nothing

For every pipeline scored on the test customers (the five of the study, plus the
pipelines the extension scripts saved in artifacts/predictions/extensions_test.csv.gz):

- MAP@100 and MAP@200 (Zheng et al., 2018): the mean precision at each thief's position
  among the N highest-scored customers. MAP@N depends on how many thieves the test set
  holds (542 of 6,356 here), so it is comparable with published values only roughly.
- Precision in the top 1% and 5% of customers, recall in the top 5% and 10%.
- Net value per 1,000 customers at the pipeline's validation-chosen threshold and when the
  top 5% are inspected, with the console's example budget: 30 per visit, 400 recovered
  per theft (the same formula as the threshold studio).

95% intervals come from a stratified bootstrap of the test customers (2,000 resamples),
and differences from standard XGBoost (the served pipeline) are paired on the same resamples.
Writes artifacts/literature_metrics.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import net_value  # noqa: E402
from src.experiment import ALL_CANDIDATES  # noqa: E402
from src.stats import interval, map_at_k, top_share  # noqa: E402

OUT = ROOT / "artifacts" / "literature_metrics.json"
COST_PER_VISIT, VALUE_PER_THEFT = 30.0, 400.0
REFERENCE = "xgboost"
EXTRA_LABELS = {"wide_deep_cnn": "Wide & Deep CNN (Zheng et al., 2018)"}


def measures(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    n = len(y)
    flags = p >= threshold
    top5 = np.zeros(n, dtype=bool)
    top5[np.argsort(-p, kind="mergesort")[:max(1, round(0.05 * n))]] = True
    return {
        "map_at_100": map_at_k(y, p, 100),
        "map_at_200": map_at_k(y, p, 200),
        "precision_top_1pct": top_share(y, p, 0.01)["precision"],
        "precision_top_5pct": top_share(y, p, 0.05)["precision"],
        "recall_top_5pct": top_share(y, p, 0.05)["recall"],
        "recall_top_10pct": top_share(y, p, 0.10)["recall"],
        "net_value_per_1000_at_threshold": 1000 * net_value(y, flags, COST_PER_VISIT, VALUE_PER_THEFT) / n,
        "net_value_per_1000_top_5pct": 1000 * net_value(y, top5, COST_PER_VISIT, VALUE_PER_THEFT) / n,
    }


def load_scores():
    thresholds = json.loads((ROOT / "models" / "baselines" / "comparison_results.json").read_text())
    test = pd.read_csv(ROOT / "artifacts" / "predictions" / "test.csv.gz", dtype={"customer_id": str}).set_index("customer_id")
    scores = {n: test[n].to_numpy(float) for n in thresholds}
    cuts = {n: float(thresholds[n]["threshold"]) for n in thresholds}
    labels = {n: thresholds[n]["label"] for n in thresholds}
    extra_path = ROOT / "artifacts" / "predictions" / "extensions_test.csv.gz"
    if extra_path.exists():
        extra = pd.read_csv(extra_path, dtype={"customer_id": str}).set_index("customer_id").loc[test.index]
        sources = {"xgboost_smote_enn_raw": ("resampling_study.json", ("fair_test", "result", "threshold")),
                   "wide_deep_cnn": ("deep_baseline.json", ("result", "threshold"))}
        for name, (file, path) in sources.items():
            record = ROOT / "artifacts" / file
            if name in extra.columns and record.exists():
                value = json.loads(record.read_text())
                for key in path:
                    value = value[key]
                scores[name], cuts[name] = extra[name].to_numpy(float), float(value)
                labels[name] = ALL_CANDIDATES.get(name, {}).get("label") or EXTRA_LABELS[name]
    return test["label"].to_numpy(int), scores, cuts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Literature metrics and budget value on the test customers")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    y, scores, cuts, labels = load_scores()
    point = {n: measures(y, scores[n], cuts[n]) for n in scores}

    rng = np.random.default_rng(args.seed)
    thieves, honest = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = {n: {m: [] for m in point[n]} for n in scores}
    for _ in range(args.resamples):
        idx = np.concatenate([rng.choice(thieves, len(thieves)), rng.choice(honest, len(honest))])
        for n in scores:
            for m, v in measures(y[idx], scores[n][idx], cuts[n]).items():
                draws[n][m].append(v)
    draws = {n: {m: np.asarray(v) for m, v in d.items()} for n, d in draws.items()}

    pipelines = {}
    for n in scores:
        pipelines[n] = {"label": labels[n], "threshold": cuts[n],
                        "metrics": {m: {"estimate": float(point[n][m]), **interval(draws[n][m])} for m in point[n]}}
        if n != REFERENCE:
            pipelines[n]["difference_from_standard_xgboost"] = {
                m: {"difference": float(point[n][m] - point[REFERENCE][m]), **interval(draws[n][m] - draws[REFERENCE][m])}
                for m in point[n]}
    result = {"population": {"split": "test", "customers": int(len(y)), "theft": int(y.sum())},
              "budget": {"cost_per_visit": COST_PER_VISIT, "value_per_theft": VALUE_PER_THEFT},
              "resamples": args.resamples, "seed": args.seed, "reference": REFERENCE, "pipelines": pipelines}
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    for n, row in pipelines.items():
        m = row["metrics"]
        print(f"{row['label']}: MAP@100 {m['map_at_100']['estimate']:.3f}, MAP@200 {m['map_at_200']['estimate']:.3f}, "
              f"P@1% {m['precision_top_1pct']['estimate']:.3f}, net/1000 {m['net_value_per_1000_at_threshold']['estimate']:.0f}")


if __name__ == "__main__":
    main()
