# Thesis figures

Every figure below comes from the committed model and the real SGCC data. Results
charts have a `.png` (300 dpi) and a `.pdf` (vector: use it in LaTeX). Screenshots
are 2× resolution PNGs of the running console.

Numbers quoted here are the ones in the figures. The hold-out test set is 8,475
customers (723 theft); the console serves a 3,000-customer sample of it (256 theft),
so console screenshots show slightly different counts from the Chapter 5 charts.

Suggested structure: Chapter 3 Methodology, Chapter 4 System design and
implementation, Chapter 5 Results and discussion. Renumber to match your template.

## Chapter 3: Data and methodology

| Figure | File | Where | Caption and what to say |
|---|---|---|---|
| 3.1 | `fig-3-1-system-architecture` | 3.1 Overview, or open Chapter 4 with it | *End-to-end architecture: offline training, committed artifacts, the FastAPI serving layer with its automated scoring pipeline, and the operations console.* Walk through the four layers top to bottom. |
| 3.2 | `fig-3-2-example-consumption` | 3.2 Dataset description | *Monthly mean consumption of an honest customer (p = 0.018) and a theft customer (p = 0.986) from the hold-out set. Shaded months have no readings.* The theft meter stops reporting in April 2015: long reporting gaps are the model's strongest signal (Fig. 5.5). |
| 3.3 | `fig-3-3-missing-readings-by-class` | 3.3 Data quality / exploratory analysis | *Share of days without a meter reading, by class, across all 42,372 customers.* 44% of honest customers have almost no gaps vs 31% of theft customers; theft customers are over-represented at 20–70% missing. Justifies keeping missing values as signal instead of imputing them. |

Methodology text to cover, with the evidence already in the repo:
- Dataset: 42,372 customers × 1,034 days (1 Jan 2014 – 31 Oct 2016), 8.5% theft (`scripts/download_data.py`).
- Features: 85 per customer (`src/features.py`); list the groups: statistics, missing/zero runs, day-over-day drops, trends, year-over-year, change points, weekday/weekend, 34-month profile.
- Split: stratified 80/20 by customer, seed 42 (33,897 train / 8,475 test).
- Model: XGBoost, Optuna TPE, 40 trials × 5-fold CV on PR-AUC; best params in `artifacts/best_params.json`.
- Threshold: maximum F1 on out-of-fold training predictions (τ = 0.228), never tuned on test data.

## Chapter 4: System design and implementation

| Figure | File | Where | Caption and what to say |
|---|---|---|---|
| 4.1 | `fig-4-1-command-center` | 4.x Operations console overview | *Command center: headline figures, the last scoring run, the investigation queue ranked by theft probability, the risk distribution and outcomes at the operating threshold.* |
| 4.2 | `fig-4-2-case-files` | 4.x Case management workflow | *Case files: every flagged customer becomes a case that moves from new → reviewing → dispatched → confirmed or cleared, each with its strongest model signal.* |
| 4.3 | `fig-4-3-case-file` | 4.x Explainable case view | *Case file for the highest-risk customer: probability gauge, reading facts, consumption history, SHAP explanation and case activity.* Links the model to an analyst decision. |
| 4.4 | `fig-4-4-consumption-daily` | same section, or 3.2 | *Daily consumption for the same customer; the shaded area has no readings.* |
| 4.5 | `fig-4-5-pipeline-scoring` | 4.x Automated scoring pipeline | *Scoring pipeline: ingest → features → score → explain → route, with per-stage timings and run history. Runs at start-up, on demand, or on a schedule.* An end-to-end run takes about 3–4 s for 3,000 customers. |
| 4.6 | `fig-4-6-pipeline-training` | 4.x Model lifecycle / retraining | *Training pipeline with the recorded stage timings of the run that produced the deployed model (28 min 50 s, mostly Optuna tuning).* Training replaces the served model, so it runs from the command line only; the console shows its record. |
| 4.7 | `fig-4-7-threshold-studio` | 4.x Operating-point selection | *Threshold studio at the trained threshold: inspections, thefts caught, hit rate and wasted visits on held-out customers.* |
| 4.8 | `fig-4-8-threshold-studio-0.50` | same section, next to 4.7 | *The same view at τ = 0.50: fewer, surer inspections (hit rate 79%) at the cost of recall (25%).* Use 4.7 and 4.8 together to show the trade-off is an operational decision. |
| 4.9 | `fig-4-9-model-performance` | 4.x Monitoring the model in the console | *Model performance page: hold-out metrics, comparison with baselines, and the features that drive the score.* |
| 4.10 | `fig-4-10-reports-scoring` | 4.x Scoring API and reporting | *Reports and scoring: PDF reports for the portfolio, an uploaded dataset or a case; a file of raw SGCC meter data checked against the model (250 customers, 7 of 17 thefts caught, ROC-AUC 0.82); single-customer scoring.* |
| 4.11 | `fig-4-11-settings-api-key` | 4.x Security | *API-key configuration: the key is entered per browser and never shipped in the frontend bundle.* Mention the security fixes: path-traversal fix, constant-time key check, upload limits, no retraining through the API. |
| 4.12 | `fig-4-12-mobile-case-file` | 4.x Responsive design (optional) | *The case file on a phone-sized screen.* |

## Chapter 5: Results and discussion

| Figure | File | Where | Caption and what to say |
|---|---|---|---|
| 5.1 | `fig-5-1-roc-curves` | 5.1 Classification performance | *ROC curves on the 8,475-customer hold-out set.* XGBoost 0.847 vs random forest 0.816 and logistic regression 0.755. |
| 5.2 | `fig-5-2-precision-recall-curves` | 5.1, right after 5.1 | *Precision–recall curves; the dashed line is the 8.5% base rate.* PR-AUC 0.506 vs 0.417 and 0.272. PR-AUC is the headline metric for an 8.5% minority class. |
| 5.3 | `fig-5-3-confusion-matrix` | 5.2 Operating point | *Confusion matrix at τ = 0.228: 312 thefts caught, 252 false alarms, 411 thefts missed, 7,500 correctly cleared.* Precision 0.553, recall 0.432, F1 0.485, MCC 0.447. |
| 5.4 | `fig-5-4-threshold-tradeoff` | 5.2 | *Precision and recall across thresholds.* Supports the discussion that τ is a business choice (inspection budget vs thefts missed). |
| 5.5 | `fig-5-5-shap-importance` | 5.3 Explainability | *Top 15 features by mean absolute SHAP value on 2,000 test customers.* Reporting gaps, volatility and range lead; long-lag monthly profile features show the value of the 34-month history. |
| 5.6 | `fig-5-6-ablation` | 5.4 Ablation study | *5-fold cross-validated scores for five pipeline variants on all 42,372 customers.* The richer feature set adds +0.067 ROC-AUC / +0.167 PR-AUC; SMOTE-ENN costs −0.024 / −0.108; sorting dates chronologically is a correctness fix worth +0.006 ROC-AUC. Source: `artifacts/ablation.json`. |
| 5.7 | `fig-5-7-shap-waterfall-case` | 5.3, as a worked example | *SHAP waterfall for one flagged customer: from the 8.4% base rate to 98.6%.* |

Tables to include in Chapter 5 (numbers from `artifacts/metrics.json`,
`models/baselines/comparison_results.json`, `artifacts/ablation.json`): the model
comparison table and the ablation table in the repository README.

Limitations to state in the discussion: SGCC labels are known to be noisy; the data
covers one utility and 2014–2016; the served demo is a sample of the hold-out set;
the model has not been validated on another utility's customers.

## Regenerating

```bash
python scripts/download_data.py            # full dataset, once
pip install matplotlib imbalanced-learn    # plotting and the SMOTE-ENN ablation only
python scripts/make_thesis_figures.py      # figures 3.2, 3.3 and 5.1–5.5
python scripts/ablation.py                 # figure 5.6 (~10 min); --plot-only redraws it
```

The screenshots (4.x, 5.7) and the architecture diagram (3.1) were captured from the
running console with a headless browser at 1440 × 960, 2× scale.
