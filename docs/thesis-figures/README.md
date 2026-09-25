# Thesis figures

Every figure below comes from the committed model and the real SGCC data. Results
charts have a `.png` (300 dpi) and a `.pdf` (vector: use it in LaTeX). Screenshots
are 2× resolution PNGs of the running console.

The protocol is the research proposal's. Customers are split 70/15/15 into training
(29,660), validation (6,356) and test (6,356; 542 thefts). Tuning, early stopping,
thresholds and the choice of the served pipeline use only training and validation
customers; the test customers are scored once. The console serves a 3,000-customer
sample of the test set (256 thefts), so console screenshots show slightly different
counts from the Chapter 5 charts.

`docs/proposal-alignment.md` maps each objective and research question to these
figures. Suggested chapters: 3 Methodology, 4 System design and implementation,
5 Results and discussion. Renumber to match your template.

## Chapter 3: Data and methodology

| Figure | File | Where | Caption and what to say |
|---|---|---|---|
| 3.1 | `fig-3-1-system-architecture` | 3.1 Overview, or open Chapter 4 with it | *End-to-end architecture: offline training (cleaning, features, 70/15/15 split, SMOTE+ENN inside CV folds, Optuna-tuned XGBoost, selection on validation), committed artifacts, the FastAPI serving layer with its automated scoring pipeline, and the operations console.* |
| 3.2 | `fig-3-2-example-consumption` | 3.4 Dataset description | *Monthly mean consumption of an honest and a theft customer from the test set, with the served model's probabilities. Shaded months have no readings.* |
| 3.3 | `fig-3-3-missing-readings-by-class` | 3.6 Data quality | *Share of days without a meter reading, by class, across all 42,372 customers.* 25.6% of all readings are missing (31.5% for theft customers, 25.1% for honest ones), and no customer is complete. Justifies keeping missingness as a feature and not applying the >10% exclusion rule. |

Tables for Chapter 3: the Appendix A tables are filled in `docs/data-quality.md`
(`python scripts/data_quality.py`).

## Chapter 4: System design and implementation

| Figure | File | Where | Caption and what to say |
|---|---|---|---|
| 4.1 | `fig-4-1-command-center` | 4.x Operations console overview | *Command center: headline figures, the last scoring run, the investigation queue ranked by theft probability, the risk distribution and outcomes at the operating threshold.* |
| 4.2 | `fig-4-2-case-files` | 4.x Case management workflow | *Case files: every flagged customer becomes a case that moves from new → reviewing → dispatched → confirmed or cleared, each with its strongest model signal.* |
| 4.3 | `fig-4-3-case-file` | 4.x Explainable case view (proposal 3.12) | *Case file for the highest-risk customer: probability gauge, reading facts, consumption history, SHAP explanation, the LIME second opinion and case activity.* When SHAP and LIME disagree the case is flagged for manual review, as section 3.12 requires. |
| 4.4 | `fig-4-4-consumption-daily` | same section, or 3.4 | *Daily consumption for the same customer; shaded stretches have no readings.* |
| 4.5 | `fig-4-5-pipeline-scoring` | 4.x Automated scoring pipeline | *Scoring pipeline: ingest → features (built exactly as in training) → score → explain → route, with per-stage timings and run history.* About 3 s for 3,000 customers. |
| 4.6 | `fig-4-6-pipeline-training` | 4.x Model lifecycle | *The training pipeline's recorded stages: load, clean, features, split, resample, tune, validate, evaluate, publish.* Training replaces the served model, so it runs from the command line only. |
| 4.7 | `fig-4-7-threshold-studio` | 4.x Operating-point selection | *Threshold studio at the trained threshold (0.499): inspections, thefts caught, hit rate and wasted visits.* |
| 4.8 | `fig-4-8-threshold-studio-0.75` | next to 4.7 | *The same view at τ = 0.75: fewer, surer inspections at the cost of recall.* The threshold is an operational decision. |
| 4.9 | `fig-4-9-model-performance` | 4.x Monitoring in the console | *Model performance page: the five pipelines with cost and significance, the effect of SMOTE+ENN on the training data, and the features that drive the score.* |
| 4.10 | `fig-4-10-reports-scoring` | 4.x Scoring API and reporting | *Reports and scoring: PDF reports for the portfolio, an uploaded dataset or a case; a file of raw SGCC meter data checked against the model; single-customer scoring.* |
| 4.11 | `fig-4-11-settings-api-key` | 4.x Security | *API-key configuration: the key is entered per browser and never shipped in the frontend bundle.* Also mention the constant-time key check, upload limits, path-safe files, and that there is no retraining through the API. |
| 4.12 | `fig-4-12-mobile-case-file` | 4.x Responsive design (optional) | *The case file on a phone-sized screen.* |

## Chapter 5: Results and discussion

| Figure | File | Research question | Caption and what to say |
|---|---|---|---|
| 5.13 | `fig-5-13-accuracy-paradox` | RQ1 / SO2 | *Accuracy hides missed thefts.* Flagging nobody scores 91.5% accuracy with zero recall. Default XGBoost at τ = 0.5 reaches 92.9% accuracy but catches only 27.7% of thefts. Open the results chapter with it. |
| 5.9 | `fig-5-9-resampling-effect` | RQ2 / SO1 | *What SMOTE and SMOTE+ENN do to the training data.* SMOTE adds 11,035 synthetic thefts and the silhouette falls (0.073 → 0.018). ENN then removes 4,539 honest boundary rows: silhouette 0.032, mean Fisher ratio 0.014 → 0.022, and theft rows with honest-majority neighbours fall from 90% to 23%. |
| 5.10 | `fig-5-10-resampling-pca` | RQ2 / SO1 | *The training data in its first two principal components, before and after SMOTE+ENN.* |
| 5.15 | `fig-5-15-tuning-history` | RQ3 / SO3 | *Optuna search, 40 trials per pipeline, 5-fold CV on training customers (resampling inside each fold).* Best CV PR-AUC 0.496 without resampling, 0.456 with SMOTE+ENN. |
| 5.14 | `fig-5-14-learning-curves` | RQ3, section 3.10 | *Training and validation PR-AUC by boosting round, with early stopping.* SMOTE+ENN training rows are almost perfectly separable (PR-AUC near 1.0) while validation stays near 0.44: synthetic rows are easier than real customers, which is the overfitting risk section 3.10 names. |
| 5.11 | `fig-5-11-model-comparison` | RQ4 / SO4 | *All five pipelines on the test customers, each at its validation-chosen threshold.* The proposed framework has the best recall (0.494) and G-Mean (0.677); standard XGBoost the best PR-AUC (0.506), precision, F1 and MCC. |
| 5.16 | `fig-5-16-fold-comparison` | RQ4 / SO4 | *10-fold cross-validation, one dot per fold.* Paired t-tests (Holm, α = 0.05) in `artifacts/significance.json`: versus standard XGBoost the proposed pipeline is significantly higher on recall and G-Mean and lower on PR-AUC, ROC-AUC, precision, F1 and MCC. |
| 5.12 | `fig-5-12-computational-cost` | RQ4 / SO4 | *Training time, inference time per customer and model size on a 4-core CPU.* All XGBoost pipelines train in seconds and score a customer in about 2 microseconds; random forest is the slowest and largest (177 MB). |
| 5.1 | `fig-5-1-roc-curves` | RQ4 | *ROC curves of the five pipelines on the test set.* |
| 5.2 | `fig-5-2-precision-recall-curves` | RQ4 | *Precision-recall curves; the dashed line is the 8.5% base rate.* PR-AUC is the headline metric for an 8.5% minority class. |
| 5.3 | `fig-5-3-confusion-matrix` | RQ4 | *The served pipeline at τ = 0.499: 231 thefts caught, 203 false alarms, 311 missed, 5,611 correctly cleared.* |
| 5.4 | `fig-5-4-threshold-tradeoff` | discussion | *Precision and recall across thresholds.* The threshold is a business choice (inspection budget vs thefts missed). |
| 5.6 | `fig-5-6-ablation` | discussion | *Ablation, one change at a time (5-fold CV, all 42,372 customers).* Features add the most (+0.061 PR-AUC from 25 to 87); tuning +0.034; cleaning −0.019; SMOTE −0.037; SMOTE+ENN −0.076. |
| 5.5 | `fig-5-5-shap-importance` | section 3.12 | *Top 15 features by mean absolute SHAP value on 2,000 test customers.* |
| 5.8 | `fig-5-8-shap-summary` | section 3.12 | *SHAP summary of the top 20 features: each dot a customer, colour the feature value.* High day-to-day volatility and long reporting gaps push towards theft; few missing reads early in the period followed by gaps later is typical of a meter that stops reporting. |
| 5.7 | `fig-5-7-shap-waterfall-case` | section 3.12 | *SHAP waterfall for one flagged customer, from the base rate to its probability.* |

Tables for Chapter 5: the comparison and ablation tables in the repository README,
and the per-metric p-values in `artifacts/significance.json`.

Limitations to state: SGCC labels are noisy; one utility, 2014–2016, China; synthetic
SMOTE rows are not real consumption series; SHAP and LIME agree on only about half
of the highest-risk cases; no validation on a Kenyan utility's data.

## Regenerating

```bash
python scripts/download_data.py          # full dataset, once
python -m src.train                      # model and artifacts (~20 min, 4 cores)
python scripts/significance.py           # figure 5.16 data (~20 min)
python scripts/ablation.py               # figure 5.6 (~25 min); --plot-only redraws it
python scripts/data_quality.py           # docs/data-quality.md
python scripts/make_thesis_figures.py    # figures 3.2, 3.3, 5.1–5.5 and 5.8–5.16
```

The screenshots (4.x, 5.7) and the architecture diagram (3.1) were captured from the
running console with a headless browser at 1440 × 960, 2× scale.
