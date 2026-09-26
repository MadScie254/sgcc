# Thesis figures

Every figure below comes from the committed model and the real SGCC data. Results
charts have a `.png` (300 dpi) and a `.pdf` (vector: use it in LaTeX). Screenshots
are 2× resolution PNGs of the running console.

The protocol is the research proposal's. Customers are split 70/15/15 into training
(29,660), validation (6,356) and test (6,356; 542 thefts). Tuning, early stopping,
calibration, thresholds and the choice of the served pipeline use only training and
validation customers; the test customers are scored once, and those predictions are
saved (`artifacts/predictions/test.csv.gz`). Every Chapter 5 chart is drawn from them.
The console's operations pages score an unlabelled 3,000-customer sample of the test
customers and never show a label. Their counts (flagged, expected thefts) are estimates;
the test-set results are on the Research evaluation page. Probabilities are
calibrated (Platt scaling on validation).

`docs/proposal-alignment.md` maps each objective and research question to these
figures. Suggested chapters: 3 Methodology, 4 System design and implementation,
5 Results and discussion. Renumber to match your template.

## Chapter 3: Data and methodology

| Figure | File | Where | Caption and what to say |
|---|---|---|---|
| 3.1 | `fig-3-1-system-architecture` | 3.1 Overview, or open Chapter 4 with it | *End-to-end architecture: offline training (cleaning, features, 70/15/15 split, SMOTE+ENN inside CV folds, Optuna-tuned XGBoost, calibration and selection on validation, one test-set scoring with saved predictions), the published artifacts with their SHA-256 manifest, the FastAPI serving layer (scoring pipeline, PostgreSQL and object storage, named keys with roles), and the console with its operations and research views.* |
| 3.2 | `fig-3-2-example-consumption` | 3.4 Dataset description | *Monthly mean consumption of an honest and a theft customer from the test set, with the served model's probabilities. Shaded months have no readings.* |
| 3.3 | `fig-3-3-missing-readings-by-class` | 3.6 Data quality | *Share of days without a meter reading, by class, across all 42,372 customers.* 25.6% of all readings are missing (31.5% for theft customers, 25.1% for honest ones), and no customer is complete. Justifies keeping missingness as a feature and not applying the >10% exclusion rule. |

Tables for Chapter 3: the Appendix A tables are filled in `docs/data-quality.md`
(`python scripts/data_quality.py`).

## Chapter 4: System design and implementation

| Figure | File | Where | Caption and what to say |
|---|---|---|---|
| 4.1 | `fig-4-1-command-center` | 4.x Operations console overview | *Command center: customers flagged at the operating threshold, expected thefts among them (sum of calibrated probabilities), the last scoring run, the investigation queue, the risk distribution, and what the threshold means on validation customers.* No labels appear in operations. |
| 4.2 | `fig-4-2-case-files` | 4.x Case management workflow | *Case files: every flagged customer becomes a case, with its strongest model signal and who last changed it.* |
| 4.3 | `fig-4-3-case-file` | 4.x Explainable case view (proposal 3.12) | *Case file for the highest-risk customer: calibrated probability, reading facts, consumption history, SHAP explanation of the raw score, the explanation-consistency check (LIME) and case activity.* Only the workflow steps allowed for the user's role are offered. |
| 4.4 | `fig-4-4-consumption-daily` | same section, or 3.4 | *Daily consumption for the same customer; shaded stretches have no readings.* |
| 4.13 | `fig-4-13-case-resolution` | 4.x Case management (proposal 3.13) | *A supervisor records the inspection outcome: a finding and an evidence reference are required to confirm theft or clear a customer.* Cases move new → reviewing → dispatched → confirmed or cleared; analysts review and dispatch, supervisors resolve and reopen (appeals), and every step is attributed. |
| 4.5 | `fig-4-5-pipeline-scoring` | 4.x Automated scoring pipeline | *Scoring pipeline: ingest → features (with the settings frozen at training) → score → explain → route, with per-stage timings, run history and who started each run.* About 3 s for 3,000 customers. |
| 4.6 | `fig-4-6-pipeline-training` | 4.x Model lifecycle | *The training pipeline's recorded stages: load, clean, features, split, resample, tune, validate, evaluate, publish; with the code commit and data hash it was trained from.* Training replaces the served model, so it runs from the command line only; the API verifies the published files against their SHA-256 manifest at startup. |
| 4.7 | `fig-4-7-threshold-studio` | 4.x Operating-point selection | *Threshold studio at the trained threshold (0.243): precision and recall on the validation customers the threshold was chosen on, and the workload in the population.* The test customers are never used to set a threshold. |
| 4.8 | `fig-4-8-threshold-studio-cost` | next to 4.7 | *The same view with an inspection budget (150 visits, 30 per visit, 400 recovered per theft): the net-value curve and the threshold that maximises it.* The threshold is an operational decision; only a supervisor can publish it. |
| 4.9 | `fig-4-9-research-evaluation` | 4.x / 5.x Research record in the console | *Research evaluation page: test-set metrics with bootstrap intervals, the five pipelines with cost and significance, calibration, the effect of SMOTE+ENN, the score's drivers and the training provenance.* Named as the test set throughout, with the limitation stated. |
| 4.10 | `fig-4-10-reports-scoring` | 4.x Scoring API and reporting | *Reports and scoring: operations and research PDFs; a file of SGCC meter data checked against the model; single-customer scoring.* |
| 4.11 | `fig-4-11-settings-audit` | 4.x Security | *Settings: the user's named key (kept for the browser tab only), service health, and the audit log of who published thresholds, changed cases and handled files.* Keys are stored as SHA-256 hashes with a role; requests are rate-limited. |
| 4.12 | `fig-4-12-mobile-case-file` | 4.x Responsive design (optional) | *The case file on a phone-sized screen.* The navigation drawer traps keyboard focus and closes with Escape. |

## Chapter 5: Results and discussion

| Figure | File | Research question | Caption and what to say |
|---|---|---|---|
| 5.13 | `fig-5-13-accuracy-paradox` | RQ1 / SO2 | *Accuracy hides missed thefts.* Flagging nobody scores 91.5% accuracy with zero recall. Default XGBoost flagging at a calibrated probability of 0.5 reaches 92.9% accuracy but catches only 24.2% of thefts. Open the results chapter with it. |
| 5.9 | `fig-5-9-resampling-effect` | RQ2 / SO1 | *What SMOTE and SMOTE+ENN do to the training data.* SMOTE adds 11,035 synthetic thefts and the silhouette falls (0.073 → 0.018). ENN then removes 4,539 honest boundary rows: silhouette 0.032, mean Fisher ratio 0.014 → 0.022, and theft rows with honest-majority neighbours fall from 90% to 23%. |
| 5.10 | `fig-5-10-resampling-pca` | RQ2 / SO1 | *The training data in its first two principal components, before and after SMOTE+ENN.* |
| 5.17 | `fig-5-17-reliability` | RQ2, section 3.10 | *Reliability diagrams on the test customers, raw scores and Platt-calibrated probabilities.* Resampling inflates the proposed pipeline's scores (ECE 0.093); calibration on validation brings it to 0.007. The served pipeline is nearly calibrated already (0.010 → 0.007). |
| 5.15 | `fig-5-15-tuning-history` | RQ3 / SO3 | *Optuna search, 40 trials per pipeline; each trial scored as the mean PR-AUC of five separately scored folds of the training customers (resampling inside each fold).* Best 0.505 without resampling, 0.457 with SMOTE+ENN. |
| 5.14 | `fig-5-14-learning-curves` | RQ3, section 3.10 | *Training and validation PR-AUC by boosting round, with early stopping.* SMOTE+ENN training rows are almost perfectly separable (PR-AUC 0.997) while validation peaks at 0.444: synthetic rows are easier than real customers, which is the overfitting risk section 3.10 names. |
| 5.11 | `fig-5-11-model-comparison` | RQ4 / SO4 | *All five pipelines on the test customers, each at its validation-chosen threshold.* The proposed framework has the best recall (0.494) and G-Mean (0.677); standard XGBoost the best PR-AUC (0.513), precision, F1 and MCC. |
| 5.16 | `fig-5-16-bootstrap-comparison` | RQ4 / SO4 | *Difference between the proposed pipeline and each other pipeline on the test customers, with 95% paired bootstrap intervals and Holm-adjusted p-values.* Against standard XGBoost the proposed pipeline is significantly worse on PR-AUC, F1 and MCC; against both SMOTE baselines significantly better; against default XGBoost no significant difference. McNemar's tests are in `artifacts/significance.json`. |
| 5.18 | `fig-5-18-workload` | RQ4, discussion | *Share of thefts found against share of customers inspected, highest probability first.* At the served threshold the utility inspects 8.2% of customers and finds 46.7% of thefts. |
| 5.12 | `fig-5-12-computational-cost` | RQ4 / SO4 | *Training time, inference time per customer and model size on a 4-core CPU.* All XGBoost pipelines train in seconds and score a customer in microseconds; random forest is the slowest and largest (177 MB). |
| 5.1 | `fig-5-1-roc-curves` | RQ4 | *ROC curves of the five pipelines on the test set.* |
| 5.2 | `fig-5-2-precision-recall-curves` | RQ4 | *Precision-recall curves; the dashed line is the 8.5% base rate.* PR-AUC is the headline metric for an 8.5% minority class. |
| 5.3 | `fig-5-3-confusion-matrix` | RQ4 | *The served pipeline at τ = 0.243: 253 thefts caught, 267 false alarms, 289 missed, 5,547 correctly cleared.* |
| 5.4 | `fig-5-4-threshold-tradeoff` | discussion | *Precision and recall across thresholds (calibrated probability), test customers.* The threshold is a business choice (inspection budget vs thefts missed); the console sets it on validation customers. |
| 5.6 | `fig-5-6-ablation` | discussion | *Ablation, one change at a time (5-fold CV on the 36,016 development customers, mean ± SD over folds).* Features add the most (+0.068 PR-AUC from 25 to 87); tuning +0.046; cleaning −0.015; SMOTE −0.054; SMOTE+ENN −0.087 against raw readings. |
| 5.5 | `fig-5-5-shap-importance` | section 3.12 | *Top 15 features by mean absolute SHAP value on 2,000 test customers.* |
| 5.8 | `fig-5-8-shap-summary` | section 3.12 | *SHAP summary of the top 20 features: each dot a customer, colour the feature value.* High day-to-day volatility and long reporting gaps push towards theft; few missing reads early in the period followed by gaps later is typical of a meter that stops reporting. |
| 5.7 | `fig-5-7-shap-waterfall-case` | section 3.12 | *SHAP waterfall for one flagged customer, from the average raw score to its raw score, then its calibrated probability.* |

Tables for Chapter 5: the comparison (with bootstrap intervals), calibration and
ablation tables in the repository README, and every interval and p-value in
`artifacts/significance.json`.

Limitations to state: SGCC labels are noisy; one utility, 2014–2016, China; the split
is random by customer, so there is no evidence of performance on later periods or other
utilities; synthetic SMOTE rows are not real consumption series; SHAP and LIME agree on
19 of the 30 highest-risk cases, which says the explanation is stable, not causal.

## Regenerating

```bash
python scripts/download_data.py          # full dataset, once
python -m src.train                      # model, artifacts, saved predictions (~20 min, 4 cores; --device cuda on a GPU)
python scripts/significance.py           # figure 5.16 data (~1 min, no refitting)
python scripts/ablation.py               # figure 5.6 (~20 min; --device cuda); --plot-only redraws it
python scripts/data_quality.py           # docs/data-quality.md
python scripts/make_thesis_figures.py    # figures 3.2, 3.3, 5.1–5.5 and 5.8–5.18 (from the saved predictions)
```

The screenshots (4.x, 5.7) were captured from the running console (production mode,
a supervisor key) with a headless browser at 1440 × 960, 2× scale, as was the
architecture diagram (3.1).
