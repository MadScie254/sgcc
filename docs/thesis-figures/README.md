# Thesis figures

Every figure below comes from the committed model and the real SGCC data. The console
serves the hybrid ensemble (CNN + XGBoost); the Chapter 5 figures made from the saved
predictions show the study's five pipelines, where standard XGBoost is the best. Results
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
| 3.2 | `fig-3-2-example-consumption` | 3.4 Dataset description | *Monthly mean consumption of an honest and a theft customer from the test set, with standard XGBoost's probabilities. Shaded months have no readings.* |
| 3.3 | `fig-3-3-missing-readings-by-class` | 3.6 Data quality | *Share of days without a meter reading, by class, across all 42,372 customers.* 25.6% of all readings are missing (31.5% for theft customers, 25.1% for honest ones), and no customer is complete. Justifies keeping missingness as a feature and not applying the >10% exclusion rule. |

Tables for Chapter 3: the Appendix A tables are filled in `docs/data-quality.md`
(`python scripts/data_quality.py`).

## Chapter 4: System design and implementation

| Figure | File | Where | Caption and what to say |
|---|---|---|---|
| 4.1 | `fig-4-1-command-center` | 4.x Operations console overview | *Command center: customers flagged at the operating threshold, expected thefts among them (sum of calibrated probabilities), the last scoring run, the investigation queue, the risk distribution, and what the threshold means on validation customers.* No labels appear in operations. |
| 4.2 | `fig-4-2-case-files` | 4.x Case management workflow | *Case files: every flagged customer becomes a case, with its strongest model signal and who last changed it.* |
| 4.3 | `fig-4-3-case-file` | 4.x Explainable case view (proposal 3.12) | *Case file for the highest-risk customer: the hybrid's calibrated probability with its CNN and XGBoost parts, reading facts, consumption history, SHAP explanation of the XGBoost part's raw score, the explanation-consistency check (LIME) and case activity.* Only the workflow steps allowed for the user's role are offered. |
| 4.4 | `fig-4-4-consumption-daily` | same section, or 3.4 | *Daily consumption for the same customer; shaded stretches have no readings.* |
| 4.14 | `fig-4-14-sequence-weeks` | 4.x Explainable case view, next to 4.3 | *What the sequence model saw: for the same customer, how much each week of readings raised the CNN part's score, in log-odds (each week replaced in turn by the customer's typical day).* SHAP explains the XGBoost part; this occlusion view explains the CNN part. |
| 4.13 | `fig-4-13-case-resolution` | 4.x Case management (proposal 3.13) | *A supervisor records the inspection outcome: a finding and an evidence reference are required to confirm theft or clear a customer.* Cases move new → reviewing → dispatched → confirmed or cleared; analysts review and dispatch, supervisors resolve and reopen (appeals), and every step is attributed. |
| 4.5 | `fig-4-5-pipeline-scoring` | 4.x Automated scoring pipeline | *Scoring pipeline: ingest → features (with the settings frozen at training) → score → explain → route, with per-stage timings, run history and who started each run.* About 3 s for 3,000 customers. |
| 4.6 | `fig-4-6-pipeline-training` | 4.x Model lifecycle | *The training pipeline's recorded stages: load, clean, features, split, resample, tune, validate, sequence (the CNN and the hybrid), evaluate, publish; with the code commit and data hash it was trained from.* Training replaces the served model, so it runs from the command line only; the API verifies the published files against their SHA-256 manifest at startup. |
| 4.7 | `fig-4-7-threshold-studio` | 4.x Operating-point selection | *Threshold studio at the trained threshold (0.216): precision and recall on the validation customers the threshold was chosen on, and the workload in the population.* The test customers are never used to set a threshold. |
| 4.8 | `fig-4-8-threshold-studio-cost` | next to 4.7 | *The same view with an inspection budget (150 visits, 30 per visit, 400 recovered per theft): the net-value curve and the threshold that maximises it.* The threshold is an operational decision; only a supervisor can publish it. |
| 4.9 | `fig-4-9-research-evaluation` | 4.x / 5.x Research record in the console | *Research evaluation page: test-set metrics of the served hybrid, the five study pipelines with cost and significance, the CNN and the hybrid, calibration, the effect of SMOTE+ENN, the score's drivers and the training provenance.* Named as the test set throughout, with the limitation stated. |
| 4.10 | `fig-4-10-reports-scoring` | 4.x Scoring API and reporting | *Reports and scoring: operations and research PDFs; a file of SGCC meter data checked against the model; single-customer scoring.* |
| 4.11 | `fig-4-11-settings-audit` | 4.x Security | *Settings: the user's named key (kept for the browser tab only), service health, and the audit log of who published thresholds, changed cases and handled files.* Keys are stored as SHA-256 hashes with a role; requests are rate-limited. |
| 4.12 | `fig-4-12-mobile-case-file` | 4.x Responsive design (optional) | *The case file on a phone-sized screen.* The navigation drawer traps keyboard focus and closes with Escape. |

## Chapter 5: Results and discussion

| Figure | File | Research question | Caption and what to say |
|---|---|---|---|
| 5.13 | `fig-5-13-accuracy-paradox` | RQ1 / SO2 | *Accuracy hides missed thefts.* Flagging nobody scores 91.5% accuracy with zero recall. Default XGBoost flagging at a calibrated probability of 0.5 reaches 92.9% accuracy but catches only 24.2% of thefts. Open the results chapter with it. |
| 5.9 | `fig-5-9-resampling-effect` | RQ2 / SO1 | *What SMOTE and SMOTE+ENN do to the training data.* SMOTE adds 11,035 synthetic thefts and the silhouette falls (0.073 → 0.018). ENN then removes 4,539 honest boundary rows: silhouette 0.032, mean Fisher ratio 0.014 → 0.022, and theft rows with honest-majority neighbours fall from 90% to 23%. |
| 5.10 | `fig-5-10-resampling-pca` | RQ2 / SO1 | *The training data in its first two principal components, before and after SMOTE+ENN.* |
| 5.17 | `fig-5-17-reliability` | RQ2, section 3.10 | *Reliability diagrams on the test customers, raw scores and Platt-calibrated probabilities.* Resampling inflates the proposed pipeline's scores (ECE 0.093); calibration on validation brings it to 0.007. Standard XGBoost is nearly calibrated already (0.010 → 0.007); the served hybrid goes from 0.025 to 0.004. |
| 5.15 | `fig-5-15-tuning-history` | RQ3 / SO3 | *Optuna search, 40 trials per pipeline; each trial scored as the mean PR-AUC of five separately scored folds of the training customers (resampling inside each fold).* Best 0.505 without resampling, 0.457 with SMOTE+ENN. |
| 5.14 | `fig-5-14-learning-curves` | RQ3, section 3.10 | *Training and validation PR-AUC by boosting round, with early stopping.* SMOTE+ENN training rows are almost perfectly separable (PR-AUC 0.997) while validation peaks at 0.444: synthetic rows are easier than real customers, which is the overfitting risk section 3.10 names. |
| 5.11 | `fig-5-11-model-comparison` | RQ4 / SO4 | *All five pipelines on the test customers, each at its validation-chosen threshold.* The proposed framework has the best recall (0.494) and G-Mean (0.677); standard XGBoost the best PR-AUC (0.513), precision, F1 and MCC. |
| 5.16 | `fig-5-16-bootstrap-comparison` | RQ4 / SO4 | *Difference between the proposed pipeline and each other pipeline on the test customers, with 95% paired bootstrap intervals and Holm-adjusted p-values.* Against standard XGBoost the proposed pipeline is significantly worse on PR-AUC, F1 and MCC; against both SMOTE baselines significantly better; against default XGBoost no significant difference. McNemar's tests are in `artifacts/significance.json`. |
| 5.18 | `fig-5-18-workload` | RQ4, discussion | *Share of thefts found against share of customers inspected, highest probability first.* At standard XGBoost's threshold the utility inspects 8.2% of customers and finds 46.7% of thefts. |
| 5.12 | `fig-5-12-computational-cost` | RQ4 / SO4 | *Training time, inference time per customer and model size on a 4-core CPU.* All XGBoost pipelines train in seconds and score a customer in microseconds; random forest is the slowest and largest (177 MB). |
| 5.1 | `fig-5-1-roc-curves` | RQ4 | *ROC curves of the five pipelines on the test set.* |
| 5.2 | `fig-5-2-precision-recall-curves` | RQ4 | *Precision-recall curves; the dashed line is the 8.5% base rate.* PR-AUC is the headline metric for an 8.5% minority class. |
| 5.3 | `fig-5-3-confusion-matrix` | RQ4 | *Standard XGBoost, the study's best pipeline, at τ = 0.243: 253 thefts caught, 267 false alarms, 289 missed, 5,547 correctly cleared.* |
| 5.4 | `fig-5-4-threshold-tradeoff` | discussion | *Precision and recall across thresholds (calibrated probability), test customers.* The threshold is a business choice (inspection budget vs thefts missed); the console sets it on validation customers. |
| 5.6 | `fig-5-6-ablation` | discussion | *Ablation, one change at a time (5-fold CV on the 36,016 development customers, mean ± SD over folds).* Features add the most (+0.068 PR-AUC from 25 to 87); tuning +0.046; cleaning −0.015; SMOTE −0.054; SMOTE+ENN −0.087 against raw readings. |
| 5.5 | `fig-5-5-shap-importance` | section 3.12 | *Top 15 features by mean absolute SHAP value on 2,000 test customers.* |
| 5.8 | `fig-5-8-shap-summary` | section 3.12 | *SHAP summary of the top 20 features: each dot a customer, colour the feature value.* High day-to-day volatility and long reporting gaps push towards theft; few missing reads early in the period followed by gaps later is typical of a meter that stops reporting. |
| 5.7 | `fig-5-7-shap-waterfall-case` | section 3.12 | *SHAP waterfall of the XGBoost part for one flagged customer, from the average raw score to its raw score; the case page shows the hybrid's calibrated probability alongside.* |

### Stress tests of the result

| Figure | File | Research question | Caption and what to say |
|---|---|---|---|
| 5.21 | `fig-5-21-resampling-sensitivity` | RQ2 / SO1 | *Resampling choices against no resampling: 5-fold CV on the 36,016 development customers, raw readings, standard XGBoost's tuned hyperparameters held fixed.* Every treatment is below no resampling (0.507), more so the more it rebalances; ENN's neighbourhood makes no difference. Tuned separately and scored on the test customers, SMOTE+ENN on raw readings (0.504) is level with standard XGBoost (0.513; no metric differs significantly) and well above the proposed pipeline on cleaned readings (0.462): the proposed pipeline's deficit comes from the cleaning step. `artifacts/resampling_study.json` |
| 5.22 | `fig-5-22-split-stability` | RQ4 / SO4, validity | *Test PR-AUC of every pipeline on six random 70/15/15 splits (circled: the study's split), tuned hyperparameters held fixed.* Standard XGBoost is first on all six (0.510 ± 0.007). The proposed pipeline's recall lead on the study's split does not hold across splits. `artifacts/robustness.json` |
| 5.23 | `fig-5-23-missingness` | validity, section 3.12 | *How much the model relies on missing readings.* Without the eight missing-reading features 0.468; consumption behaviour alone (gaps filled, no missingness left) 0.400; the missing-reading features alone 0.333; chance 0.085. The model uses both, and behaviour alone detects theft far above chance. |
| 5.24 | `fig-5-24-history-length` | practice | *Detection quality with only the most recent 3 to 34 months of readings (standard XGBoost).* PR-AUC 0.274 (3 months), 0.319 (12), 0.396 (24), 0.513 (34). `artifacts/history_length.json` |
| 5.25 | `fig-5-25-extension-pr-curves` | RQ4 / literature | *Precision-recall curves of standard XGBoost, the proposed pipeline, raw-readings SMOTE+ENN and the Wide & Deep CNN on the test customers.* The CNN (Zheng et al., 2018 style, trained on a CPU) is below XGBoost at the very top of the list and above it from about 35% recall on. `artifacts/deep_baseline.json` |
| 5.26 | `fig-5-26-hybrid-splits` | RQ4 / SO4, the served model | *The hybrid ensemble (0.55 × CNN + 0.45 × XGBoost, recalibrated) against its two parts on six random splits (CNN weight 0.55 on every split).* The hybrid is highest on all six: PR-AUC 0.633 ± 0.012, against 0.563 ± 0.017 (CNN) and 0.510 ± 0.007 (XGBoost). On the study's split it reaches 0.617 (95% CI 0.579–0.654), significantly above XGBoost (+0.104) and the CNN (+0.078). `artifacts/hybrid_study.json` |

Table for the literature comparison: MAP@100/200, precision in the top 1% and 5%, recall
in the top 5% and 10%, and value under an inspection budget for every pipeline, with
bootstrap intervals, in `artifacts/literature_metrics.json`.

Tables for Chapter 5: the comparison (with bootstrap intervals), calibration and
ablation tables in the repository README, and every interval and p-value in
`artifacts/significance.json`.

Limitations to state: SGCC labels are noisy; one utility, 2014–2016, China; the split
is random by customer, so there is no evidence of performance on later periods or other
utilities; synthetic SMOTE rows are not real consumption series; SHAP and LIME agree about
the XGBoost part on 16 of the 30 highest-risk cases, which says the explanation is stable,
not causal; the hybrid was designed after the CNN's test result was known (its weight,
calibration and threshold were chosen on validation, and the six-split check supports it);
the CNN, like XGBoost, uses reporting gaps, and the missing 2016-09-18 release date makes
September 2016 a strong week for some customers.

## Regenerating

```bash
python scripts/download_data.py          # full dataset, once
python -m src.train                      # model, artifacts, saved predictions (~25 min, 4 cores; --device cuda on a GPU)
python scripts/significance.py           # figure 5.16 data (~1 min, no refitting)
python scripts/ablation.py               # figure 5.6 (~20 min; --device cuda); --plot-only redraws it
python scripts/data_quality.py           # docs/data-quality.md
python scripts/make_thesis_figures.py    # figures 3.2, 3.3, 5.1–5.5 and 5.8–5.18 (from the saved predictions)
python scripts/resampling_study.py       # figure 5.21 (~20 min; --device cuda)
python scripts/robustness.py             # figures 5.22, 5.23 (~10 min)
python scripts/history_length.py         # figure 5.24 (~5 min)
python scripts/deep_baseline.py          # figure 5.25 (PyTorch: pip install -r requirements-research.txt); --splits
python scripts/literature_metrics.py     # MAP@N and budget table (reads the saved predictions)
python scripts/hybrid_study.py           # figure 5.26 (~30 min); --significance: the served split's tests
```

The screenshots (4.x, 5.7) were captured from the running console (production mode,
a supervisor key) with a headless browser at 1440 × 960, 2× scale, as was the
architecture diagram (3.1).
