# Proposal alignment

How the system implements the research proposal *A Hybrid Ensemble-Based Machine
Learning Framework for Electricity Theft Detection in Smart Grids under Severe
Class Imbalance* (C004/600201/2024), where every objective's evidence lives, and
what in the proposal text must be corrected before it becomes the thesis.

Numbers below come from the committed artifacts; re-running the scripts
reproduces them.

## 1. Outcome

The proposed framework (cleaned series → SMOTE+ENN on training rows → tuned
XGBoost) was built exactly as specified. It was compared under identical conditions
with standard XGBoost, random forest + SMOTE, logistic regression + SMOTE, and an
untreated default XGBoost.

On the 6,356 untouched test customers (542 thieves), each pipeline calibrated on
validation and scored at its own validation-chosen threshold:

| Pipeline | ROC-AUC | PR-AUC (95% CI) | Recall | Precision | F1 | MCC | G-Mean |
|---|---|---|---|---|---|---|---|
| SMOTE+ENN + XGBoost (proposed) | 0.828 | 0.462 (0.422–0.502) | **0.494** | 0.389 | 0.435 | 0.379 | **0.677** |
| XGBoost, no resampling | **0.850** | **0.513** (0.474–0.553) | 0.467 | **0.487** | **0.476** | **0.429** | 0.667 |
| XGBoost, default settings | 0.839 | 0.488 (0.448–0.528) | 0.483 | 0.439 | 0.460 | 0.408 | 0.675 |
| Random forest + SMOTE | 0.812 | 0.411 (0.373–0.451) | 0.432 | 0.366 | 0.396 | 0.336 | 0.634 |
| Logistic regression + SMOTE | 0.754 | 0.314 (0.278–0.354) | 0.371 | 0.318 | 0.342 | 0.277 | 0.586 |

Significance comes from a paired stratified bootstrap of the test customers
(10,000 resamples, Holm-corrected across the four comparisons of each metric) and
McNemar's exact test on the flag decisions (`artifacts/significance.json`, Fig. 5.16):

- **Against standard XGBoost**, the proposed framework is significantly worse on
  PR-AUC (−0.051, 95% CI −0.076 to −0.027, p < 0.001), ROC-AUC (−0.022), precision
  (−0.098), F1 (−0.041) and MCC (−0.050). Its recall is higher (+0.028), but the
  interval includes zero (−0.009 to +0.065, p = 0.31).
- **Against untreated default XGBoost**, no ranking difference is significant
  (PR-AUC −0.026, p = 0.07); precision is lower (−0.050, p = 0.002).
- **Against both SMOTE baselines** it is significantly better on PR-AUC, ROC-AUC, F1,
  recall and MCC. Against logistic regression + SMOTE it is better on every metric.

SMOTE+ENN therefore does not give a better model. Among the five, standard XGBoost has
the higher validation PR-AUC, which is the selection rule fixed before looking at the
test set. Section 3.13 of the proposal commits to reporting this outcome
as it is.

**Stress tests** (figures 5.21–5.25) sharpen the finding:

- The proposed pipeline changes two things, cleaning and SMOTE+ENN. With SMOTE+ENN on
  *raw* readings, tuned the same way, the test PR-AUC is 0.504, level with standard
  XGBoost (no metric differs significantly) and significantly above the proposed pipeline
  (+0.042). The deficit comes from the cleaning, which fills the gaps that carry signal.
- No resampling ratio, ENN neighbourhood or variant (SMOTE-Tomek, Borderline-SMOTE,
  ADASYN, undersampling) beats no resampling in cross-validation.
- On six random splits standard XGBoost stays first among the five pipelines, and the
  proposed pipeline's recall lead on the study's split disappears (0.444 against 0.471).
- The model uses missing readings but does not depend on them: consumption behaviour
  alone reaches a CV PR-AUC of 0.400 (chance 0.085).
- A Wide & Deep CNN in the style of Zheng et al. (2018), trained on a CPU, beats standard
  XGBoost on all six splits (PR-AUC 0.563 ± 0.017 against 0.510 ± 0.007). For RQ4 this
  means the tuned gradient-boosting model is not the ceiling: a model that reads the daily
  series directly does better at a modest CPU cost.

**The served model is a hybrid ensemble** (Fig. 5.26, `artifacts/hybrid_study.json`).
The CNN and standard XGBoost make different mistakes, so the console serves
0.55 × CNN + 0.45 × XGBoost (weight chosen on validation PR-AUC), recalibrated with Platt
scaling and thresholded on validation, by the same selection rule (validation PR-AUC
0.636 against 0.561 for the CNN and 0.501 for XGBoost). On the test customers:

| Pipeline | ROC-AUC | PR-AUC (95% CI) | Recall | Precision | F1 | MCC | G-Mean |
|---|---|---|---|---|---|---|---|
| Wide & Deep CNN | 0.880 | 0.538 (0.498–0.579) | 0.520 | 0.516 | 0.518 | 0.473 | 0.705 |
| **Hybrid: CNN + XGBoost (served)** | **0.908** | **0.617** (0.579–0.654) | **0.611** | **0.525** | **0.564** | **0.522** | **0.761** |

Against standard XGBoost the hybrid is significantly better on every metric (PR-AUC
+0.104, 95% CI +0.078 to +0.129; recall +0.144; precision +0.038; McNemar p = 0.039);
against the CNN alone on PR-AUC (+0.078), ROC-AUC, recall, F1 and MCC, but not precision.
On six random splits it is highest every time (PR-AUC 0.633 ± 0.012). This fulfils the
proposal's title, a *hybrid ensemble*, in a different form from the one proposed: the
ensemble that works combines two views of the customer (engineered features and the raw
daily series), not resampling with boosting.

An earlier version of this analysis used paired t-tests over ten cross-validation
folds of the training and validation customers. It was replaced because those
customers had already been used to choose the hyperparameters and thresholds, and the
folds' training sets overlap, so the t-test's independence assumption fails. That
earlier analysis reported a significant recall gain (+0.075) that does not hold on
the test customers.

**Calibration** (`artifacts/calibration.json`, Fig. 5.17): resampling inflates the
proposed pipeline's raw scores. Its test ECE is 0.093 and Brier score 0.080 before
calibration, and 0.007 and 0.060 after Platt scaling fitted on validation. Standard
XGBoost goes from an ECE of 0.010 to 0.007, and the served hybrid from 0.025 to 0.004. The console shows calibrated
probabilities, so "0.30" means about 30 thieves in 100 such customers.

## 2. Objectives and research questions → evidence

| Proposal item | What answers it | Where |
|---|---|---|
| **SO1 / RQ2**: effect of SMOTE+ENN on training-data quality (distribution, separability, boundary noise) | Class counts before / after SMOTE / after ENN, synthetic rows created and removed, silhouette score, Fisher discriminant ratio, boundary-noise rate; what the inflated raw scores do to calibration ; SMOTE+ENN on raw readings tuned like the other pipelines (isolates resampling from cleaning) and a grid of ratios, ENN neighbourhoods and variants | `artifacts/resampling.json`, `calibration.json`, `resampling_study.json`; Fig. 5.9, 5.10, 5.17, 5.21; Research evaluation page |
| **SO2 / RQ1**: effect of class imbalance and limits of current approaches | Accuracy paradox (flag-nobody = 91.5% accuracy, 0 recall; untreated XGBoost at τ = 0.5); ablation rows F (no scale_pos_weight) and G (default hyperparameters) | Fig. 5.13; `artifacts/ablation.json`, Fig. 5.6 |
| **SO3 / RQ3**: design the hybrid SMOTE+ENN + optimised XGBoost; effect of hyperparameter optimisation | `src/experiment.py` (the five pipelines), Optuna history, learning curves with early stopping, tuned vs default XGBoost | `artifacts/tuning.json`, `learning_curves.json`; Fig. 5.14, 5.15 |
| **SO4 / RQ4**: compare with standard XGBoost, RF + SMOTE, LR + SMOTE on recall, precision, F1, ROC-AUC, PR-AUC, G-Mean, MCC and computational efficiency | Test-set comparison of all pipelines with training time, inference time per customer and model size; paired stratified bootstrap intervals and tests (10,000 resamples, α = 0.05, Holm-adjusted) and McNemar's exact test on the saved test predictions; inspection workload against thefts found ; the comparison repeated on six random splits; MAP@100/200 and top-share precision as reported in the SGCC literature; a Wide & Deep CNN in the style of Zheng et al. (2018) trained on a CPU; the served hybrid (CNN + XGBoost) with its paired tests and six-split check | `models/baselines/comparison_results.json`, `artifacts/predictions/test.csv.gz`, `artifacts/significance.json`, `robustness.json`, `literature_metrics.json`, `deep_baseline.json`, `hybrid_study.json`; Fig. 5.1, 5.2, 5.11, 5.12, 5.16, 5.18, 5.22, 5.25, 5.26 |
| §3.6 / Appendix A.1: data quality checklist | Every table of Appendix A.1 filled from the data | `docs/data-quality.md`, `artifacts/data_quality.json` |
| §3.7: preprocessing | `src/preprocessing.py` (gaps ≤ 3 days interpolated, longer gaps filled, cap at 10,000 kWh, 3σ → median, zero runs > 30 days flagged) with a log of every action | `artifacts/preprocessing_log.json` |
| §3.8 / Appendix A.2: features in four groups; trend features | Every feature tagged statistical / temporal / trend / anomaly; 25-feature core set; overall trend, 180-day trend, trend reversals in 30-day windows | `src/feature_catalog.py`, `src/features.py`; ablation row A |
| §3.9: SMOTE (0.5, k = 5) + ENN (k = 3), training split only | `src/resampling.py`; redone inside every CV fold, never on validation or test rows (tested) | `tests/test_preprocessing.py` |
| §3.10: tuning with stratified 5-fold CV, scale_pos_weight initialised at the imbalance ratio, early stopping, overfitting check | `src/modeling.py`: the objective is the mean PR-AUC of five separately scored folds; first trial fixes scale_pos_weight = 10.72; early stopping (50 rounds) on validation; learning curves | `artifacts/tuning.json` (fold scores); Fig. 5.14, 5.15 |
| §3.10 (added): probability calibration | Platt scaling fitted on validation for every pipeline; Brier, log-loss and ECE before and after, isotonic for comparison; thresholds chosen on calibrated validation probabilities | `src/calibration.py`, `artifacts/calibration.json`; Fig. 5.17 |
| §3.11: significance testing, α = 0.05; computational efficiency | `scripts/significance.py` on the test predictions saved by training (nothing refitted); timings and sizes in the comparison | Fig. 5.12, 5.16 |
| §3.12: deployment; ranked list; probability; SHAP local and global; LIME as second layer, disagreement flagged for human review | FastAPI + React console on an unlabelled population; the hybrid's calibrated probability with its CNN and XGBoost parts per case, a SHAP waterfall of the XGBoost part and the weeks that raised the CNN part (occlusion, log-odds); SHAP summary; explanation-consistency check (LIME) on every case page and in the case PDF, stated as a check of stability, not of causation | Fig. 4.x (4.3, 4.14), 5.5, 5.7, 5.8 |
| §3.13: responsible use (flag, not punish; appeal) | Statement on every case page and PDF; resolving a case needs a supervisor, a finding and an evidence reference; reopening (appeal) needs a reason; every action is attributed and audit-logged | Case file page, reports, audit log |

## 3. What changed in the system

- **Pipeline** (`src/`): `preprocessing.py`, `resampling.py`, `pipeline.py`,
  `experiment.py`, `calibration.py`, `stats.py` and `publish.py`. `train.py` follows
  the proposal protocol end to end. It saves every pipeline's validation and test
  predictions, freezes the feature settings, feature list, calibration, threshold, code
  commit, data hash and library versions in `artifacts/pipeline.json`, and publishes
  through a staging directory with a SHA-256 manifest written last. `--quick` writes
  to `artifacts/quick/` only.
- **Evidence scripts** (`scripts/`): `significance.py` (paired bootstrap and McNemar on
  the test predictions), `ablation.py` (per-fold mean ± SD on the development
  customers), `data_quality.py`, and `make_thesis_figures.py`, which reads the saved
  predictions instead of refitting and adds figures 5.16–5.18.
- **Research and operations are separate.** The console's population carries no
  labels. Operations show flags, tiers and expected thefts. The threshold studio uses
  validation customers and a cost model. The test set appears only on the Research
  evaluation page and in the research PDF, always named as such.
- **State and access**: PostgreSQL and S3-compatible storage; named API keys with
  analyst and supervisor roles; an audit log; a case workflow with enforced steps.
- **Training on a GPU**: `python -m src.train --device cuda` and
  `python scripts/ablation.py --device cuda` (XGBoost only; the CPU run remains the
  reference, as the proposal targets CPU-only utilities).

## 4. Corrections to make in the proposal text

Page numbers are the PDF's printed page numbers.

### Facts the data contradicts

| Where | Proposal says | Correct statement |
|---|---|---|
| Abstract; §1.6; §3.4; §3.5 | 1,035 days | 1,034 daily columns: the range 1 Jan 2014 – 31 Oct 2016 spans 1,035 days, but **2016-09-18 is absent** from the release |
| §3.6 (p. 48) | missing values "below 0.01%" | **25.64%** of customer-days are missing; **no customer** has a complete record; theft customers miss more days (31.5%) than honest ones (25.1%) |
| §3.7 (p. 49) | exclude customers with > 10% missing days | Not applicable: it would remove **21,375 customers (50%) and 2,240 of the 3,615 thefts (62%)** and change the imbalance ratio. State that all customers are kept and missingness is used as a feature |
| §3.7 (p. 49) | "a full 42,372 × 1,035 panel with 4 (0.06%) missing values" | 42,372 × 1,034 with 11,233,528 missing cells before cleaning; after cleaning none (see `docs/data-quality.md` VI) |
| Abstract | "forward fill … implausible values handled with mean fill" | Gaps ≤ 3 days interpolated, longer gaps forward- then back-filled; values > 10,000 kWh capped (260 readings); 3σ outliers replaced by the customer's median (413,571 readings); no negative readings exist |
| Appendix A.1 I | ~43,850,520 records, 4 columns (ID, Date, Consumption, Label), ~450 MB | 43,812,648 customer-days in a **wide** file (CONS_NO, FLAG, 1,034 date columns), 175 MB uncompressed |
| §3.6 | data downloaded from Kaggle with API credentials | The repository downloads the SGCC release from a public GitHub mirror with a checksum and no credentials (`scripts/download_data.py`); name that source, or verify the Kaggle copy is identical |
| §1.6 | 25 features | The proposal's 25-feature core set is kept (`CORE_FEATURES`); the model uses 87 features in the same four groups. The ablation (row A vs B) reports what the extra features add |

### Method statements to align with what was done

| Where | Proposal says | Correct statement |
|---|---|---|
| §3.4 (p. 46–47) | "forty percent … training, 15% validation, 15% test"; "the information is chronological throughout" | **70 / 15 / 15**, stratified by label, split **by customer** (29,660 / 6,356 / 6,356); the split is not chronological, every customer covers the same period |
| §1.6; §3.10 | randomised hyperparameter search | Optuna (TPE), 40 trials per pipeline, maximising the mean PR-AUC over 5 stratified CV folds, each scored separately (the abstract already says Optuna) |
| §3.10 | (no calibration step) | Add: each pipeline's scores are calibrated with Platt scaling on the validation customers, and its threshold is the F1-maximising one on the calibrated validation probabilities; Brier score, log-loss and expected calibration error are reported |
| §3.10 | "since the training set is already balanced 1-1" | With sampling_strategy = 0.5 the training set after SMOTE is 1 theft : 2 honest (before ENN); scale_pos_weight is searched in 1–3 after SMOTE+ENN and 1–10.72 without resampling |
| §3.10 (p. 51–52) | AUCC "hits 0"; paragraphs on adverse drug reactions and the "BooC" model | Remove: not related to this study. Replace with the overfitting check actually run: early stopping on validation PR-AUC and the learning curves (Fig. 5.14) |
| §3.11 | t-tests over "ten CV folds"; elsewhere 5-fold | 5-fold CV for tuning only. Significance on the **test customers**: a paired stratified bootstrap (10,000 resamples) giving 95% intervals and two-sided p-values for each metric difference, Holm-corrected across the comparisons, and McNemar's exact test on the paired flag decisions (α = 0.05). Fold t-tests are not used: the folds reuse the customers that chose the hyperparameters and thresholds, and their training sets overlap |
| §3.11 (p. 54) | "Flat Classifier model", "customer churn" | Remove; cite the Wide & Deep CNN of Zheng et al. (2018) instead. It was re-implemented and trains on a CPU in minutes (`scripts/deep_baseline.py`), and is part of the served hybrid |
| §3.3; §3.11; §3.12; Appendix B phase 8 | Streamlit application | A FastAPI service and React console (ranked queue, calibrated probability, SHAP and LIME per case, PDF reports, threshold control with a cost model, case workflow with roles and audit log). Everything runs on a CPU |
| §3.12 (p. 56) | SHAP summary of 20 features | Done: Fig. 5.8 (top 20, 2,000 test customers) |
| §2.3.3 (p. 24) | "LSTM (Legacy Statistical Models)" | LSTM is Long Short-Term Memory; it is not a tree-based method |
| §3.13 (p. 57) | "Compliance with the Kenyan law on clinical science" | Kenya Data Protection Act, 2019 |

### Citations to correct

| Where | Proposal says | Correct statement |
|---|---|---|
| §1.2, §2.3.2, §3.4 | Buzau et al. (2019) analysed 47 papers on the SGCC data (accuracy 90–95%, recall 45–78%) | Buzau et al. (2019) is a study of Endesa's (Spain) industrial and commercial customers in which XGBoost performed best; it is not a review of SGCC papers. Make the accuracy point from the data instead: flagging nobody is 91.5% accurate with zero recall |
| §2.3.3, §2.4.4 | Punmiya and Choe (2019) reached 77.2% recall on SGCC | They used the Irish CER smart-meter dataset |
| §2.4.1 | Zheng et al. reached 75.8% theft recall | Zheng et al. (2018) released the SGCC data and reported MAP@100 of about 96% |
| §2.4.1, §2.5.2 | Li et al. (2019) used LSTM; Gao et al. (2020) used attention | Per their titles: Li et al. combine deep learning with random forests; Gao et al. propose a physically inspired data-driven model |
| References | — | Missing: Lundberg & Lee (2017), Ribeiro et al. (2016), Creswell (2014), Shadish et al. (2002), Lemaître et al. (2017); add Akiba et al. (2019) for Optuna, Holm (1979), Efron & Tibshirani (1993) for the bootstrap, McNemar (1947), Platt (1999) for calibration, and the Kenya Data Protection Act (2019) |

### Writing to fix

- Table of contents: every "Error! Bookmark not defined." (pp. 4–6): update the Word fields.
- Figure 2.7 (p. 43): the box label "Servier Class Imbalance" should read "Severe"; as the only figure it can be numbered 2.1.
- p. 15: "CGCC benchmark" → SGCC. p. 18: "Electrocution continues to be…" (garbled). p. 49: "pandasvale", "PLN study", "a dip coma" (garbled).
- p. 58: "bacterial discrimination", "The First Law", "Treasonous courts", "But do I care? The Columbia Encyclopedia…": rewrite the paragraph.
- p. 19–20: §1.6 last paragraph and §1.7 first paragraph are hard to follow; restate the limitations plainly (single dataset, noisy labels, no demographics, China vs Kenya).
- Keywords list "SMOTE" twice.
- List of Tables names a "Table 2.6" that does not exist.
- Appendix C: unit cost × quantity does not give the totals (Internet 3,000 × 12 months = 3,000; printing 3,000 × 5 copies = 3,000; binding 1,500 × 5 = 1,500), and "(you'll use part of it)" is a leftover note.

A corrected copy of the whole proposal, with every change highlighted in yellow and a removable summary page, is in
`docs/proposal/C004_600201_2024_Machimbo_proposal_corrected.docx`.

## 5. Limitations to state in the thesis

- One dataset (SGCC, 2014–2016, China); no validation on a Kenyan utility.
- The split is random by customer, so the test set measures performance on unseen
  customers from the same utility and period. SGCC labels are per customer, so a
  chronological test is not possible, and no second labelled utility is available.
  There is no evidence yet of performance on later periods or other utilities.
- Labels come from field inspections and are noisy.
- SMOTE interpolates in feature space: synthetic "customers" are not real consumption series.
- The console's default population is an unlabelled 3,000-customer sample of the test
  customers. A supervisor can replace it with uploaded readings.
- Calibration is fitted on 6,356 validation customers; it transfers to a new utility
  only if its theft rate is similar.
- Timings are for a 4-core CPU; they scale with hardware.
- SHAP and LIME agree (≥ 3 of the top 5 signals shared and the same direction on the
  strongest one) on 19 of the 30 highest-risk customers of the sample population. The
  console shows disagreement instead of hiding it. Agreement means a stable
  explanation, not a causal one.
