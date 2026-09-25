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

On the 6,356 untouched test customers:

| Pipeline | ROC-AUC | PR-AUC | Recall | Precision | F1 | MCC | G-Mean |
|---|---|---|---|---|---|---|---|
| SMOTE+ENN + XGBoost (proposed) | 0.828 | 0.462 | **0.494** | 0.389 | 0.435 | 0.379 | **0.677** |
| XGBoost, no resampling (served) | **0.851** | **0.506** | 0.426 | **0.532** | **0.473** | **0.433** | 0.641 |
| XGBoost, default settings | 0.839 | 0.488 | 0.483 | 0.439 | 0.460 | 0.408 | 0.675 |
| Random forest + SMOTE | 0.812 | 0.411 | 0.432 | 0.366 | 0.396 | 0.336 | 0.634 |
| Logistic regression + SMOTE | 0.754 | 0.314 | 0.371 | 0.318 | 0.342 | 0.277 | 0.586 |

The 10-fold paired t-tests (Holm-adjusted, α = 0.05) give the same picture
(`artifacts/significance.json`):

- **Against standard XGBoost**, the proposed framework catches significantly more
  thefts: recall +0.075, G-Mean +0.038.
- It is significantly worse on ranking and precision: PR-AUC −0.054, ROC-AUC −0.018,
  precision −0.143, F1 −0.041, MCC −0.054.
- **Against both SMOTE baselines** it is significantly better on PR-AUC, recall and
  G-Mean. Against logistic regression + SMOTE it is better on every metric.

SMOTE+ENN therefore shifts the operating point towards recall. It does not give a
better model overall. The console serves standard XGBoost because it has the higher
validation PR-AUC, which is the selection rule fixed before looking at the test set.
Section 3.13 of the proposal commits to reporting this outcome as it is.

## 2. Objectives and research questions → evidence

| Proposal item | What answers it | Where |
|---|---|---|
| **SO1 / RQ2**: effect of SMOTE+ENN on training-data quality (distribution, separability, boundary noise) | Class counts before / after SMOTE / after ENN, synthetic rows created and removed, silhouette score, Fisher discriminant ratio, boundary-noise rate | `artifacts/resampling.json`; Fig. 5.9, 5.10; Model performance page |
| **SO2 / RQ1**: effect of class imbalance and limits of current approaches | Accuracy paradox (flag-nobody = 91.5% accuracy, 0 recall; untreated XGBoost at τ = 0.5); ablation rows F (no scale_pos_weight) and G (default hyperparameters) | Fig. 5.13; `artifacts/ablation.json`, Fig. 5.6 |
| **SO3 / RQ3**: design the hybrid SMOTE+ENN + optimised XGBoost; effect of hyperparameter optimisation | `src/experiment.py` (the five pipelines), Optuna history, learning curves with early stopping, tuned vs default XGBoost | `artifacts/tuning.json`, `learning_curves.json`; Fig. 5.14, 5.15 |
| **SO4 / RQ4**: compare with standard XGBoost, RF + SMOTE, LR + SMOTE on recall, precision, F1, ROC-AUC, PR-AUC, G-Mean, MCC and computational efficiency | Test-set comparison of all pipelines with training time, inference time per customer and model size; 10-fold paired t-tests (α = 0.05, Holm-adjusted) and Wilcoxon tests | `models/baselines/comparison_results.json`, `artifacts/significance.json`; Fig. 5.1, 5.2, 5.11, 5.12, 5.16 |
| §3.6 / Appendix A.1: data quality checklist | Every table of Appendix A.1 filled from the data | `docs/data-quality.md`, `artifacts/data_quality.json` |
| §3.7: preprocessing | `src/preprocessing.py` (gaps ≤ 3 days interpolated, longer gaps filled, cap at 10,000 kWh, 3σ → median, zero runs > 30 days flagged) with a log of every action | `artifacts/preprocessing_log.json` |
| §3.8 / Appendix A.2: features in four groups; trend features | Every feature tagged statistical / temporal / trend / anomaly; 25-feature core set; overall trend, 180-day trend, trend reversals in 30-day windows | `src/feature_catalog.py`, `src/features.py`; ablation row A |
| §3.9: SMOTE (0.5, k = 5) + ENN (k = 3), training split only | `src/resampling.py`; redone inside every CV fold, never on validation or test rows (tested) | `tests/test_preprocessing.py` |
| §3.10: tuning with stratified 5-fold CV, scale_pos_weight initialised at the imbalance ratio, early stopping, overfitting check | `src/modeling.py`; first trial fixes scale_pos_weight = 10.72; early stopping (50 rounds) on validation; learning curves | Fig. 5.14 |
| §3.11: significance over 10 CV folds, α = 0.05; computational efficiency | `scripts/significance.py`; timings and sizes in the comparison | Fig. 5.12, 5.16 |
| §3.12: deployment; ranked list; probability; SHAP local and global; LIME as second layer, disagreement flagged for human review | FastAPI + React console; SHAP waterfall per case; SHAP summary; LIME check on every case page and in the case PDF | Fig. 4.x, 5.5, 5.7, 5.8 |
| §3.13: responsible use (flag, not punish; appeal) | Statement on every case page and PDF; "Cleared" case status | Case file page, reports |

## 3. What changed in the system

- **Pipeline** (`src/`): `preprocessing.py`, `resampling.py`, `pipeline.py`,
  `experiment.py` are new; `train.py` follows the proposal protocol end to end;
  serving applies the saved `artifacts/pipeline.json`, so uploads are cleaned and
  imputed exactly as the served model was trained.
- **Evidence scripts** (`scripts/`): `data_quality.py`, `significance.py` are new;
  `ablation.py` now varies the proposal's variables one at a time;
  `make_thesis_figures.py` adds figures 5.8 – 5.16.
- **Console**: Model performance compares all five pipelines with cost and
  significance and shows the SMOTE+ENN effect; case files carry the LIME second
  opinion and the responsible-use statement; PDFs include both.
- **Training on a GPU**: `python -m src.train --device cuda` (XGBoost only; the CPU
  run remains the reference, as the proposal targets CPU-only utilities).

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
| §1.6; §3.10 | randomised hyperparameter search | Optuna (TPE), 40 trials per pipeline, 5-fold stratified CV on PR-AUC (the abstract already says Optuna) |
| §3.10 | "since the training set is already balanced 1-1" | With sampling_strategy = 0.5 the training set after SMOTE is 1 theft : 2 honest (before ENN); scale_pos_weight is searched in 1–3 after SMOTE+ENN and 1–10.72 without resampling |
| §3.10 (p. 51–52) | AUCC "hits 0"; paragraphs on adverse drug reactions and the "BooC" model | Remove: not related to this study. Replace with the overfitting check actually run: early stopping on validation PR-AUC and the learning curves (Fig. 5.14) |
| §3.11 | t-tests over "ten CV folds"; elsewhere 5-fold | 5-fold CV for tuning; **10-fold** CV for the significance tests (paired t-test, α = 0.05, Holm-adjusted, with Wilcoxon as a check) |
| §3.11 (p. 54) | "Flat Classifier model", "customer churn" | Remove; the Wide & Deep CNN of Zheng et al. (2018) is cited as a published reference result only, not re-implemented (it needs a GPU; that is the proposal's point) |
| §3.3; §3.11; §3.12; Appendix B phase 8 | Streamlit application | A FastAPI service and React console (ranked queue, probability, SHAP and LIME per case, PDF reports, threshold control, case workflow). Everything runs on a CPU |
| §3.12 (p. 56) | SHAP summary of 20 features | Done: Fig. 5.8 (top 20, 2,000 test customers) |
| §2.3.3 (p. 24) | "LSTM (Legacy Statistical Models)" | LSTM is Long Short-Term Memory; it is not a tree-based method |
| §3.13 (p. 57) | "Compliance with the Kenyan law on clinical science" | Kenya Data Protection Act, 2019 |

### Writing to fix

- Table of contents: every "Error! Bookmark not defined." (pp. 4–6): update the Word fields.
- Figure 2.7 is referenced but not shown (p. 43).
- p. 15: "CGCC benchmark" → SGCC. p. 18: "Electrocution continues to be…" (garbled). p. 49: "pandasvale", "PLN study", "a dip coma" (garbled).
- p. 58: "bacterial discrimination", "The First Law", "Treasonous courts", "But do I care? The Columbia Encyclopedia…": rewrite the paragraph.
- p. 19–20: §1.6 last paragraph and §1.7 first paragraph are hard to follow; restate the limitations plainly (single dataset, noisy labels, no demographics, China vs Kenya).
- Keywords list "SMOTE" twice.

## 5. Limitations to state in the thesis

- One dataset (SGCC, 2014–2016, China); no validation on a Kenyan utility.
- Labels come from field inspections and are noisy.
- SMOTE interpolates in feature space: synthetic "customers" are not real consumption series.
- The console serves a 3,000-customer sample of the test set.
- Timings are for a 4-core CPU; they scale with hardware.
- SHAP and LIME agree (≥ 3 of the top 5 signals shared and the same direction on the
  strongest one) on 14 of the 30 highest-risk served customers. The console shows the
  disagreement instead of hiding it; that is the human-review rule of section 3.12.
