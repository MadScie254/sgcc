# GridSentinel — SGCC Theft Detector

Electricity-theft detection on the SGCC smart-meter dataset (42,372 customers,
daily kWh from 2014-01-01 to 2016-10-31, ~8.5% labelled theft): the SMOTE+ENN +
XGBoost framework of the research proposal evaluated against its baselines, the
winning model served by a FastAPI backend, and a React operations console.

## The console

| Page | What it does |
|---|---|
| Command center | Headline figures, the scoring pipeline's last run, the investigation queue, risk distribution and outcome at the threshold in service |
| Case files | Every flagged customer as a case (new → reviewing → dispatched → confirmed / cleared) with notes and history |
| Case file | Consumption history (monthly/daily, gaps shaded), SHAP waterfall of why it was flagged, a LIME second opinion that flags disagreement for manual review, case actions, case-file PDF |
| Pipeline | The automated scoring workflow (ingest → features → score → explain → route) with timings and run history, and the stages of the training run behind the served model |
| Threshold studio | Trade thefts caught against wasted visits on held-out customers, then publish the threshold to scoring |
| Model performance | Test metrics; all five pipelines compared (effectiveness, training and inference time, model size, significance); what SMOTE+ENN does to the training data; what drives the score (mean absolute SHAP); tuned hyperparameters |
| Reports & scoring | Portfolio, dataset and case-file PDF reports; upload SGCC meter data (or feature rows) and see how the model scores it, including thefts caught when the file has labels; batch-score a CSV; score one customer |

## Model

The research proposal's framework, **SMOTE+ENN + tuned XGBoost**, is compared with
standard XGBoost, random forest + SMOTE, logistic regression + SMOTE and an untreated
default XGBoost. All five use the same customers and the same 70/15/15 split, and
each is scored at the threshold that maximised F1 on the validation customers.

Results on the 6,356 test customers, never used for tuning, early stopping or
thresholds:

<!-- metrics:start -->
| Pipeline | ROC-AUC | PR-AUC | Recall | Precision | F1 | MCC | G-Mean | ms / customer |
|---|---|---|---|---|---|---|---|---|
| SMOTE+ENN + XGBoost (proposed) | 0.828 | 0.462 | **0.494** | 0.389 | 0.435 | 0.379 | **0.677** | 0.002 |
| **XGBoost, no resampling (served)** | **0.851** | **0.506** | 0.426 | **0.532** | **0.473** | **0.433** | 0.641 | 0.002 |
| XGBoost, default settings | 0.839 | 0.488 | 0.483 | 0.439 | 0.460 | 0.408 | 0.675 | 0.001 |
| Random forest + SMOTE | 0.812 | 0.411 | 0.432 | 0.366 | 0.396 | 0.336 | 0.634 | 0.029 |
| Logistic regression + SMOTE | 0.754 | 0.314 | 0.371 | 0.318 | 0.342 | 0.277 | 0.586 | 0.001 |

542 of the 6,356 test customers are thefts. Served threshold: 0.499.
<!-- metrics:end -->

The 10-fold paired t-tests (Holm-adjusted, α = 0.05, `scripts/significance.py`) back
this up:

- **Recall and G-Mean:** SMOTE+ENN catches significantly more thefts than standard
  XGBoost (+0.075 recall, +0.038 G-Mean).
- **Ranking and precision:** it is significantly worse on PR-AUC (−0.054), ROC-AUC,
  precision, F1 and MCC.
- **SMOTE baselines:** it beats both on PR-AUC, recall and G-Mean.

The console serves the pipeline with the higher validation PR-AUC, standard XGBoost.
`docs/proposal-alignment.md` discusses the outcome against the proposal's objectives.

What each part is worth (5-fold CV on all 42,372 customers, one change at a time,
`python scripts/ablation.py`, `artifacts/ablation.json`):

| Variant | ROC-AUC | PR-AUC |
|---|---|---|
| A. Raw readings, the proposal's 25 core features | 0.831 | 0.447 |
| **B. Raw readings, all 87 features (standard XGBoost)** | **0.854** | **0.509** |
| C. Cleaned readings (proposal section 3.7), all features | 0.847 | 0.490 |
| D. Cleaned + SMOTE | 0.831 | 0.453 |
| E. Cleaned + SMOTE+ENN (proposed) | 0.827 | 0.414 |
| F. B with scale_pos_weight = 1 | 0.856 | 0.511 |
| G. B with XGBoost's default hyperparameters | 0.839 | 0.475 |

- **Features are the main gain:** +0.061 PR-AUC from the 25 core features to all 87.
  The extra features are gaps and zero runs, drops, trends, reversals, year-over-year
  ratios, change points, and a 34-month usage profile.
- **Tuning helps:** +0.034 PR-AUC over default hyperparameters.
- **Imputation and resampling cost ranking quality.** Filling gaps smooths away theft
  signal, and SMOTE+ENN shifts the model towards recall rather than a better ranking.
- **SMOTE+ENN does clean the training data** (Objective 1, `artifacts/resampling.json`).
  ENN removes 4,539 honest boundary rows, the Fisher ratio rises from 0.014 to 0.022,
  and the share of theft rows with honest-majority neighbours falls from 90% to 23%.
  That cleaner training data does not carry over to better test-set ranking.
- The ablation pools out-of-fold predictions before scoring. That penalises
  resampled models, whose calibration varies more between folds. This is why E
  (0.414) sits below the per-fold mean of the significance run (0.452).

## Run the app

The trained model (`models/xgb_best.ubj`) and a demo population of 3,000 test
customers (`data/sgcc_demo.csv.gz`) are committed, so no training or downloads are needed.

Requirements: Python 3.11–3.13, Node.js 20+, Git. Works on Windows, Linux and
Apple Silicon Macs (Intel Macs lack a `llvmlite` wheel for SHAP).

**1. Get the code**

```bash
git clone https://github.com/MadScie254/sgcc.git
cd sgcc
```

**2. Backend** (first terminal)

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.lock
uvicorn backend.main:app --reload
```

Using an existing Anaconda environment instead (Command Prompt):

```bat
conda activate ml_env
pip install -r requirements.lock
uvicorn backend.main:app --reload
```

Install the lock file into whichever environment you use: it pins the versions the
model and the PDF reports need (for example `xgboost==3.2.0` and `fpdf2`). If
`/api/health` reports `"reports": {"available": false}`, the environment has an old
or missing fpdf2 and the message says how to fix it.

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
uvicorn backend.main:app --reload
```

Wait for `Application startup complete` (the first scoring run takes a few seconds).
`http://127.0.0.1:8000/api/health` should show `"model_loaded": true`.

**3. Frontend** (second terminal, from the repo folder)

```bash
cd frontend
npm ci
npm run dev
```

Open **http://localhost:5173**. It proxies `/api` to the backend on port 8000.
Without `ENV`/`API_KEY` set the API runs in development mode and needs no key.

**Single server instead:** run `npm run build` in `frontend/`, restart uvicorn, and open
`http://127.0.0.1:8000/`; the backend serves `frontend/dist`.

**Production-like run with a key:** set `ENV=production` and `API_KEY=<something>` before
starting uvicorn (PowerShell: `$env:ENV="production"; $env:API_KEY="..."`), then enter the
same key on the console's Settings page.

Docker: `docker compose up --build` (set `API_KEY` in `.env` first; see `.env.example`).

### Automation

The scoring pipeline runs once when the server starts and on demand from the
Pipeline page. Set `SCORING_INTERVAL_MINUTES` to also run it on a schedule.
Everything the API writes is kept in `artifacts/state/` (override with
`SGCC_STATE_DIR`): case statuses and notes, run history, a published threshold,
uploaded datasets (latest 100) and generated reports (latest 500).

### Reports

`POST /api/reports` with `{"kind": "portfolio"}`, `{"kind": "dataset", "dataset_id": ...}`
or `{"kind": "case", "customer_id": ...}` renders a PDF; `GET /api/reports/{id}/pdf`
downloads it. The console does both in one click. Uploads (`POST /api/datasets`) and
batch scoring (`POST /api/predict/batch`) accept either the SGCC layout (`CONS_NO`,
optional `FLAG`, one column per day; features are built exactly as in training) or
rows of the 87 model features. Anything else is rejected with a reason.

### Test the integrated model

With the API running:

```bash
python scripts/evaluate_api.py --url http://127.0.0.1:8000 --key "$API_KEY"
```

It checks hold-out ROC-AUC and precision through the API, that every endpoint scores
a customer identically, that SHAP values add up to each prediction, that LIME gives a
second opinion, that publishing a threshold re-flags customers and opens cases, and that
a PDF report downloads.

## Train

```bash
python scripts/download_data.py        # full SGCC dataset -> data/sgcc_full.csv (175 MB, no credentials)
python -m src.train --quick            # smoke run: 25% of customers, 5 trials
python -m src.train                    # full run: 2 × 40 Optuna trials, 5-fold CV (~20 min, 4 cores)
python -m src.train --device cuda      # the same with XGBoost on an NVIDIA GPU
python scripts/significance.py         # 10-fold paired tests between the pipelines (~20 min)
python scripts/ablation.py             # one-change-at-a-time ablation (~25 min)
python scripts/data_quality.py         # proposal Appendix A tables -> docs/data-quality.md
python scripts/make_thesis_figures.py  # docs/thesis-figures
```

The protocol follows the research proposal: series cleaning (section 3.7), features
grouped as statistical, temporal, trend and anomaly (3.8), a stratified 70/15/15
customer split, SMOTE+ENN on training rows only and redone inside every CV fold (3.9),
Optuna tuning of both XGBoost pipelines with early stopping on validation (3.10), and
one scoring of the untouched test customers for every pipeline (3.11). The XGBoost
pipeline with the higher validation PR-AUC is served. `docs/proposal-alignment.md`
maps every objective and research question to its evidence.

Training writes the model, its pipeline spec (`artifacts/pipeline.json`: how serving
must clean and impute), `artifacts/metrics.json`, `tuning.json`, `learning_curves.json`,
`resampling.json`, `preprocessing_log.json`, `feature_importance.csv`, `best_params.json`,
the pipeline comparison, and a fresh demo population. Settings live in `config.yaml`
(`model.device` sets CPU or GPU; the CPU run is the reference, as the proposal targets
CPU-only utilities). Training replaces the served model, so it runs from the
command line only; afterwards press **Run scoring now** on the Pipeline page (or restart
the API) to load the new model.

## Security

- Outside `ENV=development` the API requires `API_KEY`; every `/api/*` route except
  `/api/health` checks the `X-API-Key` header. Enter the key on the dashboard's
  **Settings** page; it is stored in that browser only and never built into the JavaScript.
- The API cannot retrain or replace the model.
- Uploads are capped at `MAX_UPLOAD_MB` (default 25), 200,000 rows and 2,000 columns.
  Batch-score CSVs neutralise cells that a spreadsheet would run as formulas.
- Report and dataset ids are validated before any file is touched.
- API docs (`/api/docs`) are only served in development.

## Deploy (Render)

`render.yaml` installs `requirements.lock`, builds the frontend, and starts uvicorn.
Connect the repository once in Render; each push to `main` redeploys. Render
generates `API_KEY`; copy it from the service's environment into the dashboard's
Settings page.

## Layout

- `src/` data loading, features, tuning, evaluation, training pipeline
- `backend/` FastAPI app (`routers/`, `services/`, `schemas.py`)
- `frontend/` React + Vite dashboard
- `models/`, `artifacts/` trained model and metrics the API reads
- `data/sgcc_demo.csv.gz` held-out customers the API serves
- `scripts/` dataset download, API evaluation, ablation study, thesis figures
- `concept_note/` project concept note
