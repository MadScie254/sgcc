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

XGBoost on 85 features per customer, tuned with Optuna on cross-validated
PR-AUC. Hold-out results (20% of customers, never seen in tuning or fitting):

<!-- metrics:start -->
| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| XGBoost (threshold 0.228) | 0.847 | 0.506 | 0.432 | 0.553 | 0.485 |
| Random forest (baseline, 0.5) | 0.816 | 0.417 | 0.115 | 0.783 | 0.200 |
| Logistic regression (baseline, 0.5) | 0.755 | 0.272 | 0.651 | 0.175 | 0.276 |

Test set: 8,475 customers (723 theft). Cross-validated PR-AUC on the training split: 0.523.
<!-- metrics:end -->

Full numbers: `artifacts/metrics.json` and `models/baselines/comparison_results.json`.

What drives it (5-fold CV on all customers, `python scripts/ablation.py`, results in `artifacts/ablation.json`):

| Variant | ROC-AUC | PR-AUC |
|---|---|---|
| 17 original features, dates in file order | 0.794 | 0.364 |
| 17 original features, dates sorted | 0.796 | 0.365 |
| 85 features, dates in file order | 0.857 | 0.523 |
| **85 features, dates sorted (this model)** | **0.862** | **0.532** |
| 85 features + SMOTE-ENN | 0.838 | 0.424 |

- **Richer features** are the main gain: missing and zero patterns kept as signal
  (XGBoost handles NaN natively), day-over-day drops, trends, year-over-year ratios,
  change points and a 34-month relative consumption profile.
- **No resampling.** SMOTE-ENN costs 0.11 PR-AUC. The decision threshold is chosen
  from out-of-fold predictions (max F1).
- **Chronological order** is a correctness fix (the raw header stores dates as
  `2014/1/1, 2014/1/10, ...`) with a small effect on scores (+0.006 ROC-AUC).

## Run the app

The trained model (`models/xgb_best.ubj`) and a demo population of 3,000 held-out
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
