# SGCC Theft Detector

Electricity-theft detection on the SGCC smart-meter dataset (42,372 customers,
daily kWh from 2014-01-01 to 2016-10-31, ~8.5% labelled theft), with a FastAPI
backend and a React dashboard.

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

What drives it:

- **Chronological order.** The raw SGCC header stores dates lexicographically
  (`2014/1/1, 2014/1/10, ...`); the loader sorts them before computing trends,
  drops, and seasonality.
- **Missing and zero patterns** are kept as signal (XGBoost handles NaN natively),
  alongside statistics, day-over-day drops, trends, year-over-year ratios,
  change points, weekday/weekend ratio, and a 34-month relative consumption profile.
- **No resampling.** SMOTE+ENN lowered cross-validated PR-AUC from ~0.53 to ~0.42.
  The decision threshold is chosen from out-of-fold predictions (max F1).

## Run the app

The trained model (`models/xgb_best.ubj`) and a demo population of 3,000 held-out
customers (`data/sgcc_demo.csv.gz`) are committed, so no training or downloads are needed.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.lock
uvicorn backend.main:app --reload  # API on http://127.0.0.1:8000
```

Frontend (second terminal):

```bash
cd frontend
npm ci
npm run dev                        # http://localhost:5173, proxies /api to :8000
```

For a single server, run `npm run build` in `frontend/` and open
`http://127.0.0.1:8000/`; the backend serves `frontend/dist`.

Docker: `docker compose up --build` (set `API_KEY` in `.env` first; see `.env.example`).

## Train

```bash
python scripts/download_data.py    # full SGCC dataset -> data/sgcc_full.csv (175 MB, no credentials)
python -m src.train --quick        # smoke run: 25% of customers, 5 trials
python -m src.train                # full run: 40 Optuna trials, 5-fold CV
```

Training writes the model, `artifacts/metrics.json`, `artifacts/feature_importance.csv`,
`artifacts/best_params.json`, the baseline comparison, and a fresh demo population.
Settings live in `config.yaml`.

## Security

- Outside `ENV=development` the API requires `API_KEY`; every `/api/*` route except
  `/api/health` checks the `X-API-Key` header. Enter the key on the dashboard's
  **Settings** page; it is stored in that browser only and never built into the JavaScript.
- `POST /api/train/jobs` overwrites the served model, so it is disabled outside
  development unless `ENABLE_TRAINING_API=1`. Clients can only change trial count,
  CV folds, timeout, and test size.
- Uploads are capped at `MAX_UPLOAD_MB` (default 25).
- API docs (`/api/docs`) are only served in development.

## Deploy (Render)

`render.yaml` installs `requirements.lock`, builds the frontend, and starts uvicorn.
Connect the repository once in Render; each push to `main` redeploys. Render
generates `API_KEY`; copy it from the service's environment into the dashboard's
Settings page.

## Layout

- `src/` data loading, features, tuning, evaluation, training pipeline
- `backend/` FastAPI app (`routers/`, `services/`, `schemas/`)
- `frontend/` React + Vite dashboard
- `models/`, `artifacts/` trained model and metrics the API reads
- `data/sgcc_demo.csv.gz` held-out customers the API serves
- `scripts/download_data.py` dataset download
- `concept_note/` project concept note
