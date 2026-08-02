# SGCC Theft Detector

Electricity theft detection with a FastAPI backend and React dashboard.

## Prerequisites

```bash
git lfs install
git lfs pull
```

Then install dependencies.

### Development

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
cd frontend
npm install
npm run dev
```

In a second terminal:

```bash
cd ..
.\.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Production-style local run

```bash
cd frontend
npm run build
cd ..
.\.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. The backend serves the built React app from `frontend/dist`.

## Already trained

- Model: `models/xgb_best.joblib`
- Metrics and SHAP artifacts: `artifacts/`

You do not need to retrain to run the app. Retrain only if you want a new model or updated metrics.

The current training flow is split across `src/train.py` for the model and test split, then `src/eval.py` for metrics and feature-importance artifacts.

## Project layout

- `backend/` FastAPI app and API routes
- `frontend/` React + Vite dashboard
- `src/` training and feature engineering code
- `models/` saved model artifacts
- `artifacts/` metrics, SHAP, scaler, and evaluation outputs

## Notes

- The old Streamlit app has been removed.
- The backend health check is `GET /api/health`.
- The frontend runs against the local API on port `8000`.
