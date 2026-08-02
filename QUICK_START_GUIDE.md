# SGCC Quick Start

## Run locally

### Development

```bash
cd C:\Users\MadScie254\Documents\GitHub\sgcc
.\.venv\Scripts\python -m pip install -r requirements.txt
cd frontend
npm install
npm run dev
```

In a second terminal:

```bash
cd C:\Users\MadScie254\Documents\GitHub\sgcc
.\.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Production-style local run

```bash
cd C:\Users\MadScie254\Documents\GitHub\sgcc\frontend
npm run build
cd ..
.\.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`.

## Trained assets

- Model: `models/xgb_best.joblib`
- Metrics and SHAP artifacts: `artifacts/`

You do not need to retrain to run the app.

## Useful endpoints

- API health: `GET /api/health`
- Customer table: `GET /api/customers`
- Explainability: `GET /api/model/feature-importance`
