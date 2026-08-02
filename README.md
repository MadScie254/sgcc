# SGCC Theft Detector

<div align="center">

**AI-Powered Electricity Theft Detection System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-ready ML pipeline for detecting electricity theft using consumption patterns.  
Built with **XGBoost**, **SHAP**, **Optuna**, **FastAPI**, and **React**.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [Deployment](#deployment)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Performance](#performance)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **SGCC Theft Detector** is an end-to-end machine learning system designed to identify electricity theft by analyzing consumption patterns. The system implements:

- **20+ engineered features** from time-series consumption data
- **SMOTE+ENN preprocessing** for handling class imbalance
- **Optuna-optimized XGBoost** classifier with recall-focused composite scoring
- **SHAP explanations** for model interpretability
- **Production-ready FastAPI backend** with a React/Vite frontend

### Key Capabilities

**High Recall Detection** — Optimized to catch theft (≥85% recall target)  
**Explainable Predictions** — SHAP-powered feature attribution  
**Interactive Dashboard** — Multi-route React app with EDA, training, prediction, explanation, upload, compare, and monitoring
**Docker Deployment** — Containerized for easy deployment  
**CI/CD Ready** — GitHub Actions workflow included  

---

## Features

### Advanced ML Pipeline

- **Feature Engineering**: Statistical, trend, temporal, and anomaly features
- **Preprocessing**: SMOTE+ENN for balanced training, MinMaxScaler normalization
- **Hyperparameter Tuning**: Optuna with 60-trial optimization
- **Composite Scoring**: `0.6 × Recall + 0.25 × Precision + 0.15 × F1`
- **Model Persistence**: Joblib serialization with versioning

### React Frontend

**4 Main Pages**:

1. **EDA** — Exploratory data analysis
   - Class distribution & imbalance analysis
   - Feature distributions (violin plots)
   - Correlation heatmaps
   - Time-series viewer with anomaly highlights
   - Customer clustering (planned)

2. **Train** — Model training interface
   - Quick train mode (5% data, 10 trials, ~5 min)
   - Full train mode (100% data, 60 trials, ~2-4 hours)
   - Custom configuration options
   - Real-time progress tracking
   - Training history & metrics

3. **Predict** — Theft detection
   - Single customer prediction (from test set)
   - Batch CSV upload
   - Manual feature input
   - Adjustable threshold slider
   - Risk gauge visualization
   - Top-3 SHAP reasons per prediction

4. **Explain** — Model explainability
   - Global SHAP summary (bar + beeswarm plots)
   - Feature importance rankings
   - Per-customer SHAP force plots
   - Interactive waterfall explanations

### Professional UI/UX

- **Dark theme** with teal (#00C2A8) and amber (#FFB020) accents
- **Metric cards** with deltas and color-coded risk levels
- **Interactive Plotly charts** with hover tooltips and drag-to-zoom
- **Responsive layout** with columns and expanders
- **Customer portrait cards** with Picsum placeholder images
- **Clean typography** and consistent spacing

### Public API Integrations

- **Open-Meteo**: Weather context for consumption patterns
- **Nominatim (OSM)**: Geocoding for customer location mapping
- **Lorem Picsum**: Placeholder images for customer portraits
- **REST Countries**: Regional metadata
- All cached with appropriate TTLs

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAW DATA (CSV)                          │
│              Kaggle: bensalem14/sgcc-dataset                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│               DATA LOADER (src/data_loader.py)              │
│  • Parse wide-format → long-format                          │
│  • Extract customer_id, day_index, consumption_kwh          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│         FEATURE ENGINEERING (src/features.py)               │
│  • Statistical: mean, median, std, coef_var, skewness      │
│  • Trend: slope_full, slope_30d, slope_90d                 │
│  • Temporal: weekday_ratio, peak_day_ratio                 │
│  • Anomaly: zero_days, sudden_drops, volatility            │
│  • Other: autocorr_lag1, missing_sequences                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│          PREPROCESSING (src/preprocessing.py)               │
│  • Train/test split (80/20, stratified)                    │
│  • SMOTE+ENN resampling (training only)                    │
│  • MinMaxScaler normalization [0,1]                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            MODELING (src/modeling.py)                       │
│  • XGBoost Classifier                                       │
│  • Optuna hyperparameter optimization                      │
│  • 5-fold stratified CV                                     │
│  • Composite score: 0.6R + 0.25P + 0.15F1                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│             EVALUATION (src/eval.py)                        │
│  • Metrics: Recall, Precision, F1, AUC, G-Mean, MCC        │
│  • Confusion matrix                                         │
│  • Error analysis (FP/FN IDs)                              │
│  • SHAP explanations (TreeExplainer)                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│        DEPLOYMENT (backend + frontend build)                │
│  • FastAPI serves the API and the built React SPA           │
│  • Docker containerization                                  │
│  • CI/CD with GitHub Actions                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- **pip** or **conda**
- **Kaggle API credentials** (for data download)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/sgcc-theft-detector.git
cd sgcc-theft-detector
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n sgcc python=3.10
conda activate sgcc
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Kaggle API

**Option A: kaggle.json file**

```bash
# Download your kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Option B: Environment variables**

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 5. Download Dataset

```bash
# Linux/Mac
bash scripts/download_data.sh

# Windows
scripts\download_data.bat
```

---

## Quick Start

### Train Model

**Quick Mode** (5% data, ~5 minutes):

```bash
python -m src.train --quick
```

**Full Mode** (100% data, ~2-4 hours):

```bash
python -m src.train
```

### Launch the Web App

```bash
# Terminal 1: API
py -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: React frontend
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open the frontend at `http://localhost:5173`.

### Make Predictions

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("models/xgb_best.joblib")
scaler = joblib.load("artifacts/scaler.joblib")

# Prepare features (example)
features = pd.DataFrame({
    'mean': [25.3],
    'median': [24.1],
    # ... other features
})

# Scale and predict
features_scaled = scaler.transform(features)
probability = model.predict_proba(features_scaled)[0, 1]

print(f"Theft probability: {probability:.1%}")
```

---

## Usage

### Training Pipeline

The `src/train.py` module orchestrates the full training pipeline:

```bash
# With custom config
python -m src.train --config custom_config.yaml

# Quick mode
python -m src.train --quick
```

**Pipeline Steps**:

1. Load raw data from CSV
2. Engineer 20+ features per customer
3. Split train/test (80/20, stratified)
4. Apply SMOTE+ENN preprocessing
5. Normalize features with MinMaxScaler
6. Optimize XGBoost with Optuna (60 trials)
7. Save model, scaler, and artifacts

**Outputs**:

- `models/xgb_best.joblib` — Trained model
- `artifacts/scaler.joblib` — Feature scaler
- `artifacts/metrics.json` — Evaluation metrics
- `artifacts/feature_importance.csv` — Feature rankings
- `artifacts/shap_values.npy` — SHAP explanations
- `artifacts/preprocess_report.json` — SMOTE+ENN stats

### Evaluation

```bash
# Run evaluation on saved model
python -m src.eval
```

Generates:

- Metrics (recall, precision, F1, AUC, G-Mean, MCC)
- Confusion matrix plot
- ROC and PR curves
- Feature importance ranking
- SHAP summary plots
- Error analysis (FP/FN customer IDs)

### Frontend Usage

**Page 1: EDA**

- View class distribution and dataset statistics
- Explore feature distributions by class (violin plots)
- Analyze feature correlations (heatmap)
- Inspect individual customer consumption time series
- Identify anomalies (zero days, sudden drops)

**Page 2: Train**

- Select training mode (Quick/Full/Custom)
- Configure hyperparameters (trials, CV folds, sample fraction)
- Monitor real-time training progress
- View training results (best score, best params, preprocessing stats)

**Page 3: Predict**

- **Single Customer**: Select from test set, predict, view SHAP reasons
- **Batch Upload**: Upload CSV with features, predict all, download results
- **Manual Input**: Enter feature values manually
- Adjust classification threshold with slider
- View risk gauge and customer portrait card

**Page 4: Explain**

- **Feature Importance**: Interactive bar chart, top-10 list, download CSV
- **Global SHAP**: Summary bar plot and beeswarm plot
- **Local Explanation**: Per-customer SHAP force plot and waterfall
- **Model Info**: Architecture, targets, feature engineering details

---

## Performance

### Target Metrics (from Concept Note)

| Metric    | Target | Achieved* |
|-----------|--------|-----------|
| Recall    | ≥85%   | TBD       |
| Precision | ≥75%   | TBD       |
| F1 Score  | ≥75%   | TBD       |
| AUC       | —      | TBD       |

*Run training to populate actual metrics.*

### Optimization Focus

The model is **recall-optimized** to minimize false negatives (missed theft cases), accepting higher false positives as a trade-off. This is achieved through:

- **Composite scoring**: `0.6 × Recall + 0.25 × Precision + 0.15 × F1`
- **SMOTE+ENN preprocessing**: Balances training data to improve minority class detection
- **Optuna hyperparameter tuning**: Searches for optimal model configuration

---

## Deployment

### Docker

**Build Image**:

```bash
docker build -t sgcc-theft-detector:latest .
```

**Run Container**:

```bash
docker run -p 8501:8501 \
  -e KAGGLE_USERNAME=$KAGGLE_USERNAME \
  -e KAGGLE_KEY=$KAGGLE_KEY \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/artifacts:/app/artifacts \
  sgcc-theft-detector:latest
```

**Using Docker Compose**:

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Docker

The Docker image builds the React frontend and serves it from the FastAPI backend on port `8000`.

```bash
docker compose up --build
```

Then open `http://localhost:8000`.

### Production Considerations

- **Model Versioning**: Use MLflow or similar for experiment tracking
- **Monitoring**: Track prediction distribution drift
- **Retraining**: Schedule periodic retraining with new data
- **Scaling**: Use Kubernetes for high-traffic deployments
- **Security**: Rotate Kaggle API keys, use secrets management
- **Logging**: Centralized logging with ELK stack or similar

---

## Project Structure

```
sgcc-theft-detector/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD
├── artifacts/                     # Training artifacts (gitignored)
│   ├── metrics.json
│   ├── feature_importance.csv
│   ├── shap_values.npy
│   ├── shap_summary.png
│   ├── shap_beeswarm.png
│   ├── scaler.joblib
│   ├── preprocess_report.json
│   ├── fp_ids.json
│   ├── fn_ids.json
│   └── test_data.pkl
├── concept_note/                  # Project documentation
│   └── CONCEPT NOTE FINAL ACT.pdf
├── data/                          # Data files (gitignored)
│   └── data_raw.csv
├── models/                        # Trained models (gitignored)
│   └── xgb_best.joblib
├── notebooks/                     # Jupyter notebooks
│   └── quick_experiments.ipynb
├── scripts/                       # Utility scripts
│   ├── download_data.sh
│   └── download_data.bat
├── src/                           # Source code
│   ├── data_loader.py             # Data loading & parsing
│   ├── features.py                # Feature engineering
│   ├── preprocessing.py           # SMOTE+ENN preprocessing
│   ├── modeling.py                # XGBoost + Optuna
│   ├── train.py                   # Training pipeline
│   └── eval.py                    # Evaluation & SHAP
├── backend/                       # FastAPI backend
├── frontend/                      # React/Vite frontend
├── tests/                         # Unit tests
│   ├── test_data_loader.py
│   └── test_features.py
├── .gitignore                     # Git ignore rules
├── config.yaml                    # Configuration file
├── docker-compose.yml             # Docker Compose config
├── Dockerfile                     # Docker image definition
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install black flake8 pytest pytest-cov

# Run tests
pytest tests/ -v

# Format code
black src/ backend/ tests/

# Lint
flake8 src/ backend/ --max-line-length=127
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **SGCC Dataset**: [bensalem14/sgcc-dataset](https://www.kaggle.com/datasets/bensalem14/sgcc-dataset) on Kaggle
- **XGBoost**: Tianqi Chen and Carlos Guestrin
- **SHAP**: Scott Lundberg et al.
- **Optuna**: Takuya Akiba et al.
- **FastAPI**: Sebastián Ramírez and contributors
- **React**: Meta

---

## Contact

For questions, issues, or collaboration:

- **GitHub Issues**: [sgcc-theft-detector/issues](https://github.com/yourusername/sgcc-theft-detector/issues)
- **Email**: your.email@example.com

---

<div align="center">

**Built with care for reliable electricity infrastructure**

SGCC Theft Detector | Production Ready | Deploy Anywhere

</div>
