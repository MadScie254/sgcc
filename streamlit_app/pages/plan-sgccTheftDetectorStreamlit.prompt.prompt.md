# Plan: SGCC End-to-End Theft Detection System

Build a production-ready, modular ML repository implementing SMOTE+ENN preprocessing, temporal/statistical feature engineering, Optuna-tuned XGBoost, SHAP explainability, and a polished multi-page Streamlit app with React-level UX, public API integrations, Docker deployment, and comprehensive EDA notebooks.

## Steps

### 1. Foundation setup
Create repo structure: `src/` modules (data_loader, features, preprocessing, modeling, train, eval, deploy_utils), `streamlit_app/` with pages (app.py, 1_EDA, 2_Train, 3_Predict, 4_Explain), `tests/`, `scripts/`, `models/`, `artifacts/`, `notebooks/`; add `requirements.txt`, `.gitignore`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `config.yaml`, `README.md`

### 2. Data pipeline
Implement `src/data_loader.py` `load_raw()` to parse wide-format CSV (detect encoding, extract customer_id from second-last column, label from last column, pivot 1000+ consumption columns to long format df_long with columns: customer_id, day_index, consumption_kwh); add `scripts/download_data.sh` with Kaggle CLI command; handle encoding/parsing edge cases with try-except and logging

### 3. Feature engineering & preprocessing
Build `src/features.py` `build_features()` computing 20+ temporal/statistical features (mean, median, std, coef_var, skewness, linear trend slopes, zero_day_count, sudden_drop_count >50%, volatility, autocorr lag-1, weekday_weekend_ratio, peak_day_ratio, missing_sequences); normalize with MinMaxScaler saved to `artifacts/scaler.joblib`; implement `src/preprocessing.py` `apply_smote_enn()` with SMOTEENN, save class-balance report to `artifacts/preprocess_report.json`

### 4. Model training & optimization
Create `src/modeling.py` with `train_xgb_with_optuna()` tuning max_depth, learning_rate, n_estimators, subsample, colsample_bytree, reg_alpha/lambda, scale_pos_weight over 60 trials optimizing composite score (0.6×recall + 0.25×precision + 0.15×f1) via stratified 5-fold CV; save best model to `models/xgb_best.joblib`, study to `artifacts/optuna_study.pkl`; implement `src/eval.py` computing recall, precision, F1, AUC, G-mean, MCC, confusion matrix, saving `artifacts/metrics.json`, `artifacts/feature_importance.csv`, `artifacts/fp_ids.json`, `artifacts/fn_ids.json`; integrate SHAP TreeExplainer generating `artifacts/shap_summary.png` and `artifacts/shap_values.npy`

### 5. Streamlit app with React-level UX
Build `streamlit_app/app.py` with sticky header (logo, model-version dropdown, customer search), cached model/scaler loading via `st.cache_resource`; implement `pages/1_EDA.py` showing class distribution, top-feature histograms, consumption heatmap, interactive Plotly timeseries, UMAP cluster scatter, cohort comparison, calendar heatmap with interpretations; `pages/2_Train.py` with quick-train demo mode (5% sample) and progress bar; `pages/3_Predict.py` with CSV upload/customer_id input, probability slider, SHAP per-sample force plot, top-3 textual reasons; `pages/4_Explain.py` with global SHAP beeswarm, feature-importance bar, interactive per-sample force plots; add customer portrait cards (Picsum placeholder images), metric chips with deltas, Plotly charts with drag-to-zoom, dark-theme CSS, teal/amber accent colors

### 6. Public API integrations & enhancements
Add cached `get_weather()` using Open-Meteo API (show weather context for flagged dates), `get_location()` via Nominatim for geocoding customer regions to Folium map on investigation panel, Picsum placeholder images for customer portraits/meter photos, REST Countries for regional metadata; implement investigative flyout panel (customer card, timeline, SHAP reasons, export button), confusion-matrix interactive threshold slider updating precision/recall live, bulk-triage checkboxes for CSV export

### 7. Notebooks, tests, CI/CD, deployment
Create `notebooks/quick_experiments.ipynb` with full EDA (10+ beautiful Plotly/seaborn graphs: distribution violin plots, correlation heatmap, time-series with anomaly highlights, cluster scatter, calendar heatmap, cohort comparison, SHAP summary, ROC/PR curves) plus interpretations and baseline training run; add `tests/test_data_loader.py` checking columns/non-empty; write `.github/workflows/ci.yml` running pytest and flake8; create `Dockerfile` installing requirements and running Streamlit on $PORT; write comprehensive `README.md` with Kaggle download instructions (KAGGLE_USERNAME/KEY setup), local run commands (venv, pip, train, streamlit), Streamlit Cloud deploy steps, performance targets (recall ≥0.85, precision ≥0.75), architecture diagram, artifact locations

## Further Considerations

### 1. Concept note extraction
PDF at `concept_note/CONCEPT NOTE FINAL ACT.pdf` is unreadable with current tools; proceed with user-specified requirements (SMOTE+ENN, XGBoost, recall-focused composite score, 20+ features) or manually extract key details for exact alignment with research design?

### 2. Data schema validation
CSV appears headerless with 1000+ consumption columns; confirm column-naming convention (e.g., day_0, day_1...) and exact customer_id/label column indices before implementing parser, or auto-detect with robust fallback logic?

### 3. Resource constraints
Training 60-trial Optuna on full dataset with 5-fold CV will take hours on laptop; provide lightweight quick-train mode (10 trials, 10% sample) for Streamlit demo vs separate offline training script for production model; document RAM requirements (16GB+ recommended)?

### 4. Deployment target
Streamlit Cloud has 1GB RAM limit; precompute all heavy artifacts (SHAP, heatmaps, trained model <100MB) during offline training and load cached; add fallback error messages if artifacts missing; provide docker-compose for local deployment alternative?

## UX & Visual Language — "React-level" in Streamlit

Think polished, component-driven, and immediately trustworthy. Use Streamlit primitives as *components*, plus HTML/JS embeds where you need micro-animations or custom controls.

### Colors & tone
- Dark, high-contrast dashboard (charcoal background, neon-accent CTA — teal / electric blue / lime). Use 2 accent colors: Action (teal #00C2A8) and Warning (amber #FFB020). Keep fonts large, roomy spacing, 12–16px base, and big metric chips.
- Use concise, direct microcopy: e.g. "Customer at risk: 87% — action: investigate", "Why this flagged: consumption dropped 72% vs rolling-30d".

### Global chrome
- App header (left): logo + "SGCC — Theft Detector"
- Header (right): quick controls: model version dropdown, date-range selector, quick-search (customer_id)
- Use a sticky top bar via `st.sidebar` or `components.v1.html` sticky CSS for global controls.

### Page scaffolding (multi-page)
- Home (overview + summary metrics)
- Customers (table + filter + map + timeline) — default landing for investigators
- Cases (ranked flagged customers, triage workflow)
- Model (train / metrics / CI) — only for admins
- Explain (global SHAP, per-customer SHAP force/why)
- Settings & Export (download artifacts, model versions)

### Layout primitives to implement
- Use `st.tabs` for subviews inside pages (e.g., EDA → [Distribution | Time-series | Cohorts])
- Use `st.columns` for responsive metric cards and small charts.
- Use `st.expander` for drill-in debug info (raw time series, raw CSV row).
- Use `st.session_state` to preserve filters & thresholds between pages.
- For micro-animations and "react-level" flair: embed Lottie animations or tiny JS (GSAP) via `st.components.v1.html` for when a model finishes training or a customer is flagged; but keep non-blocking.

### Interactivity patterns (react feel)
- Instant threshold slider (probability → label) on Predict page; update charts reactively.
- Hoverable tooltips on every chart (Plotly) with actionable text.
- Inline "investigate" button on each flagged customer row that opens a right-side panel showing timeline + SHAP.
- Drag-to-zoom on time series (Plotly), click-to-pin ranges as "evidence spans".

### Accessibility & performance
- Precompute heavy SHAP/heatmaps during training, save artifacts to `artifacts/`, load them.
- Cache everything with `st.cache_data` and `st.cache_resource`.
- Provide text alternatives and keyboard-friendly controls.

## Exact UI Components

- **Metric row (single-line)** — `st.metric` for: Total customers | Flagged today | Recall (test) | Model version.
- **Customer portrait card (left column)**: small photo (from Picsum/Unsplash), name/id, risk gauge (circular Plotly gauge), quick notes, `Investigate` button.
- **Time series viewer (center)**: Plotly line chart with consumption & rolling mean, shaded confidence bands, anomalies highlighted (vertical bars), interactive range selection.
- **Calendar heatmap (right)**: monthly/day-level view showing bursts (use seaborn / custom heatmap saved as PNG for performance).
- **Consumption distribution + violin plots by cohort** (Plotly); show mean, median markers.
- **Cohort explorer**: group customers by similar patterns using UMAP or hierarchical clustering; visualise clusters (2D scatter with cluster color) and allow selecting cluster to show sample customers.
- **SHAP section**: global summary bar chart, beeswarm, per-sample force plot embedded as HTML (shap.force_plot -> `components.v1.html`) and top-3 reasons text strip ("Top reasons this was flagged: low avg consumption, sudden 30d drop, high zero-day-count").
- **Confusion matrix + PR curve + ROC** (Plotly) with interactive threshold slider that updates precision/recall metrics live.

## The Exact List of Graphs & Text Interpretation

1. **Overview metrics row (big)**: total customers, flagged, precision@threshold, recall@threshold, model AUC.
   - Microcopy: "Recall-focused model — tuned to catch theft even at cost of false alarms."

2. **Top-of-page alert timeline**: time-series of flagged customers per day (bar).
   - Interpretation: "Spikes on dates X indicate bulk anomalies — likely meter tampering events."

3. **Customer timeline (line + rolling mean + anomalies)**: highlight drops >50% and days with zero consumption.
   - Interpretation: "Major 72% drop on 2024-03-21; last 14 days mean is 31% below historical baseline."

4. **Calendar heatmap (consumption intensity)**: shows seasonality, holidays.
   - Interpretation: "Weekend usage drops consistently — compare to peer cohort to rule-out pattern."

5. **Cluster scatter (UMAP/t-SNE)** of customer behavior with cluster labels.
   - Interpretation: "Cluster A (3,148 customers) shows frequent zero-day spikes — >60% of flagged accounts are here."

6. **SHAP global (beeswarm) + top 10 feature importances (bar)**.
   - Interpretation: "Top predictor: `sudden_drop_30d` — increases theft probability by +0.4 when true."

7. **Per-sample SHAP force plot + textual top-3 reasons** (automatic).
   - Microcopy: "Why flagged: 1) sudden_drop_30d (strong positive), 2) zero_day_count (positive), 3) avg_consumption_low (modest)."

8. **Model diagnostics**: PR curve with operating point, calibration plot, confusion matrix.
   - Interpretation: "At threshold 0.5 we get recall=0.86, precision=0.72. Moving threshold to 0.35 increases recall to 0.92 at cost of precision 0.63."

9. **Cohort comparison table** (top cohort statistics): median, skew, % flagged.

## Investigative Flows

- Click flagged customer → right-side flyout: customer card, timeline, meter photo (placeholder), location pin, last-contact notes, SHAP reasons, "export case" button to save to CSV.
- Bulk triage: checkboxes to bulk-assign customers to investigation groups, export to CSV/email (no-auth email? use Mailgun requires key — suggest download and offline).
- Audit log: list of model-retrained events (timestamp, params, metrics).

## Visual Polish & Micro-UX Details

- Use metric chips with deltas (compare current to 7-day average).
- Tiny micro-animations on training completion (Lottie success animation).
- Use subtle box-shadows for cards, rounded corners, consistent padding.
- Use strong, readable numeric formatting (e.g., 87.4% not 0.874).

## No-Auth Public APIs to Enrich the App

Below are reliable *no-auth* (or effectively no-auth) APIs you can call directly from Streamlit to add extra polish, context, or imagery.

### 1. Open-Meteo — free weather forecasts (no API key)
- **Use**: show weather at customer coordinates on the investigation panel (helps interpret low consumption during storms/outages).
- **Docs**: https://open-meteo.com/
- **Example**:
```python
import requests
url = "https://api.open-meteo.com/v1/forecast"
params = {"latitude": -1.286389, "longitude": 36.817223, "daily":"temperature_2m_max,temperature_2m_min", "timezone":"Africa/Nairobi"}
r = requests.get(url, params=params)
data = r.json()
```

### 2. IP-API (ip-api.com) — free IP geolocation
- **Use**: infer approximate location (county/region) for customers if you have IP or region hints; power quick map views.
- **Docs**: https://ip-api.com/
- **Example**:
```python
r = requests.get("http://ip-api.com/json/8.8.8.8")
print(r.json())  # country, regionName, city, lat, lon
```

### 3. Lorem Picsum (picsum.photos) — placeholder images (no auth)
- **Use**: customer portrait placeholders, banner images, or meter imagery placeholders to make the UI pop.
- **Docs**: https://picsum.photos/
- **Example URL**: `https://picsum.photos/200/200` or to list: `https://picsum.photos/v2/list?page=2&limit=100`

### 4. World Bank API — macroeconomic indicators (no auth)
- **Use**: show context, e.g., electricity consumption per capita by country or GDP trends on dashboard contextual card.
- **Docs**: https://datahelpdesk.worldbank.org/
- **Example**:
```python
r = requests.get("https://api.worldbank.org/v2/country/ken/indicator/EG.USE.ELEC.KH.PC?format=json")
data = r.json()
```

### 5. Nominatim (OpenStreetMap) geocoding — free geocoding
- **Use**: convert addresses to lat/lon for mapping customer locations or map flag clusters. Use sparingly and cache results.
- **Docs**: https://nominatim.org/
- **Example**:
```python
r = requests.get("https://nominatim.openstreetmap.org/search", params={"q":"Nairobi, Kenya","format":"json"})
coords = r.json()[0]
```

### 6. REST Countries — country metadata (no auth)
- **Use**: add country context or flag imagery for cohort dashboards.
- **Docs**: https://restcountries.com/
- **Example**:
```python
r = requests.get("https://restcountries.com/v3.1/name/kenya")
info = r.json()
```

### 7. Wikimedia / MediaWiki REST — wiki summaries (no auth, rate-limited)
- **Use**: show a short "About [region]" panel or glossary for domain terms.
- **Docs**: https://www.mediawiki.org/wiki/API:REST_API
- **Example**:
```python
r = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/Nairobi")
r.json()["extract"]
```

### Notes on usage & etiquette
Nominatim and Wikimedia have usage policies and rate-limits — cache results and respect robots. For production at scale, provision your own instance or use paid geocoding.

## How to Embed These APIs in Streamlit

- Use `requests` on the server side (not client-side JS) to avoid CORS and exposure of endpoints.
- Cache results: `@st.cache_data(ttl=86400)` for locale data, 1-hour for weather.
- Keep fallback placeholders when API fails (e.g., Picsum fallback local image).
- For maps, use Folium (render via `components.v1.html`) or `pydeck` for cluster maps; Folium can use OSM tiles (no token) and is perfect for customer clusters.

## Example Snippets You Can Paste into Streamlit

### Weather card (small):
```python
import requests, streamlit as st
@st.cache_data(ttl=3600)
def get_weather(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "daily":"temperature_2m_max,temperature_2m_min", "timezone":"Africa/Nairobi"
    })
    return r.json()
if st.button("Show weather"):
    w = get_weather(-1.286389, 36.817223)
    st.write(w["daily"]["temperature_2m_max"][0])
```

### Thumbnail image (picsum):
```python
st.image("https://picsum.photos/300/200", caption="Meter placeholder")
```

### Geocode & map (Nominatim -> Folium embed):
```python
import folium
from streamlit_folium import st_folium
r = requests.get("https://nominatim.openstreetmap.org/search", params={"q":"Nairobi, Kenya","format":"json"})
lat, lon = float(r.json()[0]["lat"]), float(r.json()[0]["lon"])
m = folium.Map(location=[lat, lon], zoom_start=10)
st_folium(m, width=700, height=400)
```

## Final Tactical Checklist

1. Add the UI skeleton pages and header/side controls as above.
2. Implement the Customer portrait card + timeline (Plotly) + SHAP force embed (precompute shap_values).
3. Add small enhancement APIs: Picsum placeholders, Open-Meteo weather card, and REST Countries for contextual info. Cache everything.
4. Add a `Map` page using Folium with OSM tiles (Nominatim geocoding cached).
5. Polish microcopy and train users with clear "what next" suggestions on every flagged case (e.g., "Call field inspector", "Schedule manual reading").

## Implementation Details (Very Specific)

### 1. Data download & loader
- Provide `scripts/download_data.sh` with `kaggle datasets download -d bensalem14/sgcc-dataset -p data --unzip`
- In `src/data_loader.py` implement `load_raw(path)` that loads the CSV (expect encoding issues) and returns a DataFrame:
  - Columns: customer_id, 1..1035 daily consumption columns or single consumption column depending on CSV; detect format and pivot if needed
  - Build `df_long` with columns: customer_id, day_index (0..1034), consumption_kwh
  - Also return `labels` series (0 honest, 1 theft)

### 2. Feature engineering (`src/features.py`)
- Implement `build_features(df_long)` that:
  - Aggregates per customer to compute these features (use rolling windows if needed):
    - **statistical**: mean, median, std, coef_var, min, max, range, skewness
    - **trend**: slope of linear regression over full series, slope on last 30/90 days
    - **temporal**: weekday_vs_weekend_ratio (if dates present), peak_day_ratio (top 10% consumption days frequency)
    - **anomaly**: zero_day_count, sudden_drop_count (days with > 50% drop vs prev), volatility_index (std/mean)
    - **other**: autocorrelation lag-1, number_of_missing_sequences (>3 days)
  - Output final features DataFrame `X` indexed by customer_id and `y` labels
  - Normalize features with `MinMaxScaler` (store scaler to `artifacts/scaler.joblib`)

### 3. SMOTE+ENN preprocessing (`src/preprocessing.py`)
- Implement `apply_smote_enn(X_train, y_train, random_state=42)` using `imblearn.combine.SMOTEENN`
- Return balanced `X_res, y_res`
- Save a short report: class counts before and after in `artifacts/preprocess_report.json`

### 4. Modeling (`src/modeling.py`)
- Implement `get_xgb_model(params=None)` returning `xgboost.XGBClassifier` with sensible defaults:
  - objective='binary:logistic'
  - eval_metric=['auc','logloss']
  - use `use_label_encoder=False`
- Implement `train_xgb_with_optuna(X, y, n_trials=60, cv=5, random_state=42)`:
  - Use Optuna to tune: max_depth, learning_rate, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda, scale_pos_weight (set as ratio or tune)
  - The objective function should optimize a composite score weighted to prioritize recall:
    - score = 0.60 * recall + 0.25 * precision + 0.15 * f1 (measured by stratified CV)
  - Use `StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)`
  - Save best params and best model to `models/xgb_best.joblib` and `artifacts/optuna_study.pkl`

### 5. Evaluation (`src/eval.py`)
- Implement `evaluate_model(model, X_test, y_test)` that returns: recall, precision, f1, auc, gmean, mcc, confusion matrix, per-class support
- Save `artifacts/metrics.json` and feature importance CSV `artifacts/feature_importance.csv`
- Implement `error_analysis(X_test, y_test, preds)` to output example customer IDs for false positives and false negatives (save lists to `artifacts/fp_ids.json` & `fn_ids.json`)

### 6. Explainability
- Integrate SHAP:
  - For tree models use TreeExplainer. Generate and save `artifacts/shap_summary.png` and `artifacts/shap_values.npy`
  - The Streamlit Explain page should render SHAP summary and per-sample force plot for interactive debugging

### 7. Streamlit app (streamlit_app/...)
- **Pages**:
  - **1_EDA.py**: load raw file or cached processed features; show class distribution, top features histograms, daily-consumption heatmap sample, interactive customer consumption timeseries
  - **2_Train.py**: button to train offline (with progress bar). But by default recommend pre-trained model; allow a small-sample quick-train mode for demo (train with 5% sample)
  - **3_Predict.py**: allow upload CSV with customer(s) to predict or input customer_id to fetch cached features and run model -> show probability and binary label (with threshold slider). Show reason via SHAP per-sample.
  - **4_Explain.py**: show global SHAP, top features, and per-sample force plot; allow exporting explanation images.
- The main `app.py` should use `st.cache_resource` / `st.cache_data` to cache model and features
- Load model with `joblib.load("models/xgb_best.joblib")` and scaler `artifacts/scaler.joblib`

### 8. Deployment & operations
- Provide `Dockerfile` that installs requirements and runs `streamlit run streamlit_app/app.py --server.port $PORT --server.enableCORS false`
- Add README with:
  - Kaggle download steps using `KAGGLE_USERNAME` & `KAGGLE_KEY` (place kaggle.json)
  - Local run commands:
    - `python -m venv venv && source venv/bin/activate`
    - `pip install -r requirements.txt`
    - `bash scripts/download_data.sh`
    - `python -m src.train` (to pretrain), or `streamlit run streamlit_app/app.py`
  - Streamlit Cloud deploy instructions: push to GitHub, add secrets (KAGGLE_USERNAME & KAGGLE_KEY), select repo on Streamlit Cloud
  - Resource recommendation: training done on laptop is slow; for full dataset use a 16GB RAM machine; model inference is lightweight and can run in Streamlit cloud.

### 9. Tests & CI
- Provide a minimal `tests/test_data_loader.py` that checks `load_raw` returns expected columns and non-empty DataFrame.
- Provide `.github/workflows/ci.yml` that runs tests and flake8 (basic).

### 10. Additional files
- Provide `notebook/quick_experiments.ipynb` with EDA, feature correlation, and a small training run that reproduces baseline metrics for replication.

## Non-Functional Requirements

- Use clear logging and exceptions
- All hyperparameters and random seeds should be controllable via a top-level `config.yaml`
- Document the expected performance target in README: aim for >= 0.85 recall and >= 0.75 precision on SGCC test split (as per the concept note)
- Use type hints for public functions and docstrings for each function
