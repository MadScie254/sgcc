# SGCC Platform Enhancements - Quick Start Guide

## What's New

The SGCC platform has been enhanced with powerful new features across **3 major phases**:

### Phase 1: Memory-Efficient Training (IMPLEMENTED)

Train models on low-resource PCs with intelligent memory management.

### Phase 3: Research Validation Dashboard (IMPLEMENTED)

Comprehensive validation of research objectives  with advanced metrics.

### Phase 5: Public API Integrations (IMPLEMENTED)

No-auth APIs for weather, geocoding, holidays, and more.

### Phase 2: Advanced EDA Backend (IMPLEMENTED)

Advanced analytics module ready for integration into EDA page.

---

## Installation

### 1. Update Dependencies

```bash
pip install -r requirements.txt
```

**New packages installed:**

- `psutil` - Memory monitoring
- `tqdm` - Progress bars
- `statsmodels` - Time-series analysis
- `ruptures` - Changepoint detection
- `geopy` - Geocoding
- `networkx` - Network graphs
- `fpdf2` - PDF generation
- `hypothesis` - Property-based testing
- `mypy`, `pre-commit` - Code quality tools

### 2. Update Configuration

The `config_ultra.yaml` now includes a **memory management** section:

```yaml
memory:
  enable_low_memory_mode: true
  max_memory_gb: 4.0  # Adjust for your PC
  chunk_size: 5000
  checkpoint_interval: 5
  checkpoint_dir: "artifacts/checkpoints"
```

**Action:** Review and adjust `max_memory_gb` based on your PC's RAM.

---

## Usage Guide

### Memory-Efficient Training

**Quick Mode** (10% data, 20 trials, ~5-10 minutes):

```bash
python -m src.memory_efficient_train --quick
```

**Full Mode** (100% data, 30 trials, ~1-2 hours):

```bash
python -m src.memory_efficient_train
```

**Resume from Checkpoint** (automatic if interrupted):

```bash
python -m src.memory_efficient_train  # Automatically resumes
```

**Disable Checkpoint Recovery**:

```bash
python -m src.memory_efficient_train --no-resume
```

**Profile Memory Usage**:

```bash
python -m src.memory_efficient_train --profile-memory
```

#### Features

- **Memory monitoring** - Tracks RAM usage in real-time
- **Automatic cleanup** - Frees memory between trials
- **Checkpoint recovery** - Resume training after crash
- **Chunked processing** - SMOTE+ENN in 5,000-sample chunks
- **Compressed artifacts** - Saves disk space

#### Expected Memory Usage

- **Quick mode**: ~1-2 GB RAM
- **Full mode**: ~2-4 GB RAM (configurable)

---

### Research Validation Dashboard

Launch the Streamlit app:

```bash
streamlit run streamlit_app/app.py
```

Then navigate to: **Research Validation** (page 8)

#### Features

**Objective 1: Detection Performance**

- Interactive precision-recall curves
- ROC curves with 95% confidence intervals
- Lift curves showing model effectiveness
- Calibration curves for probability reliability
- Confusion matrix with adjustable threshold
- Cost-benefit calculator

**Objective 2: Feature Importance**

- Top 15 features visualization
- Links to detailed SHAP analysis

**Objective 3: Model Comparison**

- Comparison tables (when baseline models run)
- Multi-metric radar charts (coming soon)

**Objective 4: Deployment Readiness**

- Deployment checklist
- Production readiness assessment

---

### API Integrations

All APIs are automatically used throughout the platform with caching and rate limiting.

#### Available APIs

1. **Weather Data** (Open-Meteo)

   ```python
   from src.api_integrations import fetch_weather_data
   
   weather = fetch_weather_data(
       latitude=36.7,
       longitude=3.2,
       start_date="2024-01-01",
       end_date="2024-01-31"
   )
   ```

2. **Geocoding** (Nominatim)

   ```python
   from src.api_integrations import geocode_address
   
   lat, lon = geocode_address("Algiers, Algeria")
   ```

3. **Holidays** (Nager.Date)

   ```python
   from src.api_integrations import get_public_holidays
   
   holidays = get_public_holidays(country_code='DZ', year=2024)
   ```

4. **Currency Conversion** (ExchangeRate-API)

   ```python
   from src.api_integrations import convert_currency
   
   usd_amount = convert_currency(1000, from_currency='DZD', to_currency='USD')
   ```

5. **User Location** (ipapi.co)

   ```python
   from src.api_integrations import get_user_location
   
   location = get_user_location()  # Auto-detect from IP
   ```

6. **Random Profiles** (RandomUser.me)

   ```python
   from src.api_integrations import generate_customer_profile
   
   profile = generate_customer_profile(seed="demo123")
   ```

#### Widget Examples

Add to any Streamlit page:

```python
from src.api_integrations import (
    display_weather_widget,
    display_location_widget,
    display_fun_fact_widget
)

# In your Streamlit app
display_weather_widget(latitude=36.7, longitude=3.2)
display_location_widget()
display_fun_fact_widget()
```

---

### Advanced EDA Functions

The `src/advanced_eda.py` module provides powerful analytics:

```python
from src.advanced_eda import (
    decompose_seasonality,
    detect_changepoints,
    calculate_acf_pacf,
    calculate_rolling_statistics,
    calculate_anomaly_scores,
    calculate_feature_importance_rf,
    perform_hypothesis_tests,
    cluster_consumption_patterns
)

# Example: Decompose time series
decomposition = decompose_seasonality(consumption_series, period=30)

# Example: Detect change points
changepoints = detect_changepoints(consumption_series, n_changepoints=5)

# Example: Statistical tests
test_results = perform_hypothesis_tests(features_df, labels)
```

**To integrate into EDA page**: Import these functions in `streamlit_app/pages/1_EDA.py` and add new tabs for advanced visualizations.

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Test Memory-Efficient Training

```bash
pytest tests/test_memory_efficient_train.py -v
```

### With Coverage Report

```bash
pytest tests/ -v --cov=src --cov-report=html
```

Then open `htmlcov/index.html` to view coverage.

---

## File Structure (New Files)

```
sgcc/
├── src/
│   ├── memory_efficient_train.py    # NEW - Memory-optimized training
│   ├── advanced_eda.py              # NEW - Advanced analytics backend
│   └── api_integrations.py          # NEW - Public API integrations
├── streamlit_app/pages/
│   └── 8_Research_Validation.py     # NEW - Research validation dashboard
├── tests/
│   └── test_memory_efficient_train.py # NEW - Memory training tests
├── config_ultra.yaml                # UPDATED - Added memory section
└── requirements.txt                 # UPDATED - Added new packages
```

---

## Next Steps

### Immediate Actions

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Test memory-efficient training** (quick mode):

   ```bash
   python -m src.memory_efficient_train --quick
   ```

3. **Launch Streamlit app**:

   ```bash
   streamlit run streamlit_app/app.py
   ```

4. **Explore Research Validation page** (page 8)

### Future Enhancements (Not Yet Implemented)

- **Phase 2**: Integrate advanced EDA functions into EDA page
- **Phase 4**: UI/UX transformation with premium design
- **Phase 6**: Automated reporting and what-if analysis
- **Phase 7**: Performance optimization and comprehensive testing

---

## Configuration Tips

### Low-Memory PC (2-4 GB RAM)

```yaml
memory:
  max_memory_gb: 2.0
  chunk_size: 3000
  checkpoint_interval: 3
```

### Standard PC (8+ GB RAM)

```yaml
memory:
  max_memory_gb: 6.0
  chunk_size: 10000
  checkpoint_interval: 10
```

### High-Performance (16+ GB RAM)

```yaml
memory:
  max_memory_gb: 12.0
  chunk_size: 20000
  checkpoint_interval: 20
```

---

## Troubleshooting

### Issue: Memory still too high

**Solution**: Reduce `chunk_size` and `max_memory_gb` in config.

### Issue: Training very slow

**Solution**: Increase `chunk_size` if you have more RAM available.

### Issue: Checkpoint not recovering

**Solution**: Check `artifacts/checkpoints/` directory exists and has write permissions.

### Issue: API calls failing

**Solution**: Check internet connection. APIs will gracefully fail and return None - the app continues to work.

### Issue: Import errors

**Solution**: Ensure you're running from project root and dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

---

## Documentation

- **Implementation Plan**: See `brain/implementation_plan.md`
- **Task List**: See `brain/task.md`
- **Main README**: See `README.md`

---

## Research Validation

The Research Validation page demonstrates:

1. **High recall detection** (≥85% target)
2. **Explainable predictions** (SHAP integration)
3. **Model comparison** (vs baselines)
4. **Production readiness** (deployment checklist)

All aligned with your research objectives!

---

## Tips

- **Start with quick mode** to verify everything works
- **Monitor memory usage** with `--profile-memory` flag
- **Check checkpoints** after every 5 trials
- **Adjust threshold** in Research Validation page to balance precision/recall
- **Use APIs sparingly** to respect rate limits (caching helps!)

---

**You're all set! Happy training and validating!**
