# 🚀 GOAT-Level Model Training Guide

## Achieving 85%+ Recall with Advanced Data Augmentation

### Quick Start (Recommended)

```bash
# One command to rule them all
python setup_goat_model.py
```

This will:
1. ✅ Generate 15,000 synthetic customers (10K honest + 5K theft)
2. ✅ Create multiple dataset mixtures
3. ✅ Run quick test (20 trials, ~10 min)
4. ✅ Run full training (150 trials, ~2-4 hours)

---

## Manual Steps

### 1. Generate Synthetic Data

```bash
# Generate augmented dataset
python -m src.data_augmentation \
  --n-honest 10000 \
  --n-theft 5000 \
  --create-mixtures
```

**Output:**
- `data/data_augmented.csv` - Full dataset (15K+ samples)
- `data/data_balanced_50_50.csv` - Perfect balance
- `data/data_realistic_imbalance.csv` - 5% theft (realistic)
- `data/data_moderate_imbalance.csv` - 20% theft

**Data Generation Features:**
- 🏠 5 customer profiles (residential low/medium/high, commercial small/large)
- 🔧 4 theft patterns (meter tampering, bypass, reverse metering, magnetic interference)
- 📊 Realistic seasonal variations
- 🎲 Random theft start months (3+ months after registration)
- 📈 20+ engineered features per customer

---

### 2. Train Ultra-Optimized Model

#### Quick Test (20 trials, ~10 minutes)
```bash
python train_ultra.py --quick
```

#### Full Training (150 trials, 2-4 hours)
```bash
python train_ultra.py
```

**Ultra Config Highlights:**
- **SMOTE**: k_neighbors=9, sampling_strategy=0.95 (95% balance!)
- **ENN**: n_neighbors=7 (maximum noise removal)
- **XGBoost**: 150 trials, 200-1200 estimators, depth 5-15
- **Scoring**: 60% recall, 25% precision, 15% F1
- **Threshold**: 0.4 (lower = higher recall)

---

## Configuration Comparison

| Parameter | Standard | Ultra-Optimized | Improvement |
|-----------|----------|-----------------|-------------|
| **Data Samples** | ~1K | 15K+ | 15x more |
| **SMOTE k_neighbors** | 5 | 9 | Better diversity |
| **Sampling Strategy** | 0.8 | 0.95 | Near-perfect balance |
| **ENN Neighbors** | 3 | 7 | Stronger noise removal |
| **Optuna Trials** | 60 | 150 | 2.5x optimization |
| **Max Estimators** | 500 | 1200 | More learning capacity |
| **Max Depth** | 10 | 15 | Deeper patterns |
| **Recall Weight** | 40% | 60% | Maximized detection |

---

## Expected Results

### Target Metrics
- ✅ **Recall**: ≥ 85% (catch 85%+ of thieves)
- ✅ **Precision**: ≥ 75% (minimize false alarms)
- ✅ **F1 Score**: ≥ 80% (balanced performance)
- ✅ **Accuracy**: ≥ 82% (overall correctness)

### Confusion Matrix (Target)
```
              Predicted
              Honest  Theft
Actual Honest  [TN]   [FP]   ← Keep FP low
      Theft    [FN]   [TP]   ← Keep FN very low (max 15%)
                ↑
        Max 15% missed thefts
```

---

## Theft Detection Patterns

### 1. **Meter Tampering** (40% reduction)
- Sudden drop in consumption
- Erratic monthly patterns
- High volatility

### 2. **Bypass** (60% reduction)
- Dramatic consumption drop
- Consistent low usage
- Near-zero months

### 3. **Reverse Metering** (50% reduction)
- Negative consumption spikes
- Inconsistent patterns
- High variability

### 4. **Magnetic Interference** (30% reduction)
- Moderate drops
- Irregular readings
- Random fluctuations

---

## Using the Models

### In Streamlit App

The prediction page automatically uses the latest model:

```python
# Model selection in app.py
model_path = "models/xgb_best_ultra.joblib"  # Ultra model
scaler_path = "models/scaler_ultra.joblib"
```

### Programmatic Usage

```python
import joblib
import pandas as pd

# Load ultra model
model = joblib.load('models/xgb_best_ultra.joblib')
scaler = joblib.load('models/scaler_ultra.joblib')

# Predict
X_scaled = scaler.transform(X_test)
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)[:, 1]

# High confidence thefts (>80%)
high_confidence = probabilities > 0.8
```

---

## Troubleshooting

### "Not enough values to unpack" error
✅ **FIXED** - Confusion matrix now properly handles dict and array formats

### Low recall (<85%)
Try these adjustments in `config_ultra.yaml`:

```yaml
preprocessing:
  smote_enn:
    smote:
      sampling_strategy: 0.98  # Even more balance

model:
  optuna:
    n_trials: 200  # More optimization
    scoring_weights:
      recall: 0.70  # Maximize recall even more
```

### Out of memory
Reduce dataset size:

```bash
python -m src.data_augmentation --n-honest 5000 --n-theft 2500
```

---

## Validation

After training, check these metrics:

```bash
# View model performance
python -c "
import joblib
model = joblib.load('models/xgb_best_ultra.joblib')
print('Model loaded successfully')
print(f'Features: {model.n_features_in_}')
print(f'Classes: {model.classes_}')
"
```

---

## Next Steps

1. ✅ Generate data: `python -m src.data_augmentation`
2. ✅ Train model: `python train_ultra.py`
3. ✅ Launch app: `streamlit run streamlit_app/app.py`
4. ✅ Test predictions on AI-powered page
5. ✅ Review AI recommendations for theft/honest cases

---

## Architecture

```
Data Pipeline:
  Original Data (1K samples)
       ↓
  Data Augmentation (15K samples)
       ↓
  SMOTE+ENN Balancing (30K samples, 95% balanced)
       ↓
  Feature Engineering (20+ features)
       ↓
  XGBoost Training (150 trials, 5-fold CV)
       ↓
  Ultra-Optimized Model (85%+ recall)
       ↓
  AI Recommendation Engine
       ↓
  Streamlit App (Canvas-level UI)
```

---

## Performance Targets

| Metric | Baseline | Target | Elite |
|--------|----------|--------|-------|
| Recall | 40-60% | **85%** | 90%+ |
| Precision | 60-70% | **75%** | 80%+ |
| F1 Score | 50-65% | **80%** | 85%+ |
| Accuracy | 70-75% | **82%** | 87%+ |

**GOAT Level = All Elite metrics achieved!** 🐐

---

## Support

For issues or improvements:
1. Check logs in `logs/` directory
2. Review Optuna study results
3. Adjust config parameters
4. Generate more synthetic data

**Remember:** More data + better balance + deeper trees = GOAT model! 🚀
