# Archived: the "ultra" / GOAT training pipeline

These files are kept for the record only. **Do not use them, and do not cite results produced with them.**

| File | Was at |
|---|---|
| `train_ultra.py` | `scripts/train_ultra.py` |
| `setup_goat_model.py` | `scripts/setup_goat_model.py` |
| `config_ultra.yaml` | `config_ultra.yaml` |
| `data_augmentation.py` | `src/data_augmentation.py` |

The pipeline targeted 85%+ recall and 75%+ precision. It was archived on 2026-09-22 after an audit found that it neither runs nor produces valid data.

## Why it does not run

- `train_ultra.py` adds `scripts/src` to `sys.path` instead of `<repo>/src`, so `from train import train_pipeline` raises `ModuleNotFoundError`.
- `setup_goat_model.py` calls `python train_ultra.py` at the repo root, which no longer exists. It also tells the user to open the removed Streamlit app.
- `data_augmentation.py` imports `faker`, which is not in the requirements.
- `train_ultra.py` reads `metrics['roc_auc']`; the metrics use the key `auc`.

## Why its data is invalid

`data_augmentation.py` generates synthetic customers as ~12–24 *monthly* summary values, then concatenates them with the real SGCC rows, which hold 1,034 *daily* readings. The columns are aligned by filling every missing column with zero. Checked against `data/data_augmented.csv` (72,372 rows):

- **All 30,000 synthetic customers have all-zero daily readings**: every daily column was zero-filled. `load_raw` reads only the daily columns, so every synthetic customer looks the same.
- Synthetic rows are 33.3% theft and real rows 8.5%. Identical all-zero rows carry both labels, so they add label noise and no signal.
- **All 42,372 real customers got `CUSTOMER_ID = "0.0"`**: the column was zero-filled, so the loader treats them as one customer.
- In `src/data_loader.py`, the exclusion list for this format is overwritten a few lines later. As a result, `IS_SYNTHETIC`, `THEFT_TYPE` and the synthetic summary columns are read as consumption values.
- The train/test split happens after augmentation, so any test metric would be measured largely on synthetic rows.
- `data/data_realistic_imbalance.csv` is byte-identical to `data/data_augmented.csv` (same LFS object). The "1:19 realistic" mixture fell back to the full augmented frame.

The augmented CSVs (`data_augmented.csv`, `data_balanced_50_50.csv`, `data_moderate_imbalance.csv`, `data_realistic_imbalance.csv`) are still in `data/`. They are derived from this generator and share the same defects.

## What replaced it

`src/train.py` with `config.yaml` trains on real data only. It tunes for PR-AUC, chooses the decision threshold from an inspection budget using out-of-fold scores on the training split, and records held-out metrics with provenance in `artifacts/metrics.json`. If synthetic augmentation is revisited, it must:

1. generate daily series in the same format as the real data;
2. be applied to the training split only;
3. keep the test split entirely real.
