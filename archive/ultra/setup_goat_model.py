"""
Complete Setup Script for Ultra-High Accuracy Model
Generates synthetic data and trains GOAT-level model.
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a command and handle errors."""
    logger.info(f"\n{'='*80}")
    logger.info(f"{description}")
    logger.info(f"{'='*80}")
    logger.info(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        return False
    return True


def main():
    """Run complete setup."""
    logger.info("\n" + "="*80)
    logger.info("GOAT-LEVEL MODEL SETUP")
    logger.info("Ultra-High Accuracy Electricity Theft Detection")
    logger.info("="*80)
    
    # Step 1: Generate augmented data
    logger.info("\nStep 1/3: Generating synthetic dataset...")
    logger.info("Creating 20,000 honest + 10,000 theft customers (30K+ total)...")
    
    if not run_command(
        "python -m src.data_augmentation --n-honest 20000 --n-theft 10000 --create-mixtures",
        "GENERATING MASSIVE AUGMENTED DATASET (30K+)"
    ):
        logger.error("Failed to generate augmented data")
        return 1
    
    # Step 2: Quick test training (20 trials)
    logger.info("\n\nStep 2/3: Quick test training (20 trials)...")
    logger.info("This validates everything works before full training...")
    
    response = input("\nRun quick test? (y/n): ").strip().lower()
    if response == 'y':
        if not run_command(
            "python train_ultra.py --config config_ultra.yaml --quick",
            "QUICK TEST TRAINING (20 TRIALS)"
        ):
            logger.error("Quick test failed")
            return 1
    else:
        logger.info("Skipping quick test")
    
    # Step 3: Full training (150 trials)
    logger.info("\n\nStep 3/3: Full ultra-optimized training...")
    logger.info("This will run 150 trials - may take 2-4 hours")
    logger.info("Target: Recall ≥ 85%")
    
    response = input("\nStart full training? (y/n): ").strip().lower()
    if response == 'y':
        if not run_command(
            "python train_ultra.py --config config_ultra.yaml",
            "FULL ULTRA-OPTIMIZED TRAINING (150 TRIALS)"
        ):
            logger.error("Full training failed")
            return 1
    else:
        logger.info("Skipping full training")
        logger.info("You can run it later with: python train_ultra.py")
    
    logger.info("\n" + "="*80)
    logger.info("SETUP COMPLETE!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("1. Run: streamlit run streamlit_app/app.py")
    logger.info("2. Navigate to Prediction page")
    logger.info("3. Test AI recommendations with high-accuracy model")
    logger.info("\nDatasets created:")
    logger.info("  - data/data_augmented.csv (full dataset)")
    logger.info("  - data/data_balanced_50_50.csv (balanced)")
    logger.info("  - data/data_realistic_imbalance.csv (5% theft)")
    logger.info("  - data/data_moderate_imbalance.csv (20% theft)")
    logger.info("\nModels will be saved to:")
    logger.info("  - models/xgb_best_ultra.joblib")
    logger.info("  - models/scaler_ultra.joblib")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
