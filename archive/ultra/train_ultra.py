"""
Ultra-Optimized Training Script for 85%+ Recall
Uses augmented dataset and optimized hyperparameters.
"""

import sys
import logging
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import from existing modules
from train import train_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config_path='config_ultra.yaml', quick=False, lightning=False):
    """
    Train ultra-optimized model for maximum recall.
    
    Args:
        config_path: Path to configuration file
        quick: If True, reduce trials to 20 for faster testing
        lightning: If True, super fast mode (10 trials, 3-fold CV, 10% sample)
    """
    logger.info("="*80)
    if lightning:
        logger.info("LIGHTNING MODE - SUPER FAST TRAINING")
    else:
        logger.info("ULTRA-OPTIMIZED TRAINING FOR 85%+ RECALL")
    logger.info("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if lightning:
        logger.info("LIGHTNING MODE: 10 trials, 3-fold CV, 40% sample (~29K customers)")
        config['model']['optuna']['n_trials'] = 10
        config['model']['optuna']['cv_folds'] = 3
        config['model']['quick_train']['n_trials'] = 10
        config['model']['quick_train']['sample_fraction'] = 0.4  # ~29K customers (similar to datasetsmall.csv)
        quick = True  # Enable quick mode features
    elif quick:
        logger.info("QUICK MODE: Reducing trials to 20")
        config['model']['optuna']['n_trials'] = 20
        config['model']['quick_train']['n_trials'] = 20
    
    # Check if augmented data exists
    data_path = Path(config['data']['raw_data_path'])
    if not data_path.exists():
        logger.warning(f"Augmented dataset not found at {data_path}")
        logger.info("Please run data augmentation first:")
        logger.info("  python -m src.data_augmentation --n-honest 20000 --n-theft 10000")
        logger.info("\nFalling back to original dataset...")
        config['data']['raw_data_path'] = 'data/data_raw.csv'
    
    # Save modified config temporarily
    temp_config = Path('temp_ultra_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"\nStarting training with config: {config_path}")
    logger.info(f"Optuna trials: {config['model']['optuna']['n_trials']}")
    logger.info(f"Target recall: ≥ 85%")
    
    # Run training pipeline
    try:
        results = train_pipeline(str(temp_config), quick_mode=quick)
        
        # Display results
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE - FINAL RESULTS")
        logger.info("="*80)
        
        if results and 'metrics' in results:
            metrics = results['metrics']
            logger.info(f"\nTest Set Performance:")
            logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
            logger.info(f"  Precision: {metrics.get('precision', 0):.4f} ({metrics.get('precision', 0)*100:.2f}%)")
            recall_status = "PASS" if metrics.get('recall', 0) >= 0.85 else "FAIL"
            logger.info(
                f"  Recall:    {metrics.get('recall', 0):.4f} "
                f"({metrics.get('recall', 0)*100:.2f}%) {recall_status}"
            )
            logger.info(f"  F1 Score:  {metrics.get('f1', 0):.4f} ({metrics.get('f1', 0)*100:.2f}%)")
            logger.info(f"  ROC AUC:   {metrics.get('roc_auc', 0):.4f} ({metrics.get('roc_auc', 0)*100:.2f}%)")
            
            # Check if target achieved
            if metrics.get('recall', 0) >= 0.85:
                logger.info("\nTARGET ACHIEVED: Recall >= 85%")
            else:
                logger.info(
                    f"\nTarget not achieved: Recall {metrics.get('recall', 0)*100:.1f}% < 85%"
                )
                logger.info("Consider:")
                logger.info("  1. Generating more synthetic data")
                logger.info("  2. Increasing SMOTE sampling_strategy to 0.98")
                logger.info("  3. Running more Optuna trials (200+)")
        
        logger.info("\n" + "="*80)
        logger.info("Model saved successfully!")
        logger.info("="*80)
        
        return results
        
    finally:
        # Cleanup temp config
        if temp_config.exists():
            temp_config.unlink()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ultra-optimized theft detection model')
    parser.add_argument('--config', type=str, default='config_ultra.yaml',
                       help='Path to config file')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 20 trials (5-10 min)')
    parser.add_argument('--lightning', action='store_true',
                       help='Lightning mode: 10 trials, 3-fold CV, 40%% sample = ~29K customers (~5 min)')
    
    args = parser.parse_args()
    
    main(args.config, args.quick, args.lightning)
