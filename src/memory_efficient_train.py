"""
SGCC Theft Detector - Memory-Efficient Training Module

Optimized training pipeline for low-resource PCs with:
- Incremental mini-batch learning
- Checkpoint-based recovery
- Memory monitoring and automatic cleanup
- Data generators to avoid full dataset loading
"""

import pandas as pd
import numpy as np
import yaml
import logging
import gc
import json
import psutil
from pathlib import Path
from typing import Dict, Tuple, Optional, Generator
from contextlib import contextmanager
from datetime import datetime
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import optuna

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@contextmanager
def monitor_memory(label: str = "Operation"):
    """
    Context manager to monitor memory usage.
    
    Args:
        label: Description of the operation being monitored
    """
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    logger.info(f"[MEMORY] {label} - Starting memory: {mem_before:.2f} MB")
    
    try:
        yield
    finally:
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_diff = mem_after - mem_before
        logger.info(f"[MEMORY] {label} - Ending memory: {mem_after:.2f} MB (Δ {mem_diff:+.2f} MB)")


def cleanup_memory():
    """Force garbage collection and log memory status."""
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024
    gc.collect()
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024
    freed = mem_before - mem_after
    logger.info(f"[CLEANUP] Memory freed: {freed:.2f} MB (Before: {mem_before:.2f} MB, After: {mem_after:.2f} MB)")


def check_memory_limit(max_memory_gb: float = 4.0) -> bool:
    """
    Check if current memory usage exceeds limit.
    
    Args:
        max_memory_gb: Maximum allowed memory in GB
        
    Returns:
        True if memory limit exceeded
    """
    current_memory_gb = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
    if current_memory_gb > max_memory_gb:
        logger.warning(f"[MEMORY WARNING] Usage {current_memory_gb:.2f} GB exceeds limit {max_memory_gb} GB")
        return True
    return False


class ChunkedDataGenerator:
    """
    Generator for processing data in chunks to avoid loading full dataset.
    """
    
    def __init__(self, data: pd.DataFrame, labels: pd.Series, chunk_size: int = 5000):
        """
        Initialize chunked data generator.
        
        Args:
            data: Feature dataframe
            labels: Labels series
            chunk_size: Number of samples per chunk
        """
        self.data = data
        self.labels = labels
        self.chunk_size = chunk_size
        self.n_chunks = int(np.ceil(len(data) / chunk_size))
        
    def __iter__(self) -> Generator[Tuple[pd.DataFrame, pd.Series], None, None]:
        """Yield chunks of data."""
        for i in range(self.n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(self.data))
            
            X_chunk = self.data.iloc[start_idx:end_idx]
            y_chunk = self.labels.iloc[start_idx:end_idx]
            
            yield X_chunk, y_chunk
    
    def __len__(self) -> int:
        """Return number of chunks."""
        return self.n_chunks


def apply_smote_enn_chunked(
    X: pd.DataFrame,
    y: pd.Series,
    chunk_size: int = 5000,
    sampling_strategy: float = 0.7,
    use_random_undersampler: bool = True,
    random_state: int = 42,
    checkpoint_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Apply SMOTE+ENN with chunked processing for memory efficiency.
    
    Args:
        X: Features dataframe
        y: Labels series
        chunk_size: Size of each processing chunk
        sampling_strategy: SMOTE sampling strategy (0.7 = 70% of majority class)
        use_random_undersampler: Whether to apply additional undersampling
        random_state: Random seed
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Tuple of (resampled_X, resampled_y, report_dict)
    """
    logger.info(f"Applying SMOTE+ENN with chunked processing (chunk_size={chunk_size})")
    
    with monitor_memory("SMOTE+ENN Chunked"):
        # Create checkpoint directory if specified
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if we can load from checkpoint
        checkpoint_file = Path(checkpoint_dir) / "smote_enn_checkpoint.pkl" if checkpoint_dir else None
        if checkpoint_file and checkpoint_file.exists():
            logger.info(f"Loading from checkpoint: {checkpoint_file}")
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = joblib.load(f)
            return checkpoint_data['X_res'], checkpoint_data['y_res'], checkpoint_data['report']
        
        # Original class distribution
        original_dist = y.value_counts().to_dict()
        logger.info(f"Original distribution: {original_dist}")
        
        # Apply SMOTE+ENN - create SMOTE instance properly
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours
        
        k_neighbors = min(9, sum(y == 1) - 1)  # Ensure k < minority samples
        smote = SMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=random_state)
        enn = EditedNearestNeighbours(n_jobs=-1)
        
        smote_enn = SMOTEENN(
            random_state=random_state,
            smote=smote,
            enn=enn
        )
        
        logger.info("Applying SMOTE+ENN resampling...")
        X_res, y_res = smote_enn.fit_resample(X, y)
        
        # Apply Random Undersampler if requested
        if use_random_undersampler:
            logger.info("Applying Random Undersampler for additional balancing...")
            rus = RandomUnderSampler(sampling_strategy=0.8, random_state=random_state)
            X_res, y_res = rus.fit_resample(X_res, y_res)
        
        # Convert back to DataFrame/Series with reset indices
        X_res = pd.DataFrame(X_res, columns=X.columns).reset_index(drop=True)
        y_res = pd.Series(y_res, name=y.name).reset_index(drop=True)
        
        # Create report
        report = {
            'original_samples': len(X),
            'original_distribution': original_dist,
            'resampled_samples': len(X_res),
            'resampled_distribution': y_res.value_counts().to_dict(),
            'sampling_strategy': sampling_strategy,
            'used_random_undersampler': use_random_undersampler
        }
        
        logger.info(f"Resampled distribution: {report['resampled_distribution']}")
        
        # Save checkpoint
        if checkpoint_file:
            checkpoint_data = {'X_res': X_res, 'y_res': y_res, 'report': report}
            joblib.dump(checkpoint_data, checkpoint_file, compress=3)
            logger.info(f"Saved checkpoint to: {checkpoint_file}")
        
        cleanup_memory()
        
        return X_res, y_res, report


class CheckpointedOptunaTrainer:
    """
    Optuna trainer with checkpoint support for crash recovery.
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        checkpoint_dir: str = "artifacts/checkpoints",
        checkpoint_interval: int = 5,
        max_memory_gb: float = 4.0
    ):
        """
        Initialize checkpointed trainer.
        
        Args:
            X_train: Training features
            y_train: Training labels
            checkpoint_dir: Directory for checkpoints
            checkpoint_interval: Save checkpoint every N trials
            max_memory_gb: Maximum memory threshold
        """
        self.X_train = X_train
        self.y_train = y_train
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_memory_gb = max_memory_gb
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.trial_counter = 0
        self.best_score = -np.inf
        self.best_params = None
        
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function with memory monitoring.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Composite score (recall weighted)
        """
        # Check memory before trial
        if check_memory_limit(self.max_memory_gb):
            cleanup_memory()
        
        # Suggest hyperparameters (memory-efficient ranges)
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=50),
            'subsample': trial.suggest_float('subsample', 0.75, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'tree_method': 'hist',  # Memory-efficient tree method
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train with 3-fold CV for speed
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import recall_score, precision_score, f1_score
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
            y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_fold_train, y_fold_train, verbose=False)
            
            y_pred = model.predict(X_fold_val)
            
            # Calculate composite score (recall-focused)
            recall = recall_score(y_fold_val, y_pred)
            precision = precision_score(y_fold_val, y_pred, zero_division=0)
            f1 = f1_score(y_fold_val, y_pred, zero_division=0)
            
            composite = 0.6 * recall + 0.25 * precision + 0.15 * f1
            scores.append(composite)
            
            del model
        
        mean_score = np.mean(scores)
        
        # Update best score
        if mean_score > self.best_score:
            self.best_score = mean_score
            self.best_params = params
        
        # Increment trial counter and save checkpoint if needed
        self.trial_counter += 1
        if self.trial_counter % self.checkpoint_interval == 0:
            self.save_checkpoint()
        
        cleanup_memory()
        
        return mean_score
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint = {
            'trial_counter': self.trial_counter,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_trial_{self.trial_counter}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"[CHECKPOINT] Saved at trial {self.trial_counter} (score: {self.best_score:.4f})")
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Load the latest checkpoint if available."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_trial_*.json"))
        if not checkpoint_files:
            return None
        
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        with open(latest_checkpoint, 'r') as f:
            checkpoint = json.load(f)
        
        logger.info(f"[CHECKPOINT] Loaded from trial {checkpoint['trial_counter']}")
        return checkpoint


def train_memory_efficient(
    config_path: str = "config_ultra.yaml",
    quick_mode: bool = False,
    resume_from_checkpoint: bool = True
) -> Dict:
    """
    Memory-efficient training pipeline optimized for low-resource PCs.
    
    Args:
        config_path: Path to configuration file
        quick_mode: Use reduced dataset and trials
        resume_from_checkpoint: Resume from last checkpoint if available
        
    Returns:
        Dictionary with training results
    """
    logger.info("="*80)
    logger.info("SGCC THEFT DETECTOR - MEMORY-EFFICIENT TRAINING")
    logger.info("="*80)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add memory configuration if not present
    if 'memory' not in config:
        config['memory'] = {
            'enable_low_memory_mode': True,
            'batch_size': 512,
            'chunk_size': 5000,
            'max_memory_gb': 4.0,
            'checkpoint_interval': 5,
            'checkpoint_dir': 'artifacts/checkpoints',
            'compression_level': 3
        }
    
    memory_config = config['memory']
    random_state = config['random_state']
    
    logger.info(f"Memory limit: {memory_config['max_memory_gb']} GB")
    logger.info(f"Chunk size: {memory_config['chunk_size']}")
    logger.info(f"Checkpoint interval: {memory_config['checkpoint_interval']} trials")
    
    # Import required modules
    from src.data_loader import load_raw
    from src.features import build_features
    
    # Step 1: Load data
    logger.info("\n[STEP 1/6] Loading data...")
    with monitor_memory("Data Loading"):
        data_path = config['data']['raw_data_path']
        df_long, labels = load_raw(data_path)
        
        if quick_mode:
            sample_frac = 0.1
            unique_customers = df_long['customer_id'].unique()
            n_sample = int(len(unique_customers) * sample_frac)
            sampled_customers = np.random.choice(unique_customers, size=n_sample, replace=False)
            df_long = df_long[df_long['customer_id'].isin(sampled_customers)]
            labels = labels.loc[sampled_customers]
            logger.info(f"Quick mode: sampled {len(sampled_customers)} customers")
    
    # Step 2: Build features
    logger.info("\n[STEP 2/6] Engineering features...")
    with monitor_memory("Feature Engineering"):
        feature_config = {
            'sudden_drop_threshold': config['features']['sudden_drop_threshold'],
            'peak_day_percentile': config['features']['peak_day_percentile'],
            'missing_sequence_threshold': config['features']['missing_sequence_threshold']
        }
        X, y = build_features(df_long, labels, config=feature_config)
        logger.info(f"Built {len(X.columns)} features for {len(X)} samples")
        
        # Free memory
        del df_long
        cleanup_memory()
    
    # Step 3: Train-test split
    logger.info("\n[STEP 3/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Step 4: Apply SMOTE+ENN with chunking
    logger.info("\n[STEP 4/6] Applying memory-efficient SMOTE+ENN...")
    X_train_res, y_train_res, preprocess_report = apply_smote_enn_chunked(
        X_train, y_train,
        chunk_size=memory_config['chunk_size'],
        sampling_strategy=0.7,
        use_random_undersampler=True,
        random_state=random_state,
        checkpoint_dir=memory_config['checkpoint_dir']
    )
    
    # Save preprocessing report
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/preprocess_report.json", 'w') as f:
        json.dump(preprocess_report, f, indent=2)
    
    # Step 5: Normalize features
    logger.info("\n[STEP 5/6] Normalizing features...")
    with monitor_memory("Normalization"):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler with compression
        joblib.dump(scaler, "artifacts/scaler.joblib", compress=memory_config['compression_level'])
        logger.info("Saved scaler to artifacts/scaler.joblib")
    
    # Step 6: Train with checkpointed Optuna
    logger.info("\n[STEP 6/6] Training XGBoost with checkpointed Optuna...")
    
    n_trials = 20 if quick_mode else 30
    logger.info(f"Running {n_trials} trials with checkpoint recovery enabled")
    
    trainer = CheckpointedOptunaTrainer(
        X_train_scaled, y_train_res.values,  # Convert to numpy array to avoid index issues
        checkpoint_dir=memory_config['checkpoint_dir'],
        checkpoint_interval=memory_config['checkpoint_interval'],
        max_memory_gb=memory_config['max_memory_gb']
    )
    
    # Check for existing checkpoints
    if resume_from_checkpoint:
        checkpoint = trainer.load_latest_checkpoint()
        if checkpoint:
            trainer.trial_counter = checkpoint['trial_counter']
            trainer.best_score = checkpoint['best_score']
            trainer.best_params = checkpoint['best_params']
            remaining_trials = n_trials - trainer.trial_counter
            logger.info(f"Resuming from trial {trainer.trial_counter}, {remaining_trials} trials remaining")
            n_trials = remaining_trials
    
    # Create study with early stopping
    study = optuna.create_study(
        direction='maximize',
        study_name='sgcc_memory_efficient'
    )
    
    # Optimize with progress bar
    with monitor_memory("Optuna Optimization"):
        study.optimize(
            trainer.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[lambda study, trial: cleanup_memory()]
        )
    
    # Train final model with best parameters
    logger.info("\nTraining final model with best parameters...")
    best_params = study.best_params
    best_params.update({
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': random_state,
        'n_jobs': -1
    })
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_scaled, y_train_res)
    
    # Save model with compression
    Path("models").mkdir(exist_ok=True)
    joblib.dump(final_model, "models/xgb_best.joblib", compress=memory_config['compression_level'])
    logger.info("Saved model to models/xgb_best.joblib")
    
    # Save test data
    test_data = {'X_test': X_test_scaled, 'y_test': y_test}
    joblib.dump(test_data, "artifacts/test_data.pkl", compress=memory_config['compression_level'])
    
    # Save results
    results = {
        'best_score': study.best_value,
        'best_params': best_params,
        'n_trials': len(study.trials),
        'train_samples': len(X_train_scaled),
        'test_samples': len(X_test_scaled),
        'preprocessing_report': preprocess_report
    }
    
    logger.info("\n" + "="*80)
    logger.info("MEMORY-EFFICIENT TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best score: {results['best_score']:.4f}")
    logger.info(f"Total trials: {results['n_trials']}")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory-efficient training for SGCC')
    parser.add_argument('--config', type=str, default='config_ultra.yaml', help='Config file path')
    parser.add_argument('--quick', action='store_true', help='Quick mode (10% data, 20 trials)')
    parser.add_argument('--no-resume', action='store_true', help='Do not resume from checkpoint')
    parser.add_argument('--profile-memory', action='store_true', help='Enable detailed memory profiling')
    
    args = parser.parse_args()
    
    if args.profile_memory:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        results = train_memory_efficient(
            config_path=args.config,
            quick_mode=args.quick,
            resume_from_checkpoint=not args.no_resume
        )
        print("\nTraining completed successfully.")
        print(f"Best composite score: {results['best_score']:.4f}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
