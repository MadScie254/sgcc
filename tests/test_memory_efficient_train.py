"""
Tests for memory-efficient training module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.memory_efficient_train import (
    monitor_memory,
    cleanup_memory,
    check_memory_limit,
    ChunkedDataGenerator,
    apply_smote_enn_chunked,
    CheckpointedOptunaTrainer
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]), name='label')
    
    return X, y


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_monitor_memory():
    """Test memory monitoring context manager."""
    with monitor_memory("Test Operation"):
        # Create some data
        _ = np.random.randn(1000, 1000)
    # Should complete without errors


def test_cleanup_memory():
    """Test memory cleanup function."""
    # Create some garbage
    _ = [np.random.randn(100, 100) for _ in range(10)]
    cleanup_memory()
    # Should complete without errors


def test_check_memory_limit():
    """Test memory limit checking."""
    # Should return False for very high limit
    assert check_memory_limit(max_memory_gb=1000.0) == False
    
    # Should return True for very low limit
    assert check_memory_limit(max_memory_gb=0.001) == True


def test_chunked_data_generator(sample_data):
    """Test chunked data generator."""
    X, y = sample_data
    chunk_size = 300
    
    generator = ChunkedDataGenerator(X, y, chunk_size=chunk_size)
    
    # Check number of chunks
    expected_chunks = int(np.ceil(len(X) / chunk_size))
    assert len(generator) == expected_chunks
    
    # Iterate through chunks
    total_samples = 0
    for X_chunk, y_chunk in generator:
        assert isinstance(X_chunk, pd.DataFrame)
        assert isinstance(y_chunk, pd.Series)
        assert len(X_chunk) == len(y_chunk)
        total_samples += len(X_chunk)
    
    assert total_samples == len(X)


def test_apply_smote_enn_chunked(sample_data, temp_checkpoint_dir):
    """Test chunked SMOTE+ENN application."""
    X, y = sample_data
    
    X_res, y_res, report = apply_smote_enn_chunked(
        X, y,
        chunk_size=500,
        sampling_strategy=0.7,
        use_random_undersampler=True,
        random_state=42,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Check outputs
    assert isinstance(X_res, pd.DataFrame)
    assert isinstance(y_res, pd.Series)
    assert len(X_res) == len(y_res)
    
    # Check report
    assert 'original_samples' in report
    assert 'resampled_samples' in report
    assert report['original_samples'] == len(X)
    assert report['resampled_samples'] == len(X_res)
    
    # Check that checkpoint was created
    checkpoint_file = Path(temp_checkpoint_dir) / "smote_enn_checkpoint.pkl"
    assert checkpoint_file.exists()


def test_checkpointed_optuna_trainer(sample_data, temp_checkpoint_dir):
    """Test checkpointed Optuna trainer."""
    X, y = sample_data
    
    # Convert to numpy arrays
    X_array = X.values
    y_array = y.values
    
    trainer = CheckpointedOptunaTrainer(
        X_array, y_array,
        checkpoint_dir=temp_checkpoint_dir,
        checkpoint_interval=2,
        max_memory_gb=4.0
    )
    
    # Run a few trials manually
    import optuna
    study = optuna.create_study(direction='maximize')
    
    for i in range(5):
        trial = study.ask()
        score = trainer.objective(trial)
        study.tell(trial, score)
    
    # Check that checkpoints were created
    checkpoint_files = list(Path(temp_checkpoint_dir).glob("checkpoint_trial_*.json"))
    assert len(checkpoint_files) > 0
    
    # Test loading checkpoint
    loaded_checkpoint = trainer.load_latest_checkpoint()
    assert loaded_checkpoint is not None
    assert 'trial_counter' in loaded_checkpoint
    assert 'best_score' in loaded_checkpoint


def test_chunked_smote_checkpoint_recovery(sample_data, temp_checkpoint_dir):
    """Test that SMOTE+ENN can recover from checkpoint."""
    X, y = sample_data
    
    # First run - creates checkpoint
    X_res1, y_res1, report1 = apply_smote_enn_chunked(
        X, y,
        chunk_size=500,
        sampling_strategy=0.7,
        random_state=42,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Second run - should load from checkpoint
    X_res2, y_res2, report2 = apply_smote_enn_chunked(
        X, y,
        chunk_size=500,
        sampling_strategy=0.7,
        random_state=42,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Results should be identical
    pd.testing.assert_frame_equal(X_res1, X_res2)
    pd.testing.assert_series_equal(y_res1, y_res2)
    assert report1 == report2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
