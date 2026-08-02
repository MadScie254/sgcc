"""
Advanced Data Augmentation for Electricity Theft Detection
Creates realistic synthetic customers (honest and fraudulent) to boost model accuracy.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()
np.random.seed(42)
Faker.seed(42)


class ElectricityDataGenerator:
    """Generate realistic electricity consumption patterns."""
    
    def __init__(self):
        self.honest_profiles = {
            'residential_low': {'base': 150, 'std': 30, 'peak_factor': 1.3},
            'residential_medium': {'base': 300, 'std': 60, 'peak_factor': 1.4},
            'residential_high': {'base': 500, 'std': 100, 'peak_factor': 1.5},
            'commercial_small': {'base': 800, 'std': 150, 'peak_factor': 1.6},
            'commercial_large': {'base': 1500, 'std': 300, 'peak_factor': 1.8},
        }
        
        self.theft_patterns = {
            'meter_tampering': {'reduction': 0.4, 'variability': 0.1},  # 40% reduction
            'bypass': {'reduction': 0.6, 'variability': 0.15},  # 60% reduction
            'reverse_metering': {'reduction': 0.5, 'variability': 0.2},  # 50% reduction
            'magnetic_interference': {'reduction': 0.3, 'variability': 0.12},  # 30% reduction
        }
    
    def generate_honest_customer(self, profile_type='residential_medium', months=12):
        """Generate honest customer consumption pattern."""
        profile = self.honest_profiles[profile_type]
        base = profile['base']
        std = profile['std']
        peak_factor = profile['peak_factor']
        
        # Generate monthly consumption with seasonal variation
        consumption = []
        for month in range(months):
            # Seasonal effect (higher in summer months 5-8 and winter 11-2)
            season_multiplier = 1.0
            month_of_year = month % 12
            if month_of_year in [5, 6, 7, 8]:  # Summer
                season_multiplier = peak_factor
            elif month_of_year in [11, 0, 1, 2]:  # Winter
                season_multiplier = peak_factor * 0.9
            
            # Add random noise
            monthly_consumption = np.random.normal(base * season_multiplier, std)
            monthly_consumption = max(0, monthly_consumption)  # No negative values
            consumption.append(monthly_consumption)
        
        return {
            'consumption_history': consumption,
            'profile_type': profile_type,
            'is_theft': 0,
            'theft_type': None
        }
    
    def generate_theft_customer(self, base_profile='residential_medium', 
                                theft_type='meter_tampering', 
                                theft_start_month=6, months=12):
        """Generate fraudulent customer with theft pattern."""
        # Start with honest pattern
        honest_data = self.generate_honest_customer(base_profile, months)
        consumption = honest_data['consumption_history'].copy()
        
        # Apply theft from start month onwards
        theft = self.theft_patterns[theft_type]
        reduction = theft['reduction']
        variability = theft['variability']
        
        for i in range(theft_start_month, months):
            # Add variability to avoid perfect detection
            actual_reduction = np.random.normal(reduction, variability)
            actual_reduction = np.clip(actual_reduction, 0.1, 0.9)  # Keep realistic
            consumption[i] = consumption[i] * (1 - actual_reduction)
        
        return {
            'consumption_history': consumption,
            'profile_type': base_profile,
            'is_theft': 1,
            'theft_type': theft_type,
            'theft_start_month': theft_start_month
        }
    
    def calculate_features(self, consumption_history):
        """Calculate features from consumption history."""
        consumption = np.array(consumption_history)
        
        features = {
            'MEAN_MONTHLY_CONSUMPTION': np.mean(consumption),
            'STD_MONTHLY_CONSUMPTION': np.std(consumption),
            'MAX_MONTHLY_CONSUMPTION': np.max(consumption),
            'MIN_MONTHLY_CONSUMPTION': np.min(consumption),
            'MEDIAN_MONTHLY_CONSUMPTION': np.median(consumption),
            
            # Trends
            'CONSUMPTION_TREND': np.polyfit(range(len(consumption)), consumption, 1)[0],
            
            # Variability metrics
            'COEFFICIENT_OF_VARIATION': np.std(consumption) / (np.mean(consumption) + 1e-6),
            
            # Sudden changes
            'MAX_CONSUMPTION_DROP': 0,
            'MONTHS_WITH_ZERO': np.sum(consumption == 0),
            'MONTHS_WITH_LOW_CONSUMPTION': np.sum(consumption < np.mean(consumption) * 0.3),
        }
        
        # Calculate max drop
        if len(consumption) > 1:
            diffs = np.diff(consumption)
            features['MAX_CONSUMPTION_DROP'] = np.min(diffs) if len(diffs) > 0 else 0
        
        # Recent vs historical
        if len(consumption) >= 6:
            recent = consumption[-3:]
            historical = consumption[:-3]
            features['RECENT_VS_HISTORICAL_RATIO'] = np.mean(recent) / (np.mean(historical) + 1e-6)
        else:
            features['RECENT_VS_HISTORICAL_RATIO'] = 1.0
        
        # Quarterly averages
        quarters = len(consumption) // 3
        if quarters > 0:
            q_avgs = [np.mean(consumption[i*3:(i+1)*3]) for i in range(quarters)]
            features['QUARTERLY_STD'] = np.std(q_avgs)
        else:
            features['QUARTERLY_STD'] = 0
        
        return features


def generate_augmented_dataset(original_data_path, output_path, 
                               n_honest=5000, n_theft=3000):
    """
    Generate augmented dataset with synthetic customers.
    
    Args:
        original_data_path: Path to original dataset
        output_path: Path to save augmented dataset
        n_honest: Number of synthetic honest customers
        n_theft: Number of synthetic theft customers
    """
    logger.info(f"Loading original dataset from {original_data_path}")
    
    # Load original data
    try:
        original_df = pd.read_csv(original_data_path)
        logger.info(f"Original dataset shape: {original_df.shape}")
    except Exception as e:
        logger.warning(f"Could not load original data: {e}. Creating from scratch.")
        original_df = pd.DataFrame()
    
    generator = ElectricityDataGenerator()
    synthetic_data = []
    
    logger.info(f"Generating {n_honest} honest customers...")
    for i in range(n_honest):
        # Randomly select profile type
        profile_type = np.random.choice(list(generator.honest_profiles.keys()))
        months = np.random.randint(12, 25)  # 12-24 months of history
        
        customer = generator.generate_honest_customer(profile_type, months)
        features = generator.calculate_features(customer['consumption_history'])
        
        # Add customer metadata
        features.update({
            'CUSTOMER_ID': f'HONEST_{i:06d}',
            'CUSTOMER_TYPE': profile_type,
            'FLAG': 0,  # Not theft
            'IS_SYNTHETIC': 1
        })
        
        synthetic_data.append(features)
        
        if (i + 1) % 1000 == 0:
            logger.info(f"  Generated {i + 1}/{n_honest} honest customers")
    
    logger.info(f"Generating {n_theft} theft customers...")
    for i in range(n_theft):
        # Randomly select profile and theft type
        profile_type = np.random.choice(list(generator.honest_profiles.keys()))
        theft_type = np.random.choice(list(generator.theft_patterns.keys()))
        months = np.random.randint(12, 25)
        theft_start = np.random.randint(3, months - 3)  # Start theft 3+ months in
        
        customer = generator.generate_theft_customer(
            profile_type, theft_type, theft_start, months
        )
        features = generator.calculate_features(customer['consumption_history'])
        
        # Add customer metadata
        features.update({
            'CUSTOMER_ID': f'THEFT_{i:06d}',
            'CUSTOMER_TYPE': profile_type,
            'THEFT_TYPE': theft_type,
            'FLAG': 1,  # Theft
            'IS_SYNTHETIC': 1
        })
        
        synthetic_data.append(features)
        
        if (i + 1) % 500 == 0:
            logger.info(f"  Generated {i + 1}/{n_theft} theft customers")
    
    # Create synthetic DataFrame
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # Combine with original data if available
    if not original_df.empty:
        # Mark original data
        original_df['IS_SYNTHETIC'] = 0
        
        # Align columns - use appropriate defaults instead of None
        all_columns = set(original_df.columns) | set(synthetic_df.columns)
        for col in all_columns:
            if col not in original_df.columns:
                if col == 'FLAG':
                    original_df[col] = 0  # Default to honest
                elif col in ['IS_SYNTHETIC', 'THEFT_TYPE', 'CUSTOMER_TYPE']:
                    original_df[col] = 0 if col == 'IS_SYNTHETIC' else 'unknown'
                else:
                    original_df[col] = 0.0  # Numeric columns default to 0
            if col not in synthetic_df.columns:
                if col == 'FLAG':
                    synthetic_df[col] = 0  # Default to honest (will be overridden)
                else:
                    synthetic_df[col] = 0.0  # Numeric columns default to 0
        
        # Combine
        augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    else:
        augmented_df = synthetic_df
    
    # Shuffle
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"AUGMENTATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Original dataset: {len(original_df) if not original_df.empty else 0} samples")
    logger.info(f"Synthetic honest: {n_honest} samples")
    logger.info(f"Synthetic theft: {n_theft} samples")
    logger.info(f"Total augmented: {len(augmented_df)} samples")
    logger.info(f"Class distribution:")
    logger.info(f"  - Honest (0): {(augmented_df['FLAG'] == 0).sum()} ({(augmented_df['FLAG'] == 0).sum() / len(augmented_df) * 100:.1f}%)")
    logger.info(f"  - Theft (1): {(augmented_df['FLAG'] == 1).sum()} ({(augmented_df['FLAG'] == 1).sum() / len(augmented_df) * 100:.1f}%)")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"{'='*60}\n")
    
    return augmented_df


def create_balanced_mixtures(augmented_df, output_dir='data'):
    """Create different dataset mixtures for experimentation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mixture 1: Balanced 50-50
    logger.info("Creating balanced 50-50 mixture...")
    theft = augmented_df[augmented_df['FLAG'] == 1]
    honest = augmented_df[augmented_df['FLAG'] == 0]
    n_samples = min(len(theft), len(honest))
    
    balanced_df = pd.concat([
        theft.sample(n_samples, random_state=42),
        honest.sample(n_samples, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    balanced_path = output_dir / 'data_balanced_50_50.csv'
    balanced_df.to_csv(balanced_path, index=False)
    logger.info(f"  Saved: {balanced_path} ({len(balanced_df)} samples)")
    
    # Mixture 2: Realistic imbalance (5% theft)
    logger.info("Creating realistic imbalance mixture (5% theft)...")
    n_theft = len(theft)
    n_honest_needed = int(n_theft * 19)  # 5% theft = 1:19 ratio
    
    if n_honest_needed <= len(honest):
        realistic_df = pd.concat([
            theft,
            honest.sample(n_honest_needed, random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        realistic_df = augmented_df
    
    realistic_path = output_dir / 'data_realistic_imbalance.csv'
    realistic_df.to_csv(realistic_path, index=False)
    logger.info(f"  Saved: {realistic_path} ({len(realistic_df)} samples)")
    
    # Mixture 3: Moderate imbalance (20% theft)
    logger.info("Creating moderate imbalance mixture (20% theft)...")
    n_theft = len(theft)
    n_honest_needed = int(n_theft * 4)  # 20% theft = 1:4 ratio
    
    if n_honest_needed <= len(honest):
        moderate_df = pd.concat([
            theft,
            honest.sample(n_honest_needed, random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        moderate_df = augmented_df
    
    moderate_path = output_dir / 'data_moderate_imbalance.csv'
    moderate_df.to_csv(moderate_path, index=False)
    logger.info(f"  Saved: {moderate_path} ({len(moderate_df)} samples)")
    
    logger.info("\nAll mixtures created successfully!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate augmented dataset')
    parser.add_argument('--original', type=str, default='data/data_raw.csv',
                       help='Path to original dataset')
    parser.add_argument('--output', type=str, default='data/data_augmented.csv',
                       help='Path to save augmented dataset')
    parser.add_argument('--n-honest', type=int, default=20000,
                       help='Number of synthetic honest customers (default: 20000)')
    parser.add_argument('--n-theft', type=int, default=10000,
                       help='Number of synthetic theft customers (default: 10000)')
    parser.add_argument('--create-mixtures', action='store_true',
                       help='Create different dataset mixtures')
    
    args = parser.parse_args()
    
    # Generate augmented dataset
    augmented_df = generate_augmented_dataset(
        args.original, 
        args.output, 
        args.n_honest, 
        args.n_theft
    )
    
    # Create mixtures if requested
    if args.create_mixtures:
        create_balanced_mixtures(augmented_df)
