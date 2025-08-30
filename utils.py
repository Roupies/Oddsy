import json
import hashlib
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

def setup_logging(config_path: str = "config/config.json") -> logging.Logger:
    """Setup professional logging configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    log_config = config['logging']
    
    # Create logs directory
    log_dir = os.path.dirname(log_config['file'])
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format=log_config['format'],
        handlers=[
            logging.FileHandler(log_config['file']),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return logging.getLogger('oddsy')

def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def generate_data_hash(df: pd.DataFrame) -> str:
    """Generate reproducible hash of dataset for versioning"""
    # Create string representation of data
    data_string = f"{df.shape}_{df.columns.tolist()}_{df.dtypes.to_dict()}"
    
    # Add sample of actual data for uniqueness
    if len(df) > 0:
        sample_data = df.head(10).to_string()
        data_string += sample_data
    
    # Generate hash
    return hashlib.md5(data_string.encode()).hexdigest()[:8]

def generate_timestamp(format_str: Optional[str] = None) -> str:
    """Generate timestamp for versioning"""
    if format_str is None:
        format_str = "%Y_%m_%d_%H%M%S"
    return datetime.now().strftime(format_str)

def version_data(df: pd.DataFrame, base_filename: str, config: Dict[str, Any]) -> str:
    """Save data with version hash and timestamp"""
    if not config['versioning']['enable_data_versioning']:
        # Save without versioning
        filepath = base_filename
        df.to_csv(filepath, index=False)
        return filepath
    
    # Generate version identifiers
    timestamp = generate_timestamp(config['versioning']['timestamp_format'])
    
    if config['versioning']['hash_features']:
        data_hash = generate_data_hash(df)
        version_id = f"{timestamp}_{data_hash}"
    else:
        version_id = timestamp
    
    # Create versioned filename
    base_name = os.path.splitext(base_filename)[0]
    extension = os.path.splitext(base_filename)[1]
    versioned_filename = f"{base_name}_v{version_id}{extension}"
    
    # Save data
    df.to_csv(versioned_filename, index=False)
    
    # Also save latest version (symlink-like)
    df.to_csv(base_filename, index=False)
    
    return versioned_filename

def version_model(model, base_filename: str, config: Dict[str, Any], 
                 metadata: Optional[Dict] = None) -> str:
    """Save model with versioning information"""
    import joblib
    
    if not config['versioning']['enable_model_versioning']:
        joblib.dump(model, base_filename)
        return base_filename
    
    timestamp = generate_timestamp(config['versioning']['timestamp_format'])
    
    # Create versioned filename
    base_name = os.path.splitext(base_filename)[0]
    extension = os.path.splitext(base_filename)[1]
    versioned_filename = f"{base_name}_v{timestamp}{extension}"
    
    # Save model
    joblib.dump(model, versioned_filename)
    
    # Save metadata alongside model
    if metadata:
        metadata_file = f"{base_name}_v{timestamp}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    # Also save as latest
    joblib.dump(model, base_filename)
    
    return versioned_filename

def validate_data_quality(X: pd.DataFrame, y: pd.Series, 
                         expected_features: list, logger: logging.Logger) -> bool:
    """Comprehensive data quality validation"""
    logger.info("Starting data quality validation...")
    
    validation_passed = True
    
    # Check shapes
    if X.shape[0] != len(y):
        logger.error(f"Shape mismatch: X={X.shape[0]}, y={len(y)}")
        validation_passed = False
    
    # Check features
    missing_features = set(expected_features) - set(X.columns)
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        validation_passed = False
    
    extra_features = set(X.columns) - set(expected_features)
    if extra_features:
        logger.warning(f"Extra features (will be ignored): {extra_features}")
    
    # Check for missing values
    missing_X = X.isnull().sum().sum()
    missing_y = y.isnull().sum()
    
    if missing_X > 0:
        logger.error(f"Missing values in features: {missing_X}")
        validation_passed = False
    
    if missing_y > 0:
        logger.error(f"Missing values in target: {missing_y}")
        validation_passed = False
    
    # Check feature normalization
    for col in X.columns:
        if col in expected_features:  # Only check expected features
            min_val, max_val = X[col].min(), X[col].max()
            if min_val < 0 or max_val > 1:
                logger.warning(f"Feature {col} not normalized: [{min_val:.3f}, {max_val:.3f}]")
    
    # Check target values
    unique_targets = set(y.unique())
    expected_targets = {0, 1, 2}  # Home, Draw, Away
    
    if unique_targets != expected_targets:
        logger.error(f"Invalid target values. Expected {expected_targets}, got {unique_targets}")
        validation_passed = False
    
    logger.info(f"Data validation {'PASSED' if validation_passed else 'FAILED'}")
    return validation_passed

def save_metrics_history(metrics: Dict[str, Any], 
                        metrics_file: str = "evaluation/metrics_history.json") -> None:
    """Append metrics to historical tracking file"""
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    # Load existing history
    history = []
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            history = json.load(f)
    
    # Add timestamp to current metrics
    metrics['timestamp'] = datetime.now().isoformat()
    
    # Append new metrics
    history.append(metrics)
    
    # Save updated history
    with open(metrics_file, 'w') as f:
        json.dump(history, f, indent=2, default=str)

def compare_with_baselines(accuracy: float, config: Dict[str, Any], 
                          logger: logging.Logger) -> Dict[str, bool]:
    """Compare model performance with defined baselines"""
    baselines = config['baselines']
    thresholds = baselines['target_thresholds']
    
    results = {
        'beats_random': accuracy > baselines['random_accuracy'],
        'beats_majority': accuracy > thresholds['minimum'],
        'achieves_good': accuracy > thresholds['good'],
        'achieves_excellent': accuracy > thresholds['excellent']
    }
    
    logger.info(f"Baseline comparisons:")
    logger.info(f"  vs Random (33.3%): {'✓' if results['beats_random'] else '✗'}")
    logger.info(f"  vs Majority ({thresholds['minimum']:.1%}): {'✓' if results['beats_majority'] else '✗'}")
    logger.info(f"  Good threshold ({thresholds['good']:.1%}): {'✓' if results['achieves_good'] else '✗'}")
    logger.info(f"  Excellent threshold ({thresholds['excellent']:.1%}): {'✓' if results['achieves_excellent'] else '✗'}")
    
    return results

def get_performance_level(accuracy: float, config: Dict[str, Any]) -> str:
    """Get performance level description"""
    thresholds = config['baselines']['target_thresholds']
    
    if accuracy > thresholds['excellent']:
        return "EXCELLENT"
    elif accuracy > thresholds['good']:
        return "GOOD"
    elif accuracy > thresholds['minimum']:
        return "ACCEPTABLE"
    else:
        return "NEEDS_IMPROVEMENT"