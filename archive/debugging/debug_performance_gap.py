#!/usr/bin/env python3
"""
Debug Performance Gap: 53.4% vs Expected 55%
===========================================

The ClusterCentroids test showed 53.4% baseline instead of expected 55%.
This script investigates the discrepancy to ensure we're comparing against
the true v2.3 production performance.

Potential causes:
1. Different hyperparameters
2. Different feature preprocessing  
3. Different train/test split
4. Model loading vs training differences
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import sys
import os
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path
sys.path.append('.')
from utils import setup_logging

def load_production_model():
    """Load actual v2.3 production model"""
    logger = setup_logging()
    logger.info("=== LOADING V2.3 PRODUCTION MODEL ===")
    
    model_file = "models/randomforest_corrected_model_2025_09_02_113228.joblib"
    
    if not os.path.exists(model_file):
        logger.error(f"❌ Production model not found: {model_file}")
        return None, None
        
    model = joblib.load(model_file)
    logger.info(f"✅ Loaded production model: {type(model).__name__}")
    
    # Load metadata for exact configuration
    metadata_file = "models/randomforest_corrected_model_2025_09_02_113228_metadata.json"
    import json
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        
    logger.info(f"📊 Model metadata:")
    logger.info(f"  Version: {metadata['version']}")
    logger.info(f"  Claimed accuracy: {metadata['accuracy']:.1%}")
    logger.info(f"  Features: {len(metadata['features'])}")
    
    return model, metadata

def test_production_model_exact_reproduction():
    """Test the exact production model on exact test set"""
    logger = setup_logging()
    logger.info("=== TESTING PRODUCTION MODEL EXACT REPRODUCTION ===")
    
    # Load production model
    model, metadata = load_production_model()
    if model is None:
        return None
        
    # Load same data
    corrected_file = "data/processed/v13_xg_corrected_features_latest.csv"
    df = pd.read_csv(corrected_file, parse_dates=['Date'])
    
    # Exact same features as production
    model_features = metadata['features']
    
    # Exact same split as production
    train_cutoff = metadata['data_split']['train_end']
    test_start = metadata['data_split']['test_start']
    
    valid_data = df.dropna(subset=model_features)
    test_data = valid_data[valid_data['Date'] >= test_start].copy()
    
    logger.info(f"📊 Test data: {len(test_data)} matches")
    
    # Exact same preprocessing
    X_test = test_data[model_features].fillna(0.5).values
    y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    
    # Test production model
    y_pred_production = model.predict(X_test)
    accuracy_production = accuracy_score(y_test, y_pred_production)
    
    logger.info(f"🎯 Production Model Accuracy: {accuracy_production:.4f} ({accuracy_production*100:.2f}%)")
    logger.info(f"📊 Expected: {metadata['accuracy']:.1%}")
    logger.info(f"📊 Difference: {(accuracy_production - metadata['accuracy'])*100:+.2f}pp")
    
    print("\n📊 Production Model Classification Report:")
    print(classification_report(y_test, y_pred_production, target_names=['Home', 'Draw', 'Away']))
    
    return {
        'accuracy': accuracy_production,
        'expected': metadata['accuracy'],
        'matches_expected': abs(accuracy_production - metadata['accuracy']) < 0.001
    }

def test_retrained_model_exact_config():
    """Train new model with exact same config as production"""
    logger = setup_logging()
    logger.info("=== TESTING RETRAINED MODEL (EXACT CONFIG) ===")
    
    # Load production metadata for exact config
    _, metadata = load_production_model()
    if metadata is None:
        return None
        
    # Load data
    corrected_file = "data/processed/v13_xg_corrected_features_latest.csv"
    df = pd.read_csv(corrected_file, parse_dates=['Date'])
    
    # Exact same features
    model_features = metadata['features']
    
    # Exact same split
    train_cutoff = metadata['data_split']['train_end'] 
    test_start = metadata['data_split']['test_start']
    
    valid_data = df.dropna(subset=model_features)
    train_data = valid_data[valid_data['Date'] <= train_cutoff].copy()
    test_data = valid_data[valid_data['Date'] >= test_start].copy()
    
    # Exact same preprocessing
    X_train = train_data[model_features].fillna(0.5).values
    X_test = test_data[model_features].fillna(0.5).values
    y_train = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    
    logger.info(f"📊 Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Exact same hyperparameters
    hyperparams = metadata['hyperparameters']
    logger.info(f"🔧 Using exact hyperparameters: {hyperparams}")
    
    # Train with exact config
    rf_exact = RandomForestClassifier(
        **hyperparams,
        random_state=42,  # Ensure reproducibility
        n_jobs=-1
    )
    
    # Same calibration as production
    rf_calibrated = CalibratedClassifierCV(rf_exact, method='isotonic', cv=3)
    rf_calibrated.fit(X_train, y_train)
    
    # Test
    y_pred_retrained = rf_calibrated.predict(X_test)
    accuracy_retrained = accuracy_score(y_test, y_pred_retrained)
    
    logger.info(f"🎯 Retrained Model Accuracy: {accuracy_retrained:.4f} ({accuracy_retrained*100:.2f}%)")
    logger.info(f"📊 Expected: {metadata['accuracy']:.1%}")
    logger.info(f"📊 Difference: {(accuracy_retrained - metadata['accuracy'])*100:+.2f}pp")
    
    print("\n📊 Retrained Model Classification Report:")
    print(classification_report(y_test, y_pred_retrained, target_names=['Home', 'Draw', 'Away']))
    
    return {
        'accuracy': accuracy_retrained,
        'expected': metadata['accuracy'],
        'matches_expected': abs(accuracy_retrained - metadata['accuracy']) < 0.01
    }

def test_clustercentroids_baseline_comparison():
    """Compare CC test baseline vs production model config"""
    logger = setup_logging()
    logger.info("=== COMPARING CC BASELINE VS PRODUCTION CONFIG ===")
    
    # CC test used these parameters (from the test file)
    cc_config = {
        'n_estimators': 300,
        'max_depth': 20,
        'min_samples_split': 5,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    # Production model metadata
    _, metadata = load_production_model()
    production_config = metadata['hyperparameters']
    
    logger.info("🔍 Configuration Comparison:")
    logger.info("CC Test Config:")
    for param, value in cc_config.items():
        if param != 'random_state' and param != 'n_jobs':
            logger.info(f"  {param}: {value}")
            
    logger.info("Production Config:")
    for param, value in production_config.items():
        logger.info(f"  {param}: {value}")
        
    # Check for differences
    differences = []
    for param in cc_config:
        if param in production_config:
            if cc_config[param] != production_config[param]:
                differences.append(f"{param}: CC={cc_config[param]} vs Prod={production_config[param]}")
                
    if differences:
        logger.warning("⚠️ Configuration differences found:")
        for diff in differences:
            logger.warning(f"  {diff}")
        return False
    else:
        logger.info("✅ Configurations match perfectly")
        return True

def main():
    """Debug the performance gap comprehensively"""
    logger = setup_logging()
    logger.info("🔍 DEBUGGING PERFORMANCE GAP: 53.4% vs 55%")
    logger.info("="*70)
    
    # Test 1: Load and test production model directly
    logger.info("\n" + "="*50)
    production_test = test_production_model_exact_reproduction()
    
    # Test 2: Retrain with exact same configuration
    logger.info("\n" + "="*50) 
    retrained_test = test_retrained_model_exact_config()
    
    # Test 3: Compare configurations
    logger.info("\n" + "="*50)
    config_match = test_clustercentroids_baseline_comparison()
    
    # Final analysis
    logger.info("\n" + "="*70)
    logger.info("🎯 PERFORMANCE GAP ANALYSIS")
    logger.info("="*70)
    
    if production_test and production_test['matches_expected']:
        logger.info("✅ Production model reproduces expected 55% performance")
    else:
        logger.warning("⚠️ Production model doesn't match expected performance")
        
    if retrained_test and retrained_test['matches_expected']:
        logger.info("✅ Retrained model matches expected performance")
    else:
        logger.warning("⚠️ Retrained model doesn't match expected performance")
        
    if config_match:
        logger.info("✅ ClusterCentroids test used correct configuration")
    else:
        logger.warning("⚠️ ClusterCentroids test used different configuration")
        
    # Root cause analysis
    logger.info("\n🔍 ROOT CAUSE ANALYSIS:")
    
    if not config_match:
        logger.info("🎯 LIKELY CAUSE: Configuration mismatch in CC test")
        logger.info("   Solution: Rerun CC test with exact production config")
    elif production_test and not production_test['matches_expected']:
        logger.info("🎯 LIKELY CAUSE: Data or model file issue")
        logger.info("   Solution: Investigate model loading or data preprocessing")
    else:
        logger.info("🎯 LIKELY CAUSE: Random variation or calibration differences")
        logger.info("   Solution: Multiple runs with different random seeds")
        
    logger.info("="*70)

if __name__ == "__main__":
    main()
    print("\n🔍 Performance gap debugging complete!")
    print("Check logs for detailed root cause analysis.")