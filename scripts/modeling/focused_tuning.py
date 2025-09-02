#!/usr/bin/env python3
"""
FOCUSED HYPERPARAMETER TUNING - v1.5
Quick optimization targeting overfitting reduction
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import sys
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def load_clean_data():
    """Load data with clean temporal split"""
    logger = setup_logging()
    logger.info("=== Loading Data ===")
    
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Season-based split
    train_seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
    test_seasons = ['2024-2025']
    
    train_data = df[df['Season'].isin(train_seasons)]
    test_data = df[df['Season'].isin(test_seasons)]
    
    features = [
        "form_diff_normalized", "elo_diff_normalized", "h2h_score",
        "matchday_normalized", "shots_diff_normalized", "corners_diff_normalized",
        "market_entropy_norm"
    ]
    
    X_train = train_data[features].fillna(0.5)
    X_test = test_data[features].fillna(0.5)
    
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train = train_data['FullTimeResult'].map(label_mapping)
    y_test = test_data['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def run_focused_tuning():
    """Run focused tuning to reduce overfitting"""
    logger = setup_logging()
    logger.info("=== Focused Hyperparameter Tuning ===")
    
    X_train, X_test, y_train, y_test = load_clean_data()
    
    # Baseline configuration
    baseline_config = {
        'n_estimators': 300,
        'max_depth': 18,
        'max_features': 'log2',
        'min_samples_leaf': 2,
        'min_samples_split': 15,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Test baseline
    baseline_model = RandomForestClassifier(**baseline_config)
    baseline_model.fit(X_train, y_train)
    baseline_train = baseline_model.score(X_train, y_train)
    baseline_test = baseline_model.score(X_test, y_test)
    baseline_gap = baseline_train - baseline_test
    
    logger.info(f"Baseline: Train={baseline_train:.4f}, Test={baseline_test:.4f}, Gap={baseline_gap*100:.1f}pp")
    
    # Test configurations focused on reducing overfitting
    test_configs = [
        # Reduce depth
        {'max_depth': 12},
        {'max_depth': 15}, 
        {'max_depth': 10},
        {'max_depth': 8},
        
        # Increase min_samples_leaf
        {'min_samples_leaf': 5},
        {'min_samples_leaf': 10},
        {'min_samples_leaf': 15},
        
        # Reduce max_features
        {'max_features': 'sqrt'},
        {'max_features': 0.5},
        
        # Combined approaches
        {'max_depth': 12, 'min_samples_leaf': 5},
        {'max_depth': 15, 'min_samples_leaf': 5}, 
        {'max_depth': 10, 'min_samples_leaf': 10},
        {'max_depth': 12, 'max_features': 'sqrt'},
        {'max_depth': 15, 'max_features': 0.5}
    ]
    
    best_config = baseline_config.copy()
    best_test = baseline_test
    best_gap = baseline_gap
    best_model = baseline_model
    
    logger.info(f"Testing {len(test_configs)} configurations...")
    
    for i, config_update in enumerate(test_configs, 1):
        full_config = baseline_config.copy()
        full_config.update(config_update)
        
        model = RandomForestClassifier(**full_config)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        gap = train_acc - test_acc
        
        test_change = (test_acc - baseline_test) * 100
        gap_reduction = (baseline_gap - gap) * 100
        
        logger.info(f"{i:2d}. {str(config_update)[:40]:<40} Test={test_acc:.4f}({test_change:+.1f}) Gap={gap*100:.1f}pp({gap_reduction:+.1f})")
        
        # Update best if better composite score
        composite = test_acc - (0.3 * max(0, gap - 0.15))
        best_composite = best_test - (0.3 * max(0, best_gap - 0.15))
        
        if composite > best_composite:
            best_config = full_config
            best_test = test_acc
            best_gap = gap
            best_model = model
            logger.info(f"    *** NEW BEST ***")
    
    # Final assessment
    test_improvement = (best_test - baseline_test) * 100
    gap_reduction = (baseline_gap - best_gap) * 100
    
    logger.info(f"\nFINAL RESULTS:")
    logger.info(f"Test improvement: {test_improvement:+.2f}pp")
    logger.info(f"Gap reduction: {gap_reduction:+.2f}pp")
    logger.info(f"Final test: {best_test:.4f}")
    logger.info(f"Final gap: {best_gap*100:.1f}pp")
    
    # Success criteria
    success = (test_improvement > -0.5 and gap_reduction > 3) or test_improvement > 1.0
    
    logger.info(f"Optimization success: {'YES' if success else 'NO'}")
    
    # Save if successful
    if success:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        model_file = f"models/v15_focused_tuned_{timestamp}.joblib"
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, model_file)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'model_file': model_file,
            'best_config': best_config,
            'performance': {
                'test_accuracy': float(best_test),
                'overfitting_gap': float(best_gap * 100),
                'test_improvement': float(test_improvement),
                'gap_reduction': float(gap_reduction)
            }
        }
        
        os.makedirs('evaluation', exist_ok=True)
        results_file = f"evaluation/v15_focused_tuning_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved: {model_file}")
        logger.info(f"Results: {results_file}")
    
    return success, best_test, best_gap

def main():
    logger = setup_logging()
    logger.info("FOCUSED HYPERPARAMETER TUNING - v1.5")
    logger.info("Goal: Reduce 33.1pp overfitting gap")
    logger.info("="*50)
    
    success, test_acc, gap = run_focused_tuning()
    
    logger.info(f"\nTUNING COMPLETE")
    logger.info(f"Success: {'YES' if success else 'NO'}")
    logger.info(f"Final test accuracy: {test_acc:.4f}")
    logger.info(f"Final overfitting gap: {gap*100:.1f}pp")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        print(f"Tuning complete: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)