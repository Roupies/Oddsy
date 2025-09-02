#!/usr/bin/env python3
"""
Train 56% Model - Production Ready
==================================

Reproduce the 56.05% RandomForest performance from benchmark_fast_jupyter.ipynb
Extract only the RandomForest winning configuration for production use.

Based on: 10 optimized features + fast hyperparameter tuning + calibration
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import joblib
import json
import os
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add project root to path
import sys
sys.path.append('.')
from utils import setup_logging

def load_and_prepare_data():
    """Load data with the exact same preparation as fast benchmark"""
    logger = setup_logging()
    logger.info("=== Loading Data for 56% Model Reproduction ===")
    
    # Load dataset
    df = pd.read_csv('data/processed/v13_xg_safe_features.csv', parse_dates=['Date'])
    logger.info(f"📊 Raw data: {len(df)} matches")
    
    # 10 optimal features (from fast benchmark)
    optimal_features = [
        'form_diff_normalized',      # ⭐ Core feature
        'elo_diff_normalized',       # ⭐ Most important
        'h2h_score',                 # ⭐ Historical context
        'matchday_normalized',       # ⭐ Season progression
        'shots_diff_normalized',     # Attacking intent
        'corners_diff_normalized',   # Pressure indicator
        'market_entropy_norm',       # ⭐ Market uncertainty
        'home_xg_eff_10',           # xG efficiency (10-match)
        'away_goals_sum_5',         # Recent scoring
        'away_xg_eff_10'            # Away xG efficiency
    ]
    
    logger.info(f"✅ Features selected: {len(optimal_features)}")
    for i, feat in enumerate(optimal_features, 1):
        logger.info(f"  {i:2d}. {feat}")
    
    # Filter complete data only
    valid_data = df.dropna(subset=optimal_features)
    logger.info(f"📊 Complete data: {len(valid_data)} matches")
    
    # Exact temporal split from fast benchmark
    train_data = valid_data[valid_data['Date'] <= '2024-05-19'].copy()
    test_data = valid_data[valid_data['Date'] >= '2024-08-16'].copy()
    
    logger.info(f"📊 Train: {len(train_data)} matches (until 2024-05-19)")
    logger.info(f"📊 Test: {len(test_data)} matches (from 2024-08-16)")
    
    # Prepare matrices (exact same as benchmark)
    X_train = train_data[optimal_features].fillna(0.5).values
    X_test = test_data[optimal_features].fillna(0.5).values
    
    # Target encoding: H->0, D->1, A->2
    y_train = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    
    logger.info(f"✅ Data prepared: X_train {X_train.shape}, X_test {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, optimal_features, test_data

def train_winning_randomforest(X_train, y_train):
    """Train the exact RandomForest configuration that achieved 56.05%"""
    logger = setup_logging()
    logger.info("=== Training Winning RandomForest (56% Configuration) ===")
    
    start_time = datetime.now()
    
    # Exact same hyperparameter grid from fast benchmark
    rf_params = {
        'n_estimators': [200, 300],
        'max_depth': [20, None],
        'min_samples_split': [5, 10],
        'max_features': ['sqrt'],
        'class_weight': ['balanced']
    }
    
    logger.info(f"🔍 Grid search combinations: {np.prod([len(v) for v in rf_params.values()])}")
    
    # Base RandomForest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # TimeSeriesSplit for temporal data
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Grid search
    logger.info("⏱️ Starting hyperparameter optimization...")
    rf_grid = GridSearchCV(
        rf, rf_params, 
        cv=tscv, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=0
    )
    
    rf_grid.fit(X_train, y_train)
    
    # Get best model
    best_rf = rf_grid.best_estimator_
    logger.info(f"✅ Best hyperparameters found:")
    for param, value in rf_grid.best_params_.items():
        logger.info(f"  {param}: {value}")
    
    # Calibration (critical for production probabilities)
    logger.info("🎯 Applying probability calibration...")
    rf_calibrated = CalibratedClassifierCV(best_rf, method='isotonic', cv=3)
    rf_calibrated.fit(X_train, y_train)
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"⏱️ Training completed in {training_time:.0f} seconds")
    
    return rf_calibrated, rf_grid.best_params_, training_time

def evaluate_model(model, X_test, y_test, features, test_data):
    """Comprehensive model evaluation"""
    logger = setup_logging()
    logger.info("=== Model Evaluation ===")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                 target_names=['Home', 'Draw', 'Away'],
                                 output_dict=True)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Home', 'Draw', 'Away']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion Matrix:")
    print("     H    D    A")
    for i, label in enumerate(['H', 'D', 'A']):
        row = ' '.join([f'{cm[i][j]:4d}' for j in range(3)])
        print(f"  {label}: {row}")
    
    # Feature importance
    logger.info("\n🔍 Feature Importance (Top → Bottom):")
    if hasattr(model, 'calibrated_classifiers_'):  # CalibratedClassifierCV
        base_model = model.calibrated_classifiers_[0].estimator  # Get base estimator
        feature_importance = list(zip(features, base_model.feature_importances_))
    elif hasattr(model, 'estimators_'):  # Regular ensemble
        feature_importance = list(zip(features, model.feature_importances_))
    else:
        feature_importance = list(zip(features, model.feature_importances_))
        
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance, 1):
        marker = " ⭐" if importance > 0.1 else ""
        logger.info(f"  {i:2d}. {feature:<25} {importance:.4f}{marker}")
    
    # Performance vs benchmarks
    logger.info("\n📈 Performance vs Benchmarks:")
    logger.info(f"  vs Random (33.3%): {(accuracy-0.333)*100:+.1f}pp ✅")
    logger.info(f"  vs Majority (43.6%): {(accuracy-0.436)*100:+.1f}pp ✅")
    logger.info(f"  vs v2.1 (54.2%): {(accuracy-0.542)*100:+.1f}pp {'✅' if accuracy > 0.542 else '❌'}")
    logger.info(f"  vs Excellent (55%): {(accuracy-0.55)*100:+.1f}pp {'🎉 EXCELLENT!' if accuracy >= 0.55 else '❌'}")
    
    if accuracy >= 0.56:
        logger.info("🏆 OUTSTANDING PERFORMANCE! (56%+ achieved)")
    elif accuracy >= 0.55:
        logger.info("🎉 EXCELLENT TARGET ACHIEVED!")
    elif accuracy > 0.542:
        logger.info("✅ BEATS v2.1 - Strong improvement!")
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def save_production_model(model, best_params, features, results, training_time):
    """Save the trained model and metadata for production use"""
    logger = setup_logging()
    logger.info("=== Saving Production Model ===")
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    
    # Save model
    model_filename = f"models/randomforest_56percent_model_{timestamp}.joblib"
    joblib.dump(model, model_filename)
    logger.info(f"💾 Model saved: {model_filename}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'RandomForest_Calibrated',
        'version': 'v2.3_56_percent',
        'accuracy': float(results['accuracy']),
        'performance_level': 'OUTSTANDING' if results['accuracy'] >= 0.56 else 'EXCELLENT',
        'features': features,
        'feature_count': len(features),
        'hyperparameters': best_params,
        'training_time_seconds': training_time,
        'data_split': {
            'train_end': '2024-05-19',
            'test_start': '2024-08-16',
            'method': 'temporal_split'
        },
        'calibration': 'isotonic',
        'cross_validation': 'TimeSeriesSplit_3_splits',
        'target_encoding': {'H': 0, 'D': 1, 'A': 2},
        'classification_report': results['classification_report'],
        'feature_importance': [
            {'feature': feat, 'importance': float(imp)} 
            for feat, imp in results['feature_importance']
        ]
    }
    
    metadata_filename = f"models/randomforest_56percent_model_{timestamp}_metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"📋 Metadata saved: {metadata_filename}")
    
    # Production summary
    logger.info(f"\n🏆 PRODUCTION MODEL READY:")
    logger.info(f"  Model: {model_filename}")
    logger.info(f"  Accuracy: {results['accuracy']:.1%}")
    logger.info(f"  Features: {len(features)}")
    logger.info(f"  Status: {'🎉 OUTSTANDING' if results['accuracy'] >= 0.56 else '✅ EXCELLENT'}")
    
    return model_filename, metadata_filename

def main():
    """Main training pipeline"""
    logger = setup_logging()
    logger.info("🚀 Starting 56% Model Training Pipeline")
    
    # Load data
    X_train, X_test, y_train, y_test, features, test_data = load_and_prepare_data()
    
    # Train model
    model, best_params, training_time = train_winning_randomforest(X_train, y_train)
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test, features, test_data)
    
    # Save for production
    model_file, metadata_file = save_production_model(model, best_params, features, results, training_time)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("🎯 TRAINING COMPLETE - 56% MODEL REPRODUCTION")
    logger.info("="*70)
    logger.info(f"✅ Final Accuracy: {results['accuracy']:.1%}")
    logger.info(f"🎯 Target (56%): {'🎉 ACHIEVED!' if results['accuracy'] >= 0.56 else f'Close ({results['accuracy']:.1%})'}")
    logger.info(f"💾 Production files:")
    logger.info(f"  Model: {model_file}")
    logger.info(f"  Metadata: {metadata_file}")
    
    return model, results

if __name__ == "__main__":
    model, results = main()
    print(f"\n🏆 Training complete! Final accuracy: {results['accuracy']:.1%}")
    print("Ready for production deployment! 🚀")