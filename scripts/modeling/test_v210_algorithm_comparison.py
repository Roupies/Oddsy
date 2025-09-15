#!/usr/bin/env python3
"""
v2.10 Algorithm Comparison
Comprehensive test of LightGBM, XGBoost vs RandomForest on v2.9 context features.
Focus: Exploit new context features with gradient boosting algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, log_loss
import lightgbm as lgb
import xgboost as xgb
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_optimized_models():
    """Get optimized model configurations for comparison."""
    
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'type': 'sklearn'
        },
        
        'LightGBM': {
            'model': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'type': 'lgb'
        },
        
        'XGBoost': {
            'model': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            'type': 'xgb'
        },
        
        'LightGBM_Tuned': {
            'model': lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=12,
                learning_rate=0.05,
                num_leaves=50,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.2,
                reg_lambda=0.2,
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'type': 'lgb'
        }
    }
    
    return models

def evaluate_model(model_config, X, y, model_name):
    """Evaluate a single model configuration."""
    
    logger.info(f"Evaluating {model_name}...")
    
    model = model_config['model']
    model_type = model_config['type']
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
    
    # Train/test split for final evaluation
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    
    # Feature importance (top 10)
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        importance_scores = model.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(importance_scores))]
        
        # Get actual feature names if available
        if hasattr(model, 'feature_name_'):
            feature_names = model.feature_name_
        
        # Sort by importance
        importance_pairs = list(zip(feature_names, importance_scores))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        feature_importance = dict(importance_pairs[:10])
    
    results = {
        'model_name': model_name,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_accuracy,
        'log_loss': logloss,
        'feature_importance': feature_importance
    }
    
    return results

def run_algorithm_comparison():
    """Run comprehensive algorithm comparison."""
    
    logger.info("🚀 Starting v2.10 Algorithm Comparison...")
    
    # Load v2.9 dataset with context features
    df = pd.read_csv('data/processed/v29_context_features_2025_09_06.csv')
    logger.info(f"Loaded dataset: {df.shape}")
    
    # Clean data
    df = df.dropna()
    logger.info(f"After cleaning: {df.shape}")
    
    # Best v2.9 feature set (baseline + context)
    feature_columns = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5',
        'days_since_last_home_normalized', 'days_since_last_away_normalized',
        'fixture_congestion_diff_normalized', 'matches_in_7d_home_normalized',
        'matches_in_7d_away_normalized', 'travel_distance_km_normalized',
        'is_midweek', 'travel_fatigue_factor'
    ]
    
    # Check feature availability
    available_features = [f for f in feature_columns if f in df.columns]
    missing_features = [f for f in feature_columns if f not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    logger.info(f"Using {len(available_features)} features for comparison")
    
    # Prepare data
    X = df[available_features].values
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_mapping).values
    
    # Get model configurations
    models = get_optimized_models()
    
    # Evaluate each model
    results = []
    for model_name, model_config in models.items():
        try:
            result = evaluate_model(model_config, X, y, model_name)
            results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
    
    # Display results
    print("\\n" + "="*90)
    print("🎯 v2.10 ALGORITHM COMPARISON RESULTS")
    print("="*90)
    
    # Sort results by test accuracy
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    baseline_accuracy = 0.5685  # v2.9 combined result
    
    for i, result in enumerate(results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
        
        print(f"\\n{rank_emoji} {result['model_name']}:")
        print(f"   • CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        print(f"   • Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"   • Log Loss: {result['log_loss']:.4f}")
        print(f"   • vs v2.9 Baseline: {(result['test_accuracy'] - baseline_accuracy)*100:+.2f}pp")
        
        if result['feature_importance'] and len(result['feature_importance']) > 0:
            print("   • Top 5 Features:")
            top_5_features = list(result['feature_importance'].items())[:5]
            for feature, importance in top_5_features:
                print(f"     - {feature}: {importance:.3f}")
    
    # Best model analysis
    best_result = results[0]
    print(f"\\n🏆 CHAMPION: {best_result['model_name']}")
    print(f"   • Final Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"   • Total Improvement: {(best_result['test_accuracy'] - 0.551)*100:+.2f}pp vs original v2.4")
    print(f"   • Algorithm Improvement: {(best_result['test_accuracy'] - baseline_accuracy)*100:+.2f}pp vs RandomForest")
    
    # Performance milestone check
    if best_result['test_accuracy'] >= 0.60:
        print("\\n🎆 ELITE PERFORMANCE ACHIEVED (≥60%)!")
    elif best_result['test_accuracy'] >= 0.58:
        print("\\n🚀 EXCELLENT PLUS PERFORMANCE (≥58%)!")
    elif best_result['test_accuracy'] >= 0.57:
        print("\\n✨ VERY GOOD PERFORMANCE (≥57%)!")
    
    # Save results
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'evaluation/reports/v210_algorithm_comparison_{timestamp}.json', index=False)
    
    logger.info("✅ v2.10 algorithm comparison complete!")
    
    return results

if __name__ == "__main__":
    results = run_algorithm_comparison()