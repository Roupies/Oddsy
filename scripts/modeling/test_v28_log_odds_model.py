#!/usr/bin/env python3
"""
v2.8 Log Odds Model Test
Quick evaluation of new log odds features vs v2.4 baseline.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def select_features(df):
    """Select features for v2.8 model test."""
    
    # v2.4 baseline features (10 core features)
    baseline_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    # New v2.8 log odds features
    log_odds_features = [
        'log_draw_home_ratio_normalized', 'log_draw_away_ratio_normalized',
        'market_balance_score_normalized', 'draw_favorability_normalized',
        'market_efficiency_gap_normalized'
    ]
    
    # Test configurations
    configs = {
        'v2.4_baseline': baseline_features,
        'v2.8_log_odds_only': log_odds_features,
        'v2.8_combined': baseline_features + log_odds_features
    }
    
    return configs

def evaluate_configuration(X, y, feature_names, config_name):
    """Evaluate a feature configuration."""
    
    logger.info(f"Testing {config_name} with {len(feature_names)} features")
    
    # Time series split (respects chronological order)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Random Forest (same as v2.4 baseline)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
    
    # Train final model for detailed analysis
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance for log odds features
    feature_importance = {}
    if len(feature_names) <= 15:  # Only for manageable number of features
        importance_scores = model.feature_importances_
        for name, importance in zip(feature_names, importance_scores):
            if 'log_' in name or 'market_balance' in name or 'draw_favorability' in name or 'market_efficiency' in name:
                feature_importance[name] = importance
    
    results = {
        'config_name': config_name,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_accuracy,
        'n_features': len(feature_names),
        'log_odds_importance': feature_importance
    }
    
    return results

def run_v28_evaluation():
    """Run comprehensive v2.8 evaluation."""
    
    logger.info("🚀 Starting v2.8 Log Odds Model evaluation...")
    
    # Load v2.8 dataset
    df = pd.read_csv('data/processed/v28_log_odds_features_2025_09_06.csv')
    logger.info(f"Loaded dataset: {df.shape}")
    
    # Remove rows with missing values
    df = df.dropna()
    logger.info(f"After cleaning: {df.shape}")
    
    # Prepare target
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_mapping).values
    
    # Get feature configurations
    feature_configs = select_features(df)
    
    # Evaluate each configuration
    results = []
    
    for config_name, feature_list in feature_configs.items():
        # Check if all features exist
        available_features = [f for f in feature_list if f in df.columns]
        missing_features = [f for f in feature_list if f not in df.columns]
        
        if missing_features:
            logger.warning(f"{config_name}: Missing features {missing_features}")
        
        if len(available_features) > 0:
            X = df[available_features].values
            result = evaluate_configuration(X, y, available_features, config_name)
            results.append(result)
        else:
            logger.error(f"No available features for {config_name}")
    
    # Display results
    print("\\n" + "="*80)
    print("🎯 v2.8 LOG ODDS FEATURES EVALUATION RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\\n📊 {result['config_name']}:")
        print(f"   • Features: {result['n_features']}")
        print(f"   • CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        print(f"   • Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"   • vs v2.4 Baseline: {(result['test_accuracy'] - 0.551)*100:+.2f}pp")
        
        if result['log_odds_importance']:
            print("   • Log Odds Feature Importance:")
            for feature, importance in sorted(result['log_odds_importance'].items(), 
                                           key=lambda x: x[1], reverse=True):
                print(f"     - {feature}: {importance:.3f}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['test_accuracy'])
    
    print(f"\\n🏆 BEST CONFIGURATION: {best_result['config_name']}")
    print(f"   • Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"   • Improvement: {(best_result['test_accuracy'] - 0.551)*100:+.2f}pp vs v2.4")
    
    # Save results
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'evaluation/reports/v28_log_odds_evaluation_{timestamp}.json', index=False)
    
    logger.info("✅ v2.8 evaluation complete!")
    
    return results

if __name__ == "__main__":
    results = run_v28_evaluation()