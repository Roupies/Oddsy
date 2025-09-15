#!/usr/bin/env python3
"""
v2.9 Context Model Test
Quick evaluation of new context features (congestion + travel) vs v2.4 baseline.
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
    """Select features for v2.9 model test."""
    
    # v2.4 baseline features (10 core features)
    baseline_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    # New v2.9 context features  
    context_features = [
        'days_since_last_home_normalized', 'days_since_last_away_normalized',
        'fixture_congestion_diff_normalized', 'matches_in_7d_home_normalized',
        'matches_in_7d_away_normalized', 'travel_distance_km_normalized',
        'is_midweek', 'travel_fatigue_factor'
    ]
    
    # Test configurations
    configs = {
        'v2.4_baseline': baseline_features,
        'v2.9_context_only': context_features,
        'v2.9_combined': baseline_features + context_features
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
    
    # Feature importance for context features
    feature_importance = {}
    if len(feature_names) <= 18:  # Only for manageable number of features
        importance_scores = model.feature_importances_
        for name, importance in zip(feature_names, importance_scores):
            if any(x in name for x in ['congestion', 'travel', 'midweek', 'days_since', 'matches_in_7d']):
                feature_importance[name] = importance
    
    results = {
        'config_name': config_name,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_accuracy,
        'n_features': len(feature_names),
        'context_importance': feature_importance
    }
    
    return results

def run_v29_evaluation():
    """Run comprehensive v2.9 evaluation."""
    
    logger.info("🚀 Starting v2.9 Context Model evaluation...")
    
    # Load v2.9 dataset
    df = pd.read_csv('data/processed/v29_context_features_2025_09_06.csv')
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
    print("🎯 v2.9 CONTEXT FEATURES EVALUATION RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\\n📊 {result['config_name']}:")
        print(f"   • Features: {result['n_features']}")
        print(f"   • CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        print(f"   • Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"   • vs v2.4 Baseline: {(result['test_accuracy'] - 0.556)*100:+.2f}pp")
        
        if result['context_importance']:
            print("   • Context Feature Importance:")
            for feature, importance in sorted(result['context_importance'].items(), 
                                           key=lambda x: x[1], reverse=True):
                print(f"     - {feature}: {importance:.3f}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['test_accuracy'])
    
    print(f"\\n🏆 BEST CONFIGURATION: {best_result['config_name']}")
    print(f"   • Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"   • Improvement: {(best_result['test_accuracy'] - 0.556)*100:+.2f}pp vs v2.4")
    
    # Context feature analysis
    print(f"\\n🔍 CONTEXT FEATURES ANALYSIS:")
    context_cols = [col for col in df.columns if any(x in col for x in 
                   ['congestion', 'travel', 'midweek', 'days_since', 'matches_in_7d'])]
    
    for col in context_cols[:8]:  # Show top 8 context features
        print(f"   • {col}: {df[col].mean():.3f} ± {df[col].std():.3f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'evaluation/reports/v29_context_evaluation_{timestamp}.json', index=False)
    
    logger.info("✅ v2.9 evaluation complete!")
    
    return results

if __name__ == "__main__":
    results = run_v29_evaluation()