#!/usr/bin/env python3
"""
v2.11 Draw Features Direct Test
Test draw-specific features with standard RandomForest before cascade implementation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_draw_features():
    """Test v2.11 draw features directly."""
    
    logger.info("🚀 Testing v2.11 draw features...")
    
    # Load v2.11 dataset
    df = pd.read_csv('data/processed/v211_draw_features_2025_09_06.csv')
    df = df.dropna()
    logger.info(f"Dataset shape: {df.shape}")
    
    # Feature sets to compare
    baseline_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    context_features = [
        'days_since_last_home_normalized', 'days_since_last_away_normalized',
        'fixture_congestion_diff_normalized', 'travel_distance_km_normalized',
        'is_midweek', 'travel_fatigue_factor'
    ]
    
    # Top draw features based on correlation
    draw_features = [
        'elo_equilibrium', 'shots_equilibrium', 'team_balance_score',
        'draw_propensity_score_normalized', 'equilibrium_score',
        'rest_equilibrium', 'mutual_streak_interaction', 'home_form_variance_normalized',
        'midseason_draw_factor', 'travel_draw_factor'
    ]
    
    # Test configurations
    configs = {
        'v2.9_baseline': baseline_features + context_features,  # Current best: 56.85%
        'v2.11_draw_only': draw_features,
        'v2.11_combined': baseline_features + context_features + draw_features
    }
    
    # Target variable
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_mapping).values
    
    results = []
    
    for config_name, feature_list in configs.items():
        logger.info(f"Testing {config_name} with {len(feature_list)} features")
        
        # Filter available features
        available_features = [f for f in feature_list if f in df.columns]
        missing_features = [f for f in feature_list if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        if len(available_features) == 0:
            continue
            
        X = df[available_features].values
        
        # Model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
        
        # Train/test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Draw-specific analysis
        draw_mask = (y_test == 1)  # Draw class
        draw_pred_mask = (y_pred == 1)
        
        draw_recall = np.sum((y_test == 1) & (y_pred == 1)) / max(np.sum(y_test == 1), 1)
        draw_precision = np.sum((y_test == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1)
        
        # Feature importance for draw features
        draw_importance = {}
        if len(available_features) <= 25:
            importance_scores = model.feature_importances_
            for name, importance in zip(available_features, importance_scores):
                if any(x in name for x in ['equilibrium', 'balance', 'draw', 'propensity']):
                    draw_importance[name] = importance
        
        result = {
            'config': config_name,
            'features': len(available_features),
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'draw_recall': draw_recall,
            'draw_precision': draw_precision,
            'draw_importance': draw_importance
        }
        
        results.append(result)
    
    # Display results
    print("\\n" + "="*80)
    print("🎯 v2.11 DRAW FEATURES DIRECT TEST RESULTS")
    print("="*80)
    
    baseline_acc = 0.5685  # v2.9 performance
    
    for result in results:
        print(f"\\n📊 {result['config']}:")
        print(f"   • Features: {result['features']}")
        print(f"   • CV Accuracy: {result['cv_accuracy']:.4f} ± {result['cv_std']:.4f}")
        print(f"   • Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"   • vs v2.9 Baseline: {(result['test_accuracy'] - baseline_acc)*100:+.2f}pp")
        print(f"   • Draw Recall: {result['draw_recall']:.3f}")
        print(f"   • Draw Precision: {result['draw_precision']:.3f}")
        
        if result['draw_importance']:
            print("   • Draw Feature Importance:")
            for feature, importance in sorted(result['draw_importance'].items(), 
                                           key=lambda x: x[1], reverse=True)[:5]:
                print(f"     - {feature}: {importance:.3f}")
    
    # Best performer
    best_result = max(results, key=lambda x: x['test_accuracy'])
    
    print(f"\\n🏆 BEST CONFIGURATION: {best_result['config']}")
    print(f"   • Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"   • Improvement: {(best_result['test_accuracy'] - baseline_acc)*100:+.2f}pp")
    print(f"   • Draw Performance: {best_result['draw_recall']:.3f} recall, {best_result['draw_precision']:.3f} precision")
    
    # Decision for cascade implementation
    if best_result['test_accuracy'] > baseline_acc + 0.005:  # >0.5pp improvement
        print("\\n✅ DRAW FEATURES SUCCESSFUL - Ready for cascade implementation!")
    else:
        print("\\n⚠️  Limited improvement - Consider feature refinement before cascade")
    
    return results

if __name__ == "__main__":
    results = test_draw_features()