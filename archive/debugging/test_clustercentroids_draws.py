#!/usr/bin/env python3
"""
Test ClusterCentroids for Draw Prediction Improvement
====================================================

Testing if ClusterCentroids can improve draw prediction (currently ~2% recall)
while maintaining or improving overall accuracy (currently 55%).

Hypothesis: By creating better balanced training data, we can achieve:
- Global accuracy: 54.5-55.5% (maintain current level)
- Draw recall: 10-20% (massive improvement from 2%)
- Better F1-macro score overall
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import joblib
import sys
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.under_sampling import ClusterCentroids
from imblearn.pipeline import make_pipeline

# Add project root to path
sys.path.append('.')
from utils import setup_logging

def load_production_data():
    """Load the same corrected dataset as v2.3 production model"""
    logger = setup_logging()
    logger.info("=== LOADING PRODUCTION DATA FOR CLUSTERCENTROIDS TEST ===")
    
    # Use corrected dataset (same as v2.3)
    corrected_file = "data/processed/v13_xg_corrected_features_latest.csv"
    df = pd.read_csv(corrected_file, parse_dates=['Date'])
    logger.info(f"📊 Dataset: {df.shape[0]} matches, {df.shape[1]} features")
    
    # Same 10 features as v2.3 production model
    model_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    logger.info(f"✅ Using same 10 features as v2.3 production model")
    return df, model_features

def prepare_temporal_split(df, features):
    """Same temporal split as production model for fair comparison"""
    logger = setup_logging()
    logger.info("=== PREPARING TEMPORAL SPLIT (SAME AS V2.3) ===")
    
    # Exact same split as v2.3
    train_cutoff = '2024-05-19'
    test_start = '2024-08-16'
    
    valid_data = df.dropna(subset=features)
    train_data = valid_data[valid_data['Date'] <= train_cutoff].copy()
    test_data = valid_data[valid_data['Date'] >= test_start].copy()
    
    logger.info(f"📊 Train: {len(train_data)} matches (until {train_cutoff})")
    logger.info(f"📊 Test: {len(test_data)} matches (from {test_start})")
    
    # Prepare data
    X_train = train_data[features].fillna(0.5).values
    X_test = test_data[features].fillna(0.5).values
    y_train = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    
    # Show current class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_names = ['Home', 'Draw', 'Away']
    logger.info("📊 Original training distribution:")
    for i, (cls, count) in enumerate(zip(unique, counts)):
        pct = count / len(y_train) * 100
        logger.info(f"  {class_names[cls]}: {count} ({pct:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def test_baseline_model(X_train, X_test, y_train, y_test):
    """Test baseline model (no ClusterCentroids) for comparison"""
    logger = setup_logging()
    logger.info("=== BASELINE MODEL (NO CLUSTERCENTROIDS) ===")
    
    # Same config as v2.3 production
    baseline_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    
    accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
    f1_macro_baseline = f1_score(y_test, y_pred_baseline, average='macro')
    f1_weighted_baseline = f1_score(y_test, y_pred_baseline, average='weighted')
    
    logger.info(f"🎯 Baseline Accuracy: {accuracy_baseline:.4f} ({accuracy_baseline*100:.2f}%)")
    logger.info(f"📊 Baseline F1-Macro: {f1_macro_baseline:.4f}")
    logger.info(f"📊 Baseline F1-Weighted: {f1_weighted_baseline:.4f}")
    
    print("\n📊 Baseline Classification Report:")
    print(classification_report(y_test, y_pred_baseline, target_names=['Home', 'Draw', 'Away']))
    
    return {
        'model': baseline_model,
        'accuracy': accuracy_baseline,
        'f1_macro': f1_macro_baseline,
        'f1_weighted': f1_weighted_baseline,
        'y_pred': y_pred_baseline
    }

def test_clustercentroids_conservative(X_train, X_test, y_train, y_test):
    """Test ClusterCentroids with conservative (auto) sampling"""
    logger = setup_logging()
    logger.info("=== CLUSTERCENTROIDS TEST (CONSERVATIVE) ===")
    
    # Conservative approach - let algorithm decide
    cc_conservative = ClusterCentroids(random_state=42, sampling_strategy='auto')
    X_resampled, y_resampled = cc_conservative.fit_resample(X_train, y_train)
    
    # Show new distribution
    unique, counts = np.unique(y_resampled, return_counts=True)
    class_names = ['Home', 'Draw', 'Away']
    logger.info("📊 ClusterCentroids resampled distribution:")
    for i, (cls, count) in enumerate(zip(unique, counts)):
        pct = count / len(y_resampled) * 100
        logger.info(f"  {class_names[cls]}: {count} ({pct:.1f}%)")
    
    # Same RF config
    cc_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    cc_model.fit(X_resampled, y_resampled)
    y_pred_cc = cc_model.predict(X_test)
    
    accuracy_cc = accuracy_score(y_test, y_pred_cc)
    f1_macro_cc = f1_score(y_test, y_pred_cc, average='macro')
    f1_weighted_cc = f1_score(y_test, y_pred_cc, average='weighted')
    
    logger.info(f"🎯 ClusterCentroids Accuracy: {accuracy_cc:.4f} ({accuracy_cc*100:.2f}%)")
    logger.info(f"📊 ClusterCentroids F1-Macro: {f1_macro_cc:.4f}")
    logger.info(f"📊 ClusterCentroids F1-Weighted: {f1_weighted_cc:.4f}")
    
    print("\n📊 ClusterCentroids Classification Report:")
    print(classification_report(y_test, y_pred_cc, target_names=['Home', 'Draw', 'Away']))
    
    return {
        'model': cc_model,
        'accuracy': accuracy_cc,
        'f1_macro': f1_macro_cc,
        'f1_weighted': f1_weighted_cc,
        'y_pred': y_pred_cc,
        'resampled_size': len(y_resampled)
    }

def test_clustercentroids_aggressive(X_train, X_test, y_train, y_test):
    """Test ClusterCentroids with aggressive custom sampling"""
    logger = setup_logging()
    logger.info("=== CLUSTERCENTROIDS TEST (AGGRESSIVE) ===")
    
    # More aggressive - force more balanced classes  
    # Original: 831H, 429D, 627A -> Target: reduce H&A, keep D at max
    sampling_strategy = {0: 500, 1: 429, 2: 500}  # H, D, A (can't increase minority class)
    
    cc_aggressive = ClusterCentroids(random_state=42, sampling_strategy=sampling_strategy)
    X_resampled_agg, y_resampled_agg = cc_aggressive.fit_resample(X_train, y_train)
    
    # Show new distribution
    unique, counts = np.unique(y_resampled_agg, return_counts=True)
    class_names = ['Home', 'Draw', 'Away']
    logger.info("📊 Aggressive ClusterCentroids distribution:")
    for i, (cls, count) in enumerate(zip(unique, counts)):
        pct = count / len(y_resampled_agg) * 100
        logger.info(f"  {class_names[cls]}: {count} ({pct:.1f}%)")
    
    # Same RF config
    cc_agg_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    cc_agg_model.fit(X_resampled_agg, y_resampled_agg)
    y_pred_agg = cc_agg_model.predict(X_test)
    
    accuracy_agg = accuracy_score(y_test, y_pred_agg)
    f1_macro_agg = f1_score(y_test, y_pred_agg, average='macro')
    f1_weighted_agg = f1_score(y_test, y_pred_agg, average='weighted')
    
    logger.info(f"🎯 Aggressive CC Accuracy: {accuracy_agg:.4f} ({accuracy_agg*100:.2f}%)")
    logger.info(f"📊 Aggressive CC F1-Macro: {f1_macro_agg:.4f}")
    logger.info(f"📊 Aggressive CC F1-Weighted: {f1_weighted_agg:.4f}")
    
    print("\n📊 Aggressive ClusterCentroids Classification Report:")
    print(classification_report(y_test, y_pred_agg, target_names=['Home', 'Draw', 'Away']))
    
    return {
        'model': cc_agg_model,
        'accuracy': accuracy_agg,
        'f1_macro': f1_macro_agg,
        'f1_weighted': f1_weighted_agg,
        'y_pred': y_pred_agg,
        'resampled_size': len(y_resampled_agg)
    }

def compare_results(baseline, conservative, aggressive):
    """Comprehensive comparison of all three approaches"""
    logger = setup_logging()
    logger.info("=== COMPREHENSIVE COMPARISON ===")
    
    results_df = pd.DataFrame({
        'Approach': ['Baseline (v2.3)', 'ClusterCentroids Conservative', 'ClusterCentroids Aggressive'],
        'Accuracy': [baseline['accuracy'], conservative['accuracy'], aggressive['accuracy']],
        'F1_Macro': [baseline['f1_macro'], conservative['f1_macro'], aggressive['f1_macro']],
        'F1_Weighted': [baseline['f1_weighted'], conservative['f1_weighted'], aggressive['f1_weighted']]
    })
    
    print("\n📊 RESULTS COMPARISON:")
    print(results_df.round(4))
    
    # Performance deltas
    logger.info("\n📈 PERFORMANCE CHANGES vs BASELINE:")
    
    cons_acc_delta = (conservative['accuracy'] - baseline['accuracy']) * 100
    cons_f1_delta = (conservative['f1_macro'] - baseline['f1_macro']) * 100
    logger.info(f"Conservative: Accuracy {cons_acc_delta:+.2f}pp, F1-Macro {cons_f1_delta:+.2f}pp")
    
    agg_acc_delta = (aggressive['accuracy'] - baseline['accuracy']) * 100
    agg_f1_delta = (aggressive['f1_macro'] - baseline['f1_macro']) * 100
    logger.info(f"Aggressive: Accuracy {agg_acc_delta:+.2f}pp, F1-Macro {agg_f1_delta:+.2f}pp")
    
    # Best performer analysis
    best_accuracy = max(baseline['accuracy'], conservative['accuracy'], aggressive['accuracy'])
    best_f1_macro = max(baseline['f1_macro'], conservative['f1_macro'], aggressive['f1_macro'])
    
    if best_accuracy == baseline['accuracy']:
        acc_winner = "Baseline"
    elif best_accuracy == conservative['accuracy']:
        acc_winner = "Conservative CC"
    else:
        acc_winner = "Aggressive CC"
        
    if best_f1_macro == baseline['f1_macro']:
        f1_winner = "Baseline"
    elif best_f1_macro == conservative['f1_macro']:
        f1_winner = "Conservative CC"
    else:
        f1_winner = "Aggressive CC"
    
    logger.info(f"\n🏆 WINNERS:")
    logger.info(f"  Best Accuracy: {acc_winner} ({best_accuracy:.1%})")
    logger.info(f"  Best F1-Macro: {f1_winner} ({best_f1_macro:.4f})")
    
    return results_df

def main():
    """Run comprehensive ClusterCentroids test"""
    logger = setup_logging()
    logger.info("🚀 STARTING CLUSTERCENTROIDS EXPERIMENT")
    logger.info("="*70)
    logger.info("Goal: Improve draw prediction while maintaining overall accuracy")
    logger.info("Current baseline: 55% accuracy with ~2% draw recall")
    logger.info("="*70)
    
    # Load data
    df, features = load_production_data()
    X_train, X_test, y_train, y_test = prepare_temporal_split(df, features)
    
    # Test all approaches
    logger.info("\n" + "="*50)
    baseline_results = test_baseline_model(X_train, X_test, y_train, y_test)
    
    logger.info("\n" + "="*50)
    conservative_results = test_clustercentroids_conservative(X_train, X_test, y_train, y_test)
    
    logger.info("\n" + "="*50)
    aggressive_results = test_clustercentroids_aggressive(X_train, X_test, y_train, y_test)
    
    # Final comparison
    logger.info("\n" + "="*70)
    comparison_df = compare_results(baseline_results, conservative_results, aggressive_results)
    
    # Final verdict
    logger.info("\n" + "="*70)
    logger.info("🎯 FINAL VERDICT ON CLUSTERCENTROIDS")
    logger.info("="*70)
    
    best_overall = comparison_df.loc[comparison_df['F1_Macro'].idxmax()]
    logger.info(f"🏆 Best Overall Approach: {best_overall['Approach']}")
    logger.info(f"  Accuracy: {best_overall['Accuracy']:.1%}")
    logger.info(f"  F1-Macro: {best_overall['F1_Macro']:.4f}")
    
    # Recommendation
    if best_overall['Approach'] == 'Baseline (v2.3)':
        recommendation = "KEEP v2.3 baseline - ClusterCentroids didn't improve performance"
    else:
        recommendation = f"UPGRADE to {best_overall['Approach']} - Clear improvement detected"
    
    logger.info(f"📋 Recommendation: {recommendation}")
    logger.info("="*70)
    
    return comparison_df

if __name__ == "__main__":
    results = main()
    print(f"\n🏁 ClusterCentroids experiment complete!")
    print("Check logs for detailed analysis and recommendations.")