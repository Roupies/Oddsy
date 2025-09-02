#!/usr/bin/env python3
"""
TRAIN v2.0 XG-ENHANCED MODEL
Compare xG features against v1.5 baseline performance
Target: Beat 52.11% baseline with Expected Goals intelligence
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def load_enhanced_dataset():
    """
    Load the xG-enhanced dataset from Phase 2.0.4
    """
    logger = setup_logging()
    logger.info("=== LOADING XG-ENHANCED DATASET ===")
    
    # Find most recent enhanced dataset
    data_dir = 'data/processed'
    enhanced_files = [f for f in os.listdir(data_dir) if f.startswith('premier_league_xg_enhanced')]
    
    if not enhanced_files:
        logger.error("❌ No enhanced dataset found")
        return None
    
    latest_file = sorted(enhanced_files)[-1]
    filepath = os.path.join(data_dir, latest_file)
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info(f"✅ Enhanced dataset loaded: {filepath}")
    logger.info(f"📊 Shape: {df.shape}")
    logger.info(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

def prepare_xg_features(df):
    """
    Prepare xG feature sets for comparison
    """
    logger = setup_logging()
    logger.info("=== PREPARING XG FEATURE SETS ===")
    
    # Current v1.5 baseline features (shots-based)
    v15_features = [
        'elo_diff_normalized',
        'form_diff_normalized', 
        'h2h_score',
        'matchday_normalized',
        'shots_diff_normalized',
        'corners_diff_normalized'
    ]
    
    # New xG-based features (replace shots with xG intelligence)
    xg_core_features = [
        'HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG',
        'Home_GoalsVsXG', 'Away_GoalsVsXG'
    ]
    
    xg_advanced_features = [
        'xg_form_diff_normalized',
        'xga_form_diff_normalized', 
        'xg_efficiency_diff_normalized'
    ]
    
    # v2.0 Feature Set 1: Replace shots with basic xG
    v20_basic_features = [
        'elo_diff_normalized',
        'form_diff_normalized',
        'h2h_score', 
        'matchday_normalized',
        'XG_Diff',  # Replace shots_diff with xG_Diff
        'corners_diff_normalized'
    ]
    
    # v2.0 Feature Set 2: Full xG integration
    v20_full_features = [
        'elo_diff_normalized',
        'form_diff_normalized',
        'h2h_score',
        'matchday_normalized', 
        'XG_Diff',
        'corners_diff_normalized',
        'xg_form_diff_normalized',
        'xga_form_diff_normalized',
        'xg_efficiency_diff_normalized'
    ]
    
    # v2.0 Feature Set 3: xG-only approach
    v20_xg_only_features = [
        'HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG',
        'xg_form_diff_normalized',
        'xga_form_diff_normalized',
        'xg_efficiency_diff_normalized',
        'Home_GoalsVsXG', 'Away_GoalsVsXG'
    ]
    
    feature_sets = {
        'v1.5_baseline': v15_features,
        'v2.0_basic': v20_basic_features, 
        'v2.0_full': v20_full_features,
        'v2.0_xg_only': v20_xg_only_features
    }
    
    # Validate feature availability
    available_features = df.columns.tolist()
    for set_name, features in feature_sets.items():
        missing = [f for f in features if f not in available_features]
        if missing:
            logger.warning(f"⚠️  {set_name} missing features: {missing}")
        else:
            logger.info(f"✅ {set_name}: {len(features)} features ready")
    
    return feature_sets

def prepare_ml_data(df):
    """
    Prepare data for ML training with proper target encoding
    """
    logger = setup_logging()
    logger.info("=== PREPARING ML DATA ===")
    
    # Handle missing values in xG features
    xg_columns = ['HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG', 'Home_GoalsVsXG', 'Away_GoalsVsXG']
    for col in xg_columns:
        if col in df.columns:
            missing_before = df[col].isnull().sum()
            if missing_before > 0:
                # Fill with neutral/average values
                if 'Diff' in col:
                    df[col] = df[col].fillna(0)  # Neutral difference
                elif 'GoalsVsXG' in col:
                    df[col] = df[col].fillna(1.0)  # Average efficiency
                else:
                    df[col] = df[col].fillna(1.5)  # Average xG per team
                logger.info(f"Filled {missing_before} missing values in {col}")
    
    # Encode target variable
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_mapping)
    
    # Validate target encoding
    unique_targets = sorted(y.dropna().unique())
    logger.info(f"✅ Target encoded: {dict(zip(['H', 'D', 'A'], unique_targets))}")
    logger.info(f"Target distribution: {y.value_counts().sort_index().to_dict()}")
    
    return df, y

def evaluate_feature_set(X, y, feature_set_name, n_splits=5):
    """
    Evaluate a feature set using temporal cross-validation
    """
    logger = setup_logging()
    logger.info(f"🔬 EVALUATING {feature_set_name.upper()}")
    
    # Temporal cross-validation (respects chronological order)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Random Forest with balanced class weights
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation scores
    cv_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
    
    # Calculate metrics
    mean_accuracy = cv_scores.mean()
    std_accuracy = cv_scores.std()
    
    logger.info(f"📊 {feature_set_name} Results:")
    logger.info(f"   Cross-validation accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    logger.info(f"   Individual folds: {[f'{score:.3f}' for score in cv_scores]}")
    
    # Baselines comparison
    baselines = {
        'random': 0.333,
        'majority_class': 0.436,  # Approximate home win rate
        'good_target': 0.520,
        'excellent_target': 0.550
    }
    
    logger.info(f"📈 Baseline Comparisons:")
    for baseline_name, baseline_value in baselines.items():
        diff = mean_accuracy - baseline_value
        status = "✅" if diff > 0 else "❌"
        logger.info(f"   vs {baseline_name} ({baseline_value:.1%}): {diff:+.1%} {status}")
    
    return {
        'feature_set': feature_set_name,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'cv_scores': cv_scores.tolist(),
        'features_count': X.shape[1]
    }

def compare_feature_importance(df, feature_sets, target):
    """
    Compare feature importance across different sets
    """
    logger = setup_logging()
    logger.info("=== FEATURE IMPORTANCE ANALYSIS ===")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    
    importance_results = {}
    
    for set_name, features in feature_sets.items():
        if all(f in df.columns for f in features):
            X = df[features].copy()
            
            # Handle any remaining missing values
            X = X.fillna(X.mean())
            
            # Fit model
            rf_model.fit(X, target)
            
            # Get importance
            importance_dict = dict(zip(features, rf_model.feature_importances_))
            importance_results[set_name] = importance_dict
            
            logger.info(f"🎯 {set_name.upper()} Feature Importance:")
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_importance:
                logger.info(f"   {feature}: {importance:.3f}")
    
    return importance_results

def save_v20_results(results, feature_sets):
    """
    Save v2.0 evaluation results
    """
    logger = setup_logging()
    logger.info("=== SAVING V2.0 RESULTS ===")
    
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    
    # Save detailed results
    results_data = {
        'timestamp': timestamp,
        'evaluation_results': results,
        'feature_sets': feature_sets,
        'baseline_comparison': {
            'v15_performance': 0.5211,  # Known baseline
            'target_performance': 0.5500,  # Target for v2.0
            'improvements': {}
        }
    }
    
    # Calculate improvements vs baseline
    baseline_score = 0.5211
    for result in results:
        improvement = result['mean_accuracy'] - baseline_score
        results_data['baseline_comparison']['improvements'][result['feature_set']] = {
            'absolute_improvement': float(improvement),
            'relative_improvement': float(improvement / baseline_score),
            'beats_baseline': bool(improvement > 0)
        }
    
    # Save results
    import json
    results_file = f"evaluation/v20_xg_evaluation_{timestamp}.json"
    os.makedirs('evaluation', exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"✅ Results saved: {results_file}")
    
    # Generate summary report
    best_result = max(results, key=lambda x: x['mean_accuracy'])
    logger.info(f"\n🏆 BEST PERFORMING MODEL:")
    logger.info(f"   Model: {best_result['feature_set']}")
    logger.info(f"   Accuracy: {best_result['mean_accuracy']:.3f} ± {best_result['std_accuracy']:.3f}")
    logger.info(f"   Features: {best_result['features_count']}")
    
    baseline_improvement = best_result['mean_accuracy'] - baseline_score
    logger.info(f"\n📊 vs v1.5 BASELINE (52.11%):")
    logger.info(f"   Improvement: {baseline_improvement:+.3f} ({baseline_improvement/baseline_score:+.1%})")
    
    target_gap = 0.55 - best_result['mean_accuracy']
    logger.info(f"\n🎯 vs TARGET (55.0%):")
    logger.info(f"   Gap: {target_gap:.3f} ({'ACHIEVED' if target_gap <= 0 else 'REMAINING'})")
    
    return results_file, best_result

def main():
    """
    Main v2.0 evaluation pipeline
    """
    logger = setup_logging()
    logger.info("🚀 v2.0 XG-ENHANCED MODEL EVALUATION")
    logger.info("=" * 60)
    logger.info("Comparing xG features vs v1.5 shots-based baseline")
    logger.info("Target: Beat 52.11% baseline, achieve 55%+ accuracy")
    logger.info("=" * 60)
    
    # Load enhanced dataset
    df = load_enhanced_dataset()
    if df is None:
        return False
    
    # Prepare ML data
    df, y = prepare_ml_data(df)
    
    # Prepare feature sets
    feature_sets = prepare_xg_features(df)
    
    # Sort by date for proper temporal validation
    df = df.sort_values('Date').reset_index(drop=True)
    y = y.reindex(df.index)
    
    # Evaluate all feature sets
    results = []
    
    for set_name, features in feature_sets.items():
        if all(f in df.columns for f in features):
            logger.info(f"\n{'='*50}")
            logger.info(f"TESTING {set_name.upper()}")
            logger.info(f"{'='*50}")
            
            X = df[features].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Ensure all features are numeric
            for col in X.columns:
                if X[col].dtype == 'object':
                    logger.warning(f"Converting {col} to numeric")
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            # Evaluate feature set
            result = evaluate_feature_set(X, y, set_name)
            results.append(result)
        else:
            logger.warning(f"⚠️  Skipping {set_name} - missing features")
    
    if not results:
        logger.error("❌ No feature sets could be evaluated")
        return False
    
    # Feature importance analysis
    logger.info(f"\n{'='*60}")
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info(f"{'='*60}")
    
    importance_results = compare_feature_importance(df, feature_sets, y)
    
    # Save results and generate report
    results_file, best_result = save_v20_results(results, feature_sets)
    
    logger.info(f"\n🎉 V2.0 EVALUATION COMPLETE")
    logger.info(f"Results saved: {results_file}")
    logger.info(f"Best model: {best_result['feature_set']} ({best_result['mean_accuracy']:.1%})")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        print(f"v2.0 evaluation: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)