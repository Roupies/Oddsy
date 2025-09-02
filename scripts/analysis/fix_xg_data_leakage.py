#!/usr/bin/env python3
"""
FIX XG DATA LEAKAGE - Remove features that use actual match results
Create clean xG dataset and re-test model performance
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def clean_xg_dataset():
    """
    Remove data leakage features from xG dataset
    """
    logger = setup_logging()
    logger.info("🧹 CLEANING XG DATASET - REMOVING DATA LEAKAGE")
    
    # Load corrupted dataset
    df = pd.read_csv('data/processed/premier_league_xg_enhanced_2025_08_31_195341.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info(f"📊 Original dataset: {df.shape}")
    logger.info(f"Original columns: {len(df.columns)}")
    
    # Identify data leakage features (use actual match results)
    leakage_features = [
        'Home_GoalsVsXG',      # Uses HomeGoals (result we want to predict)
        'Away_GoalsVsXG',      # Uses AwayGoals (result we want to predict)
        'xg_efficiency_home',   # Calculated from goals vs xG
        'xg_efficiency_away',   # Calculated from goals vs xG
        'xg_efficiency_diff',   # Difference of efficiency features
        'xg_efficiency_diff_normalized'  # Normalized efficiency diff
    ]
    
    # Remove leakage features
    logger.info(f"🗑️  REMOVING DATA LEAKAGE FEATURES:")
    for feature in leakage_features:
        if feature in df.columns:
            df = df.drop(feature, axis=1)
            logger.info(f"   ❌ Removed: {feature}")
        else:
            logger.info(f"   ℹ️  Not found: {feature}")
    
    logger.info(f"✅ Cleaned dataset: {df.shape}")
    logger.info(f"Cleaned columns: {len(df.columns)}")
    
    # Identify legitimate xG features
    legitimate_xg_features = [
        'HomeXG',              # Raw expected goals (legitimate)
        'AwayXG',              # Raw expected goals (legitimate) 
        'XG_Diff',             # Difference in expected goals (legitimate)
        'Total_XG',            # Total expected goals (legitimate)
        'xg_form_home',        # Rolling xG form (calculated historically)
        'xg_form_away',        # Rolling xG form (calculated historically)
        'xga_form_home',       # Rolling xGA form (calculated historically)
        'xga_form_away',       # Rolling xGA form (calculated historically)
        'xg_form_diff',        # Form differences (legitimate)
        'xga_form_diff',       # Form differences (legitimate) 
        'xg_form_diff_normalized',   # Normalized form diff
        'xga_form_diff_normalized'   # Normalized defensive form diff
    ]
    
    # Verify legitimate features exist
    logger.info(f"\n✅ LEGITIMATE XG FEATURES:")
    available_xg_features = []
    for feature in legitimate_xg_features:
        if feature in df.columns:
            available_xg_features.append(feature)
            logger.info(f"   ✅ Available: {feature}")
        else:
            logger.warning(f"   ❌ Missing: {feature}")
    
    logger.info(f"\nTotal legitimate xG features: {len(available_xg_features)}")
    
    # Save cleaned dataset
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    cleaned_file = f"data/processed/premier_league_xg_cleaned_{timestamp}.csv"
    df.to_csv(cleaned_file, index=False)
    
    logger.info(f"💾 Cleaned dataset saved: {cleaned_file}")
    
    return df, available_xg_features, cleaned_file

def test_cleaned_xg_features(df, xg_features):
    """
    Test model performance with cleaned xG features
    """
    logger = setup_logging()
    logger.info("🧪 TESTING CLEANED XG FEATURES")
    
    # Prepare target
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_mapping)
    
    # Test different feature combinations
    feature_sets = {
        'v1.5_baseline': [
            'elo_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized'
        ],
        'xg_basic': [
            'HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG'
        ],
        'xg_with_traditional': [
            'elo_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'corners_diff_normalized',
            'HomeXG', 'AwayXG', 'XG_Diff'
        ],
        'xg_advanced': [
            'HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG',
            'xg_form_diff_normalized', 'xga_form_diff_normalized'
        ]
    }
    
    # Sort data temporally
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    y_sorted = y.reindex(df_sorted.index)
    
    results = []
    
    for set_name, features in feature_sets.items():
        # Check if all features are available
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.warning(f"⚠️  Skipping {set_name} - missing: {missing_features}")
            continue
        
        logger.info(f"\n🔬 TESTING {set_name.upper()}")
        logger.info(f"Features: {features}")
        
        # Prepare features
        X = df_sorted[features].copy()
        X = X.fillna(X.mean())
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        cv_scores = cross_val_score(rf_model, X, y_sorted, cv=tscv, scoring='accuracy', n_jobs=-1)
        
        mean_accuracy = cv_scores.mean()
        std_accuracy = cv_scores.std()
        
        logger.info(f"📊 Results:")
        logger.info(f"   Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
        logger.info(f"   Folds: {[f'{score:.3f}' for score in cv_scores]}")
        
        # Compare to baselines
        baseline_52 = 0.5211
        improvement = mean_accuracy - baseline_52
        logger.info(f"   vs v1.5 baseline (52.11%): {improvement:+.3f} ({improvement/baseline_52:+.1%})")
        
        if mean_accuracy > 0.55:
            logger.info("   🎯 EXCELLENT: > 55% target achieved!")
        elif mean_accuracy > 0.52:
            logger.info("   ✅ GOOD: > 52% target achieved")
        elif mean_accuracy > 0.436:
            logger.info("   ⚠️  MARGINAL: Beats majority class")
        else:
            logger.info("   ❌ POOR: Below majority class baseline")
        
        results.append({
            'feature_set': set_name,
            'accuracy': mean_accuracy,
            'std': std_accuracy,
            'improvement_vs_baseline': improvement,
            'features': features
        })
    
    return results

def save_corrected_evaluation(results):
    """
    Save corrected evaluation results
    """
    logger = setup_logging()
    logger.info("💾 SAVING CORRECTED EVALUATION RESULTS")
    
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    
    # Find best result
    if results:
        best_result = max(results, key=lambda x: x['accuracy'])
        
        logger.info(f"\n🏆 BEST PERFORMING MODEL (CORRECTED):")
        logger.info(f"   Model: {best_result['feature_set']}")
        logger.info(f"   Accuracy: {best_result['accuracy']:.3f} ± {best_result['std']:.3f}")
        logger.info(f"   Features: {len(best_result['features'])}")
        logger.info(f"   vs v1.5 baseline: {best_result['improvement_vs_baseline']:+.3f}")
        
        # Save results
        import json
        results_file = f"evaluation/v20_corrected_evaluation_{timestamp}.json"
        os.makedirs('evaluation', exist_ok=True)
        
        results_data = {
            'timestamp': timestamp,
            'corrected_results': results,
            'data_leakage_removed': [
                'Home_GoalsVsXG', 'Away_GoalsVsXG', 
                'xg_efficiency_home', 'xg_efficiency_away',
                'xg_efficiency_diff', 'xg_efficiency_diff_normalized'
            ],
            'best_model': best_result
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"✅ Corrected results saved: {results_file}")
        
        return best_result['accuracy']
    
    return None

def main():
    """
    Main pipeline to fix data leakage and retest
    """
    logger = setup_logging()
    logger.info("🚨 FIXING XG DATA LEAKAGE AND RETESTING")
    logger.info("=" * 60)
    logger.info("Removing features that use actual match results")
    logger.info("=" * 60)
    
    # Clean dataset
    df_cleaned, xg_features, cleaned_file = clean_xg_dataset()
    
    # Test with cleaned features
    results = test_cleaned_xg_features(df_cleaned, xg_features)
    
    # Save corrected results
    best_accuracy = save_corrected_evaluation(results)
    
    logger.info(f"\n🎯 CORRECTED EVALUATION COMPLETE")
    if best_accuracy:
        logger.info(f"Best accuracy (without data leakage): {best_accuracy:.1%}")
        
        # Reality check
        if best_accuracy < 0.60:
            logger.info("✅ REALISTIC: Performance in expected range")
        else:
            logger.warning("⚠️  STILL SUSPICIOUS: > 60% may indicate remaining issues")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        print(f"Data leakage fix: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)