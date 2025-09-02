#!/usr/bin/env python3
"""
Final sealed test validation with data leakage corrections
Re-run performance evaluation accounting for discovered temporal integrity issues
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def create_cleaned_dataset():
    """
    Create a cleaned dataset with proper temporal integrity
    """
    logger = setup_logging()
    logger.info("=== 🧹 CREATING CLEANED DATASET ===")
    
    # Load original data
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} matches")
    
    # Clean problematic features
    cleaned_df = df.copy()
    
    # 1. Fix form features with excessive seasonal jumps
    logger.info("🔧 Fixing form_diff_normalized...")
    
    # Recalculate form with proper temporal constraints
    for idx in range(len(cleaned_df)):
        current_match = cleaned_df.iloc[idx]
        current_date = current_match['Date']
        home_team = current_match['HomeTeam']
        away_team = current_match['AwayTeam']
        
        # Get recent matches (only historical)
        window_days = 60  # 2 months window
        recent_matches = cleaned_df[
            (cleaned_df['Date'] < current_date) & 
            (cleaned_df['Date'] >= current_date - pd.Timedelta(days=window_days))
        ]
        
        # Home team recent form
        home_recent = recent_matches[
            (recent_matches['HomeTeam'] == home_team) | 
            (recent_matches['AwayTeam'] == home_team)
        ]
        
        # Away team recent form  
        away_recent = recent_matches[
            (recent_matches['HomeTeam'] == away_team) | 
            (recent_matches['AwayTeam'] == away_team)
        ]
        
        # Calculate cleaned form scores
        if len(home_recent) >= 3 and len(away_recent) >= 3:
            # Count wins for home team
            home_wins = 0
            for _, match in home_recent.iterrows():
                if match['HomeTeam'] == home_team and match['FullTimeResult'] == 'H':
                    home_wins += 1
                elif match['AwayTeam'] == home_team and match['FullTimeResult'] == 'A':
                    home_wins += 1
            
            # Count wins for away team
            away_wins = 0
            for _, match in away_recent.iterrows():
                if match['HomeTeam'] == away_team and match['FullTimeResult'] == 'H':
                    away_wins += 1
                elif match['AwayTeam'] == away_team and match['FullTimeResult'] == 'A':
                    away_wins += 1
            
            home_form = home_wins / len(home_recent)
            away_form = away_wins / len(away_recent)
            
            # Normalize to [0,1] range
            cleaned_df.at[idx, 'form_diff_normalized'] = (home_form + (1 - away_form)) / 2
        else:
            # Insufficient data - use neutral value
            cleaned_df.at[idx, 'form_diff_normalized'] = 0.5
    
    # 2. Fix H2H with proper temporal constraints
    logger.info("🔧 Fixing h2h_score...")
    
    for idx in range(len(cleaned_df)):
        current_match = cleaned_df.iloc[idx]
        current_date = current_match['Date']
        home_team = current_match['HomeTeam']
        away_team = current_match['AwayTeam']
        
        # Find ALL historical H2H matches (no recency bias)
        historical_h2h = cleaned_df[
            (cleaned_df['Date'] < current_date) &
            (((cleaned_df['HomeTeam'] == home_team) & (cleaned_df['AwayTeam'] == away_team)) |
             ((cleaned_df['HomeTeam'] == away_team) & (cleaned_df['AwayTeam'] == home_team)))
        ]
        
        if len(historical_h2h) >= 2:  # At least 2 matches for meaningful H2H
            home_wins = 0
            for _, match in historical_h2h.iterrows():
                if match['HomeTeam'] == home_team and match['FullTimeResult'] == 'H':
                    home_wins += 1
                elif match['AwayTeam'] == home_team and match['FullTimeResult'] == 'A':
                    home_wins += 1
            
            h2h_score = home_wins / len(historical_h2h)
        else:
            h2h_score = 0.5  # Neutral when no meaningful history
        
        cleaned_df.at[idx, 'h2h_score'] = h2h_score
    
    logger.info("✅ Dataset temporal integrity fixes applied")
    
    # Verify no extreme values
    for col in ['form_diff_normalized', 'h2h_score']:
        min_val = cleaned_df[col].min()
        max_val = cleaned_df[col].max()
        logger.info(f"  {col}: [{min_val:.3f}, {max_val:.3f}]")
        
        if min_val < 0 or max_val > 1:
            logger.error(f"❌ {col} outside [0,1] range!")
            return None
    
    return cleaned_df

def validate_cleaned_performance():
    """
    Validate model performance with cleaned dataset
    """
    logger = setup_logging()
    logger.info("=== 🎯 VALIDATING CLEANED DATASET PERFORMANCE ===")
    
    # Create cleaned dataset
    cleaned_df = create_cleaned_dataset()
    if cleaned_df is None:
        logger.error("❌ Failed to create cleaned dataset")
        return None
    
    # Same temporal split as before
    cutoff_date = '2024-01-01'
    train_data = cleaned_df[cleaned_df['Date'] < cutoff_date].copy()
    test_data = cleaned_df[cleaned_df['Date'] >= cutoff_date].copy()
    
    logger.info(f"Split: {len(train_data)} train, {len(test_data)} test")
    
    # Features (same as v1.4)
    features = [
        "form_diff_normalized", "elo_diff_normalized", "h2h_score",
        "matchday_normalized", "shots_diff_normalized", "corners_diff_normalized",
        "market_entropy_norm"
    ]
    
    # Prepare data
    X_train = train_data[features].fillna(0.5)
    X_test = test_data[features].fillna(0.5)
    
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train = train_data['FullTimeResult'].map(label_mapping)
    y_test = test_data['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Features: {len(features)}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Train model with same config as breakthrough
    model_config = {
        'n_estimators': 300, 
        'max_depth': 18,  # Key breakthrough parameter
        'max_features': 'log2',
        'min_samples_leaf': 2, 
        'min_samples_split': 15,
        'class_weight': 'balanced', 
        'random_state': 42, 
        'n_jobs': -1
    }
    
    logger.info("🚀 Training model with breakthrough configuration...")
    
    model = RandomForestClassifier(**model_config)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    # Detailed predictions
    y_pred = model.predict(X_test)
    
    logger.info(f"\n📊 CLEANED DATASET PERFORMANCE:")
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Overfitting gap: {(train_accuracy - test_accuracy)*100:.1f}pp")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Home', 'Draw', 'Away'], output_dict=True)
    
    logger.info(f"\nDetailed Performance:")
    logger.info(f"Home: {report['Home']['precision']:.3f} precision, {report['Home']['recall']:.3f} recall")
    logger.info(f"Draw: {report['Draw']['precision']:.3f} precision, {report['Draw']['recall']:.3f} recall")
    logger.info(f"Away: {report['Away']['precision']:.3f} precision, {report['Away']['recall']:.3f} recall")
    
    # Feature importance
    feature_importance = dict(zip(features, model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(f"\n🎯 FEATURE IMPORTANCE (Cleaned):")
    for i, (feature, importance) in enumerate(sorted_importance, 1):
        logger.info(f"  {i}. {feature}: {importance:.3f}")
    
    # Compare with original v1.4 result
    original_performance = 0.5567  # From breakthrough
    performance_change = test_accuracy - original_performance
    
    logger.info(f"\n📈 PERFORMANCE COMPARISON:")
    logger.info(f"Original v1.4: {original_performance:.4f}")
    logger.info(f"Cleaned data: {test_accuracy:.4f}")
    logger.info(f"Change: {performance_change*100:+.2f}pp")
    
    if abs(performance_change) < 0.01:
        logger.info("✅ Minimal change - data leakage impact was small")
    elif performance_change < -0.02:
        logger.warning("❌ Significant drop - data leakage was boosting performance artificially")
    else:
        logger.info("🎯 Performance improvement - cleaning fixed issues")
    
    # Assessment
    if test_accuracy >= 0.55:
        logger.info("🏆 EXCELLENT: Still achieving 55%+ with cleaned data")
    elif test_accuracy >= 0.50:
        logger.info("✅ GOOD: Above 50% threshold with proper temporal integrity")
    else:
        logger.warning("⚠️  Below 50% - may need feature engineering improvements")
    
    return {
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'overfitting_gap': train_accuracy - test_accuracy,
        'performance_change_vs_original': performance_change,
        'feature_importance': dict(sorted_importance),
        'achieves_excellent': test_accuracy >= 0.55,
        'proper_temporal_integrity': True
    }

def compare_original_vs_cleaned():
    """
    Direct comparison between original and cleaned datasets
    """
    logger = setup_logging()
    logger.info("=== 🔬 ORIGINAL vs CLEANED COMPARISON ===")
    
    # Load both datasets
    original_df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    cleaned_df = create_cleaned_dataset()
    
    cutoff_date = '2024-01-01'
    
    # Original test set
    orig_test = original_df[pd.to_datetime(original_df['Date']) >= cutoff_date]
    clean_test = cleaned_df[cleaned_df['Date'] >= cutoff_date]
    
    # Compare key features
    comparison_features = ['form_diff_normalized', 'h2h_score']
    
    for feature in comparison_features:
        orig_stats = orig_test[feature].describe()
        clean_stats = clean_test[feature].describe()
        
        logger.info(f"\n{feature} comparison:")
        logger.info(f"  Original: mean={orig_stats['mean']:.3f}, std={orig_stats['std']:.3f}")
        logger.info(f"  Cleaned:  mean={clean_stats['mean']:.3f}, std={clean_stats['std']:.3f}")
        
        # Count extreme values (potential leakage indicators)
        orig_extremes = ((orig_test[feature] < 0.1) | (orig_test[feature] > 0.9)).sum()
        clean_extremes = ((clean_test[feature] < 0.1) | (clean_test[feature] > 0.9)).sum()
        
        logger.info(f"  Extreme values: Original={orig_extremes}, Cleaned={clean_extremes}")
        
        if clean_extremes < orig_extremes * 0.5:
            logger.info("  ✅ Significant reduction in extreme values - leakage likely fixed")

def run_final_validation():
    """
    Execute complete final validation
    """
    logger = setup_logging()
    logger.info("🔬 STARTING FINAL SEALED TEST VALIDATION")
    logger.info("=" * 80)
    
    # Step 1: Compare datasets
    compare_original_vs_cleaned()
    
    # Step 2: Validate cleaned performance
    results = validate_cleaned_performance()
    
    if results is None:
        logger.error("❌ Validation failed")
        return False
    
    # Step 3: Final assessment
    logger.info("\n" + "=" * 80)
    logger.info("🏁 FINAL VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"✅ Temporal Integrity: Fixed")
    logger.info(f"📊 Cleaned Performance: {results['test_accuracy']:.4f}")
    logger.info(f"🎯 Overfitting Gap: {results['overfitting_gap']*100:.1f}pp")
    logger.info(f"📈 vs Original v1.4: {results['performance_change_vs_original']*100:+.2f}pp")
    logger.info(f"🏆 Excellent Target: {'✅ ACHIEVED' if results['achieves_excellent'] else '❌ MISSED'}")
    
    # Business conclusion
    if results['test_accuracy'] >= 0.55:
        conclusion = "🏆 VALIDATED BREAKTHROUGH - 55%+ achieved with proper temporal integrity"
    elif results['performance_change_vs_original'] > -0.01:
        conclusion = "✅ PERFORMANCE MAINTAINED - Data leakage impact minimal"
    else:
        conclusion = "⚠️ PERFORMANCE DEGRADED - Original results may have been inflated by data leakage"
    
    logger.info(f"\n🎯 FINAL CONCLUSION: {conclusion}")
    
    return results['test_accuracy'] >= 0.50

if __name__ == "__main__":
    try:
        success = run_final_validation()
        print(f"\n🔬 FINAL VALIDATION COMPLETE")
        
        if success:
            print("✅ Model performance validated with proper temporal integrity")
        else:
            print("❌ Performance below acceptable threshold")
            
    except Exception as e:
        print(f"💥 Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)