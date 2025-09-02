#!/usr/bin/env python3
"""
AUDIT v2.0 RESULTS - INVESTIGATE SUSPICIOUS 64.3% ACCURACY
Something is wrong - need to find the data leakage or overfitting issue
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def load_and_inspect_data():
    """
    Load data and inspect for obvious issues
    """
    logger = setup_logging()
    logger.info("🔍 AUDIT: LOADING AND INSPECTING DATA")
    
    # Load enhanced dataset
    data_files = [f for f in os.listdir('data/processed') if f.startswith('premier_league_xg_enhanced')]
    latest_file = sorted(data_files)[-1]
    filepath = f"data/processed/{latest_file}"
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info(f"📊 Dataset: {df.shape}")
    logger.info(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Check for obvious data leakage indicators
    logger.info(f"\n🔍 CHECKING FOR DATA LEAKAGE INDICATORS:")
    
    # Check if xG perfectly predicts results
    df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    # Correlation between xG and actual results
    xg_result_correlation = df['XG_Diff'].corr(df['target'])
    logger.info(f"XG_Diff vs Result correlation: {xg_result_correlation:.3f}")
    
    # Check for perfect predictions
    df['xg_prediction'] = np.where(df['XG_Diff'] > 0.5, 0,  # Home win
                                  np.where(df['XG_Diff'] < -0.5, 2, 1))  # Away win, Draw
    
    xg_accuracy = (df['xg_prediction'] == df['target']).mean()
    logger.info(f"Simple xG rule accuracy: {xg_accuracy:.3f}")
    
    if xg_accuracy > 0.6:
        logger.error("❌ SMOKING GUN: xG perfectly predicts results - MAJOR DATA LEAKAGE")
    
    # Check xG feature distributions
    logger.info(f"\n📊 XG FEATURE STATISTICS:")
    xg_features = ['HomeXG', 'AwayXG', 'XG_Diff', 'Home_GoalsVsXG', 'Away_GoalsVsXG']
    
    for feature in xg_features:
        if feature in df.columns:
            logger.info(f"{feature}: mean={df[feature].mean():.3f}, std={df[feature].std():.3f}")
            logger.info(f"  Range: [{df[feature].min():.3f}, {df[feature].max():.3f}]")
    
    return df

def audit_temporal_integrity(df):
    """
    Check if temporal integrity is maintained
    """
    logger = setup_logging()
    logger.info(f"\n🕐 AUDIT: TEMPORAL INTEGRITY")
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Check if any rolling features use future data
    logger.info("Checking rolling feature calculations...")
    
    # Manually calculate xG form for first few matches
    sample_matches = df.head(20)
    
    for idx, match in sample_matches.iterrows():
        date = match['Date']
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Count previous matches for home team
        prev_home = df[(df['Date'] < date) & 
                      ((df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team))]
        
        logger.info(f"Match {idx+1} ({date.strftime('%Y-%m-%d')}): {home_team} vs {away_team}")
        logger.info(f"  Previous matches for {home_team}: {len(prev_home)}")
        logger.info(f"  xg_form_home value: {match['xg_form_home']:.3f}")
        
        if idx >= 10:  # Just check first 10
            break

def audit_cross_validation_method(df):
    """
    Audit the cross-validation method used
    """
    logger = setup_logging()
    logger.info(f"\n🔬 AUDIT: CROSS-VALIDATION METHOD")
    
    # Prepare data exactly as in original script
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_mapping)
    
    # xG-only features
    xg_features = [
        'HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG',
        'xg_form_diff_normalized', 'xga_form_diff_normalized', 
        'xg_efficiency_diff_normalized', 'Home_GoalsVsXG', 'Away_GoalsVsXG'
    ]
    
    X = df[xg_features].copy()
    X = X.fillna(X.mean())
    
    logger.info(f"Features: {xg_features}")
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Sort by date (critical!)
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    X_sorted = X.reindex(df_sorted.index)
    y_sorted = y.reindex(df_sorted.index)
    
    # Manual time series split to see what's happening
    tscv = TimeSeriesSplit(n_splits=5)
    
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
    )
    
    fold_accuracies = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
        X_train, X_test = X_sorted.iloc[train_idx], X_sorted.iloc[test_idx]
        y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]
        
        # Get date ranges
        train_dates = df_sorted.iloc[train_idx]['Date']
        test_dates = df_sorted.iloc[test_idx]['Date']
        
        logger.info(f"\nFold {fold_idx + 1}:")
        logger.info(f"  Train: {train_dates.min()} to {train_dates.max()} ({len(train_idx)} matches)")
        logger.info(f"  Test:  {test_dates.min()} to {test_dates.max()} ({len(test_idx)} matches)")
        
        # Check for temporal leakage
        if train_dates.max() > test_dates.min():
            logger.error("❌ TEMPORAL LEAKAGE: Training data contains future dates!")
        
        # Fit and predict
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
        
        logger.info(f"  Fold {fold_idx + 1} accuracy: {accuracy:.3f}")
        
        # Check training accuracy (overfitting indicator)
        train_pred = rf_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        logger.info(f"  Training accuracy: {train_accuracy:.3f}")
        
        if train_accuracy > 0.9:
            logger.warning("⚠️ POSSIBLE OVERFITTING: Training accuracy > 90%")
    
    overall_accuracy = np.mean(fold_accuracies)
    logger.info(f"\n📊 Overall CV accuracy: {overall_accuracy:.3f}")
    
    return fold_accuracies

def audit_feature_leakage(df):
    """
    Check if xG features contain future information
    """
    logger = setup_logging()
    logger.info(f"\n🕵️ AUDIT: FEATURE LEAKAGE INVESTIGATION")
    
    # The smoking gun test: Do xG values perfectly correlate with match results?
    logger.info("Testing xG vs actual goals correlation...")
    
    # Check if HomeXG/AwayXG are suspiciously close to actual goals
    df['HomeGoals'] = df.get('HomeGoals', 0)  # Might not exist
    df['AwayGoals'] = df.get('AwayGoals', 0)
    
    if 'HomeGoals' in df.columns and df['HomeGoals'].notna().sum() > 0:
        home_correlation = df['HomeXG'].corr(df['HomeGoals'])
        away_correlation = df['AwayXG'].corr(df['AwayGoals'])
        
        logger.info(f"HomeXG vs HomeGoals correlation: {home_correlation:.3f}")
        logger.info(f"AwayXG vs AwayGoals correlation: {away_correlation:.3f}")
        
        if home_correlation > 0.9 or away_correlation > 0.9:
            logger.error("❌ SMOKING GUN: xG values are too perfectly correlated with actual goals!")
    
    # Check Goals vs xG efficiency features
    logger.info("\nChecking Goals vs xG efficiency features...")
    
    if 'Home_GoalsVsXG' in df.columns:
        efficiency_stats = df['Home_GoalsVsXG'].describe()
        logger.info(f"Home_GoalsVsXG distribution:")
        logger.info(f"  Mean: {efficiency_stats['mean']:.3f}")
        logger.info(f"  Std: {efficiency_stats['std']:.3f}")
        logger.info(f"  Range: [{efficiency_stats['min']:.3f}, {efficiency_stats['max']:.3f}]")
        
        # Check for impossible values
        if efficiency_stats['max'] > 10 or efficiency_stats['min'] < 0:
            logger.warning("⚠️ SUSPICIOUS: Impossible efficiency values detected")

def investigate_data_source():
    """
    Investigate the data source for potential issues
    """
    logger = setup_logging()
    logger.info(f"\n📋 AUDIT: DATA SOURCE INVESTIGATION")
    
    # Check original vs enhanced datasets
    logger.info("Comparing original vs enhanced datasets...")
    
    # Load original dataset for comparison
    original_files = [f for f in os.listdir('data/processed') if 'v13_complete' in f]
    if original_files:
        original_file = sorted(original_files)[-1]
        original_df = pd.read_csv(f"data/processed/{original_file}")
        logger.info(f"Original dataset: {original_df.shape}")
        
        # Check if enhanced dataset has the same matches
        enhanced_files = [f for f in os.listdir('data/processed') if f.startswith('premier_league_xg_enhanced')]
        enhanced_file = sorted(enhanced_files)[-1]
        enhanced_df = pd.read_csv(f"data/processed/{enhanced_file}")
        logger.info(f"Enhanced dataset: {enhanced_df.shape}")
        
        if len(enhanced_df) != len(original_df):
            logger.warning(f"⚠️ SIZE MISMATCH: Original {len(original_df)} vs Enhanced {len(enhanced_df)}")

def main():
    """
    Complete audit of v2.0 results
    """
    logger = setup_logging()
    logger.info("🚨 v2.0 RESULTS AUDIT - INVESTIGATING 64.3% ACCURACY")
    logger.info("=" * 60)
    logger.info("Hypothesis: Data leakage or overfitting causing unrealistic performance")
    logger.info("=" * 60)
    
    # Load and inspect data
    df = load_and_inspect_data()
    
    # Audit temporal integrity
    audit_temporal_integrity(df)
    
    # Audit cross-validation
    cv_accuracies = audit_cross_validation_method(df)
    
    # Audit feature leakage
    audit_feature_leakage(df)
    
    # Investigate data source
    investigate_data_source()
    
    logger.info(f"\n🏁 AUDIT COMPLETE")
    logger.info(f"Suspicion level: {'HIGH' if np.mean(cv_accuracies) > 0.6 else 'MEDIUM'}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        print(f"Audit: {'COMPLETED' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)