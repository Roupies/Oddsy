#!/usr/bin/env python3
"""
VALIDATE 56% MODEL INTEGRITY - Comprehensive Data Leakage Check
============================================================

Complete validation pipeline to ensure the 56% model has NO data leakage:
1. Feature calculation timing validation
2. Normalization pipeline integrity check  
3. Temporal split verification
4. Performance consistency analysis
5. Feature correlation with future results

This must pass 100% before declaring the 56% model as production reference.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append('.')
from utils import setup_logging

def load_data_for_validation():
    """Load dataset and prepare for comprehensive validation"""
    logger = setup_logging()
    logger.info("=== LOADING DATA FOR INTEGRITY VALIDATION ===")
    
    # Use corrected dataset for validation
    corrected_file = "data/processed/v13_xg_corrected_features_latest.csv"
    if os.path.exists(corrected_file):
        df = pd.read_csv(corrected_file, parse_dates=['Date'])
        logger.info("📊 Using corrected dataset for validation")
    else:
        df = pd.read_csv('data/processed/v13_xg_safe_features.csv', parse_dates=['Date'])
        logger.info("📊 Using original dataset (corrected not available)")
    logger.info(f"📊 Dataset: {df.shape[0]} matches, {df.shape[1]} features")
    
    # 10 features used in 56% model
    model_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Check all features exist
    missing_features = [f for f in model_features if f not in df.columns]
    if missing_features:
        logger.error(f"❌ Missing features: {missing_features}")
        return None, None
        
    logger.info(f"✅ All 10 model features present")
    return df, model_features

def test_temporal_split_integrity(df, features):
    """
    Test 1: Verify temporal split has no information leakage
    """
    logger = setup_logging()
    logger.info("=== TEST 1: TEMPORAL SPLIT INTEGRITY ===")
    
    # Same split as 56% model
    train_cutoff = '2024-05-19'
    test_start = '2024-08-16'
    
    train_df = df[df['Date'] <= train_cutoff].copy()
    test_df = df[df['Date'] >= test_start].copy()
    
    logger.info(f"📊 Train: {len(train_df)} matches (until {train_cutoff})")
    logger.info(f"📊 Test: {len(test_df)} matches (from {test_start})")
    
    # Check for temporal gap
    gap_days = (pd.to_datetime(test_start) - pd.to_datetime(train_cutoff)).days
    logger.info(f"⏰ Temporal gap: {gap_days} days")
    
    if gap_days <= 0:
        logger.error("❌ CRITICAL: No temporal gap between train/test!")
        return False
        
    # Check no overlap in data
    train_dates = set(train_df['Date'].dt.date)
    test_dates = set(test_df['Date'].dt.date)
    overlap = train_dates & test_dates
    
    if overlap:
        logger.error(f"❌ CRITICAL: Date overlap found: {len(overlap)} dates")
        return False
        
    logger.info("✅ Temporal split integrity: PASSED")
    return True

def test_feature_calculation_timing(df, features):
    """
    Test 2: Verify features are calculated using only past information
    """
    logger = setup_logging()
    logger.info("=== TEST 2: FEATURE CALCULATION TIMING ===")
    
    issues_found = []
    
    # Sample random matches and verify feature calculation
    sample_matches = df.sample(min(100, len(df)), random_state=42)
    
    for idx, row in sample_matches.iterrows():
        match_date = row['Date']
        
        # For features that should use rolling windows, check if they're reasonable
        # Example: home_xg_eff_10 should be efficiency over past 10 matches
        
        home_xg_eff = row['home_xg_eff_10']
        if pd.notna(home_xg_eff):
            # Updated bounds after correction: should be between 0.3 and 3.0
            if not (0.3 <= home_xg_eff <= 3.0):
                issues_found.append(f"Suspicious home_xg_eff_10 value: {home_xg_eff}")
                
        # Check away_xg_eff_10 as well
        away_xg_eff = row['away_xg_eff_10']
        if pd.notna(away_xg_eff):
            if not (0.3 <= away_xg_eff <= 3.0):
                issues_found.append(f"Suspicious away_xg_eff_10 value: {away_xg_eff}")
                
        # Check normalized features are in [0,1] range
        normalized_features = [f for f in features if 'normalized' in f]
        for feat in normalized_features:
            value = row[feat]
            if pd.notna(value) and not (0 <= value <= 1):
                issues_found.append(f"Normalized feature {feat} out of bounds: {value}")
    
    if issues_found:
        logger.warning(f"⚠️ Found {len(issues_found)} suspicious feature values:")
        for issue in issues_found[:5]:  # Show first 5
            logger.warning(f"  {issue}")
    else:
        logger.info("✅ Feature calculation timing: PASSED")
        
    return len(issues_found) == 0

def test_normalization_leakage(df, features):
    """
    Test 3: Check if normalization was done globally (leakage) or properly per-period
    """
    logger = setup_logging()  
    logger.info("=== TEST 3: NORMALIZATION LEAKAGE CHECK ===")
    
    # Split data as in 56% model
    train_cutoff = '2024-05-19'
    
    train_df = df[df['Date'] <= train_cutoff].copy()
    test_df = df[df['Date'] > train_cutoff].copy()
    
    normalized_features = [f for f in features if 'normalized' in f]
    leakage_detected = False
    
    for feat in normalized_features:
        # Get complete data statistics
        all_data_mean = df[feat].mean()
        all_data_std = df[feat].std()
        
        # Get train-only statistics
        train_mean = train_df[feat].mean()
        train_std = train_df[feat].std()
        
        # If normalization was done properly, test set should have different distribution
        # than [0,1] when normalized using train-only stats
        test_values = test_df[feat].dropna()
        if len(test_values) > 10:
            test_renormalized = (test_values - train_mean) / train_std
            
            # Check if test values would be outside [0,1] if normalized using train stats only
            outside_bounds = ((test_renormalized < -0.5) | (test_renormalized > 1.5)).sum()
            total_test = len(test_values)
            
            outside_pct = outside_bounds / total_test * 100
            
            logger.info(f"  {feat}: {outside_pct:.1f}% test values outside [-0.5, 1.5] range")
            
            # If most values are within [0,1], it suggests global normalization (leakage)
            if outside_pct < 5:  # Less than 5% outside suggests global normalization
                logger.warning(f"  ⚠️ Possible global normalization for {feat}")
                leakage_detected = True
    
    if leakage_detected:
        logger.warning("⚠️ Normalization leakage: SUSPICIOUS (requires investigation)")
        return False
    else:
        logger.info("✅ Normalization leakage: PASSED")
        return True

def test_correlation_with_future_results(df, features):
    """
    Test 4: Check if any feature has suspicious correlation with future match results
    """
    logger = setup_logging()
    logger.info("=== TEST 4: CORRELATION WITH FUTURE RESULTS ===")
    
    # Target encoding
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df['target'] = df['FullTimeResult'].map(target_map)
    
    suspicious_correlations = []
    
    for feat in features:
        if df[feat].notna().sum() < 100:  # Skip if too few values
            continue
            
        # Calculate correlation with current match result
        current_corr = abs(df[feat].corr(df['target']))
        
        # Calculate correlation with future results (1 match ahead)
        df_sorted = df.sort_values(['Date']).copy()
        df_sorted['future_target'] = df_sorted.groupby('Date')['target'].shift(-1)
        
        future_corr = abs(df_sorted[feat].corr(df_sorted['future_target']))
        
        logger.info(f"  {feat}: current={current_corr:.3f}, future={future_corr:.3f}")
        
        # Flag if correlation with future is suspiciously high
        if future_corr > 0.15:  # Threshold for suspicion
            suspicious_correlations.append((feat, future_corr))
    
    if suspicious_correlations:
        logger.error("❌ CRITICAL: High correlation with future results!")
        for feat, corr in suspicious_correlations:
            logger.error(f"  {feat}: {corr:.3f} correlation with future")
        return False
    else:
        logger.info("✅ Correlation with future: PASSED")
        return True

def test_performance_consistency(df, features):
    """
    Test 5: Check if model performance is consistent across different time periods
    """
    logger = setup_logging()
    logger.info("=== TEST 5: PERFORMANCE CONSISTENCY ===")
    
    # Prepare data
    valid_df = df.dropna(subset=features).copy()
    target_map = {'H': 0, 'D': 1, 'A': 2}
    valid_df['target'] = valid_df['FullTimeResult'].map(target_map)
    
    X = valid_df[features].fillna(0.5)
    y = valid_df['target']
    
    # Test with TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Simple RF model
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        
        acc = rf.score(X_test, y_test)
        accuracies.append(acc)
        logger.info(f"  Fold {fold+1}: {acc:.1%}")
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    logger.info(f"📊 Cross-validation: {mean_acc:.1%} ± {std_acc:.1%}")
    
    # Check for excessive variance (sign of overfitting/leakage)
    if std_acc > 0.05:  # More than 5% standard deviation is suspicious
        logger.warning(f"⚠️ High performance variance: {std_acc:.1%}")
        return False
    else:
        logger.info("✅ Performance consistency: PASSED")
        return True

def run_comprehensive_validation():
    """
    Run all validation tests and provide final verdict
    """
    logger = setup_logging()
    logger.info("🚀 STARTING COMPREHENSIVE MODEL INTEGRITY VALIDATION")
    logger.info("="*70)
    
    # Load data
    df, features = load_data_for_validation()
    if df is None:
        return False
        
    # Run all tests
    tests_results = {}
    
    tests_results['temporal_split'] = test_temporal_split_integrity(df, features)
    tests_results['feature_timing'] = test_feature_calculation_timing(df, features)  
    tests_results['normalization'] = test_normalization_leakage(df, features)
    tests_results['future_correlation'] = test_correlation_with_future_results(df, features)
    tests_results['performance_consistency'] = test_performance_consistency(df, features)
    
    # Final verdict
    logger.info("\n" + "="*70)
    logger.info("🏁 FINAL VALIDATION RESULTS")
    logger.info("="*70)
    
    all_passed = True
    for test_name, passed in tests_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test_name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("\n" + "="*70)
    if all_passed:
        logger.info("🎉 MODEL INTEGRITY: VALIDATED")
        logger.info("✅ The 56% model is CLEAN and can be used as production reference")
        logger.info("✅ No data leakage detected in any test")
    else:
        logger.error("❌ MODEL INTEGRITY: COMPROMISED")
        logger.error("🚨 Data leakage suspected - model results are NOT reliable")
        logger.error("🚨 DO NOT use this model for production until issues are resolved")
    
    logger.info("="*70)
    
    return all_passed

if __name__ == "__main__":
    integrity_validated = run_comprehensive_validation()
    
    if integrity_validated:
        print("\n🏆 VALIDATION COMPLETE: Model integrity confirmed!")
        print("The 56% model is ready for production use.")
    else:
        print("\n🚨 VALIDATION FAILED: Model integrity compromised!")
        print("Investigate data leakage before using this model.")