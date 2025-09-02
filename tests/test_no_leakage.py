#!/usr/bin/env python3
"""
TEST NO DATA LEAKAGE - Comprehensive validation
Detects data leakage in xG features using multiple methods:
- Correlation analysis with targets
- Temporal integrity validation  
- Train vs test performance gaps
- Feature shift verification
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import setup_logging

# Configuration
SAFE_DATASET_CSV = 'data/processed/v13_xg_safe_features.csv'
CORRELATION_THRESHOLD = 0.80  # Suspicious correlation threshold
TRAIN_TEST_CUTOFF = '2024-01-01'  # Split date for train/test gap analysis

def load_safe_dataset():
    """
    Load the safe dataset for leakage testing
    """
    logger = setup_logging()
    logger.info("📊 LOADING SAFE DATASET FOR LEAKAGE TESTING")
    
    if not os.path.exists(SAFE_DATASET_CSV):
        logger.error(f"❌ Safe dataset not found: {SAFE_DATASET_CSV}")
        logger.info("Run build_xg_safe_features.py first")
        return None
    
    df = pd.read_csv(SAFE_DATASET_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info(f"✅ Safe dataset loaded: {df.shape}")
    logger.info(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

def test_dangerous_features_removed(df):
    """
    Test 1: Verify dangerous features have been removed
    """
    logger = setup_logging()
    logger.info("🔍 TEST 1: DANGEROUS FEATURES REMOVAL")
    
    dangerous_features = [
        'HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG',
        'HomeGoals', 'AwayGoals', 
        'Home_GoalsVsXG', 'Away_GoalsVsXG'
    ]
    
    found_dangerous = []
    
    for feature in dangerous_features:
        if feature in df.columns:
            found_dangerous.append(feature)
    
    if found_dangerous:
        logger.error(f"❌ DANGEROUS FEATURES STILL PRESENT:")
        for feature in found_dangerous:
            logger.error(f"  {feature}")
        return False
    else:
        logger.info("✅ All dangerous features removed")
        return True

def test_correlation_analysis(df):
    """
    Test 2: Check for suspicious correlations with target
    """
    logger = setup_logging()
    logger.info("📈 TEST 2: CORRELATION ANALYSIS")
    
    if 'FullTimeResult' not in df.columns:
        logger.warning("⚠️ No target variable found - skipping correlation test")
        return True
    
    # Encode target for correlation analysis
    target_mapping = {'H': 1, 'D': 0, 'A': -1}
    y_numeric = df['FullTimeResult'].map(target_mapping)
    
    # Find numeric features (potential xG features)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    xg_features = [col for col in numeric_features 
                  if any(keyword in col.lower() for keyword in ['xg', 'roll', 'eff'])]
    
    logger.info(f"Testing {len(xg_features)} xG-related features for suspicious correlations")
    
    suspicious_features = []
    
    for feature in xg_features:
        if feature in df.columns and not df[feature].isnull().all():
            correlation = abs(df[feature].corr(y_numeric))
            
            if correlation > CORRELATION_THRESHOLD:
                suspicious_features.append((feature, correlation))
                logger.warning(f"⚠️ SUSPICIOUS: {feature} correlation = {correlation:.3f}")
            else:
                logger.info(f"✅ {feature}: {correlation:.3f}")
    
    if suspicious_features:
        logger.error(f"❌ {len(suspicious_features)} features with suspicious correlations > {CORRELATION_THRESHOLD}")
        return False
    else:
        logger.info(f"✅ All features have correlations < {CORRELATION_THRESHOLD}")
        return True

def test_temporal_integrity(df):
    """
    Test 3: Verify rolling features have proper NaN patterns in early matches
    """
    logger = setup_logging()
    logger.info("🕐 TEST 3: TEMPORAL INTEGRITY")
    
    # Sort by date
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    # Check first 100 matches (should have many NaN in rolling features)
    early_matches = df_sorted.head(100)
    
    rolling_features = [col for col in df.columns if 'roll' in col and 'normalized' in col]
    
    if not rolling_features:
        logger.info("ℹ️ No rolling features found to test")
        return True
    
    logger.info(f"Testing temporal integrity for {len(rolling_features)} rolling features")
    
    issues_found = []
    
    for feature in rolling_features:
        if feature in early_matches.columns:
            nan_count = early_matches[feature].isnull().sum()
            nan_percentage = (nan_count / 100) * 100
            
            # Early matches should have some NaN (teams haven't played enough games)
            if nan_percentage < 10:  # Less than 10% NaN is suspicious
                issues_found.append((feature, nan_percentage))
                logger.warning(f"⚠️ SUSPICIOUS: {feature} only {nan_percentage:.1f}% NaN in early matches")
            else:
                logger.info(f"✅ {feature}: {nan_percentage:.1f}% NaN in early matches (good)")
    
    if issues_found:
        logger.warning(f"⚠️ {len(issues_found)} features may not be properly shifted")
        return False
    else:
        logger.info("✅ All rolling features show proper temporal patterns")
        return True

def test_train_test_performance_gap(df):
    """
    Test 4: Check for suspicious train/test performance gaps
    """
    logger = setup_logging()
    logger.info("🎯 TEST 4: TRAIN/TEST PERFORMANCE GAP")
    
    if 'FullTimeResult' not in df.columns:
        logger.warning("⚠️ No target variable - skipping train/test gap test")
        return True
    
    # Temporal split
    cutoff_date = pd.to_datetime(TRAIN_TEST_CUTOFF)
    train_mask = df['Date'] < cutoff_date
    test_mask = df['Date'] >= cutoff_date
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    if len(train_df) == 0 or len(test_df) == 0:
        logger.warning(f"⚠️ Insufficient data for train/test split at {TRAIN_TEST_CUTOFF}")
        return True
    
    logger.info(f"Train period: {train_df['Date'].min()} to {train_df['Date'].max()} ({len(train_df)} matches)")
    logger.info(f"Test period: {test_df['Date'].min()} to {test_df['Date'].max()} ({len(test_df)} matches)")
    
    # Prepare features (exclude dangerous columns)
    feature_cols = [col for col in df.columns 
                   if col not in ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult']
                   and df[col].dtype in ['float64', 'int64']]
    
    # Remove any remaining dangerous features
    safe_feature_cols = [col for col in feature_cols 
                        if not any(danger in col for danger in ['HomeGoals', 'AwayGoals', 'HomeXG', 'AwayXG'])]
    
    logger.info(f"Using {len(safe_feature_cols)} features for train/test gap analysis")
    
    # Prepare data
    X_train = train_df[safe_feature_cols].fillna(0.5)
    X_test = test_df[safe_feature_cols].fillna(0.5)
    
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train = train_df['FullTimeResult'].map(target_mapping)
    y_test = test_df['FullTimeResult'].map(target_mapping)
    
    # Train simple model
    rf_model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=8,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate on both sets
    train_pred = rf_model.predict(X_train)
    test_pred = rf_model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    gap = train_accuracy - test_accuracy
    
    logger.info(f"📊 PERFORMANCE ANALYSIS:")
    logger.info(f"  Train accuracy: {train_accuracy:.3f}")
    logger.info(f"  Test accuracy: {test_accuracy:.3f}")
    logger.info(f"  Gap: {gap:.3f}")
    
    # Analyze gap
    if gap > 0.15:  # Large gap suggests overfitting/leakage
        logger.error(f"❌ LARGE PERFORMANCE GAP: {gap:.3f} (>0.15)")
        logger.error("This suggests possible data leakage or severe overfitting")
        return False
    elif gap > 0.08:
        logger.warning(f"⚠️ MODERATE PERFORMANCE GAP: {gap:.3f}")
        logger.warning("Monitor for potential issues")
        return True
    else:
        logger.info(f"✅ ACCEPTABLE PERFORMANCE GAP: {gap:.3f}")
        return True

def test_feature_shift_validation(df):
    """
    Test 5: Verify rolling features are actually shifted
    """
    logger = setup_logging()
    logger.info("📊 TEST 5: FEATURE SHIFT VALIDATION")
    
    # This test checks if rolling features avoid perfect equality with raw values
    # (A basic sanity check - not foolproof but catches obvious mistakes)
    
    rolling_features = [col for col in df.columns if 'roll' in col and not 'normalized' in col]
    
    if not rolling_features:
        logger.info("ℹ️ No raw rolling features found to validate shift")
        return True
    
    logger.info(f"Validating shift for {len(rolling_features)} rolling features")
    
    # Check if any rolling feature is suspiciously constant or has perfect patterns
    issues_found = []
    
    for feature in rolling_features:
        if feature in df.columns:
            # Check variance
            variance = df[feature].var()
            if variance < 1e-10:  # Essentially constant
                issues_found.append((feature, f"constant (var={variance:.2e})"))
            
            # Check for obvious patterns
            non_null_values = df[feature].dropna()
            if len(non_null_values) > 100:
                # Check if too many identical values (suspicious)
                value_counts = non_null_values.value_counts()
                most_common_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
                if most_common_freq > len(non_null_values) * 0.8:  # 80% same value
                    issues_found.append((feature, f"too uniform ({most_common_freq}/{len(non_null_values)} identical)"))
    
    if issues_found:
        logger.warning(f"⚠️ {len(issues_found)} features have suspicious patterns:")
        for feature, issue in issues_found:
            logger.warning(f"  {feature}: {issue}")
        return False
    else:
        logger.info("✅ Rolling features show expected variation patterns")
        return True

def run_all_tests():
    """
    Run all data leakage tests
    """
    logger = setup_logging()
    logger.info("🧪 COMPREHENSIVE DATA LEAKAGE TESTING")
    logger.info("=" * 60)
    
    # Load dataset
    df = load_safe_dataset()
    if df is None:
        return False
    
    # Run tests
    test_results = []
    
    logger.info("\n" + "="*50)
    test_results.append(("Dangerous Features Removal", test_dangerous_features_removed(df)))
    
    logger.info("\n" + "="*50)
    test_results.append(("Correlation Analysis", test_correlation_analysis(df)))
    
    logger.info("\n" + "="*50)
    test_results.append(("Temporal Integrity", test_temporal_integrity(df)))
    
    logger.info("\n" + "="*50)
    test_results.append(("Train/Test Gap", test_train_test_performance_gap(df)))
    
    logger.info("\n" + "="*50)
    test_results.append(("Feature Shift Validation", test_feature_shift_validation(df)))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📋 TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if passed:
            passed_tests += 1
    
    logger.info(f"\n📊 OVERALL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - No data leakage detected!")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} TESTS FAILED - Potential data leakage detected")
        return False

def main():
    """
    Main testing pipeline
    """
    logger = setup_logging()
    logger.info("🛡️ DATA LEAKAGE VALIDATION SUITE")
    logger.info("Testing xG-safe features for temporal integrity and leakage")
    logger.info("=" * 60)
    
    success = run_all_tests()
    
    if success:
        logger.info("\n✅ VALIDATION COMPLETE - Dataset is safe for ML training")
    else:
        logger.error("\n❌ VALIDATION FAILED - Review and fix issues before training")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        print(f"Data leakage testing: {'SUCCESS - No leakage detected' if success else 'FAILED - Potential leakage found'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)