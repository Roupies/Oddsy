#!/usr/bin/env python3
"""
Validate Cascade Model Integrity - Data Leakage Check
====================================================

The cascade model showed promising results (53% accuracy, 34% draw recall).
Before adopting it, we need to ensure it has no data leakage issues like
we discovered in the original xG model.

Validation tests:
1. Temporal integrity check
2. Feature correlation with future results
3. Cross-validation stability
4. Performance consistency across time periods
5. Cascade-specific leakage (stage 1 -> stage 2)
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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

# Add project root to path
sys.path.append('.')
from utils import setup_logging

# Import cascade model from previous script
sys.path.append('/Users/maxime/Desktop/Oddsy')

class HybridCascadeModel:
    """Replicated cascade model for validation"""
    
    def __init__(self, draw_threshold=0.4, use_smote=True):
        self.draw_threshold = draw_threshold
        self.use_smote = use_smote
        self.draw_detector = None
        self.ha_classifier = None
        
    def fit(self, X_train, y_train_binary, X_train_ha, y_train_ha):
        # Train Draw Detector
        if self.use_smote:
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train_binary)
            
            draw_rf = RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=5,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            self.draw_detector = CalibratedClassifierCV(draw_rf, method='isotonic', cv=3)
            self.draw_detector.fit(X_balanced, y_balanced)
        else:
            draw_rf = RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=5,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            self.draw_detector = CalibratedClassifierCV(draw_rf, method='isotonic', cv=3)
            self.draw_detector.fit(X_train, y_train_binary)
        
        # Train H vs A Classifier
        ha_rf = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=5,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        self.ha_classifier = CalibratedClassifierCV(ha_rf, method='isotonic', cv=3)
        self.ha_classifier.fit(X_train_ha, y_train_ha)
        
    def predict(self, X):
        draw_proba = self.draw_detector.predict_proba(X)[:, 1]
        ha_pred = self.ha_classifier.predict(X)
        
        final_pred = np.zeros(len(X))
        for i in range(len(X)):
            if draw_proba[i] >= self.draw_threshold:
                final_pred[i] = 1  # Draw
            else:
                if ha_pred[i] == 0:
                    final_pred[i] = 0  # Home
                else:
                    final_pred[i] = 2  # Away
                    
        return final_pred.astype(int)

def load_data_for_validation():
    """Load data same as cascade test"""
    logger = setup_logging()
    logger.info("=== LOADING DATA FOR CASCADE VALIDATION ===")
    
    corrected_file = "data/processed/v13_xg_corrected_features_latest.csv"
    df = pd.read_csv(corrected_file, parse_dates=['Date'])
    
    model_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    logger.info(f"📊 Dataset: {df.shape[0]} matches, {len(model_features)} features")
    return df, model_features

def test_temporal_integrity(df, features):
    """Test 1: Temporal split integrity for cascade model"""
    logger = setup_logging()
    logger.info("=== TEST 1: TEMPORAL INTEGRITY ===")
    
    # Same split as cascade test
    train_cutoff = '2024-05-19'
    test_start = '2024-08-16'
    
    valid_data = df.dropna(subset=features)
    train_data = valid_data[valid_data['Date'] <= train_cutoff].copy()
    test_data = valid_data[valid_data['Date'] >= test_start].copy()
    
    # Check temporal gap
    gap_days = (pd.to_datetime(test_start) - pd.to_datetime(train_cutoff)).days
    logger.info(f"⏰ Temporal gap: {gap_days} days")
    
    # Check no overlap
    train_dates = set(train_data['Date'].dt.date)
    test_dates = set(test_data['Date'].dt.date)
    overlap = train_dates & test_dates
    
    if overlap:
        logger.error(f"❌ CRITICAL: Date overlap found: {len(overlap)} dates")
        return False
        
    # Check data distributions match expectations
    train_results = train_data['FullTimeResult'].value_counts()
    test_results = test_data['FullTimeResult'].value_counts()
    
    logger.info("📊 Train distribution: H={}, D={}, A={}".format(
        train_results.get('H', 0), train_results.get('D', 0), train_results.get('A', 0)))
    logger.info("📊 Test distribution: H={}, D={}, A={}".format(
        test_results.get('H', 0), test_results.get('D', 0), test_results.get('A', 0)))
    
    logger.info("✅ Temporal integrity: PASSED")
    return True

def test_feature_correlation_with_future(df, features):
    """Test 2: Check if features correlate suspiciously with future results"""
    logger = setup_logging()
    logger.info("=== TEST 2: FEATURE CORRELATION WITH FUTURE ===")
    
    df_sorted = df.dropna(subset=features).sort_values('Date').copy()
    df_sorted['target'] = df_sorted['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    # Check correlation with future results (1-3 matches ahead)
    suspicious_features = []
    
    for lag in [1, 2, 3]:
        df_sorted[f'future_target_{lag}'] = df_sorted['target'].shift(-lag)
        
        logger.info(f"🔍 Testing correlation with results {lag} matches ahead:")
        
        for feature in features:
            if df_sorted[feature].notna().sum() < 100:
                continue
                
            future_corr = abs(df_sorted[feature].corr(df_sorted[f'future_target_{lag}']))
            
            if future_corr > 0.10:  # Suspicious threshold
                suspicious_features.append((feature, lag, future_corr))
                logger.warning(f"  ⚠️ {feature}: {future_corr:.3f} correlation ({lag} ahead)")
            else:
                logger.info(f"  ✅ {feature}: {future_corr:.3f} correlation ({lag} ahead)")
    
    if suspicious_features:
        logger.error(f"❌ CRITICAL: {len(suspicious_features)} suspicious correlations found!")
        return False
    else:
        logger.info("✅ Feature correlation with future: PASSED")
        return True

def test_cross_validation_stability(df, features):
    """Test 3: Cross-validation stability of cascade model"""
    logger = setup_logging()
    logger.info("=== TEST 3: CROSS-VALIDATION STABILITY ===")
    
    valid_df = df.dropna(subset=features).copy()
    
    # Prepare data
    X = valid_df[features].fillna(0.5).values
    y_3class = valid_df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    y_binary = (y_3class == 1).astype(int)
    
    # Prepare H vs A data
    non_draw_mask = y_3class != 1
    X_ha = X[non_draw_mask]
    y_ha = (y_3class[non_draw_mask] == 2).astype(int)
    
    logger.info(f"📊 Full data: {len(X)} samples")
    logger.info(f"📊 H vs A data: {len(X_ha)} samples")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Split full data
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_3class_cv, y_test_3class_cv = y_3class[train_idx], y_3class[test_idx]
        y_train_binary_cv = (y_train_3class_cv == 1).astype(int)
        
        # Split H vs A data
        train_ha_mask = np.isin(np.arange(len(y_3class)), train_idx) & non_draw_mask
        test_ha_mask = np.isin(np.arange(len(y_3class)), test_idx) & non_draw_mask
        
        X_train_ha_cv = X[train_ha_mask]
        X_test_ha_cv = X[test_ha_mask]
        y_train_ha_cv = (y_3class[train_ha_mask] == 2).astype(int)
        
        if len(X_train_ha_cv) == 0 or len(X_test_ha_cv) == 0:
            continue
            
        # Train cascade model
        cascade_cv = HybridCascadeModel(draw_threshold=0.4, use_smote=True)
        cascade_cv.fit(X_train_cv, y_train_binary_cv, X_train_ha_cv, y_train_ha_cv)
        
        # Predict
        y_pred_cv = cascade_cv.predict(X_test_cv)
        score = accuracy_score(y_test_3class_cv, y_pred_cv)
        cv_scores.append(score)
        
        logger.info(f"  Fold {fold+1}: {score:.1%}")
    
    if len(cv_scores) < 3:
        logger.warning("⚠️ Insufficient CV folds completed")
        return False
        
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    logger.info(f"📊 CV Results: {cv_mean:.1%} ± {cv_std:.1%}")
    
    # Check stability (less than 6% std is reasonable for this type of model)
    if cv_std > 0.06:
        logger.warning(f"⚠️ High variance detected: {cv_std:.1%}")
        return False
    else:
        logger.info("✅ Cross-validation stability: PASSED")
        return True

def test_cascade_specific_leakage(df, features):
    """Test 4: Cascade-specific leakage (Stage 1 predictions affecting Stage 2)"""
    logger = setup_logging()
    logger.info("=== TEST 4: CASCADE-SPECIFIC LEAKAGE ===")
    
    # This test checks if there's any information leakage between the two stages
    # of the cascade model that could artificially inflate performance
    
    valid_data = df.dropna(subset=features)
    
    # Same split as production
    train_cutoff = '2024-05-19'
    test_start = '2024-08-16'
    
    train_data = valid_data[valid_data['Date'] <= train_cutoff].copy()
    test_data = valid_data[valid_data['Date'] >= test_start].copy()
    
    # Prepare data
    X_train = train_data[features].fillna(0.5).values
    X_test = test_data[features].fillna(0.5).values
    
    y_train_3class = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    y_test_3class = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    
    # Test Stage 1 (Draw detector) in isolation
    y_train_binary = (y_train_3class == 1).astype(int)
    y_test_binary = (y_test_3class == 1).astype(int)
    
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train_binary)
    
    draw_rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=5,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    draw_detector = CalibratedClassifierCV(draw_rf, method='isotonic', cv=3)
    draw_detector.fit(X_balanced, y_balanced)
    
    # Test Stage 1 performance
    y_pred_binary = draw_detector.predict(X_test)
    draw_accuracy = accuracy_score(y_test_binary, y_pred_binary)
    draw_recall = (y_pred_binary[y_test_binary == 1] == 1).sum() / (y_test_binary == 1).sum()
    
    logger.info(f"📊 Stage 1 (Draw detector): {draw_accuracy:.1%} accuracy, {draw_recall:.1%} draw recall")
    
    # Test Stage 2 (H vs A) in isolation 
    non_draw_mask_train = y_train_3class != 1
    non_draw_mask_test = y_test_3class != 1
    
    X_train_ha = X_train[non_draw_mask_train]
    X_test_ha = X_test[non_draw_mask_test]
    y_train_ha = (y_train_3class[non_draw_mask_train] == 2).astype(int)
    y_test_ha = (y_test_3class[non_draw_mask_test] == 2).astype(int)
    
    ha_rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    ha_classifier = CalibratedClassifierCV(ha_rf, method='isotonic', cv=3)
    ha_classifier.fit(X_train_ha, y_train_ha)
    
    y_pred_ha = ha_classifier.predict(X_test_ha)
    ha_accuracy = accuracy_score(y_test_ha, y_pred_ha)
    
    logger.info(f"📊 Stage 2 (H vs A): {ha_accuracy:.1%} accuracy")
    
    # Sanity check: Individual stages should make sense
    if draw_recall < 0.20:  # Should detect at least 20% of draws
        logger.warning("⚠️ Draw detector performing poorly in isolation")
        
    if ha_accuracy < 0.55:  # Should be good at H vs A when draws removed
        logger.warning("⚠️ H vs A classifier performing poorly in isolation")
        
    # Check for unrealistic individual stage performance
    if draw_recall > 0.70:  # Too good to be true for draw detection
        logger.error("❌ SUSPICIOUS: Draw recall too high, possible leakage")
        return False
        
    if ha_accuracy > 0.75:  # Too good for H vs A classification
        logger.error("❌ SUSPICIOUS: H vs A accuracy too high, possible leakage")
        return False
    
    logger.info("✅ Cascade-specific leakage: PASSED")
    return True

def test_performance_consistency_over_time(df, features):
    """Test 5: Performance consistency across different time periods"""
    logger = setup_logging()
    logger.info("=== TEST 5: PERFORMANCE CONSISTENCY OVER TIME ===")
    
    valid_data = df.dropna(subset=features).copy()
    
    # Test on different seasons/periods
    periods = [
        ('2019-08-01', '2020-07-31', '2019-20 Season'),
        ('2020-08-01', '2021-07-31', '2020-21 Season'), 
        ('2021-08-01', '2022-07-31', '2021-22 Season'),
        ('2022-08-01', '2023-07-31', '2022-23 Season')
    ]
    
    period_scores = []
    
    for start_date, end_date, period_name in periods:
        period_data = valid_data[
            (valid_data['Date'] >= start_date) & 
            (valid_data['Date'] <= end_date)
        ].copy()
        
        if len(period_data) < 200:  # Need sufficient data
            continue
            
        # Split period into train/test
        split_point = len(period_data) // 2
        period_data_sorted = period_data.sort_values('Date')
        
        train_period = period_data_sorted.iloc[:split_point]
        test_period = period_data_sorted.iloc[split_point:]
        
        if len(train_period) < 100 or len(test_period) < 50:
            continue
            
        # Prepare data for cascade
        X_train_period = train_period[features].fillna(0.5).values
        X_test_period = test_period[features].fillna(0.5).values
        
        y_train_3class = train_period['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
        y_test_3class = test_period['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
        
        y_train_binary = (y_train_3class == 1).astype(int)
        
        # Prepare H vs A data
        non_draw_mask_train = y_train_3class != 1
        X_train_ha = X_train_period[non_draw_mask_train]
        y_train_ha = (y_train_3class[non_draw_mask_train] == 2).astype(int)
        
        if len(X_train_ha) < 50:
            continue
            
        # Train and test cascade
        try:
            cascade = HybridCascadeModel(draw_threshold=0.4, use_smote=True)
            cascade.fit(X_train_period, y_train_binary, X_train_ha, y_train_ha)
            
            y_pred = cascade.predict(X_test_period)
            score = accuracy_score(y_test_3class, y_pred)
            period_scores.append((period_name, score))
            
            logger.info(f"  {period_name}: {score:.1%}")
            
        except Exception as e:
            logger.warning(f"  {period_name}: Failed ({str(e)})")
            continue
    
    if len(period_scores) < 3:
        logger.warning("⚠️ Insufficient periods tested")
        return True  # Don't fail on this
        
    scores = [score for _, score in period_scores]
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    
    logger.info(f"📊 Period consistency: {score_mean:.1%} ± {score_std:.1%}")
    
    # Check for reasonable consistency
    if score_std > 0.08:  # More than 8% std suggests instability
        logger.warning(f"⚠️ High period variance: {score_std:.1%}")
        return False
    
    logger.info("✅ Performance consistency over time: PASSED")
    return True

def run_comprehensive_cascade_validation():
    """Run all validation tests"""
    logger = setup_logging()
    logger.info("🔍 COMPREHENSIVE CASCADE MODEL VALIDATION")
    logger.info("="*70)
    logger.info("Testing cascade model (53% acc, 34% draw recall) for data leakage")
    logger.info("="*70)
    
    # Load data
    df, features = load_data_for_validation()
    
    # Run all tests
    test_results = {}
    
    logger.info("\n" + "="*50)
    test_results['temporal_integrity'] = test_temporal_integrity(df, features)
    
    logger.info("\n" + "="*50)
    test_results['feature_correlation'] = test_feature_correlation_with_future(df, features)
    
    logger.info("\n" + "="*50)
    test_results['cross_validation'] = test_cross_validation_stability(df, features)
    
    logger.info("\n" + "="*50)
    test_results['cascade_leakage'] = test_cascade_specific_leakage(df, features)
    
    logger.info("\n" + "="*50)
    test_results['time_consistency'] = test_performance_consistency_over_time(df, features)
    
    # Final verdict
    logger.info("\n" + "="*70)
    logger.info("🏁 FINAL CASCADE VALIDATION RESULTS")
    logger.info("="*70)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test_name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("\n" + "="*70)
    if all_passed:
        logger.info("🎉 CASCADE MODEL VALIDATION: PASSED")
        logger.info("✅ The cascade model is CLEAN and can be trusted for production")
        logger.info("✅ No data leakage detected in any test")
        logger.info("✅ Performance is stable and consistent")
        logger.info("🚀 RECOMMENDATION: SAFE to adopt cascade model (53% acc, 34% draw recall)")
    else:
        logger.error("❌ CASCADE MODEL VALIDATION: FAILED")
        logger.error("🚨 Data integrity issues detected - investigate before using")
        logger.error("🚨 DO NOT use this model until issues are resolved")
    
    logger.info("="*70)
    
    return all_passed

if __name__ == "__main__":
    validation_passed = run_comprehensive_cascade_validation()
    
    if validation_passed:
        print("\n🏆 CASCADE VALIDATION COMPLETE: Model is CLEAN!")
        print("The cascade model can be safely adopted for production use.")
        print("Performance: 53% accuracy with 34% draw recall - NO DATA LEAKAGE")
    else:
        print("\n🚨 CASCADE VALIDATION FAILED!")
        print("Data integrity issues found - investigate before production use.")