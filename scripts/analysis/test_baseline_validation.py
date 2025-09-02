#!/usr/bin/env python3
"""
Test rapide pour valider notre baseline avec le dataset de référence
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def test_baseline_with_reference_data():
    """Test avec le dataset de référence validé"""
    logger = setup_logging()
    logger.info("🔍 TEST BASELINE AVEC DATASET DE RÉFÉRENCE")
    
    # Load reference dataset
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Prepare v1.5 baseline features (exactement comme avant)
    v15_features = [
        'elo_diff_normalized',
        'form_diff_normalized', 
        'h2h_score',
        'matchday_normalized',
        'shots_diff_normalized',
        'corners_diff_normalized'
    ]
    
    # Target encoding
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(target_mapping)
    
    # Features
    X = df[v15_features].copy()
    X = X.fillna(X.mean())
    
    # Sort by date
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    X_sorted = X.reindex(df_sorted.index)
    y_sorted = y.reindex(df_sorted.index)
    
    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(rf_model, X_sorted, y_sorted, cv=tscv, scoring='accuracy', n_jobs=-1)
    
    mean_accuracy = cv_scores.mean()
    std_accuracy = cv_scores.std()
    
    logger.info(f"📊 BASELINE TEST RESULTS:")
    logger.info(f"   Dataset: v13_complete_with_dates.csv")
    logger.info(f"   Features: {len(v15_features)} features")
    logger.info(f"   Cross-validation accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    logger.info(f"   Individual folds: {[f'{score:.3f}' for score in cv_scores]}")
    
    # Compare to expected baseline
    expected_baseline = 0.5211
    diff = mean_accuracy - expected_baseline
    logger.info(f"   vs Expected baseline (52.11%): {diff:+.3f}")
    
    if abs(diff) < 0.02:  # Within 2%
        logger.info("✅ BASELINE VALIDATED - Results consistent")
    else:
        logger.warning(f"⚠️  BASELINE MISMATCH - Difference: {diff:+.1%}")
    
    return mean_accuracy

def quick_audit_xg_features():
    """Audit rapide des features xG pour data leakage"""
    logger = setup_logging()
    logger.info("\n🕵️ AUDIT RAPIDE DES FEATURES XG")
    
    df = pd.read_csv('data/processed/premier_league_xg_enhanced_2025_08_31_195341.csv')
    
    # Check suspicious efficiency features
    efficiency_features = ['Home_GoalsVsXG', 'Away_GoalsVsXG']
    
    for feature in efficiency_features:
        if feature in df.columns:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            min_val = df[feature].min()
            max_val = df[feature].max()
            
            logger.info(f"{feature}: mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
            
            # Red flags
            if mean_val > 2.0 or max_val > 10.0:
                logger.error(f"❌ SUSPICIOUS: {feature} has unrealistic values")
            
            # Check for perfect correlation with results
            target_mapping = {'H': 0, 'D': 1, 'A': 2}
            y = df['FullTimeResult'].map(target_mapping)
            
            correlation = df[feature].corr(y)
            logger.info(f"   Correlation with results: {correlation:.3f}")
            
            if abs(correlation) > 0.3:
                logger.error(f"❌ DATA LEAKAGE: {feature} too correlated with results!")

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("🚨 VALIDATION RAPIDE DU BASELINE ET AUDIT XG")
    logger.info("=" * 50)
    
    # Test baseline
    baseline_accuracy = test_baseline_with_reference_data()
    
    # Quick xG audit
    quick_audit_xg_features()
    
    logger.info(f"\n🏁 VALIDATION COMPLETE")
    logger.info(f"Baseline accuracy: {baseline_accuracy:.1%}")