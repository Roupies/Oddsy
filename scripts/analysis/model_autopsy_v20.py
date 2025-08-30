import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def model_autopsy_v20():
    """
    CRITICAL DIAGNOSTIC: Why are odds features detrimental?
    30-minute EXPRESS analysis to identify root causes.
    
    Analysis Plan:
    1. Baseline performance verification
    2. Feature correlation matrix (identify redundancies)  
    3. Feature importance analysis
    4. Individual odds feature impact
    5. Recommendation: Fix baseline vs Advanced features
    """
    
    logger = setup_logging()
    logger.info("=== 🔍 MODEL AUTOPSY v2.0 - Express Diagnostic ===")
    
    # Load v2.0 dataset
    v20_file = "data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv"
    logger.info(f"Loading dataset: {v20_file}")
    
    df = pd.read_csv(v20_file)
    logger.info(f"Dataset: {df.shape}")
    
    # Feature definitions
    v12_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'home_advantage', 'matchday_normalized', 'season_period_numeric',
        'shots_diff_normalized', 'corners_diff_normalized'
    ]
    
    odds_features = [
        'market_home_advantage_norm', 'market_entropy_norm', 'pinnacle_home_advantage_norm'
    ]
    
    all_features = v12_features + odds_features
    
    # Prepare target
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Features: {len(v12_features)} baseline + {len(odds_features)} odds = {len(all_features)} total")
    
    # =========================
    # 1. BASELINE VERIFICATION
    # =========================
    logger.info("\n🔍 1. BASELINE PERFORMANCE VERIFICATION")
    
    X_baseline = df[v12_features]
    
    # Quick RF test on baseline
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv_baseline = cross_val_score(rf_baseline, X_baseline, y, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')
    baseline_mean = cv_baseline.mean()
    
    logger.info(f"  v1.2 Baseline: {baseline_mean:.4f} (±{cv_baseline.std():.4f})")
    logger.info(f"  Target 53.0%: {'✅ ACHIEVED' if baseline_mean >= 0.530 else '❌ FAILED'}")
    logger.info(f"  Historical 52.2%: {'✅ BEAT' if baseline_mean >= 0.522 else '❌ WORSE'}")
    
    if baseline_mean < 0.525:
        logger.warning("  🚨 BASELINE ISSUE: Model underperforming - investigate baseline first!")
        baseline_issue = True
    else:
        logger.info("  ✅ BASELINE OK: Issue likely with odds features")
        baseline_issue = False
    
    # =========================
    # 2. CORRELATION ANALYSIS
    # =========================
    logger.info("\n🔍 2. FEATURE CORRELATION MATRIX")
    
    X_all = df[all_features]
    corr_matrix = X_all.corr()
    
    # Focus on cross-correlations between baseline and odds
    logger.info("  Cross-correlations (Baseline vs Odds):")
    for odds_feat in odds_features:
        for baseline_feat in v12_features:
            corr = corr_matrix.loc[odds_feat, baseline_feat]
            if abs(corr) > 0.3:  # Significant correlation
                logger.info(f"    {odds_feat} vs {baseline_feat}: {corr:.3f}")
    
    # Highest correlations with odds features
    logger.info("\n  Highest correlations with odds features:")
    for odds_feat in odds_features:
        correlations = corr_matrix[odds_feat].abs().sort_values(ascending=False)
        # Exclude self-correlation
        correlations = correlations[correlations.index != odds_feat]
        top_corr = correlations.iloc[0]
        top_feature = correlations.index[0]
        
        logger.info(f"    {odds_feat} most correlated with {top_feature}: {top_corr:.3f}")
        
        if top_corr > 0.7:
            logger.warning(f"    🚨 HIGH REDUNDANCY: {odds_feat} ≈ {top_feature}")
        elif top_corr > 0.5:
            logger.warning(f"    ⚠️ MODERATE REDUNDANCY: {odds_feat} ~ {top_feature}")
    
    # =========================
    # 3. FEATURE IMPORTANCE
    # =========================
    logger.info("\n🔍 3. FEATURE IMPORTANCE ANALYSIS")
    
    # Train RF on full dataset to get feature importance
    rf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_full.fit(X_all, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': rf_full.feature_importances_,
        'type': ['baseline'] * len(v12_features) + ['odds'] * len(odds_features)
    }).sort_values('importance', ascending=False)
    
    logger.info("  Feature Importance Ranking:")
    for i, row in importance_df.iterrows():
        logger.info(f"    {row['importance']:.3f} - {row['feature']} ({row['type']})")
    
    # Check if odds features are bottom performers
    bottom_3 = importance_df.tail(3)['type'].tolist()
    if bottom_3.count('odds') >= 2:
        logger.warning("  🚨 ODDS FEATURES IN BOTTOM 3 - Clear signal they're not adding value")
        odds_low_importance = True
    else:
        logger.info("  ✅ ODDS FEATURES NOT ALL LOW IMPORTANCE")
        odds_low_importance = False
    
    # =========================
    # 4. INDIVIDUAL ODDS IMPACT
    # =========================
    logger.info("\n🔍 4. INDIVIDUAL ODDS FEATURE IMPACT")
    
    baseline_performance = baseline_mean
    
    for odds_feat in odds_features:
        # Test baseline + single odds feature
        test_features = v12_features + [odds_feat]
        X_test = df[test_features]
        
        cv_test = cross_val_score(rf_baseline, X_test, y, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')
        test_mean = cv_test.mean()
        
        impact = (test_mean - baseline_performance) * 100
        
        if impact > 0.2:
            status = "✅ POSITIVE"
        elif impact > -0.2:
            status = "🔸 NEUTRAL"  
        else:
            status = "❌ NEGATIVE"
        
        logger.info(f"  {odds_feat}: {test_mean:.4f} ({impact:+.2f}pp) {status}")
    
    # =========================
    # 5. PERFORMANCE COMPARISON
    # =========================
    logger.info("\n🔍 5. MODEL PERFORMANCE COMPARISON")
    
    # v1.2 baseline
    logger.info(f"  v1.2 Baseline (8 features): {baseline_mean:.4f}")
    
    # v2.0 with odds
    X_v20 = df[all_features]
    cv_v20 = cross_val_score(rf_baseline, X_v20, y, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')
    v20_mean = cv_v20.mean()
    
    logger.info(f"  v2.0 Enhanced (11 features): {v20_mean:.4f}")
    
    odds_impact = (v20_mean - baseline_mean) * 100
    logger.info(f"  Odds Impact: {odds_impact:+.2f}pp")
    
    # =========================
    # 6. DIAGNOSIS & RECOMMENDATION
    # =========================
    logger.info("\n🎯 6. DIAGNOSTIC SUMMARY")
    
    # Collect evidence
    evidence = {
        'baseline_underperforms': baseline_mean < 0.525,
        'target_not_achieved': baseline_mean < 0.530,
        'odds_detrimental': odds_impact < -0.5,
        'high_redundancy': any([corr_matrix.loc[odds_feat, baseline_feat] > 0.7 
                               for odds_feat in odds_features 
                               for baseline_feat in v12_features]),
        'odds_low_importance': odds_low_importance
    }
    
    logger.info("  Evidence Summary:")
    for condition, result in evidence.items():
        logger.info(f"    {condition}: {'TRUE' if result else 'FALSE'}")
    
    # Make recommendation
    logger.info("\n🚀 FINAL RECOMMENDATION:")
    
    if evidence['baseline_underperforms']:
        recommendation = "FIX_BASELINE_FIRST"
        logger.info("  ❌ CRITICAL: Fix baseline model before adding odds features")
        logger.info("     - Baseline < 52.5% indicates fundamental issues")
        logger.info("     - Investigate: Elo bug, feature engineering, data leakage")
        logger.info("     - Adding odds to broken baseline = lipstick on a pig")
        
    elif evidence['high_redundancy']:
        recommendation = "FEATURE_SELECTION_REQUIRED"
        logger.info("  ⚠️ REDUNDANCY: Odds features too similar to baseline")
        logger.info("     - High correlation indicates circular dependency")
        logger.info("     - Try: Feature selection, L1 regularization")
        logger.info("     - Or: Different odds features (value betting, market movement)")
        
    elif evidence['odds_detrimental'] and evidence['odds_low_importance']:
        recommendation = "ADVANCED_FEATURES_NEEDED"
        logger.info("  🔄 PIVOT: Current odds features not working")
        logger.info("     - Baseline is OK, but current odds add noise")
        logger.info("     - Try: Value betting signals, market movements, bookmaker patterns")
        logger.info("     - Current features too basic/redundant")
        
    else:
        recommendation = "DEEP_INVESTIGATION_REQUIRED"
        logger.info("  🔍 UNCLEAR: Mixed signals require deeper analysis")
        logger.info("     - Run longer diagnostic with more models")
        logger.info("     - Check data quality, feature engineering bugs")
        
    # Save diagnostic results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "baseline_performance": float(baseline_mean),
        "v20_performance": float(v20_mean), 
        "odds_impact_pp": float(odds_impact),
        "evidence": evidence,
        "recommendation": recommendation,
        "feature_importance": importance_df.to_dict('records'),
        "correlation_matrix": corr_matrix.to_dict()
    }
    
    import json
    results_file = f"models/diagnostic_v20_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📊 Diagnostic saved: {results_file}")
    logger.info("=== 🔍 MODEL AUTOPSY COMPLETED ===")
    
    return {
        'recommendation': recommendation,
        'baseline_performance': baseline_mean,
        'v20_performance': v20_mean,
        'odds_impact': odds_impact,
        'evidence': evidence,
        'results_file': results_file
    }

if __name__ == "__main__":
    result = model_autopsy_v20()
    print(f"\n🎯 AUTOPSY RESULTS:")
    print(f"Baseline: {result['baseline_performance']:.4f}")
    print(f"v2.0: {result['v20_performance']:.4f}")  
    print(f"Odds Impact: {result['odds_impact']:+.2f}pp")
    print(f"Recommendation: {result['recommendation']}")