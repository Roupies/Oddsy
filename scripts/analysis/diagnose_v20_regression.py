import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_classif

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def diagnose_v20_regression():
    """
    Diagnose why v2.0 advanced features performed worse than v1.3
    """
    
    logger = setup_logging()
    logger.info("=== 🔍 DIAGNOSING v2.0 REGRESSION ===")
    
    # Load v2.0 advanced dataset
    df_v20 = pd.read_csv("data/processed/premier_league_v20_advanced_2025_08_30_210808.csv")
    
    # Load v1.3 dataset for comparison
    df_v13 = pd.read_csv("data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv")
    
    logger.info(f"v2.0 dataset: {df_v20.shape}")
    logger.info(f"v1.3 dataset: {df_v13.shape}")
    
    # Define feature sets
    v20_features = [
        'elo_diff_normalized', 'form_diff_normalized', 
        'line_movement_normalized', 'sharp_public_divergence_norm', 
        'market_inefficiency_norm', 'market_velocity_norm',
        'elo_market_interaction_norm', 'form_sharp_interaction_norm',
        'line_h2h_interaction_norm', 'velocity_elo_interaction_norm',
        'time_weighted_form_diff_norm', 'volatility_diff_norm'
    ]
    
    v13_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'home_advantage', 'matchday_normalized', 'season_period_numeric',
        'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm'  # The one good odds feature from v1.3
    ]
    
    # Prepare targets
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_v20 = df_v20['FullTimeResult'].map(label_mapping)
    y_v13 = df_v13['FullTimeResult'].map(label_mapping)
    
    # Prepare feature matrices
    X_v20 = df_v20[v20_features].fillna(0.5)
    X_v13 = df_v13[v13_features].fillna(0.5)
    
    logger.info(f"v2.0 features ({len(v20_features)}): {v20_features}")
    logger.info(f"v1.3 features ({len(v13_features)}): {v13_features}")
    
    # Cross-validation setup
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Standard RandomForest for comparison
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features='log2',
        min_samples_leaf=2, min_samples_split=15, 
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # Test v1.3 performance
    logger.info("\n📊 TESTING v1.3 BASELINE")
    cv_scores_v13 = cross_val_score(rf_model, X_v13, y_v13, cv=tscv, scoring='accuracy')
    mean_v13 = cv_scores_v13.mean()
    logger.info(f"v1.3 CV Scores: {[f'{s:.4f}' for s in cv_scores_v13]}")
    logger.info(f"v1.3 Mean: {mean_v13:.4f}")
    
    # Test v2.0 performance  
    logger.info("\n📊 TESTING v2.0 ADVANCED")
    cv_scores_v20 = cross_val_score(rf_model, X_v20, y_v20, cv=tscv, scoring='accuracy')
    mean_v20 = cv_scores_v20.mean()
    logger.info(f"v2.0 CV Scores: {[f'{s:.4f}' for s in cv_scores_v20]}")
    logger.info(f"v2.0 Mean: {mean_v20:.4f}")
    
    # Regression analysis
    regression = (mean_v13 - mean_v20) * 100
    logger.info(f"\n🔍 REGRESSION ANALYSIS")
    logger.info(f"Performance Change: {regression:+.1f}pp (v1.3 → v2.0)")
    
    if regression > 0:
        logger.warning(f"⚠️ SIGNIFICANT REGRESSION: v2.0 is {regression:.1f}pp worse than v1.3")
    else:
        logger.info(f"✅ IMPROVEMENT: v2.0 is {abs(regression):.1f}pp better than v1.3")
    
    # Feature importance comparison
    logger.info(f"\n📈 FEATURE IMPORTANCE ANALYSIS")
    
    # v1.3 feature importance
    rf_model.fit(X_v13, y_v13)
    v13_importance = list(zip(v13_features, rf_model.feature_importances_))
    v13_importance.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("v1.3 Feature Importance:")
    for i, (feature, importance) in enumerate(v13_importance):
        logger.info(f"  {i+1:2d}. {feature:<30}: {importance:.4f}")
    
    # v2.0 feature importance
    rf_model.fit(X_v20, y_v20)
    v20_importance = list(zip(v20_features, rf_model.feature_importances_))
    v20_importance.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("\nv2.0 Feature Importance:")
    for i, (feature, importance) in enumerate(v20_importance):
        logger.info(f"  {i+1:2d}. {feature:<30}: {importance:.4f}")
    
    # Test core features only from v2.0
    logger.info(f"\n🔬 TESTING CORE FEATURES ONLY")
    core_features = ['elo_diff_normalized', 'form_diff_normalized', 'line_movement_normalized', 
                     'sharp_public_divergence_norm', 'market_inefficiency_norm']
    
    X_core = df_v20[core_features].fillna(0.5)
    cv_scores_core = cross_val_score(rf_model, X_core, y_v20, cv=tscv, scoring='accuracy')
    mean_core = cv_scores_core.mean()
    logger.info(f"Core features ({len(core_features)}) CV: {[f'{s:.4f}' for s in cv_scores_core]}")
    logger.info(f"Core features mean: {mean_core:.4f}")
    
    # Progressive feature addition test
    logger.info(f"\n⚡ PROGRESSIVE FEATURE ADDITION TEST")
    
    progressive_features = [
        ['elo_diff_normalized'],
        ['elo_diff_normalized', 'form_diff_normalized'],  
        ['elo_diff_normalized', 'form_diff_normalized', 'line_movement_normalized'],
        ['elo_diff_normalized', 'form_diff_normalized', 'line_movement_normalized', 'market_inefficiency_norm'],
        core_features,
        v20_features
    ]
    
    for i, feature_set in enumerate(progressive_features):
        X_prog = df_v20[feature_set].fillna(0.5)
        cv_scores_prog = cross_val_score(rf_model, X_prog, y_v20, cv=tscv, scoring='accuracy')
        mean_prog = cv_scores_prog.mean()
        logger.info(f"Step {i+1} ({len(feature_set):2d} features): {mean_prog:.4f}")
    
    # Correlation analysis of v2.0 features
    logger.info(f"\n🔗 CORRELATION ANALYSIS")
    correlation_matrix = X_v20.corr()
    
    # Find high correlations (> 0.7)
    high_corr_pairs = []
    for i in range(len(v20_features)):
        for j in range(i+1, len(v20_features)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr_pairs.append((v20_features[i], v20_features[j], corr))
    
    if high_corr_pairs:
        logger.warning(f"High correlations found (>0.7):")
        for feat1, feat2, corr in high_corr_pairs:
            logger.warning(f"  {feat1} ↔ {feat2}: {corr:.3f}")
    else:
        logger.info("No high correlations (>0.7) found among features")
    
    # Statistical significance test 
    logger.info(f"\n📊 STATISTICAL SIGNIFICANCE")
    from scipy.stats import ttest_rel
    
    try:
        t_stat, p_value = ttest_rel(cv_scores_v13, cv_scores_v20)
        logger.info(f"Paired t-test: t={t_stat:.3f}, p={p_value:.3f}")
        if p_value < 0.05:
            logger.warning(f"⚠️ SIGNIFICANT DIFFERENCE: p<0.05 confirms regression")
        else:
            logger.info(f"✅ NO SIGNIFICANT DIFFERENCE: p≥0.05")
    except:
        logger.warning("Could not perform statistical significance test")
    
    # Recommendations
    logger.info(f"\n💡 DIAGNOSTIC RECOMMENDATIONS")
    
    if regression > 1.0:
        logger.info("🚨 CRITICAL REGRESSION DETECTED:")
        logger.info("  1. Feature engineering complexity is hurting performance")
        logger.info("  2. Consider reverting to simpler feature set")
        logger.info("  3. Individual feature validation needed")
        logger.info("  4. Possible overfitting with 12 features")
        recommendation = "REVERT_TO_SIMPLER_FEATURES"
    elif regression > 0.5:
        logger.info("⚠️ MODERATE REGRESSION:")  
        logger.info("  1. Some advanced features may be redundant")
        logger.info("  2. Try feature selection with fewer features")
        logger.info("  3. Validate feature engineering logic")
        recommendation = "FEATURE_SELECTION_REQUIRED"
    else:
        logger.info("✅ MINOR IMPACT:")
        logger.info("  1. Performance change not significant")
        logger.info("  2. Continue with hyperparameter tuning")
        recommendation = "CONTINUE_OPTIMIZATION"
    
    logger.info("=== 🔍 DIAGNOSTIC COMPLETED ===")
    
    return {
        'v13_score': mean_v13,
        'v20_score': mean_v20,
        'regression_pp': regression,
        'core_features_score': mean_core,
        'recommendation': recommendation,
        'high_correlations': len(high_corr_pairs)
    }

if __name__ == "__main__":
    result = diagnose_v20_regression()
    print(f"\n🔍 REGRESSION DIAGNOSTIC RESULTS:")
    print(f"v1.3 Score: {result['v13_score']:.4f}")
    print(f"v2.0 Score: {result['v20_score']:.4f}")
    print(f"Regression: {result['regression_pp']:+.1f}pp")
    print(f"Core Features Score: {result['core_features_score']:.4f}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"High Correlations: {result['high_correlations']}")