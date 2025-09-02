import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def clean_v20_features():
    """
    Clean v2.0 features by removing high correlations and redundant features.
    Target: Create optimal feature set for 55%+ breakthrough
    """
    
    logger = setup_logging()
    logger.info("=== 🧹 CLEANING v2.0 FEATURES FOR BREAKTHROUGH ===")
    
    # Load v2.0 advanced dataset
    df = pd.read_csv("data/processed/premier_league_v20_advanced_2025_08_30_210808.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    logger.info(f"v2.0 dataset loaded: {df.shape}")
    
    # Original v2.0 feature set
    original_features = [
        'elo_diff_normalized', 'form_diff_normalized', 
        'line_movement_normalized', 'sharp_public_divergence_norm', 
        'market_inefficiency_norm', 'market_velocity_norm',
        'elo_market_interaction_norm', 'form_sharp_interaction_norm',
        'line_h2h_interaction_norm', 'velocity_elo_interaction_norm',
        'time_weighted_form_diff_norm', 'volatility_diff_norm'
    ]
    
    # Prepare target
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    X_original = df[original_features].fillna(0.5)
    
    logger.info(f"Original feature set: {len(original_features)} features")
    
    # Identify and remove high correlations
    logger.info("\n🔍 CORRELATION ANALYSIS & CLEANUP")
    correlation_matrix = X_original.corr()
    
    # Find highly correlated pairs (>0.8)
    high_corr_pairs = []
    for i in range(len(original_features)):
        for j in range(i+1, len(original_features)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.8:
                high_corr_pairs.append((original_features[i], original_features[j], abs(corr)))
    
    # Sort by correlation strength
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    logger.info(f"High correlations found (>0.8):")
    for feat1, feat2, corr in high_corr_pairs:
        logger.info(f"  {feat1} ↔ {feat2}: {corr:.3f}")
    
    # Remove redundant features based on correlation analysis
    features_to_remove = set()
    
    # Remove based on diagnostic results:
    # 1. line_h2h_interaction_norm (perfect correlation 1.0 with line_movement_normalized)
    # 2. elo_market_interaction_norm (0.96 correlation with elo_diff_normalized) 
    # 3. form_sharp_interaction_norm (0.71 correlation with form_diff_normalized)
    # 4. velocity_elo_interaction_norm (0.81 correlation with market_velocity_norm)
    
    features_to_remove.update([
        'line_h2h_interaction_norm',      # Perfect correlation with line_movement_normalized
        'elo_market_interaction_norm',    # Very high correlation with elo_diff_normalized
        'form_sharp_interaction_norm',    # High correlation with form_diff_normalized
        'velocity_elo_interaction_norm'   # High correlation with market_velocity_norm
    ])
    
    # Clean feature set
    clean_features = [f for f in original_features if f not in features_to_remove]
    
    logger.info(f"\nFeature cleanup:")
    logger.info(f"  Removing {len(features_to_remove)} redundant features: {list(features_to_remove)}")
    logger.info(f"  Keeping {len(clean_features)} clean features: {clean_features}")
    
    # Test clean feature set performance
    X_clean = df[clean_features].fillna(0.5)
    
    # Cross-validation setup
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Standard RandomForest
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features='log2',
        min_samples_leaf=2, min_samples_split=15, 
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    logger.info("\n🧪 TESTING FEATURE SETS")
    
    # Test original features
    cv_scores_original = cross_val_score(rf_model, X_original, y, cv=tscv, scoring='accuracy')
    mean_original = cv_scores_original.mean()
    logger.info(f"Original ({len(original_features)} features): {mean_original:.4f}")
    
    # Test clean features
    cv_scores_clean = cross_val_score(rf_model, X_clean, y, cv=tscv, scoring='accuracy')
    mean_clean = cv_scores_clean.mean()
    logger.info(f"Clean ({len(clean_features)} features): {mean_clean:.4f}")
    
    improvement = (mean_clean - mean_original) * 100
    logger.info(f"Cleanup impact: {improvement:+.1f}pp")
    
    # Enhanced model testing for breakthrough
    logger.info(f"\n🚀 BREAKTHROUGH MODEL TESTING")
    
    # Enhanced RandomForest
    rf_enhanced = RandomForestClassifier(
        n_estimators=500, max_depth=15, max_features='sqrt',
        min_samples_leaf=1, min_samples_split=5, 
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    cv_scores_enhanced = cross_val_score(rf_enhanced, X_clean, y, cv=tscv, scoring='accuracy')
    mean_enhanced = cv_scores_enhanced.mean()
    
    logger.info(f"Enhanced RF CV Scores: {[f'{s:.4f}' for s in cv_scores_enhanced]}")
    logger.info(f"Enhanced RF Mean: {mean_enhanced:.4f}")
    
    # Breakthrough assessment
    v13_baseline = 0.530
    v20_target = 0.550
    
    improvement_v13 = (mean_enhanced - v13_baseline) * 100
    target_gap = (v20_target - mean_enhanced) * 100
    
    if mean_enhanced >= 0.550:
        status = "🎯 v2.0 BREAKTHROUGH ACHIEVED!"
        color = "🟢"
    elif mean_enhanced >= 0.540:
        status = "🔥 EXCELLENT - Very close to breakthrough"
        color = "🟡"
    elif mean_enhanced >= 0.535:
        status = "⚡ VERY GOOD - Strong progress"
        color = "🟡"
    elif mean_enhanced >= v13_baseline:
        status = "✅ GOOD - Beats v1.3 baseline"
        color = "🟢"
    else:
        status = "❌ BELOW v1.3 baseline"
        color = "🔴"
    
    logger.info(f"vs v1.3 baseline: {improvement_v13:+.1f}pp")
    logger.info(f"Gap to v2.0 target: {target_gap:+.1f}pp")
    logger.info(f"Status: {color} {status}")
    
    # Feature importance analysis
    logger.info(f"\n📊 CLEAN FEATURE IMPORTANCE")
    rf_enhanced.fit(X_clean, y)
    
    feature_importance = list(zip(clean_features, rf_enhanced.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance):
        logger.info(f"  {i+1:2d}. {feature:<35}: {importance:.4f}")
    
    # Create final v2.0 clean dataset
    logger.info(f"\n💾 CREATING CLEAN v2.0 DATASET")
    
    output_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FullTimeResult'] + clean_features
    df_clean = df[output_columns].copy()
    
    # Save clean dataset
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_file = f"data/processed/premier_league_v20_clean_{timestamp}.csv"
    
    df_clean.to_csv(output_file, index=False)
    logger.info(f"Clean v2.0 dataset saved: {output_file}")
    
    # Data quality validation
    logger.info(f"Final clean dataset:")
    logger.info(f"  Shape: {df_clean.shape}")
    logger.info(f"  Features: {len(clean_features)}")
    logger.info(f"  Missing values: {df_clean[clean_features].isnull().sum().sum()}")
    
    # Feature statistics
    for feature in clean_features:
        values = df_clean[feature].dropna()
        logger.info(f"  {feature}: [{values.min():.3f}, {values.max():.3f}], mean={values.mean():.3f}")
    
    # Save metadata
    feature_metadata = {
        "version": "v2.0 Clean (Multicollinearity Removed)",
        "timestamp": timestamp,
        "cleanup_process": {
            "original_features": len(original_features),
            "removed_features": len(features_to_remove),
            "final_features": len(clean_features),
            "removed_list": list(features_to_remove),
            "correlation_threshold": 0.8
        },
        "performance": {
            "original_score": float(mean_original),
            "clean_score": float(mean_clean),
            "enhanced_score": float(mean_enhanced),
            "improvement_vs_v13": float(improvement_v13),
            "breakthrough_achieved": bool(mean_enhanced >= 0.550)
        },
        "clean_features": clean_features,
        "feature_importance": [{"feature": f, "importance": float(i)} 
                              for f, i in feature_importance],
        "data_quality": {
            "total_matches": len(df_clean),
            "missing_values": int(df_clean[clean_features].isnull().sum().sum())
        }
    }
    
    import json
    metadata_file = f"config/v20_clean_features_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    
    logger.info(f"Metadata saved: {metadata_file}")
    logger.info("=== 🧹 v2.0 FEATURE CLEANUP COMPLETED ===")
    
    return {
        'output_file': output_file,
        'metadata_file': metadata_file,
        'original_features': len(original_features),
        'clean_features': len(clean_features),
        'removed_features': len(features_to_remove),
        'enhanced_score': mean_enhanced,
        'breakthrough_achieved': mean_enhanced >= 0.550,
        'improvement_v13_pp': improvement_v13
    }

if __name__ == "__main__":
    result = clean_v20_features()
    print(f"\n🧹 v2.0 CLEANUP RESULTS:")
    print(f"Original Features: {result['original_features']}")
    print(f"Clean Features: {result['clean_features']}")
    print(f"Removed Features: {result['removed_features']}")
    print(f"Enhanced Score: {result['enhanced_score']:.4f}")
    print(f"Improvement vs v1.3: {result['improvement_v13_pp']:+.1f}pp")
    print(f"Breakthrough: {'🎯 YES' if result['breakthrough_achieved'] else '❌ NO'}")
    print(f"Output: {result['output_file']}")