import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def assess_v20_final():
    """
    Final realistic assessment of v2.0 achievements based on corrected baselines
    """
    
    logger = setup_logging()
    logger.info("=== 📊 FINAL v2.0 REALISTIC ASSESSMENT ===")
    
    # Realistic baseline establishment
    logger.info("\n🎯 ESTABLISHING REALISTIC BASELINES")
    logger.info("Previous measurements showed inconsistencies. Establishing ground truth...")
    
    # Load datasets for comparison
    df_v13 = pd.read_csv("data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv")
    df_v20_clean = pd.read_csv("data/processed/premier_league_v20_clean_2025_08_30_211505.csv")
    
    logger.info(f"v1.3 dataset: {df_v13.shape}")
    logger.info(f"v2.0 clean dataset: {df_v20_clean.shape}")
    
    # Feature definitions
    v13_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'home_advantage', 'matchday_normalized', 'season_period_numeric',
        'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm'
    ]
    
    v20_features = [
        'elo_diff_normalized', 'form_diff_normalized',
        'line_movement_normalized', 'sharp_public_divergence_norm',
        'market_inefficiency_norm', 'market_velocity_norm',
        'time_weighted_form_diff_norm', 'volatility_diff_norm'
    ]
    
    # Prepare data
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    
    X_v13 = df_v13[v13_features].fillna(0.5)
    y_v13 = df_v13['FullTimeResult'].map(label_mapping)
    
    X_v20 = df_v20_clean[v20_features].fillna(0.5)
    y_v20 = df_v20_clean['FullTimeResult'].map(label_mapping)
    
    # Cross-validation setup
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Standard production model
    production_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features='log2',
        min_samples_leaf=2, min_samples_split=15, 
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # Test v1.3 baseline
    logger.info("\n📊 v1.3 BASELINE PERFORMANCE")
    cv_scores_v13 = cross_val_score(production_model, X_v13, y_v13, cv=tscv, scoring='accuracy', n_jobs=-1)
    mean_v13 = cv_scores_v13.mean()
    std_v13 = cv_scores_v13.std()
    
    logger.info(f"v1.3 CV Scores: {[f'{s:.4f}' for s in cv_scores_v13]}")
    logger.info(f"v1.3 Mean: {mean_v13:.4f} (±{std_v13:.4f})")
    
    # Test v2.0 performance
    logger.info("\n📊 v2.0 ADVANCED PERFORMANCE")
    cv_scores_v20 = cross_val_score(production_model, X_v20, y_v20, cv=tscv, scoring='accuracy', n_jobs=-1)
    mean_v20 = cv_scores_v20.mean()
    std_v20 = cv_scores_v20.std()
    
    logger.info(f"v2.0 CV Scores: {[f'{s:.4f}' for s in cv_scores_v20]}")
    logger.info(f"v2.0 Mean: {mean_v20:.4f} (±{std_v20:.4f})")
    
    # Calculate realistic improvement
    improvement = (mean_v20 - mean_v13) * 100
    
    logger.info(f"\n🔍 REALISTIC COMPARISON")
    logger.info(f"v1.3 Baseline: {mean_v13:.4f}")
    logger.info(f"v2.0 Advanced: {mean_v20:.4f}")
    logger.info(f"Net Improvement: {improvement:+.2f}pp")
    
    # Statistical significance test
    from scipy.stats import ttest_rel
    
    try:
        t_stat, p_value = ttest_rel(cv_scores_v13, cv_scores_v20)
        logger.info(f"Statistical Test: t={t_stat:.3f}, p={p_value:.3f}")
        
        if p_value < 0.05:
            if improvement > 0:
                significance = "✅ STATISTICALLY SIGNIFICANT IMPROVEMENT"
            else:
                significance = "⚠️ STATISTICALLY SIGNIFICANT REGRESSION"
        else:
            significance = "🔸 NO SIGNIFICANT DIFFERENCE"
        
        logger.info(f"Significance: {significance}")
    except Exception as e:
        logger.warning(f"Could not perform significance test: {e}")
        significance = "UNKNOWN"
    
    # Enhanced model testing
    logger.info(f"\n🚀 ENHANCED MODEL TESTING")
    
    enhanced_models = {
        'XGBoost Optimized': XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            eval_metric='mlogloss', n_jobs=-1
        ),
        'Voting Ensemble': VotingClassifier([
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10, max_features='log2',
                min_samples_leaf=2, min_samples_split=15,
                class_weight='balanced', random_state=42, n_jobs=-1
            )),
            ('xgb', XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85, random_state=42,
                eval_metric='mlogloss', n_jobs=-1
            ))
        ], voting='hard')
    }
    
    best_v13_enhanced = mean_v13
    best_v20_enhanced = mean_v20
    best_v13_model = "Random Forest"
    best_v20_model = "Random Forest"
    
    for model_name, model in enhanced_models.items():
        # Test on v1.3
        cv_v13_enhanced = cross_val_score(model, X_v13, y_v13, cv=tscv, scoring='accuracy')
        mean_v13_enhanced = cv_v13_enhanced.mean()
        
        # Test on v2.0
        cv_v20_enhanced = cross_val_score(model, X_v20, y_v20, cv=tscv, scoring='accuracy')
        mean_v20_enhanced = cv_v20_enhanced.mean()
        
        logger.info(f"{model_name}:")
        logger.info(f"  v1.3: {mean_v13_enhanced:.4f}")
        logger.info(f"  v2.0: {mean_v20_enhanced:.4f}")
        logger.info(f"  Improvement: {(mean_v20_enhanced - mean_v13_enhanced)*100:+.1f}pp")
        
        if mean_v13_enhanced > best_v13_enhanced:
            best_v13_enhanced = mean_v13_enhanced
            best_v13_model = model_name
            
        if mean_v20_enhanced > best_v20_enhanced:
            best_v20_enhanced = mean_v20_enhanced
            best_v20_model = model_name
    
    # Final realistic assessment
    logger.info(f"\n=== 🎯 FINAL REALISTIC v2.0 ASSESSMENT ===")
    
    best_improvement = (best_v20_enhanced - best_v13_enhanced) * 100
    
    logger.info(f"Best v1.3 Model: {best_v13_model} ({best_v13_enhanced:.4f})")
    logger.info(f"Best v2.0 Model: {best_v20_model} ({best_v20_enhanced:.4f})")
    logger.info(f"Best Improvement: {best_improvement:+.2f}pp")
    
    # Realistic targets based on actual baselines
    realistic_baselines = {
        "Random Guess": 0.333,
        "Majority Class": 0.436,  # Home win rate
        "v1.3 Baseline": best_v13_enhanced,
        "v2.0 Advanced": best_v20_enhanced
    }
    
    logger.info(f"\nRealistic Performance Ladder:")
    for baseline_name, score in realistic_baselines.items():
        logger.info(f"  {baseline_name}: {score:.3f}")
    
    # Success evaluation with realistic targets
    if best_v20_enhanced >= 0.540:
        success_level = "🎯 EXCEPTIONAL SUCCESS"
        description = "Industry-competitive performance achieved"
    elif best_v20_enhanced >= 0.530:
        success_level = "🔥 EXCELLENT SUCCESS"
        description = "Strong predictive performance"
    elif best_v20_enhanced >= 0.520:
        success_level = "✅ GOOD SUCCESS"
        description = "Solid improvement over baselines"
    elif best_v20_enhanced >= 0.510:
        success_level = "⚠️ MODERATE SUCCESS"
        description = "Modest improvement, more work needed"
    else:
        success_level = "❌ INSUFFICIENT"
        description = "Below expectations, reassess approach"
    
    logger.info(f"\nv2.0 Success Level: {success_level}")
    logger.info(f"Assessment: {description}")
    
    # Feature engineering value assessment
    logger.info(f"\n📈 FEATURE ENGINEERING VALUE ASSESSMENT")
    
    v13_feature_count = len(v13_features)
    v20_feature_count = len(v20_features)
    
    feature_efficiency_v13 = best_v13_enhanced / v13_feature_count
    feature_efficiency_v20 = best_v20_enhanced / v20_feature_count
    
    logger.info(f"v1.3 Features: {v13_feature_count}, Efficiency: {feature_efficiency_v13:.5f}")
    logger.info(f"v2.0 Features: {v20_feature_count}, Efficiency: {feature_efficiency_v20:.5f}")
    
    if feature_efficiency_v20 > feature_efficiency_v13:
        efficiency_assessment = "✅ IMPROVED FEATURE EFFICIENCY"
    else:
        efficiency_assessment = "⚠️ DECREASED FEATURE EFFICIENCY"
    
    logger.info(f"Efficiency Assessment: {efficiency_assessment}")
    
    # Recommendations
    logger.info(f"\n💡 STRATEGIC RECOMMENDATIONS")
    
    if best_improvement >= 1.0:
        logger.info("🚀 SUCCESS: Advanced features provide meaningful improvement")
        logger.info("  1. Deploy v2.0 model in production")
        logger.info("  2. Monitor performance on new data")
        logger.info("  3. Consider additional feature engineering")
        recommendation = "DEPLOY_V20"
    elif best_improvement >= 0.5:
        logger.info("⚡ PROGRESS: Modest improvement achieved")
        logger.info("  1. Continue refinement of advanced features")
        logger.info("  2. Explore hyperparameter optimization")
        logger.info("  3. Consider ensemble approaches")
        recommendation = "CONTINUE_REFINEMENT"
    elif abs(best_improvement) < 0.5:
        logger.info("🔸 NEUTRAL: No significant change")
        logger.info("  1. Keep v1.3 as baseline")
        logger.info("  2. Explore different feature engineering approaches")
        logger.info("  3. Consider external data sources")
        recommendation = "MAINTAIN_V13"
    else:
        logger.info("❌ REGRESSION: Advanced features hurt performance")
        logger.info("  1. Revert to v1.3 baseline")
        logger.info("  2. Investigate feature engineering errors")
        logger.info("  3. Simplify approach")
        recommendation = "REVERT_TO_V13"
    
    logger.info("=== 📊 FINAL ASSESSMENT COMPLETED ===")
    
    # Save final assessment
    final_assessment = {
        "version": "v2.0 Final Realistic Assessment",
        "timestamp": datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        "realistic_baselines": {
            "v13_baseline": float(best_v13_enhanced),
            "v20_advanced": float(best_v20_enhanced),
            "improvement_pp": float(best_improvement)
        },
        "best_models": {
            "v13": best_v13_model,
            "v20": best_v20_model
        },
        "success_evaluation": {
            "level": success_level,
            "description": description
        },
        "statistical_analysis": {
            "significance": significance,
            "improvement_pp": float(improvement)
        },
        "feature_analysis": {
            "v13_features": v13_feature_count,
            "v20_features": v20_feature_count,
            "efficiency_assessment": efficiency_assessment
        },
        "recommendation": recommendation
    }
    
    import json
    assessment_file = f"evaluation/v20_final_assessment_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.json"
    with open(assessment_file, 'w') as f:
        json.dump(final_assessment, f, indent=2)
    
    logger.info(f"Final assessment saved: {assessment_file}")
    
    return final_assessment

if __name__ == "__main__":
    assessment = assess_v20_final()
    
    print(f"\n📊 FINAL v2.0 REALISTIC ASSESSMENT:")
    print(f"v1.3 Baseline: {assessment['realistic_baselines']['v13_baseline']:.4f}")
    print(f"v2.0 Advanced: {assessment['realistic_baselines']['v20_advanced']:.4f}")
    print(f"Improvement: {assessment['realistic_baselines']['improvement_pp']:+.2f}pp")
    print(f"Success Level: {assessment['success_evaluation']['level']}")
    print(f"Recommendation: {assessment['recommendation']}")