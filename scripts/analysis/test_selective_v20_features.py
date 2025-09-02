import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def test_selective_v20_features():
    """
    Test adding 1-2 best v2.0 features individually to v1.3 stable baseline
    Goal: See if selective integration can improve without regression
    """
    
    logger = setup_logging()
    logger.info("=== 🧪 SELECTIVE v2.0 FEATURE INTEGRATION TEST ===")
    
    # Load datasets
    df_v13 = pd.read_csv("data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv")
    df_v20 = pd.read_csv("data/processed/premier_league_v20_clean_2025_08_30_211505.csv")
    
    logger.info(f"v1.3 dataset: {df_v13.shape}")
    logger.info(f"v2.0 dataset: {df_v20.shape}")
    
    # v1.3 stable baseline (proven 51.37% with current measurement)
    v13_baseline_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 
        'corners_diff_normalized', 'market_entropy_norm'
    ]
    
    # Best v2.0 features to test (from feature importance analysis)
    candidate_v20_features = [
        'time_weighted_form_diff_norm',    # High F-score, temporal intelligence
        'line_movement_normalized',        # Market sentiment signal
        'sharp_public_divergence_norm',    # Smart money vs public
        'market_inefficiency_norm'         # Cross-bookmaker disagreement
    ]
    
    logger.info(f"v1.3 baseline ({len(v13_baseline_features)}): {v13_baseline_features}")
    logger.info(f"v2.0 candidates ({len(candidate_v20_features)}): {candidate_v20_features}")
    
    # Create merged dataset for testing
    # Use v1.3 as base and add v2.0 features where available
    
    # Match datasets by creating simple match keys
    df_v13_test = df_v13.copy()
    df_v20_test = df_v20.copy()
    
    # For testing, we'll use the v2.0 clean dataset which has the advanced features
    # and test combinations with v1.3 baseline features
    
    # Cross-validation setup
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Model (same as successful v1.3)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features='log2',
        min_samples_leaf=2, min_samples_split=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # Prepare target
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    
    results = {}
    
    # Test baseline first (using v2.0 dataset for consistency)
    logger.info("\n📊 BASELINE PERFORMANCE TEST")
    
    # Check which baseline features are available in v2.0 dataset
    available_baseline = [f for f in v13_baseline_features if f in df_v20.columns]
    logger.info(f"Available baseline features in v2.0: {available_baseline}")
    
    if len(available_baseline) < len(v13_baseline_features):
        logger.warning(f"Missing baseline features. Using available subset.")
    
    # Test baseline with available features
    X_baseline = df_v20[available_baseline].fillna(0.5)
    y = df_v20['FullTimeResult'].map(label_mapping)
    
    cv_baseline = cross_val_score(model, X_baseline, y, cv=tscv, scoring='accuracy')
    baseline_score = cv_baseline.mean()
    
    logger.info(f"Baseline ({len(available_baseline)} features): {baseline_score:.4f}")
    results['baseline'] = {'score': baseline_score, 'features': available_baseline}
    
    # Test each v2.0 feature addition individually
    logger.info("\n🧪 INDIVIDUAL FEATURE ADDITION TESTS")
    
    best_addition = None
    best_improvement = 0
    
    for candidate_feature in candidate_v20_features:
        if candidate_feature in df_v20.columns:
            # Create augmented feature set
            augmented_features = available_baseline + [candidate_feature]
            X_augmented = df_v20[augmented_features].fillna(0.5)
            
            # Test performance
            cv_augmented = cross_val_score(model, X_augmented, y, cv=tscv, scoring='accuracy')
            augmented_score = cv_augmented.mean()
            improvement = (augmented_score - baseline_score) * 100
            
            if improvement > 0:
                status = f"✅ +{improvement:.1f}pp"
            else:
                status = f"❌ {improvement:+.1f}pp"
            
            logger.info(f"{candidate_feature}: {augmented_score:.4f} {status}")
            
            results[f'baseline + {candidate_feature}'] = {
                'score': augmented_score,
                'improvement_pp': improvement,
                'features': augmented_features
            }
            
            # Track best addition
            if improvement > best_improvement:
                best_improvement = improvement
                best_addition = candidate_feature
        else:
            logger.warning(f"Feature {candidate_feature} not available in v2.0 dataset")
    
    # Test best 2-feature combination
    if best_addition and best_improvement > 0:
        logger.info(f"\n🚀 TESTING BEST 2-FEATURE COMBINATION")
        
        # Try adding the second-best feature
        remaining_candidates = [f for f in candidate_v20_features if f != best_addition and f in df_v20.columns]
        
        best_combo_score = baseline_score
        best_combo_features = available_baseline + [best_addition]
        
        for second_feature in remaining_candidates[:2]:  # Test top 2
            combo_features = available_baseline + [best_addition, second_feature]
            X_combo = df_v20[combo_features].fillna(0.5)
            
            cv_combo = cross_val_score(model, X_combo, y, cv=tscv, scoring='accuracy')
            combo_score = cv_combo.mean()
            combo_improvement = (combo_score - baseline_score) * 100
            
            logger.info(f"+ {best_addition} + {second_feature}: {combo_score:.4f} ({combo_improvement:+.1f}pp)")
            
            if combo_score > best_combo_score:
                best_combo_score = combo_score
                best_combo_features = combo_features
        
        results['best_combination'] = {
            'score': best_combo_score,
            'improvement_pp': (best_combo_score - baseline_score) * 100,
            'features': best_combo_features
        }
    
    # Final assessment
    logger.info(f"\n=== 🎯 SELECTIVE INTEGRATION ASSESSMENT ===")
    
    if best_addition and best_improvement > 0.5:
        logger.info(f"✅ BENEFICIAL INTEGRATION FOUND:")
        logger.info(f"  Best Addition: {best_addition}")
        logger.info(f"  Improvement: +{best_improvement:.1f}pp")
        recommendation = f"ADD_{best_addition.upper()}"
    elif best_addition and best_improvement > 0:
        logger.info(f"🔸 MARGINAL IMPROVEMENT:")
        logger.info(f"  Best Addition: {best_addition}")
        logger.info(f"  Improvement: +{best_improvement:.1f}pp")
        recommendation = "MARGINAL_BENEFIT"
    else:
        logger.info(f"❌ NO BENEFICIAL INTEGRATION:")
        logger.info(f"  All additions hurt or neutral performance")
        recommendation = "KEEP_BASELINE_ONLY"
    
    logger.info(f"Recommendation: {recommendation}")
    logger.info("=== 🧪 SELECTIVE INTEGRATION TEST COMPLETED ===")
    
    return {
        'baseline_score': baseline_score,
        'best_addition': best_addition,
        'best_improvement': best_improvement,
        'recommendation': recommendation,
        'detailed_results': results
    }

if __name__ == "__main__":
    result = test_selective_v20_features()
    print(f"\n🧪 SELECTIVE v2.0 INTEGRATION RESULTS:")
    print(f"Baseline Score: {result['baseline_score']:.4f}")
    print(f"Best Addition: {result['best_addition']}")
    print(f"Best Improvement: {result['best_improvement']:+.1f}pp")
    print(f"Recommendation: {result['recommendation']}")