import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def scientific_reproducibility_test():
    """
    Scientific approach to resolve 53.05% vs 51.37% mystery:
    1. Control randomness with fixed random_state
    2. Compare selected features across runs
    3. Test multiple seeds to understand variance range
    """
    
    logger = setup_logging()
    logger.info("=== 🔬 SCIENTIFIC REPRODUCIBILITY TEST ===")
    
    # Load dataset (same as all previous tests)
    df = pd.read_csv("data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv")
    logger.info(f"Dataset: {df.shape}")
    
    # Clean features (same as successful cleanup)
    clean_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'season_period_numeric',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm'
    ]
    
    # Prepare data
    X_clean = df[clean_features].fillna(0.5)
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Features: {clean_features}")
    logger.info(f"Feature matrix: {X_clean.shape}")
    
    # Cross-validation setup with CONTROLLED randomness
    cv_folds = 5
    
    # Test seeds to understand performance variance
    test_seeds = [0, 42, 99, 123, 999, 1337, 2025]
    
    logger.info(f"Testing {len(test_seeds)} random seeds: {test_seeds}")
    
    # =======================
    # EXPERIMENT 1: MULTIPLE SEEDS WITH XGBOOST + SELECTKBEST
    # =======================
    logger.info("\n🧪 EXPERIMENT 1: Multiple Seeds - XGBoost + SelectKBest k=7")
    
    results = {}
    
    for seed in test_seeds:
        logger.info(f"\n--- Testing Seed {seed} ---")
        
        # Create TimeSeriesSplit with controlled randomness
        # Note: TimeSeriesSplit doesn't use random_state, but we'll control the model
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Feature selection with controlled randomness
        np.random.seed(seed)  # Control numpy randomness
        
        selector = SelectKBest(score_func=f_classif, k=7)
        X_selected = selector.fit_transform(X_clean, y)
        
        # Get selected features
        selected_features = [clean_features[i] for i in range(len(clean_features)) if selector.get_support()[i]]
        
        # XGBoost model with controlled randomness
        try:
            from xgboost import XGBClassifier
            
            xgb_model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, 
                random_state=seed,  # CRITICAL: Control XGBoost randomness
                eval_metric='mlogloss', n_jobs=1  # Single thread for consistency
            )
            
            # Cross-validation with controlled randomness
            cv_scores = cross_val_score(xgb_model, X_selected, y, cv=tscv, scoring='accuracy')
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            logger.info(f"  Selected features: {selected_features}")
            logger.info(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
            logger.info(f"  Mean: {mean_score:.4f} (±{std_score:.4f})")
            
            # Check if this matches the mysterious 53.05%
            if abs(mean_score - 0.5305) < 0.001:
                logger.info(f"  🎯 FOUND 53.05% with seed {seed}!")
            
            results[seed] = {
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores.tolist(),
                'selected_features': selected_features,
                'n_features_selected': len(selected_features)
            }
            
        except ImportError:
            logger.warning(f"XGBoost not available for seed {seed}")
            continue
    
    # =======================
    # EXPERIMENT 2: FEATURE SELECTION STABILITY
    # =======================
    logger.info("\n🧪 EXPERIMENT 2: Feature Selection Stability Analysis")
    
    # Analyze which features are consistently selected
    feature_selection_counts = {}
    for feature in clean_features:
        feature_selection_counts[feature] = 0
    
    for seed, result in results.items():
        for feature in result['selected_features']:
            feature_selection_counts[feature] += 1
    
    logger.info("Feature selection frequency across seeds:")
    sorted_features = sorted(feature_selection_counts.items(), key=lambda x: x[1], reverse=True)
    for feature, count in sorted_features:
        pct = (count / len(results)) * 100
        logger.info(f"  {feature}: {count}/{len(results)} ({pct:.0f}%)")
    
    # =======================
    # EXPERIMENT 3: PERFORMANCE STATISTICS
    # =======================
    logger.info("\n🧪 EXPERIMENT 3: Performance Statistics Analysis")
    
    scores = [result['mean_score'] for result in results.values()]
    
    if scores:
        min_score = min(scores)
        max_score = max(scores)
        mean_score_across_seeds = np.mean(scores)
        std_score_across_seeds = np.std(scores)
        
        logger.info(f"Performance across {len(scores)} seeds:")
        logger.info(f"  Minimum: {min_score:.4f}")
        logger.info(f"  Maximum: {max_score:.4f}")
        logger.info(f"  Mean: {mean_score_across_seeds:.4f}")
        logger.info(f"  Std Dev: {std_score_across_seeds:.4f}")
        logger.info(f"  Range: {(max_score - min_score)*100:.1f}pp")
        
        # Check if 53.05% falls within this range
        within_range = min_score <= 0.5305 <= max_score
        logger.info(f"  53.05% within range: {'✅ YES' if within_range else '❌ NO'}")
        
        # Statistical significance of variance
        if std_score_across_seeds > 0.005:  # > 0.5pp standard deviation
            logger.warning(f"⚠️ HIGH VARIANCE: {std_score_across_seeds*100:.1f}pp std dev")
        else:
            logger.info(f"✅ LOW VARIANCE: {std_score_across_seeds*100:.1f}pp std dev")
    
    # =======================
    # EXPERIMENT 4: COMPARE WITH ORIGINAL 53.05% FEATURES
    # =======================
    logger.info("\n🧪 EXPERIMENT 4: Compare with Original 53.05% Features")
    
    # From the original successful run, these were the selected features:
    original_53_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 
        'matchday_normalized', 'shots_diff_normalized', 
        'corners_diff_normalized', 'market_entropy_norm'
    ]
    
    logger.info(f"Original 53.05% features: {original_53_features}")
    
    # Test this exact feature combination with multiple seeds
    logger.info("Testing original feature combination with multiple seeds:")
    
    original_combo_results = {}
    
    for seed in test_seeds[:3]:  # Test with first 3 seeds
        X_original = df[original_53_features].fillna(0.5)
        
        try:
            xgb_model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, 
                random_state=seed,
                eval_metric='mlogloss', n_jobs=1
            )
            
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = cross_val_score(xgb_model, X_original, y, cv=tscv, scoring='accuracy')
            mean_score = cv_scores.mean()
            
            logger.info(f"  Seed {seed}: {mean_score:.4f}")
            
            if abs(mean_score - 0.5305) < 0.001:
                logger.info(f"    🎯 REPRODUCED 53.05% with original features + seed {seed}!")
            
            original_combo_results[seed] = mean_score
            
        except ImportError:
            continue
    
    # =======================
    # SCIENTIFIC CONCLUSION
    # =======================
    logger.info("\n=== 🔬 SCIENTIFIC CONCLUSION ===")
    
    if scores:
        # Determine the "true" performance range
        confidence_interval_95 = {
            'lower': mean_score_across_seeds - 2*std_score_across_seeds,
            'upper': mean_score_across_seeds + 2*std_score_across_seeds
        }
        
        logger.info(f"TRUE PERFORMANCE ESTIMATE:")
        logger.info(f"  Point Estimate: {mean_score_across_seeds:.4f}")
        logger.info(f"  95% Confidence Interval: [{confidence_interval_95['lower']:.4f}, {confidence_interval_95['upper']:.4f}]")
        
        # Scientific recommendation
        if 0.5305 <= confidence_interval_95['upper']:
            logger.info(f"✅ SCIENTIFIC VERDICT: 53.05% is within plausible range")
            recommendation = "USE_RANGE_ESTIMATE"
            stable_estimate = f"{confidence_interval_95['lower']:.3f} - {confidence_interval_95['upper']:.3f}"
        else:
            logger.info(f"❌ SCIENTIFIC VERDICT: 53.05% likely an outlier")
            recommendation = "USE_MEAN_ESTIMATE" 
            stable_estimate = f"{mean_score_across_seeds:.4f}"
        
        # Most stable feature set
        most_stable_features = [feat for feat, count in sorted_features if count >= len(results) * 0.8]
        logger.info(f"MOST STABLE FEATURES (80%+ selection): {most_stable_features}")
        
    else:
        recommendation = "INSUFFICIENT_DATA"
        stable_estimate = "UNKNOWN"
    
    logger.info("=== 🔬 SCIENTIFIC TEST COMPLETED ===")
    
    return {
        'performance_range': {
            'min': min(scores) if scores else None,
            'max': max(scores) if scores else None,
            'mean': mean_score_across_seeds if scores else None,
            'std': std_score_across_seeds if scores else None
        },
        'confidence_interval_95': confidence_interval_95 if scores else None,
        'stable_estimate': stable_estimate,
        'recommendation': recommendation,
        'most_stable_features': most_stable_features if scores else [],
        'detailed_results': results
    }

if __name__ == "__main__":
    result = scientific_reproducibility_test()
    
    print(f"\n🔬 SCIENTIFIC REPRODUCIBILITY RESULTS:")
    if result['performance_range']['min']:
        print(f"Performance Range: {result['performance_range']['min']:.4f} - {result['performance_range']['max']:.4f}")
        print(f"Mean ± Std: {result['performance_range']['mean']:.4f} ± {result['performance_range']['std']:.4f}")
        print(f"Stable Estimate: {result['stable_estimate']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Most Stable Features: {result['most_stable_features']}")
    else:
        print("No valid results obtained")