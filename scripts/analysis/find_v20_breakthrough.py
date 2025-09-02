import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def find_v20_breakthrough():
    """
    Strategic search for v2.0 breakthrough by testing optimal feature combinations
    """
    
    logger = setup_logging()
    logger.info("=== 🎯 STRATEGIC v2.0 BREAKTHROUGH SEARCH ===")
    
    # Load the successful v1.3 dataset
    df_v13 = pd.read_csv("data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv")
    logger.info(f"v1.3 successful dataset: {df_v13.shape}")
    
    # Load v2.0 clean features for advanced odds
    df_v20_clean = pd.read_csv("data/processed/premier_league_v20_clean_2025_08_30_211505.csv")
    logger.info(f"v2.0 clean dataset: {df_v20_clean.shape}")
    
    # Prepare targets
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_v13 = df_v13['FullTimeResult'].map(label_mapping)
    y_v20 = df_v20_clean['FullTimeResult'].map(label_mapping)
    
    # Cross-validation setup
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Enhanced models for breakthrough testing
    models = {
        'Random Forest Enhanced': RandomForestClassifier(
            n_estimators=500, max_depth=15, max_features='sqrt',
            min_samples_leaf=1, min_samples_split=5, 
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost Enhanced': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, random_state=42,
            eval_metric='mlogloss', n_jobs=-1
        )
    }
    
    # Strategic feature combinations to test
    feature_combinations = {
        "v1.3 Original (Successful)": [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'home_advantage', 'matchday_normalized', 'season_period_numeric',
            'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm'  # The successful odds feature
        ],
        "v1.3 + Core Odds (Minimal)": [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'home_advantage', 'matchday_normalized', 'season_period_numeric',
            'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'line_movement_normalized'
        ],
        "v1.3 + Sharp Money": [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'home_advantage', 'matchday_normalized', 'season_period_numeric',
            'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'sharp_public_divergence_norm'
        ],
        "Best Combined": [
            'elo_diff_normalized', 'form_diff_normalized',
            'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'line_movement_normalized',
            'sharp_public_divergence_norm', 'time_weighted_form_diff_norm'
        ]
    }
    
    results = {}
    best_combination = None
    best_score = 0
    
    logger.info("\n🔍 TESTING STRATEGIC FEATURE COMBINATIONS")
    
    for combo_name, features in feature_combinations.items():
        logger.info(f"\n🧪 Testing: {combo_name}")
        logger.info(f"Features ({len(features)}): {features}")
        
        # Determine which dataset to use
        if combo_name == "v1.3 Original (Successful)":
            X = df_v13[features].fillna(0.5)
            y = y_v13
        else:
            # For other combinations, we need to merge v1.3 baseline features with v2.0 advanced
            # Use v1.3 dataset as base and try to add features from it
            available_features = [f for f in features if f in df_v13.columns]
            if len(available_features) == len(features):
                X = df_v13[features].fillna(0.5)
                y = y_v13
            else:
                logger.warning(f"  Some features not available in v1.3 dataset - skipping")
                continue
        
        combo_results = {}
        
        for model_name, model in models.items():
            try:
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                # Performance assessment
                v13_baseline = 0.530
                v20_target = 0.550
                improvement = (mean_score - v13_baseline) * 100
                target_gap = (v20_target - mean_score) * 100
                
                if mean_score >= 0.550:
                    status = "🎯 BREAKTHROUGH!"
                elif mean_score >= 0.540:
                    status = "🔥 EXCELLENT"
                elif mean_score >= 0.535:
                    status = "⚡ VERY GOOD"
                elif mean_score >= v13_baseline:
                    status = "✅ GOOD"
                else:
                    status = "❌ BELOW BASELINE"
                
                logger.info(f"    {model_name}: {mean_score:.4f} (±{std_score:.4f}) {status}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_combination = f"{combo_name} + {model_name}"
                
                combo_results[model_name] = {
                    'mean': float(mean_score),
                    'std': float(std_score),
                    'improvement': float(improvement),
                    'target_gap': float(target_gap),
                    'status': status
                }
                
            except Exception as e:
                logger.error(f"    {model_name}: Error - {str(e)}")
        
        results[combo_name] = {
            'features': features,
            'feature_count': len(features),
            'results': combo_results
        }
    
    # Find the optimal baseline from v1.3
    logger.info("\n🎯 OPTIMAL BASELINE SEARCH")
    
    # Test v1.3 without least important features
    v13_core_features = [
        'elo_diff_normalized',        # Most important
        'market_entropy_norm',        # Best odds feature  
        'shots_diff_normalized',      # Strong performance feature
        'corners_diff_normalized',    # Strong performance feature
        'form_diff_normalized'        # Core strength feature
    ]
    
    logger.info(f"Testing v1.3 core features: {v13_core_features}")
    X_core = df_v13[v13_core_features].fillna(0.5)
    
    for model_name, model in models.items():
        cv_scores = cross_val_score(model, X_core, y_v13, cv=tscv, scoring='accuracy')
        mean_score = cv_scores.mean()
        
        if mean_score >= 0.550:
            status = "🎯 BREAKTHROUGH!"
        elif mean_score >= 0.540:
            status = "🔥 EXCELLENT"
        else:
            status = f"{mean_score:.4f}"
        
        logger.info(f"  Core {model_name}: {mean_score:.4f} {status}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_combination = f"v1.3 Core + {model_name}"
    
    # Test hyperparameter optimization on best combination
    logger.info(f"\n⚡ HYPERPARAMETER OPTIMIZATION")
    
    if best_score < 0.550:
        logger.info("Testing hyperparameter variations for breakthrough...")
        
        # Aggressive RandomForest
        rf_aggressive = RandomForestClassifier(
            n_estimators=800, max_depth=20, max_features='sqrt',
            min_samples_leaf=1, min_samples_split=2, 
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        )
        
        cv_scores = cross_val_score(rf_aggressive, X_core, y_v13, cv=tscv, scoring='accuracy')
        mean_aggressive = cv_scores.mean()
        
        logger.info(f"  Aggressive RF: {mean_aggressive:.4f}")
        
        if mean_aggressive > best_score:
            best_score = mean_aggressive
            best_combination = "v1.3 Core + Aggressive RF"
    
    # Final assessment
    logger.info(f"\n=== 🎯 BREAKTHROUGH ASSESSMENT ===")
    logger.info(f"Best Combination: {best_combination}")
    logger.info(f"Best Score: {best_score:.4f}")
    
    breakthrough_achieved = best_score >= 0.550
    
    if breakthrough_achieved:
        logger.info(f"🎯 v2.0 BREAKTHROUGH ACHIEVED! ({best_score:.1%})")
        recommendation = "PRODUCTION_READY"
    elif best_score >= 0.540:
        logger.info(f"🔥 EXCELLENT PROGRESS! ({best_score:.1%})")
        logger.info(f"Only {((0.550 - best_score) * 100):.1f}pp from breakthrough")
        recommendation = "FINAL_OPTIMIZATION"
    elif best_score >= 0.535:
        logger.info(f"⚡ VERY GOOD PROGRESS! ({best_score:.1%})")
        recommendation = "CONTINUE_FEATURE_ENGINEERING"
    else:
        logger.info(f"❌ INSUFFICIENT PROGRESS ({best_score:.1%})")
        recommendation = "REASSESS_APPROACH"
    
    logger.info(f"Recommendation: {recommendation}")
    logger.info("=== 🎯 BREAKTHROUGH SEARCH COMPLETED ===")
    
    return {
        'best_combination': best_combination,
        'best_score': best_score,
        'breakthrough_achieved': breakthrough_achieved,
        'recommendation': recommendation,
        'all_results': results
    }

if __name__ == "__main__":
    result = find_v20_breakthrough()
    print(f"\n🎯 BREAKTHROUGH SEARCH RESULTS:")
    print(f"Best: {result['best_combination']}")
    print(f"Score: {result['best_score']:.4f}")
    print(f"Breakthrough: {'🎯 YES' if result['breakthrough_achieved'] else '❌ NO'}")
    print(f"Recommendation: {result['recommendation']}")