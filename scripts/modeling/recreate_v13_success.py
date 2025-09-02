import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def recreate_v13_success():
    """
    Recreate the successful v1.3 combination that achieved 53.05%:
    v1.2 baseline (8 features) + market_entropy_norm = 9 features total
    """
    
    logger = setup_logging()
    logger.info("=== 🎯 RECREATING v1.3 SUCCESS (53.05%) ===")
    
    # Load the v2.0 dataset that has market_entropy_norm
    df = pd.read_csv("data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv")
    logger.info(f"Dataset loaded: {df.shape}")
    
    # Exact v1.3 successful feature set (from model_cleanup_v2.py results)
    v13_successful_features = [
        'form_diff_normalized', 
        'elo_diff_normalized', 
        'h2h_score',
        'matchday_normalized', 
        'season_period_numeric',
        'shots_diff_normalized', 
        'corners_diff_normalized',
        'market_entropy_norm'  # The one successful odds feature
    ]
    
    logger.info(f"v1.3 successful features ({len(v13_successful_features)}): {v13_successful_features}")
    
    # Prepare data
    X = df[v13_successful_features].fillna(0.5)
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts(normalize=True).sort_index().round(3).to_dict()}")
    
    # Cross-validation setup
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Exact model from successful run (XGBoost Conservative from model_cleanup_v2.py)
    logger.info("\n🎯 RECREATING SUCCESSFUL v1.3 MODEL")
    
    xgb_conservative = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='mlogloss', n_jobs=-1
    )
    
    cv_scores = cross_val_score(xgb_conservative, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    logger.info(f"XGBoost Conservative CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    logger.info(f"XGBoost Conservative Mean: {mean_score:.4f} (±{std_score:.4f})")
    
    # Test other models with same feature set
    logger.info("\n📊 TESTING OTHER MODELS WITH v1.3 FEATURES")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=12, max_features='log2',
            min_samples_leaf=2, min_samples_split=15, 
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost Enhanced': XGBClassifier(
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
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric='mlogloss', n_jobs=-1
            ))
        ], voting='hard')
    }
    
    best_model = None
    best_score = mean_score  # Start with XGBoost Conservative
    best_name = 'XGBoost Conservative'
    
    results = {
        'XGBoost Conservative': {
            'mean': float(mean_score),
            'std': float(std_score),
            'cv_scores': cv_scores.tolist()
        }
    }
    
    for model_name, model in models.items():
        cv_scores_model = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
        mean_model = cv_scores_model.mean()
        std_model = cv_scores_model.std()
        
        logger.info(f"{model_name}: {mean_model:.4f} (±{std_model:.4f})")
        
        if mean_model > best_score:
            best_score = mean_model
            best_model = model
            best_name = model_name
        
        results[model_name] = {
            'mean': float(mean_model),
            'std': float(std_model),
            'cv_scores': cv_scores_model.tolist()
        }
    
    # Assessment
    logger.info(f"\n=== v1.3 RECREATION RESULTS ===")
    logger.info(f"Best Model: {best_name}")
    logger.info(f"Best Score: {best_score:.4f}")
    
    target_53 = 0.530
    if best_score >= target_53:
        logger.info(f"✅ SUCCESS: Reproduced v1.3 performance ({best_score:.1%} ≥ {target_53:.1%})")
        status = "SUCCESS"
    else:
        gap = (target_53 - best_score) * 100
        logger.info(f"❌ SHORTFALL: {gap:.1f}pp below v1.3 target ({best_score:.1%} vs {target_53:.1%})")
        status = "SHORTFALL"
    
    # Now test strategic feature additions for v2.0 breakthrough
    if best_score >= 0.525:  # If we have a good baseline
        logger.info(f"\n🚀 TESTING v2.0 BREAKTHROUGH ADDITIONS")
        
        # Load advanced odds features for strategic addition
        df_advanced = pd.read_csv("data/processed/premier_league_v20_clean_2025_08_30_211505.csv")
        
        # Create match keys for merging
        df['match_key'] = df['Date'].astype(str) + '_' + df.index.astype(str)  # Simple key
        
        # Test strategic feature additions
        breakthrough_tests = {
            "v1.3 + Line Movement": v13_successful_features + ['line_movement_normalized'],
            "v1.3 + Sharp Money": v13_successful_features + ['sharp_public_divergence_norm'],
            "v1.3 + Time Weighted Form": v13_successful_features + ['time_weighted_form_diff_norm'],
            "v1.3 + Market Inefficiency": v13_successful_features + ['market_inefficiency_norm']
        }
        
        # For now, test with available features in current dataset
        available_tests = {
            "v1.3 + Sharp Money (proxy)": v13_successful_features + ['pinnacle_home_advantage_norm'],
            "v1.3 + Market Strength": v13_successful_features + ['market_home_advantage_norm']
        }
        
        for test_name, test_features in available_tests.items():
            available_features = [f for f in test_features if f in df.columns]
            if len(available_features) == len(test_features):
                X_test = df[test_features].fillna(0.5)
                
                # Use best model from above
                if best_name == 'XGBoost Conservative':
                    test_model = xgb_conservative
                else:
                    test_model = models[best_name]
                
                cv_scores_test = cross_val_score(test_model, X_test, y, cv=tscv, scoring='accuracy')
                mean_test = cv_scores_test.mean()
                
                improvement = (mean_test - best_score) * 100
                
                if mean_test >= 0.550:
                    breakthrough_status = "🎯 BREAKTHROUGH!"
                elif mean_test >= 0.540:
                    breakthrough_status = "🔥 EXCELLENT"
                elif improvement > 0:
                    breakthrough_status = f"✅ +{improvement:.1f}pp"
                else:
                    breakthrough_status = f"❌ {improvement:+.1f}pp"
                
                logger.info(f"{test_name}: {mean_test:.4f} {breakthrough_status}")
                
                if mean_test >= 0.550:
                    logger.info(f"🎯 v2.0 BREAKTHROUGH FOUND!")
                    
                    return {
                        'breakthrough_achieved': True,
                        'breakthrough_features': test_features,
                        'breakthrough_score': mean_test,
                        'best_model': best_name
                    }
    
    logger.info("=== v1.3 RECREATION COMPLETED ===")
    
    return {
        'v13_reproduced': best_score >= target_53,
        'best_score': best_score,
        'best_model': best_name,
        'status': status,
        'breakthrough_achieved': False,
        'all_results': results
    }

if __name__ == "__main__":
    result = recreate_v13_success()
    print(f"\n🎯 v1.3 RECREATION RESULTS:")
    print(f"v1.3 Reproduced: {'✅ YES' if result['v13_reproduced'] else '❌ NO'}")
    print(f"Best Score: {result['best_score']:.4f}")
    print(f"Best Model: {result['best_model']}")
    print(f"Status: {result['status']}")
    print(f"Breakthrough: {'🎯 YES' if result.get('breakthrough_achieved', False) else '❌ NO'}")