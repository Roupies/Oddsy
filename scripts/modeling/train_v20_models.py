import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def train_v20_models():
    """
    Train v2.0 models with odds integration.
    Compare v1.2 baseline (8 features) vs v2.0 enhanced (11 features).
    
    Target: 53%+ accuracy to prove odds value
    """
    
    logger = setup_logging()
    logger.info("=== v2.0 Model Training & Validation ===")
    
    # Load v2.0 enhanced dataset
    v20_file = "data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv"
    logger.info(f"Loading v2.0 dataset: {v20_file}")
    
    df = pd.read_csv(v20_file)
    logger.info(f"Dataset loaded: {df.shape}")
    
    # Feature definitions
    v12_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'home_advantage', 'matchday_normalized', 'season_period_numeric',
        'shots_diff_normalized', 'corners_diff_normalized'
    ]
    
    odds_features = [
        'market_home_advantage_norm', 'market_entropy_norm', 'pinnacle_home_advantage_norm'
    ]
    
    v20_features = v12_features + odds_features
    
    # Prepare target
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Features defined:")
    logger.info(f"  v1.2 baseline ({len(v12_features)}): {v12_features}")
    logger.info(f"  Odds features ({len(odds_features)}): {odds_features}")
    logger.info(f"  v2.0 total ({len(v20_features)}): {len(v20_features)}")
    logger.info(f"Target distribution: {y.value_counts(normalize=True).sort_index().round(3).to_dict()}")
    
    # Cross-validation setup
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Model definitions
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=12, max_features='log2',
            min_samples_leaf=2, min_samples_split=15, 
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            eval_metric='mlogloss', n_jobs=-1
        ),
        'Voting Ensemble': VotingClassifier([
            ('rf', RandomForestClassifier(
                n_estimators=300, max_depth=12, max_features='log2',
                min_samples_leaf=2, min_samples_split=15,
                class_weight='balanced', random_state=42, n_jobs=-1
            )),
            ('xgb', XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                eval_metric='mlogloss', n_jobs=-1
            ))
        ], voting='hard')
    }
    
    # Test configurations
    test_configs = {
        'v1.2 Baseline (8 features)': v12_features,
        'v2.0 Enhanced (11 features)': v20_features
    }
    
    results = {}
    
    # Test all configurations
    logger.info("\n=== Model Training & Cross-Validation ===")
    
    for config_name, features in test_configs.items():
        logger.info(f"\nTesting: {config_name}")
        logger.info(f"Features: {features}")
        
        X = df[features]
        logger.info(f"Feature matrix: {X.shape}")
        
        # Verify no missing values
        if X.isnull().sum().sum() > 0:
            logger.error(f"Missing values detected in {config_name}!")
            continue
        
        config_results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n  Training {model_name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            logger.info(f"    CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
            logger.info(f"    Mean: {mean_score:.4f} (±{std_score:.4f})")
            
            # Performance evaluation
            baseline_v12 = 0.522  # v1.2 baseline from results
            improvement = (mean_score - baseline_v12) * 100
            
            if mean_score >= 0.530:
                status = "✅ TARGET ACHIEVED"
            elif mean_score >= 0.525:
                status = "⚠️ CLOSE TO TARGET"
            else:
                status = "❌ BELOW TARGET"
            
            logger.info(f"    vs v1.2 baseline: {improvement:+.1f}pp ({status})")
            
            config_results[model_name] = {
                'cv_scores': cv_scores.tolist(),
                'mean': float(mean_score),
                'std': float(std_score),
                'improvement_pp': float(improvement),
                'status': status
            }
        
        results[config_name] = {
            'features': features,
            'feature_count': len(features),
            'results': config_results
        }
    
    # Find best configuration
    best_config = None
    best_model = None
    best_score = 0
    
    logger.info("\n=== Performance Comparison ===")
    
    for config_name, config_data in results.items():
        logger.info(f"\n{config_name}:")
        
        for model_name, model_results in config_data['results'].items():
            score = model_results['mean']
            improvement = model_results['improvement_pp']
            status = model_results['status']
            
            logger.info(f"  {model_name}: {score:.4f} ({improvement:+.1f}pp) {status}")
            
            if score > best_score:
                best_score = score
                best_config = config_name
                best_model = model_name
    
    # Detailed comparison
    if 'v1.2 Baseline (8 features)' in results and 'v2.0 Enhanced (11 features)' in results:
        logger.info(f"\n=== v1.2 vs v2.0 Impact Analysis ===")
        
        v12_results = results['v1.2 Baseline (8 features)']['results']
        v20_results = results['v2.0 Enhanced (11 features)']['results']
        
        for model_name in models.keys():
            if model_name in v12_results and model_name in v20_results:
                v12_score = v12_results[model_name]['mean']
                v20_score = v20_results[model_name]['mean']
                odds_impact = (v20_score - v12_score) * 100
                
                logger.info(f"{model_name}:")
                logger.info(f"  v1.2: {v12_score:.4f}")
                logger.info(f"  v2.0: {v20_score:.4f}")
                logger.info(f"  Odds Impact: {odds_impact:+.2f}pp")
                
                if odds_impact >= 1.0:
                    logger.info(f"  ✅ SIGNIFICANT ODDS BOOST")
                elif odds_impact >= 0.5:
                    logger.info(f"  ⚠️ MODERATE ODDS BOOST")
                elif odds_impact >= 0.0:
                    logger.info(f"  🔸 MARGINAL ODDS BOOST")
                else:
                    logger.info(f"  ❌ ODDS DETRIMENTAL")
    
    # Success evaluation
    target_achieved = best_score >= 0.530
    logger.info(f"\n=== v2.0 Success Evaluation ===")
    logger.info(f"Best Model: {best_config} + {best_model}")
    logger.info(f"Best Score: {best_score:.4f}")
    logger.info(f"Target 53.0%: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
    
    if target_achieved:
        logger.info(f"🎯 v2.0 SUCCESS: Odds integration proves valuable!")
        recommendation = "PROCEED_TO_ADVANCED_FEATURES"
    elif best_score >= 0.525:
        logger.info(f"⚠️ v2.0 PROMISING: Close to target, try advanced features")
        recommendation = "TRY_ADVANCED_FEATURES"
    else:
        logger.info(f"❌ v2.0 INSUFFICIENT: Odds not adding enough value")
        recommendation = "REVIEW_ODDS_FEATURES"
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    results_file = f"models/v20_training_results_{timestamp}.json"
    
    final_results = {
        "version": "v2.0 Odds Integration",
        "timestamp": timestamp,
        "best_config": best_config,
        "best_model": best_model,
        "best_score": float(best_score),
        "target_achieved": target_achieved,
        "recommendation": recommendation,
        "baseline_comparison": {
            "v12_baseline": 0.522,
            "improvement_pp": float((best_score - 0.522) * 100)
        },
        "feature_analysis": {
            "v12_features": v12_features,
            "odds_features": odds_features,
            "v20_features": v20_features
        },
        "detailed_results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Results saved: {results_file}")
    logger.info("=== v2.0 Training Completed ===")
    
    return {
        'best_config': best_config,
        'best_model': best_model,
        'best_score': best_score,
        'target_achieved': target_achieved,
        'recommendation': recommendation,
        'results_file': results_file
    }

if __name__ == "__main__":
    result = train_v20_models()
    print(f"\nv2.0 Training Results:")
    print(f"Best: {result['best_config']} + {result['best_model']}")
    print(f"Score: {result['best_score']:.4f}")
    print(f"Target 53%: {'✅ ACHIEVED' if result['target_achieved'] else '❌ FAILED'}")
    print(f"Recommendation: {result['recommendation']}")