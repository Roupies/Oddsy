import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score
import joblib

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def cleanup_and_optimize_v2():
    """
    MODEL CLEANUP v2.0: Remove useless features, reduce overfitting, optimize for 53%+ target
    
    Cleanup Plan:
    1. Remove useless features (home_advantage = zero variance)
    2. Remove redundant odds features (0.9+ correlation)
    3. Test reduced complexity models
    4. Feature selection optimization
    5. Target 53%+ accuracy with clean model
    """
    
    logger = setup_logging()
    logger.info("=== 🧹 MODEL CLEANUP v2.0 - Optimization for 53%+ Target ===")
    
    # Load dataset
    df_file = "data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv"
    logger.info(f"Loading dataset: {df_file}")
    
    df = pd.read_csv(df_file)
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
    
    logger.info(f"Starting features: {len(all_features)} total")
    logger.info(f"Target: 53.0%+ accuracy")
    
    # =========================
    # 1. REMOVE USELESS FEATURES
    # =========================
    logger.info("\n🧹 1. REMOVING USELESS FEATURES")
    
    useless_features = []
    
    # Remove zero variance features
    for feature in all_features:
        if df[feature].std() < 0.001:
            useless_features.append(feature)
            logger.info(f"  🗑️ Removing {feature}: zero variance (std={df[feature].std():.6f})")
    
    # Remove redundant odds features (high correlation with baseline)
    correlation_threshold = 0.85
    X_temp = df[all_features]
    corr_matrix = X_temp.corr()
    
    redundant_features = []
    for odds_feat in odds_features:
        if odds_feat in useless_features:
            continue
            
        for baseline_feat in v12_features:
            if baseline_feat in useless_features:
                continue
                
            corr = abs(corr_matrix.loc[odds_feat, baseline_feat])
            if corr > correlation_threshold:
                redundant_features.append(odds_feat)
                logger.info(f"  🗑️ Removing {odds_feat}: high correlation with {baseline_feat} ({corr:.3f})")
                break
    
    # Also check for perfect correlation between odds features
    for i, feat1 in enumerate(odds_features):
        if feat1 in useless_features or feat1 in redundant_features:
            continue
        for feat2 in odds_features[i+1:]:
            if feat2 in useless_features or feat2 in redundant_features:
                continue
            corr = abs(corr_matrix.loc[feat1, feat2])
            if corr > 0.99:
                redundant_features.append(feat2)
                logger.info(f"  🗑️ Removing {feat2}: perfect correlation with {feat1} ({corr:.3f})")
    
    # Create clean feature set
    features_to_remove = list(set(useless_features + redundant_features))
    clean_features = [f for f in all_features if f not in features_to_remove]
    
    logger.info(f"  Removed features: {features_to_remove}")
    logger.info(f"  Clean features: {clean_features}")
    logger.info(f"  Feature reduction: {len(all_features)} → {len(clean_features)}")
    
    # =========================
    # 2. REDUCED COMPLEXITY MODELS
    # =========================
    logger.info("\n🔧 2. TESTING REDUCED COMPLEXITY MODELS")
    
    X_clean = df[clean_features]
    
    # Define models with reduced complexity to prevent overfitting
    models = {
        'Random Forest Conservative': RandomForestClassifier(
            n_estimators=100,    # Reduced from 300
            max_depth=8,         # Reduced from 12
            max_features='sqrt', # More conservative
            min_samples_leaf=5,  # Increased from 2
            min_samples_split=20, # Increased from 15
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Random Forest Moderate': RandomForestClassifier(
            n_estimators=200,    
            max_depth=10,        
            max_features='log2',
            min_samples_leaf=3,  
            min_samples_split=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost Conservative': XGBClassifier(
            n_estimators=100,    # Reduced from 200
            max_depth=3,         # Reduced from 4
            learning_rate=0.08,  # Slightly higher
            subsample=0.8,       # More conservative
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        ),
        'XGBoost Moderate': XGBClassifier(
            n_estimators=150,    
            max_depth=4,        
            learning_rate=0.06,  
            subsample=0.9,       
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )
    }
    
    # Cross-validation setup
    cv = TimeSeriesSplit(n_splits=5)
    
    model_results = {}
    
    logger.info("  Testing reduced complexity models:")
    for model_name, model in models.items():
        logger.info(f"\n    {model_name}:")
        
        cv_scores = cross_val_score(model, X_clean, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        # Check training accuracy to assess overfitting
        model.fit(X_clean, y)
        train_score = model.score(X_clean, y)
        overfitting_gap = train_score - mean_score
        
        logger.info(f"      CV: {mean_score:.4f} (±{std_score:.4f})")
        logger.info(f"      Train: {train_score:.4f}")
        logger.info(f"      Overfitting gap: {overfitting_gap:.4f}")
        
        # Performance vs target
        target_achievement = "✅ TARGET MET" if mean_score >= 0.530 else f"❌ -{(0.530-mean_score)*100:.1f}pp SHORT"
        logger.info(f"      vs 53%: {target_achievement}")
        
        model_results[model_name] = {
            'cv_mean': mean_score,
            'cv_std': std_score,
            'train_score': train_score,
            'overfitting_gap': overfitting_gap,
            'meets_target': mean_score >= 0.530
        }
    
    # =========================
    # 3. FEATURE SELECTION OPTIMIZATION
    # =========================
    logger.info("\n🎯 3. FEATURE SELECTION OPTIMIZATION")
    
    # Find best model from previous step
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['cv_mean'])
    best_model = models[best_model_name]
    best_score = model_results[best_model_name]['cv_mean']
    
    logger.info(f"  Best model so far: {best_model_name} ({best_score:.4f})")
    
    # Try different feature selection approaches if we haven't hit target
    if best_score < 0.530:
        logger.info("  Target not achieved, trying feature selection...")
        
        # Approach 1: Select K Best features
        feature_selection_results = {}
        
        for k in [4, 5, 6, 7]:
            if k >= len(clean_features):
                continue
                
            logger.info(f"\n    Trying top {k} features:")
            
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X_clean, y)
            
            selected_features = [clean_features[i] for i in selector.get_support(indices=True)]
            logger.info(f"      Selected: {selected_features}")
            
            cv_scores = cross_val_score(best_model, X_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)
            mean_score = cv_scores.mean()
            
            logger.info(f"      CV: {mean_score:.4f}")
            target_status = "✅" if mean_score >= 0.530 else "❌"
            logger.info(f"      Target: {target_status}")
            
            feature_selection_results[f'Top_{k}_features'] = {
                'features': selected_features,
                'cv_mean': mean_score,
                'meets_target': mean_score >= 0.530
            }
        
        # Approach 2: Recursive Feature Elimination
        logger.info(f"\n    Trying RFE feature selection:")
        
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=5)
        X_rfe = rfe.fit_transform(X_clean, y)
        
        rfe_features = [clean_features[i] for i in range(len(clean_features)) if rfe.support_[i]]
        logger.info(f"      RFE selected: {rfe_features}")
        
        cv_scores_rfe = cross_val_score(best_model, X_rfe, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_score_rfe = cv_scores_rfe.mean()
        
        logger.info(f"      RFE CV: {mean_score_rfe:.4f}")
        target_status = "✅" if mean_score_rfe >= 0.530 else "❌"
        logger.info(f"      Target: {target_status}")
        
        feature_selection_results['RFE'] = {
            'features': rfe_features,
            'cv_mean': mean_score_rfe,
            'meets_target': mean_score_rfe >= 0.530
        }
    
    # =========================
    # 4. ENSEMBLE WITH CLEAN FEATURES
    # =========================
    logger.info("\n🤝 4. ENSEMBLE WITH CLEAN FEATURES")
    
    # Create ensemble with best performing models
    top_models = sorted(model_results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:2]
    logger.info(f"  Creating ensemble from top 2 models: {[name for name, _ in top_models]}")
    
    ensemble = VotingClassifier([
        ('model1', models[top_models[0][0]]),
        ('model2', models[top_models[1][0]])
    ], voting='hard')
    
    cv_ensemble = cross_val_score(ensemble, X_clean, y, cv=cv, scoring='accuracy', n_jobs=-1)
    ensemble_mean = cv_ensemble.mean()
    ensemble_std = cv_ensemble.std()
    
    logger.info(f"  Ensemble CV: {ensemble_mean:.4f} (±{ensemble_std:.4f})")
    target_status = "✅ TARGET MET" if ensemble_mean >= 0.530 else f"❌ -{(0.530-ensemble_mean)*100:.1f}pp SHORT"
    logger.info(f"  vs 53%: {target_status}")
    
    # =========================
    # 5. FINAL RESULTS & RECOMMENDATIONS
    # =========================
    logger.info("\n🎯 5. CLEANUP RESULTS SUMMARY")
    
    # Find best overall result
    all_results = {
        **{name: data['cv_mean'] for name, data in model_results.items()},
        'Ensemble': ensemble_mean
    }
    
    if best_score < 0.530 and 'feature_selection_results' in locals():
        all_results.update({name: data['cv_mean'] for name, data in feature_selection_results.items()})
    
    best_approach = max(all_results.keys(), key=lambda x: all_results[x])
    best_final_score = all_results[best_approach]
    
    logger.info(f"  Best approach: {best_approach}")
    logger.info(f"  Best score: {best_final_score:.4f}")
    logger.info(f"  Target 53%: {'✅ ACHIEVED' if best_final_score >= 0.530 else '❌ NOT ACHIEVED'}")
    
    # Improvement analysis
    baseline_score = 0.5221  # From previous investigation
    improvement = (best_final_score - baseline_score) * 100
    
    logger.info(f"\n  Improvement Analysis:")
    logger.info(f"    Baseline v1.2: {baseline_score:.4f}")
    logger.info(f"    Best cleaned: {best_final_score:.4f}")
    logger.info(f"    Improvement: {improvement:+.2f}pp")
    
    # Create recommendations
    logger.info(f"\n🚀 RECOMMENDATIONS:")
    
    if best_final_score >= 0.530:
        recommendation = "SUCCESS_ACHIEVED"
        logger.info(f"  ✅ SUCCESS: Target 53%+ achieved with model cleanup!")
        logger.info(f"     - Use: {best_approach}")
        logger.info(f"     - Features removed: {len(features_to_remove)}")
        logger.info(f"     - Performance: {best_final_score:.4f}")
        
    elif improvement >= 0.5:
        recommendation = "SIGNIFICANT_IMPROVEMENT" 
        logger.info(f"  ⚠️ PROMISING: Significant improvement but target missed")
        logger.info(f"     - Best approach: {best_approach}")
        logger.info(f"     - Try: Advanced odds features next")
        logger.info(f"     - Gap to target: {(0.530-best_final_score)*100:.1f}pp")
        
    else:
        recommendation = "NEED_ADVANCED_FEATURES"
        logger.info(f"  ❌ INSUFFICIENT: Cleanup not enough, need advanced features")
        logger.info(f"     - Current gap: {(0.530-best_final_score)*100:.1f}pp")
        logger.info(f"     - Proceed to Option B: Advanced odds features")
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    
    # Save best model if target achieved
    if best_final_score >= 0.530:
        if best_approach == 'Ensemble':
            best_model_obj = ensemble
            best_features = clean_features
        elif best_approach in model_results:
            best_model_obj = models[best_approach]  
            best_features = clean_features
        elif 'feature_selection_results' in locals() and best_approach in feature_selection_results:
            best_model_obj = best_model
            best_features = feature_selection_results[best_approach]['features']
        else:
            best_model_obj = models[best_model_name]
            best_features = clean_features
        
        # Train and save
        X_final = df[best_features]
        best_model_obj.fit(X_final, y)
        
        model_file = f"models/cleaned_model_v2_{timestamp}.joblib"
        joblib.dump(best_model_obj, model_file)
        logger.info(f"  💾 Best model saved: {model_file}")
        
        # Save feature configuration
        feature_config = {
            "version": "v2.0_cleaned",
            "timestamp": timestamp,
            "best_approach": best_approach,
            "best_features": best_features,
            "removed_features": features_to_remove,
            "performance": float(best_final_score),
            "target_achieved": True
        }
        
        config_file = f"config/features_v2_cleaned_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(feature_config, f, indent=2)
        
        logger.info(f"  📝 Feature config saved: {config_file}")
    
    # Save complete results
    results_file = f"models/cleanup_results_v2_{timestamp}.json"
    
    results = {
        "version": "v2.0 Model Cleanup",
        "timestamp": timestamp,
        "features_removed": features_to_remove,
        "clean_features": clean_features,
        "best_approach": best_approach,
        "best_score": float(best_final_score),
        "target_achieved": bool(best_final_score >= 0.530),
        "improvement_pp": float(improvement),
        "recommendation": recommendation,
        "model_results": {k: {**v, 'cv_mean': float(v['cv_mean']), 'cv_std': float(v['cv_std']), 
                             'train_score': float(v['train_score']), 'overfitting_gap': float(v['overfitting_gap'])} 
                         for k, v in model_results.items()},
        "ensemble_performance": {
            "cv_mean": float(ensemble_mean),
            "cv_std": float(ensemble_std),
            "meets_target": bool(ensemble_mean >= 0.530)
        }
    }
    
    if 'feature_selection_results' in locals():
        results["feature_selection_results"] = feature_selection_results
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"  📊 Complete results saved: {results_file}")
    logger.info("=== 🧹 MODEL CLEANUP COMPLETED ===")
    
    return {
        'recommendation': recommendation,
        'best_approach': best_approach,
        'best_score': best_final_score,
        'target_achieved': best_final_score >= 0.530,
        'improvement_pp': improvement,
        'features_removed': features_to_remove,
        'clean_features': clean_features,
        'results_file': results_file
    }

if __name__ == "__main__":
    result = cleanup_and_optimize_v2()
    print(f"\n🎯 CLEANUP RESULTS:")
    print(f"Best Approach: {result['best_approach']}")
    print(f"Best Score: {result['best_score']:.4f}")
    print(f"Target 53%: {'✅ ACHIEVED' if result['target_achieved'] else '❌ MISSED'}")
    print(f"Improvement: {result['improvement_pp']:+.2f}pp")
    print(f"Features Removed: {len(result['features_removed'])}")
    print(f"Recommendation: {result['recommendation']}")