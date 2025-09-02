import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import joblib

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def train_v20_breakthrough():
    """
    Train v2.0 BREAKTHROUGH models with advanced feature engineering.
    
    Test the complete Phase 2.0.1 + 2.0.2 feature set:
    - 7 core features (Elo, form, advanced odds)  
    - 5 advanced features (interactions, time-decay, volatility)
    
    TARGET: True v2.0 breakthrough to 55%+ accuracy
    """
    
    logger = setup_logging()
    logger.info("=== 🚀 v2.0 BREAKTHROUGH MODEL TRAINING ===")
    
    # Load v2.0 advanced dataset
    v20_file = "data/processed/premier_league_v20_advanced_2025_08_30_210808.csv"
    logger.info(f"Loading v2.0 breakthrough dataset: {v20_file}")
    
    df = pd.read_csv(v20_file)
    df['Date'] = pd.to_datetime(df['Date'])
    logger.info(f"Dataset loaded: {df.shape}")
    
    # Feature definitions
    selected_features = [
        'elo_diff_normalized', 'form_diff_normalized', 
        'line_movement_normalized', 'sharp_public_divergence_norm', 
        'market_inefficiency_norm', 'market_velocity_norm',
        'elo_market_interaction_norm', 'form_sharp_interaction_norm',
        'line_h2h_interaction_norm', 'velocity_elo_interaction_norm',
        'time_weighted_form_diff_norm', 'volatility_diff_norm'
    ]
    
    # Prepare features and target
    X = df[selected_features].copy()
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    
    logger.info(f"v2.0 feature set ({len(selected_features)} features):")
    for i, feature in enumerate(selected_features):
        logger.info(f"  {i+1:2d}. {feature}")
    
    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts(normalize=True).sort_index().round(3).to_dict()}")
    
    # Verify no missing values
    if X.isnull().sum().sum() > 0:
        logger.error("Missing values detected!")
        return None
    
    # Cross-validation setup
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Enhanced model definitions for v2.0
    models = {
        'Random Forest Enhanced': RandomForestClassifier(
            n_estimators=500, max_depth=15, max_features='sqrt',
            min_samples_leaf=1, min_samples_split=10, 
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost Enhanced': XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, random_state=42,
            eval_metric='mlogloss', n_jobs=-1
        ),
        'Logistic Regression L2': LogisticRegression(
            C=1.0, penalty='l2', max_iter=1000, 
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Voting Ensemble Pro': VotingClassifier([
            ('rf', RandomForestClassifier(
                n_estimators=400, max_depth=15, max_features='sqrt',
                min_samples_leaf=1, min_samples_split=10,
                class_weight='balanced', random_state=42, n_jobs=-1
            )),
            ('xgb', XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.03,
                subsample=0.85, colsample_bytree=0.85, random_state=42,
                eval_metric='mlogloss', n_jobs=-1
            )),
            ('lr', LogisticRegression(
                C=1.0, penalty='l2', max_iter=1000,
                class_weight='balanced', random_state=42, n_jobs=-1
            ))
        ], voting='soft')  # Use soft voting for probability averaging
    }
    
    results = {}
    best_model_name = None
    best_score = 0
    
    # Train and evaluate all models
    logger.info("\n=== v2.0 BREAKTHROUGH MODEL TRAINING ===")
    
    for model_name, model in models.items():
        logger.info(f"\n🚀 Training {model_name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        logger.info(f"    CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
        logger.info(f"    Mean: {mean_score:.4f} (±{std_score:.4f})")
        
        # Performance evaluation against targets
        v13_baseline = 0.530  # v1.3 achievement
        target_v20 = 0.550   # v2.0 breakthrough target
        
        improvement_v13 = (mean_score - v13_baseline) * 100
        target_gap = (target_v20 - mean_score) * 100
        
        # Status determination
        if mean_score >= 0.550:
            status = "🎯 v2.0 BREAKTHROUGH ACHIEVED!"
            color = "🟢"
        elif mean_score >= 0.540:
            status = "🔥 EXCELLENT - Close to breakthrough"
            color = "🟡"
        elif mean_score >= 0.535:
            status = "⚡ VERY GOOD - Strong progress"
            color = "🟡"  
        elif mean_score >= v13_baseline:
            status = "✅ GOOD - Beats v1.3 baseline"
            color = "🟢"
        else:
            status = "❌ BELOW v1.3 BASELINE"
            color = "🔴"
        
        logger.info(f"    vs v1.3 baseline: {improvement_v13:+.1f}pp")
        logger.info(f"    Gap to v2.0 target: {target_gap:+.1f}pp")
        logger.info(f"    Status: {color} {status}")
        
        # Track best model
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = model_name
        
        # Store results
        results[model_name] = {
            'cv_scores': cv_scores.tolist(),
            'mean': float(mean_score),
            'std': float(std_score),
            'improvement_v13_pp': float(improvement_v13),
            'target_gap_pp': float(target_gap),
            'status': status,
            'breakthrough_achieved': bool(mean_score >= 0.550)
        }
        
        # Log-loss for probability quality (if model supports predict_proba)
        if hasattr(model, 'predict_proba'):
            try:
                model.fit(X, y)
                y_prob = model.predict_proba(X)
                logloss = log_loss(y, y_prob)
                logger.info(f"    Log-Loss: {logloss:.4f} (lower is better)")
                results[model_name]['log_loss'] = float(logloss)
            except:
                pass
    
    # Overall v2.0 assessment
    logger.info(f"\n=== 🎯 v2.0 BREAKTHROUGH ASSESSMENT ===")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Best Score: {best_score:.4f}")
    
    breakthrough_achieved = best_score >= 0.550
    
    if breakthrough_achieved:
        logger.info(f"🎯 v2.0 BREAKTHROUGH ACHIEVED! ({best_score:.1%})")
        logger.info(f"🚀 SUCCESS: Advanced feature engineering delivers breakthrough performance!")
        recommendation = "PRODUCTION_READY"
        success_level = "BREAKTHROUGH"
    elif best_score >= 0.540:
        logger.info(f"🔥 EXCELLENT PERFORMANCE! ({best_score:.1%})")
        logger.info(f"⚡ VERY CLOSE: Only {((0.550 - best_score) * 100):.1f}pp from breakthrough")
        recommendation = "FINE_TUNE_HYPERPARAMETERS"
        success_level = "EXCELLENT"
    elif best_score >= 0.535:
        logger.info(f"⚡ VERY GOOD PROGRESS! ({best_score:.1%})")
        logger.info(f"✅ SIGNIFICANT: {((best_score - 0.530) * 100):.1f}pp improvement over v1.3")
        recommendation = "EXPLORE_ADDITIONAL_FEATURES"
        success_level = "VERY_GOOD"
    elif best_score >= 0.530:
        logger.info(f"✅ MODERATE SUCCESS ({best_score:.1%})")
        logger.info(f"🔸 PROGRESS: Advanced features show improvement over v1.3")
        recommendation = "FEATURE_INTERACTION_ANALYSIS"
        success_level = "MODERATE"
    else:
        logger.info(f"❌ BELOW v1.3 BASELINE ({best_score:.1%})")
        logger.info(f"⚠️ REGRESSION: Feature complexity may be hurting performance")
        recommendation = "FEATURE_SELECTION_REVIEW"
        success_level = "REGRESSION"
    
    # Feature importance analysis (using best Random Forest model)
    logger.info(f"\n📊 FEATURE IMPORTANCE ANALYSIS")
    
    rf_model = RandomForestClassifier(
        n_estimators=500, max_depth=15, max_features='sqrt',
        min_samples_leaf=1, min_samples_split=10, 
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_model.fit(X, y)
    
    feature_importance = list(zip(selected_features, rf_model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("Top 10 most important features:")
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        logger.info(f"  {i+1:2d}. {feature:<35}: {importance:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    results_file = f"models/v20_breakthrough_results_{timestamp}.json"
    
    final_results = {
        "version": "v2.0 BREAKTHROUGH",
        "timestamp": timestamp,
        "phases_completed": ["2.0.1 Advanced Odds", "2.0.2 Advanced Engineering"],
        "best_model": best_model_name,
        "best_score": float(best_score),
        "breakthrough_achieved": bool(breakthrough_achieved),
        "success_level": success_level,
        "recommendation": recommendation,
        "performance_benchmarks": {
            "v13_baseline": 0.530,
            "v20_target": 0.550,
            "best_achieved": float(best_score),
            "improvement_v13_pp": float((best_score - 0.530) * 100),
            "target_gap_pp": float((0.550 - best_score) * 100)
        },
        "feature_analysis": {
            "total_features": len(selected_features),
            "selected_features": selected_features,
            "top_10_importance": [{"feature": f, "importance": float(i)} 
                                for f, i in feature_importance[:10]]
        },
        "detailed_results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Results saved: {results_file}")
    
    # Save best model if breakthrough achieved
    if breakthrough_achieved:
        best_model = models[best_model_name]
        best_model.fit(X, y)
        
        model_file = f"models/v20_breakthrough_{best_model_name.lower().replace(' ', '_')}_{timestamp}.joblib"
        joblib.dump(best_model, model_file)
        logger.info(f"Breakthrough model saved: {model_file}")
        
        final_results['saved_model'] = model_file
    
    logger.info("=== 🚀 v2.0 BREAKTHROUGH TRAINING COMPLETED ===")
    
    return {
        'best_model': best_model_name,
        'best_score': best_score,
        'breakthrough_achieved': breakthrough_achieved,
        'success_level': success_level,
        'recommendation': recommendation,
        'results_file': results_file,
        'feature_count': len(selected_features)
    }

if __name__ == "__main__":
    result = train_v20_breakthrough()
    
    if result:
        print(f"\n🎯 v2.0 BREAKTHROUGH RESULTS:")
        print(f"Best Model: {result['best_model']}")
        print(f"Best Score: {result['best_score']:.4f}")
        print(f"Success Level: {result['success_level']}")
        print(f"Breakthrough: {'🎯 YES' if result['breakthrough_achieved'] else '❌ NO'}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Features Used: {result['feature_count']}")
    else:
        print("❌ Training failed")