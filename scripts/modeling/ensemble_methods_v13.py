import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging, load_config

def ensemble_methods_v13():
    """
    v1.3 Ensemble Methods: Combine multiple optimized models
    
    Tests:
    1. Individual optimized models (baseline)
    2. Voting Classifier (hard + soft voting)
    3. Stacking Classifier with meta-learner
    4. Weighted ensemble with custom weights
    """
    
    logger = setup_logging()
    logger.info("=== v1.3 Ensemble Methods Development ===")
    
    # Load clean v1.2 dataset
    input_file = "data/processed/premier_league_ml_ready.csv"
    logger.info(f"Loading dataset: {input_file}")
    
    df = pd.read_csv(input_file)
    
    # Use v1.2 final features
    features_v12 = [
        'form_diff_normalized',
        'elo_diff_normalized', 
        'h2h_score',
        'home_advantage',
        'matchday_normalized',
        'season_period_numeric',
        'shots_diff_normalized',
        'corners_diff_normalized'
    ]
    
    # Prepare data
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    X = df[features_v12]
    
    logger.info(f"Dataset prepared: X{X.shape}, y{y.shape}")
    logger.info(f"Features: {features_v12}")
    
    # Setup cross-validation
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Define individual optimized models
    models = {
        'Random Forest Optimized': RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            max_features='log2',
            min_samples_leaf=2,
            min_samples_split=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost Conservative': XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        ),
        'LightGBM Tuned': LGBMClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1  # Suppress output
        )
    }
    
    # Test individual models first
    logger.info("\\n=== Phase 1: Individual Model Performance ===")
    individual_results = {}
    
    for name, model in models.items():
        logger.info(f"\\nTesting {name}...")
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        logger.info(f"  CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
        logger.info(f"  Mean: {mean_score:.4f} (±{std_score:.4f})")
        
        individual_results[name] = {
            'cv_scores': cv_scores.tolist(),
            'mean': float(mean_score),
            'std': float(std_score)
        }
    
    # Create ensemble models
    logger.info("\\n=== Phase 2: Ensemble Models ===")
    
    # Voting Classifier (Hard Voting)
    logger.info("\\nTesting Voting Classifier (Hard Voting)...")
    voting_hard = VotingClassifier(
        estimators=[
            ('rf', models['Random Forest Optimized']),
            ('xgb', models['XGBoost Conservative']),
            ('lgb', models['LightGBM Tuned'])
        ],
        voting='hard'
    )
    
    cv_scores_voting_hard = cross_val_score(voting_hard, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
    mean_voting_hard = cv_scores_voting_hard.mean()
    std_voting_hard = cv_scores_voting_hard.std()
    
    logger.info(f"  CV Scores: {[f'{s:.3f}' for s in cv_scores_voting_hard]}")
    logger.info(f"  Mean: {mean_voting_hard:.4f} (±{std_voting_hard:.4f})")
    
    # Voting Classifier (Soft Voting)
    logger.info("\\nTesting Voting Classifier (Soft Voting)...")
    voting_soft = VotingClassifier(
        estimators=[
            ('rf', models['Random Forest Optimized']),
            ('xgb', models['XGBoost Conservative']),
            ('lgb', models['LightGBM Tuned'])
        ],
        voting='soft'
    )
    
    cv_scores_voting_soft = cross_val_score(voting_soft, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
    mean_voting_soft = cv_scores_voting_soft.mean()
    std_voting_soft = cv_scores_voting_soft.std()
    
    logger.info(f"  CV Scores: {[f'{s:.3f}' for s in cv_scores_voting_soft]}")
    logger.info(f"  Mean: {mean_voting_soft:.4f} (±{std_voting_soft:.4f})")
    
    # Stacking Classifier
    logger.info("\\nTesting Stacking Classifier...")
    stacking = StackingClassifier(
        estimators=[
            ('rf', models['Random Forest Optimized']),
            ('xgb', models['XGBoost Conservative']),
            ('lgb', models['LightGBM Tuned'])
        ],
        final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
        cv=3,  # Internal CV for stacking
        n_jobs=-1
    )
    
    cv_scores_stacking = cross_val_score(stacking, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
    mean_stacking = cv_scores_stacking.mean()
    std_stacking = cv_scores_stacking.std()
    
    logger.info(f"  CV Scores: {[f'{s:.3f}' for s in cv_scores_stacking]}")
    logger.info(f"  Mean: {mean_stacking:.4f} (±{std_stacking:.4f})")
    
    # Collect ensemble results
    ensemble_results = {
        'Voting Hard': {
            'cv_scores': cv_scores_voting_hard.tolist(),
            'mean': float(mean_voting_hard),
            'std': float(std_voting_hard)
        },
        'Voting Soft': {
            'cv_scores': cv_scores_voting_soft.tolist(),
            'mean': float(mean_voting_soft),
            'std': float(std_voting_soft)
        },
        'Stacking': {
            'cv_scores': cv_scores_stacking.tolist(),
            'mean': float(mean_stacking),
            'std': float(std_stacking)
        }
    }
    
    # Find best model overall
    all_results = {**individual_results, **ensemble_results}
    best_model_name = max(all_results.keys(), key=lambda x: all_results[x]['mean'])
    best_score = all_results[best_model_name]['mean']
    
    # Performance comparison
    logger.info("\\n=== Phase 3: Performance Comparison ===")
    logger.info("Individual Models:")
    for name, result in individual_results.items():
        logger.info(f"  {name}: {result['mean']:.4f} (±{result['std']:.4f})")
    
    logger.info("\\nEnsemble Models:")
    for name, result in ensemble_results.items():
        logger.info(f"  {name}: {result['mean']:.4f} (±{result['std']:.4f})")
    
    logger.info(f"\\nBest Model: {best_model_name} with {best_score:.4f}")
    
    # Compare against baselines
    baseline_v12 = 0.522  # v1.2 baseline
    improvement = (best_score - baseline_v12) * 100
    
    logger.info(f"\\nBaseline Comparison:")
    logger.info(f"  v1.2 Baseline: {baseline_v12:.3f}")
    logger.info(f"  v1.3 Best: {best_score:.3f}")
    logger.info(f"  Improvement: {improvement:+.1f} percentage points")
    
    # Train and save best model
    if best_model_name in ensemble_results:
        # It's an ensemble model
        if best_model_name == 'Voting Hard':
            best_model = voting_hard
        elif best_model_name == 'Voting Soft':
            best_model = voting_soft
        else:  # Stacking
            best_model = stacking
    else:
        # It's an individual model
        best_model = models[best_model_name]
    
    logger.info(f"\\nTraining best model ({best_model_name}) on full dataset...")
    best_model.fit(X, y)
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    
    # Save best model
    model_file = f"models/ensemble_v13_best_{timestamp}.joblib"
    joblib.dump(best_model, model_file)
    logger.info(f"Best model saved: {model_file}")
    
    # Save complete results
    results_file = f"models/ensemble_v13_results_{timestamp}.json"
    
    results = {
        "version": "v1.3 Ensemble Methods",
        "timestamp": timestamp,
        "best_model": best_model_name,
        "best_score": float(best_score),
        "baseline_v12": baseline_v12,
        "improvement": float(improvement),
        "individual_models": individual_results,
        "ensemble_models": ensemble_results,
        "features": features_v12,
        "data_info": {
            "samples": len(X),
            "features": len(features_v12)
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved: {results_file}")
    
    # Final summary
    logger.info("\\n" + "="*60)
    logger.info("v1.3 ENSEMBLE METHODS SUMMARY")
    logger.info("="*60)
    logger.info(f"Best Configuration: {best_model_name}")
    logger.info(f"Performance: {best_score:.3f} (±{all_results[best_model_name]['std']:.3f})")
    logger.info(f"Improvement vs v1.2: {improvement:+.1f}pp")
    
    if improvement > 0.5:
        logger.info("✅ SIGNIFICANT IMPROVEMENT: Ensemble methods successful!")
    elif improvement > 0.0:
        logger.info("⚠️ MARGINAL IMPROVEMENT: Consider individual model")
    else:
        logger.info("❌ NO IMPROVEMENT: Stick with v1.2 baseline")
    
    logger.info("="*60)
    
    return {
        'best_model_name': best_model_name,
        'best_score': best_score,
        'improvement': improvement,
        'model_file': model_file,
        'results_file': results_file,
        'all_results': all_results
    }

if __name__ == "__main__":
    result = ensemble_methods_v13()
    print(f"\\nv1.3 Ensemble Methods completed!")
    print(f"Best: {result['best_model_name']} ({result['best_score']:.4f})")
    print(f"Improvement: {result['improvement']:+.1f}pp vs v1.2")