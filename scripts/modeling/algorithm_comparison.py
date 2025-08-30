import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging, load_config

def algorithm_comparison(config_path='config/config.json'):
    """
    Compare optimized Random Forest vs XGBoost performance.
    Uses TimeSeriesSplit for fair temporal comparison.
    """
    # Setup logging and load config
    logger = setup_logging(config_path)
    config = load_config(config_path)
    
    logger.info("=== v1.2 Algorithm Comparison: RF vs XGBoost ===")
    
    # Load the latest versioned data
    logger.info("Loading ML-ready data...")
    
    processed_dir = config['data']['output_dir']
    X_files = [f for f in os.listdir(processed_dir) if f.startswith('X_features_v') and f.endswith('.csv')]
    y_files = [f for f in os.listdir(processed_dir) if f.startswith('y_target_v') and f.endswith('.csv')]
    
    X_file = sorted(X_files)[-1]
    y_file = sorted(y_files)[-1]
    
    X = pd.read_csv(os.path.join(processed_dir, X_file))
    y_df = pd.read_csv(os.path.join(processed_dir, y_file))
    y = y_df['target']
    
    logger.info(f"Data loaded: X{X.shape}, y{y.shape}")
    
    # Setup cross-validation
    cv_folds = config['training']['cv_folds']
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Define algorithms with optimized parameters
    algorithms = {
        'Random Forest (Optimized)': RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            max_features='log2',
            min_samples_leaf=2,
            min_samples_split=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost (Tuned)': XGBClassifier(
            n_estimators=300,
            max_depth=6,  # XGBoost typically uses shallower trees
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',  # For multiclass
            n_jobs=-1
        ),
        'XGBoost (Conservative)': XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        logger.info(f"\\nTesting {name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            algorithm, X, y, 
            cv=tscv, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        logger.info(f"{name} Results:")
        logger.info(f"  CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
        logger.info(f"  Mean CV Score: {mean_score:.4f} (±{std_score:.4f})")
        
        # Train on full dataset for feature importance
        algorithm.fit(X, y)
        
        # Get feature importance (different methods for different algorithms)
        if hasattr(algorithm, 'feature_importances_'):
            importances = algorithm.feature_importances_
        else:
            importances = np.zeros(len(X.columns))  # Fallback
            
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"  Top 3 Features:")
        for _, row in feature_importance.head(3).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")
        
        results[name] = {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_score': float(mean_score),
            'std_cv_score': float(std_score),
            'feature_importance': [
                {"feature": row['feature'], "importance": float(row['importance'])}
                for _, row in feature_importance.iterrows()
            ]
        }
    
    # Find best algorithm
    best_algorithm = max(results.keys(), key=lambda x: results[x]['mean_cv_score'])
    best_score = results[best_algorithm]['mean_cv_score']
    
    logger.info(f"\\n=== Algorithm Comparison Results ===")
    logger.info("Performance Ranking:")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_cv_score'], reverse=True)
    for i, (name, result) in enumerate(sorted_results, 1):
        score = result['mean_cv_score']
        std = result['std_cv_score']
        logger.info(f"{i}. {name}: {score:.4f} (±{std:.4f})")
    
    # Compare against baselines
    baseline_v11 = 0.512
    baseline_v12_rf = 0.522  # From hyperparameter tuning
    
    logger.info(f"\\nComparison vs Baselines:")
    logger.info(f"v1.1 Baseline: {baseline_v11:.3f}")
    logger.info(f"v1.2 RF Optimized: {baseline_v12_rf:.3f}")
    logger.info(f"Best Algorithm ({best_algorithm}): {best_score:.3f}")
    
    improvement_v11 = (best_score - baseline_v11) * 100
    improvement_v12 = (best_score - baseline_v12_rf) * 100
    
    logger.info(f"Improvement vs v1.1: {improvement_v11:+.1f} percentage points")
    logger.info(f"Improvement vs v1.2 RF: {improvement_v12:+.1f} percentage points")
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    results_file = f"models/algorithm_comparison_{timestamp}.json"
    
    comparison_results = {
        "timestamp": timestamp,
        "best_algorithm": best_algorithm,
        "best_score": float(best_score),
        "baseline_v11": baseline_v11,
        "baseline_v12_rf": baseline_v12_rf,
        "improvement_vs_v11": float(improvement_v11),
        "improvement_vs_v12_rf": float(improvement_v12),
        "algorithms": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"\\nResults saved: {results_file}")
    
    # Train and save best model
    best_model_config = algorithms[best_algorithm]
    best_model_config.fit(X, y)
    
    model_file = f"models/best_algorithm_{timestamp}.joblib"
    joblib.dump(best_model_config, model_file)
    logger.info(f"Best model saved: {model_file}")
    
    if improvement_v12 > 0:
        logger.info(f"✅ {best_algorithm} outperforms RF by {improvement_v12:+.1f}pp!")
    else:
        logger.info(f"⚠️ Random Forest remains the best choice")
    
    return {
        'best_algorithm': best_algorithm,
        'best_score': best_score,
        'improvement_v11': improvement_v11,
        'improvement_v12': improvement_v12,
        'results': results
    }

if __name__ == "__main__":
    results = algorithm_comparison()
    print(f"Algorithm comparison completed. Best: {results['best_algorithm']} ({results['best_score']:.4f})")