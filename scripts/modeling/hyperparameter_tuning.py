import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging, load_config

def hyperparameter_tuning(config_path='config/config.json'):
    """
    Comprehensive hyperparameter tuning for Random Forest using GridSearchCV.
    Uses TimeSeriesSplit to maintain temporal integrity.
    """
    # Setup logging and load config
    logger = setup_logging(config_path)
    config = load_config(config_path)
    
    logger.info("=== v1.2 Hyperparameter Tuning Started ===")
    
    # Load the latest versioned data
    logger.info("Loading ML-ready data...")
    
    # Find the latest versioned files
    processed_dir = config['data']['output_dir']
    X_files = [f for f in os.listdir(processed_dir) if f.startswith('X_features_v') and f.endswith('.csv')]
    y_files = [f for f in os.listdir(processed_dir) if f.startswith('y_target_v') and f.endswith('.csv')]
    
    if not X_files or not y_files:
        raise FileNotFoundError("No versioned ML data found. Run prepare_ml_data.py first.")
    
    # Get latest versions
    X_file = sorted(X_files)[-1]
    y_file = sorted(y_files)[-1]
    
    logger.info(f"Loading features from: {X_file}")
    logger.info(f"Loading target from: {y_file}")
    
    X = pd.read_csv(os.path.join(processed_dir, X_file))
    y_df = pd.read_csv(os.path.join(processed_dir, y_file))
    y = y_df['target']
    
    logger.info(f"Data loaded: X{X.shape}, y{y.shape}")
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 10, 12, 15],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 5, 8],
        'max_features': ['sqrt', 'log2', 0.8],
        'class_weight': ['balanced']  # Keep balanced for class imbalance
    }
    
    logger.info("Parameter grid defined:")
    for param, values in param_grid.items():
        logger.info(f"  {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    logger.info(f"Total parameter combinations: {total_combinations}")
    
    # Setup cross-validation (TimeSeriesSplit for temporal data)
    cv_folds = config['training']['cv_folds']
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    logger.info(f"Using TimeSeriesSplit with {cv_folds} folds")
    
    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Setup GridSearchCV
    logger.info("Starting GridSearchCV (this may take several minutes)...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,  # Use all CPU cores
        verbose=1,  # Show progress
        return_train_score=True
    )
    
    # Perform hyperparameter search
    grid_search.fit(X, y)
    
    logger.info("GridSearchCV completed!")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    logger.info("Best parameters:")
    for param, value in grid_search.best_params_.items():
        logger.info(f"  {param}: {value}")
    
    # Get detailed results
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters...")
    best_rf = grid_search.best_estimator_
    best_rf.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 5 most important features (optimized):")
    for _, row in feature_importance.head().iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    
    # Save optimized model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_file = f"{model_dir}/random_forest_optimized_{timestamp}.joblib"
    joblib.dump(best_rf, model_file)
    logger.info(f"Optimized model saved: {model_file}")
    
    # Save hyperparameter tuning results
    results_file = f"{model_dir}/hyperparameter_results_{timestamp}.json"
    tuning_results = {
        "best_score": float(grid_search.best_score_),
        "best_params": grid_search.best_params_,
        "baseline_score": 0.512,  # v1.1 baseline
        "improvement": float(grid_search.best_score_ - 0.512),
        "total_combinations_tested": total_combinations,
        "cv_folds": cv_folds,
        "timestamp": timestamp,
        "feature_importance": [
            {"feature": row['feature'], "importance": float(row['importance'])}
            for _, row in feature_importance.iterrows()
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    logger.info(f"Hyperparameter results saved: {results_file}")
    
    # Save detailed CV results
    cv_results_file = f"{model_dir}/cv_results_{timestamp}.csv"
    results_df.to_csv(cv_results_file, index=False)
    logger.info(f"Detailed CV results saved: {cv_results_file}")
    
    # Performance summary
    logger.info("=== v1.2 Hyperparameter Tuning Summary ===")
    logger.info(f"v1.1 Baseline: 51.2%")
    logger.info(f"v1.2 Optimized: {grid_search.best_score_*100:.1f}%")
    improvement = (grid_search.best_score_ - 0.512) * 100
    logger.info(f"Improvement: {improvement:+.1f} percentage points")
    
    if improvement > 0:
        logger.info("✅ Hyperparameter tuning successful!")
    else:
        logger.info("⚠️ No improvement from hyperparameter tuning")
    
    return {
        'best_model': best_rf,
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'improvement': improvement,
        'results_file': results_file
    }

if __name__ == "__main__":
    results = hyperparameter_tuning()
    print(f"Hyperparameter tuning completed. Best score: {results['best_score']:.4f}")