import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging, load_config

def finalize_v12_clean():
    """
    Create final v1.2 clean version:
    - Use 8 original features (no noisy new features)
    - Use Random Forest with optimized parameters (more stable than XGBoost)
    - Establish solid baseline for v2
    """
    
    logger = setup_logging()
    logger.info("=== v1.2 Clean Finalization ===")
    
    # Load base ML dataset
    input_file = "data/processed/premier_league_ml_ready.csv"
    logger.info(f"Loading clean dataset: {input_file}")
    
    df = pd.read_csv(input_file)
    logger.info(f"Dataset loaded: {df.shape}")
    
    # Define final v1.2 clean features (8 features, no redundancy, no noise)
    features_v12_clean = [
        'form_diff_normalized',
        'elo_diff_normalized', 
        'h2h_score',
        'home_advantage',
        'matchday_normalized',
        'season_period_numeric',
        'shots_diff_normalized',
        'corners_diff_normalized'
    ]
    
    logger.info(f"v1.2 Clean Features ({len(features_v12_clean)}):")
    for i, feature in enumerate(features_v12_clean, 1):
        logger.info(f"  {i}. {feature}")
    
    # Prepare data
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    X = df[features_v12_clean]
    
    logger.info(f"Final dataset: X{X.shape}, y{y.shape}")
    
    # Verify data quality
    logger.info("Data quality check:")
    logger.info(f"  Missing values: {X.isnull().sum().sum()}")
    logger.info(f"  Target distribution: {y.value_counts().sort_index().to_dict()}")
    
    for feature in features_v12_clean:
        min_val, max_val = X[feature].min(), X[feature].max()
        logger.info(f"  {feature}: [{min_val:.3f}, {max_val:.3f}]")
    
    # Define final Random Forest model (optimized parameters from hyperparameter tuning)
    rf_final = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        max_features='log2',
        min_samples_leaf=2,
        min_samples_split=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Final Random Forest Configuration:")
    for param, value in rf_final.get_params().items():
        if param in ['n_estimators', 'max_depth', 'max_features', 'min_samples_leaf', 'min_samples_split', 'class_weight']:
            logger.info(f"  {param}: {value}")
    
    # Perform final cross-validation
    logger.info("\\nPerforming final temporal cross-validation...")
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    cv_scores = cross_val_score(rf_final, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    logger.info("Final Cross-Validation Results:")
    logger.info(f"  CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
    logger.info(f"  Mean: {mean_score:.4f} (±{std_score:.4f})")
    
    # Train final model on full dataset
    logger.info("\\nTraining final model on complete dataset...")
    rf_final.fit(X, y)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features_v12_clean,
        'importance': rf_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Final Feature Importance:")
    for _, row in feature_importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save final model and results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    
    # Save model
    model_file = f"models/random_forest_v12_final_{timestamp}.joblib"
    joblib.dump(rf_final, model_file)
    logger.info(f"\\nFinal model saved: {model_file}")
    
    # Save feature configuration
    features_file = "config/features_v12_final.json"
    with open(features_file, 'w') as f:
        json.dump(features_v12_clean, f, indent=2)
    logger.info(f"Final features saved: {features_file}")
    
    # Save complete results
    results_file = f"models/v12_final_results_{timestamp}.json"
    
    results = {
        "version": "v1.2 Final Clean",
        "timestamp": timestamp,
        "model_type": "Random Forest (Optimized)",
        "features_count": len(features_v12_clean),
        "features": features_v12_clean,
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 12,
            "max_features": "log2",
            "min_samples_leaf": 2,
            "min_samples_split": 15,
            "class_weight": "balanced"
        },
        "performance": {
            "cv_mean": float(mean_score),
            "cv_std": float(std_score),
            "cv_scores": cv_scores.tolist()
        },
        "feature_importance": [
            {"feature": row['feature'], "importance": float(row['importance'])}
            for _, row in feature_importance.iterrows()
        ],
        "baselines_comparison": {
            "v1.1_baseline": 0.512,
            "improvement_vs_v11": float((mean_score - 0.512) * 100),
            "target_50_percent": 0.50,
            "beats_target": mean_score > 0.50
        },
        "data_quality": {
            "samples": len(X),
            "missing_values": int(X.isnull().sum().sum()),
            "target_distribution": {
                "Home": int(y[y==0].count()),
                "Draw": int(y[y==1].count()),
                "Away": int(y[y==2].count())
            }
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Complete results saved: {results_file}")
    
    # Performance summary
    logger.info("\\n" + "="*50)
    logger.info("v1.2 FINAL CLEAN SUMMARY")
    logger.info("="*50)
    logger.info(f"Model: Random Forest (Optimized)")
    logger.info(f"Features: {len(features_v12_clean)} clean features")
    logger.info(f"Performance: {mean_score:.3f}% (±{std_score:.3f}%)")
    
    improvement = (mean_score - 0.512) * 100
    logger.info(f"Improvement vs v1.1: {improvement:+.1f} percentage points")
    
    if mean_score > 0.52:
        logger.info("✅ EXCELLENT: Exceeds 52% target")
    elif mean_score > 0.50:
        logger.info("✅ GOOD: Exceeds 50% target")
    else:
        logger.info("⚠️ Needs improvement")
    
    logger.info("\\nReady for v2 development with solid baseline!")
    logger.info("="*50)
    
    return {
        'model_file': model_file,
        'features_file': features_file,
        'results_file': results_file,
        'final_score': mean_score,
        'improvement': improvement,
        'features_count': len(features_v12_clean)
    }

if __name__ == "__main__":
    result = finalize_v12_clean()
    print(f"\\nv1.2 Clean finalization completed!")
    print(f"Final performance: {result['final_score']:.3f}")
    print(f"Improvement vs v1.1: {result['improvement']:+.1f}pp")
    print(f"Model saved: {result['model_file']}")