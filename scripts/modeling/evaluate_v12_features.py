import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging, load_config

def evaluate_v12_features():
    """
    Test the impact of new v1.2 features on model performance.
    Compare 8 original features vs 11 enhanced features.
    """
    
    logger = setup_logging()
    logger.info("=== v1.2 Enhanced Features Evaluation ===")
    
    # Load enhanced dataset
    enhanced_files = [f for f in os.listdir('data/processed/') 
                     if f.startswith('premier_league_ml_ready_v12_') and f.endswith('.csv')]
    
    if not enhanced_files:
        raise FileNotFoundError("No v1.2 enhanced dataset found. Run enhance_features_v12.py first.")
    
    latest_enhanced = sorted(enhanced_files)[-1]
    enhanced_path = f"data/processed/{latest_enhanced}"
    
    logger.info(f"Loading enhanced dataset: {enhanced_path}")
    df_enhanced = pd.read_csv(enhanced_path)
    
    # Load feature configurations
    with open('config/features_v12.json', 'r') as f:
        features_v12 = json.load(f)
    
    # Original 8 features (v1.1)
    features_v11 = [
        'form_diff_normalized',
        'elo_diff_normalized', 
        'h2h_score',
        'home_advantage',
        'matchday_normalized',
        'season_period_numeric',
        'shots_diff_normalized',
        'corners_diff_normalized'
    ]
    
    logger.info(f"Features v1.1: {len(features_v11)}")
    logger.info(f"Features v1.2: {len(features_v12)}")
    logger.info(f"New features: {[f for f in features_v12 if f not in features_v11]}")
    
    # Prepare datasets
    # Encode target
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df_enhanced['FullTimeResult'].map(label_mapping)
    
    X_v11 = df_enhanced[features_v11]
    X_v12 = df_enhanced[features_v12]
    
    logger.info(f"Dataset shapes - v1.1: {X_v11.shape}, v1.2: {X_v12.shape}, target: {y.shape}")
    
    # Setup cross-validation
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Test configurations
    models = {
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
        'Random Forest Optimized': RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            max_features='log2',
            min_samples_leaf=2,
            min_samples_split=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"\\n--- Testing {model_name} ---")
        
        # Test v1.1 features (8 features)
        logger.info("Testing with v1.1 features (8 features)...")
        cv_scores_v11 = cross_val_score(model, X_v11, y, cv=tscv, scoring='accuracy', n_jobs=-1)
        mean_v11 = cv_scores_v11.mean()
        std_v11 = cv_scores_v11.std()
        
        logger.info(f"v1.1 Results: {mean_v11:.4f} (±{std_v11:.4f})")
        logger.info(f"  CV Scores: {[f'{s:.3f}' for s in cv_scores_v11]}")
        
        # Test v1.2 features (11 features)
        logger.info("Testing with v1.2 features (11 features)...")
        cv_scores_v12 = cross_val_score(model, X_v12, y, cv=tscv, scoring='accuracy', n_jobs=-1)
        mean_v12 = cv_scores_v12.mean()
        std_v12 = cv_scores_v12.std()
        
        logger.info(f"v1.2 Results: {mean_v12:.4f} (±{std_v12:.4f})")
        logger.info(f"  CV Scores: {[f'{s:.3f}' for s in cv_scores_v12]}")
        
        # Calculate improvement
        improvement = (mean_v12 - mean_v11) * 100
        logger.info(f"Feature Impact: {improvement:+.2f} percentage points")
        
        # Feature importance for v1.2
        model.fit(X_v12, y)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features_v12,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 5 Features (v1.2):")
            for _, row in feature_importance.head().iterrows():
                is_new = "🆕" if row['feature'] not in features_v11 else ""
                logger.info(f"  {row['feature']}: {row['importance']:.4f} {is_new}")
        
        results[model_name] = {
            'v11_score': float(mean_v11),
            'v11_std': float(std_v11),
            'v12_score': float(mean_v12),
            'v12_std': float(std_v12),
            'improvement': float(improvement),
            'cv_scores_v11': cv_scores_v11.tolist(),
            'cv_scores_v12': cv_scores_v12.tolist()
        }
    
    # Find best configuration
    best_config = max(results.keys(), 
                     key=lambda x: results[x]['v12_score'])
    best_score = results[best_config]['v12_score']
    
    logger.info(f"\\n=== v1.2 Feature Enhancement Results ===")
    logger.info("Performance Summary:")
    
    for model_name, result in results.items():
        logger.info(f"\\n{model_name}:")
        logger.info(f"  v1.1 (8 features):  {result['v11_score']:.4f} ± {result['v11_std']:.4f}")
        logger.info(f"  v1.2 (11 features): {result['v12_score']:.4f} ± {result['v12_std']:.4f}")
        logger.info(f"  Improvement: {result['improvement']:+.2f}pp")
    
    logger.info(f"\\nBest Configuration: {best_config} with {best_score:.4f}")
    
    # Historical comparison
    baselines = {
        'v1.1 Baseline': 0.512,
        'v1.2 Optimized (no new features)': 0.523
    }
    
    logger.info("\\nHistorical Comparison:")
    for baseline_name, baseline_score in baselines.items():
        improvement = (best_score - baseline_score) * 100
        logger.info(f"  vs {baseline_name}: {improvement:+.1f}pp")
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    results_file = f"models/v12_feature_evaluation_{timestamp}.json"
    
    evaluation_results = {
        "timestamp": timestamp,
        "best_config": best_config,
        "best_score": float(best_score),
        "features_v11_count": len(features_v11),
        "features_v12_count": len(features_v12),
        "new_features": [f for f in features_v12 if f not in features_v11],
        "models": results,
        "baselines": baselines
    }
    
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"\\nResults saved: {results_file}")
    
    # Determine if new features helped
    best_improvement = results[best_config]['improvement']
    
    if best_improvement > 0.5:
        logger.info(f"✅ New features provide significant improvement: +{best_improvement:.1f}pp")
        recommendation = "USE_V12"
    elif best_improvement > 0.0:
        logger.info(f"⚠️ New features provide marginal improvement: +{best_improvement:.1f}pp")
        recommendation = "USE_V12_CAUTIOUSLY"
    else:
        logger.info(f"❌ New features do not improve performance: {best_improvement:.1f}pp")
        recommendation = "STICK_WITH_V11"
    
    return {
        'best_config': best_config,
        'best_score': best_score,
        'improvement': best_improvement,
        'recommendation': recommendation,
        'results': results
    }

if __name__ == "__main__":
    result = evaluate_v12_features()
    print(f"v1.2 Feature evaluation completed.")
    print(f"Best: {result['best_config']} ({result['best_score']:.4f})")
    print(f"Improvement: {result['improvement']:+.2f}pp")
    print(f"Recommendation: {result['recommendation']}")