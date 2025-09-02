import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def deploy_v13_stable():
    """
    Deploy v1.3 as stable production model (53.05% proven performance)
    
    This creates the final production-ready model with:
    - Proven 53.05% accuracy
    - 7 optimally selected features
    - Full validation and metadata
    """
    
    logger = setup_logging()
    logger.info("=== 🚀 DEPLOYING v1.3 STABLE PRODUCTION MODEL ===")
    
    # Load dataset (same as successful v1.3)
    df = pd.read_csv("data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv")
    logger.info(f"Production dataset loaded: {df.shape}")
    
    # Clean features (remove redundant ones like in successful cleanup)
    all_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'season_period_numeric',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm'
    ]
    
    # Prepare data
    X = df[all_features].fillna(0.5)
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Clean features ({len(all_features)}): {all_features}")
    
    # Optimal feature selection (SelectKBest k=7 as proven successful)
    logger.info("\n🎯 OPTIMAL FEATURE SELECTION (k=7)")
    selector = SelectKBest(score_func=f_classif, k=7)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected features
    selected_features = [all_features[i] for i in range(len(all_features)) if selector.get_support()[i]]
    logger.info(f"Selected features: {selected_features}")
    
    # Production model (same configuration that achieved 53.05%)
    production_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features='log2',
        min_samples_leaf=2, min_samples_split=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # Validation
    logger.info("\n📊 PRODUCTION MODEL VALIDATION")
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    cv_scores = cross_val_score(production_model, X_selected, y, cv=tscv, scoring='accuracy')
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    logger.info(f"Production CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    logger.info(f"Production Mean: {mean_score:.4f} (±{std_score:.4f})")
    
    # Validation against target
    target_expected = 0.5305
    if abs(mean_score - target_expected) < 0.002:
        validation_status = "✅ VALIDATED - Matches expected performance"
    else:
        gap = (mean_score - target_expected) * 100
        validation_status = f"⚠️ DEVIATION: {gap:+.1f}pp from expected"
    
    logger.info(f"Validation: {validation_status}")
    
    # Train final model on full dataset
    logger.info("\n🔧 TRAINING FINAL PRODUCTION MODEL")
    production_model.fit(X_selected, y)
    
    # Feature importance
    feature_importance = list(zip(selected_features, production_model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("Production model feature importance:")
    for i, (feature, importance) in enumerate(feature_importance):
        logger.info(f"  {i+1}. {feature:<30}: {importance:.4f}")
    
    # Save production model
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    model_file = f"models/production_v13_stable_{timestamp}.joblib"
    
    # Save model with feature selector
    production_package = {
        'model': production_model,
        'feature_selector': selector,
        'feature_names': all_features,
        'selected_features': selected_features,
        'label_mapping': label_mapping
    }
    
    joblib.dump(production_package, model_file)
    logger.info(f"Production model saved: {model_file}")
    
    # Save production metadata
    production_metadata = {
        "version": "v1.3 Stable Production",
        "timestamp": timestamp,
        "model_type": "Random Forest with SelectKBest",
        "performance": {
            "cv_mean": float(mean_score),
            "cv_std": float(std_score),
            "cv_scores": cv_scores.tolist(),
            "target_expected": float(target_expected),
            "validation_status": validation_status
        },
        "features": {
            "total_available": len(all_features),
            "selected_count": len(selected_features),
            "selection_method": "SelectKBest k=7",
            "all_features": all_features,
            "selected_features": selected_features,
            "feature_importance": [{"feature": f, "importance": float(i)} 
                                  for f, i in feature_importance]
        },
        "model_config": {
            "algorithm": "RandomForestClassifier",
            "n_estimators": 300,
            "max_depth": 12,
            "max_features": "log2",
            "min_samples_leaf": 2,
            "min_samples_split": 15,
            "class_weight": "balanced",
            "random_state": 42
        },
        "data_quality": {
            "total_samples": len(df),
            "training_samples": int(len(y)),
            "missing_values": int(X.isnull().sum().sum()),
            "target_distribution": y.value_counts(normalize=True).to_dict()
        },
        "deployment": {
            "status": "PRODUCTION_READY",
            "baseline_beaten": {
                "random_guess": 0.333,
                "majority_class": 0.436,
                "improvement_vs_random": float((mean_score - 0.333) * 100),
                "improvement_vs_majority": float((mean_score - 0.436) * 100)
            }
        }
    }
    
    metadata_file = f"models/production_v13_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(production_metadata, f, indent=2)
    
    logger.info(f"Production metadata saved: {metadata_file}")
    
    # Usage example
    logger.info("\n📖 PRODUCTION MODEL USAGE EXAMPLE")
    logger.info("To use this model in production:")
    logger.info(f"  model_package = joblib.load('{model_file}')")
    logger.info("  model = model_package['model']")
    logger.info("  selector = model_package['feature_selector']")
    logger.info("  # New data: X_new with all 8 features")
    logger.info("  X_selected = selector.transform(X_new)")
    logger.info("  predictions = model.predict(X_selected)")
    logger.info("  probabilities = model.predict_proba(X_selected)")
    
    logger.info("\n=== 🚀 v1.3 STABLE PRODUCTION DEPLOYMENT COMPLETED ===")
    
    return {
        'model_file': model_file,
        'metadata_file': metadata_file,
        'performance': float(mean_score),
        'selected_features': selected_features,
        'validation_status': validation_status,
        'production_ready': abs(mean_score - target_expected) < 0.005
    }

if __name__ == "__main__":
    result = deploy_v13_stable()
    print(f"\n🚀 v1.3 STABLE PRODUCTION DEPLOYMENT:")
    print(f"Performance: {result['performance']:.4f}")
    print(f"Selected Features: {result['selected_features']}")
    print(f"Validation: {result['validation_status']}")
    print(f"Production Ready: {'✅ YES' if result['production_ready'] else '❌ NO'}")
    print(f"Model: {result['model_file']}")
    print(f"Metadata: {result['metadata_file']}")