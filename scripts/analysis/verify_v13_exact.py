import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def verify_v13_exact():
    """
    Test with EXACT features that achieved 53.05% in cleanup_results_v2
    """
    
    logger = setup_logging()
    logger.info("=== 🔍 VERIFYING EXACT v1.3 SUCCESS (53.05%) ===")
    
    # Load dataset
    df = pd.read_csv("data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv")
    logger.info(f"Dataset: {df.shape}")
    
    # EXACT features from cleanup_results_v2.json that achieved 53.05%
    exact_v13_features = [
        "form_diff_normalized",
        "elo_diff_normalized", 
        "h2h_score",
        "matchday_normalized",
        "season_period_numeric",
        "shots_diff_normalized",
        "corners_diff_normalized",
        "market_entropy_norm"
    ]
    
    logger.info(f"EXACT v1.3 features ({len(exact_v13_features)}): {exact_v13_features}")
    
    # Check if all features exist
    missing_features = [f for f in exact_v13_features if f not in df.columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        return
    
    # Prepare data
    X = df[exact_v13_features].fillna(0.5)
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Feature matrix: {X.shape}")
    
    # Cross-validation setup (same as original)
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Test different models to find which one gave 53.05%
    models_to_test = {
        'Random Forest Conservative': RandomForestClassifier(
            n_estimators=100, max_depth=10, max_features='log2',
            min_samples_leaf=3, min_samples_split=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Random Forest Standard': RandomForestClassifier(
            n_estimators=300, max_depth=12, max_features='log2',
            min_samples_leaf=2, min_samples_split=15,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'SelectKBest Top 7': None  # Will implement feature selection
    }
    
    logger.info("\n🧪 TESTING MODELS TO REPRODUCE 53.05%")
    
    best_score = 0
    best_model = None
    
    for model_name, model in models_to_test.items():
        if model_name == 'SelectKBest Top 7':
            # Try feature selection (the approach was "Top_7_features")
            from sklearn.feature_selection import SelectKBest, f_classif
            
            selector = SelectKBest(score_func=f_classif, k=7)
            X_selected = selector.fit_transform(X, y)
            
            selected_features = [exact_v13_features[i] for i in range(len(exact_v13_features)) if selector.get_support()[i]]
            logger.info(f"  Selected 7 features: {selected_features}")
            
            model = RandomForestClassifier(
                n_estimators=300, max_depth=12, max_features='log2',
                min_samples_leaf=2, min_samples_split=15,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            
            cv_scores = cross_val_score(model, X_selected, y, cv=tscv, scoring='accuracy')
        else:
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        logger.info(f"{model_name}: {mean_score:.4f} (±{std_score:.4f})")
        logger.info(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
        
        if abs(mean_score - 0.5305) < 0.001:
            logger.info(f"  🎯 FOUND THE 53.05% RESULT!")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model_name
    
    # Compare with what I tested before
    logger.info(f"\n📊 COMPARISON WITH PREVIOUS TESTS")
    logger.info(f"Exact v1.3 (8 features): {best_score:.4f}")
    
    # Previous test included home_advantage - let's test with that too
    previous_features = exact_v13_features + ['home_advantage']
    if 'home_advantage' in df.columns:
        X_prev = df[previous_features].fillna(0.5)
        
        model = RandomForestClassifier(
            n_estimators=300, max_depth=12, max_features='log2',
            min_samples_leaf=2, min_samples_split=15,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        
        cv_scores_prev = cross_val_score(model, X_prev, y, cv=tscv, scoring='accuracy')
        mean_prev = cv_scores_prev.mean()
        
        logger.info(f"Previous test (9 features): {mean_prev:.4f}")
        logger.info(f"Difference: {(best_score - mean_prev)*100:+.1f}pp")
    
    logger.info(f"\n=== VERIFICATION RESULTS ===")
    logger.info(f"Best reproduction: {best_score:.4f} with {best_model}")
    
    if abs(best_score - 0.5305) < 0.005:
        logger.info(f"✅ SUCCESSFULLY REPRODUCED 53.05% RESULT")
        status = "REPRODUCED"
    else:
        gap = (0.5305 - best_score) * 100
        logger.info(f"❌ COULD NOT REPRODUCE: {gap:+.1f}pp gap")
        status = "NOT_REPRODUCED"
    
    return {
        'best_score': best_score,
        'target_score': 0.5305,
        'reproduced': abs(best_score - 0.5305) < 0.005,
        'best_model': best_model,
        'status': status
    }

if __name__ == "__main__":
    result = verify_v13_exact()
    print(f"\n🔍 EXACT v1.3 VERIFICATION:")
    print(f"Best Score: {result['best_score']:.4f}")
    print(f"Target: {result['target_score']:.4f}")
    print(f"Reproduced: {'✅ YES' if result['reproduced'] else '❌ NO'}")
    print(f"Status: {result['status']}")