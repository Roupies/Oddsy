import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def investigate_performance_gap():
    """
    Investigate the 1.7pp gap between:
    - model_cleanup_v2.py: 53.05% (original)
    - deploy_v13_stable.py: 51.37% (reproduction)
    
    Find the exact source of this discrepancy.
    """
    
    logger = setup_logging()
    logger.info("=== 🔍 INVESTIGATING 1.7PP PERFORMANCE GAP ===")
    logger.info("Target: Understand 53.05% vs 51.37% difference")
    
    # Load same dataset
    df = pd.read_csv("data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv")
    logger.info(f"Dataset: {df.shape}")
    
    # Same features as successful cleanup
    clean_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'season_period_numeric',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm'
    ]
    
    logger.info(f"Clean features ({len(clean_features)}): {clean_features}")
    
    # Prepare data
    X_clean = df[clean_features].fillna(0.5)
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y = df['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Feature matrix: {X_clean.shape}")
    logger.info(f"Target distribution: {y.value_counts(normalize=True).sort_index().round(3).to_dict()}")
    
    # Cross-validation setup
    cv_folds = 5
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    logger.info(f"CV setup: {cv_folds} folds, TimeSeriesSplit")
    
    # =======================
    # TEST 1: EXACT MODEL FROM CLEANUP SCRIPT
    # =======================
    logger.info("\n🧪 TEST 1: EXACT MODEL FROM model_cleanup_v2.py")
    
    # This is the model used in cleanup script line 137-142
    xgb_conservative = None
    try:
        from xgboost import XGBClassifier
        xgb_conservative = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='mlogloss', n_jobs=-1
        )
        
        cv_scores = cross_val_score(xgb_conservative, X_clean, y, cv=tscv, scoring='accuracy')
        mean_score = cv_scores.mean()
        
        logger.info(f"XGBoost Conservative: {mean_score:.4f}")
        logger.info(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
        
        if abs(mean_score - 0.5305) < 0.001:
            logger.info("  🎯 FOUND THE 53.05% RESULT!")
    except ImportError:
        logger.warning("XGBoost not available")
    
    # =======================  
    # TEST 2: FEATURE SELECTION WITH TOP 7
    # =======================
    logger.info("\n🧪 TEST 2: SelectKBest k=7 (Original Method)")
    
    selector = SelectKBest(score_func=f_classif, k=7)
    X_selected = selector.fit_transform(X_clean, y)
    
    selected_features = [clean_features[i] for i in range(len(clean_features)) if selector.get_support()[i]]
    logger.info(f"Selected features: {selected_features}")
    
    # Test with same XGBoost if available
    if xgb_conservative:
        cv_scores_selected = cross_val_score(xgb_conservative, X_selected, y, cv=tscv, scoring='accuracy')
        mean_selected = cv_scores_selected.mean()
        
        logger.info(f"XGBoost + SelectKBest k=7: {mean_selected:.4f}")
        logger.info(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores_selected]}")
        
        if abs(mean_selected - 0.5305) < 0.001:
            logger.info("  🎯 FOUND THE 53.05% RESULT!")
    
    # Test with RandomForest (what I used in deploy script)
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features='log2',
        min_samples_leaf=2, min_samples_split=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    cv_scores_rf = cross_val_score(rf_model, X_selected, y, cv=tscv, scoring='accuracy')
    mean_rf = cv_scores_rf.mean()
    
    logger.info(f"RandomForest + SelectKBest k=7: {mean_rf:.4f}")
    logger.info(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores_rf]}")
    
    # =======================
    # TEST 3: RANDOM STATE INVESTIGATION  
    # =======================
    logger.info("\n🧪 TEST 3: RANDOM STATE SENSITIVITY")
    
    if xgb_conservative:
        random_states = [42, 0, 1, 123, 999]
        
        for rs in random_states:
            # Create fresh TimeSeriesSplit with different random state
            tscv_rs = TimeSeriesSplit(n_splits=cv_folds)
            
            xgb_rs = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=rs,
                eval_metric='mlogloss', n_jobs=-1
            )
            
            cv_scores_rs = cross_val_score(xgb_rs, X_selected, y, cv=tscv_rs, scoring='accuracy')
            mean_rs = cv_scores_rs.mean()
            
            logger.info(f"  Random State {rs}: {mean_rs:.4f}")
            
            if abs(mean_rs - 0.5305) < 0.001:
                logger.info(f"    🎯 FOUND 53.05% with random_state={rs}!")
    
    # =======================
    # TEST 4: CV FOLD ANALYSIS
    # =======================
    logger.info("\n🧪 TEST 4: CV FOLD DETAILED ANALYSIS")
    
    if xgb_conservative:
        # Manual CV to inspect individual folds
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_selected)):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            xgb_conservative.fit(X_train, y_train)
            fold_score = xgb_conservative.score(X_test, y_test)
            fold_results.append(fold_score)
            
            logger.info(f"  Fold {fold_idx + 1}: {fold_score:.4f} (train: {len(train_idx)}, test: {len(test_idx)})")
        
        manual_mean = np.mean(fold_results)
        logger.info(f"Manual CV Mean: {manual_mean:.4f}")
        
        if abs(manual_mean - 0.5305) < 0.001:
            logger.info("  🎯 FOUND 53.05% with manual CV!")
    
    # =======================
    # TEST 5: DATA HASH VERIFICATION
    # =======================
    logger.info("\n🧪 TEST 5: DATA INTEGRITY CHECK")
    
    import hashlib
    
    # Hash the feature matrix to verify data consistency
    X_bytes = X_selected.tobytes()
    data_hash = hashlib.md5(X_bytes).hexdigest()[:8]
    logger.info(f"Feature matrix hash: {data_hash}")
    
    # Hash the target
    y_bytes = y.values.tobytes()  
    target_hash = hashlib.md5(y_bytes).hexdigest()[:8]
    logger.info(f"Target hash: {target_hash}")
    
    # =======================
    # SUMMARY & CONCLUSION
    # =======================
    logger.info("\n=== 🔍 INVESTIGATION SUMMARY ===")
    
    # Compare all our results
    results = {
        "XGBoost Full Features": mean_score if xgb_conservative else None,
        "XGBoost + SelectKBest k=7": mean_selected if xgb_conservative else None, 
        "RandomForest + SelectKBest k=7": mean_rf,
        "Target (Original)": 0.5305
    }
    
    logger.info("Performance comparison:")
    for method, score in results.items():
        if score:
            gap = (score - 0.5305) * 100 if score else 0
            logger.info(f"  {method}: {score:.4f} ({gap:+.1f}pp from target)")
    
    # Find closest match
    closest_method = None
    smallest_gap = float('inf')
    
    for method, score in results.items():
        if score and method != "Target (Original)":
            gap = abs(score - 0.5305)
            if gap < smallest_gap:
                smallest_gap = gap
                closest_method = method
    
    if smallest_gap < 0.001:
        logger.info(f"✅ SUCCESS: {closest_method} reproduces 53.05%")
        explanation = "REPRODUCED"
    elif smallest_gap < 0.005:
        logger.info(f"🔸 CLOSE: {closest_method} within 0.5pp of 53.05%")
        explanation = "CLOSE_MATCH"
    else:
        logger.info(f"❌ MYSTERY: Cannot reproduce 53.05% (closest: {closest_method})")
        explanation = "MYSTERY_PERSISTS"
    
    logger.info("=== 🔍 INVESTIGATION COMPLETED ===")
    
    return {
        'explanation': explanation,
        'closest_method': closest_method,
        'smallest_gap': smallest_gap,
        'data_hash': data_hash,
        'target_hash': target_hash,
        'all_results': results
    }

if __name__ == "__main__":
    result = investigate_performance_gap()
    print(f"\n🔍 PERFORMANCE GAP INVESTIGATION:")
    print(f"Explanation: {result['explanation']}")
    print(f"Closest Method: {result['closest_method']}")
    print(f"Smallest Gap: {result['smallest_gap']:.4f}")
    print(f"Data Hash: {result['data_hash']}")
    print(f"Target Hash: {result['target_hash']}")