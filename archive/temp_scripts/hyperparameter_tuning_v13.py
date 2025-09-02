#!/usr/bin/env python3
"""
Hyperparameter tuning for 7-feature v1.3 model with market intelligence
Goal: Exploit market_entropy_norm properly through optimized parameters
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(__file__))
from utils import setup_logging

def hyperparameter_tuning_v13():
    """
    Optimize hyperparameters for 7-feature model with market intelligence
    """
    logger = setup_logging()
    logger.info("=== 🔧 HYPERPARAMETER TUNING v1.3 (7 FEATURES) ===")
    
    # Load dataset
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Temporal split for tuning
    cutoff_date = '2024-01-01'
    train_dev_data = df[df['Date'] < cutoff_date].copy()
    sealed_test_data = df[df['Date'] >= cutoff_date].copy()
    
    logger.info(f"Dataset: {len(df)} matches")
    logger.info(f"Tuning on: {len(train_dev_data)} matches (2019-2023)")
    logger.info(f"Final test: {len(sealed_test_data)} matches (2024-2025)")
    
    # v1.3 complete features
    v13_features = [
        "form_diff_normalized", "elo_diff_normalized", "h2h_score",
        "matchday_normalized", "shots_diff_normalized", "corners_diff_normalized",
        "market_entropy_norm"
    ]
    
    # Prepare data
    X_train_dev = train_dev_data[v13_features].fillna(0.5)
    X_test = sealed_test_data[v13_features].fillna(0.5)
    
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train_dev = train_dev_data['FullTimeResult'].map(label_mapping)
    y_test = sealed_test_data['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Features: {v13_features}")
    logger.info(f"Train shape: {X_train_dev.shape}")
    
    # Baseline performance (current configuration)
    logger.info(f"\n📊 BASELINE PERFORMANCE")
    logger.info("-" * 50)
    
    baseline_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features='log2',
        min_samples_leaf=2, min_samples_split=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # CV on training data
    tscv = TimeSeriesSplit(n_splits=5)
    from sklearn.model_selection import cross_val_score
    baseline_cv = cross_val_score(baseline_model, X_train_dev, y_train_dev, cv=tscv, scoring='accuracy')
    baseline_cv_mean = baseline_cv.mean()
    
    # Test on sealed
    baseline_model.fit(X_train_dev, y_train_dev)
    baseline_sealed = baseline_model.score(X_test, y_test)
    
    logger.info(f"Baseline CV: {baseline_cv_mean:.4f} ± {baseline_cv.std():.4f}")
    logger.info(f"Baseline Sealed: {baseline_sealed:.4f}")
    
    # Hyperparameter grid - focused on exploiting 7 features
    logger.info(f"\n🔧 HYPERPARAMETER OPTIMIZATION")
    logger.info("-" * 50)
    
    param_grid = {
        'n_estimators': [200, 400, 600],  # More trees for stability
        'max_depth': [10, 15, 20],        # Deeper trees for complex patterns
        'max_features': ['sqrt', 'log2', 0.6, 0.8],  # Different feature sampling
        'min_samples_leaf': [1, 2, 4],    # Allow more granular splits
        'min_samples_split': [10, 20, 30] # Control overfitting
    }
    
    # Estimate search space
    total_combinations = 1
    for param, values in param_grid.items():
        total_combinations *= len(values)
    
    logger.info(f"Search space: {total_combinations} combinations")
    logger.info(f"With 5-fold CV: ~{total_combinations * 5} model fits")
    logger.info(f"Estimated time: ~{total_combinations * 5 * 2 / 60:.0f} minutes")
    
    # Use smaller grid for speed
    if total_combinations > 100:
        logger.info("⚠️ Large search space - using focused grid")
        param_grid = {
            'n_estimators': [300, 500],     # Current + higher
            'max_depth': [12, 18],          # Current + deeper
            'max_features': ['log2', 0.7],  # Current + more features
            'min_samples_leaf': [2, 3],     # Around current
            'min_samples_split': [15, 25]   # Around current
        }
        
        total_combinations = 1
        for param, values in param_grid.items():
            total_combinations *= len(values)
        logger.info(f"Focused grid: {total_combinations} combinations")
    
    # Grid search with TimeSeriesSplit
    start_time = time.time()
    
    rf = RandomForestClassifier(
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=1,  # Use single job since RF already parallel
        verbose=1,
        return_train_score=True
    )
    
    logger.info("Starting grid search...")
    grid_search.fit(X_train_dev, y_train_dev)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Grid search completed in {elapsed_time/60:.1f} minutes")
    
    # Results analysis
    logger.info(f"\n🏆 OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    logger.info(f"Best CV score: {best_cv_score:.4f}")
    logger.info(f"Improvement over baseline: {(best_cv_score - baseline_cv_mean)*100:+.2f}pp")
    logger.info(f"Best parameters: {best_params}")
    
    # Test best model on sealed data
    best_model = grid_search.best_estimator_
    best_sealed_score = best_model.score(X_test, y_test)
    
    logger.info(f"\n🔐 SEALED TEST RESULTS")
    logger.info("-" * 40)
    logger.info(f"Best model sealed score: {best_sealed_score:.4f}")
    logger.info(f"Baseline sealed score:   {baseline_sealed:.4f}")
    logger.info(f"Improvement:             {(best_sealed_score - baseline_sealed)*100:+.2f}pp")
    
    # Statistical significance of improvement
    improvement = best_sealed_score - baseline_sealed
    if improvement > 0.01:  # >1pp improvement
        significance = "🟢 SIGNIFICANT IMPROVEMENT"
    elif improvement > 0.005:  # >0.5pp
        significance = "🟡 MARGINAL IMPROVEMENT"
    elif improvement > -0.005:  # <0.5pp loss
        significance = "🔵 NO MEANINGFUL CHANGE"
    else:
        significance = "🔴 PERFORMANCE DEGRADATION"
    
    logger.info(f"Significance: {significance}")
    
    # Feature importance comparison
    logger.info(f"\n🎯 FEATURE IMPORTANCE COMPARISON")
    logger.info("-" * 50)
    
    baseline_importance = dict(zip(v13_features, baseline_model.feature_importances_))
    best_importance = dict(zip(v13_features, best_model.feature_importances_))
    
    logger.info("Feature importance changes:")
    for feature in v13_features:
        baseline_imp = baseline_importance[feature]
        best_imp = best_importance[feature]
        change = (best_imp - baseline_imp) * 100
        
        marker = " 🎯" if feature == "market_entropy_norm" else ""
        logger.info(f"  {feature:25s}: {baseline_imp:.3f} → {best_imp:.3f} ({change:+.1f}%){marker}")
    
    # Market intelligence analysis
    market_baseline = baseline_importance['market_entropy_norm']
    market_best = best_importance['market_entropy_norm']
    market_rank_baseline = sorted(baseline_importance.items(), key=lambda x: x[1], reverse=True)
    market_rank_best = sorted(best_importance.items(), key=lambda x: x[1], reverse=True)
    
    market_pos_baseline = [i for i, (f, _) in enumerate(market_rank_baseline, 1) if f == 'market_entropy_norm'][0]
    market_pos_best = [i for i, (f, _) in enumerate(market_rank_best, 1) if f == 'market_entropy_norm'][0]
    
    logger.info(f"\n📊 MARKET INTELLIGENCE EXPLOITATION:")
    logger.info(f"  Baseline ranking: #{market_pos_baseline} ({market_baseline:.3f})")
    logger.info(f"  Optimized ranking: #{market_pos_best} ({market_best:.3f})")
    logger.info(f"  Importance change: {((market_best/market_baseline)-1)*100:+.1f}%")
    
    if market_pos_best < market_pos_baseline:
        logger.info(f"  ✅ Market intelligence better exploited (rank improved)")
    elif market_best > market_baseline:
        logger.info(f"  ✅ Market intelligence importance increased")
    else:
        logger.info(f"  ⚠️ Limited improvement in market intelligence exploitation")
    
    # Top parameter insights
    logger.info(f"\n💡 PARAMETER INSIGHTS:")
    if best_params['n_estimators'] > 300:
        logger.info("  • More trees beneficial (ensemble stability)")
    if best_params['max_depth'] > 12:
        logger.info("  • Deeper trees capture complex market patterns")
    if isinstance(best_params['max_features'], float) and best_params['max_features'] > 0.5:
        logger.info("  • Using more features per split (market interactions)")
    if best_params['min_samples_leaf'] < 2:
        logger.info("  • More granular predictions (individual match patterns)")
    
    # Business impact
    logger.info(f"\n💰 BUSINESS IMPACT")
    logger.info("-" * 40)
    
    if best_sealed_score >= 0.55:
        verdict = "🏆 EXCELLENT - Industry competitive"
    elif best_sealed_score >= 0.54:
        verdict = "✅ STRONG - Solid foundation"
    elif best_sealed_score >= baseline_sealed:
        verdict = "🟢 IMPROVED - Step forward"
    else:
        verdict = "⚠️ NO IMPROVEMENT - Need different approach"
    
    logger.info(f"Final performance: {best_sealed_score:.4f} ({verdict})")
    
    # Save optimized model
    import joblib
    model_filename = f"models/v13_optimized_{time.strftime('%Y_%m_%d_%H%M%S')}.joblib"
    joblib.dump(best_model, model_filename)
    logger.info(f"Optimized model saved: {model_filename}")
    
    # Final recommendations
    logger.info(f"\n🚀 NEXT STEPS")
    logger.info("=" * 40)
    
    if improvement > 0.01:
        logger.info("✅ HYPERPARAMETER TUNING SUCCESSFUL")
        logger.info("   → Use optimized model as v1.3 final")
        logger.info("   → Ready for v2.0 feature development")
    elif improvement > 0.005:
        logger.info("⚡ MARGINAL TUNING SUCCESS")  
        logger.info("   → Consider ensemble methods or different algorithms")
        logger.info("   → Focus on new features for major gains")
    else:
        logger.info("⚠️ LIMITED TUNING GAINS")
        logger.info("   → Current architecture near optimal")
        logger.info("   → Major improvements need new features/approaches")
    
    results = {
        'baseline_cv': float(baseline_cv_mean),
        'baseline_sealed': float(baseline_sealed),
        'best_cv': float(best_cv_score),
        'best_sealed': float(best_sealed_score),
        'improvement': float(improvement),
        'best_params': best_params,
        'market_intelligence_rank': int(market_pos_best),
        'tuning_successful': improvement > 0.005
    }
    
    return results

if __name__ == "__main__":
    try:
        results = hyperparameter_tuning_v13()
        print(f"\n🔧 HYPERPARAMETER TUNING COMPLETE")
        print(f"Baseline: {results['baseline_sealed']:.4f}")
        print(f"Optimized: {results['best_sealed']:.4f}")
        print(f"Improvement: {results['improvement']*100:+.2f}pp")
        print(f"Success: {results['tuning_successful']}")
        
    except Exception as e:
        print(f"❌ Tuning failed: {e}")
        import traceback
        traceback.print_exc()