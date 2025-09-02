#!/usr/bin/env python3
"""
ALGORITHM BENCHMARKING - v1.5
Compare RandomForest vs XGBoost vs LightGBM with clean methodology
Use optimized hyperparameters from previous tuning
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced algorithms
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def load_clean_data():
    """Load data with clean temporal split"""
    logger = setup_logging()
    logger.info("=== Loading Data ===")
    
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Season-based split (definitive methodology)
    train_seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
    test_seasons = ['2024-2025']
    
    train_data = df[df['Season'].isin(train_seasons)]
    test_data = df[df['Season'].isin(test_seasons)]
    
    features = [
        "form_diff_normalized", "elo_diff_normalized", "h2h_score",
        "matchday_normalized", "shots_diff_normalized", "corners_diff_normalized",
        "market_entropy_norm"
    ]
    
    X_train = train_data[features].fillna(0.5)
    X_test = test_data[features].fillna(0.5)
    
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train = train_data['FullTimeResult'].map(label_mapping)
    y_test = test_data['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def benchmark_random_forest(X_train, X_test, y_train, y_test):
    """Benchmark Random Forest with optimized config"""
    logger = setup_logging()
    logger.info("=== RandomForest Benchmark ===")
    
    # Use optimized config from successful tuning
    rf_config = {
        'n_estimators': 300,
        'max_depth': 10,        # Optimized from 18
        'min_samples_leaf': 10, # Optimized from 2
        'min_samples_split': 15,
        'max_features': 'log2',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    logger.info(f"RandomForest config: {rf_config}")
    
    rf_model = RandomForestClassifier(**rf_config)
    rf_model.fit(X_train, y_train)
    
    train_acc = rf_model.score(X_train, y_train)
    test_acc = rf_model.score(X_test, y_test)
    gap = train_acc - test_acc
    
    y_pred = rf_model.predict(X_test)
    
    logger.info(f"RandomForest Results:")
    logger.info(f"  Train: {train_acc:.4f}")
    logger.info(f"  Test:  {test_acc:.4f}")
    logger.info(f"  Gap:   {gap*100:.1f}pp")
    
    return {
        'algorithm': 'RandomForest',
        'model': rf_model,
        'config': rf_config,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap,
        'predictions': y_pred
    }

def benchmark_xgboost(X_train, X_test, y_train, y_test):
    """Benchmark XGBoost if available"""
    logger = setup_logging()
    
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available - skipping")
        return None
    
    logger.info("=== XGBoost Benchmark ===")
    
    # XGBoost config optimized for similar performance characteristics
    xgb_config = {
        'n_estimators': 300,
        'max_depth': 8,           # Slightly lower for XGB
        'learning_rate': 0.1,
        'min_child_weight': 10,   # Similar to min_samples_leaf
        'subsample': 0.8,
        'colsample_bytree': 0.8,  # Similar to max_features
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }
    
    logger.info(f"XGBoost config: {xgb_config}")
    
    xgb_model = xgb.XGBClassifier(**xgb_config)
    xgb_model.fit(X_train, y_train, verbose=False)
    
    train_acc = xgb_model.score(X_train, y_train)
    test_acc = xgb_model.score(X_test, y_test)
    gap = train_acc - test_acc
    
    y_pred = xgb_model.predict(X_test)
    
    logger.info(f"XGBoost Results:")
    logger.info(f"  Train: {train_acc:.4f}")
    logger.info(f"  Test:  {test_acc:.4f}")
    logger.info(f"  Gap:   {gap*100:.1f}pp")
    
    return {
        'algorithm': 'XGBoost',
        'model': xgb_model,
        'config': xgb_config,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap,
        'predictions': y_pred
    }

def benchmark_lightgbm(X_train, X_test, y_train, y_test):
    """Benchmark LightGBM if available"""
    logger = setup_logging()
    
    if not LIGHTGBM_AVAILABLE:
        logger.warning("LightGBM not available - skipping")
        return None
    
    logger.info("=== LightGBM Benchmark ===")
    
    # LightGBM config optimized for similar characteristics
    lgb_config = {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.1,
        'min_child_samples': 10,  # Similar to min_samples_leaf
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    logger.info(f"LightGBM config: {lgb_config}")
    
    lgb_model = lgb.LGBMClassifier(**lgb_config)
    lgb_model.fit(X_train, y_train)
    
    train_acc = lgb_model.score(X_train, y_train)
    test_acc = lgb_model.score(X_test, y_test)
    gap = train_acc - test_acc
    
    y_pred = lgb_model.predict(X_test)
    
    logger.info(f"LightGBM Results:")
    logger.info(f"  Train: {train_acc:.4f}")
    logger.info(f"  Test:  {test_acc:.4f}")
    logger.info(f"  Gap:   {gap*100:.1f}pp")
    
    return {
        'algorithm': 'LightGBM', 
        'model': lgb_model,
        'config': lgb_config,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap,
        'predictions': y_pred
    }

def compare_algorithms(results, y_test):
    """Compare all algorithm results"""
    logger = setup_logging()
    logger.info("=== Algorithm Comparison ===")
    
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        logger.error("No valid results to compare")
        return None, None
    
    logger.info(f"Comparing {len(valid_results)} algorithms:")
    logger.info(f"{'Algorithm':<12} {'Train':<8} {'Test':<8} {'Gap':<8} {'Score'}")
    logger.info("-" * 50)
    
    best_algorithm = None
    best_score = -1
    
    for result in valid_results:
        algo = result['algorithm']
        train = result['train_acc']
        test = result['test_acc']
        gap = result['gap']
        
        # Composite score: test performance - gap penalty
        score = test - (0.2 * max(0, gap - 0.15))  # 15% gap tolerance
        
        logger.info(f"{algo:<12} {train:<8.3f} {test:<8.3f} {gap*100:<6.1f}pp {score:<8.3f}")
        
        if score > best_score:
            best_score = score
            best_algorithm = result
    
    logger.info(f"\nBest Algorithm: {best_algorithm['algorithm']}")
    
    # Detailed comparison with baseline
    baseline_test = 0.5184  # Original definitive baseline
    baseline_gap = 0.331   # Original gap
    
    logger.info(f"\nComparison vs Definitive Baseline:")
    logger.info(f"Baseline: Test={baseline_test:.4f}, Gap={baseline_gap*100:.1f}pp")
    logger.info(f"Best:     Test={best_algorithm['test_acc']:.4f}, Gap={best_algorithm['gap']*100:.1f}pp")
    
    test_improvement = (best_algorithm['test_acc'] - baseline_test) * 100
    gap_improvement = (baseline_gap - best_algorithm['gap']) * 100
    
    logger.info(f"Improvements: Test{test_improvement:+.2f}pp, Gap{gap_improvement:+.2f}pp")
    
    # Performance assessment
    final_test = best_algorithm['test_acc']
    if final_test >= 0.525:
        assessment = "🏆 EXCELLENT - Significant improvement achieved"
    elif final_test >= 0.520:
        assessment = "✅ GOOD - Meaningful improvement"
    elif final_test >= baseline_test:
        assessment = "📈 IMPROVED - Better than baseline"
    else:
        assessment = "⚠️ DECLINED - Below baseline"
    
    logger.info(f"\nAssessment: {assessment}")
    
    return best_algorithm, valid_results

def save_benchmark_results(best_algorithm, all_results):
    """Save benchmarking results"""
    logger = setup_logging()
    logger.info("=== Saving Results ===")
    
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    
    # Save best model
    model_file = f"models/v15_best_{best_algorithm['algorithm'].lower()}_{timestamp}.joblib"
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_algorithm['model'], model_file)
    
    # Prepare results summary
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.5_algorithm_benchmarking',
        'best_algorithm': best_algorithm['algorithm'],
        'model_file': model_file,
        'best_config': best_algorithm['config'],
        'performance': {
            'test_accuracy': float(best_algorithm['test_acc']),
            'train_accuracy': float(best_algorithm['train_acc']),
            'overfitting_gap': float(best_algorithm['gap'] * 100)
        },
        'comparison_results': []
    }
    
    for result in all_results:
        results_summary['comparison_results'].append({
            'algorithm': result['algorithm'],
            'test_accuracy': float(result['test_acc']),
            'train_accuracy': float(result['train_acc']),
            'overfitting_gap': float(result['gap'] * 100),
            'config': result['config']
        })
    
    # Save results
    os.makedirs('evaluation', exist_ok=True)
    results_file = f"evaluation/v15_algorithm_benchmark_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"Best model saved: {model_file}")
    logger.info(f"Results saved: {results_file}")
    
    return results_summary

def main():
    """Main algorithm benchmarking pipeline"""
    logger = setup_logging()
    logger.info("ALGORITHM BENCHMARKING - v1.5")
    logger.info("=" * 60)
    logger.info("Compare RandomForest vs XGBoost vs LightGBM")
    logger.info("Using clean temporal split + optimized hyperparameters")
    logger.info("=" * 60)
    
    # Check available algorithms
    available = []
    if True: available.append("RandomForest")
    if XGBOOST_AVAILABLE: available.append("XGBoost")
    if LIGHTGBM_AVAILABLE: available.append("LightGBM")
    
    logger.info(f"Available algorithms: {available}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_clean_data()
    
    # Benchmark algorithms
    results = []
    
    # Always test RandomForest (baseline)
    rf_result = benchmark_random_forest(X_train, X_test, y_train, y_test)
    results.append(rf_result)
    
    # Test XGBoost if available
    if XGBOOST_AVAILABLE:
        xgb_result = benchmark_xgboost(X_train, X_test, y_train, y_test)
        if xgb_result:
            results.append(xgb_result)
    
    # Test LightGBM if available
    if LIGHTGBM_AVAILABLE:
        lgb_result = benchmark_lightgbm(X_train, X_test, y_train, y_test)
        if lgb_result:
            results.append(lgb_result)
    
    # Compare results
    best_algorithm, all_results = compare_algorithms(results, y_test)
    
    if best_algorithm is None:
        logger.error("No valid results to compare")
        return False
    
    # Save results
    summary = save_benchmark_results(best_algorithm, all_results)
    
    # Final assessment
    logger.info(f"\nALGORITHM BENCHMARKING COMPLETE")
    logger.info(f"Best algorithm: {best_algorithm['algorithm']}")
    logger.info(f"Test accuracy: {best_algorithm['test_acc']:.4f}")
    logger.info(f"Overfitting gap: {best_algorithm['gap']*100:.1f}pp")
    
    success = best_algorithm['test_acc'] >= 0.5184  # Beat definitive baseline
    logger.info(f"Success: {'YES' if success else 'NO'}")
    
    if success:
        logger.info("🎯 v1.5 optimization complete - Ready for production")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        print(f"Algorithm benchmarking complete: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)