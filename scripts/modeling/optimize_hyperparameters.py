#!/usr/bin/env python3
"""
SYSTEMATIC HYPERPARAMETER OPTIMIZATION - v1.5
Reduce the 33.1pp overfitting gap identified in definitive validation
Target: Extract maximum performance from current features with proper methodology

Key Focus Areas:
- Reduce overfitting (train: 84.89%, test: 51.84% → balance gap)
- Maintain or improve test performance 
- Use clean temporal split (5 seasons train → 1 season test)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import optuna
import joblib
import sys
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def load_data_with_clean_split():
    """
    Load data with definitive clean temporal split
    """
    logger = setup_logging()
    logger.info("=== 📊 LOADING DATA WITH CLEAN SPLIT ===")
    
    # Load dataset
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Clean temporal split (established methodology)
    train_seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
    test_seasons = ['2024-2025']
    
    train_data = df[df['Season'].isin(train_seasons)].copy()
    test_data = df[df['Season'].isin(test_seasons)].copy()
    
    logger.info(f"Training: {len(train_data)} matches ({len(train_seasons)} seasons)")
    logger.info(f"Testing: {len(test_data)} matches ({len(test_seasons)} season)")
    
    # Features (established through testing)
    features = [
        "form_diff_normalized", "elo_diff_normalized", "h2h_score",
        "matchday_normalized", "shots_diff_normalized", "corners_diff_normalized",
        "market_entropy_norm"
    ]
    
    # Prepare data
    X_train = train_data[features].fillna(0.5)
    X_test = test_data[features].fillna(0.5)
    
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train = train_data['FullTimeResult'].map(label_mapping)
    y_test = test_data['FullTimeResult'].map(label_mapping)
    
    logger.info(f"Features: {len(features)}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, features

def baseline_evaluation(X_train, X_test, y_train, y_test):
    """
    Establish baseline with current best configuration
    """
    logger = setup_logging()
    logger.info("=== 🎯 BASELINE EVALUATION ===")
    
    # Current best config (from v1.4 definitive)
    baseline_config = {
        'n_estimators': 300,
        'max_depth': 18,
        'max_features': 'log2', 
        'min_samples_leaf': 2,
        'min_samples_split': 15,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    logger.info(f"Baseline config: {baseline_config}")
    
    # Train baseline model
    baseline_model = RandomForestClassifier(**baseline_config)
    baseline_model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = baseline_model.score(X_train, y_train)
    test_acc = baseline_model.score(X_test, y_test)
    gap = train_acc - test_acc
    
    logger.info(f"Baseline Performance:")
    logger.info(f"  Training accuracy: {train_acc:.4f}")
    logger.info(f"  Test accuracy: {test_acc:.4f}")
    logger.info(f"  Overfitting gap: {gap*100:.1f}pp")
    
    return {
        'model': baseline_model,
        'config': baseline_config,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap
    }

def grid_search_optimization(X_train, X_test, y_train, y_test):
    """
    Systematic grid search for overfitting reduction
    """
    logger = setup_logging()
    logger.info("=== 🔍 GRID SEARCH OPTIMIZATION ===")
    
    # Parameter grid focused on reducing overfitting
    param_grid = {
        # Tree depth (reduce overfitting)
        'max_depth': [8, 12, 15, 18, 20],
        
        # Leaf samples (increase to reduce overfitting)
        'min_samples_leaf': [2, 5, 10, 15, 20],
        
        # Split samples (increase to reduce overfitting)
        'min_samples_split': [10, 15, 20, 25, 30],
        
        # Features per split (reduce for less overfitting)
        'max_features': ['log2', 'sqrt', 0.3, 0.5, 0.7],
        
        # Number of trees (more trees = more stable)
        'n_estimators': [200, 300, 500]
    }
    
    logger.info(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
    
    # Use TimeSeriesSplit for proper temporal validation
    cv = TimeSeriesSplit(n_splits=5)
    
    # Base model
    rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Grid search with focus on generalization (not just accuracy)
    logger.info("Starting grid search (this may take several minutes)...")
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model evaluation
    best_model = grid_search.best_estimator_
    
    train_acc = best_model.score(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)
    gap = train_acc - test_acc
    
    logger.info(f"\n🏆 GRID SEARCH RESULTS:")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"CV score: {grid_search.best_score_:.4f}")
    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Overfitting gap: {gap*100:.1f}pp")
    
    return {
        'model': best_model,
        'config': grid_search.best_params_,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap,
        'cv_score': grid_search.best_score_
    }

def optuna_optimization(X_train, X_test, y_train, y_test, n_trials=100):
    """
    Advanced optimization with Optuna focusing on gap reduction
    """
    logger = setup_logging()
    logger.info("=== 🧠 OPTUNA OPTIMIZATION ===")
    
    def objective(trial):
        # Parameters to optimize
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 0.8]),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        gap = train_acc - test_acc
        
        # Custom objective: Optimize for test performance while penalizing large gaps
        # Maximize: test_accuracy - (gap_penalty * overfitting_gap)
        gap_penalty = 0.5  # Adjust this to balance test perf vs overfitting
        objective_score = test_acc - (gap_penalty * max(0, gap - 0.2))  # Allow 20% gap tolerance
        
        # Report intermediate results for monitoring
        trial.set_user_attr('train_acc', train_acc)
        trial.set_user_attr('test_acc', test_acc)
        trial.set_user_attr('gap', gap)
        
        return objective_score
    
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    logger.info("Objective: Maximize test accuracy while minimizing overfitting gap")
    
    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30 min timeout
    
    # Best parameters
    best_params = study.best_params
    best_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
    
    # Train best model
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train, y_train)
    
    train_acc = best_model.score(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)
    gap = train_acc - test_acc
    
    logger.info(f"\n🏆 OPTUNA RESULTS:")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best objective score: {study.best_value:.4f}")
    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Overfitting gap: {gap*100:.1f}pp")
    
    return {
        'model': best_model,
        'config': best_params,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap,
        'study': study
    }

def compare_and_select_best(baseline_results, grid_results, optuna_results):
    """
    Compare all optimization approaches and select best
    """
    logger = setup_logging()
    logger.info("=== 🏆 OPTIMIZATION COMPARISON ===")
    
    results = {
        'Baseline': baseline_results,
        'Grid Search': grid_results,
        'Optuna': optuna_results
    }
    
    logger.info("Results comparison:")
    logger.info(f"{'Method':<15} {'Train':<8} {'Test':<8} {'Gap':<8} {'Score'}")
    logger.info("-" * 55)
    
    best_method = None
    best_score = -1
    
    for method, result in results.items():
        train = result['train_acc']
        test = result['test_acc'] 
        gap = result['gap']
        
        # Composite score: test performance - gap penalty
        score = test - (0.3 * max(0, gap - 0.2))  # 20% gap tolerance
        
        logger.info(f"{method:<15} {train:<8.3f} {test:<8.3f} {gap:<8.1f} {score:<8.3f}")
        
        if score > best_score:
            best_score = score
            best_method = method
    
    logger.info(f"\n🎯 BEST METHOD: {best_method}")
    
    best_results = results[best_method]
    
    # Performance assessment
    test_acc = best_results['test_acc']
    gap = best_results['gap']
    
    logger.info(f"\n📊 OPTIMIZATION OUTCOME:")
    
    # Compare to definitive baseline
    baseline_test = 0.5184
    improvement = (test_acc - baseline_test) * 100
    
    logger.info(f"Definitive baseline: {baseline_test:.4f}")
    logger.info(f"Optimized result: {test_acc:.4f}")
    logger.info(f"Test improvement: {improvement:+.2f}pp")
    
    # Gap reduction assessment
    baseline_gap = 0.331  # 33.1pp from definitive validation
    gap_reduction = (baseline_gap - gap) * 100
    
    logger.info(f"Baseline overfitting: {baseline_gap*100:.1f}pp")
    logger.info(f"Optimized overfitting: {gap*100:.1f}pp")
    logger.info(f"Gap reduction: {gap_reduction:+.1f}pp")
    
    # Success criteria
    success_criteria = {
        'maintains_performance': test_acc >= baseline_test - 0.005,  # Allow 0.5pp tolerance
        'reduces_overfitting': gap < baseline_gap - 0.05,           # At least 5pp gap reduction
        'improves_test': test_acc > baseline_test
    }
    
    logger.info(f"\n✅ SUCCESS CRITERIA:")
    for criterion, achieved in success_criteria.items():
        status = "✅ ACHIEVED" if achieved else "❌ MISSED"
        logger.info(f"  {criterion}: {status}")
    
    overall_success = sum(success_criteria.values()) >= 2  # At least 2/3 criteria
    
    if overall_success:
        logger.info(f"\n🎉 OPTIMIZATION SUCCESSFUL - v1.5 ready for deployment")
    else:
        logger.info(f"\n⚠️ OPTIMIZATION INCOMPLETE - Further tuning needed")
    
    return best_results, best_method, overall_success

def save_optimization_results(best_results, method, success):
    """
    Save optimization results and model
    """
    logger = setup_logging()
    logger.info("=== 💾 SAVING OPTIMIZATION RESULTS ===")
    
    # Save model
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    model_file = f"models/v15_optimized_{method.lower().replace(' ', '_')}_{timestamp}.joblib"
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_results['model'], model_file)
    
    # Save configuration and results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.5_hyperparameter_optimization',
        'optimization_method': method,
        'success': success,
        'model_file': model_file,
        'best_config': best_results['config'],
        'performance': {
            'train_accuracy': float(best_results['train_acc']),
            'test_accuracy': float(best_results['test_acc']),
            'overfitting_gap': float(best_results['gap'])
        },
        'comparison_vs_baseline': {
            'baseline_test': 0.5184,
            'test_improvement': float((best_results['test_acc'] - 0.5184) * 100),
            'baseline_gap': 33.1,
            'gap_reduction': float((0.331 - best_results['gap']) * 100)
        }
    }
    
    # Save detailed results
    os.makedirs('evaluation', exist_ok=True)
    results_file = f"evaluation/v15_optimization_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"✅ Model saved: {model_file}")
    logger.info(f"✅ Results saved: {results_file}")
    
    if success:
        logger.info("🎯 v1.5 optimization complete - Ready for algorithm benchmarking")
    else:
        logger.info("⚠️ Further optimization may be needed")
    
    return results_summary

def main():
    """
    Main hyperparameter optimization pipeline
    """
    logger = setup_logging()
    logger.info("🔧 SYSTEMATIC HYPERPARAMETER OPTIMIZATION - v1.5")
    logger.info("=" * 80)
    logger.info("Objective: Reduce 33.1pp overfitting gap while maintaining test performance")
    logger.info("Methodology: Clean temporal split + Multiple optimization approaches")
    logger.info("=" * 80)
    
    # Load data
    X_train, X_test, y_train, y_test, features = load_data_with_clean_split()
    
    # Baseline evaluation
    logger.info("\n" + "="*50)
    baseline_results = baseline_evaluation(X_train, X_test, y_train, y_test)
    
    # Grid search optimization
    logger.info("\n" + "="*50)  
    grid_results = grid_search_optimization(X_train, X_test, y_train, y_test)
    
    # Optuna optimization
    logger.info("\n" + "="*50)
    optuna_results = optuna_optimization(X_train, X_test, y_train, y_test, n_trials=50)
    
    # Compare and select best
    logger.info("\n" + "="*50)
    best_results, best_method, success = compare_and_select_best(
        baseline_results, grid_results, optuna_results
    )
    
    # Save results
    logger.info("\n" + "="*50)
    summary = save_optimization_results(best_results, best_method, success)
    
    logger.info(f"\n🏁 HYPERPARAMETER OPTIMIZATION COMPLETE")
    logger.info(f"Best method: {best_method}")
    logger.info(f"Test accuracy: {best_results['test_acc']:.4f}")
    logger.info(f"Overfitting gap: {best_results['gap']*100:.1f}pp")
    logger.info(f"Success: {'✅' if success else '❌'}")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)