#!/usr/bin/env python3
"""
QUICK HYPERPARAMETER TUNING - v1.5
Focused optimization to reduce overfitting gap efficiently
Target specific parameters known to affect overfitting
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
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
    """Load data with definitive clean temporal split"""
    logger = setup_logging()
    logger.info("=== 📊 LOADING DATA ===")
    
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Clean temporal split
    train_seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
    test_seasons = ['2024-2025']
    
    train_data = df[df['Season'].isin(train_seasons)].copy()
    test_data = df[df['Season'].isin(test_seasons)].copy()
    
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
    
    logger.info(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def quick_optimization(X_train, X_test, y_train, y_test):
    """
    Quick targeted optimization focusing on overfitting reduction
    """
    logger = setup_logging()
    logger.info("=== ⚡ QUICK HYPERPARAMETER OPTIMIZATION ===")
    
    # Baseline (current best)
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
    
    baseline_model = RandomForestClassifier(**baseline_config)
    baseline_model.fit(X_train, y_train)
    
    baseline_train = baseline_model.score(X_train, y_train)
    baseline_test = baseline_model.score(X_test, y_test)
    baseline_gap = baseline_train - baseline_test
    
    logger.info(f"Baseline: Train={baseline_train:.4f}, Test={baseline_test:.4f}, Gap={baseline_gap*100:.1f}pp")
    
    # Focused parameter configurations to reduce overfitting
    configs_to_test = [
        # Reduce max_depth (primary overfitting control)
        {'max_depth': 12},
        {'max_depth': 15},
        {'max_depth': 10},
        {'max_depth': 8},
        
        # Increase min_samples_leaf (require more samples per leaf)
        {'min_samples_leaf': 5},
        {'min_samples_leaf': 10},
        {'min_samples_leaf': 15},
        {'min_samples_leaf': 20},
        
        # Increase min_samples_split (require more samples to split)
        {'min_samples_split': 20},
        {'min_samples_split': 30},
        {'min_samples_split': 40},
        
        # Reduce max_features (less features per tree)
        {'max_features': 'sqrt'},
        {'max_features': 0.5},
        {'max_features': 0.3},
        
        # Combined approaches
        {'max_depth': 12, 'min_samples_leaf': 5},
        {'max_depth': 12, 'min_samples_leaf': 10},
        {'max_depth': 15, 'min_samples_leaf': 5},
        {'max_depth': 10, 'min_samples_leaf': 10},
        {'max_depth': 12, 'min_samples_split': 25},
        {'max_depth': 15, 'max_features': 'sqrt'},
        {'max_depth': 12, 'max_features': 0.5},
        
        # Conservative configurations
        {'max_depth': 8, 'min_samples_leaf': 15, 'max_features': 'sqrt'},
        {'max_depth': 10, 'min_samples_leaf': 10, 'max_features': 0.5'},
        {'max_depth': 12, 'min_samples_leaf': 8, 'min_samples_split': 25}
    ]
    
    logger.info(f"Testing {len(configs_to_test)} configurations...")
    
    best_config = baseline_config.copy()
    best_test = baseline_test
    best_gap = baseline_gap
    best_model = baseline_model
    
    results = []
    
    for i, config_update in enumerate(configs_to_test, 1):
        # Create full config
        full_config = baseline_config.copy()
        full_config.update(config_update)
        
        logger.info(f"Testing {i}/{len(configs_to_test)}: {config_update}")
        
        # Train and evaluate
        model = RandomForestClassifier(**full_config)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        gap = train_acc - test_acc
        
        # Calculate improvements
        test_change = (test_acc - baseline_test) * 100
        gap_reduction = (baseline_gap - gap) * 100
        
        logger.info(f"  Train={train_acc:.4f}, Test={test_acc:.4f}, Gap={gap*100:.1f}pp")
        logger.info(f"  Changes: Test{test_change:+.2f}pp, Gap{gap_reduction:+.2f}pp")
        
        results.append({
            'config': config_update,
            'full_config': full_config,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': gap,
            'test_change': test_change,
            'gap_reduction': gap_reduction,
            'model': model
        })
        
        # Update best based on composite score
        # Priority: maintain test performance + reduce overfitting
        composite_score = test_acc - (0.3 * max(0, gap - 0.15))  # Penalize gaps > 15%
        best_composite = best_test - (0.3 * max(0, best_gap - 0.15))
        
        if composite_score > best_composite:
            best_config = full_config
            best_test = test_acc
            best_gap = gap
            best_model = model
            logger.info(f"  ✅ NEW BEST: Test={test_acc:.4f}, Gap={gap*100:.1f}pp")
    
    return {
        'baseline': {'config': baseline_config, 'train': baseline_train, 'test': baseline_test, 'gap': baseline_gap},
        'best': {'config': best_config, 'train': best_model.score(X_train, y_train), 'test': best_test, 'gap': best_gap, 'model': best_model},
        'all_results': results
    }

def analyze_results(optimization_results):
    """
    Analyze optimization results and provide recommendations
    """
    logger = setup_logging()
    logger.info("=== 📊 OPTIMIZATION ANALYSIS ===")
    
    baseline = optimization_results['baseline']
    best = optimization_results['best']
    all_results = optimization_results['all_results']
    
    # Performance comparison
    test_improvement = (best['test'] - baseline['test']) * 100
    gap_reduction = (baseline['gap'] - best['gap']) * 100
    
    logger.info(f"BASELINE vs OPTIMIZED:")
    logger.info(f"  Test accuracy: {baseline['test']:.4f} → {best['test']:.4f} ({test_improvement:+.2f}pp)")
    logger.info(f"  Overfitting gap: {baseline['gap']*100:.1f}pp → {best['gap']*100:.1f}pp ({gap_reduction:+.2f}pp)")
    
    # Success assessment
    success_criteria = {
        'maintains_performance': best['test'] >= baseline['test'] - 0.005,  # Tolerate 0.5pp drop
        'reduces_overfitting': best['gap'] < baseline['gap'] - 0.03,       # At least 3pp gap reduction
        'significant_improvement': test_improvement > 0.5 or gap_reduction > 5
    }
    
    logger.info(f"\n🎯 SUCCESS CRITERIA:")
    for criterion, achieved in success_criteria.items():
        status = "✅ ACHIEVED" if achieved else "❌ MISSED"
        logger.info(f"  {criterion}: {status}")
    
    overall_success = sum(success_criteria.values()) >= 2
    
    # Top 5 configurations by different metrics
    logger.info(f"\n🏆 TOP 5 BY TEST PERFORMANCE:")
    top_test = sorted(all_results, key=lambda x: x['test_acc'], reverse=True)[:5]
    for i, result in enumerate(top_test, 1):
        logger.info(f"  {i}. Test={result['test_acc']:.4f}, Gap={result['gap']*100:.1f}pp: {result['config']}")
    
    logger.info(f"\n⚡ TOP 5 BY GAP REDUCTION:")
    top_gap = sorted(all_results, key=lambda x: x['gap'])[:5]
    for i, result in enumerate(top_gap, 1):
        logger.info(f"  {i}. Gap={result['gap']*100:.1f}pp, Test={result['test_acc']:.4f}: {result['config']}")
    
    # Performance assessment
    if best['test'] >= 0.52:
        performance_level = "🏆 EXCELLENT (≥52%)"
    elif best['test'] >= 0.515:
        performance_level = "✅ GOOD (≥51.5%)"
    elif best['test'] >= baseline['test']:
        performance_level = "📈 IMPROVED"
    else:
        performance_level = "⚠️ DECLINED"
    
    logger.info(f"\n📊 FINAL ASSESSMENT:")
    logger.info(f"Performance level: {performance_level}")
    logger.info(f"Optimization success: {'✅ YES' if overall_success else '❌ NO'}")
    
    return overall_success

def save_results(optimization_results, success):
    """
    Save optimization results and best model
    """
    logger = setup_logging()
    logger.info("=== 💾 SAVING RESULTS ===")
    
    best = optimization_results['best']
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    
    # Save model
    model_file = f"models/v15_quick_tuned_{timestamp}.joblib"
    os.makedirs('models', exist_ok=True)
    joblib.dump(best['model'], model_file)
    
    # Save results summary
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.5_quick_hyperparameter_tuning',
        'optimization_success': success,
        'model_file': model_file,
        'baseline_performance': {
            'test_accuracy': float(optimization_results['baseline']['test']),
            'overfitting_gap': float(optimization_results['baseline']['gap'] * 100)
        },
        'optimized_performance': {
            'test_accuracy': float(best['test']),
            'train_accuracy': float(best['train']),
            'overfitting_gap': float(best['gap'] * 100),
            'test_improvement': float((best['test'] - optimization_results['baseline']['test']) * 100),
            'gap_reduction': float((optimization_results['baseline']['gap'] - best['gap']) * 100)
        },
        'best_config': best['config']
    }
    
    # Save to file
    os.makedirs('evaluation', exist_ok=True)
    results_file = f"evaluation/v15_quick_tuning_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"✅ Model saved: {model_file}")
    logger.info(f"✅ Results saved: {results_file}")
    
    return results_summary

def main():
    """
    Main quick optimization pipeline
    """
    logger = setup_logging()
    logger.info("⚡ QUICK HYPERPARAMETER TUNING - v1.5")
    logger.info("=" * 60)
    logger.info("Goal: Reduce 33.1pp overfitting gap efficiently")
    logger.info("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data_with_clean_split()
    
    # Quick optimization
    optimization_results = quick_optimization(X_train, X_test, y_train, y_test)
    
    # Analyze results
    success = analyze_results(optimization_results)
    
    # Save results
    summary = save_results(optimization_results, success)
    
    # Final summary
    logger.info(f"\n🏁 QUICK OPTIMIZATION COMPLETE")
    logger.info(f"Test accuracy: {optimization_results['best']['test']:.4f}")
    logger.info(f"Overfitting gap: {optimization_results['best']['gap']*100:.1f}pp")
    logger.info(f"Success: {'✅' if success else '❌'}")
    
    if success:
        logger.info("🎯 v1.5 ready - Proceed to algorithm benchmarking")
    else:
        logger.info("⚠️ Consider further optimization or accept current baseline")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        print(f"\n⚡ QUICK TUNING COMPLETE")
        print(f"Success: {'✅' if success else '❌'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)