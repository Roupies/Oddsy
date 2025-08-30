import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def debug_baseline_v12():
    """
    CRITICAL DEBUGGING: Why is v1.2 baseline 50.95% instead of 52.2%?
    
    Investigation Plan:
    1. Compare with historical v1.3 results (52.4%)
    2. Check data quality and feature ranges
    3. Test different CV strategies
    4. Validate against known baselines
    5. Identify the exact bug/issue
    """
    
    logger = setup_logging()
    logger.info("=== 🐛 BASELINE DEBUGGER v1.2 - Critical Investigation ===")
    
    # Load current dataset
    current_file = "data/processed/premier_league_ml_ready_v20_2025_08_30_201711.csv"
    logger.info(f"Loading current dataset: {current_file}")
    
    df_current = pd.read_csv(current_file)
    logger.info(f"Current dataset: {df_current.shape}")
    
    # v1.2 baseline features
    v12_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'home_advantage', 'matchday_normalized', 'season_period_numeric',
        'shots_diff_normalized', 'corners_diff_normalized'
    ]
    
    # Prepare target
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_current = df_current['FullTimeResult'].map(label_mapping)
    X_current = df_current[v12_features]
    
    logger.info(f"Features: {v12_features}")
    logger.info(f"Target distribution: {y_current.value_counts(normalize=True).sort_index().round(3).to_dict()}")
    
    # =========================
    # 1. COMPARE WITH HISTORICAL RESULTS
    # =========================
    logger.info("\n🔍 1. HISTORICAL PERFORMANCE COMPARISON")
    
    # Load historical results from v1.3 (should have v1.2 baseline)
    historical_files = [
        "models/ensemble_v13_results_2025_08_30_173917.json",
        "models/feature_interactions_v13_results_2025_08_30_174314.json"
    ]
    
    import json
    historical_baselines = {}
    
    for file in historical_files:
        if os.path.exists(file):
            logger.info(f"  Loading historical results: {file}")
            with open(file, 'r') as f:
                data = json.load(f)
                
            # Extract baseline performance
            if 'individual_models' in data:
                for model, results in data['individual_models'].items():
                    if 'Random Forest' in model:
                        historical_baselines[f"{file}_RF"] = results['mean']
            
            if 'all_configurations' in data:
                for config, config_data in data['all_configurations'].items():
                    if 'Baseline' in config and 'Random Forest' in config_data.get('results', {}):
                        rf_result = config_data['results']['Random Forest']
                        historical_baselines[f"{file}_Baseline_RF"] = rf_result['mean']
    
    logger.info("  Historical v1.2 baselines found:")
    for source, performance in historical_baselines.items():
        logger.info(f"    {source}: {performance:.4f}")
    
    if historical_baselines:
        avg_historical = np.mean(list(historical_baselines.values()))
        logger.info(f"    Average historical: {avg_historical:.4f}")
    else:
        logger.warning("    No historical baselines found!")
        avg_historical = 0.522  # Use documented baseline
    
    # =========================
    # 2. TEST CURRENT MODEL WITH SAME SETTINGS
    # =========================
    logger.info("\n🔍 2. CURRENT MODEL PERFORMANCE")
    
    # Use same RF settings as historical
    rf_current = RandomForestClassifier(
        n_estimators=300, max_depth=12, max_features='log2',
        min_samples_leaf=2, min_samples_split=15, 
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # Test with TimeSeriesSplit (same as historical)
    cv_current = cross_val_score(rf_current, X_current, y_current, 
                                cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')
    current_mean = cv_current.mean()
    
    logger.info(f"  Current performance: {current_mean:.4f} (±{cv_current.std():.4f})")
    logger.info(f"  CV scores: {[f'{s:.3f}' for s in cv_current]}")
    
    performance_gap = (current_mean - avg_historical) * 100
    logger.info(f"  Gap vs historical: {performance_gap:+.2f}pp")
    
    if performance_gap < -1.0:
        logger.error("  🚨 SIGNIFICANT PERFORMANCE DROP DETECTED!")
    
    # =========================
    # 3. DATA QUALITY INVESTIGATION
    # =========================
    logger.info("\n🔍 3. DATA QUALITY ANALYSIS")
    
    # Check feature statistics
    logger.info("  Feature Statistics:")
    for feature in v12_features:
        values = X_current[feature]
        logger.info(f"    {feature}:")
        logger.info(f"      Range: [{values.min():.3f}, {values.max():.3f}]")
        logger.info(f"      Mean: {values.mean():.3f}, Std: {values.std():.3f}")
        logger.info(f"      Missing: {values.isnull().sum()}")
        
        # Check for anomalies
        if values.std() < 0.001:
            logger.warning(f"      🚨 ZERO VARIANCE: {feature} has no variation!")
        if (values == 0.5).mean() > 0.8:
            logger.warning(f"      🚨 DEFAULT VALUES: {feature} is mostly 0.5 (neutral)!")
        if values.isnull().sum() > 0:
            logger.error(f"      🚨 MISSING DATA: {feature} has {values.isnull().sum()} NaN values!")
    
    # Check target distribution
    target_dist = y_current.value_counts(normalize=True).sort_index()
    logger.info(f"  Target distribution: H={target_dist[0]:.3f}, D={target_dist[1]:.3f}, A={target_dist[2]:.3f}")
    
    expected_dist = {'H': 0.436, 'D': 0.230, 'A': 0.334}  # Natural PL distribution
    
    for i, outcome in enumerate(['H', 'D', 'A']):
        expected = expected_dist[outcome]
        actual = target_dist[i]
        diff = abs(actual - expected)
        if diff > 0.05:
            logger.warning(f"    🚨 TARGET DISTRIBUTION ANOMALY: {outcome} is {actual:.3f}, expected ~{expected:.3f}")
    
    # =========================
    # 4. FEATURE ENGINEERING VALIDATION
    # =========================
    logger.info("\n🔍 4. FEATURE ENGINEERING VALIDATION")
    
    # Check for data leakage - features should not be perfect predictors
    logger.info("  Data Leakage Check:")
    
    rf_leak_check = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_leak_check.fit(X_current, y_current)
    
    train_accuracy = rf_leak_check.score(X_current, y_current)
    logger.info(f"    Training accuracy: {train_accuracy:.4f}")
    
    if train_accuracy > 0.95:
        logger.error("    🚨 POSSIBLE DATA LEAKAGE: Perfect training accuracy!")
    elif train_accuracy > 0.80:
        logger.warning("    ⚠️ SUSPICIOUS: Very high training accuracy - check for leakage")
    else:
        logger.info("    ✅ Training accuracy looks normal")
    
    # Check individual feature predictive power
    logger.info("  Individual Feature Predictive Power:")
    for feature in v12_features:
        X_single = X_current[[feature]]
        cv_single = cross_val_score(rf_leak_check, X_single, y_current, 
                                  cv=TimeSeriesSplit(n_splits=3), scoring='accuracy')
        single_mean = cv_single.mean()
        logger.info(f"    {feature}: {single_mean:.3f}")
        
        if single_mean > 0.60:
            logger.warning(f"      ⚠️ HIGH SINGLE FEATURE POWER: {feature} alone gives {single_mean:.3f}")
    
    # =========================
    # 5. CROSS-VALIDATION METHODOLOGY CHECK
    # =========================
    logger.info("\n🔍 5. CROSS-VALIDATION METHODOLOGY")
    
    # Test different CV strategies
    cv_strategies = {
        'TimeSeriesSplit (n=5)': TimeSeriesSplit(n_splits=5),
        'TimeSeriesSplit (n=3)': TimeSeriesSplit(n_splits=3),
    }
    
    for cv_name, cv_strategy in cv_strategies.items():
        cv_scores = cross_val_score(rf_current, X_current, y_current, 
                                  cv=cv_strategy, scoring='accuracy')
        cv_mean = cv_scores.mean()
        logger.info(f"  {cv_name}: {cv_mean:.4f} (±{cv_scores.std():.4f})")
        logger.info(f"    Scores: {[f'{s:.3f}' for s in cv_scores]}")
    
    # =========================
    # 6. BASELINE VALIDATION TESTS
    # =========================
    logger.info("\n🔍 6. BASELINE VALIDATION TESTS")
    
    # Test naive baselines
    from sklearn.dummy import DummyClassifier
    
    # Majority class baseline
    dummy_majority = DummyClassifier(strategy='most_frequent')
    cv_majority = cross_val_score(dummy_majority, X_current, y_current, 
                                cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')
    majority_mean = cv_majority.mean()
    
    logger.info(f"  Majority class baseline: {majority_mean:.4f}")
    logger.info(f"    Expected: ~0.436 (Home win frequency)")
    
    # Random baseline
    dummy_random = DummyClassifier(strategy='uniform', random_state=42)
    cv_random = cross_val_score(dummy_random, X_current, y_current, 
                              cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')
    random_mean = cv_random.mean()
    
    logger.info(f"  Random baseline: {random_mean:.4f}")
    logger.info(f"    Expected: ~0.333 (Random guess)")
    
    # Check if our model beats baselines
    beats_majority = current_mean > majority_mean
    beats_random = current_mean > random_mean
    
    logger.info(f"  Beats majority: {'✅' if beats_majority else '❌'} ({(current_mean - majority_mean)*100:+.1f}pp)")
    logger.info(f"  Beats random: {'✅' if beats_random else '❌'} ({(current_mean - random_mean)*100:+.1f}pp)")
    
    # =========================
    # 7. DIAGNOSIS & RECOMMENDATIONS
    # =========================
    logger.info("\n🎯 7. DEBUGGING DIAGNOSIS")
    
    # Collect evidence
    issues_found = []
    
    if performance_gap < -1.0:
        issues_found.append("PERFORMANCE_REGRESSION")
    
    # Check for zero variance features
    zero_variance_features = []
    mostly_neutral_features = []
    for feature in v12_features:
        values = X_current[feature]
        if values.std() < 0.001:
            zero_variance_features.append(feature)
        if (values == 0.5).mean() > 0.8:
            mostly_neutral_features.append(feature)
    
    if zero_variance_features:
        issues_found.append(f"ZERO_VARIANCE_FEATURES: {zero_variance_features}")
    if mostly_neutral_features:
        issues_found.append(f"MOSTLY_NEUTRAL_FEATURES: {mostly_neutral_features}")
    
    if train_accuracy > 0.95:
        issues_found.append("POSSIBLE_DATA_LEAKAGE")
    
    if not beats_majority:
        issues_found.append("FAILS_MAJORITY_BASELINE")
    
    logger.info("  Issues Identified:")
    if issues_found:
        for issue in issues_found:
            logger.error(f"    🚨 {issue}")
    else:
        logger.info("    ✅ No obvious issues detected")
    
    # Make recommendation
    logger.info("\n🚀 DEBUGGING RECOMMENDATION:")
    
    if zero_variance_features or mostly_neutral_features:
        recommendation = "FEATURE_ENGINEERING_BUG"
        logger.info("  🔧 FEATURE ENGINEERING ISSUE DETECTED")
        logger.info("     - Features are not properly calculated or normalized")
        logger.info("     - Check feature engineering pipeline for bugs")
        logger.info(f"     - Zero variance: {zero_variance_features}")
        logger.info(f"     - Mostly neutral: {mostly_neutral_features}")
        
    elif performance_gap < -2.0:
        recommendation = "DATA_QUALITY_ISSUE"
        logger.info("  📊 DATA QUALITY ISSUE")
        logger.info("     - Significant performance regression detected")
        logger.info("     - Check dataset integrity and preprocessing")
        logger.info("     - Compare with original processed data")
        
    elif train_accuracy > 0.95:
        recommendation = "DATA_LEAKAGE_SUSPECTED"
        logger.info("  🔒 POSSIBLE DATA LEAKAGE")
        logger.info("     - Training accuracy too high")
        logger.info("     - Check temporal integrity of features")
        logger.info("     - Ensure no future information in historical features")
        
    else:
        recommendation = "METHODOLOGY_ISSUE"
        logger.info("  🔬 METHODOLOGY ISSUE")
        logger.info("     - Model performance inconsistent with historical results")
        logger.info("     - Check cross-validation methodology")
        logger.info("     - Verify model hyperparameters match historical setup")
    
    # Save debugging results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    debug_results = {
        "timestamp": timestamp,
        "current_performance": float(current_mean),
        "historical_average": float(avg_historical) if historical_baselines else None,
        "performance_gap_pp": float(performance_gap),
        "majority_baseline": float(majority_mean),
        "random_baseline": float(random_mean),
        "beats_majority": bool(beats_majority),
        "beats_random": bool(beats_random),
        "issues_found": issues_found,
        "recommendation": recommendation,
        "feature_stats": {
            feature: {
                "mean": float(X_current[feature].mean()),
                "std": float(X_current[feature].std()),
                "min": float(X_current[feature].min()),
                "max": float(X_current[feature].max()),
                "mostly_neutral": bool((X_current[feature] == 0.5).mean() > 0.8)
            } for feature in v12_features
        }
    }
    
    results_file = f"models/baseline_debug_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(debug_results, f, indent=2)
    
    logger.info(f"\n📊 Debug results saved: {results_file}")
    logger.info("=== 🐛 BASELINE DEBUGGING COMPLETED ===")
    
    return {
        'recommendation': recommendation,
        'current_performance': current_mean,
        'performance_gap': performance_gap,
        'issues_found': issues_found,
        'results_file': results_file
    }

if __name__ == "__main__":
    result = debug_baseline_v12()
    print(f"\n🎯 DEBUGGING RESULTS:")
    print(f"Current Performance: {result['current_performance']:.4f}")
    print(f"Performance Gap: {result['performance_gap']:+.2f}pp")
    print(f"Issues Found: {len(result['issues_found'])}")
    print(f"Recommendation: {result['recommendation']}")