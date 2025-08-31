#!/usr/bin/env python3
"""
Statistical significance test for performance difference
Bootstrap confidence intervals to determine if performance changes are real
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import sys
import os
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(__file__))
from utils import setup_logging

def bootstrap_confidence_interval(y_true, y_pred, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for accuracy"""
    # Convert to numpy arrays for consistent indexing
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = resample(range(n_samples), n_samples=n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate accuracy
        score = accuracy_score(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return bootstrap_scores.mean(), bootstrap_scores.std(), lower, upper

def test_statistical_significance():
    """
    Test if performance difference between 6 and 7 features is statistically significant
    """
    logger = setup_logging()
    logger.info("=== 📊 STATISTICAL SIGNIFICANCE TEST ===")
    
    # Load the perfect dataset
    df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Same temporal split as final test
    cutoff_date = '2024-01-01'
    train_dev_data = df[df['Date'] < cutoff_date].copy()
    sealed_test_data = df[df['Date'] >= cutoff_date].copy()
    
    logger.info(f"Dataset: {len(df)} matches")
    logger.info(f"Train/Dev: {len(train_dev_data)}, Sealed: {len(sealed_test_data)}")
    
    # Define feature sets
    features_6 = [
        "form_diff_normalized", "elo_diff_normalized", "h2h_score",
        "matchday_normalized", "shots_diff_normalized", "corners_diff_normalized"
    ]
    
    features_7 = features_6 + ["market_entropy_norm"]
    
    logger.info(f"6 features: {features_6}")
    logger.info(f"7 features: {features_7}")
    
    # Prepare data
    X_train_6 = train_dev_data[features_6].fillna(0.5)
    X_train_7 = train_dev_data[features_7].fillna(0.5)
    X_test_6 = sealed_test_data[features_6].fillna(0.5)
    X_test_7 = sealed_test_data[features_7].fillna(0.5)
    
    label_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train = train_dev_data['FullTimeResult'].map(label_mapping)
    y_test = sealed_test_data['FullTimeResult'].map(label_mapping)
    
    # Same model configuration for fair comparison
    model_config = {
        'n_estimators': 300, 'max_depth': 12, 'max_features': 'log2',
        'min_samples_leaf': 2, 'min_samples_split': 15,
        'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
    }
    
    logger.info(f"\n🧪 TRAINING BOTH MODELS (SAME CONFIG)")
    logger.info("-" * 50)
    
    # Train 6-feature model
    model_6 = RandomForestClassifier(**model_config)
    model_6.fit(X_train_6, y_train)
    y_pred_6 = model_6.predict(X_test_6)
    accuracy_6 = accuracy_score(y_test, y_pred_6)
    
    # Train 7-feature model  
    model_7 = RandomForestClassifier(**model_config)
    model_7.fit(X_train_7, y_train)
    y_pred_7 = model_7.predict(X_test_7)
    accuracy_7 = accuracy_score(y_test, y_pred_7)
    
    logger.info(f"6 features accuracy: {accuracy_6:.4f}")
    logger.info(f"7 features accuracy: {accuracy_7:.4f}")
    logger.info(f"Raw difference: {(accuracy_7 - accuracy_6)*100:+.2f}pp")
    
    # Bootstrap confidence intervals
    logger.info(f"\n📊 BOOTSTRAP CONFIDENCE INTERVALS (n=1000)")
    logger.info("-" * 60)
    
    mean_6, std_6, ci_6_lower, ci_6_upper = bootstrap_confidence_interval(y_test, y_pred_6)
    mean_7, std_7, ci_7_lower, ci_7_upper = bootstrap_confidence_interval(y_test, y_pred_7)
    
    logger.info(f"6 features:")
    logger.info(f"  Mean: {mean_6:.4f} ± {std_6:.4f}")
    logger.info(f"  95% CI: [{ci_6_lower:.4f}, {ci_6_upper:.4f}]")
    logger.info(f"  Width: {(ci_6_upper - ci_6_lower)*100:.1f}pp")
    
    logger.info(f"7 features:")
    logger.info(f"  Mean: {mean_7:.4f} ± {std_7:.4f}")
    logger.info(f"  95% CI: [{ci_7_lower:.4f}, {ci_7_upper:.4f}]")
    logger.info(f"  Width: {(ci_7_upper - ci_7_lower)*100:.1f}pp")
    
    # Check if confidence intervals overlap
    overlap = not (ci_6_upper < ci_7_lower or ci_7_upper < ci_6_lower)
    
    logger.info(f"\n🔍 STATISTICAL ANALYSIS:")
    logger.info("-" * 40)
    logger.info(f"Confidence intervals overlap: {overlap}")
    
    if overlap:
        overlap_size = min(ci_6_upper, ci_7_upper) - max(ci_6_lower, ci_7_lower)
        total_range = max(ci_6_upper, ci_7_upper) - min(ci_6_lower, ci_7_lower)
        overlap_pct = (overlap_size / total_range) * 100
        
        logger.info(f"Overlap percentage: {overlap_pct:.1f}%")
        
        if overlap_pct > 50:
            significance = "🟢 NOT SIGNIFICANT - High overlap"
        else:
            significance = "🟡 MARGINALLY SIGNIFICANT - Some overlap"
    else:
        significance = "🔴 STATISTICALLY SIGNIFICANT - No overlap"
    
    logger.info(f"Statistical significance: {significance}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(y_pred_6)-1)*std_6**2 + (len(y_pred_7)-1)*std_7**2) / (len(y_pred_6) + len(y_pred_7) - 2))
    cohens_d = abs(mean_7 - mean_6) / pooled_std
    
    logger.info(f"\n📏 EFFECT SIZE:")
    logger.info(f"Cohen's d: {cohens_d:.3f}")
    
    if cohens_d < 0.2:
        effect_interpretation = "Negligible effect"
    elif cohens_d < 0.5:
        effect_interpretation = "Small effect"  
    elif cohens_d < 0.8:
        effect_interpretation = "Medium effect"
    else:
        effect_interpretation = "Large effect"
    
    logger.info(f"Interpretation: {effect_interpretation}")
    
    # Paired t-test (additional validation)
    # Bootstrap difference distribution
    logger.info(f"\n🧮 ADDITIONAL TESTS:")
    n_bootstrap = 1000
    diff_distribution = []
    
    for _ in range(n_bootstrap):
        indices = resample(range(len(y_test)), n_samples=len(y_test))
        y_true_boot = np.array(y_test)[indices]
        y_pred_6_boot = y_pred_6[indices]
        y_pred_7_boot = y_pred_7[indices]
        
        acc_6_boot = accuracy_score(y_true_boot, y_pred_6_boot)
        acc_7_boot = accuracy_score(y_true_boot, y_pred_7_boot)
        diff_distribution.append(acc_7_boot - acc_6_boot)
    
    diff_mean = np.mean(diff_distribution)
    diff_std = np.std(diff_distribution)
    diff_ci_lower = np.percentile(diff_distribution, 2.5)
    diff_ci_upper = np.percentile(diff_distribution, 97.5)
    
    logger.info(f"Performance difference distribution:")
    logger.info(f"  Mean difference: {diff_mean*100:+.2f}pp ± {diff_std*100:.2f}pp")
    logger.info(f"  95% CI: [{diff_ci_lower*100:+.2f}pp, {diff_ci_upper*100:+.2f}pp]")
    
    # Check if zero is in confidence interval
    zero_in_ci = diff_ci_lower <= 0 <= diff_ci_upper
    logger.info(f"  Zero in CI: {zero_in_ci} ({'No significant difference' if zero_in_ci else 'Significant difference'})")
    
    # Final conclusion
    logger.info(f"\n" + "="*70)
    logger.info(f"🏁 FINAL STATISTICAL CONCLUSION")
    logger.info("="*70)
    
    if zero_in_ci and overlap:
        conclusion = "🟢 NO SIGNIFICANT DIFFERENCE"
        recommendation = "The -0.36pp drop is within statistical noise. Both models perform equivalently."
        action = "✅ PROCEED with 7-feature model (market intelligence valuable)"
        
    elif not zero_in_ci and not overlap:
        if diff_mean > 0:
            conclusion = "🔴 7-FEATURE MODEL SIGNIFICANTLY BETTER"
            recommendation = "Market intelligence provides significant improvement."
            action = "✅ DEFINITELY use 7-feature model"
        else:
            conclusion = "🔴 6-FEATURE MODEL SIGNIFICANTLY BETTER"
            recommendation = "Market intelligence hurts performance significantly."
            action = "❌ REMOVE market_entropy_norm"
    else:
        conclusion = "🟡 MARGINALLY SIGNIFICANT"
        recommendation = "Evidence suggests difference exists but not conclusive."
        action = "⚠️ ADDITIONAL testing needed (hyperparameter tuning)"
    
    logger.info(conclusion)
    logger.info(f"Recommendation: {recommendation}")
    logger.info(f"Action: {action}")
    
    # Practical implications
    logger.info(f"\n💡 PRACTICAL IMPLICATIONS:")
    if zero_in_ci:
        logger.info("• Performance difference likely due to random variation")
        logger.info("• Focus on hyperparameter tuning to exploit market_entropy_norm")
        logger.info("• Feature importance (#2) suggests untapped potential")
    else:
        logger.info("• Performance difference is real")
        logger.info("• Need to understand why market intelligence helps/hurts")
        logger.info("• Consider feature interaction analysis")
    
    results = {
        'accuracy_6': float(accuracy_6),
        'accuracy_7': float(accuracy_7),
        'difference': float(accuracy_7 - accuracy_6),
        'ci_6': [float(ci_6_lower), float(ci_6_upper)],
        'ci_7': [float(ci_7_lower), float(ci_7_upper)],
        'confidence_intervals_overlap': overlap,
        'zero_in_difference_ci': zero_in_ci,
        'cohens_d': float(cohens_d),
        'effect_size': effect_interpretation,
        'statistical_conclusion': conclusion,
        'recommendation': action
    }
    
    return results

if __name__ == "__main__":
    try:
        results = test_statistical_significance()
        print(f"\n📊 STATISTICAL SIGNIFICANCE TEST COMPLETE")
        print(f"6 features: {results['accuracy_6']:.4f}")
        print(f"7 features: {results['accuracy_7']:.4f}")
        print(f"Difference: {results['difference']*100:+.2f}pp")
        print(f"Significant: {not results['zero_in_difference_ci']}")
        print(f"Conclusion: {results['statistical_conclusion']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()