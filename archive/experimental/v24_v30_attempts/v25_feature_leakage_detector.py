#!/usr/bin/env python3
"""
v25 Feature Leakage Detector
Systematic investigation of individual features for data leakage causing Stage 1 overfitting.

Key Analysis:
1. Single-feature models to isolate overfitting sources
2. Temporal consistency validation per feature
3. Train/test distribution analysis
4. Future information correlation detection
5. Feature stability across time periods

Target: Identify which of the 10 v2.4 features contain subtle data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load v2.4 dataset with comprehensive preprocessing."""
    print("📊 Loading v2.4 dataset...")
    
    # Load the exact dataset that showed overfitting
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v13_xg_corrected_features_2025_09_02_113048.csv'
    df = pd.read_csv(data_path)
    
    print(f"Dataset loaded: {len(df)} matches")
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Target encoding for Stage 1 (Draw vs Non-Draw)
    df['stage1_target'] = (df['FullTimeResult'] == 'D').astype(int)  # 1 for Draw, 0 for Non-Draw
    
    # The exact 10 v2.4 features that showed overfitting
    v24_features = [
        'elo_diff_normalized',
        'market_entropy_norm', 
        'home_xg_eff_10',
        'away_xg_eff_10',
        'shots_diff_normalized',
        'corners_diff_normalized',
        'matchday_normalized',
        'form_diff_normalized',
        'h2h_score',
        'away_goals_sum_5'
    ]
    
    # Verify all features exist
    missing_features = [f for f in v24_features if f not in df.columns]
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        return None, None, None, None
    
    print(f"✅ All 10 v2.4 features found")
    
    # Temporal split (same as cascade model)
    cutoff_date = pd.to_datetime('2023-05-01')  # 89-day gap before 2023-24 season
    
    train_mask = df['Date'] < cutoff_date
    test_mask = df['Date'] >= cutoff_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Train split: {len(train_df)} matches (before {cutoff_date.strftime('%Y-%m-%d')})")
    print(f"Test split: {len(test_df)} matches (from {cutoff_date.strftime('%Y-%m-%d')})")
    
    # Draw distribution analysis
    train_draw_rate = train_df['stage1_target'].mean()
    test_draw_rate = test_df['stage1_target'].mean()
    
    print(f"Train draw rate: {train_draw_rate:.3f}")
    print(f"Test draw rate: {test_draw_rate:.3f}")
    print(f"Distribution shift: {abs(train_draw_rate - test_draw_rate):.3f}")
    
    return train_df, test_df, v24_features, cutoff_date

def analyze_single_feature(feature_name, train_df, test_df):
    """Comprehensive analysis of a single feature for data leakage."""
    print(f"\n🔍 Analyzing feature: {feature_name}")
    print("=" * 50)
    
    results = {
        'feature': feature_name,
        'train_cv_f1': 0,
        'test_f1': 0,
        'overfitting_gap': 0,
        'train_mean': 0,
        'test_mean': 0,
        'distribution_shift': 0,
        'missing_values': 0,
        'temporal_stability': 0,
        'leakage_risk': 'LOW'
    }
    
    # 1. Basic statistics
    train_values = train_df[feature_name].dropna()
    test_values = test_df[feature_name].dropna()
    
    results['train_mean'] = train_values.mean()
    results['test_mean'] = test_values.mean()
    results['distribution_shift'] = abs(results['train_mean'] - results['test_mean'])
    results['missing_values'] = (train_df[feature_name].isna().sum() + test_df[feature_name].isna().sum())
    
    print(f"Train mean: {results['train_mean']:.4f}")
    print(f"Test mean: {results['test_mean']:.4f}")
    print(f"Distribution shift: {results['distribution_shift']:.4f}")
    print(f"Missing values: {results['missing_values']}")
    
    # 2. Temporal stability (coefficient of variation across time periods)
    train_df_sorted = train_df.sort_values('Date')
    n_periods = 4
    period_size = len(train_df_sorted) // n_periods
    
    period_means = []
    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < n_periods - 1 else len(train_df_sorted)
        period_data = train_df_sorted.iloc[start_idx:end_idx]
        period_mean = period_data[feature_name].mean()
        if not pd.isna(period_mean):
            period_means.append(period_mean)
    
    if len(period_means) > 1:
        results['temporal_stability'] = np.std(period_means) / np.mean(period_means) if np.mean(period_means) != 0 else 0
    
    print(f"Temporal stability (CV): {results['temporal_stability']:.4f}")
    
    # 3. Single-feature model performance
    try:
        # Prepare single feature data
        X_train = train_df[[feature_name]].fillna(train_df[feature_name].median())
        y_train = train_df['stage1_target']
        X_test = test_df[[feature_name]].fillna(train_df[feature_name].median())
        y_test = test_df['stage1_target']
        
        # Single feature model with SMOTE (same as cascade)
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Simple RF model
        rf_model = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        # Cross-validation on training data
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(rf_model, X_train_balanced, y_train_balanced, 
                                   cv=tscv, scoring='f1', n_jobs=-1)
        results['train_cv_f1'] = cv_scores.mean()
        
        # Test performance
        rf_model.fit(X_train_balanced, y_train_balanced)
        y_pred = rf_model.predict(X_test)
        results['test_f1'] = f1_score(y_test, y_pred)
        
        # Overfitting gap
        results['overfitting_gap'] = results['train_cv_f1'] - results['test_f1']
        
        print(f"Cross-validation F1: {results['train_cv_f1']:.3f}")
        print(f"Test F1: {results['test_f1']:.3f}")
        print(f"Overfitting gap: {results['overfitting_gap']:.3f}")
        
        # Risk assessment
        if results['overfitting_gap'] > 0.3:
            results['leakage_risk'] = 'CRITICAL'
        elif results['overfitting_gap'] > 0.15:
            results['leakage_risk'] = 'HIGH'
        elif results['overfitting_gap'] > 0.05:
            results['leakage_risk'] = 'MEDIUM'
        else:
            results['leakage_risk'] = 'LOW'
            
        print(f"Leakage risk: {results['leakage_risk']}")
        
    except Exception as e:
        print(f"❌ Model analysis failed: {str(e)}")
        results['leakage_risk'] = 'ERROR'
    
    return results

def main():
    """Main feature leakage investigation."""
    print("🔬 v25 Feature Leakage Detector")
    print("=" * 60)
    print("Systematic investigation of 10 v2.4 features for data leakage")
    print("Target: Identify source of 78.71% Stage 1 overfitting gap")
    print()
    
    # Load data
    train_df, test_df, v24_features, cutoff_date = load_and_prepare_data()
    if train_df is None:
        print("❌ Failed to load data")
        return
    
    # Analyze each feature individually
    all_results = []
    
    for feature in v24_features:
        results = analyze_single_feature(feature, train_df, test_df)
        all_results.append(results)
    
    # Summary analysis
    print("\n📋 FEATURE LEAKAGE ANALYSIS SUMMARY")
    print("=" * 60)
    
    results_df = pd.DataFrame(all_results)
    
    # Sort by overfitting gap (descending)
    results_df = results_df.sort_values('overfitting_gap', ascending=False)
    
    print("Features ranked by overfitting gap (highest = most suspicious):")
    print()
    
    for idx, row in results_df.iterrows():
        risk_emoji = {
            'CRITICAL': '🔴',
            'HIGH': '🟠', 
            'MEDIUM': '🟡',
            'LOW': '🟢',
            'ERROR': '❌'
        }
        
        print(f"{risk_emoji.get(row['leakage_risk'], '❓')} {row['feature']:<25} "
              f"Gap: {row['overfitting_gap']:>6.3f} "
              f"CV: {row['train_cv_f1']:>5.3f} "
              f"Test: {row['test_f1']:>5.3f} "
              f"Risk: {row['leakage_risk']}")
    
    # Critical findings
    critical_features = results_df[results_df['leakage_risk'] == 'CRITICAL']
    high_risk_features = results_df[results_df['leakage_risk'] == 'HIGH']
    
    print(f"\n🚨 CRITICAL FINDINGS:")
    print(f"Features with CRITICAL leakage risk: {len(critical_features)}")
    print(f"Features with HIGH leakage risk: {len(high_risk_features)}")
    
    if len(critical_features) > 0:
        print(f"\n🔴 CRITICAL FEATURES (>30% overfitting gap):")
        for _, row in critical_features.iterrows():
            print(f"  - {row['feature']}: {row['overfitting_gap']:.1%} gap")
    
    if len(high_risk_features) > 0:
        print(f"\n🟠 HIGH RISK FEATURES (>15% overfitting gap):")
        for _, row in high_risk_features.iterrows():
            print(f"  - {row['feature']}: {row['overfitting_gap']:.1%} gap")
    
    # Save detailed results
    timestamp = pd.Timestamp.now().strftime('%Y_%m_%d_%H%M%S')
    output_file = f'evaluation/reports/feature_leakage_analysis_{timestamp}.json'
    
    results_dict = {
        'analysis_timestamp': timestamp,
        'dataset_path': '/Users/maxime/Desktop/Oddsy/data/processed/v13_xg_corrected_features_2025_09_02_113048.csv',
        'temporal_cutoff': cutoff_date.strftime('%Y-%m-%d'),
        'train_matches': len(train_df),
        'test_matches': len(test_df),
        'features_analyzed': len(v24_features),
        'critical_features': len(critical_features),
        'high_risk_features': len(high_risk_features),
        'feature_results': results_df.to_dict('records')
    }
    
    import json
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print(f"\n✅ Feature leakage investigation complete!")
    
    # Next steps recommendation
    if len(critical_features) > 0 or len(high_risk_features) > 0:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Remove or fix features with CRITICAL/HIGH leakage risk")
        print(f"2. Test cascade model with clean feature subset")
        print(f"3. Validate performance without leaky features")
    else:
        print(f"\n🤔 UNEXPECTED: No high-risk features found")
        print(f"Investigation may need to look at feature interactions or other factors")

if __name__ == "__main__":
    main()