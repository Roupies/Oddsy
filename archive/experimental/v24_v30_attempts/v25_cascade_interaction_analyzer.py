#!/usr/bin/env python3
"""
v25 Cascade Interaction Analyzer
Investigation of feature interactions causing cascade overfitting despite clean individual features.

Key Discovery: Individual features show NEGATIVE overfitting (better test than CV performance)
But cascade model shows 78.71% POSITIVE overfitting gap.

Hypothesis: Feature interactions or SMOTE+cascade architecture creating synthetic overfitting.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_cascade_data():
    """Load and prepare data exactly as cascade model."""
    print("📊 Loading cascade data with exact v2.4 preprocessing...")
    
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v13_xg_corrected_features_2025_09_02_113048.csv'
    df = pd.read_csv(data_path)
    
    # Convert Date and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Target encoding
    df['stage1_target'] = (df['FullTimeResult'] == 'D').astype(int)
    
    # v2.4 features
    features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized', 
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    # Temporal split (exact cascade methodology)
    cutoff_date = pd.to_datetime('2023-05-01')
    train_mask = df['Date'] < cutoff_date
    test_mask = df['Date'] >= cutoff_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Train: {len(train_df)} matches | Test: {len(test_df)} matches")
    
    return train_df, test_df, features

def test_progressive_feature_combinations(train_df, test_df, features):
    """Test progressive feature combinations to isolate interaction effects."""
    print("\n🔬 PROGRESSIVE FEATURE COMBINATION ANALYSIS")
    print("=" * 60)
    
    results = []
    
    # Test 1: Single best feature (market_entropy_norm had best individual performance)
    print(f"1️⃣ Testing single best feature: market_entropy_norm")
    result = test_feature_subset(train_df, test_df, ['market_entropy_norm'])
    result['combination'] = 'single_best'
    result['n_features'] = 1
    results.append(result)
    
    # Test 2: Top 2 features
    top_features = ['market_entropy_norm', 'home_xg_eff_10']  # Best 2 from individual analysis
    print(f"2️⃣ Testing top 2 features: {top_features}")
    result = test_feature_subset(train_df, test_df, top_features)
    result['combination'] = 'top_2'
    result['n_features'] = 2
    results.append(result)
    
    # Test 3: Top 5 features
    top5_features = ['market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10', 
                     'shots_diff_normalized', 'elo_diff_normalized']
    print(f"3️⃣ Testing top 5 features")
    result = test_feature_subset(train_df, test_df, top5_features)
    result['combination'] = 'top_5'
    result['n_features'] = 5
    results.append(result)
    
    # Test 4: All 10 features (current v2.4)
    print(f"4️⃣ Testing all 10 features (current v2.4)")
    result = test_feature_subset(train_df, test_df, features)
    result['combination'] = 'all_10'
    result['n_features'] = 10
    results.append(result)
    
    return results

def test_smote_impact(train_df, test_df, features):
    """Test cascade model with and without SMOTE to isolate SMOTE impact."""
    print("\n⚖️ SMOTE IMPACT ANALYSIS")
    print("=" * 40)
    
    results = []
    
    # Test 1: Without SMOTE
    print("1️⃣ Testing cascade WITHOUT SMOTE")
    result = test_cascade_variant(train_df, test_df, features, use_smote=False)
    result['variant'] = 'no_smote'
    results.append(result)
    
    # Test 2: With SMOTE (current)
    print("2️⃣ Testing cascade WITH SMOTE (current)")
    result = test_cascade_variant(train_df, test_df, features, use_smote=True)
    result['variant'] = 'with_smote'
    results.append(result)
    
    return results

def test_feature_subset(train_df, test_df, feature_subset):
    """Test cascade model with specific feature subset."""
    
    # Prepare data
    X_train = train_df[feature_subset].fillna(train_df[feature_subset].median())
    y_train = train_df['stage1_target']
    X_test = test_df[feature_subset].fillna(train_df[feature_subset].median())
    y_test = test_df['stage1_target']
    
    # SMOTE oversampling
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"   Original train balance: {y_train.sum()}/{len(y_train)} = {y_train.mean():.3f}")
    print(f"   SMOTE train balance: {y_train_balanced.sum()}/{len(y_train_balanced)} = {y_train_balanced.mean():.3f}")
    
    # Stage 1 Model (Draw vs Non-Draw)
    stage1_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=42
    )
    
    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(stage1_model, X_train_balanced, y_train_balanced, 
                               cv=tscv, scoring='f1', n_jobs=-1)
    cv_f1 = cv_scores.mean()
    
    # Test performance
    stage1_model.fit(X_train_balanced, y_train_balanced)
    y_pred = stage1_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    
    # Results
    overfitting_gap = cv_f1 - test_f1
    
    print(f"   CV F1: {cv_f1:.3f} | Test F1: {test_f1:.3f} | Gap: {overfitting_gap:.3f}")
    
    return {
        'features': feature_subset,
        'cv_f1': cv_f1,
        'test_f1': test_f1,
        'overfitting_gap': overfitting_gap,
        'train_samples': len(X_train_balanced),
        'test_samples': len(X_test)
    }

def test_cascade_variant(train_df, test_df, features, use_smote=True):
    """Test cascade model variant with/without SMOTE."""
    
    # Prepare data
    X_train = train_df[features].fillna(train_df[features].median())
    y_train = train_df['stage1_target']
    X_test = test_df[features].fillna(train_df[features].median())
    y_test = test_df['stage1_target']
    
    if use_smote:
        # SMOTE oversampling
        smote = SMOTE(random_state=42)
        X_train_processed, y_train_processed = smote.fit_resample(X_train, y_train)
        print(f"   SMOTE: {len(X_train)} → {len(X_train_processed)} samples")
    else:
        # No SMOTE
        X_train_processed, y_train_processed = X_train, y_train
        print(f"   No SMOTE: {len(X_train)} samples")
    
    # Stage 1 Model
    stage1_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=42
    )
    
    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(stage1_model, X_train_processed, y_train_processed, 
                               cv=tscv, scoring='f1', n_jobs=-1)
    cv_f1 = cv_scores.mean()
    
    # Test performance
    stage1_model.fit(X_train_processed, y_train_processed)
    y_pred = stage1_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    
    overfitting_gap = cv_f1 - test_f1
    
    print(f"   CV F1: {cv_f1:.3f} | Test F1: {test_f1:.3f} | Gap: {overfitting_gap:.3f}")
    
    return {
        'cv_f1': cv_f1,
        'test_f1': test_f1,
        'overfitting_gap': overfitting_gap,
        'use_smote': use_smote,
        'train_samples': len(X_train_processed)
    }

def analyze_synthetic_data_quality(train_df, features):
    """Analyze SMOTE synthetic data quality."""
    print("\n🔍 SMOTE SYNTHETIC DATA QUALITY ANALYSIS")
    print("=" * 50)
    
    # Prepare original data
    X_train = train_df[features].fillna(train_df[features].median())
    y_train = train_df['stage1_target']
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    
    # Identify synthetic samples
    original_indices = set(range(len(X_train)))
    synthetic_mask = np.array([i not in original_indices for i in range(len(X_smote))])
    
    X_original = X_smote[~synthetic_mask]
    X_synthetic = X_smote[synthetic_mask]
    
    print(f"Original samples: {len(X_original)}")
    print(f"Synthetic samples: {len(X_synthetic)}")
    
    # Compare feature distributions
    print(f"\n📊 Feature Distribution Comparison:")
    print(f"{'Feature':<25} {'Original Mean':<15} {'Synthetic Mean':<15} {'Difference':<12}")
    print("-" * 70)
    
    for feature in features:
        orig_mean = X_original[feature].mean()
        synth_mean = X_synthetic[feature].mean() if len(X_synthetic) > 0 else np.nan
        diff = abs(orig_mean - synth_mean) if not pd.isna(synth_mean) else np.nan
        
        print(f"{feature:<25} {orig_mean:<15.4f} {synth_mean:<15.4f} {diff:<12.4f}")
    
    return {
        'original_count': len(X_original),
        'synthetic_count': len(X_synthetic),
        'feature_stats': {
            feature: {
                'original_mean': X_original[feature].mean(),
                'synthetic_mean': X_synthetic[feature].mean() if len(X_synthetic) > 0 else None
            } for feature in features
        }
    }

def main():
    """Main cascade interaction analysis."""
    print("🔬 v25 Cascade Interaction Analyzer")
    print("=" * 60)
    print("Hypothesis: Feature interactions or SMOTE causing cascade overfitting")
    print("Individual features: ALL show negative overfitting gaps")
    print("Cascade model: 78.71% positive overfitting gap")
    print()
    
    # Load data
    train_df, test_df, features = load_cascade_data()
    
    # Test 1: Progressive feature combinations
    combination_results = test_progressive_feature_combinations(train_df, test_df, features)
    
    # Test 2: SMOTE impact
    smote_results = test_smote_impact(train_df, test_df, features)
    
    # Test 3: Synthetic data quality
    synthetic_analysis = analyze_synthetic_data_quality(train_df, features)
    
    # Summary Analysis
    print("\n📋 INTERACTION ANALYSIS SUMMARY")
    print("=" * 50)
    
    print("\n🔢 Feature Combination Results:")
    for result in combination_results:
        gap_status = "🔴 CRITICAL" if result['overfitting_gap'] > 0.3 else "🟡 MEDIUM" if result['overfitting_gap'] > 0.1 else "🟢 LOW"
        print(f"{result['combination']:<12} ({result['n_features']} feat): "
              f"Gap {result['overfitting_gap']:>7.3f} | {gap_status}")
    
    print("\n⚖️ SMOTE Impact Results:")
    for result in smote_results:
        gap_status = "🔴 CRITICAL" if result['overfitting_gap'] > 0.3 else "🟡 MEDIUM" if result['overfitting_gap'] > 0.1 else "🟢 LOW"
        print(f"{result['variant']:<12}: Gap {result['overfitting_gap']:>7.3f} | {gap_status}")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime('%Y_%m_%d_%H%M%S')
    results_dict = {
        'analysis_timestamp': timestamp,
        'combination_results': combination_results,
        'smote_results': smote_results,
        'synthetic_analysis': synthetic_analysis
    }
    
    import json
    output_file = f'evaluation/reports/cascade_interaction_analysis_{timestamp}.json'
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results_dict, f, indent=2, default=convert_numpy)
    
    print(f"\n💾 Analysis results saved to: {output_file}")
    print(f"\n✅ Cascade interaction analysis complete!")

if __name__ == "__main__":
    main()