#!/usr/bin/env python3
"""
Test Refined v2.5 - Only Expected Surprise Factor
Remove correlated match_stakes_normalized, keep only expected_surprise_factor
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def quick_cascade_test(features, set_name):
    """Quick cascade test for feature comparison."""
    print(f"\n🧪 {set_name}")
    print("-" * 40)
    
    # Load data
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v25_meta_features_2025_09_06_004117.csv'
    df = pd.read_csv(data_path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['stage1_target'] = (df['FullTimeResult'] == 'D').astype(int)
    df['stage2_target'] = df['FullTimeResult'].map({'H': 0, 'A': 1})
    
    # Temporal split
    cutoff_date = pd.to_datetime('2023-05-01')
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    # Stage 1: Draw Detection
    X_train_s1 = train_df[features].fillna(train_df[features].median())
    y_train_s1 = train_df['stage1_target']
    
    smote = SMOTE(random_state=42)
    X_train_s1_balanced, y_train_s1_balanced = smote.fit_resample(X_train_s1, y_train_s1)
    
    stage1_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    stage1_model.fit(X_train_s1_balanced, y_train_s1_balanced)
    
    # Stage 2: Home vs Away
    train_non_draw = train_df[train_df['stage1_target'] == 0].copy()
    X_train_s2 = train_non_draw[features].fillna(train_non_draw[features].median())
    y_train_s2 = train_non_draw['stage2_target']
    
    stage2_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    stage2_model.fit(X_train_s2, y_train_s2)
    
    # Test with optimal threshold
    X_test = test_df[features].fillna(train_df[features].median())
    y_test_true = test_df['FullTimeResult']
    
    stage1_proba = stage1_model.predict_proba(X_test)[:, 1]
    draw_mask = stage1_proba >= 0.7  # Optimal threshold
    
    y_pred = np.full(len(X_test), 'D', dtype=object)
    if (~draw_mask).sum() > 0:
        non_draw_features = X_test[~draw_mask]
        stage2_pred = stage2_model.predict(non_draw_features)
        y_pred[~draw_mask] = np.where(stage2_pred == 0, 'H', 'A')
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_true, y_pred, labels=['H', 'D', 'A'], average=None, zero_division=0
    )
    f1_macro = np.mean(f1)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': stage1_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Features ({len(features)}): {len([f for f in features if 'v25' in f or f in ['expected_surprise_factor', 'match_stakes_normalized']])} new")
    print(f"Accuracy: {accuracy:.1%} | F1-Macro: {f1_macro:.3f}")
    print(f"Draw predictions: {draw_mask.sum()}/{len(draw_mask)} ({draw_mask.mean():.1%})")
    
    # Show new feature importance if any
    new_features = [f for f in features if f in ['expected_surprise_factor', 'match_stakes_normalized']]
    if new_features:
        print("New feature importance:")
        for feature in new_features:
            if feature in importance_df['feature'].values:
                imp = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
                print(f"  {feature}: {imp:.3f}")
    
    return accuracy, f1_macro, importance_df

def main():
    """Test refined v2.5 approaches."""
    print("🔧 REFINED v2.5 TESTING")
    print("=" * 50)
    print("Hypothesis: Remove correlated match_stakes, keep expected_surprise_factor")
    
    # Define feature sets
    v24_baseline = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    # Test scenarios
    scenarios = [
        ("v2.4 BASELINE", v24_baseline),
        ("v2.5 REFINED (+ expected_surprise_factor)", v24_baseline + ['expected_surprise_factor']),
        ("v2.5 STAKES ONLY (+ match_stakes_normalized)", v24_baseline + ['match_stakes_normalized']),
        ("v2.5 BOTH META", v24_baseline + ['expected_surprise_factor', 'match_stakes_normalized'])
    ]
    
    results = []
    
    for name, features in scenarios:
        accuracy, f1_macro, importance = quick_cascade_test(features, name)
        results.append({
            'name': name,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'features': len(features)
        })
    
    # Analysis
    print(f"\n📊 REFINED v2.5 RESULTS")
    print("=" * 50)
    
    baseline_accuracy = results[0]['accuracy']
    
    for result in results:
        improvement = (result['accuracy'] - baseline_accuracy) * 100
        status = "🟢" if improvement > 1 else "🟡" if improvement > 0 else "🔴"
        print(f"{status} {result['name']:<35} {result['accuracy']:.1%} ({improvement:+.1f}pp)")
    
    # Best performer analysis
    best_result = max(results, key=lambda x: x['accuracy'])
    if best_result['name'] != "v2.4 BASELINE":
        print(f"\n🏆 BEST PERFORMER: {best_result['name']}")
        print(f"Improvement: +{(best_result['accuracy'] - baseline_accuracy)*100:.1f}pp")
        
        if best_result['accuracy'] >= 0.565:  # v2.5 target
            print("✅ v2.5 TARGET ACHIEVED!")
        else:
            gap = 0.565 - best_result['accuracy']
            print(f"⚠️ Target gap: {gap*100:.1f}pp remaining")
    else:
        print(f"\n🔴 NO IMPROVEMENT FOUND")
        print("Meta-features are not adding value to the model")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATION:")
    if best_result['name'] == "v2.5 REFINED (+ expected_surprise_factor)":
        print("✅ Add expected_surprise_factor to v2.4 baseline")
        print("❌ Drop match_stakes_normalized (too correlated)")
    elif best_result['name'] == "v2.4 BASELINE":
        print("❌ Drop meta-features approach - no value added")
        print("🎯 Consider Sprint v2.6 (momentum features) instead")
    else:
        print(f"✅ Adopt {best_result['name']} approach")

if __name__ == "__main__":
    main()