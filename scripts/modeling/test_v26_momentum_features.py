#!/usr/bin/env python3
"""
Test v2.6 Momentum Features Performance
Compare v2.4 baseline vs v2.6 with momentum features using cascade model.
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
    print("-" * 50)
    
    # Load v2.6 data
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v26_momentum_features_2025_09_06_004609.csv'
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
    
    stage1_model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5,
                                        min_samples_split=10, random_state=42, n_jobs=-1)
    stage1_model.fit(X_train_s1_balanced, y_train_s1_balanced)
    
    # Stage 2: Home vs Away
    train_non_draw = train_df[train_df['stage1_target'] == 0].copy()
    X_train_s2 = train_non_draw[features].fillna(train_non_draw[features].median())
    y_train_s2 = train_non_draw['stage2_target']
    
    stage2_model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=3,
                                        min_samples_split=8, random_state=42, n_jobs=-1)
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
    
    print(f"Features ({len(features)})")
    print(f"Accuracy: {accuracy:.1%} | F1-Macro: {f1_macro:.3f}")
    print(f"Draw predictions: {draw_mask.sum()}/{len(draw_mask)} ({draw_mask.mean():.1%})")
    print(f"Class F1: Home={f1[0]:.3f}, Draw={f1[1]:.3f}, Away={f1[2]:.3f}")
    
    # Show momentum feature importance if any
    momentum_features = [f for f in features if 'acceleration' in f or 'time_decay' in f]
    if momentum_features:
        print("Momentum feature importance:")
        for feature in momentum_features:
            if feature in importance_df['feature'].values:
                imp = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
                print(f"  {feature}: {imp:.3f}")
    
    print("Top 5 overall features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return accuracy, f1_macro, importance_df

def comprehensive_v26_test():
    """Comprehensive test of v2.6 momentum features."""
    print("🚀 v2.6 MOMENTUM FEATURES COMPREHENSIVE TEST")
    print("=" * 60)
    print("Sprint v2.6: Advanced Temporal Dynamics")
    print("Target: 55.4% → 58%+ accuracy through momentum detection")
    print()
    
    # Define feature sets
    v24_baseline = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    v26_momentum_features = ['form_acceleration_normalized', 'time_decay_form_normalized']
    
    # Test scenarios to isolate the impact of each momentum feature
    scenarios = [
        ("v2.4 BASELINE", v24_baseline),
        ("v2.6 + Form Acceleration", v24_baseline + ['form_acceleration_normalized']),
        ("v2.6 + Time Decay Form", v24_baseline + ['time_decay_form_normalized']),
        ("v2.6 FULL MOMENTUM", v24_baseline + v26_momentum_features),
        ("MOMENTUM ONLY", v26_momentum_features)
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
    print(f"\n📊 v2.6 MOMENTUM RESULTS SUMMARY")
    print("=" * 60)
    
    baseline_accuracy = results[0]['accuracy']  # v2.4 baseline
    
    for result in results:
        improvement = (result['accuracy'] - baseline_accuracy) * 100
        
        if improvement >= 2.5:
            status = "🟢 EXCELLENT"
        elif improvement >= 1.5:
            status = "🟡 GOOD"
        elif improvement > 0:
            status = "🟠 MINOR"
        else:
            status = "🔴 WORSE"
        
        print(f"{status} {result['name']:<25} {result['accuracy']:.1%} ({improvement:+.1f}pp) F1:{result['f1_macro']:.3f}")
    
    # Best performer analysis
    best_result = max(results[1:], key=lambda x: x['accuracy'])  # Exclude baseline from best
    best_improvement = (best_result['accuracy'] - baseline_accuracy) * 100
    
    print(f"\n🏆 BEST MOMENTUM APPROACH: {best_result['name']}")
    print(f"Improvement: +{best_improvement:.1f}pp")
    
    # Target achievement
    target_accuracy = 0.58  # 58% target
    if best_result['accuracy'] >= target_accuracy:
        print(f"✅ v2.6 TARGET ACHIEVED! ({best_result['accuracy']:.1%} ≥ 58%)")
        success_status = "SUCCESS"
    elif best_improvement >= 2:
        print(f"🟡 STRONG IMPROVEMENT - Target missed by {(target_accuracy - best_result['accuracy'])*100:.1f}pp")
        success_status = "PARTIAL SUCCESS"
    elif best_improvement > 0:
        print(f"🟠 MINOR IMPROVEMENT - Target missed by {(target_accuracy - best_result['accuracy'])*100:.1f}pp")
        success_status = "MINOR SUCCESS"
    else:
        print(f"🔴 NO IMPROVEMENT - Momentum features don't add value")
        success_status = "FAILURE"
    
    # Feature correlation analysis
    if best_improvement > 0:
        print(f"\n🔍 SUCCESS ANALYSIS:")
        if 'Form Acceleration' in best_result['name']:
            print("✅ Form acceleration (momentum derivative) is the key!")
            print("💡 Short-term vs long-term form comparison adds predictive value")
        elif 'Time Decay' in best_result['name']:
            print("✅ Time decay weighting (EMA) is the key!")
            print("💡 Recent matches weighted more heavily improves predictions")
        elif 'FULL MOMENTUM' in best_result['name']:
            print("✅ Combined momentum features work best!")
            print("💡 Both acceleration and time weighting contribute")
    
    # Recommendations
    print(f"\n💡 SPRINT v2.6 RECOMMENDATIONS:")
    
    if success_status == "SUCCESS":
        print(f"🎯 ADOPT v2.6 MOMENTUM APPROACH")
        print(f"  - Use {best_result['name']} as new baseline")
        print(f"  - Continue to Sprint v2.7 (H2H Intelligence)")
        
    elif success_status in ["PARTIAL SUCCESS", "MINOR SUCCESS"]:
        print(f"🤔 MOMENTUM SHOWS PROMISE")
        print(f"  - Consider refining momentum calculation")
        print(f"  - Test different acceleration windows (2vs8, 4vs12)")
        print(f"  - Or proceed to Sprint v2.7 for cumulative improvements")
        
    else:
        print(f"❌ ABANDON MOMENTUM APPROACH")
        print(f"  - Skip to Sprint v2.7 (H2H Intelligence)")
        print(f"  - Or investigate other temporal patterns")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime('%Y_%m_%d_%H%M%S')
    results_dict = {
        'test_timestamp': timestamp,
        'baseline_accuracy': float(baseline_accuracy),
        'best_accuracy': float(best_result['accuracy']),
        'best_approach': best_result['name'],
        'improvement_pp': float(best_improvement),
        'target_achieved': best_result['accuracy'] >= target_accuracy,
        'success_status': success_status
    }
    
    import json
    output_file = f'evaluation/reports/v26_momentum_test_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results_dict

if __name__ == "__main__":
    comprehensive_v26_test()