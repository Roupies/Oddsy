#!/usr/bin/env python3
"""
Test v2.5 Meta-Features Performance
Compare v2.4 baseline vs v2.5 with meta-features using cascade model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def test_feature_set(train_df, test_df, features, set_name):
    """Test a specific feature set with cascade model."""
    print(f"\n🧪 TESTING {set_name}")
    print("=" * 50)
    print(f"Features ({len(features)}): {', '.join(features)}")
    
    # Prepare data
    X_train = train_df[features].fillna(train_df[features].median())
    y_train_stage1 = train_df['stage1_target']
    X_test = test_df[features].fillna(train_df[features].median())
    y_test_true = test_df['FullTimeResult']
    
    # Stage 1: Draw vs Non-Draw with SMOTE
    smote = SMOTE(random_state=42)
    X_train_s1_balanced, y_train_s1_balanced = smote.fit_resample(X_train, y_train_stage1)
    
    stage1_model = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=5,
        min_samples_split=10, random_state=42, n_jobs=-1
    )
    
    # Cross-validation on Stage 1
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(stage1_model, X_train_s1_balanced, y_train_s1_balanced,
                               cv=tscv, scoring='f1', n_jobs=-1)
    stage1_cv_f1 = cv_scores.mean()
    
    # Train Stage 1
    stage1_model.fit(X_train_s1_balanced, y_train_s1_balanced)
    
    # Stage 2: Home vs Away
    train_non_draw = train_df[train_df['stage1_target'] == 0].copy()
    X_train_s2 = train_non_draw[features].fillna(train_non_draw[features].median())
    y_train_s2 = train_non_draw['stage2_target']
    
    stage2_model = RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_leaf=3,
        min_samples_split=8, random_state=42, n_jobs=-1
    )
    stage2_model.fit(X_train_s2, y_train_s2)
    
    # Test cascade with optimal threshold (0.7 from previous analysis)
    optimal_threshold = 0.7
    stage1_proba = stage1_model.predict_proba(X_test)[:, 1]
    draw_mask = stage1_proba >= optimal_threshold
    
    y_pred = np.full(len(X_test), 'D', dtype=object)
    if (~draw_mask).sum() > 0:
        non_draw_features = X_test[~draw_mask]
        stage2_pred = stage2_model.predict(non_draw_features)
        y_pred[~draw_mask] = np.where(stage2_pred == 0, 'H', 'A')
    
    # Calculate performance metrics
    overall_accuracy = accuracy_score(y_test_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_true, y_pred, labels=['H', 'D', 'A'], average=None, zero_division=0
    )
    f1_macro = np.mean(f1)
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': stage1_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"📊 Performance Results:")
    print(f"  Global Accuracy: {overall_accuracy:.1%}")
    print(f"  F1-Macro: {f1_macro:.3f}")
    print(f"  Draw Predictions: {draw_mask.sum()}/{len(draw_mask)} ({draw_mask.mean():.1%})")
    print(f"  Class F1 scores: Home={f1[0]:.3f}, Draw={f1[1]:.3f}, Away={f1[2]:.3f}")
    print(f"  Stage 1 CV F1: {stage1_cv_f1:.3f}")
    
    print(f"\n🔍 Top 5 Feature Importances:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return {
        'set_name': set_name,
        'overall_accuracy': overall_accuracy,
        'f1_macro': f1_macro,
        'draw_prediction_rate': draw_mask.mean(),
        'draw_f1': f1[1],
        'stage1_cv_f1': stage1_cv_f1,
        'feature_importance': feature_importance
    }

def compare_feature_sets():
    """Compare v2.4 baseline vs v2.5 meta-features."""
    print("🔬 v2.5 Meta-Features Performance Test")
    print("=" * 60)
    print("Comparison: v2.4 baseline vs v2.5 with context intelligence")
    print("Target: 53.8% → 56.5% accuracy improvement")
    print()
    
    # Load v2.5 dataset
    print("📊 Loading v2.5 dataset with meta-features...")
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v25_meta_features_2025_09_06_004117.csv'
    df = pd.read_csv(data_path)
    
    # Data preparation
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['stage1_target'] = (df['FullTimeResult'] == 'D').astype(int)
    df['stage2_target'] = df['FullTimeResult'].map({'H': 0, 'A': 1})
    
    # Temporal split
    cutoff_date = pd.to_datetime('2023-05-01')
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    print(f"Train: {len(train_df)} matches | Test: {len(test_df)} matches")
    print(f"True draw rate: {test_df['stage1_target'].mean():.1%}")
    
    # Define feature sets
    v24_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    v25_meta_features = ['match_stakes_normalized', 'expected_surprise_factor']
    v25_features = v24_features + v25_meta_features
    
    # Test 1: v2.4 Baseline
    result_v24 = test_feature_set(train_df, test_df, v24_features, "v2.4 BASELINE")
    
    # Test 2: v2.5 with Meta-Features
    result_v25 = test_feature_set(train_df, test_df, v25_features, "v2.5 WITH META-FEATURES")
    
    # Test 3: Meta-Features Only (to understand their standalone value)
    result_meta_only = test_feature_set(train_df, test_df, v25_meta_features, "META-FEATURES ONLY")
    
    # Comparison Analysis
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    accuracy_improvement = (result_v25['overall_accuracy'] - result_v24['overall_accuracy']) * 100
    f1_improvement = result_v25['f1_macro'] - result_v24['f1_macro']
    
    print(f"📈 v2.4 → v2.5 Improvement:")
    print(f"  Global Accuracy: {result_v24['overall_accuracy']:.1%} → {result_v25['overall_accuracy']:.1%} ({accuracy_improvement:+.1f}pp)")
    print(f"  F1-Macro: {result_v24['f1_macro']:.3f} → {result_v25['f1_macro']:.3f} ({f1_improvement:+.3f})")
    print(f"  Draw F1: {result_v24['draw_f1']:.3f} → {result_v25['draw_f1']:.3f} ({result_v25['draw_f1'] - result_v24['draw_f1']:+.3f})")
    
    # Target achievement analysis
    print(f"\n🎯 Target Achievement Analysis:")
    print(f"  Current v2.4: {result_v24['overall_accuracy']:.1%}")
    print(f"  v2.5 Achieved: {result_v25['overall_accuracy']:.1%}")
    print(f"  v2.5 Target: 56.5%")
    
    target_gap = 0.565 - result_v25['overall_accuracy']
    if target_gap <= 0:
        print(f"  ✅ TARGET EXCEEDED by {-target_gap*100:.1f}pp!")
    else:
        print(f"  ⚠️ Target missed by {target_gap*100:.1f}pp")
    
    # Feature importance comparison
    print(f"\n🔍 NEW META-FEATURES IMPORTANCE:")
    v25_importance = result_v25['feature_importance']
    meta_importances = v25_importance[v25_importance['feature'].isin(v25_meta_features)]
    
    for _, row in meta_importances.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Success assessment
    print(f"\n🏆 SPRINT v2.5 ASSESSMENT:")
    if accuracy_improvement >= 2:
        status = "🟢 SUCCESS"
        assessment = "Target improvement achieved"
    elif accuracy_improvement >= 1:
        status = "🟡 PARTIAL SUCCESS"
        assessment = "Meaningful improvement, below target"
    elif accuracy_improvement > 0:
        status = "🟠 MINOR IMPROVEMENT"
        assessment = "Small improvement, investigate further"
    else:
        status = "🔴 NO IMPROVEMENT"
        assessment = "Meta-features not adding value"
    
    print(f"  Status: {status}")
    print(f"  Assessment: {assessment}")
    print(f"  Improvement: {accuracy_improvement:+.1f}pp")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime('%Y_%m_%d_%H%M%S')
    results_dict = {
        'test_timestamp': timestamp,
        'v24_accuracy': float(result_v24['overall_accuracy']),
        'v25_accuracy': float(result_v25['overall_accuracy']),
        'accuracy_improvement_pp': float(accuracy_improvement),
        'target_achieved': target_gap <= 0,
        'meta_feature_importance': {
            row['feature']: float(row['importance']) 
            for _, row in meta_importances.iterrows()
        }
    }
    
    import json
    output_file = f'evaluation/reports/v25_meta_features_test_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results_dict

if __name__ == "__main__":
    compare_feature_sets()