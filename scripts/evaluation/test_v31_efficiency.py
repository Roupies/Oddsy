#!/usr/bin/env python3
"""
Test v3.1 Efficiency Features against baseline
Evaluate if xG over/under-performance patterns break through 55.51% ceiling
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_v31_efficiency_features():
    """Test v3.1 efficiency features against baseline performance."""
    
    logger.info("🚀 Testing v3.1 Efficiency Features...")
    
    # Load v3.1 dataset with efficiency features
    df = pd.read_csv('data/processed/v31_efficiency_features_2025_09_06.csv')
    logger.info(f"Loaded v3.1 dataset: {df.shape}")
    
    # Baseline features (from v2.4 production)
    baseline_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    # New efficiency features
    efficiency_features = [col for col in df.columns if 
                          'finishing_efficiency' in col or 
                          'goalkeeping_efficiency' in col or
                          'net_performance_factor' in col or
                          'advantage' in col or
                          ('finishing' in col and ('hot' in col or 'cold' in col))]
    
    logger.info(f"Baseline features: {len(baseline_features)}")
    logger.info(f"New efficiency features: {len(efficiency_features)}")
    
    # Combined feature set
    combined_features = baseline_features + efficiency_features
    available_combined = [f for f in combined_features if f in df.columns]
    available_baseline = [f for f in baseline_features if f in df.columns]
    
    logger.info(f"Available baseline: {len(available_baseline)}")
    logger.info(f"Available combined: {len(available_combined)}")
    
    # Clean data
    df_clean = df.dropna(subset=available_combined + ['FullTimeResult'])
    logger.info(f"Clean dataset: {df_clean.shape}")
    
    # Train/test split (temporal - last 20% for test)
    split_idx = int(len(df_clean) * 0.8)
    df_train = df_clean[:split_idx]
    df_test = df_clean[split_idx:]
    
    logger.info(f"Train: {len(df_train)}, Test: {len(df_test)}")
    
    # Target encoding
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train = df_train['FullTimeResult'].map(target_mapping)
    y_test = df_test['FullTimeResult'].map(target_mapping)
    
    # Model configuration (same as baseline)
    model_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    results = {}
    
    # Test 1: Baseline performance
    logger.info("Testing baseline features...")
    model_baseline = RandomForestClassifier(**model_params)
    X_train_baseline = df_train[available_baseline]
    X_test_baseline = df_test[available_baseline]
    
    model_baseline.fit(X_train_baseline, y_train)
    y_pred_baseline = model_baseline.predict(X_test_baseline)
    
    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')
    
    results['baseline'] = {
        'accuracy': baseline_accuracy,
        'f1_macro': baseline_f1,
        'features': len(available_baseline)
    }
    
    # Test 2: Combined features (baseline + efficiency)
    logger.info("Testing combined features (baseline + efficiency)...")
    model_combined = RandomForestClassifier(**model_params)
    X_train_combined = df_train[available_combined]
    X_test_combined = df_test[available_combined]
    
    model_combined.fit(X_train_combined, y_train)
    y_pred_combined = model_combined.predict(X_test_combined)
    
    combined_accuracy = accuracy_score(y_test, y_pred_combined)
    combined_f1 = f1_score(y_test, y_pred_combined, average='macro')
    
    results['combined'] = {
        'accuracy': combined_accuracy,
        'f1_macro': combined_f1,
        'features': len(available_combined)
    }
    
    # Test 3: Efficiency features only
    logger.info("Testing efficiency features only...")
    efficiency_only = [f for f in efficiency_features if f in df.columns and 
                      df[f].notna().sum() > len(df) * 0.8]  # At least 80% coverage
    
    if len(efficiency_only) >= 5:  # Need minimum features
        model_efficiency = RandomForestClassifier(**model_params)
        X_train_efficiency = df_train[efficiency_only]
        X_test_efficiency = df_test[efficiency_only]
        
        model_efficiency.fit(X_train_efficiency, y_train)
        y_pred_efficiency = model_efficiency.predict(X_test_efficiency)
        
        efficiency_accuracy = accuracy_score(y_test, y_pred_efficiency)
        efficiency_f1 = f1_score(y_test, y_pred_efficiency, average='macro')
        
        results['efficiency_only'] = {
            'accuracy': efficiency_accuracy,
            'f1_macro': efficiency_f1,
            'features': len(efficiency_only)
        }
    else:
        logger.warning("Insufficient efficiency features for standalone test")
        results['efficiency_only'] = None
    
    # Test 4: Smart feature selection on combined set
    logger.info("Testing smart feature selection...")
    selector = SelectKBest(score_func=f_classif, k=min(15, len(available_combined)))
    X_train_selected = selector.fit_transform(X_train_combined, y_train)
    X_test_selected = selector.transform(X_test_combined)
    
    selected_feature_names = [available_combined[i] for i in selector.get_support(indices=True)]
    
    model_selected = RandomForestClassifier(**model_params)
    model_selected.fit(X_train_selected, y_train)
    y_pred_selected = model_selected.predict(X_test_selected)
    
    selected_accuracy = accuracy_score(y_test, y_pred_selected)
    selected_f1 = f1_score(y_test, y_pred_selected, average='macro')
    
    results['selected'] = {
        'accuracy': selected_accuracy,
        'f1_macro': selected_f1,
        'features': len(selected_feature_names),
        'feature_names': selected_feature_names
    }
    
    # Generate comprehensive report
    print("\\n" + "="*80)
    print("🎯 v3.1 EFFICIENCY FEATURES EVALUATION")
    print("="*80)
    
    print(f"\\n📊 PERFORMANCE COMPARISON:")
    print(f"   🔄 Baseline (10 features):     {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"   ⚡ Combined ({len(available_combined)} features):    {combined_accuracy:.4f} ({combined_accuracy*100:.2f}%)")
    
    if results['efficiency_only']:
        efficiency_acc = results['efficiency_only']['accuracy']
        print(f"   🎯 Efficiency Only ({results['efficiency_only']['features']} features): {efficiency_acc:.4f} ({efficiency_acc*100:.2f}%)")
    
    print(f"   🧠 Smart Selected ({results['selected']['features']} features): {selected_accuracy:.4f} ({selected_accuracy*100:.2f}%)")
    
    # Calculate improvements
    combined_improvement = (combined_accuracy - baseline_accuracy) * 100
    selected_improvement = (selected_accuracy - baseline_accuracy) * 100
    
    print(f"\\n📈 IMPROVEMENT ANALYSIS:")
    print(f"   • Combined vs Baseline: {combined_improvement:+.2f}pp")
    print(f"   • Selected vs Baseline: {selected_improvement:+.2f}pp")
    print(f"   • Test set size: {len(y_test)} matches")
    
    # Breakthrough assessment
    breakthrough_threshold = 0.5651  # 56.51% - our current best
    print(f"\\n🚀 BREAKTHROUGH ASSESSMENT:")
    print(f"   • Current ceiling: 55.51%")
    print(f"   • Combined result: {combined_accuracy*100:.2f}%")
    print(f"   • Selected result: {selected_accuracy*100:.2f}%")
    
    if combined_accuracy > 0.5551:
        breakthrough_amount = (combined_accuracy - 0.5551) * 100
        print(f"   ✅ BREAKTHROUGH! +{breakthrough_amount:.2f}pp improvement!")
        status = "SUCCESS"
    elif selected_accuracy > 0.5551:
        breakthrough_amount = (selected_accuracy - 0.5551) * 100
        print(f"   ✅ BREAKTHROUGH! (Selected) +{breakthrough_amount:.2f}pp improvement!")
        status = "SUCCESS"
    else:
        print(f"   📊 No breakthrough - efficiency features don't improve ceiling")
        status = "NO_IMPROVEMENT"
    
    # Feature importance analysis
    if status == "SUCCESS":
        print(f"\\n⭐ BEST FEATURE SET (Smart Selected):")
        importance_scores = model_selected.feature_importances_
        feature_importance = list(zip(selected_feature_names, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for feature, importance in feature_importance[:10]:
            feature_type = "🎯 NEW" if feature in efficiency_features else "📊 BASE"
            print(f"   {feature_type} {feature}: {importance:.3f}")
    
    # Business implications
    print(f"\\n💰 BUSINESS IMPLICATIONS:")
    if status == "SUCCESS":
        print(f"   • xG efficiency patterns provide genuine predictive value")
        print(f"   • Over/under-performance detection is football-relevant")
        print(f"   • Ready for Phase 2: Player-level data integration")
    else:
        print(f"   • xG efficiency features redundant with existing metrics")
        print(f"   • Focus should shift to external data sources")
        print(f"   • Consider Phase 2: Player absence/injury features")
    
    print(f"\\n📋 NEXT STEPS:")
    if status == "SUCCESS":
        print(f"   1. Deploy v3.1 as new production model")
        print(f"   2. Begin Phase 2: Player data pipeline design")
        print(f"   3. Test player absence impact on predictions")
    else:
        print(f"   1. Archive v3.1 as research experiment")
        print(f"   2. Focus on external data: player injuries, suspensions") 
        print(f"   3. Consider alternative approaches: ensemble methods")
    
    # Classification report for best model
    if status == "SUCCESS":
        best_pred = y_pred_selected if selected_accuracy > combined_accuracy else y_pred_combined
        print(f"\\n📊 DETAILED CLASSIFICATION (Best Model):")
        print(classification_report(y_test, best_pred, 
                                  target_names=['Home', 'Draw', 'Away'], 
                                  digits=3))
    
    logger.info(f"✅ v3.1 evaluation complete! Status: {status}")
    return results, status

if __name__ == "__main__":
    results, status = test_v31_efficiency_features()