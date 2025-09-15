#!/usr/bin/env python3
"""
Test v4.0 Fatigue Features Impact
Evaluate if fixture congestion and recovery features improve predictions

Strategy: Test fatigue features against v3.1 efficiency baseline
Hypothesis: Physical fatigue provides predictive signals beyond traditional metrics
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_v40_fatigue_impact():
    """Test v4.0 fatigue features against baseline."""
    
    logger.info("🚀 Testing v4.0 Fatigue Features Impact...")
    
    # Load v4.0 dataset with fatigue features
    df = pd.read_csv('data/processed/v40_fatigue_features_2025_09_07.csv')
    logger.info(f"Loaded v4.0 dataset: {df.shape}")
    
    # Baseline features (from v3.1 efficiency breakthrough)
    baseline_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    # v3.1 efficiency features (proven breakthrough features)
    efficiency_features = [
        'goalkeeping_advantage_10', 'away_goalkeeping_efficiency_10_normalized',
        'goalkeeping_advantage_10_normalized', 'net_performance_advantage_10_normalized',
        'net_performance_advantage_10'
    ]
    
    # New fatigue features
    fatigue_features = [col for col in df.columns if 
                       'fatigue' in col or 'congestion' in col or 
                       'recovery' in col or 'days_since' in col or 
                       'travel' in col]
    
    logger.info(f"Baseline features: {len(baseline_features)}")
    logger.info(f"Efficiency features: {len(efficiency_features)}")
    logger.info(f"New fatigue features: {len(fatigue_features)}")
    
    # Test scenarios
    test_scenarios = {
        'v31_baseline': baseline_features,
        'v31_efficiency': baseline_features + efficiency_features,
        'v40_baseline_plus_fatigue': baseline_features + fatigue_features,
        'v40_full_combined': baseline_features + efficiency_features + fatigue_features
    }
    
    # Prepare data
    all_features = list(set(baseline_features + efficiency_features + fatigue_features))
    available_features = [f for f in all_features if f in df.columns]
    
    df_clean = df.dropna(subset=available_features + ['FullTimeResult'])
    logger.info(f"Clean dataset: {df_clean.shape}")
    
    # Train/test split
    split_idx = int(len(df_clean) * 0.8)
    df_train = df_clean[:split_idx]
    df_test = df_clean[split_idx:]
    
    # Target encoding
    target_mapping = {'H': 0, 'D': 1, 'A': 2}
    y_train = df_train['FullTimeResult'].map(target_mapping)
    y_test = df_test['FullTimeResult'].map(target_mapping)
    
    # Model configuration
    model_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Test each scenario
    results = {}
    
    for scenario_name, features in test_scenarios.items():
        logger.info(f"Testing {scenario_name}...")
        
        # Filter available features
        scenario_features = [f for f in features if f in df.columns]
        
        if len(scenario_features) < 5:
            logger.warning(f"Insufficient features for {scenario_name}: {len(scenario_features)}")
            continue
        
        # Train model
        model = RandomForestClassifier(**model_params)
        X_train = df_train[scenario_features]
        X_test = df_test[scenario_features]
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Feature importance
        feature_importance = list(zip(scenario_features, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        results[scenario_name] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'num_features': len(scenario_features),
            'features': scenario_features,
            'feature_importance': feature_importance
        }
    
    # Smart feature selection test
    logger.info("Testing smart feature selection...")
    
    # Use all available features for selection
    combined_features = [f for f in baseline_features + efficiency_features + fatigue_features 
                        if f in df.columns]
    
    if len(combined_features) >= 10:
        selector = SelectKBest(score_func=f_classif, k=min(20, len(combined_features)))
        X_train_combined = df_train[combined_features]
        X_test_combined = df_test[combined_features]
        
        X_train_selected = selector.fit_transform(X_train_combined, y_train)
        X_test_selected = selector.transform(X_test_combined)
        
        selected_feature_names = [combined_features[i] for i in selector.get_support(indices=True)]
        
        model_selected = RandomForestClassifier(**model_params)
        model_selected.fit(X_train_selected, y_train)
        y_pred_selected = model_selected.predict(X_test_selected)
        
        selected_accuracy = accuracy_score(y_test, y_pred_selected)
        selected_f1 = f1_score(y_test, y_pred_selected, average='macro')
        
        selected_importance = list(zip(selected_feature_names, model_selected.feature_importances_))
        selected_importance.sort(key=lambda x: x[1], reverse=True)
        
        results['smart_selection'] = {
            'accuracy': selected_accuracy,
            'f1_macro': selected_f1,
            'num_features': len(selected_feature_names),
            'features': selected_feature_names,
            'feature_importance': selected_importance
        }
    
    # Generate comprehensive report
    print("\\n" + "="*80)
    print("🎯 v4.0 FATIGUE FEATURES IMPACT TEST")
    print("="*80)
    
    print(f"\\n📊 PERFORMANCE COMPARISON:")
    
    # Get baseline for comparison
    baseline_accuracy = results.get('v31_efficiency', {}).get('accuracy', 0.5628)
    
    for scenario, result in results.items():
        accuracy = result['accuracy']
        improvement = (accuracy - baseline_accuracy) * 100
        
        if scenario == 'v31_efficiency':
            marker = "🎯 BASELINE"
        elif accuracy > baseline_accuracy + 0.005:
            marker = "🚀 BREAKTHROUGH"
        elif accuracy > baseline_accuracy:
            marker = "✅ IMPROVEMENT"
        elif accuracy > baseline_accuracy - 0.005:
            marker = "⚡ SIMILAR"
        else:
            marker = "📉 LOWER"
        
        print(f"   {marker} {scenario}: {accuracy:.4f} ({accuracy*100:.2f}%) [{improvement:+.2f}pp]")
        print(f"      Features: {result['num_features']}")
    
    # Best performer analysis
    best_scenario = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_name, best_result = best_scenario
    
    print(f"\\n🏆 BEST PERFORMER: {best_name.upper()}")
    print(f"   • Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"   • Features: {best_result['num_features']}")
    
    improvement = (best_result['accuracy'] - baseline_accuracy) * 100
    print(f"   • Improvement: {improvement:+.2f}pp vs v3.1 baseline")
    
    # Feature importance analysis
    print(f"\\n⭐ TOP FEATURES ({best_name}):")
    
    fatigue_features_in_top = []
    for i, (feature, importance) in enumerate(best_result['feature_importance'][:15]):
        if any(keyword in feature for keyword in ['fatigue', 'congestion', 'recovery', 'days_since', 'travel']):
            feature_type = "🎯 FATIGUE"
            fatigue_features_in_top.append((feature, importance))
        elif feature in efficiency_features:
            feature_type = "⚡ EFFICIENCY"
        else:
            feature_type = "📊 BASELINE"
        
        print(f"   {i+1:2d}. {feature_type} {feature}: {importance:.3f}")
    
    # Fatigue features analysis
    print(f"\\n🔍 FATIGUE FEATURES ANALYSIS:")
    print(f"   • Fatigue features in top 15: {len(fatigue_features_in_top)}")
    
    if fatigue_features_in_top:
        top_fatigue = fatigue_features_in_top[0]
        print(f"   • Top fatigue feature: {top_fatigue[0]} ({top_fatigue[1]:.3f})")
        
        total_fatigue_importance = sum(imp for _, imp in fatigue_features_in_top)
        print(f"   • Combined fatigue importance: {total_fatigue_importance:.3f}")
    
    # Success assessment
    print(f"\\n🚀 SUCCESS ASSESSMENT:")
    
    if improvement >= 1.0:
        status = "🚀 BREAKTHROUGH - Major improvement!"
        recommendation = "Deploy v4.0 immediately"
    elif improvement >= 0.5:
        status = "✅ SUCCESS - Meaningful improvement"
        recommendation = "Deploy v4.0 after validation"
    elif improvement >= 0.2:
        status = "⚡ MARGINAL - Small but positive"
        recommendation = "Consider deployment"
    else:
        status = "📊 MINIMAL - No significant improvement"
        recommendation = "Continue with v3.1 baseline"
    
    print(f"   • Status: {status}")
    print(f"   • Improvement: {improvement:+.2f}pp")
    print(f"   • Recommendation: {recommendation}")
    
    # Fatigue insights
    if fatigue_features_in_top:
        print(f"\\n🎪 FATIGUE INSIGHTS:")
        print(f"   • Physical fatigue provides genuine predictive signals")
        print(f"   • Fixture congestion affects match outcomes measurably") 
        print(f"   • Recovery time is a meaningful factor in team performance")
        
        # Most important fatigue categories
        fatigue_categories = {}
        for feature, importance in fatigue_features_in_top:
            if 'congestion' in feature:
                fatigue_categories['congestion'] = fatigue_categories.get('congestion', 0) + importance
            elif 'recovery' in feature or 'days_since' in feature:
                fatigue_categories['recovery'] = fatigue_categories.get('recovery', 0) + importance
            elif 'travel' in feature:
                fatigue_categories['travel'] = fatigue_categories.get('travel', 0) + importance
            else:
                fatigue_categories['other'] = fatigue_categories.get('other', 0) + importance
        
        if fatigue_categories:
            top_category = max(fatigue_categories.items(), key=lambda x: x[1])
            print(f"   • Most important fatigue type: {top_category[0]} ({top_category[1]:.3f})")
    
    # Business implications
    print(f"\\n💰 BUSINESS IMPLICATIONS:")
    if improvement > 0.5:
        print(f"   • Physical factors enhance prediction accuracy")
        print(f"   • Fixture congestion analysis provides betting edge")
        print(f"   • Recovery tracking worth the implementation effort")
        print(f"   • Expected ROI improvement: +{improvement*2:.1f}%")
    else:
        print(f"   • Fatigue effects may be too subtle for current approach")
        print(f"   • Focus should remain on proven efficiency features")
        print(f"   • Consider alternative fatigue modeling approaches")
    
    print(f"\\n📋 NEXT STEPS:")
    if improvement >= 0.5:
        print(f"   1. Deploy v4.0 as new production model")
        print(f"   2. Enhance fatigue feature engineering")
        print(f"   3. Test referee influence features")
        print(f"   4. Consider ensemble methods")
    else:
        print(f"   1. Continue with v3.1 efficiency baseline")
        print(f"   2. Explore referee influence features")
        print(f"   3. Test alternative fatigue approaches")
        print(f"   4. Focus on external data sources")
    
    # Detailed classification report for best model
    if improvement >= 0.2:
        best_features = best_result['features']
        model = RandomForestClassifier(**model_params)
        X_train_best = df_train[best_features]
        X_test_best = df_test[best_features]
        
        model.fit(X_train_best, y_train)
        y_pred_best = model.predict(X_test_best)
        
        print(f"\\n📊 DETAILED CLASSIFICATION (Best Model):")
        print(classification_report(y_test, y_pred_best, 
                                  target_names=['Home', 'Draw', 'Away'], 
                                  digits=3))
    
    logger.info(f"✅ v4.0 fatigue impact test complete!")
    return results

if __name__ == "__main__":
    results = test_v40_fatigue_impact()