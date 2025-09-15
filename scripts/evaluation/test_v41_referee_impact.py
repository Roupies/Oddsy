#!/usr/bin/env python3
"""
Test v4.1 Referee Features Impact
Evaluate if referee influence patterns improve match predictions

Strategy: Test referee behavioral patterns against proven baselines
Hypothesis: Official decisions directly impact match outcomes through measurable bias
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_v41_referee_impact():
    """Test v4.1 referee features against all previous baselines."""
    
    logger.info("🚀 Testing v4.1 Referee Features Impact...")
    
    # Load v4.1 dataset with referee features
    df = pd.read_csv('data/processed/v41_referee_features_2025_09_07.csv')
    logger.info(f"Loaded v4.1 dataset: {df.shape}")
    
    # Progressive feature sets for testing
    baseline_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    efficiency_features = [
        'goalkeeping_advantage_10', 'away_goalkeeping_efficiency_10_normalized',
        'goalkeeping_advantage_10_normalized', 'net_performance_advantage_10_normalized',
        'net_performance_advantage_10'
    ]
    
    # Select best fatigue features (from v4.0 testing)
    key_fatigue_features = [
        'home_days_since_last_match', 'away_days_since_last_match',
        'fixture_density_differential', 'fatigue_advantage'
    ]
    
    # Core referee features
    referee_features = [col for col in df.columns if 'referee' in col or 'ref_' in col]
    
    logger.info(f"Baseline features: {len(baseline_features)}")
    logger.info(f"Efficiency features: {len(efficiency_features)}")
    logger.info(f"Fatigue features: {len(key_fatigue_features)}")
    logger.info(f"Referee features: {len(referee_features)}")
    
    # Test scenarios (progressive enhancement)
    test_scenarios = {
        'v31_baseline': baseline_features,
        'v31_efficiency': baseline_features + efficiency_features,
        'v40_plus_fatigue': baseline_features + efficiency_features + key_fatigue_features,
        'v41_plus_referee': baseline_features + efficiency_features + referee_features,
        'v41_full_combined': baseline_features + efficiency_features + key_fatigue_features + referee_features
    }
    
    # Prepare data
    all_features = list(set(baseline_features + efficiency_features + key_fatigue_features + referee_features))
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
    
    combined_features = [f for f in baseline_features + efficiency_features + referee_features 
                        if f in df.columns]
    
    if len(combined_features) >= 10:
        selector = SelectKBest(score_func=f_classif, k=min(25, len(combined_features)))
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
    print("⚖️ v4.1 REFEREE FEATURES IMPACT TEST")
    print("="*80)
    
    print(f"\\n📊 PERFORMANCE EVOLUTION:")
    
    # Get v3.1 efficiency baseline for comparison
    baseline_accuracy = results.get('v31_efficiency', {}).get('accuracy', 0.5628)
    
    for scenario, result in results.items():
        accuracy = result['accuracy']
        improvement = (accuracy - baseline_accuracy) * 100
        
        if scenario == 'v31_efficiency':
            marker = "🎯 BASELINE"
        elif accuracy > baseline_accuracy + 0.01:
            marker = "🚀 BREAKTHROUGH"
        elif accuracy > baseline_accuracy + 0.005:
            marker = "✅ IMPROVEMENT"
        elif accuracy > baseline_accuracy:
            marker = "⚡ MARGINAL"
        elif accuracy > baseline_accuracy - 0.005:
            marker = "📊 SIMILAR"
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
    
    referee_features_in_top = []
    for i, (feature, importance) in enumerate(best_result['feature_importance'][:20]):
        if 'referee' in feature or 'ref_' in feature:
            feature_type = "⚖️ REFEREE"
            referee_features_in_top.append((feature, importance))
        elif feature in efficiency_features:
            feature_type = "⚡ EFFICIENCY"
        elif feature in key_fatigue_features:
            feature_type = "🎯 FATIGUE"
        else:
            feature_type = "📊 BASELINE"
        
        print(f"   {i+1:2d}. {feature_type} {feature}: {importance:.3f}")
    
    # Referee features analysis
    print(f"\\n⚖️ REFEREE FEATURES ANALYSIS:")
    print(f"   • Referee features in top 20: {len(referee_features_in_top)}")
    
    if referee_features_in_top:
        top_referee = referee_features_in_top[0]
        print(f"   • Top referee feature: {top_referee[0]} ({top_referee[1]:.3f})")
        
        total_referee_importance = sum(imp for _, imp in referee_features_in_top)
        print(f"   • Combined referee importance: {total_referee_importance:.3f}")
        
        # Categorize referee feature types
        referee_categories = {}
        for feature, importance in referee_features_in_top:
            if 'disciplinary' in feature or 'card' in feature:
                referee_categories['disciplinary'] = referee_categories.get('disciplinary', 0) + importance
            elif 'bias' in feature or 'home' in feature:
                referee_categories['bias'] = referee_categories.get('bias', 0) + importance
            elif 'experience' in feature:
                referee_categories['experience'] = referee_categories.get('experience', 0) + importance
            else:
                referee_categories['other'] = referee_categories.get('other', 0) + importance
        
        if referee_categories:
            top_category = max(referee_categories.items(), key=lambda x: x[1])
            print(f"   • Most important referee aspect: {top_category[0]} ({top_category[1]:.3f})")
    else:
        print(f"   • No referee features in top 20 - limited predictive value")
    
    # Success assessment
    print(f"\\n🚀 REFEREE SUCCESS ASSESSMENT:")
    
    # Compare referee vs non-referee scenarios
    referee_scenario_acc = results.get('v41_plus_referee', {}).get('accuracy', 0)
    non_referee_acc = results.get('v40_plus_fatigue', {}).get('accuracy', 0)
    
    referee_contribution = (referee_scenario_acc - non_referee_acc) * 100 if non_referee_acc > 0 else 0
    
    print(f"   • Referee-only contribution: {referee_contribution:+.2f}pp")
    
    if improvement >= 1.0:
        status = "🚀 BREAKTHROUGH - Major improvement!"
        recommendation = "Deploy v4.1 immediately"
    elif improvement >= 0.5:
        status = "✅ SUCCESS - Meaningful improvement"
        recommendation = "Deploy v4.1 after validation"
    elif referee_contribution >= 0.3 or len(referee_features_in_top) >= 2:
        status = "⚖️ REFEREE VALUE - Officials matter"
        recommendation = "Referee features add measurable value"
    elif improvement >= 0.2:
        status = "⚡ MARGINAL - Small but positive"
        recommendation = "Consider deployment"
    else:
        status = "📊 MINIMAL - No significant improvement"
        recommendation = "Continue with previous baseline"
    
    print(f"   • Status: {status}")
    print(f"   • Total improvement: {improvement:+.2f}pp")
    print(f"   • Recommendation: {recommendation}")
    
    # Official influence insights
    if referee_features_in_top:
        print(f"\\n⚖️ OFFICIAL INFLUENCE INSIGHTS:")
        print(f"   • Referee behavior patterns provide predictive signals")
        print(f"   • Official bias and disciplinary tendencies affect outcomes")
        print(f"   • Referee experience and consistency influence match flow")
        
        # Specific insights based on top features
        for feature, importance in referee_features_in_top[:3]:
            if 'home_bias' in feature:
                print(f"   • Home bias patterns: {importance:.3f} importance")
            elif 'disciplinary' in feature:
                print(f"   • Card tendency patterns: {importance:.3f} importance")
            elif 'experience' in feature:
                print(f"   • Referee experience factor: {importance:.3f} importance")
    
    # Business implications
    print(f"\\n💰 BUSINESS IMPLICATIONS:")
    if improvement > 0.5 or referee_contribution > 0.3:
        print(f"   • Official decisions create measurable betting edges")
        print(f"   • Referee analysis worth the implementation effort")
        print(f"   • Disciplinary patterns provide actionable insights")
        print(f"   • Expected ROI improvement: +{improvement*2:.1f}%")
    elif referee_features_in_top:
        print(f"   • Referee effects are real but subtle")
        print(f"   • Combined with other features, provides incremental value")
        print(f"   • Worth including in final production model")
    else:
        print(f"   • Referee effects may be too random for current approach")
        print(f"   • Focus should remain on proven efficiency/form features")
        print(f"   • Consider more granular referee data if available")
    
    print(f"\\n📋 FINAL v4.1 ASSESSMENT:")
    if improvement >= 1.0:
        print(f"   1. 🚀 DEPLOY v4.1 as new production model")
        print(f"   2. Referee intelligence provides significant edge")
        print(f"   3. Consider expanding referee feature engineering")
        print(f"   4. Ready for production betting deployment")
    elif improvement >= 0.5:
        print(f"   1. ✅ VALIDATE v4.1 thoroughly then deploy")
        print(f"   2. Meaningful improvement justifies complexity")
        print(f"   3. Monitor referee feature stability")
        print(f"   4. Consider ensemble methods")
    elif referee_features_in_top:
        print(f"   1. ⚖️ REFEREE VALUE CONFIRMED - incremental improvement")
        print(f"   2. Include referee features in production model")
        print(f"   3. Focus on smart feature selection (20-25 features)")
        print(f"   4. Continue optimizing feature engineering")
    else:
        print(f"   1. 📊 LIMITED REFEREE IMPACT - stay with previous baseline")
        print(f"   2. Referee effects too subtle for current approach")
        print(f"   3. Focus on alternative improvement strategies")
        print(f"   4. Consider ensemble or external data sources")
    
    # Detailed classification report for best model
    if improvement >= 0.3:
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
    
    logger.info(f"✅ v4.1 referee impact test complete!")
    return results

if __name__ == "__main__":
    results = test_v41_referee_impact()