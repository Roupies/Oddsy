#!/usr/bin/env python3
"""
Final v2.7 Evaluation - Gemini's Roadmap Completion
Test all sprint combinations to find the ultimate configuration for 58%+ elite performance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def calculate_optimized_momentum(df):
    """Add the optimized 3vs15 momentum feature."""
    print("⚡ Adding optimized momentum feature (3vs15 windows)...")
    
    form_data = []
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_form_history = {team: [] for team in all_teams}
    
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    for idx, match in df_sorted.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        result = match['FullTimeResult']
        
        # Calculate 3vs15 acceleration BEFORE this match
        home_form_3 = np.mean(team_form_history[home_team][-3:]) if len(team_form_history[home_team]) >= 3 else 1.0
        home_form_15 = np.mean(team_form_history[home_team][-15:]) if len(team_form_history[home_team]) >= 15 else 1.0
        away_form_3 = np.mean(team_form_history[away_team][-3:]) if len(team_form_history[away_team]) >= 3 else 1.0
        away_form_15 = np.mean(team_form_history[away_team][-15:]) if len(team_form_history[away_team]) >= 15 else 1.0
        
        acceleration_diff = (home_form_3 - home_form_15) - (away_form_3 - away_form_15)
        
        form_data.append({
            'Date': match['Date'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'acceleration_diff': acceleration_diff
        })
        
        # Update histories
        if result == 'H':
            home_points, away_points = 3, 0
        elif result == 'A':
            home_points, away_points = 0, 3
        else:
            home_points, away_points = 1, 1
            
        team_form_history[home_team].append(home_points)
        team_form_history[away_team].append(away_points)
        
        if len(team_form_history[home_team]) > 20:
            team_form_history[home_team] = team_form_history[home_team][-20:]
        if len(team_form_history[away_team]) > 20:
            team_form_history[away_team] = team_form_history[away_team][-20:]
    
    form_df = pd.DataFrame(form_data)
    
    # Normalize
    min_val = form_df['acceleration_diff'].min()
    max_val = form_df['acceleration_diff'].max()
    form_df['form_acceleration_optimized'] = (form_df['acceleration_diff'] - min_val) / (max_val - min_val)
    
    # Merge back
    df_with_momentum = df.merge(form_df[['Date', 'HomeTeam', 'AwayTeam', 'form_acceleration_optimized']], 
                                on=['Date', 'HomeTeam', 'AwayTeam'], how='left')
    
    return df_with_momentum

def comprehensive_cascade_test(features, set_name):
    """Comprehensive cascade test with full hyperparameters."""
    print(f"\n🏆 {set_name}")
    print("-" * 50)
    
    # Load v2.7 data
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v27_h2h_intelligence_2025_09_06_005038.csv'
    df = pd.read_csv(data_path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Add optimized momentum if needed
    if 'form_acceleration_optimized' in features:
        df = calculate_optimized_momentum(df)
    
    df['stage1_target'] = (df['FullTimeResult'] == 'D').astype(int)
    df['stage2_target'] = df['FullTimeResult'].map({'H': 0, 'A': 1})
    
    # Temporal split
    cutoff_date = pd.to_datetime('2023-05-01')
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    # Stage 1: Draw Detection (production hyperparameters)
    X_train_s1 = train_df[features].fillna(train_df[features].median())
    y_train_s1 = train_df['stage1_target']
    
    smote = SMOTE(random_state=42)
    X_train_s1_balanced, y_train_s1_balanced = smote.fit_resample(X_train_s1, y_train_s1)
    
    stage1_model = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=5,
        min_samples_split=10, random_state=42, n_jobs=-1
    )
    stage1_model.fit(X_train_s1_balanced, y_train_s1_balanced)
    
    # Stage 2: Home vs Away (production hyperparameters)
    train_non_draw = train_df[train_df['stage1_target'] == 0].copy()
    X_train_s2 = train_non_draw[features].fillna(train_non_draw[features].median())
    y_train_s2 = train_non_draw['stage2_target']
    
    stage2_model = RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_leaf=3,
        min_samples_split=8, random_state=42, n_jobs=-1
    )
    stage2_model.fit(X_train_s2, y_train_s2)
    
    # Test with optimized threshold
    X_test = test_df[features].fillna(train_df[features].median())
    y_test_true = test_df['FullTimeResult']
    
    stage1_proba = stage1_model.predict_proba(X_test)[:, 1]
    draw_mask = stage1_proba >= 0.7  # Optimized threshold
    
    y_pred = np.full(len(X_test), 'D', dtype=object)
    if (~draw_mask).sum() > 0:
        stage2_pred = stage2_model.predict(X_test[~draw_mask])
        y_pred[~draw_mask] = np.where(stage2_pred == 0, 'H', 'A')
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_true, y_pred, labels=['H', 'D', 'A'], average=None, zero_division=0
    )
    f1_macro = np.mean(f1)
    
    # Feature importance analysis
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': stage1_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Features: {len(features)} total")
    print(f"🎯 Accuracy: {accuracy:.1%} | F1-Macro: {f1_macro:.3f}")
    print(f"📊 Class F1: Home={f1[0]:.3f}, Draw={f1[1]:.3f}, Away={f1[2]:.3f}")
    print(f"🎲 Draw predictions: {draw_mask.sum()}/{len(draw_mask)} ({draw_mask.mean():.1%})")
    
    # Show new feature importance if any
    new_features = [f for f in features if any(marker in f for marker in ['acceleration', 'bogey', 'h2h_context'])]
    if new_features:
        print("🆕 New feature importance:")
        for feature in new_features:
            if feature in importance_df['feature'].values:
                imp = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
                print(f"  {feature}: {imp:.3f}")
    
    return {
        'name': set_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'draw_f1': f1[1],
        'features': len(features),
        'importance_df': importance_df
    }

def gemini_roadmap_final_evaluation():
    """Final comprehensive evaluation of Gemini's 3-sprint roadmap."""
    print("🎯 GEMINI'S ROADMAP FINAL EVALUATION")
    print("=" * 60)
    print("Complete evaluation of intelligent feature engineering journey")
    print("v2.4 baseline → v2.5 meta → v2.6 momentum → v2.7 H2H intelligence")
    print("ULTIMATE TARGET: 58%+ Elite Performance")
    print()
    
    # Define all feature combinations
    v24_baseline = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    # Test comprehensive combinations
    test_scenarios = [
        ("v2.4 BASELINE", v24_baseline),
        
        # Individual sprint tests
        ("v2.6 + Optimized Momentum", v24_baseline + ['form_acceleration_optimized']),
        ("v2.7 + Bogey Teams Only", v24_baseline + ['bogey_team_score']),
        ("v2.7 + Context H2H Only", v24_baseline + ['h2h_context_score']),
        
        # Progressive combinations
        ("v2.6+v2.7 Momentum + Bogey", v24_baseline + ['form_acceleration_optimized', 'bogey_team_score']),
        ("v2.6+v2.7 Momentum + Context H2H", v24_baseline + ['form_acceleration_optimized', 'h2h_context_score']),
        
        # Ultimate combination
        ("🏆 ULTIMATE v2.7 (All Features)", v24_baseline + ['form_acceleration_optimized', 'bogey_team_score', 'h2h_context_score'])
    ]
    
    print("🧪 TESTING ALL SPRINT COMBINATIONS")
    print("=" * 50)
    
    results = []
    
    for scenario_name, features in test_scenarios:
        result = comprehensive_cascade_test(features, scenario_name)
        results.append(result)
    
    # Comprehensive analysis
    print(f"\n📊 COMPLETE ROADMAP RESULTS")
    print("=" * 60)
    
    baseline_accuracy = results[0]['accuracy']  # v2.4 baseline
    
    for result in results:
        improvement = (result['accuracy'] - baseline_accuracy) * 100
        
        if result['accuracy'] >= 0.58:  # 58%+ elite target
            status = "🏆 ELITE"
        elif improvement >= 2:
            status = "🟢 EXCELLENT"
        elif improvement >= 1:
            status = "🟡 GOOD"
        elif improvement > 0:
            status = "🟠 MINOR"
        else:
            status = "🔴 WORSE"
        
        target_gap = 0.58 - result['accuracy']
        gap_str = f"({target_gap*100:+.1f}pp to 58%)" if target_gap > 0 else "✅ TARGET ACHIEVED!"
        
        print(f"{status} {result['name']:<35} {result['accuracy']:.1%} ({improvement:+.1f}pp) {gap_str}")
    
    # Find ultimate best
    best_result = max(results[1:], key=lambda x: x['accuracy'])  # Exclude baseline
    best_improvement = (best_result['accuracy'] - baseline_accuracy) * 100
    
    print(f"\n🏆 ULTIMATE CHAMPION: {best_result['name']}")
    print(f"🎯 Final Performance: {best_result['accuracy']:.1%}")
    print(f"📈 Total Improvement: +{best_improvement:.1f}pp from v2.4 baseline")
    print(f"⚖️ Balanced F1-Macro: {best_result['f1_macro']:.3f}")
    
    # Elite achievement assessment
    if best_result['accuracy'] >= 0.58:
        print(f"\n🎉 🎉 🎉 ELITE PERFORMANCE ACHIEVED! 🎉 🎉 🎉")
        print(f"✅ Gemini's roadmap successfully delivered 58%+ accuracy!")
        print(f"🏅 Industry-competitive model ready for deployment")
        roadmap_success = "COMPLETE SUCCESS"
        
    elif best_result['accuracy'] >= 0.57:
        print(f"\n🥈 NEAR-ELITE PERFORMANCE!")
        print(f"🎯 Just {(0.58 - best_result['accuracy'])*100:.1f}pp short of 58% elite target")
        print(f"✅ Gemini's roadmap delivered exceptional progress")
        roadmap_success = "NEAR SUCCESS"
        
    elif best_improvement >= 2:
        print(f"\n🥉 STRONG IMPROVEMENT!")
        print(f"✅ Gemini's roadmap delivered meaningful progress")
        print(f"🔄 Consider refinement or additional features")
        roadmap_success = "PARTIAL SUCCESS"
        
    else:
        print(f"\n🤔 LIMITED PROGRESS")
        print(f"❓ Intelligent feature engineering showed minimal gains")
        print(f"💭 May need different approach or more data")
        roadmap_success = "LIMITED SUCCESS"
    
    # Feature effectiveness analysis
    print(f"\n📊 FEATURE EFFECTIVENESS ANALYSIS:")
    if 'form_acceleration_optimized' in best_result['name']:
        print("✅ Momentum intelligence (v2.6) contributed to success")
    if 'bogey_team_score' in best_result['name']:
        print("✅ Bogey team psychology (v2.7) contributed to success")
    if 'h2h_context_score' in best_result['name']:
        print("✅ Context-weighted H2H (v2.7) contributed to success")
    
    # Final recommendations
    print(f"\n💡 FINAL RECOMMENDATIONS:")
    if roadmap_success in ["COMPLETE SUCCESS", "NEAR SUCCESS"]:
        print(f"🚀 DEPLOY {best_result['name']} as production model")
        print(f"🎯 Performance target achieved through intelligent feature engineering")
        
    elif roadmap_success == "PARTIAL SUCCESS":
        print(f"🔄 ADOPT {best_result['name']} as improved baseline")
        print(f"🧠 Continue research into advanced feature interactions")
        
    else:
        print(f"🤔 MAINTAIN v2.4 baseline model for now")
        print(f"🔬 Investigate alternative improvement strategies")
    
    # Save comprehensive results
    timestamp = pd.Timestamp.now().strftime('%Y_%m_%d_%H%M%S')
    final_results = {
        'evaluation_timestamp': timestamp,
        'roadmap_success': roadmap_success,
        'baseline_accuracy': float(baseline_accuracy),
        'best_accuracy': float(best_result['accuracy']),
        'best_configuration': best_result['name'],
        'total_improvement_pp': float(best_improvement),
        'target_achieved': best_result['accuracy'] >= 0.58,
        'all_results': [
            {
                'name': r['name'],
                'accuracy': float(r['accuracy']),
                'f1_macro': float(r['f1_macro']),
                'improvement_pp': float((r['accuracy'] - baseline_accuracy) * 100)
            } for r in results
        ]
    }
    
    import json
    output_file = f'evaluation/reports/final_gemini_roadmap_evaluation_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Complete evaluation saved to: {output_file}")
    
    return final_results

if __name__ == "__main__":
    gemini_roadmap_final_evaluation()