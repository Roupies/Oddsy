#!/usr/bin/env python3
"""
Optimize Momentum Windows
Test different acceleration windows to maximize the +0.5pp improvement from form acceleration.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def calculate_form_points(result):
    """Convert result to points."""
    if result == 'H':
        return 3, 0
    elif result == 'A':
        return 0, 3
    else:
        return 1, 1

def calculate_custom_acceleration(df, short_window, long_window):
    """Calculate form acceleration with custom windows."""
    print(f"📊 Calculating {short_window}vs{long_window} acceleration...")
    
    form_data = []
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_form_history = {team: [] for team in all_teams}
    
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    for idx, match in df_sorted.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        result = match['FullTimeResult']
        
        # Calculate form BEFORE this match
        home_short = np.mean(team_form_history[home_team][-short_window:]) if len(team_form_history[home_team]) >= short_window else 1.0
        home_long = np.mean(team_form_history[home_team][-long_window:]) if len(team_form_history[home_team]) >= long_window else 1.0
        away_short = np.mean(team_form_history[away_team][-short_window:]) if len(team_form_history[away_team]) >= short_window else 1.0
        away_long = np.mean(team_form_history[away_team][-long_window:]) if len(team_form_history[away_team]) >= long_window else 1.0
        
        # Form acceleration
        home_acceleration = home_short - home_long
        away_acceleration = away_short - away_long
        acceleration_diff = home_acceleration - away_acceleration
        
        form_data.append({
            'Date': match['Date'],
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'acceleration_diff': acceleration_diff
        })
        
        # Update histories
        home_points, away_points = calculate_form_points(result)
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
    form_df['acceleration_normalized'] = (form_df['acceleration_diff'] - min_val) / (max_val - min_val)
    
    return form_df

def test_acceleration_window(df, short_window, long_window):
    """Test specific acceleration window configuration."""
    
    # Calculate acceleration for this window
    form_df = calculate_custom_acceleration(df, short_window, long_window)
    
    # Merge with main data
    df_test = df.merge(form_df[['Date', 'HomeTeam', 'AwayTeam', 'acceleration_normalized']], 
                      on=['Date', 'HomeTeam', 'AwayTeam'], how='left')
    
    # Prepare features
    v24_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10',
        'shots_diff_normalized', 'corners_diff_normalized', 'matchday_normalized',
        'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
    ]
    
    test_features = v24_features + ['acceleration_normalized']
    
    # Temporal split
    cutoff_date = pd.to_datetime('2023-05-01')
    train_df = df_test[df_test['Date'] < cutoff_date].copy()
    test_df = df_test[df_test['Date'] >= cutoff_date].copy()
    
    # Quick cascade test
    X_train = train_df[test_features].fillna(train_df[test_features].median())
    y_train = (train_df['FullTimeResult'] == 'D').astype(int)
    X_test = test_df[test_features].fillna(train_df[test_features].median())
    y_test_true = test_df['FullTimeResult']
    
    # Stage 1: Draw detection
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    stage1_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    stage1_model.fit(X_train_balanced, y_train_balanced)
    
    # Stage 2: Home vs Away
    train_non_draw = train_df[train_df['FullTimeResult'] != 'D'].copy()
    X_train_s2 = train_non_draw[test_features].fillna(train_non_draw[test_features].median())
    y_train_s2 = train_non_draw['FullTimeResult'].map({'H': 0, 'A': 1})
    
    stage2_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    stage2_model.fit(X_train_s2, y_train_s2)
    
    # Prediction
    stage1_proba = stage1_model.predict_proba(X_test)[:, 1]
    draw_mask = stage1_proba >= 0.7
    
    y_pred = np.full(len(X_test), 'D', dtype=object)
    if (~draw_mask).sum() > 0:
        stage2_pred = stage2_model.predict(X_test[~draw_mask])
        y_pred[~draw_mask] = np.where(stage2_pred == 0, 'H', 'A')
    
    accuracy = accuracy_score(y_test_true, y_pred)
    
    # Get feature importance for acceleration
    acceleration_importance = 0
    feature_names = list(test_features)
    if 'acceleration_normalized' in feature_names:
        acc_idx = feature_names.index('acceleration_normalized')
        acceleration_importance = stage1_model.feature_importances_[acc_idx]
    
    return accuracy, acceleration_importance

def optimize_acceleration_windows():
    """Test different acceleration windows to find optimal configuration."""
    print("🔧 OPTIMIZING MOMENTUM ACCELERATION WINDOWS")
    print("=" * 60)
    print("Current best: 3vs10 windows gave +0.5pp improvement")
    print("Testing various short vs long window combinations")
    print()
    
    # Load data
    data_path = '/Users/maxime/Desktop/Oddsy/data/processed/v13_xg_corrected_features_2025_09_02_113048.csv'
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Test different window combinations
    window_configs = [
        # Short vs Long comparisons
        (2, 8),   # Very recent vs medium-term
        (3, 10),  # Current best (baseline)
        (2, 10),  # Very recent vs long-term
        (4, 12),  # Slightly longer vs longer-term
        (3, 15),  # Current short vs very long-term
        (5, 15),  # Medium vs very long-term
        (2, 6),   # Tighter window
        (4, 8),   # Medium comparison
    ]
    
    print(f"Testing {len(window_configs)} window configurations...")
    print(f"{'Short':<6} {'Long':<6} {'Accuracy':<9} {'Improvement':<12} {'Acc_Imp':<8} {'Status':<10}")
    print("-" * 65)
    
    results = []
    baseline_accuracy = None
    
    for short, long in window_configs:
        accuracy, acc_importance = test_acceleration_window(df, short, long)
        
        if (short, long) == (3, 10):  # Our baseline
            baseline_accuracy = accuracy
            improvement_pp = 0
            status = "BASELINE"
        else:
            improvement_pp = (accuracy - baseline_accuracy) * 100 if baseline_accuracy else 0
            if improvement_pp > 0.3:
                status = "🟢 BETTER"
            elif improvement_pp > 0:
                status = "🟡 SLIGHT"
            elif improvement_pp > -0.3:
                status = "🟠 SIMILAR"
            else:
                status = "🔴 WORSE"
        
        print(f"{short:<6} {long:<6} {accuracy:<9.1%} {improvement_pp:<12.1f}pp {acc_importance:<8.3f} {status:<10}")
        
        results.append({
            'short_window': short,
            'long_window': long,
            'accuracy': accuracy,
            'improvement_pp': improvement_pp,
            'acceleration_importance': acc_importance
        })
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print(f"\n🏆 OPTIMAL CONFIGURATION:")
    print(f"Windows: {best_result['short_window']} vs {best_result['long_window']}")
    print(f"Accuracy: {best_result['accuracy']:.1%}")
    print(f"Improvement: {best_result['improvement_pp']:+.1f}pp vs 3vs10 baseline")
    print(f"Feature Importance: {best_result['acceleration_importance']:.3f}")
    
    # Performance assessment
    if best_result['improvement_pp'] > 0.5:
        print(f"\n✅ SIGNIFICANT IMPROVEMENT FOUND!")
        print(f"Recommend using {best_result['short_window']}vs{best_result['long_window']} windows")
    elif best_result['improvement_pp'] > 0.2:
        print(f"\n🟡 MODEST IMPROVEMENT FOUND")
        print(f"Consider using {best_result['short_window']}vs{best_result['long_window']} windows")
    else:
        print(f"\n🟠 MINIMAL IMPROVEMENT")
        print(f"Current 3vs10 windows remain optimal")
    
    # Final recommendation
    if best_result['accuracy'] >= 0.56:  # 56%+ would be meaningful progress toward 58% target
        print(f"\n🎯 MOMENTUM OPTIMIZATION SUCCESSFUL!")
        print(f"Ready for v2.6 production with optimized windows")
    else:
        print(f"\n🤔 MOMENTUM OPTIMIZATION COMPLETE")
        print(f"Consider proceeding to Sprint v2.7 (H2H Intelligence)")
    
    return best_result

if __name__ == "__main__":
    optimize_acceleration_windows()