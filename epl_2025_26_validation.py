#!/usr/bin/env python3
"""
🏆 EPL 2025-26 VALIDATION - Enhanced Baseline Champion v2.4
===========================================================

Validation stricte sur vraies données EPL 2025-26 uniquement
Train: 2019-2025 / Test: EPL 2025-26 réelle
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import functions from TEST BOSS
import sys
sys.path.append('.')

def load_baseline_champion_v23():
    """Load reference model"""
    try:
        model = joblib.load('models/production/baseline_champion_v23.joblib')
        print("✅ Loaded Baseline Champion v2.3")
        return model
    except Exception as e:
        print(f"❌ Failed to load Baseline v2.3: {e}")
        return None

def create_enhanced_champion_v24():
    """Create Enhanced v2.4 with same architecture as v2.3"""
    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        max_features="sqrt", 
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
    return model

def extract_bet365_features(df):
    """Extract B365 features (same as TEST BOSS)"""
    df_enhanced = df.copy()
    
    # Market probabilities (normalized)
    inverse_odds = 1 / df[['B365H', 'B365D', 'B365A']].fillna(2.0)
    prob_sum = inverse_odds.sum(axis=1)
    df_enhanced['market_prob_home_b365'] = inverse_odds['B365H'] / prob_sum
    df_enhanced['market_prob_draw_b365'] = inverse_odds['B365D'] / prob_sum
    df_enhanced['market_prob_away_b365'] = inverse_odds['B365A'] / prob_sum
    
    # Geometric features
    df_enhanced['parity_gap_b365'] = abs(df['B365H'] - df['B365A'])
    df_enhanced['draw_premium_b365'] = df['B365D'] / (df['B365H'] + df['B365A'])
    df_enhanced['favorite_side_b365'] = (df['B365H'] < df['B365A']).astype(int)
    
    return df_enhanced

def calculate_rolling_features_optimized(df):
    """Optimized rolling features calculation"""
    df_enhanced = df.copy()
    df_enhanced['Date'] = pd.to_datetime(df_enhanced['Date'])
    df_enhanced = df_enhanced.sort_values('Date').reset_index(drop=True)
    
    # Initialize features
    df_enhanced['shot_accuracy_diff_roll'] = np.nan
    df_enhanced['booking_points_diff_roll'] = np.nan
    df_enhanced['corners_avg_roll'] = np.nan
    
    # Get all teams
    all_teams = set(df_enhanced['HomeTeam'].unique()) | set(df_enhanced['AwayTeam'].unique())
    
    # Pre-calculate team stats per date for efficiency
    team_stats = {}
    
    for team in all_teams:
        team_home = df_enhanced[df_enhanced['HomeTeam'] == team].copy()
        team_away = df_enhanced[df_enhanced['AwayTeam'] == team].copy()
        
        # Process home matches
        for idx in team_home.index:
            match_date = df_enhanced.loc[idx, 'Date']
            prev_home = team_home[(team_home['Date'] < match_date) & (team_home.index < idx)].tail(5)
            
            if len(prev_home) >= 3:
                # Calculate stats
                shot_acc = (prev_home['HST'] / prev_home['HS'].replace(0, np.nan)).fillna(0).mean()
                shot_acc = np.clip(shot_acc, 0, 1)
                booking_avg = prev_home['HY'].mean()
                corners_avg = prev_home['HC'].mean()
                
                # Store in main dataframe
                df_enhanced.loc[idx, 'shot_accuracy_diff_roll'] = shot_acc
                df_enhanced.loc[idx, 'booking_points_diff_roll'] = booking_avg 
                df_enhanced.loc[idx, 'corners_avg_roll'] = corners_avg
        
        # Process away matches
        for idx in team_away.index:
            match_date = df_enhanced.loc[idx, 'Date']
            prev_away = team_away[(team_away['Date'] < match_date) & (team_away.index < idx)].tail(5)
            
            if len(prev_away) >= 3:
                shot_acc = (prev_away['AST'] / prev_away['AS'].replace(0, np.nan)).fillna(0).mean()
                shot_acc = np.clip(shot_acc, 0, 1)
                booking_avg = prev_away['AY'].mean()
                corners_avg = prev_away['AC'].mean()
                
                # For away matches, we need to combine with home team stats
                home_team = df_enhanced.loc[idx, 'HomeTeam']
                home_idx_matches = df_enhanced[df_enhanced['HomeTeam'] == home_team]
                home_idx_before = home_idx_matches[(home_idx_matches['Date'] < match_date) & 
                                                 (home_idx_matches.index < idx)].tail(5)
                
                if len(home_idx_before) >= 3:
                    home_shot_acc = (home_idx_before['HST'] / home_idx_before['HS'].replace(0, np.nan)).fillna(0).mean()
                    home_shot_acc = np.clip(home_shot_acc, 0, 1)
                    home_booking = home_idx_before['HY'].mean()
                    home_corners = home_idx_before['HC'].mean()
                    
                    # Set differential features
                    df_enhanced.loc[idx, 'shot_accuracy_diff_roll'] = home_shot_acc - shot_acc
                    df_enhanced.loc[idx, 'booking_points_diff_roll'] = home_booking - booking_avg
                    df_enhanced.loc[idx, 'corners_avg_roll'] = home_corners - corners_avg
    
    return df_enhanced

def run_epl_2025_26_validation():
    """
    Validation stricte EPL 2025-26
    Train: Toutes données 2019-2025
    Test: EPL 2025-26 uniquement 
    """
    print("🏆 EPL 2025-26 VALIDATION")
    print("=" * 50)
    
    # Load data
    print("📊 Loading historical data...")
    raw_files = [
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2019_20.csv",
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2020_21.csv", 
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2021_22.csv",
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2022_23.csv",
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2023_24.csv",
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2024_25.csv",
        "/Users/maxime/Desktop/Oddsy/data/raw/E0 (9).csv"
    ]
    
    raw_dfs = []
    for file_path in raw_files:
        if os.path.exists(file_path):
            df_temp = pd.read_csv(file_path)
            raw_dfs.append(df_temp)
    
    raw_data = pd.concat(raw_dfs, ignore_index=True)
    
    # Load processed features
    processed_data = pd.read_csv("/Users/maxime/Desktop/Oddsy/data/processed/v_auto_update_20250922_093416.csv")
    
    # Merge
    raw_data['Date'] = pd.to_datetime(raw_data['Date'], errors='coerce')
    processed_data['Date'] = pd.to_datetime(processed_data['Date'])
    raw_data = raw_data.dropna(subset=['Date'])
    
    merge_cols = ['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'HS', 'AS', 'HST', 'AST', 'HY', 'AY', 'HC', 'AC']
    raw_subset = raw_data[merge_cols].copy()
    
    df_merged = processed_data.merge(raw_subset, on=['Date', 'HomeTeam', 'AwayTeam'], how='inner')
    print(f"✅ Merged: {len(df_merged)} matches")
    
    # Add enhanced features
    print("🔧 Adding enhanced features...")
    df_with_b365 = extract_bet365_features(df_merged)
    df_enhanced = calculate_rolling_features_optimized(df_with_b365)
    
    # Split: Train (2019-2025) vs Test (EPL 2025-26 only)
    df_sorted = df_enhanced.sort_values('Date').reset_index(drop=True)
    
    # Define EPL 2025-26 season (approximate dates)
    epl_2025_26_start = pd.to_datetime('2025-08-01')
    
    train_data = df_sorted[df_sorted['Date'] < epl_2025_26_start].copy()
    test_data = df_sorted[df_sorted['Date'] >= epl_2025_26_start].copy()
    
    print(f"📊 Train (2019-2025): {len(train_data)} matches")
    print(f"📊 Test (EPL 2025-26): {len(test_data)} matches")
    
    if len(test_data) == 0:
        print("❌ No EPL 2025-26 data found!")
        return
    
    # Prepare features
    base_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    enhanced_features = base_features + [
        'market_prob_home_b365', 'market_prob_draw_b365', 'market_prob_away_b365',
        'parity_gap_b365', 'draw_premium_b365', 'favorite_side_b365',
        'shot_accuracy_diff_roll', 'booking_points_diff_roll', 'corners_avg_roll'
    ]
    
    # Filter complete cases
    train_complete = train_data[base_features + ['FullTimeResult']].notna().all(axis=1)
    test_complete = test_data[base_features + ['FullTimeResult']].notna().all(axis=1)
    
    train_base = train_data[train_complete].copy()
    test_base = test_data[test_complete].copy()
    
    print(f"📊 Complete train: {len(train_base)} matches")  
    print(f"📊 Complete test: {len(test_base)} matches")
    
    # Target mapping
    target_map = {'H': 0, 'D': 1, 'A': 2}
    
    X_train_base = train_base[base_features].fillna(0.5)
    y_train = train_base['FullTimeResult'].map(target_map)
    
    X_test_base = test_base[base_features].fillna(0.5) 
    y_test = test_base['FullTimeResult'].map(target_map)
    
    print(f"📊 Test EPL 2025-26 distribution: {test_base['FullTimeResult'].value_counts().to_dict()}")
    
    # Test 1: Baseline Champion v2.3
    print("\n🏆 Testing Baseline Champion v2.3 on EPL 2025-26...")
    baseline_model = load_baseline_champion_v23()
    
    if baseline_model:
        try:
            y_pred_baseline = baseline_model.predict(X_test_base)
            accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
            print(f"✅ Baseline v2.3 EPL 2025-26: {accuracy_baseline:.4f}")
        except Exception as e:
            print(f"❌ Baseline prediction failed: {e}")
            accuracy_baseline = 0
    else:
        accuracy_baseline = 0
    
    # Test 2: Enhanced Champion v2.4
    print("\n🚀 Training Enhanced Champion v2.4...")
    
    # Check enhanced features availability
    train_enhanced_complete = train_data[enhanced_features + ['FullTimeResult']].notna().all(axis=1)  
    test_enhanced_complete = test_data[enhanced_features + ['FullTimeResult']].notna().all(axis=1)
    
    train_enhanced = train_data[train_enhanced_complete].copy()
    test_enhanced = test_data[test_enhanced_complete].copy()
    
    if len(train_enhanced) < 1000 or len(test_enhanced) < 10:
        print("⚠️  Insufficient enhanced data, using imputation...")
        X_train_enhanced = train_base[enhanced_features].fillna(0.5)
        X_test_enhanced = test_base[enhanced_features].fillna(0.5)
        y_train_enhanced = y_train
        y_test_enhanced = y_test
    else:
        X_train_enhanced = train_enhanced[enhanced_features]
        X_test_enhanced = test_enhanced[enhanced_features]
        y_train_enhanced = train_enhanced['FullTimeResult'].map(target_map)
        y_test_enhanced = test_enhanced['FullTimeResult'].map(target_map)
    
    print(f"📊 Enhanced train: {len(X_train_enhanced)} matches")
    print(f"📊 Enhanced test: {len(X_test_enhanced)} matches")
    
    # Train Enhanced v2.4
    enhanced_model = create_enhanced_champion_v24()
    enhanced_model.fit(X_train_enhanced, y_train_enhanced)
    
    y_pred_enhanced = enhanced_model.predict(X_test_enhanced)
    accuracy_enhanced = accuracy_score(y_test_enhanced, y_pred_enhanced)
    
    print(f"✅ Enhanced v2.4 EPL 2025-26: {accuracy_enhanced:.4f}")
    
    # Results
    improvement = accuracy_enhanced - accuracy_baseline
    
    print("\n" + "=" * 50)
    print("🏆 EPL 2025-26 VALIDATION RESULTS") 
    print("=" * 50)
    print(f"Baseline Champion v2.3:  {accuracy_baseline:.4f}")
    print(f"Enhanced Champion v2.4:  {accuracy_enhanced:.4f}")
    print(f"Improvement:             {improvement:+.4f}")
    print(f"Test matches:            {len(test_enhanced)} EPL 2025-26")
    
    # Decision
    if accuracy_enhanced > accuracy_baseline and accuracy_enhanced > 0.45:
        decision = "✅ CHAMPION CANDIDAT"
        reason = "Supérieur sur EPL 2025-26 réelle"
    elif accuracy_enhanced > accuracy_baseline:
        decision = "⚠️  AMÉLIORATION MARGINALE"
        reason = "Gain positif mais limité"
    else:
        decision = "❌ PERFORMANCE INSUFFISANTE"
        reason = "Pas d'amélioration vs baseline"
    
    print(f"\nDécision: {decision}")
    print(f"Reason: {reason}")
    
    # Detailed breakdown
    if len(y_test_enhanced) > 0:
        print(f"\n📊 Détail Enhanced v2.4 sur EPL 2025-26:")
        labels = ['H', 'D', 'A']
        report = classification_report(y_test_enhanced, y_pred_enhanced, 
                                     target_names=labels, output_dict=True)
        
        for label in labels:
            precision = report[label]['precision']
            recall = report[label]['recall']
            print(f"  {label}: Precision {precision:.3f}, Recall {recall:.3f}")
    
    return {
        'accuracy_baseline_epl': accuracy_baseline,
        'accuracy_enhanced_epl': accuracy_enhanced,
        'improvement_epl': improvement,
        'decision': decision,
        'reason': reason,
        'test_matches': len(test_enhanced)
    }

if __name__ == "__main__":
    results = run_epl_2025_26_validation()
    print("\n📋 FINAL VALIDATION SUMMARY:")
    for key, value in results.items():
        print(f"{key}: {value}")