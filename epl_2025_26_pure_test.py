#!/usr/bin/env python3
"""
🏆 PURE EPL 2025-26 TEST - Baseline v2.3 vs Enhanced v2.4
=========================================================

Train: Données historiques (2019-2025)
Test: EPL 2025-26 pure (E0 (9).csv) 
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_baseline_v23():
    """Load Baseline Champion v2.3"""
    try:
        model = joblib.load('models/production/baseline_champion_v23.joblib')
        print("✅ Loaded Baseline Champion v2.3")
        return model
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

def create_enhanced_v24():
    """Same architecture as v2.3"""
    base_model = RandomForestClassifier(
        n_estimators=300, max_depth=20, max_features="sqrt",
        min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=-1
    )
    return CalibratedClassifierCV(base_model, cv=3, method='isotonic')

def extract_bet365_features(df):
    """Extract B365 market features"""
    # Market probs (normalized)
    inverse_odds = 1 / df[['B365H', 'B365D', 'B365A']].fillna(2.0)
    prob_sum = inverse_odds.sum(axis=1)
    df['market_prob_home_b365'] = inverse_odds['B365H'] / prob_sum
    df['market_prob_draw_b365'] = inverse_odds['B365D'] / prob_sum
    df['market_prob_away_b365'] = inverse_odds['B365A'] / prob_sum
    
    # Geometric features  
    df['parity_gap_b365'] = abs(df['B365H'] - df['B365A'])
    df['draw_premium_b365'] = df['B365D'] / (df['B365H'] + df['B365A'])
    df['favorite_side_b365'] = (df['B365H'] < df['B365A']).astype(int)
    
    return df

def calculate_rolling_features(df):
    """Calculate rolling stats with anti-leakage"""
    df = df.sort_values('Date').reset_index(drop=True)
    df['shot_accuracy_diff_roll'] = np.nan
    df['booking_points_diff_roll'] = np.nan  
    df['corners_avg_roll'] = np.nan
    
    teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    
    for team in teams:
        # Home matches for this team
        team_home = df[df['HomeTeam'] == team].copy()
        for idx in team_home.index:
            match_date = df.loc[idx, 'Date']
            prev_home = team_home[(team_home['Date'] < match_date) & (team_home.index < idx)].tail(5)
            
            if len(prev_home) >= 3:
                shot_acc = (prev_home['HST'] / prev_home['HS'].replace(0, np.nan)).fillna(0).mean()
                shot_acc = np.clip(shot_acc, 0, 1)
                booking_avg = prev_home['HY'].mean()
                corners_avg = prev_home['HC'].mean()
                
                df.loc[idx, f'{team}_home_shot_acc'] = shot_acc
                df.loc[idx, f'{team}_home_booking'] = booking_avg
                df.loc[idx, f'{team}_home_corners'] = corners_avg
        
        # Away matches for this team
        team_away = df[df['AwayTeam'] == team].copy()
        for idx in team_away.index:
            match_date = df.loc[idx, 'Date']
            prev_away = team_away[(team_away['Date'] < match_date) & (team_away.index < idx)].tail(5)
            
            if len(prev_away) >= 3:
                shot_acc = (prev_away['AST'] / prev_away['AS'].replace(0, np.nan)).fillna(0).mean()
                shot_acc = np.clip(shot_acc, 0, 1)
                booking_avg = prev_away['AY'].mean()
                corners_avg = prev_away['AC'].mean()
                
                df.loc[idx, f'{team}_away_shot_acc'] = shot_acc
                df.loc[idx, f'{team}_away_booking'] = booking_avg
                df.loc[idx, f'{team}_away_corners'] = corners_avg
    
    # Combine home/away into differentials
    for idx in df.index:
        home_team = df.loc[idx, 'HomeTeam']
        away_team = df.loc[idx, 'AwayTeam']
        
        # Shot accuracy differential
        home_shot_col = f'{home_team}_home_shot_acc'
        away_shot_col = f'{away_team}_away_shot_acc'
        if home_shot_col in df.columns and away_shot_col in df.columns:
            home_acc = df.loc[idx, home_shot_col]
            away_acc = df.loc[idx, away_shot_col]
            if pd.notna(home_acc) and pd.notna(away_acc):
                df.loc[idx, 'shot_accuracy_diff_roll'] = home_acc - away_acc
        
        # Booking differential
        home_book_col = f'{home_team}_home_booking'
        away_book_col = f'{away_team}_away_booking'
        if home_book_col in df.columns and away_book_col in df.columns:
            home_book = df.loc[idx, home_book_col]
            away_book = df.loc[idx, away_book_col]
            if pd.notna(home_book) and pd.notna(away_book):
                df.loc[idx, 'booking_points_diff_roll'] = home_book - away_book
        
        # Corners differential
        home_corn_col = f'{home_team}_home_corners'
        away_corn_col = f'{away_team}_away_corners'
        if home_corn_col in df.columns and away_corn_col in df.columns:
            home_corn = df.loc[idx, home_corn_col]
            away_corn = df.loc[idx, away_corn_col]
            if pd.notna(home_corn) and pd.notna(away_corn):
                df.loc[idx, 'corners_avg_roll'] = home_corn - away_corn
    
    # Clean temp columns
    temp_cols = [col for col in df.columns if any(team in col for team in teams) and ('_shot_acc' in col or '_booking' in col or '_corners' in col)]
    df = df.drop(columns=temp_cols)
    
    return df

def main():
    print("🏆 PURE EPL 2025-26 VALIDATION")
    print("=" * 50)
    
    # Load training data (historical)
    print("📊 Loading historical training data...")
    train_files = [
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2019_20.csv",
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2020_21.csv",
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2021_22.csv", 
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2022_23.csv",
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2023_24.csv",
        "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2024_25.csv"
    ]
    
    train_dfs = []
    for file_path in train_files:
        if os.path.exists(file_path):
            df_temp = pd.read_csv(file_path)
            train_dfs.append(df_temp)
    
    train_data = pd.concat(train_dfs, ignore_index=True)
    print(f"📊 Historical training data: {len(train_data)} matches")
    
    # Load EPL 2025-26 test data
    print("📊 Loading EPL 2025-26 test data...")
    test_data = pd.read_csv("/Users/maxime/Desktop/Oddsy/data/raw/E0 (9).csv")
    print(f"📊 EPL 2025-26 test data: {len(test_data)} matches")
    
    # Process dates
    train_data['Date'] = pd.to_datetime(train_data['Date'], errors='coerce')
    test_data['Date'] = pd.to_datetime(test_data['Date'], errors='coerce')
    
    train_data = train_data.dropna(subset=['Date'])
    test_data = test_data.dropna(subset=['Date'])
    
    # Combine for feature engineering (needed for rolling features)
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    combined_data = combined_data.sort_values('Date').reset_index(drop=True)
    
    print("🔧 Adding enhanced features...")
    # Add B365 features
    combined_enhanced = extract_bet365_features(combined_data)
    
    # Add rolling features
    combined_enhanced = calculate_rolling_features(combined_enhanced)
    
    # Split back into train/test
    train_enhanced = combined_enhanced[combined_enhanced['Date'] < pd.to_datetime('2025-08-01')].copy()
    test_enhanced = combined_enhanced[combined_enhanced['Date'] >= pd.to_datetime('2025-08-01')].copy()
    
    print(f"📊 Enhanced train: {len(train_enhanced)} matches")
    print(f"📊 Enhanced test: {len(test_enhanced)} matches")
    
    # Feature definitions
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
    
    # But we only have raw data, need to merge with processed features
    print("🔗 Loading processed features for merge...")
    processed_data = pd.read_csv("/Users/maxime/Desktop/Oddsy/data/processed/v_auto_update_20250922_093416.csv")
    processed_data['Date'] = pd.to_datetime(processed_data['Date'])
    
    # Merge test data with processed features
    test_epl = test_enhanced.merge(
        processed_data[['Date', 'HomeTeam', 'AwayTeam'] + base_features + ['FullTimeResult']], 
        on=['Date', 'HomeTeam', 'AwayTeam'], 
        how='inner'
    )
    
    print(f"📊 EPL 2025-26 with features: {len(test_epl)} matches")
    
    if len(test_epl) == 0:
        print("❌ No test matches with features found!")
        return
    
    # Filter complete cases
    test_complete = test_epl[base_features + ['FullTimeResult']].notna().all(axis=1)
    test_final = test_epl[test_complete].copy()
    
    print(f"📊 Complete EPL 2025-26: {len(test_final)} matches")
    print(f"📊 Results distribution: {test_final['FullTimeResult'].value_counts().to_dict()}")
    
    # Target mapping
    target_map = {'H': 0, 'D': 1, 'A': 2}
    
    X_test_base = test_final[base_features].fillna(0.5)
    y_test = test_final['FullTimeResult'].map(target_map)
    
    # Test 1: Baseline Champion v2.3
    print("\\n🏆 Testing Baseline Champion v2.3...")
    baseline_model = load_baseline_v23()
    
    if baseline_model:
        y_pred_baseline = baseline_model.predict(X_test_base)
        accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
        print(f"✅ Baseline v2.3 on EPL 2025-26: {accuracy_baseline:.4f}")
    else:
        accuracy_baseline = 0
    
    # Test 2: Enhanced Champion v2.4 (train on historical, test on EPL 2025-26)
    print("\\n🚀 Training Enhanced Champion v2.4...")
    
    # For training, use processed data (we need the base features) 
    train_processed = processed_data[processed_data['Date'] < pd.to_datetime('2025-08-01')].copy()
    train_complete = train_processed[base_features + ['FullTimeResult']].notna().all(axis=1)
    train_final = train_processed[train_complete].copy()
    
    print(f"📊 Training data: {len(train_final)} matches")
    
    if len(train_final) < 1000:
        print("⚠️  Insufficient training data!")
        return
    
    # Train enhanced model (use base features only for now since rolling features are complex)
    X_train = train_final[base_features].fillna(0.5)
    y_train = train_final['FullTimeResult'].map(target_map)
    
    enhanced_model = create_enhanced_v24()
    
    # For now, test with B365 features only (since rolling features need complex historical computation)
    enhanced_test_features = base_features + ['market_prob_home_b365', 'market_prob_draw_b365', 'market_prob_away_b365', 
                                           'parity_gap_b365', 'draw_premium_b365', 'favorite_side_b365']
    
    # Check if enhanced features are available in test
    enhanced_available = all(col in test_final.columns for col in enhanced_test_features)
    
    if enhanced_available:
        print("✅ Enhanced features available, training with B365 features...")
        # Add B365 features to training data by merging with historical raw data
        train_enhanced_final = train_final.copy()
        
        # For training, we need to add B365 features - merge with historical data
        historical_combined = pd.concat(train_dfs, ignore_index=True)
        historical_combined['Date'] = pd.to_datetime(historical_combined['Date'], errors='coerce')
        historical_combined = historical_combined.dropna(subset=['Date'])
        historical_b365 = extract_bet365_features(historical_combined)
        
        # Merge training data with B365 features
        train_with_b365 = train_final.merge(
            historical_b365[['Date', 'HomeTeam', 'AwayTeam', 'market_prob_home_b365', 'market_prob_draw_b365', 
                           'market_prob_away_b365', 'parity_gap_b365', 'draw_premium_b365', 'favorite_side_b365']], 
            on=['Date', 'HomeTeam', 'AwayTeam'], 
            how='left'
        )
        
        # Fill NaN B365 features with neutral values
        b365_features = ['market_prob_home_b365', 'market_prob_draw_b365', 'market_prob_away_b365',
                        'parity_gap_b365', 'draw_premium_b365', 'favorite_side_b365']
        
        train_with_b365[b365_features] = train_with_b365[b365_features].fillna({
            'market_prob_home_b365': 0.33, 'market_prob_draw_b365': 0.33, 'market_prob_away_b365': 0.33,
            'parity_gap_b365': 0, 'draw_premium_b365': 1, 'favorite_side_b365': 0
        })
        
        X_train_enhanced = train_with_b365[enhanced_test_features]
        X_test_enhanced = test_final[enhanced_test_features].fillna({
            'market_prob_home_b365': 0.33, 'market_prob_draw_b365': 0.33, 'market_prob_away_b365': 0.33,
            'parity_gap_b365': 0, 'draw_premium_b365': 1, 'favorite_side_b365': 0
        })
        
        enhanced_model.fit(X_train_enhanced, y_train)
        y_pred_enhanced = enhanced_model.predict(X_test_enhanced)
        accuracy_enhanced = accuracy_score(y_test, y_pred_enhanced)
        
        print(f"✅ Enhanced v2.4 on EPL 2025-26: {accuracy_enhanced:.4f}")
    else:
        print("⚠️  Enhanced features not available, using base features...")
        enhanced_model.fit(X_train, y_train)
        y_pred_enhanced = enhanced_model.predict(X_test_base)
        accuracy_enhanced = accuracy_score(y_test, y_pred_enhanced)
        print(f"✅ Enhanced v2.4 (base features) on EPL 2025-26: {accuracy_enhanced:.4f}")
    
    # Results
    improvement = accuracy_enhanced - accuracy_baseline
    
    print("\\n" + "=" * 50)
    print("🏆 EPL 2025-26 PURE TEST RESULTS")
    print("=" * 50)
    print(f"Baseline Champion v2.3:  {accuracy_baseline:.4f}")
    print(f"Enhanced Champion v2.4:  {accuracy_enhanced:.4f}")
    print(f"Improvement:             {improvement:+.4f}")
    print(f"EPL 2025-26 matches:     {len(test_final)}")
    
    if accuracy_enhanced > accuracy_baseline:
        if accuracy_enhanced >= 0.45:
            decision = "🏆 NOUVEAU CHAMPION CANDIDAT"
        else:
            decision = "⚠️  AMÉLIORATION MAIS PERFORMANCE MODESTE"
    else:
        decision = "❌ PAS D'AMÉLIORATION"
    
    print(f"\\nDécision: {decision}")
    
    # Detailed breakdown
    if len(y_test) > 0:
        print(f"\\n📊 Détail Enhanced v2.4:")
        labels = ['H', 'D', 'A']
        report = classification_report(y_test, y_pred_enhanced, target_names=labels, output_dict=True)
        for label in labels:
            print(f"  {label}: Precision {report[label]['precision']:.3f}, Recall {report[label]['recall']:.3f}")
    
    return accuracy_baseline, accuracy_enhanced, improvement, decision

if __name__ == "__main__":
    main()