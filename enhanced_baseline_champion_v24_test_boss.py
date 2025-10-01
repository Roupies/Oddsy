#!/usr/bin/env python3
"""
🧪 TEST BOSS - Enhanced Baseline Champion v2.4
==============================================

Phase 1: Bet365 Features + Rolling Features Test
Objective: 53.16% → 53.5%+ validation on same temporal split
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings('ignore')

# ======================================================================
# PHASE 1: BET365 FEATURES ENGINEERING
# ======================================================================

def extract_bet365_features(df):
    """
    Extract Bet365 market intelligence features
    FAIL FAST if B365 columns missing
    """
    print("🎯 Phase 1: Extracting Bet365 Features...")
    
    # CRITICAL: Assert B365 availability
    required_cols = ['B365H', 'B365D', 'B365A']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ FAIL FAST: Missing B365 columns: {missing_cols}")
    
    # Check NA percentage
    na_pct = df[required_cols].isna().mean().mean()
    if na_pct > 0.05:
        raise ValueError(f"❌ FAIL FAST: Too many NAs in B365 data: {na_pct:.2%}")
    
    df_enhanced = df.copy()
    
    # Feature 1: Market probabilities (normalized)
    print("  → market_probs_b365 (normalized)")
    inverse_odds = 1 / df[['B365H', 'B365D', 'B365A']].fillna(2.0)
    prob_sum = inverse_odds.sum(axis=1)
    df_enhanced['market_prob_home_b365'] = inverse_odds['B365H'] / prob_sum
    df_enhanced['market_prob_draw_b365'] = inverse_odds['B365D'] / prob_sum  
    df_enhanced['market_prob_away_b365'] = inverse_odds['B365A'] / prob_sum
    
    # Feature 2: Parity gap (chose parity_gap over market_consistency)
    print("  → parity_gap_b365")
    df_enhanced['parity_gap_b365'] = abs(df['B365H'] - df['B365A'])
    
    # Feature 3: Draw premium  
    print("  → draw_premium_b365")
    df_enhanced['draw_premium_b365'] = df['B365D'] / (df['B365H'] + df['B365A'])
    
    # Feature 4: Favorite side (binary)
    print("  → favorite_side_b365")
    df_enhanced['favorite_side_b365'] = (df['B365H'] < df['B365A']).astype(int)
    
    print(f"✅ Bet365 features added. NA check: {df_enhanced[['market_prob_home_b365', 'parity_gap_b365', 'draw_premium_b365', 'favorite_side_b365']].isna().mean().mean():.3f}")
    
    return df_enhanced

# ======================================================================
# PHASE 2: ROLLING FEATURES WITH ANTI-LEAKAGE
# ======================================================================

def calculate_rolling_features(df):
    """
    Calculate rolling features with strict temporal validation
    Shift +1, minimum k≥3 observations
    """
    print("🎯 Phase 2: Calculating Rolling Features (Anti-leakage)...")
    
    # Required columns for rolling features
    required_cols = ['HS', 'AS', 'HST', 'AST', 'HY', 'AY', 'HC', 'AC', 'Date', 'HomeTeam', 'AwayTeam']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Missing columns for rolling features: {missing_cols}")
        return df
    
    df_enhanced = df.copy()
    df_enhanced['Date'] = pd.to_datetime(df_enhanced['Date'])
    df_enhanced = df_enhanced.sort_values('Date').reset_index(drop=True)
    
    # Initialize new feature columns
    df_enhanced['shot_accuracy_diff_roll'] = np.nan
    df_enhanced['booking_points_diff_roll'] = np.nan  
    df_enhanced['corners_avg_roll'] = np.nan
    
    print("  → Processing team rolling statistics...")
    
    # Group by team for rolling calculations
    all_teams = set(df_enhanced['HomeTeam'].unique()) | set(df_enhanced['AwayTeam'].unique())
    
    for team in all_teams:
        # Get all matches for this team (home and away)
        team_home_matches = df_enhanced[df_enhanced['HomeTeam'] == team].copy()
        team_away_matches = df_enhanced[df_enhanced['AwayTeam'] == team].copy()
        
        # Calculate rolling stats for home matches
        for idx in team_home_matches.index:
            match_date = df_enhanced.loc[idx, 'Date']
            
            # Get previous 5 home matches for this team (SHIFT +1)
            prev_home = team_home_matches[
                (team_home_matches['Date'] < match_date) & 
                (team_home_matches.index < idx)
            ].tail(5)
            
            if len(prev_home) >= 3:  # Minimum k≥3 observations
                # Shot accuracy (handle division by zero)
                home_shots = prev_home['HS']
                home_sot = prev_home['HST']
                home_acc = (home_sot / home_shots.replace(0, np.nan)).fillna(0).mean()
                home_acc = np.clip(home_acc, 0, 1)  # Clip to [0,1]
                
                # Booking points (using yellow cards as proxy since HBP not in all datasets)
                home_bookings = prev_home['HY'].mean()
                
                # Corners
                home_corners = prev_home['HC'].mean()
                
                # Store home team stats (will be used when this team plays away)
                df_enhanced.loc[idx, f'{team}_shot_acc_roll'] = home_acc
                df_enhanced.loc[idx, f'{team}_bookings_roll'] = home_bookings
                df_enhanced.loc[idx, f'{team}_corners_roll'] = home_corners
        
        # Calculate rolling stats for away matches  
        for idx in team_away_matches.index:
            match_date = df_enhanced.loc[idx, 'Date']
            
            # Get previous 5 away matches for this team (SHIFT +1)
            prev_away = team_away_matches[
                (team_away_matches['Date'] < match_date) & 
                (team_away_matches.index < idx)
            ].tail(5)
            
            if len(prev_away) >= 3:  # Minimum k≥3 observations
                # Shot accuracy (handle division by zero)
                away_shots = prev_away['AS']
                away_sot = prev_away['AST']
                away_acc = (away_sot / away_shots.replace(0, np.nan)).fillna(0).mean()
                away_acc = np.clip(away_acc, 0, 1)  # Clip to [0,1]
                
                # Booking points
                away_bookings = prev_away['AY'].mean()
                
                # Corners
                away_corners = prev_away['AC'].mean()
                
                # Store away team stats
                df_enhanced.loc[idx, f'{team}_shot_acc_roll'] = away_acc
                df_enhanced.loc[idx, f'{team}_bookings_roll'] = away_bookings  
                df_enhanced.loc[idx, f'{team}_corners_roll'] = away_corners
    
    # Combine home/away rolling stats into final features
    print("  → Combining home/away rolling statistics...")
    for idx in df_enhanced.index:
        home_team = df_enhanced.loc[idx, 'HomeTeam']
        away_team = df_enhanced.loc[idx, 'AwayTeam']
        
        # Shot accuracy difference
        home_acc_col = f'{home_team}_shot_acc_roll'
        away_acc_col = f'{away_team}_shot_acc_roll'
        if home_acc_col in df_enhanced.columns and away_acc_col in df_enhanced.columns:
            home_acc = df_enhanced.loc[idx, home_acc_col]
            away_acc = df_enhanced.loc[idx, away_acc_col]
            if pd.notna(home_acc) and pd.notna(away_acc):
                df_enhanced.loc[idx, 'shot_accuracy_diff_roll'] = home_acc - away_acc
        
        # Booking points difference  
        home_book_col = f'{home_team}_bookings_roll'
        away_book_col = f'{away_team}_bookings_roll'
        if home_book_col in df_enhanced.columns and away_book_col in df_enhanced.columns:
            home_book = df_enhanced.loc[idx, home_book_col]
            away_book = df_enhanced.loc[idx, away_book_col]
            if pd.notna(home_book) and pd.notna(away_book):
                df_enhanced.loc[idx, 'booking_points_diff_roll'] = home_book - away_book
        
        # Corners average
        home_corn_col = f'{home_team}_corners_roll'
        away_corn_col = f'{away_team}_corners_roll'
        if home_corn_col in df_enhanced.columns and away_corn_col in df_enhanced.columns:
            home_corn = df_enhanced.loc[idx, home_corn_col]
            away_corn = df_enhanced.loc[idx, away_corn_col]
            if pd.notna(home_corn) and pd.notna(away_corn):
                df_enhanced.loc[idx, 'corners_avg_roll'] = home_corn - away_corn
    
    # Clean up temporary team columns
    team_cols = [col for col in df_enhanced.columns if any(team in col for team in all_teams) and '_roll' in col]
    df_enhanced = df_enhanced.drop(columns=team_cols)
    
    # Report rolling features stats
    rolling_cols = ['shot_accuracy_diff_roll', 'booking_points_diff_roll', 'corners_avg_roll']
    na_stats = df_enhanced[rolling_cols].isna().mean()
    print(f"✅ Rolling features calculated. NA rates: {dict(na_stats)}")
    
    return df_enhanced

# ======================================================================
# BASELINE CHAMPION v2.3 vs ENHANCED v2.4 COMPARISON
# ======================================================================

def load_baseline_champion_v23():
    """Load the reference Baseline Champion v2.3"""
    try:
        model = joblib.load('models/production/baseline_champion_v23.joblib')
        print("✅ Loaded Baseline Champion v2.3")
        return model
    except Exception as e:
        print(f"❌ Failed to load Baseline Champion v2.3: {e}")
        return None

def create_enhanced_champion_v24():
    """Create Enhanced Baseline Champion v2.4 with same architecture"""
    print("🏗️ Creating Enhanced Baseline Champion v2.4...")
    
    # Same architecture as v2.3
    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        max_features="sqrt",
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    # Same calibration as v2.3
    model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
    
    return model

def run_ab_test(df_enhanced):
    """
    A/B test Baseline v2.3 vs Enhanced v2.4
    Same temporal split: 1900 train / 380 test
    """
    print("🧪 Phase 3: A/B Test Baseline v2.3 vs Enhanced v2.4")
    print("=" * 60)
    
    # Prepare data - same split as previous tests
    df_sorted = df_enhanced.sort_values('Date').reset_index(drop=True)
    
    # Use complete data only
    required_base_cols = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    enhanced_cols = [
        'market_prob_home_b365', 'market_prob_draw_b365', 'market_prob_away_b365',
        'parity_gap_b365', 'draw_premium_b365', 'favorite_side_b365',
        'shot_accuracy_diff_roll', 'booking_points_diff_roll', 'corners_avg_roll'
    ]
    
    # Filter complete cases for base features
    base_complete = df_sorted[required_base_cols + ['FullTimeResult']].notna().all(axis=1)
    df_base = df_sorted[base_complete].copy()
    
    print(f"📊 Dataset: {len(df_base)} complete matches for base features")
    
    # Use same 1900/380 split
    if len(df_base) < 2280:
        print(f"⚠️  Dataset has only {len(df_base)} matches")
        train_size = min(1900, len(df_base) - 100)
        test_size = min(380, len(df_base) - train_size)
    else:
        train_size = 1900
        test_size = 380
    
    train_df = df_base.iloc[:train_size].copy()
    test_df = df_base.iloc[train_size:train_size + test_size].copy()
    
    print(f"📊 Train: {len(train_df)} matches ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"📊 Test: {len(test_df)} matches ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    # Prepare baseline features (v2.3)
    X_train_base = train_df[required_base_cols].fillna(0.5)
    X_test_base = test_df[required_base_cols].fillna(0.5)
    
    # Target mapping
    target_map = {'H': 0, 'D': 1, 'A': 2}
    y_train = train_df['FullTimeResult'].map(target_map)
    y_test = test_df['FullTimeResult'].map(target_map)
    
    print(f"📊 Train distribution: {y_train.value_counts().to_dict()}")
    print(f"📊 Test distribution: {y_test.value_counts().to_dict()}")
    
    # Test 1: Load and test Baseline Champion v2.3
    print("\\n🏆 Testing Baseline Champion v2.3...")
    baseline_model = load_baseline_champion_v23()
    
    if baseline_model is not None:
        try:
            y_pred_baseline = baseline_model.predict(X_test_base)
            accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
            print(f"✅ Baseline v2.3 Accuracy: {accuracy_baseline:.4f}")
        except Exception as e:
            print(f"❌ Baseline v2.3 prediction failed: {e}")
            accuracy_baseline = 0
    else:
        accuracy_baseline = 0
    
    # Test 2: Train and test Enhanced Champion v2.4  
    print("\\n🚀 Training Enhanced Champion v2.4...")
    
    # Prepare enhanced features (v2.4)
    # Check availability of enhanced features
    available_enhanced = [col for col in enhanced_cols if col in train_df.columns]
    print(f"  Available enhanced features: {len(available_enhanced)}/{len(enhanced_cols)}")
    
    all_features_v24 = required_base_cols + available_enhanced
    
    # Filter for complete enhanced cases
    enhanced_complete = train_df[all_features_v24].notna().all(axis=1)
    train_df_enhanced = train_df[enhanced_complete].copy()
    
    enhanced_complete_test = test_df[all_features_v24].notna().all(axis=1) 
    test_df_enhanced = test_df[enhanced_complete_test].copy()
    
    if len(train_df_enhanced) < 1000 or len(test_df_enhanced) < 100:
        print(f"⚠️  Not enough complete enhanced data: {len(train_df_enhanced)} train, {len(test_df_enhanced)} test")
        print("  Falling back to basic imputation...")
        X_train_enhanced = train_df[all_features_v24].fillna(0.5)
        X_test_enhanced = test_df[all_features_v24].fillna(0.5)
        y_train_enhanced = y_train
        y_test_enhanced = y_test
    else:
        X_train_enhanced = train_df_enhanced[all_features_v24]
        X_test_enhanced = test_df_enhanced[all_features_v24]  
        y_train_enhanced = train_df_enhanced['FullTimeResult'].map(target_map)
        y_test_enhanced = test_df_enhanced['FullTimeResult'].map(target_map)
        
        print(f"📊 Enhanced train: {len(X_train_enhanced)} matches")
        print(f"📊 Enhanced test: {len(X_test_enhanced)} matches")
    
    # Train Enhanced v2.4
    enhanced_model = create_enhanced_champion_v24()
    enhanced_model.fit(X_train_enhanced, y_train_enhanced)
    
    y_pred_enhanced = enhanced_model.predict(X_test_enhanced)
    accuracy_enhanced = accuracy_score(y_test_enhanced, y_pred_enhanced)
    
    print(f"✅ Enhanced v2.4 Accuracy: {accuracy_enhanced:.4f}")
    
    # Results comparison
    print("\\n" + "=" * 60)
    print("🏆 TEST BOSS RESULTS")
    print("=" * 60)
    print(f"Baseline Champion v2.3:  {accuracy_baseline:.4f}")
    print(f"Enhanced Champion v2.4:  {accuracy_enhanced:.4f}")
    
    improvement = accuracy_enhanced - accuracy_baseline
    print(f"Improvement:             {improvement:+.4f}")
    
    # Decision criteria
    if accuracy_enhanced > 0.535:
        decision = "🟢 GO"
        reason = "Exceeds 53.5% threshold"
    elif accuracy_enhanced >= 0.5316:
        decision = "🟡 INVESTIGATE" 
        reason = "Marginal gain - analyze feature contribution"
    else:
        decision = "🔴 NO-GO"
        reason = "Performance degradation vs baseline"
    
    print(f"\\nDecision: {decision}")
    print(f"Reason: {reason}")
    
    return {
        'accuracy_baseline': accuracy_baseline,
        'accuracy_enhanced': accuracy_enhanced, 
        'improvement': improvement,
        'decision': decision,
        'reason': reason,
        'features_baseline': len(required_base_cols),
        'features_enhanced': len(all_features_v24),
        'enhanced_features': available_enhanced
    }

# ======================================================================
# MAIN TEST BOSS EXECUTION
# ======================================================================

def main():
    """Execute TEST BOSS validation"""
    print("🧪 TEST BOSS - Enhanced Baseline Champion v2.4")
    print("Goal: Validate 53.16% → 53.5%+ improvement potential")
    print("=" * 60)
    
    try:
        # Load historical raw data with B365 odds
        print("📊 Loading historical raw data with B365 odds...")
        raw_files = [
            "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2019_20.csv",
            "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2020_21.csv", 
            "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2021_22.csv",
            "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2022_23.csv",
            "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2023_24.csv",
            "/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/football_data_2024_25.csv",
            "/Users/maxime/Desktop/Oddsy/data/raw/E0 (9).csv"  # 2025 season
        ]
        
        raw_dfs = []
        for file_path in raw_files:
            if os.path.exists(file_path):
                try:
                    df_temp = pd.read_csv(file_path)
                    print(f"  Loaded: {os.path.basename(file_path)} ({len(df_temp)} rows)")
                    raw_dfs.append(df_temp)
                except Exception as e:
                    print(f"  Failed to load: {os.path.basename(file_path)} - {e}")
        
        if not raw_dfs:
            raise ValueError("No raw data files could be loaded")
        
        # Combine all raw data
        raw_data = pd.concat(raw_dfs, ignore_index=True)
        print(f"📊 Combined raw data: {len(raw_data)} total rows")
        
        # Load processed features dataset  
        print("📊 Loading processed features dataset...")
        processed_data = pd.read_csv("/Users/maxime/Desktop/Oddsy/data/processed/v_auto_update_20250922_093416.csv")
        
        # Merge B365 odds with processed features
        print("🔗 Merging B365 odds with processed features...")
        
        # Simple merge on Date, HomeTeam, AwayTeam - handle mixed date formats
        try:
            raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='mixed', dayfirst=True, errors='coerce')
        except:
            # Fallback: try with errors='coerce' to handle malformed dates
            raw_data['Date'] = pd.to_datetime(raw_data['Date'], errors='coerce')
        
        # Remove rows with invalid dates
        raw_data = raw_data.dropna(subset=['Date'])
        processed_data['Date'] = pd.to_datetime(processed_data['Date'])
        
        # Merge key columns from raw data
        merge_cols = ['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'HS', 'AS', 'HST', 'AST', 'HY', 'AY', 'HC', 'AC']
        raw_subset = raw_data[merge_cols].copy()
        
        # Merge datasets
        df_merged = processed_data.merge(
            raw_subset,
            on=['Date', 'HomeTeam', 'AwayTeam'],
            how='inner',
            suffixes=('', '_raw')
        )
        
        print(f"✅ Merged dataset: {len(df_merged)} matches")
        
        # Phase 1: Extract Bet365 features
        df_with_b365 = extract_bet365_features(df_merged)
        
        # Phase 2: Calculate rolling features
        df_enhanced = calculate_rolling_features(df_with_b365)
        
        # Phase 3: Run A/B test
        results = run_ab_test(df_enhanced)
        
        # Final report
        print("\\n" + "=" * 60)
        print("📋 TEST BOSS FINAL REPORT")
        print("=" * 60)
        for key, value in results.items():
            print(f"{key}: {value}")
        
        return results
        
    except Exception as e:
        print(f"❌ TEST BOSS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()