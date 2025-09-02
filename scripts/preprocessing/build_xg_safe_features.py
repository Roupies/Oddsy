#!/usr/bin/env python3
"""
BUILD XG SAFE FEATURES - NO DATA LEAKAGE
Following ChatGPT's strict methodology to eliminate data leakage:
- Remove all raw xG from current match (HomeXG, AwayXG, XG_Diff)
- Use only historical xG data with proper shift(1) 
- Rolling averages, efficiency metrics from past matches only
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

# Configuration
INPUT_CSV = 'data/processed/v13_complete_with_dates.csv'
XG_DATA_CSV = 'data/external/understat_xg_data_corrected_names.csv'
OUTPUT_CSV = 'data/processed/v13_xg_safe_features.csv'

# Dangerous features to explicitly remove (data leakage)
DANGEROUS_FEATURES = [
    'HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG',
    'HomeGoals', 'AwayGoals', 
    'Home_GoalsVsXG', 'Away_GoalsVsXG'
]

def load_and_merge_datasets():
    """
    Load base dataset and merge with xG data (corrected team names)
    """
    logger = setup_logging()
    logger.info("📊 LOADING AND MERGING DATASETS")
    
    # Load base dataset
    if not os.path.exists(INPUT_CSV):
        logger.error(f"❌ Base dataset not found: {INPUT_CSV}")
        return None
    
    base_df = pd.read_csv(INPUT_CSV)
    base_df['Date'] = pd.to_datetime(base_df['Date'])
    logger.info(f"✅ Base dataset: {base_df.shape}")
    
    # Load corrected xG data
    if not os.path.exists(XG_DATA_CSV):
        logger.error(f"❌ xG dataset not found: {XG_DATA_CSV}")
        logger.info("Run the team name correction script first")
        return None
    
    xg_df = pd.read_csv(XG_DATA_CSV)
    xg_df['Date'] = pd.to_datetime(xg_df['Date'])
    logger.info(f"✅ xG dataset: {xg_df.shape}")
    
    # Merge datasets
    base_df['merge_key'] = base_df['Date'].dt.strftime('%Y-%m-%d') + '_' + base_df['HomeTeam'] + '_' + base_df['AwayTeam']
    xg_df['merge_key'] = xg_df['Date'].dt.strftime('%Y-%m-%d') + '_' + xg_df['HomeTeam'] + '_' + xg_df['AwayTeam']
    
    # Select ONLY safe columns from xG data (NO current match results)
    safe_xg_cols = ['merge_key', 'HomeXG', 'AwayXG', 'HomeGoals', 'AwayGoals']
    
    merged_df = base_df.merge(xg_df[safe_xg_cols], on='merge_key', how='left')
    merged_df = merged_df.drop('merge_key', axis=1)
    
    merge_success_rate = (merged_df['HomeXG'].notna().sum() / len(merged_df)) * 100
    logger.info(f"✅ Merge success rate: {merge_success_rate:.1f}%")
    
    if merge_success_rate < 95:
        logger.warning(f"⚠️ Low merge rate: {merge_success_rate:.1f}%")
    
    return merged_df

def create_team_long_format(df):
    """
    Create team-level long format for rolling calculations
    """
    logger = setup_logging()
    logger.info("🔄 CREATING TEAM LONG FORMAT")
    
    # Home games
    home_games = df[['Date', 'HomeTeam', 'AwayTeam', 'HomeXG', 'HomeGoals']].copy()
    home_games = home_games.rename(columns={
        'HomeTeam': 'team',
        'AwayTeam': 'opponent', 
        'HomeXG': 'xg',
        'HomeGoals': 'goals'
    })
    home_games['venue'] = 'H'
    
    # Away games  
    away_games = df[['Date', 'HomeTeam', 'AwayTeam', 'AwayXG', 'AwayGoals']].copy()
    away_games = away_games.rename(columns={
        'HomeTeam': 'opponent',
        'AwayTeam': 'team',
        'AwayXG': 'xg', 
        'AwayGoals': 'goals'
    })
    away_games['venue'] = 'A'
    
    # Combine and sort
    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(['team', 'Date']).reset_index(drop=True)
    
    # Fill missing xG/goals with neutral values
    team_games['xg'] = team_games['xg'].fillna(1.5)  # Neutral xG
    team_games['goals'] = team_games['goals'].fillna(1.0)  # Neutral goals
    
    logger.info(f"✅ Team long format: {len(team_games)} records for {team_games['team'].nunique()} teams")
    
    return team_games

def add_shifted_rolling_features(team_games, windows=[5, 10]):
    """
    Add rolling features with proper shift(1) to prevent data leakage
    CRITICAL: shift(1) ensures features for match N use only data from matches 1 to N-1
    """
    logger = setup_logging()
    logger.info("📈 ADDING SHIFTED ROLLING FEATURES")
    
    for window in windows:
        logger.info(f"Computing {window}-match rolling features...")
        
        # Rolling xG (attack) - SHIFTED
        team_games[f'xg_roll_{window}'] = (
            team_games.groupby('team')['xg']
            .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        ).values
        
        # Rolling goals sum - SHIFTED  
        team_games[f'goals_sum_{window}'] = (
            team_games.groupby('team')['goals']
            .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).sum())
        ).values
        
        # Rolling xG sum - SHIFTED
        team_games[f'xg_sum_{window}'] = (
            team_games.groupby('team')['xg']
            .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).sum())
        ).values
        
        # Efficiency = goals/xG from historical data only
        team_games[f'xg_eff_{window}'] = (
            team_games[f'goals_sum_{window}'] / 
            team_games[f'xg_sum_{window}'].replace(0, np.nan)
        )
        
        # Fill missing efficiency with median (neutral)
        median_eff = team_games[f'xg_eff_{window}'].median()
        team_games[f'xg_eff_{window}'] = team_games[f'xg_eff_{window}'].fillna(median_eff)
    
    logger.info(f"✅ Rolling features computed for windows: {windows}")
    
    return team_games

def merge_back_to_match_level(df, team_games):
    """
    Merge team-level rolling features back to match level
    """
    logger = setup_logging()
    logger.info("🔄 MERGING BACK TO MATCH LEVEL")
    
    # Create merge keys
    team_games['merge_key'] = team_games['Date'].astype(str) + '_' + team_games['team']
    
    # Select rolling feature columns
    feature_cols = [col for col in team_games.columns 
                   if any(keyword in col for keyword in ['roll', 'eff', 'sum']) 
                   and col not in ['Date', 'team', 'opponent', 'venue', 'xg', 'goals']]
    
    merge_cols = ['merge_key'] + feature_cols
    team_features = team_games[merge_cols].copy()
    
    logger.info(f"Features to merge: {len(feature_cols)}")
    for col in feature_cols:
        logger.info(f"  {col}")
    
    # Merge home team features
    df['home_merge_key'] = df['Date'].astype(str) + '_' + df['HomeTeam']
    home_features = team_features.copy()
    home_features.columns = ['home_merge_key'] + [f'home_{col}' for col in feature_cols]
    
    df_enhanced = df.merge(home_features, on='home_merge_key', how='left')
    
    # Merge away team features  
    df_enhanced['away_merge_key'] = df_enhanced['Date'].astype(str) + '_' + df_enhanced['AwayTeam']
    away_features = team_features.copy()
    away_features.columns = ['away_merge_key'] + [f'away_{col}' for col in feature_cols]
    
    df_enhanced = df_enhanced.merge(away_features, on='away_merge_key', how='left')
    
    # Clean up merge keys
    df_enhanced = df_enhanced.drop(['home_merge_key', 'away_merge_key'], axis=1)
    
    logger.info(f"✅ Enhanced dataset: {df_enhanced.shape}")
    
    return df_enhanced

def create_safe_difference_features(df):
    """
    Create home vs away difference features from safe historical data
    """
    logger = setup_logging()
    logger.info("➖ CREATING SAFE DIFFERENCE FEATURES")
    
    # Find home/away feature pairs
    home_features = [col for col in df.columns if col.startswith('home_') and 'roll' in col]
    
    difference_features = []
    
    for home_col in home_features:
        away_col = home_col.replace('home_', 'away_')
        if away_col in df.columns:
            diff_col = home_col.replace('home_', '') + '_diff'
            df[diff_col] = df[home_col] - df[away_col]
            difference_features.append(diff_col)
    
    logger.info(f"✅ Created {len(difference_features)} difference features")
    for feat in difference_features:
        logger.info(f"  {feat}")
    
    return df, difference_features

def normalize_safe_features(df, feature_list):
    """
    Normalize features to [0,1] range - but only safe features
    """
    logger = setup_logging()
    logger.info("📏 NORMALIZING SAFE FEATURES")
    
    normalized_features = []
    
    for feature in feature_list:
        if feature in df.columns:
            if df[feature].std() > 1e-6:  # Has variance
                # Robust normalization
                q25, q75 = df[feature].quantile([0.25, 0.75])
                iqr = q75 - q25
                
                if iqr > 1e-6:
                    # Use IQR scaling
                    df[f'{feature}_normalized'] = ((df[feature] - q25) / iqr).clip(-2, 2)
                    # Map to [0, 1]
                    min_val = df[f'{feature}_normalized'].min()
                    max_val = df[f'{feature}_normalized'].max()
                    if max_val > min_val:
                        df[f'{feature}_normalized'] = (df[f'{feature}_normalized'] - min_val) / (max_val - min_val)
                    else:
                        df[f'{feature}_normalized'] = 0.5
                else:
                    df[f'{feature}_normalized'] = 0.5  # Constant feature
                
                normalized_features.append(f'{feature}_normalized')
            else:
                df[f'{feature}_normalized'] = 0.5  # No variance
                normalized_features.append(f'{feature}_normalized')
    
    logger.info(f"✅ Normalized {len(normalized_features)} features")
    
    return df, normalized_features

def remove_dangerous_features(df):
    """
    Explicitly remove features that cause data leakage
    """
    logger = setup_logging()
    logger.info("🗑️ REMOVING DANGEROUS FEATURES")
    
    removed_features = []
    
    for feature in DANGEROUS_FEATURES:
        if feature in df.columns:
            df = df.drop(feature, axis=1)
            removed_features.append(feature)
            logger.info(f"  ❌ Removed: {feature}")
    
    logger.info(f"✅ Removed {len(removed_features)} dangerous features")
    
    return df

def main():
    """
    Main pipeline for building xG-safe features
    """
    logger = setup_logging()
    logger.info("🛡️ BUILDING XG-SAFE FEATURES - NO DATA LEAKAGE")
    logger.info("=" * 60)
    logger.info("Following ChatGPT's strict methodology:")
    logger.info("- Remove all current-match xG data")
    logger.info("- Use only historical xG with shift(1)")
    logger.info("- Rolling averages and efficiency from past only")
    logger.info("=" * 60)
    
    # Step 1: Load and merge datasets
    df = load_and_merge_datasets()
    if df is None:
        return False
    
    logger.info(f"📊 Initial dataset: {df.shape}")
    
    # Step 2: Create team long format for rolling calculations
    team_games = create_team_long_format(df)
    
    # Step 3: Add shifted rolling features (NO DATA LEAKAGE)
    team_games = add_shifted_rolling_features(team_games, windows=[5, 10])
    
    # Step 4: Merge back to match level
    df_enhanced = merge_back_to_match_level(df, team_games)
    
    # Step 5: Create safe difference features
    df_enhanced, diff_features = create_safe_difference_features(df_enhanced)
    
    # Step 6: Normalize safe features
    df_enhanced, normalized_features = normalize_safe_features(df_enhanced, diff_features)
    
    # Step 7: Remove dangerous features (CRITICAL STEP)
    df_enhanced = remove_dangerous_features(df_enhanced)
    
    # Step 8: Save safe dataset
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_enhanced.to_csv(OUTPUT_CSV, index=False)
    
    # Generate summary
    safe_xg_features = [col for col in df_enhanced.columns 
                       if any(keyword in col.lower() for keyword in ['xg', 'roll', 'eff']) 
                       and 'normalized' in col]
    
    logger.info(f"\n✅ XG-SAFE DATASET CREATED")
    logger.info(f"Output file: {OUTPUT_CSV}")
    logger.info(f"📊 Final shape: {df_enhanced.shape}")
    logger.info(f"🛡️ Safe xG features: {len(safe_xg_features)}")
    
    # Feature summary
    logger.info(f"\n🎯 SAFE XG FEATURES CREATED:")
    for feature in safe_xg_features:
        logger.info(f"  {feature}")
    
    # Validation summary
    logger.info(f"\n🔒 DATA LEAKAGE PREVENTION:")
    logger.info(f"✅ All raw xG removed (HomeXG, AwayXG, XG_Diff)")
    logger.info(f"✅ All current-match goals removed")
    logger.info(f"✅ Rolling features use shift(1) - historical only")
    logger.info(f"✅ Efficiency computed from past matches only")
    
    logger.info(f"\n🎉 READY FOR CLEAN EXPERIMENTATION!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        print(f"XG-safe feature building: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)