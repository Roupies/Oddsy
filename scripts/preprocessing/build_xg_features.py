#!/usr/bin/env python3
"""
BUILD XG FEATURES v2.1 - SMART TEMPORAL XG ENGINEERING
Following ChatGPT's blueprint for proper xG feature engineering:
- Rolling averages with shift(1) to prevent data leakage
- Efficiency metrics from historical data only
- Momentum and trend analysis
- H2H and contextual interactions
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

def rolling_stats_grouped(df, team_col='team', date_col='Date', value_col='xg', window=5):
    """
    Return rolling mean per team at match level aligned so that match t feature uses past matches only.
    Critical: uses shift(1) to prevent data leakage
    """
    df = df.sort_values([team_col, date_col]).reset_index(drop=True)
    
    # Compute rolling mean per team, but shift so current match not included
    rolled_values = []
    
    for team in df[team_col].unique():
        team_mask = df[team_col] == team
        team_data = df[team_mask][value_col]
        
        # Apply shift(1) then rolling mean
        team_rolled = team_data.shift(1).rolling(window, min_periods=1).mean()
        rolled_values.extend(team_rolled.tolist())
    
    # Sort df same way and assign
    df = df.sort_values([team_col, date_col]).reset_index(drop=True)
    df[f'{value_col}_rolling_{window}'] = rolled_values
    
    return df

def rolling_sum_grouped(df, team_col='team', date_col='Date', value_col='goals', window=5):
    """
    Return rolling sum per team with shift(1) for efficiency calculations
    """
    df = df.sort_values([team_col, date_col]).reset_index(drop=True)
    
    # Compute rolling sum per team, but shift so current match not included
    rolled_values = []
    
    for team in df[team_col].unique():
        team_mask = df[team_col] == team
        team_data = df[team_mask][value_col]
        
        # Apply shift(1) then rolling sum
        team_rolled = team_data.shift(1).rolling(window, min_periods=1).sum()
        rolled_values.extend(team_rolled.tolist())
    
    # Sort df same way and assign
    df = df.sort_values([team_col, date_col]).reset_index(drop=True)
    df[f'{value_col}_sum_{window}'] = rolled_values
    
    return df

def build_team_level_features(matches_df):
    """
    Build rolling features at team level, then merge back to match level
    """
    logger = setup_logging()
    logger.info("🔧 BUILDING TEAM-LEVEL ROLLING FEATURES")
    
    # Ensure we have required columns
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'HomeXG', 'AwayXG']
    missing_cols = [col for col in required_cols if col not in matches_df.columns]
    if missing_cols:
        logger.error(f"❌ Missing columns: {missing_cols}")
        return None
    
    # Create team-level long format
    home_games = matches_df[['Date', 'HomeTeam', 'AwayTeam', 'HomeXG', 'AwayXG']].copy()
    home_games['team'] = home_games['HomeTeam']
    home_games['opponent'] = home_games['AwayTeam']
    home_games['xg'] = home_games['HomeXG']
    home_games['xga'] = home_games['AwayXG']  # Goals against = opponent's xG
    home_games['venue'] = 'H'
    
    away_games = matches_df[['Date', 'HomeTeam', 'AwayTeam', 'HomeXG', 'AwayXG']].copy()
    away_games['team'] = away_games['AwayTeam']
    away_games['opponent'] = away_games['HomeTeam']
    away_games['xg'] = away_games['AwayXG']
    away_games['xga'] = away_games['HomeXG']  # Goals against = opponent's xG
    away_games['venue'] = 'A'
    
    # Combine all team games
    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(['team', 'Date']).reset_index(drop=True)
    
    logger.info(f"📊 Team games created: {len(team_games)} records for {team_games['team'].nunique()} teams")
    
    # Add goals if available (for efficiency calculations)
    if 'HomeGoals' in matches_df.columns and 'AwayGoals' in matches_df.columns:
        logger.info("✅ Goals data available - will compute efficiency metrics")
        # Add goals to home games
        home_games['goals'] = matches_df['HomeGoals']
        away_games['goals'] = matches_df['AwayGoals']
        team_games = pd.concat([
            home_games[['Date', 'team', 'opponent', 'xg', 'xga', 'venue', 'goals']],
            away_games[['Date', 'team', 'opponent', 'xg', 'xga', 'venue', 'goals']]
        ], ignore_index=True)
        team_games = team_games.sort_values(['team', 'Date']).reset_index(drop=True)
        has_goals = True
    else:
        logger.warning("⚠️ No goals data - efficiency metrics will be skipped")
        has_goals = False
    
    # Build rolling features for different windows
    windows = [5, 10]
    
    for window in windows:
        logger.info(f"Computing {window}-match rolling features...")
        
        # Rolling xG (attack)
        team_games = rolling_stats_grouped(team_games, 'team', 'Date', 'xg', window)
        
        # Rolling xGA (defense)
        team_games = rolling_stats_grouped(team_games, 'team', 'Date', 'xga', window)
        
        if has_goals:
            # Rolling goals and efficiency (with shift to prevent leakage)
            team_games = rolling_sum_grouped(team_games, 'team', 'Date', 'goals', window)
            team_games = rolling_sum_grouped(team_games, 'team', 'Date', 'xg', window)
            
            # Efficiency = goals / xG (avoid division by zero)
            team_games[f'xg_eff_{window}'] = (
                team_games[f'goals_sum_{window}'] / 
                np.maximum(team_games[f'xg_sum_{window}'], 1e-6)
            )
    
    # Momentum features (recent vs previous period)
    if len(windows) >= 2:
        short_window, long_window = windows[0], windows[1]
        
        # xG momentum = recent form - previous form
        team_games[f'xg_momentum'] = (
            team_games[f'xg_rolling_{short_window}'] - 
            team_games[f'xg_rolling_{short_window}'].groupby(team_games['team']).shift(short_window)
        )
        
        # Defensive momentum
        team_games[f'xga_momentum'] = (
            team_games[f'xga_rolling_{short_window}'] - 
            team_games[f'xga_rolling_{short_window}'].groupby(team_games['team']).shift(short_window)
        )
    
    logger.info(f"✅ Team-level features built: {len([col for col in team_games.columns if 'rolling' in col or 'momentum' in col or 'eff' in col])} new features")
    
    return team_games

def merge_team_features_to_matches(matches_df, team_games):
    """
    Merge team-level rolling features back to match-level dataset
    """
    logger = setup_logging()
    logger.info("🔄 MERGING TEAM FEATURES TO MATCH LEVEL")
    
    # Create merge keys
    team_games['merge_key'] = team_games['Date'].astype(str) + '_' + team_games['team']
    
    # Prepare feature columns to merge
    feature_cols = [col for col in team_games.columns 
                   if any(keyword in col for keyword in ['rolling', 'momentum', 'eff', 'sum']) 
                   and col not in ['Date', 'team', 'opponent', 'venue', 'xg', 'xga', 'goals']]
    
    merge_cols = ['merge_key'] + feature_cols
    team_features = team_games[merge_cols].copy()
    
    logger.info(f"Features to merge: {len(feature_cols)} features")
    for col in feature_cols[:5]:  # Show first 5
        logger.info(f"  {col}")
    if len(feature_cols) > 5:
        logger.info(f"  ... and {len(feature_cols) - 5} more")
    
    # Merge home team features
    matches_df['home_merge_key'] = matches_df['Date'].astype(str) + '_' + matches_df['HomeTeam']
    home_features = team_features.copy()
    home_features.columns = ['home_merge_key'] + [f'home_{col}' for col in feature_cols]
    
    matches_enhanced = matches_df.merge(home_features, on='home_merge_key', how='left')
    
    # Merge away team features
    matches_enhanced['away_merge_key'] = matches_enhanced['Date'].astype(str) + '_' + matches_enhanced['AwayTeam']
    away_features = team_features.copy()
    away_features.columns = ['away_merge_key'] + [f'away_{col}' for col in feature_cols]
    
    matches_enhanced = matches_enhanced.merge(away_features, on='away_merge_key', how='left')
    
    # Clean up merge keys
    matches_enhanced = matches_enhanced.drop(['home_merge_key', 'away_merge_key'], axis=1)
    
    logger.info(f"✅ Match-level dataset enhanced: {matches_enhanced.shape}")
    
    return matches_enhanced

def create_difference_features(df):
    """
    Create home vs away difference features from team-level features
    """
    logger = setup_logging()
    logger.info("➖ CREATING DIFFERENCE FEATURES")
    
    # Find paired home/away features
    home_features = [col for col in df.columns if col.startswith('home_') and any(keyword in col for keyword in ['rolling', 'momentum', 'eff'])]
    
    difference_features = []
    
    for home_col in home_features:
        away_col = home_col.replace('home_', 'away_')
        if away_col in df.columns:
            diff_col = home_col.replace('home_', '') + '_diff'
            df[diff_col] = df[home_col] - df[away_col]
            difference_features.append(diff_col)
    
    logger.info(f"✅ Created {len(difference_features)} difference features")
    for feat in difference_features[:5]:
        logger.info(f"  {feat}")
    if len(difference_features) > 5:
        logger.info(f"  ... and {len(difference_features) - 5} more")
    
    return df, difference_features

def create_interaction_features(df):
    """
    Create contextual interaction features
    """
    logger = setup_logging()
    logger.info("🔗 CREATING INTERACTION FEATURES")
    
    interaction_features = []
    
    # Core interactions if base features exist
    if 'elo_diff_normalized' in df.columns and 'xg_rolling_5_diff' in df.columns:
        df['elo_xg_interaction'] = df['elo_diff_normalized'] * df['xg_rolling_5_diff']
        interaction_features.append('elo_xg_interaction')
    
    if 'market_entropy_norm' in df.columns and 'xg_rolling_5_diff' in df.columns:
        df['market_xg_interaction'] = df['market_entropy_norm'] * df['xg_rolling_5_diff']
        interaction_features.append('market_xg_interaction')
    
    # Form-based interactions
    if 'form_diff_normalized' in df.columns and 'xg_eff_5_diff' in df.columns:
        df['form_efficiency_interaction'] = df['form_diff_normalized'] * df['xg_eff_5_diff']
        interaction_features.append('form_efficiency_interaction')
    
    logger.info(f"✅ Created {len(interaction_features)} interaction features")
    for feat in interaction_features:
        logger.info(f"  {feat}")
    
    return df, interaction_features

def normalize_xg_features(df, feature_list):
    """
    Normalize xG features to [0, 1] range using robust scaling
    """
    logger = setup_logging()
    logger.info("📏 NORMALIZING XG FEATURES")
    
    from sklearn.preprocessing import RobustScaler
    
    normalized_features = []
    scaler = RobustScaler()
    
    for feature in feature_list:
        if feature in df.columns:
            # Only normalize if feature has variance
            if df[feature].std() > 1e-6:
                # Robust scaling (less sensitive to outliers)
                scaled_values = scaler.fit_transform(df[[feature]])
                
                # Map to [0, 1] range
                min_val = scaled_values.min()
                max_val = scaled_values.max()
                if max_val > min_val:
                    normalized_values = (scaled_values - min_val) / (max_val - min_val)
                    df[f'{feature}_normalized'] = normalized_values.flatten()
                    normalized_features.append(f'{feature}_normalized')
                else:
                    # Constant feature - set to 0.5 (neutral)
                    df[f'{feature}_normalized'] = 0.5
                    normalized_features.append(f'{feature}_normalized')
            else:
                # No variance - set to neutral
                df[f'{feature}_normalized'] = 0.5
                normalized_features.append(f'{feature}_normalized')
    
    logger.info(f"✅ Normalized {len(normalized_features)} features")
    
    return df, normalized_features

def validate_no_data_leakage(df):
    """
    Validate that no features use future information
    """
    logger = setup_logging()
    logger.info("🛡️ VALIDATING NO DATA LEAKAGE")
    
    # Check for suspicious correlations with target
    if 'FullTimeResult' in df.columns:
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        y = df['FullTimeResult'].map(target_mapping)
        
        suspicious_features = []
        xg_features = [col for col in df.columns if any(keyword in col.lower() for keyword in ['xg', 'rolling', 'momentum', 'eff'])]
        
        for feature in xg_features:
            if df[feature].dtype in ['float64', 'int64']:
                correlation = abs(df[feature].corr(y))
                if correlation > 0.4:  # Suspiciously high correlation
                    suspicious_features.append((feature, correlation))
        
        if suspicious_features:
            logger.warning("⚠️ SUSPICIOUS FEATURES (high correlation with target):")
            for feature, corr in suspicious_features:
                logger.warning(f"  {feature}: {corr:.3f}")
        else:
            logger.info("✅ No suspicious correlations detected")
    
    # Check for NaN values in critical periods
    early_matches = df.head(100)  # First 100 matches
    xg_features = [col for col in df.columns if 'rolling' in col]
    
    for feature in xg_features:
        nan_count = early_matches[feature].isnull().sum()
        if nan_count > 80:  # Most early matches shouldn't have rolling data
            logger.info(f"✅ {feature}: {nan_count}/100 NaN in early matches (expected)")
        elif nan_count < 20:
            logger.warning(f"⚠️ {feature}: Only {nan_count}/100 NaN in early matches (suspicious)")

def main():
    """
    Main pipeline for building smart xG features
    """
    logger = setup_logging()
    logger.info("🚀 XG FEATURE ENGINEERING v2.1 - SMART TEMPORAL APPROACH")
    logger.info("=" * 60)
    logger.info("Following ChatGPT blueprint:")
    logger.info("- Rolling averages with temporal safety (shift)")
    logger.info("- Efficiency from historical data only") 
    logger.info("- Momentum and contextual interactions")
    logger.info("=" * 60)
    
    # Load base dataset (v13 with traditional features) and merge with xG data from external
    base_files = [f for f in os.listdir('data/processed') if 'v13_complete_with_dates' in f]
    if not base_files:
        logger.error("❌ No base dataset (v13_complete_with_dates.csv) found")
        return False
    
    base_df = pd.read_csv('data/processed/v13_complete_with_dates.csv')
    base_df['Date'] = pd.to_datetime(base_df['Date'])
    
    # Load original xG data with goals
    xg_files = [f for f in os.listdir('data/external') if f.startswith('understat_xg_data')]
    if not xg_files:
        logger.error("❌ No xG data found in data/external")
        return False
    
    latest_xg_file = sorted(xg_files)[-1]
    xg_df = pd.read_csv(f"data/external/{latest_xg_file}")
    xg_df['Date'] = pd.to_datetime(xg_df['Date'])
    
    logger.info(f"📊 Base dataset: {base_df.shape}")
    logger.info(f"📊 XG dataset: {xg_df.shape}")
    
    # Merge base dataset with xG data (including HomeGoals, AwayGoals)
    # Create merge keys
    base_df['merge_key'] = base_df['Date'].dt.strftime('%Y-%m-%d') + '_' + base_df['HomeTeam'] + '_' + base_df['AwayTeam']
    xg_df['merge_key'] = xg_df['Date'].dt.strftime('%Y-%m-%d') + '_' + xg_df['HomeTeam'] + '_' + xg_df['AwayTeam']
    
    # Select xG columns to merge (excluding data leakage features)
    xg_cols_to_merge = [
        'merge_key', 'HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG', 'HomeGoals', 'AwayGoals', 'UnderstatID'
    ]
    
    # Merge
    df = base_df.merge(xg_df[xg_cols_to_merge], on='merge_key', how='left')
    df = df.drop('merge_key', axis=1)
    
    merge_success_rate = (df['HomeXG'].notna().sum() / len(df)) * 100
    logger.info(f"✅ Merge success rate: {merge_success_rate:.1f}%")
    
    if merge_success_rate < 95:
        logger.warning(f"⚠️ Low merge rate - some matches missing xG data")
    
    # Fill missing xG values with neutral defaults
    df['HomeXG'] = df['HomeXG'].fillna(1.5)
    df['AwayXG'] = df['AwayXG'].fillna(1.5)
    df['XG_Diff'] = df['XG_Diff'].fillna(0.0)
    df['Total_XG'] = df['Total_XG'].fillna(3.0)
    df['HomeGoals'] = df['HomeGoals'].fillna(1.0)  # Fill missing goals too
    df['AwayGoals'] = df['AwayGoals'].fillna(1.0)
    
    logger.info(f"📊 Input dataset: {df.shape}")
    logger.info(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Step 1: Build team-level rolling features
    team_games = build_team_level_features(df)
    if team_games is None:
        return False
    
    # Step 2: Merge back to match level
    df_enhanced = merge_team_features_to_matches(df, team_games)
    
    # Step 3: Create difference features
    df_enhanced, diff_features = create_difference_features(df_enhanced)
    
    # Step 4: Create interaction features
    df_enhanced, interaction_features = create_interaction_features(df_enhanced)
    
    # Step 5: Normalize new features
    all_new_features = diff_features + interaction_features
    df_enhanced, normalized_features = normalize_xg_features(df_enhanced, all_new_features)
    
    # Step 6: Validate no data leakage
    validate_no_data_leakage(df_enhanced)
    
    # Step 7: Save enhanced dataset
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    output_file = f"data/processed/premier_league_xg_v21_{timestamp}.csv"
    df_enhanced.to_csv(output_file, index=False)
    
    # Generate feature summary
    all_features = [col for col in df_enhanced.columns if any(keyword in col for keyword in ['xg', 'rolling', 'momentum', 'eff', 'interaction'])]
    
    logger.info(f"\n✅ XG v2.1 FEATURE ENGINEERING COMPLETE")
    logger.info(f"Enhanced dataset: {output_file}")
    logger.info(f"📊 Final shape: {df_enhanced.shape}")
    logger.info(f"🎯 Total xG features: {len(all_features)}")
    
    # Save feature list for experiments
    feature_config = {
        'timestamp': timestamp,
        'dataset_file': output_file,
        'total_features': len(all_features),
        'feature_categories': {
            'rolling_features': [f for f in all_features if 'rolling' in f],
            'momentum_features': [f for f in all_features if 'momentum' in f],
            'efficiency_features': [f for f in all_features if 'eff' in f],
            'interaction_features': interaction_features,
            'difference_features': diff_features,
            'normalized_features': normalized_features
        }
    }
    
    import json
    config_file = f"config/xg_v21_features_{timestamp}.json"
    os.makedirs('config', exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(feature_config, f, indent=2)
    
    logger.info(f"📋 Feature config saved: {config_file}")
    logger.info(f"\n🎉 Ready for progressive experimentation!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        print(f"XG v2.1 feature engineering: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)