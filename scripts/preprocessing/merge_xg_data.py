#!/usr/bin/env python3
"""
MERGE XG DATA WITH EXISTING DATASET
Integrate Understat xG features with current v1.5 Premier League dataset
Create enhanced dataset for v2.0 development
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

def load_datasets():
    """
    Load both existing Premier League data and scraped xG data
    """
    logger = setup_logging()
    logger.info("=== LOADING DATASETS ===")
    
    # Load existing dataset
    existing_data_path = 'data/processed/v13_complete_with_dates.csv'
    if not os.path.exists(existing_data_path):
        logger.error(f"❌ Existing dataset not found: {existing_data_path}")
        return None, None
    
    existing_df = pd.read_csv(existing_data_path)
    existing_df['Date'] = pd.to_datetime(existing_df['Date'])
    
    logger.info(f"✅ Existing dataset: {existing_df.shape}")
    logger.info(f"   Date range: {existing_df['Date'].min()} to {existing_df['Date'].max()}")
    
    # Load xG data (find most recent file)
    xg_data_dir = 'data/external'
    if not os.path.exists(xg_data_dir):
        logger.error(f"❌ xG data directory not found: {xg_data_dir}")
        return None, None
    
    xg_files = [f for f in os.listdir(xg_data_dir) if f.startswith('understat_xg_data')]
    if not xg_files:
        logger.error("❌ No xG data files found")
        return None, None
    
    # Use most recent xG file
    latest_xg_file = sorted(xg_files)[-1]
    xg_data_path = os.path.join(xg_data_dir, latest_xg_file)
    
    xg_df = pd.read_csv(xg_data_path)
    xg_df['Date'] = pd.to_datetime(xg_df['Date'])
    
    logger.info(f"✅ xG dataset: {xg_df.shape}")
    logger.info(f"   File: {latest_xg_file}")
    logger.info(f"   Date range: {xg_df['Date'].min()} to {xg_df['Date'].max()}")
    
    return existing_df, xg_df

def normalize_team_names(df, team_columns):
    """
    Normalize team names for consistent matching
    """
    logger = setup_logging()
    logger.info("=== NORMALIZING TEAM NAMES ===")
    
    # Common team name variations
    name_mappings = {
        # Understat variations -> Our standard names
        'Manchester United': 'Man United',
        'Manchester City': 'Man City',
        'Tottenham': 'Tottenham',  # May need adjustment
        'Leicester': 'Leicester',
        'Wolverhampton Wanderers': 'Wolves',
        'Brighton': 'Brighton',
        'Newcastle United': 'Newcastle',
        'Norwich': 'Norwich',
        'Sheffield United': 'Sheffield United',
        'West Ham': 'West Ham',
        'Nottingham Forest': "Nott'm Forest"  # Common variation
    }
    
    df_normalized = df.copy()
    
    for col in team_columns:
        for understat_name, standard_name in name_mappings.items():
            df_normalized[col] = df_normalized[col].replace(understat_name, standard_name)
    
    logger.info(f"✅ Team names normalized for columns: {team_columns}")
    return df_normalized

def analyze_match_overlap(existing_df, xg_df):
    """
    Analyze overlap between datasets for matching strategy
    """
    logger = setup_logging()
    logger.info("=== ANALYZING DATASET OVERLAP ===")
    
    # Create match signatures for both datasets
    existing_signatures = set()
    for _, row in existing_df.iterrows():
        signature = f"{row['Date'].strftime('%Y-%m-%d')}_{row['HomeTeam']}_{row['AwayTeam']}"
        existing_signatures.add(signature)
    
    xg_signatures = set()
    for _, row in xg_df.iterrows():
        signature = f"{row['Date'].strftime('%Y-%m-%d')}_{row['HomeTeam']}_{row['AwayTeam']}"
        xg_signatures.add(signature)
    
    # Analyze overlap
    common_matches = existing_signatures.intersection(xg_signatures)
    existing_only = existing_signatures - xg_signatures
    xg_only = xg_signatures - common_matches
    
    logger.info(f"📊 OVERLAP ANALYSIS:")
    logger.info(f"   Existing dataset: {len(existing_signatures)} matches")
    logger.info(f"   xG dataset: {len(xg_signatures)} matches")
    logger.info(f"   Common matches: {len(common_matches)}")
    logger.info(f"   Existing only: {len(existing_only)}")
    logger.info(f"   xG only: {len(xg_only)}")
    
    match_rate = len(common_matches) / len(existing_signatures) * 100
    logger.info(f"   Match rate: {match_rate:.1f}%")
    
    if len(existing_only) > 0:
        logger.warning(f"⚠️  {len(existing_only)} matches in existing dataset not found in xG data")
        # Show first few examples
        examples = list(existing_only)[:5]
        for example in examples:
            logger.info(f"     Example: {example}")
    
    if len(xg_only) > 0:
        logger.info(f"ℹ️  {len(xg_only)} extra matches in xG dataset")
    
    return match_rate >= 95.0  # Require 95%+ match rate

def merge_datasets(existing_df, xg_df):
    """
    Merge datasets on Date, HomeTeam, AwayTeam
    """
    logger = setup_logging()
    logger.info("=== MERGING DATASETS ===")
    
    # Normalize team names in both datasets
    existing_normalized = normalize_team_names(existing_df, ['HomeTeam', 'AwayTeam'])
    xg_normalized = normalize_team_names(xg_df, ['HomeTeam', 'AwayTeam'])
    
    # Create merge keys
    existing_normalized['merge_key'] = (
        existing_normalized['Date'].dt.strftime('%Y-%m-%d') + '_' +
        existing_normalized['HomeTeam'] + '_' + 
        existing_normalized['AwayTeam']
    )
    
    xg_normalized['merge_key'] = (
        xg_normalized['Date'].dt.strftime('%Y-%m-%d') + '_' +
        xg_normalized['HomeTeam'] + '_' +
        xg_normalized['AwayTeam']
    )
    
    # Select xG columns to merge
    xg_columns = [
        'merge_key', 'HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG',
        'Home_GoalsVsXG', 'Away_GoalsVsXG', 'UnderstatID'
    ]
    
    xg_to_merge = xg_normalized[xg_columns].copy()
    
    # Perform merge
    merged_df = existing_normalized.merge(
        xg_to_merge, 
        on='merge_key', 
        how='left'
    )
    
    # Drop merge key
    merged_df = merged_df.drop('merge_key', axis=1)
    
    # Check merge success
    xg_missing = merged_df['HomeXG'].isnull().sum()
    merge_success_rate = (len(merged_df) - xg_missing) / len(merged_df) * 100
    
    logger.info(f"📊 MERGE RESULTS:")
    logger.info(f"   Merged dataset shape: {merged_df.shape}")
    logger.info(f"   Matches with xG data: {len(merged_df) - xg_missing}/{len(merged_df)}")
    logger.info(f"   Merge success rate: {merge_success_rate:.1f}%")
    
    if xg_missing > 0:
        logger.warning(f"⚠️  {xg_missing} matches missing xG data")
    
    return merged_df, merge_success_rate >= 95.0

def engineer_xg_features(df):
    """
    Engineer advanced xG-based features
    """
    logger = setup_logging()
    logger.info("=== ENGINEERING xG FEATURES ===")
    
    # Sort by date for rolling calculations
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Fill missing xG values with neutral defaults
    xg_columns = ['HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG', 'Home_GoalsVsXG', 'Away_GoalsVsXG']
    for col in xg_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0 if col in ['XG_Diff'] else 1.5)  # Neutral values
    
    # Calculate rolling xG features for each team
    teams = df['HomeTeam'].unique()
    
    logger.info(f"Calculating rolling xG features for {len(teams)} teams...")
    
    # Initialize new feature columns
    df['xg_form_home'] = 0.0
    df['xg_form_away'] = 0.0
    df['xga_form_home'] = 0.0  # Expected goals against
    df['xga_form_away'] = 0.0
    df['xg_efficiency_home'] = 1.0
    df['xg_efficiency_away'] = 1.0
    
    # Rolling window
    window_size = 5  # Last 5 matches
    
    for team in teams:
        # Get all matches for this team (home and away)
        team_matches = df[
            (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
        ].copy()
        
        if len(team_matches) == 0:
            continue
        
        # Calculate rolling metrics
        team_xg_for = []
        team_xg_against = []
        team_goals_for = []
        team_goals_against = []
        
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                # Team playing at home
                xg_for = match['HomeXG']
                xg_against = match['AwayXG'] 
                goals_for = match.get('HomeGoals', 0)
                goals_against = match.get('AwayGoals', 0)
            else:
                # Team playing away
                xg_for = match['AwayXG']
                xg_against = match['HomeXG']
                goals_for = match.get('AwayGoals', 0)
                goals_against = match.get('HomeGoals', 0)
            
            team_xg_for.append(xg_for)
            team_xg_against.append(xg_against)
            team_goals_for.append(goals_for)
            team_goals_against.append(goals_against)
        
        # Calculate rolling averages
        team_matches['rolling_xg_for'] = pd.Series(team_xg_for).rolling(window=window_size, min_periods=1).mean()
        team_matches['rolling_xg_against'] = pd.Series(team_xg_against).rolling(window=window_size, min_periods=1).mean()
        team_matches['rolling_goals_for'] = pd.Series(team_goals_for).rolling(window=window_size, min_periods=1).mean()
        
        # Update main dataframe
        for idx, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                df.loc[df.index == idx, 'xg_form_home'] = match['rolling_xg_for']
                df.loc[df.index == idx, 'xga_form_home'] = match['rolling_xg_against']
                if match['rolling_xg_for'] > 0:
                    df.loc[df.index == idx, 'xg_efficiency_home'] = match['rolling_goals_for'] / match['rolling_xg_for']
            else:
                df.loc[df.index == idx, 'xg_form_away'] = match['rolling_xg_for']
                df.loc[df.index == idx, 'xga_form_away'] = match['rolling_xg_against']
                if match['rolling_xg_for'] > 0:
                    df.loc[df.index == idx, 'xg_efficiency_away'] = match['rolling_goals_for'] / match['rolling_xg_for']
    
    # Create normalized difference features
    df['xg_form_diff'] = df['xg_form_home'] - df['xg_form_away']
    df['xga_form_diff'] = df['xga_form_away'] - df['xga_form_home']  # Lower xGA is better
    df['xg_efficiency_diff'] = df['xg_efficiency_home'] - df['xg_efficiency_away']
    
    # Normalize to [0, 1] range
    for feature in ['xg_form_diff', 'xga_form_diff', 'xg_efficiency_diff']:
        min_val = df[feature].min()
        max_val = df[feature].max()
        if max_val > min_val:
            df[f'{feature}_normalized'] = (df[feature] - min_val) / (max_val - min_val)
        else:
            df[f'{feature}_normalized'] = 0.5
    
    logger.info(f"✅ Engineered xG features:")
    new_features = [
        'HomeXG', 'AwayXG', 'XG_Diff', 'Total_XG',
        'xg_form_diff_normalized', 'xga_form_diff_normalized', 'xg_efficiency_diff_normalized'
    ]
    for feature in new_features:
        if feature in df.columns:
            logger.info(f"   {feature}: [{df[feature].min():.3f}, {df[feature].max():.3f}]")
    
    return df

def save_enhanced_dataset(df):
    """
    Save the enhanced dataset with xG features
    """
    logger = setup_logging()
    logger.info("=== SAVING ENHANCED DATASET ===")
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    filename = f"premier_league_xg_enhanced_{timestamp}.csv"
    filepath = f"data/processed/{filename}"
    
    # Ensure directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    # Save
    df.to_csv(filepath, index=False)
    
    logger.info(f"✅ Enhanced dataset saved: {filepath}")
    logger.info(f"📊 Shape: {df.shape}")
    logger.info(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Feature summary
    xg_features = [col for col in df.columns if 'xg' in col.lower() or 'XG' in col]
    logger.info(f"🎯 xG Features ({len(xg_features)}): {xg_features}")
    
    return filepath

def main():
    """
    Main pipeline for merging xG data with existing dataset
    """
    logger = setup_logging()
    logger.info("🔗 XG DATA INTEGRATION PIPELINE")
    logger.info("=" * 60)
    logger.info("Merging Understat xG data with existing Premier League dataset")
    logger.info("=" * 60)
    
    # Load datasets
    existing_df, xg_df = load_datasets()
    if existing_df is None or xg_df is None:
        return False
    
    # Analyze overlap
    overlap_good = analyze_match_overlap(existing_df, xg_df)
    if not overlap_good:
        logger.warning("⚠️ Low overlap rate - proceed with caution")
    
    # Merge datasets
    merged_df, merge_success = merge_datasets(existing_df, xg_df)
    if not merge_success:
        logger.error("❌ Merge failed - insufficient match rate")
        return False
    
    # Engineer xG features
    enhanced_df = engineer_xg_features(merged_df)
    
    # Save enhanced dataset
    saved_file = save_enhanced_dataset(enhanced_df)
    
    logger.info(f"\n🎉 XG INTEGRATION SUCCESS")
    logger.info(f"Enhanced dataset: {saved_file}")
    logger.info(f"Ready for v2.0 model training")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        print(f"xG integration: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)