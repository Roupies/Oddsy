#!/usr/bin/env python3
"""
Fix All Datasets - Ensure Complete 2280 Matches
Create fixed versions of all major datasets with identical base data and no missing matches

Strategy: Start from complete raw data, rebuild each feature set with proper imputation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_complete_base_data():
    """Load complete base data (2280 matches) from raw source."""
    
    logger.info("Loading complete base data...")
    
    # Try multiple sources for complete data
    raw_paths = [
        'data/raw/premier_league_2019_2024.csv',
        'data/raw/PremierLeague.csv'
    ]
    
    for raw_path in raw_paths:
        if Path(raw_path).exists():
            df_raw = pd.read_csv(raw_path)
            if len(df_raw) >= 2280:
                logger.info(f"Using complete raw data from {raw_path}: {len(df_raw)} matches")
                # Take exactly 2280 most recent matches if more available
                df_raw = df_raw.tail(2280).reset_index(drop=True)
                return df_raw
    
    # Fallback: combine backup files
    logger.info("Combining backup files...")
    backup_files = [
        'data/raw/football_data_backup/football_data_2019_20.csv',
        'data/raw/football_data_backup/football_data_2020_21.csv', 
        'data/raw/football_data_backup/football_data_2021_22.csv',
        'data/raw/football_data_backup/football_data_2022_23.csv',
        'data/raw/football_data_backup/football_data_2023_24.csv',
        'data/raw/football_data_backup/football_data_2024_25.csv'
    ]
    
    all_data = []
    for file_path in backup_files:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            all_data.append(df)
            logger.info(f"Loaded {file_path}: {len(df)} matches")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined['Date'] = pd.to_datetime(combined['Date'])
        combined = combined.sort_values('Date').reset_index(drop=True)
        logger.info(f"Combined complete dataset: {len(combined)} matches")
        return combined
    
    raise FileNotFoundError("No complete base data found!")

def intelligent_imputation(df, feature_name, strategy='median'):
    """Intelligently impute missing values using contextual information."""
    
    missing_count = df[feature_name].isna().sum()
    if missing_count == 0:
        return df
    
    logger.info(f"Imputing {missing_count} missing values for {feature_name}")
    
    if strategy == 'team_median':
        # Use team-specific medians
        if 'home' in feature_name.lower():
            for team in df['HomeTeam'].unique():
                team_mask = (df['HomeTeam'] == team) & df[feature_name].notna()
                if team_mask.sum() > 0:
                    team_median = df.loc[team_mask, feature_name].median()
                    fill_mask = (df['HomeTeam'] == team) & df[feature_name].isna()
                    df.loc[fill_mask, feature_name] = team_median
        
        elif 'away' in feature_name.lower():
            for team in df['AwayTeam'].unique():
                team_mask = (df['AwayTeam'] == team) & df[feature_name].notna()
                if team_mask.sum() > 0:
                    team_median = df.loc[team_mask, feature_name].median()
                    fill_mask = (df['AwayTeam'] == team) & df[feature_name].isna()
                    df.loc[fill_mask, feature_name] = team_median
    
    # Final fallback - global median/mode
    remaining_missing = df[feature_name].isna().sum()
    if remaining_missing > 0:
        if df[feature_name].dtype in ['float64', 'int64']:
            fill_value = df[feature_name].median()
        else:
            fill_value = df[feature_name].mode()[0] if len(df[feature_name].mode()) > 0 else 'unknown'
        
        df.loc[df[feature_name].isna(), feature_name] = fill_value
        logger.info(f"Used {strategy} fillvalue {fill_value} for {remaining_missing} values")
    
    return df

def create_fixed_v13_dataset(base_df):
    """Create fixed v2.3 (v13) dataset with complete matches."""
    
    logger.info("Creating fixed v2.3 (v13) dataset...")
    
    # Load original v13 for feature engineering logic
    try:
        original_v13 = pd.read_csv('data/processed/v13_xg_corrected_features_latest.csv')
        v13_features = [
            'elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10',
            'away_xg_eff_10', 'shots_diff_normalized', 'corners_diff_normalized',
            'matchday_normalized', 'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
        ]
        
        # Merge base data with v13 features (matching on date/teams)
        base_df['Date'] = pd.to_datetime(base_df['Date'])
        original_v13['Date'] = pd.to_datetime(original_v13['Date'])
        
        # Merge key
        base_df['match_key'] = base_df['Date'].dt.strftime('%Y-%m-%d') + '_' + base_df['HomeTeam'] + '_' + base_df['AwayTeam']
        original_v13['match_key'] = original_v13['Date'].dt.strftime('%Y-%m-%d') + '_' + original_v13['HomeTeam'] + '_' + original_v13['AwayTeam']
        
        # Merge features
        merged = base_df.merge(
            original_v13[['match_key'] + v13_features], 
            on='match_key', 
            how='left'
        )
        
        # Impute missing features
        for feature in v13_features:
            if feature in merged.columns:
                merged = intelligent_imputation(merged, feature, 'team_median')
        
        # Clean up
        merged = merged.drop(columns=['match_key'])
        
        output_path = 'data/processed/v13_xg_corrected_features_fixed_complete.csv'
        merged.to_csv(output_path, index=False)
        
        logger.info(f"✅ Created fixed v13: {len(merged)} matches → {output_path}")
        return merged
        
    except Exception as e:
        logger.error(f"Failed to create fixed v13: {str(e)}")
        return None

def create_fixed_v31_dataset(base_df):
    """Create fixed v3.1 efficiency dataset with complete matches."""
    
    logger.info("Creating fixed v3.1 efficiency dataset...")
    
    try:
        original_v31 = pd.read_csv('data/processed/v31_efficiency_features_2025_09_06.csv')
        v31_features = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'goalkeeping_advantage_10',
            'away_goalkeeping_efficiency_10_normalized', 'goalkeeping_advantage_10_normalized',
            'net_performance_advantage_10_normalized', 'net_performance_advantage_10',
            'goalkeeping_advantage_5_normalized', 'away_xg_eff_10', 'matchday_normalized',
            'form_diff_normalized', 'h2h_score', 'away_goals_sum_5'
        ]
        
        # Merge process
        base_df['Date'] = pd.to_datetime(base_df['Date'])
        original_v31['Date'] = pd.to_datetime(original_v31['Date'])
        
        base_df['match_key'] = base_df['Date'].dt.strftime('%Y-%m-%d') + '_' + base_df['HomeTeam'] + '_' + base_df['AwayTeam']
        original_v31['match_key'] = original_v31['Date'].dt.strftime('%Y-%m-%d') + '_' + original_v31['HomeTeam'] + '_' + original_v31['AwayTeam']
        
        merged = base_df.merge(
            original_v31[['match_key'] + v31_features], 
            on='match_key', 
            how='left'
        )
        
        # Impute missing efficiency features intelligently
        for feature in v31_features:
            if feature in merged.columns:
                merged = intelligent_imputation(merged, feature, 'team_median')
        
        merged = merged.drop(columns=['match_key'])
        
        output_path = 'data/processed/v31_efficiency_features_fixed_complete.csv'
        merged.to_csv(output_path, index=False)
        
        logger.info(f"✅ Created fixed v31: {len(merged)} matches → {output_path}")
        return merged
        
    except Exception as e:
        logger.error(f"Failed to create fixed v31: {str(e)}")
        return None

def create_fixed_v40_dataset(base_df):
    """Create fixed v4.0 fatigue dataset with complete matches."""
    
    logger.info("Creating fixed v4.0 fatigue dataset...")
    
    try:
        original_v40 = pd.read_csv('data/processed/v40_fatigue_features_2025_09_07.csv')
        
        # Key v4.0 features (baseline + efficiency + fatigue)
        v40_features = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'form_diff_normalized', 'home_xg_eff_10',
            'away_xg_eff_10', 'h2h_score', 'matchday_normalized', 'away_goals_sum_5',
            'goalkeeping_advantage_10_normalized', 'away_goalkeeping_efficiency_10_normalized',
            'goalkeeping_advantage_10', 'net_performance_advantage_10',
            'net_performance_advantage_10_normalized', 'fatigue_advantage', 
            'home_days_since_last_match', 'away_days_since_last_match', 
            'fixture_density_differential'
        ]
        
        # Merge process
        base_df['Date'] = pd.to_datetime(base_df['Date'])
        original_v40['Date'] = pd.to_datetime(original_v40['Date'])
        
        base_df['match_key'] = base_df['Date'].dt.strftime('%Y-%m-%d') + '_' + base_df['HomeTeam'] + '_' + base_df['AwayTeam']
        original_v40['match_key'] = original_v40['Date'].dt.strftime('%Y-%m-%d') + '_' + original_v40['HomeTeam'] + '_' + original_v40['AwayTeam']
        
        merged = base_df.merge(
            original_v40[['match_key'] + [f for f in v40_features if f in original_v40.columns]], 
            on='match_key', 
            how='left'
        )
        
        # Impute all missing features
        available_v40_features = [f for f in v40_features if f in merged.columns]
        for feature in available_v40_features:
            merged = intelligent_imputation(merged, feature, 'team_median')
        
        merged = merged.drop(columns=['match_key'])
        
        output_path = 'data/processed/v40_fatigue_features_fixed_complete.csv'
        merged.to_csv(output_path, index=False)
        
        logger.info(f"✅ Created fixed v40: {len(merged)} matches → {output_path}")
        return merged
        
    except Exception as e:
        logger.error(f"Failed to create fixed v40: {str(e)}")
        return None

def main():
    """Main function to create all fixed datasets."""
    
    logger.info("🚀 Starting complete dataset repair for fair comparison...")
    
    # Load complete base data (2280 matches)
    base_df = load_complete_base_data()
    logger.info(f"Base complete dataset: {len(base_df)} matches")
    
    if len(base_df) != 2280:
        logger.warning(f"Expected 2280 matches, got {len(base_df)}")
    
    # Create all fixed datasets
    results = {}
    
    # v2.3 (v13)
    v13_fixed = create_fixed_v13_dataset(base_df.copy())
    if v13_fixed is not None:
        results['v13'] = len(v13_fixed)
    
    # v3.1 efficiency
    v31_fixed = create_fixed_v31_dataset(base_df.copy())
    if v31_fixed is not None:
        results['v31'] = len(v31_fixed)
    
    # v4.0 fatigue
    v40_fixed = create_fixed_v40_dataset(base_df.copy())
    if v40_fixed is not None:
        results['v40'] = len(v40_fixed)
    
    # v4.1 is already fixed
    v41_path = Path('data/processed/v41_referee_features_fixed_2025_09_07.csv')
    if v41_path.exists():
        v41_df = pd.read_csv(v41_path)
        results['v41'] = len(v41_df)
        logger.info(f"✅ v4.1 already fixed: {len(v41_df)} matches")
    
    # Summary
    print(f"\\n📊 DATASET REPAIR SUMMARY:")
    print(f"   • Base matches: {len(base_df)}")
    for version, count in results.items():
        status = "✅ COMPLETE" if count == 2280 else f"⚠️ {count} matches"
        print(f"   • {version}: {status}")
    
    if all(count == 2280 for count in results.values()):
        print(f"\\n🎯 SUCCESS: All datasets now have complete 2280 matches!")
    else:
        print(f"\\n❌ Some datasets still have missing matches")
    
    return results

if __name__ == "__main__":
    main()