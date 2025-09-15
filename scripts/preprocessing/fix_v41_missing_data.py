#!/usr/bin/env python3
"""
Fix v4.1 Dataset - Repair Missing Efficiency Features
Intelligently impute missing goalkeeping efficiency values to recover complete 2280 matches

Strategy: Use team-specific historical averages and league defaults for missing efficiency data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_missing_efficiency_features(df):
    """Intelligently impute missing efficiency features."""
    
    logger.info("Fixing missing efficiency features...")
    
    # Features with missing data
    efficiency_features = [
        'goalkeeping_advantage_10_normalized',
        'away_goalkeeping_efficiency_10_normalized', 
        'goalkeeping_advantage_10',
        'net_performance_advantage_10',
        'net_performance_advantage_10_normalized'
    ]
    
    for feature in efficiency_features:
        if feature in df.columns:
            missing_before = df[feature].isna().sum()
            
            if missing_before > 0:
                logger.info(f"Fixing {feature}: {missing_before} missing values")
                
                # Strategy 1: Use team-specific median for similar matches
                if 'home' in feature.lower():
                    # Home team efficiency - use home team historical median
                    for team in df['HomeTeam'].unique():
                        team_mask = (df['HomeTeam'] == team) & df[feature].notna()
                        if team_mask.sum() > 0:
                            team_median = df.loc[team_mask, feature].median()
                            fill_mask = (df['HomeTeam'] == team) & df[feature].isna()
                            df.loc[fill_mask, feature] = team_median
                
                elif 'away' in feature.lower():
                    # Away team efficiency - use away team historical median
                    for team in df['AwayTeam'].unique():
                        team_mask = (df['AwayTeam'] == team) & df[feature].notna()
                        if team_mask.sum() > 0:
                            team_median = df.loc[team_mask, feature].median()
                            fill_mask = (df['AwayTeam'] == team) & df[feature].isna()
                            df.loc[fill_mask, feature] = team_median
                
                else:
                    # General efficiency features - use match-context median
                    # Group by season and use seasonal median
                    df['Date'] = pd.to_datetime(df['Date'])
                    df['Season'] = df['Date'].dt.year
                    
                    for season in df['Season'].unique():
                        season_mask = (df['Season'] == season) & df[feature].notna()
                        if season_mask.sum() > 0:
                            season_median = df.loc[season_mask, feature].median()
                            fill_mask = (df['Season'] == season) & df[feature].isna()
                            df.loc[fill_mask, feature] = season_median
                
                # Strategy 2: Final fallback - use global median
                remaining_missing = df[feature].isna().sum()
                if remaining_missing > 0:
                    global_median = df[feature].median()
                    df.loc[df[feature].isna(), feature] = global_median
                    logger.info(f"Used global median {global_median:.3f} for {remaining_missing} remaining values")
                
                missing_after = df[feature].isna().sum()
                logger.info(f"✅ {feature}: {missing_before} → {missing_after} missing")
    
    return df

def fix_other_missing_features(df):
    """Fix other minor missing features."""
    
    # Fix away_goals_sum_5 (13 missing)
    if 'away_goals_sum_5' in df.columns:
        missing = df['away_goals_sum_5'].isna().sum()
        if missing > 0:
            logger.info(f"Fixing away_goals_sum_5: {missing} missing values")
            
            # Use team-specific median goals scored
            for team in df['AwayTeam'].unique():
                team_mask = (df['AwayTeam'] == team) & df['away_goals_sum_5'].notna()
                if team_mask.sum() > 0:
                    team_median = df.loc[team_mask, 'away_goals_sum_5'].median()
                    fill_mask = (df['AwayTeam'] == team) & df['away_goals_sum_5'].isna()
                    df.loc[fill_mask, 'away_goals_sum_5'] = team_median
            
            # Final fallback
            remaining = df['away_goals_sum_5'].isna().sum()
            if remaining > 0:
                global_median = df['away_goals_sum_5'].median()
                df.loc[df['away_goals_sum_5'].isna(), 'away_goals_sum_5'] = global_median
            
            logger.info(f"✅ away_goals_sum_5: {missing} → {df['away_goals_sum_5'].isna().sum()} missing")
    
    return df

def create_fixed_v41_dataset():
    """Create complete v4.1 dataset with all 2280 matches."""
    
    logger.info("🚀 Starting v4.1 dataset repair...")
    
    # Load current v4.1 dataset
    input_path = Path('data/processed/v41_referee_features_2025_09_07.csv')
    output_path = Path('data/processed/v41_referee_features_fixed_2025_09_07.csv')
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded v4.1 dataset: {df.shape}")
    
    # Check missing data before
    production_features = [
        'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
        'corners_diff_normalized', 'form_diff_normalized', 'home_xg_eff_10',
        'away_xg_eff_10', 'h2h_score', 'matchday_normalized', 'away_goals_sum_5',
        'goalkeeping_advantage_10_normalized', 'away_goalkeeping_efficiency_10_normalized',
        'goalkeeping_advantage_10', 'net_performance_advantage_10',
        'net_performance_advantage_10_normalized', 'fatigue_advantage', 
        'home_days_since_last_match', 'away_days_since_last_match', 
        'fixture_density_differential', 'referee_bias_index_weighted',
        'referee_home_bias_index', 'referee_disciplinary_index', 
        'referee_home_impact_score', 'referee_experience_factor'
    ]
    
    available_features = [f for f in production_features if f in df.columns]
    df_clean_before = df.dropna(subset=available_features + ['FullTimeResult'])
    logger.info(f"Clean matches before repair: {len(df_clean_before)}/{len(df)} ({len(df_clean_before)/len(df)*100:.1f}%)")
    
    # Fix missing efficiency features
    df = fix_missing_efficiency_features(df)
    
    # Fix other missing features
    df = fix_other_missing_features(df)
    
    # Check results after repair
    df_clean_after = df.dropna(subset=available_features + ['FullTimeResult'])
    logger.info(f"Clean matches after repair: {len(df_clean_after)}/{len(df)} ({len(df_clean_after)/len(df)*100:.1f}%)")
    
    # Save fixed dataset
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved fixed v4.1 dataset to {output_path}")
    
    print(f"\n📊 REPAIR SUMMARY:")
    print(f"  • Total matches: {len(df)}")
    print(f"  • Complete matches before: {len(df_clean_before)}")
    print(f"  • Complete matches after: {len(df_clean_after)}")
    print(f"  • Matches recovered: +{len(df_clean_after) - len(df_clean_before)}")
    print(f"  • Success rate: {len(df_clean_after)/len(df)*100:.1f}%")
    
    if len(df_clean_after) == len(df):
        print(f"  🎯 PERFECT: All 2280 matches recovered!")
    else:
        remaining_issues = len(df) - len(df_clean_after)
        print(f"  ⚠️ Still missing: {remaining_issues} matches need manual inspection")
    
    return df

if __name__ == "__main__":
    fixed_df = create_fixed_v41_dataset()