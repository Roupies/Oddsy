#!/usr/bin/env python3
"""
v3.1 Efficiency Features - The "Moneyball" Approach
Instead of asking "Who creates most chances?", ask "Who overperforms their chances?"
Focus: finishing_efficiency and goalkeeping_efficiency over rolling windows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_efficiency_features(df):
    """Calculate xG efficiency features that measure over/under-performance."""
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize efficiency columns
    efficiency_columns = [
        'home_finishing_efficiency_5', 'home_finishing_efficiency_10',
        'away_finishing_efficiency_5', 'away_finishing_efficiency_10',
        'home_goalkeeping_efficiency_5', 'home_goalkeeping_efficiency_10', 
        'away_goalkeeping_efficiency_5', 'away_goalkeeping_efficiency_10',
        'home_net_performance_factor_5', 'home_net_performance_factor_10',
        'away_net_performance_factor_5', 'away_net_performance_factor_10'
    ]
    
    for col in efficiency_columns:
        df[col] = np.nan
    
    # Track team performance for rolling calculations
    team_performance = {}  # team -> list of (goals_for, xg_for, goals_against, xg_against)
    
    logger.info("Calculating xG efficiency features...")
    
    for i, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Get actual goals and xG (assuming we have these columns)
        # If exact column names differ, adjust accordingly
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            goals_home = row['FTHG'] 
            goals_away = row['FTAG']
        else:
            # Fallback - try to extract from result or skip
            continue
            
        if 'home_xg_sum_5' in df.columns:
            # Use existing xG data if available
            xg_home = row.get('home_xg_sum_5', 0) / 5  # Convert sum to average per match
            xg_away = row.get('away_xg_sum_5', 0) / 5
        else:
            # Skip if no xG data
            continue
        
        # Calculate efficiencies for teams (only if they have history)
        for team, team_type in [(home_team, 'home'), (away_team, 'away')]:
            if team in team_performance:
                history = team_performance[team]
                
                for window in [5, 10]:
                    if len(history) >= min(window, 3):  # Need at least 3 matches
                        recent_history = history[-window:]
                        
                        # Calculate finishing efficiency
                        total_goals_for = sum(h[0] for h in recent_history)
                        total_xg_for = sum(h[1] for h in recent_history)
                        
                        if total_xg_for > 0:
                            finishing_eff = total_goals_for / total_xg_for
                        else:
                            finishing_eff = 1.0  # Neutral if no xG data
                        
                        # Calculate goalkeeping efficiency (goals against / xG against)
                        total_goals_against = sum(h[2] for h in recent_history)
                        total_xg_against = sum(h[3] for h in recent_history)
                        
                        if total_xg_against > 0:
                            goalkeeping_eff = total_goals_against / total_xg_against
                        else:
                            goalkeeping_eff = 1.0
                        
                        # Net performance factor (finishing good - goalkeeping bad)
                        net_performance = finishing_eff - goalkeeping_eff
                        
                        # Store in dataframe
                        df.loc[i, f'{team_type}_finishing_efficiency_{window}'] = finishing_eff
                        df.loc[i, f'{team_type}_goalkeeping_efficiency_{window}'] = goalkeeping_eff
                        df.loc[i, f'{team_type}_net_performance_factor_{window}'] = net_performance
        
        # Update team histories after calculating features for this match
        # Home team performance (goals_for, xg_for, goals_against, xg_against)
        if home_team not in team_performance:
            team_performance[home_team] = []
        team_performance[home_team].append((goals_home, xg_home, goals_away, xg_away))
        
        # Away team performance  
        if away_team not in team_performance:
            team_performance[away_team] = []
        team_performance[away_team].append((goals_away, xg_away, goals_home, xg_home))
        
        # Keep only last 15 matches for memory efficiency
        for team in [home_team, away_team]:
            if len(team_performance[team]) > 15:
                team_performance[team] = team_performance[team][-15:]
    
    logger.info("xG efficiency features calculation complete")
    return df

def create_efficiency_derived_features(df):
    """Create derived features from efficiency metrics."""
    
    # Efficiency advantages (home vs away)
    for window in [5, 10]:
        # Finishing advantage
        home_finishing_col = f'home_finishing_efficiency_{window}'
        away_finishing_col = f'away_finishing_efficiency_{window}'
        
        if home_finishing_col in df.columns and away_finishing_col in df.columns:
            df[f'finishing_advantage_{window}'] = df[home_finishing_col] - df[away_finishing_col]
        
        # Goalkeeping advantage (lower is better for goalkeeping)
        home_gk_col = f'home_goalkeeping_efficiency_{window}'
        away_gk_col = f'away_goalkeeping_efficiency_{window}'
        
        if home_gk_col in df.columns and away_gk_col in df.columns:
            df[f'goalkeeping_advantage_{window}'] = df[away_gk_col] - df[home_gk_col]
        
        # Net performance advantage
        home_net_col = f'home_net_performance_factor_{window}'
        away_net_col = f'away_net_performance_factor_{window}'
        
        if home_net_col in df.columns and away_net_col in df.columns:
            df[f'net_performance_advantage_{window}'] = df[home_net_col] - df[away_net_col]
    
    # Create "hot streak" indicators (very high efficiency)
    for team_type in ['home', 'away']:
        for window in [5, 10]:
            finishing_col = f'{team_type}_finishing_efficiency_{window}'
            if finishing_col in df.columns:
                # Hot finishing streak (>1.3x expected goals)
                df[f'{team_type}_hot_finishing_{window}'] = (df[finishing_col] > 1.3).astype(int)
                
                # Cold finishing streak (<0.7x expected goals) 
                df[f'{team_type}_cold_finishing_{window}'] = (df[finishing_col] < 0.7).astype(int)
    
    logger.info("Derived efficiency features created")
    return df

def normalize_efficiency_features(df):
    """Normalize efficiency features for ML."""
    
    efficiency_features = [col for col in df.columns if 
                          'efficiency' in col or 'advantage' in col or 
                          'net_performance' in col]
    
    for feature in efficiency_features:
        if df[feature].notna().sum() > 10:  # Only if we have enough data
            # Use robust scaling (less sensitive to outliers)
            q75, q25 = df[feature].quantile(0.75), df[feature].quantile(0.25)
            iqr = q75 - q25
            
            if iqr > 0:
                median = df[feature].median()
                df[f'{feature}_normalized'] = (df[feature] - median) / iqr
                # Cap outliers
                df[f'{feature}_normalized'] = df[f'{feature}_normalized'].clip(-3, 3)
            else:
                df[f'{feature}_normalized'] = 0.5
    
    logger.info(f"Normalized {len(efficiency_features)} efficiency features")
    return df

def build_v31_dataset():
    """Build v3.1 dataset with xG efficiency features."""
    
    # Load base dataset with xG data
    base_path = Path('data/processed/v13_xg_corrected_features_latest.csv')
    output_path = Path('data/processed/v31_efficiency_features_2025_09_06.csv')
    
    logger.info(f"Loading base dataset from {base_path}")
    df = pd.read_csv(base_path)
    
    logger.info(f"Input dataset shape: {df.shape}")
    
    # Need to get actual goals data - check if we have it
    if 'FTHG' not in df.columns:
        logger.warning("No FTHG/FTAG columns found. Attempting to load from raw data...")
        
        # Try to merge with raw data to get goals
        try:
            import glob
            raw_files = glob.glob('/Users/maxime/Desktop/Oddsy/data/raw/football_data_backup/*.csv')
            
            all_raw = []
            for file in sorted(raw_files):
                season_data = pd.read_csv(file)
                season_data['Date'] = pd.to_datetime(season_data['Date'], format='%d/%m/%Y', errors='coerce')
                all_raw.append(season_data)
            
            raw_df = pd.concat(all_raw, ignore_index=True)
            
            # Ensure consistent date format for merging
            raw_df['Date'] = pd.to_datetime(raw_df['Date'], format='%d/%m/%Y', errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Merge to get goals data
            df = df.merge(
                raw_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']],
                on=['Date', 'HomeTeam', 'AwayTeam'],
                how='left'
            )
            
            logger.info(f"Successfully merged goals data. New shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load goals data: {str(e)}")
            return None
    
    # Calculate efficiency features
    df = calculate_efficiency_features(df)
    
    # Create derived features
    df = create_efficiency_derived_features(df)
    
    # Normalize features
    df = normalize_efficiency_features(df)
    
    # Save enhanced dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Saved v3.1 dataset to {output_path}")
    logger.info(f"Output dataset shape: {df.shape}")
    
    # Analyze new features
    efficiency_features = [col for col in df.columns if 
                          'efficiency' in col or 'advantage' in col or 'performance' in col]
    
    print("\n🎯 NEW EFFICIENCY FEATURES:")
    for feature in sorted(efficiency_features[:15]):  # Show first 15
        if df[feature].notna().sum() > 0:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            coverage = df[feature].notna().sum() / len(df) * 100
            print(f"  • {feature}: mean={mean_val:.3f}, std={std_val:.3f}, coverage={coverage:.1f}%")
    
    print(f"\n📊 FEATURE SUMMARY:")
    print(f"  • Total new efficiency features: {len(efficiency_features)}")
    print(f"  • Dataset coverage: {df[efficiency_features].notna().all(axis=1).sum()}/{len(df)} matches")
    
    return df

if __name__ == "__main__":
    logger.info("🚀 Starting v3.1 Efficiency Features build...")
    df = build_v31_dataset()
    if df is not None:
        logger.info("✅ v3.1 Efficiency Features build complete!")
    else:
        logger.error("❌ v3.1 build failed")