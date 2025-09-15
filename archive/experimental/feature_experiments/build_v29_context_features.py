#!/usr/bin/env python3
"""
v2.9 Context Features Builder
Implements fixture congestion and travel distance features for EPL prediction.
Focus: Real football dynamics that impact team performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from geopy.distance import geodesic

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_stadium_coordinates():
    """Load EPL stadium coordinates."""
    coords_path = Path('data/external/epl_stadium_coordinates.csv')
    coords_df = pd.read_csv(coords_path)
    
    # Create mapping from team name to coordinates
    coords_dict = {}
    for _, row in coords_df.iterrows():
        coords_dict[row['team']] = (row['latitude'], row['longitude'])
    
    logger.info(f"Loaded coordinates for {len(coords_dict)} stadiums")
    return coords_dict

def calculate_fixture_congestion(df):
    """Calculate fixture congestion features."""
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize congestion features
    df['days_since_last_home'] = np.nan
    df['days_since_last_away'] = np.nan
    df['fixture_congestion_diff'] = np.nan
    df['matches_in_7d_home'] = 0
    df['matches_in_7d_away'] = 0
    df['is_midweek'] = 0
    
    # Track last match date for each team
    last_match = {}
    
    for i, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        match_date = row['Date']
        
        # Check if midweek (Tuesday, Wednesday, Thursday)
        if match_date.weekday() in [1, 2, 3]:  # 1=Tuesday, 2=Wednesday, 3=Thursday
            df.loc[i, 'is_midweek'] = 1
        
        # Calculate days since last match for each team
        if home_team in last_match:
            days_since_home = (match_date - last_match[home_team]).days
            df.loc[i, 'days_since_last_home'] = days_since_home
        
        if away_team in last_match:
            days_since_away = (match_date - last_match[away_team]).days
            df.loc[i, 'days_since_last_away'] = days_since_away
        
        # Calculate fixture congestion difference (positive = home team has advantage)
        if not pd.isna(df.loc[i, 'days_since_last_home']) and not pd.isna(df.loc[i, 'days_since_last_away']):
            df.loc[i, 'fixture_congestion_diff'] = df.loc[i, 'days_since_last_away'] - df.loc[i, 'days_since_last_home']
        
        # Count matches in last 7 days for each team
        week_ago = match_date - timedelta(days=7)
        
        # Count home team matches
        home_recent = df[(df['Date'] > week_ago) & (df['Date'] < match_date) & 
                        ((df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team))]
        df.loc[i, 'matches_in_7d_home'] = len(home_recent)
        
        # Count away team matches
        away_recent = df[(df['Date'] > week_ago) & (df['Date'] < match_date) & 
                        ((df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team))]
        df.loc[i, 'matches_in_7d_away'] = len(away_recent)
        
        # Update last match dates
        last_match[home_team] = match_date
        last_match[away_team] = match_date
    
    logger.info("Calculated fixture congestion features")
    return df

def calculate_travel_distance(df, coords_dict):
    """Calculate travel distance features."""
    
    df['travel_distance_km'] = np.nan
    df['travel_fatigue_factor'] = np.nan
    
    for i, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Handle team name variations
        home_coords = coords_dict.get(home_team)
        away_coords = coords_dict.get(away_team)
        
        if home_coords and away_coords:
            # Calculate great circle distance
            distance_km = geodesic(away_coords, home_coords).kilometers
            df.loc[i, 'travel_distance_km'] = distance_km
            
            # Travel fatigue factor (normalized 0-1, max ~600km for Newcastle-Bournemouth)
            max_distance = 600  # Approximate max EPL travel distance
            df.loc[i, 'travel_fatigue_factor'] = min(distance_km / max_distance, 1.0)
        else:
            logger.warning(f"Missing coordinates for {home_team} vs {away_team}")
    
    logger.info(f"Calculated travel distance for {df['travel_distance_km'].notna().sum()} matches")
    return df

def normalize_context_features(df):
    """Normalize context features for ML."""
    
    # Features to normalize
    context_features = [
        'days_since_last_home', 'days_since_last_away', 'fixture_congestion_diff',
        'matches_in_7d_home', 'matches_in_7d_away', 'travel_distance_km'
    ]
    
    for feature in context_features:
        if feature in df.columns and df[feature].notna().sum() > 0:
            # Handle potential infinite values
            df[feature] = df[feature].fillna(df[feature].median())
            
            min_val = df[feature].min()
            max_val = df[feature].max()
            
            if max_val > min_val:
                df[f'{feature}_normalized'] = (df[feature] - min_val) / (max_val - min_val)
            else:
                df[f'{feature}_normalized'] = 0.5  # Default for constant values
    
    logger.info(f"Normalized {len(context_features)} context features")
    return df

def build_v29_dataset():
    """Build v2.9 dataset with context features."""
    
    # Load existing dataset (v2.4 baseline)
    input_path = Path('data/processed/v13_xg_corrected_features_latest.csv')
    output_path = Path('data/processed/v29_context_features_2025_09_06.csv')
    
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    logger.info(f"Input dataset shape: {df.shape}")
    
    # Load stadium coordinates
    coords_dict = load_stadium_coordinates()
    
    # Calculate fixture congestion features
    df = calculate_fixture_congestion(df)
    
    # Calculate travel distance features
    df = calculate_travel_distance(df, coords_dict)
    
    # Normalize features
    df = normalize_context_features(df)
    
    # Save enhanced dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Saved v2.9 dataset to {output_path}")
    logger.info(f"Output dataset shape: {df.shape}")
    
    # Feature analysis
    context_features = [col for col in df.columns if any(x in col for x in 
                       ['congestion', 'travel', 'midweek', 'days_since', 'matches_in_7d'])]
    
    print("\\n🎯 New Context Features:")
    for feature in context_features:
        if df[feature].notna().sum() > 0:
            print(f"  • {feature}: mean={df[feature].mean():.3f}, std={df[feature].std():.3f}, coverage={df[feature].notna().sum()}/{len(df)}")
    
    return df

if __name__ == "__main__":
    logger.info("🚀 Starting v2.9 Context Features build...")
    df = build_v29_dataset()
    logger.info("✅ v2.9 Context Features build complete!")