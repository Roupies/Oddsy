#!/usr/bin/env python3
"""
v4.0 Fatigue Features - The Physical Performance Edge
Focus: Fixture congestion, recovery time, and cumulative load impact

Strategy: Quantify team fatigue through calendar analysis and player load
Hypothesis: Teams with insufficient recovery perform worse than expected
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_fixture_density_features(df):
    """Calculate fixture congestion and recovery features."""
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize fatigue features
    fatigue_features = [
        'home_days_since_last_match', 'away_days_since_last_match',
        'home_matches_in_last_14_days', 'away_matches_in_last_14_days',
        'home_fixture_congestion_index', 'away_fixture_congestion_index',
        'home_recovery_advantage', 'away_recovery_advantage',
        'fixture_density_differential'
    ]
    
    for feature in fatigue_features:
        df[feature] = np.nan
    
    # Track each team's match history for fatigue calculation
    team_match_history = {}  # team -> [(date, is_home), ...]
    
    logger.info("Calculating fixture density and recovery features...")
    
    for i, row in df.iterrows():
        match_date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Calculate features for both teams
        for team, team_type in [(home_team, 'home'), (away_team, 'away')]:
            
            if team in team_match_history:
                history = team_match_history[team]
                
                # Feature 1: Days since last match
                if len(history) > 0:
                    last_match_date = history[-1][0]
                    days_since = (match_date - last_match_date).days
                    df.loc[i, f'{team_type}_days_since_last_match'] = days_since
                else:
                    df.loc[i, f'{team_type}_days_since_last_match'] = 7  # Default weekly
                
                # Feature 2: Matches in last 14 days
                cutoff_date = match_date - timedelta(days=14)
                recent_matches = [h for h in history if h[0] >= cutoff_date]
                df.loc[i, f'{team_type}_matches_in_last_14_days'] = len(recent_matches)
                
                # Feature 3: Fixture Congestion Index (weighted by recency)
                if len(recent_matches) > 0:
                    # Weight recent matches more heavily
                    congestion_score = 0
                    for match_date_hist, _ in recent_matches:
                        days_ago = (match_date - match_date_hist).days
                        if days_ago <= 3:
                            weight = 1.0  # Very recent
                        elif days_ago <= 7:
                            weight = 0.7  # Recent
                        else:
                            weight = 0.3  # Less recent
                        congestion_score += weight
                    
                    df.loc[i, f'{team_type}_fixture_congestion_index'] = congestion_score
                else:
                    df.loc[i, f'{team_type}_fixture_congestion_index'] = 0.0
            else:
                # First match for this team in dataset
                df.loc[i, f'{team_type}_days_since_last_match'] = 7
                df.loc[i, f'{team_type}_matches_in_last_14_days'] = 0
                df.loc[i, f'{team_type}_fixture_congestion_index'] = 0.0
        
        # Update match history after calculating features
        is_home_for_home_team = True
        is_home_for_away_team = False
        
        if home_team not in team_match_history:
            team_match_history[home_team] = []
        team_match_history[home_team].append((match_date, is_home_for_home_team))
        
        if away_team not in team_match_history:
            team_match_history[away_team] = []
        team_match_history[away_team].append((match_date, is_home_for_away_team))
        
        # Keep only last 30 matches for memory efficiency
        for team in [home_team, away_team]:
            if len(team_match_history[team]) > 30:
                team_match_history[team] = team_match_history[team][-30:]
    
    # Calculate differential/advantage features
    df['fixture_density_differential'] = (
        df['away_fixture_congestion_index'] - df['home_fixture_congestion_index']
    )  # Positive = home advantage (away team more fatigued)
    
    df['home_recovery_advantage'] = (
        df['away_days_since_last_match'] - df['home_days_since_last_match']
    )  # Positive = home team had less rest (disadvantage)
    
    df['away_recovery_advantage'] = -df['home_recovery_advantage']
    
    logger.info("Fixture density features calculation complete")
    return df

def calculate_travel_fatigue_features(df):
    """Calculate travel-based fatigue (simplified European travel simulation)."""
    
    # Simplified travel distance matrix (approximate km between major cities)
    travel_distances = {
        ('London', 'Manchester'): 300,
        ('London', 'Liverpool'): 350,
        ('London', 'Newcastle'): 450,
        ('London', 'Birmingham'): 200,
        ('Manchester', 'Liverpool'): 50,
        ('Manchester', 'Newcastle'): 200,
        ('Birmingham', 'Liverpool'): 150,
        # Add more as needed
    }
    
    # Team city mappings (simplified)
    team_cities = {
        'Arsenal': 'London', 'Chelsea': 'London', 'Tottenham': 'London',
        'West Ham': 'London', 'Crystal Palace': 'London', 'Fulham': 'London',
        'Brentford': 'London',
        'Man City': 'Manchester', 'Man United': 'Manchester',
        'Liverpool': 'Liverpool', 'Everton': 'Liverpool',
        'Newcastle': 'Newcastle',
        'Aston Villa': 'Birmingham', 'Wolves': 'Birmingham',
        'Brighton': 'Brighton', 'Bournemouth': 'Bournemouth',
        'Nottm Forest': 'Nottingham', 'Sheffield United': 'Sheffield',
        'Burnley': 'Burnley', 'Luton': 'Luton'
    }
    
    # Calculate travel distance for away team
    df['away_travel_distance'] = 0
    
    for i, row in df.iterrows():
        home_city = team_cities.get(row['HomeTeam'], 'Unknown')
        away_city = team_cities.get(row['AwayTeam'], 'Unknown')
        
        if home_city != 'Unknown' and away_city != 'Unknown':
            # Check both directions in travel matrix
            distance = travel_distances.get((away_city, home_city), 
                                          travel_distances.get((home_city, away_city), 100))
            df.loc[i, 'away_travel_distance'] = distance
    
    # Travel fatigue index (longer distance = more fatigue)
    df['away_travel_fatigue_index'] = df['away_travel_distance'] / 500  # Normalize
    
    logger.info("Travel fatigue features created")
    return df

def create_fatigue_derived_features(df):
    """Create derived features from basic fatigue metrics."""
    
    # Fatigue severity categories
    df['home_severe_congestion'] = (df['home_fixture_congestion_index'] > 2.0).astype(int)
    df['away_severe_congestion'] = (df['away_fixture_congestion_index'] > 2.0).astype(int)
    
    # Recovery deficit (insufficient rest)
    df['home_recovery_deficit'] = np.maximum(0, 3 - df['home_days_since_last_match'])
    df['away_recovery_deficit'] = np.maximum(0, 3 - df['away_days_since_last_match'])
    
    # Combined fatigue score (congestion + recovery deficit + travel)
    df['home_total_fatigue_score'] = (
        df['home_fixture_congestion_index'] + 
        df['home_recovery_deficit'] * 0.5
    )
    
    df['away_total_fatigue_score'] = (
        df['away_fixture_congestion_index'] + 
        df['away_recovery_deficit'] * 0.5 +
        df['away_travel_fatigue_index'] * 0.3
    )
    
    # Fatigue advantage (positive = home advantage)
    df['fatigue_advantage'] = df['away_total_fatigue_score'] - df['home_total_fatigue_score']
    
    # High-impact fatigue scenarios
    df['home_critical_fatigue'] = (
        (df['home_days_since_last_match'] <= 3) & 
        (df['home_matches_in_last_14_days'] >= 3)
    ).astype(int)
    
    df['away_critical_fatigue'] = (
        (df['away_days_since_last_match'] <= 3) & 
        (df['away_matches_in_last_14_days'] >= 3) &
        (df['away_travel_distance'] > 200)
    ).astype(int)
    
    logger.info("Derived fatigue features created")
    return df

def normalize_fatigue_features(df):
    """Normalize fatigue features for ML consumption."""
    
    fatigue_features = [col for col in df.columns if 
                       'fatigue' in col or 'congestion' in col or 
                       'recovery' in col or 'travel' in col]
    
    # Remove binary/categorical features from normalization
    numeric_features = []
    for feature in fatigue_features:
        if not feature.endswith(('_deficit', '_advantage', '_index', '_score')):
            continue
        if df[feature].dtype in ['int64', 'float64'] and df[feature].nunique() > 2:
            numeric_features.append(feature)
    
    for feature in numeric_features:
        if df[feature].notna().sum() > 10:
            # Robust scaling
            q75, q25 = df[feature].quantile(0.75), df[feature].quantile(0.25)
            iqr = q75 - q25
            
            if iqr > 0:
                median = df[feature].median()
                df[f'{feature}_normalized'] = (df[feature] - median) / iqr
                df[f'{feature}_normalized'] = df[f'{feature}_normalized'].clip(-3, 3)
            else:
                df[f'{feature}_normalized'] = 0.5
    
    logger.info(f"Normalized fatigue features for ML consumption")
    return df

def build_v40_fatigue_dataset():
    """Build v4.0 dataset with fatigue and fixture congestion features."""
    
    # Load v3.1 baseline
    base_path = Path('data/processed/v31_efficiency_features_2025_09_06.csv')
    output_path = Path('data/processed/v40_fatigue_features_2025_09_07.csv')
    
    logger.info(f"Loading v3.1 baseline from {base_path}")
    df = pd.read_csv(base_path)
    
    logger.info(f"Input dataset shape: {df.shape}")
    
    # Calculate fixture density features
    df = calculate_fixture_density_features(df)
    
    # Calculate travel fatigue
    df = calculate_travel_fatigue_features(df)
    
    # Create derived features
    df = create_fatigue_derived_features(df)
    
    # Normalize features
    df = normalize_fatigue_features(df)
    
    # Save enhanced dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Saved v4.0 dataset to {output_path}")
    logger.info(f"Output dataset shape: {df.shape}")
    
    # Analyze new features
    fatigue_features = [col for col in df.columns if 
                       'fatigue' in col or 'congestion' in col or 
                       'recovery' in col or 'travel' in col or 
                       'days_since' in col or 'matches_in' in col]
    
    print("\n🎯 NEW FATIGUE FEATURES:")
    for feature in sorted(fatigue_features[:20]):  # Show first 20
        if df[feature].notna().sum() > 0:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            coverage = df[feature].notna().sum() / len(df) * 100
            print(f"  • {feature}: mean={mean_val:.3f}, std={std_val:.3f}, coverage={coverage:.1f}%")
    
    print(f"\n📊 FEATURE SUMMARY:")
    print(f"  • Total new fatigue features: {len(fatigue_features)}")
    print(f"  • Dataset coverage: {df[fatigue_features].notna().all(axis=1).sum()}/{len(df)} matches")
    
    # Fatigue insights
    print(f"\n🔍 FATIGUE INSIGHTS:")
    
    # Average recovery time
    avg_home_recovery = df['home_days_since_last_match'].mean()
    avg_away_recovery = df['away_days_since_last_match'].mean()
    print(f"  • Average home recovery: {avg_home_recovery:.1f} days")
    print(f"  • Average away recovery: {avg_away_recovery:.1f} days")
    
    # Congestion frequency
    severe_home_congestion = (df['home_severe_congestion'] == 1).sum()
    severe_away_congestion = (df['away_severe_congestion'] == 1).sum()
    print(f"  • Severe home congestion: {severe_home_congestion} matches ({severe_home_congestion/len(df)*100:.1f}%)")
    print(f"  • Severe away congestion: {severe_away_congestion} matches ({severe_away_congestion/len(df)*100:.1f}%)")
    
    # Critical fatigue scenarios
    critical_home = (df['home_critical_fatigue'] == 1).sum()
    critical_away = (df['away_critical_fatigue'] == 1).sum()
    print(f"  • Critical home fatigue: {critical_home} matches ({critical_home/len(df)*100:.1f}%)")
    print(f"  • Critical away fatigue: {critical_away} matches ({critical_away/len(df)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    logger.info("🚀 Starting v4.0 Fatigue Features build...")
    df = build_v40_fatigue_dataset()
    if df is not None:
        logger.info("✅ v4.0 Fatigue Features build complete!")
    else:
        logger.error("❌ v4.0 build failed")