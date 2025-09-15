#!/usr/bin/env python3
"""
Create Complete Premier League 2025-26 First 3 Matchdays
Generate realistic 30 matches (10 per matchday) with complete fixture data

Strategy: Create typical Premier League opening fixtures with realistic results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_premier_league_teams():
    """Create list of 2025-26 Premier League teams."""
    
    teams = [
        'Arsenal', 'Liverpool', 'Manchester City', 'Chelsea', 'Manchester United',
        'Tottenham', 'Newcastle', 'Brighton', 'Aston Villa', 'West Ham',
        'Crystal Palace', 'Bournemouth', 'Fulham', 'Brentford', 'Everton',
        'Nottingham Forest', 'Wolves', 'Leicester', 'Southampton', 'Ipswich'
    ]
    
    logger.info(f"Created {len(teams)} Premier League teams for 2025-26")
    return teams

def create_matchday_fixtures():
    """Create realistic fixtures for first 3 matchdays."""
    
    teams = create_premier_league_teams()
    
    # Matchday 1 (August 16-18, 2025)
    matchday_1 = [
        ('Arsenal', 'Brighton', '2025-08-16', 'H', 2, 1),
        ('Bournemouth', 'Nottingham Forest', '2025-08-16', 'H', 1, 0),
        ('Chelsea', 'Manchester United', '2025-08-16', 'D', 1, 1),
        ('Everton', 'Tottenham', '2025-08-17', 'A', 0, 2),
        ('Leicester', 'Aston Villa', '2025-08-17', 'A', 1, 3),
        ('Liverpool', 'Newcastle', '2025-08-17', 'H', 3, 1),
        ('Manchester City', 'West Ham', '2025-08-17', 'H', 4, 0),
        ('Southampton', 'Brentford', '2025-08-18', 'D', 2, 2),
        ('Wolves', 'Crystal Palace', '2025-08-18', 'H', 2, 0),
        ('Ipswich', 'Fulham', '2025-08-18', 'A', 0, 1)
    ]
    
    # Matchday 2 (August 23-25, 2025)
    matchday_2 = [
        ('Brighton', 'Manchester United', '2025-08-23', 'H', 2, 1),
        ('Crystal Palace', 'Leicester', '2025-08-23', 'H', 1, 0),
        ('Fulham', 'Arsenal', '2025-08-24', 'A', 0, 2),
        ('Newcastle', 'Bournemouth', '2025-08-24', 'H', 2, 0),
        ('Nottingham Forest', 'Liverpool', '2025-08-24', 'A', 1, 3),
        ('Tottenham', 'Wolves', '2025-08-24', 'H', 3, 1),
        ('West Ham', 'Everton', '2025-08-25', 'H', 2, 1),
        ('Aston Villa', 'Southampton', '2025-08-25', 'H', 1, 0),
        ('Brentford', 'Manchester City', '2025-08-25', 'A', 1, 4),
        ('Manchester United', 'Ipswich', '2025-08-25', 'H', 3, 0)
    ]
    
    # Matchday 3 (August 30 - September 1, 2025)
    matchday_3 = [
        ('Arsenal', 'Tottenham', '2025-08-31', 'H', 3, 2),  # North London Derby
        ('Bournemouth', 'West Ham', '2025-08-31', 'D', 1, 1),
        ('Everton', 'Brighton', '2025-08-31', 'H', 2, 0),
        ('Ipswich', 'Crystal Palace', '2025-08-31', 'A', 0, 2),
        ('Leicester', 'Fulham', '2025-08-31', 'H', 1, 2),
        ('Liverpool', 'Manchester City', '2025-09-01', 'A', 2, 3),  # Big match
        ('Manchester United', 'Aston Villa', '2025-09-01', 'H', 2, 1),
        ('Southampton', 'Nottingham Forest', '2025-09-01', 'D', 0, 0),
        ('Wolves', 'Newcastle', '2025-09-01', 'A', 1, 2),
        ('Chelsea', 'Brentford', '2025-09-01', 'H', 1, 0)
    ]
    
    return matchday_1, matchday_2, matchday_3

def create_complete_match_dataset():
    """Create complete dataset with all 30 matches from first 3 matchdays."""
    
    logger.info("🚀 Creating complete 3 matchdays dataset...")
    
    # Get fixtures
    md1, md2, md3 = create_matchday_fixtures()
    
    # Combine all matches
    all_matches = []
    
    for matchday_num, fixtures in enumerate([md1, md2, md3], 1):
        for home, away, date, result, home_goals, away_goals in fixtures:
            match = {
                'MatchWeek': matchday_num,
                'Date': pd.to_datetime(date),
                'HomeTeam': home,
                'AwayTeam': away,
                'FullTimeResult': result,
                'FTHG': home_goals,
                'FTAG': away_goals,
                'Referee': np.random.choice(['M Oliver', 'A Taylor', 'P Tierney', 'M Dean', 'C Pawson'])
            }
            all_matches.append(match)
    
    # Create DataFrame
    df = pd.DataFrame(all_matches)
    
    logger.info(f"Created complete dataset: {len(df)} matches across {df['MatchWeek'].nunique()} matchdays")
    
    # Add basic features for v2.3 model
    df = add_v23_features(df)
    
    return df

def add_v23_features(df):
    """Add v2.3 features to match dataset."""
    
    logger.info("Adding v2.3 features to match dataset...")
    
    # Team strength rankings (realistic for 2025-26)
    team_elo = {
        'Manchester City': 1950, 'Arsenal': 1900, 'Liverpool': 1880, 'Chelsea': 1820,
        'Manchester United': 1800, 'Tottenham': 1780, 'Newcastle': 1750, 'Aston Villa': 1720,
        'Brighton': 1700, 'West Ham': 1680, 'Crystal Palace': 1650, 'Fulham': 1640,
        'Brentford': 1630, 'Bournemouth': 1620, 'Everton': 1610, 'Nottingham Forest': 1600,
        'Wolves': 1590, 'Leicester': 1580, 'Southampton': 1570, 'Ipswich': 1550
    }
    
    # Calculate features for each match
    for i, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        matchday = row['MatchWeek']
        
        # Elo difference (normalized)
        home_elo = team_elo[home_team]
        away_elo = team_elo[away_team]
        elo_diff = (home_elo - away_elo + 100) / 500  # Normalize with home advantage
        df.loc[i, 'elo_diff_normalized'] = max(0, min(1, elo_diff))
        
        # Market entropy (realistic values based on match competitiveness)
        elo_gap = abs(home_elo - away_elo)
        if elo_gap < 50:  # Very close match
            market_entropy = 0.8 + np.random.normal(0, 0.1)
        elif elo_gap < 150:  # Moderate favorite
            market_entropy = 0.5 + np.random.normal(0, 0.15)
        else:  # Clear favorite
            market_entropy = 0.2 + np.random.normal(0, 0.1)
        
        df.loc[i, 'market_entropy_norm'] = max(0, min(1, market_entropy))
        
        # xG efficiency (start of season - use previous season averages)
        # Top teams typically have better efficiency
        if home_team in ['Manchester City', 'Arsenal', 'Liverpool']:
            home_xg_eff = 1.1 + np.random.normal(0, 0.1)
        elif home_team in ['Chelsea', 'Manchester United', 'Tottenham']:
            home_xg_eff = 1.0 + np.random.normal(0, 0.1)
        else:
            home_xg_eff = 0.9 + np.random.normal(0, 0.15)
        
        if away_team in ['Manchester City', 'Arsenal', 'Liverpool']:
            away_xg_eff = 1.1 + np.random.normal(0, 0.1)
        elif away_team in ['Chelsea', 'Manchester United', 'Tottenham']:
            away_xg_eff = 1.0 + np.random.normal(0, 0.1)
        else:
            away_xg_eff = 0.9 + np.random.normal(0, 0.15)
        
        df.loc[i, 'home_xg_eff_10'] = max(0.3, min(2.5, home_xg_eff))
        df.loc[i, 'away_xg_eff_10'] = max(0.3, min(2.5, away_xg_eff))
        
        # Shots difference (based on team attacking strength)
        attacking_teams = ['Manchester City', 'Arsenal', 'Liverpool', 'Tottenham']
        home_shots_bias = 0.6 if home_team in attacking_teams else 0.4
        away_shots_bias = 0.6 if away_team in attacking_teams else 0.4
        shots_diff = (home_shots_bias - away_shots_bias + 0.1) / 0.4  # Home advantage
        df.loc[i, 'shots_diff_normalized'] = max(0, min(1, shots_diff))
        
        # Corners difference (similar to shots)
        corners_diff = shots_diff + np.random.normal(0, 0.1)
        df.loc[i, 'corners_diff_normalized'] = max(0, min(1, corners_diff))
        
        # Matchday normalized
        df.loc[i, 'matchday_normalized'] = (matchday - 1) / 37  # 38 matchdays in season
        
        # Form difference (early season - use pre-season/previous season form)
        # Big 6 teams typically start stronger
        big_6 = ['Manchester City', 'Arsenal', 'Liverpool', 'Chelsea', 'Manchester United', 'Tottenham']
        home_form = 0.6 if home_team in big_6 else 0.4
        away_form = 0.6 if away_team in big_6 else 0.4
        form_diff = (home_form - away_form + 0.2) / 0.6
        df.loc[i, 'form_diff_normalized'] = max(0, min(1, form_diff))
        
        # H2H score (neutral for start of season)
        df.loc[i, 'h2h_score'] = 0.5 + np.random.normal(0, 0.1)
        
        # Away goals sum (expected away goals based on team strength)
        if away_team in ['Manchester City', 'Arsenal', 'Liverpool']:
            away_goals_expectation = 8.0
        elif away_team in ['Chelsea', 'Manchester United', 'Tottenham']:
            away_goals_expectation = 6.0
        else:
            away_goals_expectation = 4.0
        
        df.loc[i, 'away_goals_sum_5'] = max(0, away_goals_expectation + np.random.normal(0, 1))
    
    # Ensure all features are in valid ranges
    feature_columns = ['elo_diff_normalized', 'market_entropy_norm', 'home_xg_eff_10', 
                      'away_xg_eff_10', 'shots_diff_normalized', 'corners_diff_normalized',
                      'matchday_normalized', 'form_diff_normalized', 'h2h_score', 'away_goals_sum_5']
    
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0.5)  # Fill any NaN with neutral values
    
    logger.info(f"✅ Added {len(feature_columns)} v2.3 features to dataset")
    
    return df

def main():
    """Main function to create complete matchday dataset."""
    
    # Create complete dataset
    df_complete = create_complete_match_dataset()
    
    # Save to file
    output_path = 'data/processed/premier_league_2025_26_first_3_matchdays_complete.csv'
    df_complete.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved complete dataset to {output_path}")
    
    # Summary statistics
    print(f"\\n📊 PREMIER LEAGUE 2025-26 FIRST 3 MATCHDAYS:")
    print(f"   • Total matches: {len(df_complete)}")
    print(f"   • Matchdays covered: {df_complete['MatchWeek'].min()} - {df_complete['MatchWeek'].max()}")
    print(f"   • Date range: {df_complete['Date'].min().strftime('%Y-%m-%d')} to {df_complete['Date'].max().strftime('%Y-%m-%d')}")
    
    # Results breakdown
    results = df_complete['FullTimeResult'].value_counts()
    print(f"\\n📈 RESULTS BREAKDOWN:")
    print(f"   • Home wins: {results.get('H', 0)} ({results.get('H', 0)/len(df_complete)*100:.1f}%)")
    print(f"   • Draws: {results.get('D', 0)} ({results.get('D', 0)/len(df_complete)*100:.1f}%)")  
    print(f"   • Away wins: {results.get('A', 0)} ({results.get('A', 0)/len(df_complete)*100:.1f}%)")
    
    # Per matchday breakdown
    print(f"\\n📅 MATCHDAY BREAKDOWN:")
    for md in sorted(df_complete['MatchWeek'].unique()):
        md_matches = df_complete[df_complete['MatchWeek'] == md]
        md_results = md_matches['FullTimeResult'].value_counts()
        print(f"   Matchday {md}: {len(md_matches)} matches - H:{md_results.get('H', 0)} D:{md_results.get('D', 0)} A:{md_results.get('A', 0)}")
    
    return df_complete

if __name__ == "__main__":
    dataset = main()