#!/usr/bin/env python3
"""
v2.11 Draw-Specific Features Builder
Creates intelligent features focused on draw prediction based on thesis recommendations.
Focus: Equilibrium indicators, momentum variance, and temporal patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_equilibrium_features(df):
    """Calculate draw-specific equilibrium features."""
    
    # 1. Team Strength Equilibrium (closer = more draws)
    df['elo_diff_abs'] = np.abs(df['elo_diff_normalized'] - 0.5)  # Distance from balance
    df['elo_equilibrium'] = 1 - df['elo_diff_abs']  # Closer to 1 = more balanced
    
    # 2. Form Equilibrium (similar recent form = draws)
    df['form_diff_abs'] = np.abs(df['form_diff_normalized'] - 0.5)
    df['form_equilibrium'] = 1 - df['form_diff_abs']
    
    # 3. Shot Balance Indicator
    df['shots_diff_abs'] = np.abs(df['shots_diff_normalized'] - 0.5)
    df['shots_equilibrium'] = 1 - df['shots_diff_abs']
    
    # 4. Corner Balance (possession proxy)
    df['corners_diff_abs'] = np.abs(df['corners_diff_normalized'] - 0.5)
    df['corners_equilibrium'] = 1 - df['corners_diff_abs']
    
    # 5. Overall Team Balance Score (composite)
    equilibrium_features = ['elo_equilibrium', 'form_equilibrium', 'shots_equilibrium', 'corners_equilibrium']
    df['team_balance_score'] = df[equilibrium_features].mean(axis=1)
    
    logger.info("Calculated equilibrium features")
    return df

def calculate_market_draw_signals(df):
    """Calculate market-based draw indicators."""
    
    # 1. Close Odds Indicator (tight markets = draws)
    # Using market probabilities to detect close matches
    if 'market_home_prob_norm' in df.columns and 'market_away_prob_norm' in df.columns:
        df['prob_diff'] = np.abs(df['market_home_prob_norm'] - df['market_away_prob_norm'])
        df['close_odds_indicator'] = (df['prob_diff'] < 0.15).astype(int)  # Tight market
        
        # Market draw favorability
        df['draw_vs_favorites'] = df['market_draw_prob_norm'] / (df['market_home_prob_norm'] + df['market_away_prob_norm'])
        
    else:
        logger.warning("Market probability columns not found")
        df['close_odds_indicator'] = 0
        df['draw_vs_favorites'] = 0.5
    
    # 2. Enhanced market entropy for draw detection
    if 'market_entropy_norm' in df.columns:
        # High entropy often correlates with draws (uncertain outcomes)
        df['high_uncertainty'] = (df['market_entropy_norm'] > df['market_entropy_norm'].quantile(0.7)).astype(int)
    else:
        df['high_uncertainty'] = 0
    
    logger.info("Calculated market draw signals")
    return df

def calculate_temporal_momentum_features(df):
    """Calculate momentum and variance features for draw prediction."""
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize new columns
    df['home_form_variance'] = np.nan
    df['away_form_variance'] = np.nan
    df['home_streak_type'] = 0  # 0=none, 1=win_streak, -1=loss_streak, 2=draw_streak
    df['away_streak_type'] = 0
    df['mutual_streak_interaction'] = 0
    
    # Track recent results for variance calculation
    team_results = {}  # team -> list of recent results (W=3, D=1, L=0)
    team_streaks = {}  # team -> current streak info
    
    for i, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        result = row['FullTimeResult']
        
        # Calculate form variance if we have history
        for team, team_type in [(home_team, 'home'), (away_team, 'away')]:
            if team in team_results and len(team_results[team]) >= 3:
                variance = np.var(team_results[team])
                df.loc[i, f'{team_type}_form_variance'] = variance
            
            # Calculate streak information
            if team in team_streaks:
                streak_info = team_streaks[team]
                df.loc[i, f'{team_type}_streak_type'] = streak_info['type']
        
        # Mutual streak interaction (both teams on similar streaks = more draws)
        if not pd.isna(df.loc[i, 'home_streak_type']) and not pd.isna(df.loc[i, 'away_streak_type']):
            home_streak = df.loc[i, 'home_streak_type']
            away_streak = df.loc[i, 'away_streak_type']
            
            # Similar streaks (both winning, both losing, both drawing) = draw tendency
            if home_streak == away_streak and home_streak != 0:
                df.loc[i, 'mutual_streak_interaction'] = 1
        
        # Update team results and streaks after match
        # Home team result
        if result == 'H':
            home_result, away_result = 3, 0  # Win=3, Loss=0
        elif result == 'D':
            home_result, away_result = 1, 1  # Draw=1
        else:  # 'A'
            home_result, away_result = 0, 3
        
        # Update results history (keep last 5 matches)
        for team, team_result in [(home_team, home_result), (away_team, away_result)]:
            if team not in team_results:
                team_results[team] = []
            team_results[team].append(team_result)
            if len(team_results[team]) > 5:
                team_results[team] = team_results[team][-5:]
            
            # Update streak tracking
            if team not in team_streaks:
                team_streaks[team] = {'type': 0, 'count': 0, 'last_result': None}
            
            streak = team_streaks[team]
            if team_result == 3:  # Win
                if streak['last_result'] == 3:
                    streak['count'] += 1
                else:
                    streak = {'type': 1, 'count': 1, 'last_result': 3}
            elif team_result == 1:  # Draw
                if streak['last_result'] == 1:
                    streak['count'] += 1
                else:
                    streak = {'type': 2, 'count': 1, 'last_result': 1}
            else:  # Loss
                if streak['last_result'] == 0:
                    streak['count'] += 1
                else:
                    streak = {'type': -1, 'count': 1, 'last_result': 0}
            
            team_streaks[team] = streak
    
    # Normalize variance features
    for col in ['home_form_variance', 'away_form_variance']:
        if df[col].notna().sum() > 0:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[f'{col}_normalized'] = 0.5
    
    logger.info("Calculated temporal momentum features")
    return df

def calculate_context_draw_features(df):
    """Calculate context features that influence draw probability."""
    
    # 1. Rest Equilibrium (both teams equally rested = more draws)
    if 'days_since_last_home_normalized' in df.columns and 'days_since_last_away_normalized' in df.columns:
        rest_diff = np.abs(df['days_since_last_home_normalized'] - df['days_since_last_away_normalized'])
        df['rest_equilibrium'] = 1 - rest_diff  # Close to 1 = both equally rested
    else:
        df['rest_equilibrium'] = 0.5
    
    # 2. Midweek fatigue effect (tired teams = more draws)
    if 'is_midweek' in df.columns:
        df['midweek_draw_factor'] = df['is_midweek'] * 1.2  # Midweek games have more draws
    else:
        df['midweek_draw_factor'] = 0
    
    # 3. Travel fatigue neutralizer (long travel = home advantage reduced = more draws)
    if 'travel_fatigue_factor' in df.columns:
        df['travel_draw_factor'] = df['travel_fatigue_factor'] * 0.8  # High travel = more draws
    else:
        df['travel_draw_factor'] = 0
    
    # 4. Season progression draw tendency (mid-season = more draws due to fatigue)
    if 'matchday_normalized' in df.columns:
        # U-shaped curve: draws more common in mid-season
        season_progress = df['matchday_normalized']
        df['midseason_draw_factor'] = 1 - np.abs(season_progress - 0.5) * 2  # Peak at mid-season
    else:
        df['midseason_draw_factor'] = 0.5
    
    logger.info("Calculated context draw features")
    return df

def create_composite_draw_score(df):
    """Create final composite draw prediction score."""
    
    # Weight different categories of draw indicators
    equilibrium_features = ['team_balance_score', 'rest_equilibrium']
    market_features = ['close_odds_indicator', 'high_uncertainty', 'draw_vs_favorites']
    context_features = ['midweek_draw_factor', 'travel_draw_factor', 'midseason_draw_factor']
    momentum_features = ['mutual_streak_interaction']
    
    # Add variance features if available
    variance_features = []
    for col in ['home_form_variance_normalized', 'away_form_variance_normalized']:
        if col in df.columns:
            variance_features.append(col)
    
    # Calculate weighted composite score
    equilibrium_weight = 0.4
    market_weight = 0.3
    context_weight = 0.2
    momentum_weight = 0.1
    
    df['equilibrium_score'] = df[equilibrium_features].mean(axis=1) * equilibrium_weight
    df['market_score'] = df[market_features].mean(axis=1) * market_weight
    df['context_score'] = df[context_features].mean(axis=1) * context_weight
    df['momentum_score'] = df[momentum_features].mean(axis=1) * momentum_weight
    
    # Add variance component if available
    if variance_features:
        # High variance = unpredictable = more draws
        df['variance_score'] = df[variance_features].mean(axis=1) * 0.1
        df['draw_propensity_score'] = (df['equilibrium_score'] + df['market_score'] + 
                                     df['context_score'] + df['momentum_score'] + df['variance_score'])
    else:
        df['draw_propensity_score'] = (df['equilibrium_score'] + df['market_score'] + 
                                     df['context_score'] + df['momentum_score'])
    
    # Normalize final score
    min_val = df['draw_propensity_score'].min()
    max_val = df['draw_propensity_score'].max()
    if max_val > min_val:
        df['draw_propensity_score_normalized'] = (df['draw_propensity_score'] - min_val) / (max_val - min_val)
    else:
        df['draw_propensity_score_normalized'] = 0.5
    
    logger.info("Created composite draw prediction score")
    return df

def build_v211_dataset():
    """Build v2.11 dataset with draw-specific features."""
    
    # Load existing v2.9 dataset (has context features)
    input_path = Path('data/processed/v29_context_features_2025_09_06.csv')
    output_path = Path('data/processed/v211_draw_features_2025_09_06.csv')
    
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    logger.info(f"Input dataset shape: {df.shape}")
    
    # Calculate all draw-specific features
    df = calculate_equilibrium_features(df)
    df = calculate_market_draw_signals(df)
    df = calculate_temporal_momentum_features(df)
    df = calculate_context_draw_features(df)
    df = create_composite_draw_score(df)
    
    # Save enhanced dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Saved v2.11 dataset to {output_path}")
    logger.info(f"Output dataset shape: {df.shape}")
    
    # Feature analysis
    draw_features = [col for col in df.columns if any(x in col for x in 
                    ['equilibrium', 'balance', 'draw', 'streak', 'variance', 'propensity'])]
    
    print("\\n🎯 New Draw-Specific Features:")
    for feature in sorted(draw_features):
        if df[feature].notna().sum() > 0:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            # Check correlation with actual draws
            is_draw = (df['FullTimeResult'] == 'D').astype(int)
            correlation = df[feature].corr(is_draw) if df[feature].notna().sum() > 10 else 0
            print(f"  • {feature}: mean={mean_val:.3f}, std={std_val:.3f}, draw_corr={correlation:.3f}")
    
    return df

if __name__ == "__main__":
    logger.info("🚀 Starting v2.11 Draw Features build...")
    df = build_v211_dataset()
    logger.info("✅ v2.11 Draw Features build complete!")