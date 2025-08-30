import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging, load_config

def enhance_features_v12():
    """
    Add 3 new simple but effective features for v1.2:
    1. elo_home_advantage: Elo difference with home advantage boost
    2. form_momentum: Acceleration of recent form (improving vs declining)
    3. h2h_recent_weight: Head-to-head weighted by recency
    """
    
    logger = setup_logging()
    logger.info("=== v1.2 Feature Enhancement Started ===")
    
    # Load the base ML-ready dataset
    input_file = "data/processed/premier_league_ml_ready.csv"
    logger.info(f"Loading base dataset: {input_file}")
    
    df = pd.read_csv(input_file)
    logger.info(f"Base dataset loaded: {df.shape}")
    
    # Convert Date to datetime for calculations
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date']).reset_index(drop=True)
    
    logger.info("Adding new features...")
    
    # Feature 1: Enhanced Elo with Home Advantage
    logger.info("1. Creating elo_home_advantage feature...")
    
    # Current elo_diff ranges from ~0.2 to 0.8
    # Add a home advantage boost (typically 50-100 Elo points = ~0.05-0.1 in normalized scale)
    home_advantage_boost = 0.05  # Conservative boost
    
    df['elo_home_advantage'] = df['elo_diff_normalized'] + home_advantage_boost
    # Ensure it stays in [0,1] range
    df['elo_home_advantage'] = np.clip(df['elo_home_advantage'], 0, 1)
    
    # Feature 2: Form Momentum (acceleration)
    logger.info("2. Creating form_momentum feature...")
    
    # We'll use a simplified approach since we have form_diff_normalized
    # Form momentum = how form is trending (improving vs declining)
    # This is a proxy - in real implementation we'd need historical form data
    
    teams = []
    for _, row in df.iterrows():
        teams.extend([row['HomeTeam'], row['AwayTeam']])
    unique_teams = list(set(teams))
    
    # Initialize form momentum dictionary
    team_form_momentum = {team: 0.5 for team in unique_teams}  # Neutral start
    
    form_momentum_list = []
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Get current momentum for both teams
        home_momentum = team_form_momentum.get(home_team, 0.5)
        away_momentum = team_form_momentum.get(away_team, 0.5)
        
        # Form momentum difference (home advantage perspective)
        momentum_diff = home_momentum - away_momentum
        # Normalize to [0,1]
        momentum_normalized = (momentum_diff + 1) / 2  # Range [-1,1] -> [0,1]
        
        form_momentum_list.append(momentum_normalized)
        
        # Update momentum based on result (simple learning)
        result = row['FullTimeResult']
        if result == 'H':
            team_form_momentum[home_team] = min(1.0, team_form_momentum[home_team] + 0.1)
            team_form_momentum[away_team] = max(0.0, team_form_momentum[away_team] - 0.05)
        elif result == 'A':
            team_form_momentum[away_team] = min(1.0, team_form_momentum[away_team] + 0.1)
            team_form_momentum[home_team] = max(0.0, team_form_momentum[home_team] - 0.05)
        # Draw: slight momentum decay
        else:
            team_form_momentum[home_team] = max(0.0, team_form_momentum[home_team] - 0.02)
            team_form_momentum[away_team] = max(0.0, team_form_momentum[away_team] - 0.02)
    
    df['form_momentum'] = form_momentum_list
    
    # Feature 3: Enhanced H2H with Recency Weighting
    logger.info("3. Creating h2h_recent_weight feature...")
    
    # Current h2h_score is 0-1, we'll add recency weighting
    # More recent H2H meetings have higher impact
    
    # Simple approach: weight current h2h_score by match recency in season
    # Later matches in season get higher H2H weight (teams know each other better)
    season_progress = df['matchday_normalized']  # 0 (early) to 1 (late season)
    
    # H2H with recency: base h2h_score boosted by season progress
    recency_weight = 0.5 + 0.5 * season_progress  # Range [0.5, 1.0]
    df['h2h_recent_weight'] = df['h2h_score'] * recency_weight
    
    # Ensure all new features are in [0,1] range
    for feature in ['elo_home_advantage', 'form_momentum', 'h2h_recent_weight']:
        df[feature] = np.clip(df[feature], 0, 1)
    
    logger.info("Feature statistics:")
    new_features = ['elo_home_advantage', 'form_momentum', 'h2h_recent_weight']
    for feature in new_features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        mean_val = df[feature].mean()
        logger.info(f"  {feature}: [{min_val:.3f}, {max_val:.3f}], mean={mean_val:.3f}")
    
    # Save enhanced dataset
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_file = f"data/processed/premier_league_ml_ready_v12_{timestamp}.csv"
    
    df.to_csv(output_file, index=False)
    logger.info(f"Enhanced dataset saved: {output_file}")
    
    # Update feature list for v1.2
    updated_features = [
        'form_diff_normalized',
        'elo_diff_normalized', 
        'h2h_score',
        'home_advantage',
        'matchday_normalized',
        'season_period_numeric',
        'shots_diff_normalized',
        'corners_diff_normalized',
        # New v1.2 features
        'elo_home_advantage',
        'form_momentum', 
        'h2h_recent_weight'
    ]
    
    # Save updated feature configuration
    features_v12_file = "config/features_v12.json"
    import json
    with open(features_v12_file, 'w') as f:
        json.dump(updated_features, f, indent=2)
    
    logger.info(f"Updated feature list saved: {features_v12_file}")
    logger.info(f"Total features: {len(updated_features)} (was 8, now 11)")
    
    logger.info("=== v1.2 Feature Enhancement Completed ===")
    
    return {
        'enhanced_dataset': output_file,
        'feature_config': features_v12_file,
        'new_features': new_features,
        'total_features': len(updated_features)
    }

if __name__ == "__main__":
    result = enhance_features_v12()
    print(f"Feature enhancement completed. New dataset: {result['enhanced_dataset']}")
    print(f"Total features: {result['total_features']} (+{len(result['new_features'])})")