import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import setup_logging

def extract_odds_features():
    """
    Extract and engineer bookmaker odds features from football_data_backup files.
    Convert odds to implied probabilities and create market consensus features.
    
    This represents v2.0 development: integrating market intelligence with 
    our v1.2 baseline features for significant performance improvement.
    """
    
    logger = setup_logging()
    logger.info("=== v2.0 Odds Data Extraction Started ===")
    
    # Load all backup files
    backup_dir = "data/raw/football_data_backup/"
    backup_files = [
        "football_data_2019_20.csv",
        "football_data_2020_21.csv", 
        "football_data_2021_22.csv",
        "football_data_2022_23.csv",
        "football_data_2023_24.csv",
        "football_data_2024_25.csv"
    ]
    
    all_odds_data = []
    
    for file in backup_files:
        filepath = os.path.join(backup_dir, file)
        logger.info(f"Processing {file}...")
        
        df = pd.read_csv(filepath)
        logger.info(f"  Loaded: {df.shape[0]} matches")
        
        # Extract season from filename
        season = file.replace('football_data_', '').replace('.csv', '')
        df['Season'] = season
        
        all_odds_data.append(df)
    
    # Combine all seasons
    combined_df = pd.concat(all_odds_data, ignore_index=True)
    logger.info(f"Combined dataset: {combined_df.shape}")
    
    # Convert Date to datetime
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%d/%m/%Y')
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    logger.info("Engineering odds features...")
    
    # ===== CORE ODDS FEATURES =====
    
    # 1. Market Consensus (Average Odds)
    logger.info("1. Creating market consensus features...")
    
    # Convert odds to implied probabilities
    combined_df['market_prob_home'] = 1 / combined_df['AvgH']
    combined_df['market_prob_draw'] = 1 / combined_df['AvgD'] 
    combined_df['market_prob_away'] = 1 / combined_df['AvgA']
    
    # Normalize probabilities (remove overround)
    total_prob = (combined_df['market_prob_home'] + 
                  combined_df['market_prob_draw'] + 
                  combined_df['market_prob_away'])
    
    combined_df['market_prob_home_norm'] = combined_df['market_prob_home'] / total_prob
    combined_df['market_prob_draw_norm'] = combined_df['market_prob_draw'] / total_prob
    combined_df['market_prob_away_norm'] = combined_df['market_prob_away'] / total_prob
    
    # Market strength differential
    combined_df['market_home_advantage'] = (combined_df['market_prob_home_norm'] - 
                                          combined_df['market_prob_away_norm'])
    
    # Normalize to [0,1]
    min_val = combined_df['market_home_advantage'].min()
    max_val = combined_df['market_home_advantage'].max()
    combined_df['market_home_advantage_norm'] = ((combined_df['market_home_advantage'] - min_val) / 
                                               (max_val - min_val))
    
    # 2. Market Uncertainty (Implied Volatility)
    logger.info("2. Creating market uncertainty features...")
    
    # Entropy measure: how uncertain is the market?
    epsilon = 1e-10  # Avoid log(0)
    combined_df['market_entropy'] = -(
        combined_df['market_prob_home_norm'] * np.log(combined_df['market_prob_home_norm'] + epsilon) +
        combined_df['market_prob_draw_norm'] * np.log(combined_df['market_prob_draw_norm'] + epsilon) +
        combined_df['market_prob_away_norm'] * np.log(combined_df['market_prob_away_norm'] + epsilon)
    )
    
    # Normalize entropy to [0,1]
    max_entropy = np.log(3)  # Maximum entropy for 3 outcomes
    combined_df['market_entropy_norm'] = combined_df['market_entropy'] / max_entropy
    
    # Draw probability (market expects tight game)
    combined_df['market_draw_prob_norm'] = combined_df['market_prob_draw_norm']
    
    # 3. Bookmaker Disagreement (Market Efficiency)
    logger.info("3. Creating bookmaker disagreement features...")
    
    # Standard deviation of odds across bookmakers (for Home wins)
    bookmaker_home_odds = combined_df[['B365H', 'BWH', 'PSH', 'WHH', 'VCH']].fillna(method='ffill', axis=1)
    combined_df['odds_disagreement_home'] = bookmaker_home_odds.std(axis=1, skipna=True)
    
    # Normalize disagreement
    max_disagreement = combined_df['odds_disagreement_home'].max()
    if max_disagreement > 0:
        combined_df['odds_disagreement_norm'] = (combined_df['odds_disagreement_home'] / max_disagreement)
    else:
        combined_df['odds_disagreement_norm'] = 0
    
    # 4. Over/Under Market Features
    logger.info("4. Creating over/under market features...")
    
    # Goals expectation from over/under 2.5 market
    if 'Avg>2.5' in combined_df.columns and 'Avg<2.5' in combined_df.columns:
        combined_df['over25_prob'] = 1 / combined_df['Avg>2.5']
        combined_df['under25_prob'] = 1 / combined_df['Avg<2.5']
        
        # Normalize O/U probabilities
        ou_total = combined_df['over25_prob'] + combined_df['under25_prob']
        combined_df['over25_prob_norm'] = combined_df['over25_prob'] / ou_total
        
        # Expected goals proxy (higher over25 prob = more goals expected)
        combined_df['market_expected_goals'] = combined_df['over25_prob_norm']
    else:
        combined_df['market_expected_goals'] = 0.5  # Neutral
    
    # 5. Premium Bookmaker Features (Pinnacle - sharp money)
    logger.info("5. Creating premium bookmaker features...")
    
    if all(col in combined_df.columns for col in ['PSH', 'PSD', 'PSA']):
        # Pinnacle probabilities (sharp money indicator)
        combined_df['pinnacle_prob_home'] = 1 / combined_df['PSH']
        combined_df['pinnacle_prob_draw'] = 1 / combined_df['PSD']
        combined_df['pinnacle_prob_away'] = 1 / combined_df['PSA']
        
        # Normalize Pinnacle probabilities
        pinnacle_total = (combined_df['pinnacle_prob_home'] + 
                         combined_df['pinnacle_prob_draw'] + 
                         combined_df['pinnacle_prob_away'])
        
        combined_df['pinnacle_home_advantage'] = ((combined_df['pinnacle_prob_home'] - 
                                                 combined_df['pinnacle_prob_away']) / pinnacle_total)
        
        # Normalize to [0,1]
        min_pin = combined_df['pinnacle_home_advantage'].min()
        max_pin = combined_df['pinnacle_home_advantage'].max()
        if max_pin > min_pin:
            combined_df['pinnacle_home_advantage_norm'] = ((combined_df['pinnacle_home_advantage'] - min_pin) / 
                                                         (max_pin - min_pin))
        else:
            combined_df['pinnacle_home_advantage_norm'] = 0.5
    else:
        combined_df['pinnacle_home_advantage_norm'] = combined_df['market_home_advantage_norm']
    
    # Handle missing values
    odds_features = [
        'market_home_advantage_norm', 'market_entropy_norm', 'market_draw_prob_norm',
        'odds_disagreement_norm', 'market_expected_goals', 'pinnacle_home_advantage_norm'
    ]
    
    for feature in odds_features:
        combined_df[feature] = combined_df[feature].fillna(0.5)  # Neutral for missing
        combined_df[feature] = np.clip(combined_df[feature], 0, 1)  # Ensure [0,1] range
    
    # Feature statistics
    logger.info("Odds features statistics:")
    for feature in odds_features:
        min_val = combined_df[feature].min()
        max_val = combined_df[feature].max()
        mean_val = combined_df[feature].mean()
        logger.info(f"  {feature}: [{min_val:.3f}, {max_val:.3f}], mean={mean_val:.3f}")
    
    # Select core columns for output
    output_columns = [
        'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Season',
        # Match statistics
        'HS', 'AS', 'HST', 'AST', 'HC', 'AC',
        # Odds features  
        'market_home_advantage_norm', 'market_entropy_norm', 'market_draw_prob_norm',
        'odds_disagreement_norm', 'market_expected_goals', 'pinnacle_home_advantage_norm',
        # Raw odds for reference
        'AvgH', 'AvgD', 'AvgA'
    ]
    
    # Filter to columns that exist
    existing_columns = [col for col in output_columns if col in combined_df.columns]
    output_df = combined_df[existing_columns].copy()
    
    # Save extracted data
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_file = f"data/processed/odds_features_extracted_{timestamp}.csv"
    
    output_df.to_csv(output_file, index=False)
    logger.info(f"Odds features saved: {output_file}")
    logger.info(f"Output shape: {output_df.shape}")
    
    # Data quality check
    logger.info("Data quality validation:")
    logger.info(f"  Date range: {output_df['Date'].min()} to {output_df['Date'].max()}")
    logger.info(f"  Missing values: {output_df.isnull().sum().sum()}")
    logger.info(f"  Seasons covered: {sorted(output_df['Season'].unique())}")
    
    logger.info("=== v2.0 Odds Data Extraction Completed ===")
    
    return {
        'output_file': output_file,
        'features_created': odds_features,
        'total_matches': len(output_df),
        'seasons': sorted(output_df['Season'].unique()),
        'feature_stats': {feat: {
            'min': float(output_df[feat].min()),
            'max': float(output_df[feat].max()), 
            'mean': float(output_df[feat].mean())
        } for feat in odds_features if feat in output_df.columns}
    }

if __name__ == "__main__":
    result = extract_odds_features()
    print(f"\nv2.0 Odds extraction completed!")
    print(f"Output: {result['output_file']}")
    print(f"Features created: {len(result['features_created'])}")
    print(f"Matches processed: {result['total_matches']}")
    print(f"Seasons: {result['seasons']}")