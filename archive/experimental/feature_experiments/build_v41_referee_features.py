#!/usr/bin/env python3
"""
v4.1 Referee Features - The Official Influence Factor
Integrate referee tendencies and biases into match predictions

Strategy: Referee behavior directly impacts match outcomes through cards/penalties
Focus: Disciplinary indices, home bias, and severity patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_referee_database():
    """Load referee statistics database."""
    
    referee_db_path = Path('data/external/referee_database_2025_09_07.csv')
    if not referee_db_path.exists():
        logger.error(f"Referee database not found: {referee_db_path}")
        return None
    
    referee_db = pd.read_csv(referee_db_path)
    logger.info(f"Loaded referee database: {len(referee_db)} referees")
    
    # Create referee lookup dictionary
    referee_lookup = {}
    for _, row in referee_db.iterrows():
        referee_lookup[row['referee_name']] = {
            'disciplinary_index': row['disciplinary_index'],
            'home_bias_index': row['home_bias_index'],
            'severity_index': row['severity_index'],
            'yellow_card_bias': row['yellow_card_bias'],
            'total_matches': row['total_matches']
        }
    
    return referee_lookup

def load_referee_match_data():
    """Load historical referee-match assignments."""
    
    referee_matches_path = Path('data/external/referee_matches_2025_09_07.csv')
    if not referee_matches_path.exists():
        logger.error(f"Referee matches not found: {referee_matches_path}")
        return None
    
    referee_matches = pd.read_csv(referee_matches_path)
    referee_matches['Date'] = pd.to_datetime(referee_matches['Date'])
    
    logger.info(f"Loaded referee match assignments: {len(referee_matches)} matches")
    return referee_matches

def create_referee_features(df, referee_lookup, referee_matches):
    """Create referee influence features for each match."""
    
    logger.info("Creating referee influence features...")
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Initialize referee features
    referee_features = [
        'referee_disciplinary_index',  # Cards tendency vs league avg
        'referee_home_bias_index',     # Home win rate vs league avg
        'referee_severity_index',      # Red card tendency vs league avg
        'referee_yellow_card_bias',    # Home vs away card bias
        'referee_experience_factor',   # Total matches experience
        'referee_strictness_category', # Categorical strictness level
        'referee_bias_category'        # Categorical bias level
    ]
    
    for feature in referee_features:
        df[feature] = np.nan
    
    # Match referee data with main dataset
    matches_found = 0
    
    for i, row in df.iterrows():
        match_date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Find referee for this match
        referee_match = referee_matches[
            (referee_matches['Date'] == match_date) &
            (referee_matches['HomeTeam'] == home_team) &
            (referee_matches['AwayTeam'] == away_team)
        ]
        
        if len(referee_match) > 0:
            referee_name = referee_match.iloc[0]['Referee']
            
            if referee_name in referee_lookup:
                ref_stats = referee_lookup[referee_name]
                
                # Core referee indices
                df.loc[i, 'referee_disciplinary_index'] = ref_stats['disciplinary_index']
                df.loc[i, 'referee_home_bias_index'] = ref_stats['home_bias_index']
                df.loc[i, 'referee_severity_index'] = ref_stats['severity_index']
                df.loc[i, 'referee_yellow_card_bias'] = ref_stats['yellow_card_bias']
                
                # Experience factor (log scale for diminishing returns)
                experience = ref_stats['total_matches']
                df.loc[i, 'referee_experience_factor'] = np.log(experience + 1) / np.log(200)  # Normalized
                
                matches_found += 1
    
    logger.info(f"Matched referee data for {matches_found}/{len(df)} matches ({matches_found/len(df)*100:.1f}%)")
    
    # Create categorical features for missing data handling
    df['referee_strictness_category'] = pd.cut(
        df['referee_disciplinary_index'], 
        bins=[0, 0.8, 1.2, float('inf')], 
        labels=['lenient', 'average', 'strict'],
        include_lowest=True
    ).astype(str)
    
    df['referee_bias_category'] = pd.cut(
        df['referee_home_bias_index'], 
        bins=[0, 0.85, 1.15, float('inf')], 
        labels=['away_biased', 'neutral', 'home_biased'],
        include_lowest=True
    ).astype(str)
    
    # Handle missing values with league averages
    league_avg_disciplinary = 1.0  # By definition (normalized)
    league_avg_home_bias = 1.0
    league_avg_severity = 1.0
    
    df['referee_disciplinary_index'] = df['referee_disciplinary_index'].fillna(league_avg_disciplinary)
    df['referee_home_bias_index'] = df['referee_home_bias_index'].fillna(league_avg_home_bias)
    df['referee_severity_index'] = df['referee_severity_index'].fillna(league_avg_severity)
    df['referee_yellow_card_bias'] = df['referee_yellow_card_bias'].fillna(0.0)
    df['referee_experience_factor'] = df['referee_experience_factor'].fillna(0.5)
    
    # Replace 'nan' strings with 'unknown' for categorical features
    df['referee_strictness_category'] = df['referee_strictness_category'].replace('nan', 'unknown')
    df['referee_bias_category'] = df['referee_bias_category'].replace('nan', 'unknown')
    
    logger.info("Referee influence features created")
    return df

def create_referee_derived_features(df):
    """Create derived features from referee indices."""
    
    # Referee impact scenarios
    df['referee_high_card_risk'] = (df['referee_disciplinary_index'] > 1.2).astype(int)
    df['referee_home_advantage_boost'] = (df['referee_home_bias_index'] > 1.1).astype(int)
    df['referee_away_advantage_boost'] = (df['referee_home_bias_index'] < 0.9).astype(int)
    
    # Combined referee impact score
    # Positive = favors home team, Negative = favors away team
    df['referee_home_impact_score'] = (
        (df['referee_home_bias_index'] - 1.0) * 2.0 +  # Home bias effect
        (df['referee_yellow_card_bias']) * 0.5  # Card bias effect (positive = more away cards)
    )
    
    # Referee game flow impact
    df['referee_disruption_factor'] = (
        df['referee_disciplinary_index'] * 0.7 +
        df['referee_severity_index'] * 0.3
    )
    
    # Experience-weighted indices
    exp_weight = df['referee_experience_factor']
    df['referee_disciplinary_index_weighted'] = df['referee_disciplinary_index'] * exp_weight
    df['referee_bias_index_weighted'] = df['referee_home_bias_index'] * exp_weight
    
    logger.info("Derived referee features created")
    return df

def encode_categorical_referee_features(df):
    """Encode categorical referee features for ML."""
    
    # One-hot encode categorical features
    strictness_dummies = pd.get_dummies(df['referee_strictness_category'], prefix='ref_strictness')
    bias_dummies = pd.get_dummies(df['referee_bias_category'], prefix='ref_bias')
    
    # Add to dataframe
    df = pd.concat([df, strictness_dummies, bias_dummies], axis=1)
    
    # Drop original categorical columns
    df = df.drop(columns=['referee_strictness_category', 'referee_bias_category'])
    
    logger.info("Categorical referee features encoded")
    return df

def build_v41_referee_dataset():
    """Build v4.1 dataset with referee influence features."""
    
    # Load v4.0 fatigue dataset as base
    base_path = Path('data/processed/v40_fatigue_features_2025_09_07.csv')
    output_path = Path('data/processed/v41_referee_features_2025_09_07.csv')
    
    logger.info(f"Loading v4.0 baseline from {base_path}")
    df = pd.read_csv(base_path)
    
    logger.info(f"Input dataset shape: {df.shape}")
    
    # Load referee data
    referee_lookup = load_referee_database()
    referee_matches = load_referee_match_data()
    
    if referee_lookup is None or referee_matches is None:
        logger.error("Failed to load referee data")
        return None
    
    # Create referee features
    df = create_referee_features(df, referee_lookup, referee_matches)
    
    # Create derived features
    df = create_referee_derived_features(df)
    
    # Encode categorical features
    df = encode_categorical_referee_features(df)
    
    # Save enhanced dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Saved v4.1 dataset to {output_path}")
    logger.info(f"Output dataset shape: {df.shape}")
    
    # Analyze new features
    referee_features = [col for col in df.columns if 'referee' in col or 'ref_' in col]
    
    print("\n⚖️ NEW REFEREE FEATURES:")
    for feature in sorted(referee_features[:15]):  # Show first 15
        if df[feature].notna().sum() > 0:
            if df[feature].dtype in ['int64', 'float64']:
                mean_val = df[feature].mean()
                std_val = df[feature].std()
                print(f"  • {feature}: mean={mean_val:.3f}, std={std_val:.3f}")
            else:
                unique_vals = df[feature].nunique()
                print(f"  • {feature}: {unique_vals} unique values")
    
    print(f"\n📊 FEATURE SUMMARY:")
    print(f"  • Total new referee features: {len(referee_features)}")
    print(f"  • Referee data coverage: {(df['referee_disciplinary_index'] != 1.0).sum()}/{len(df)} matches")
    
    # Referee influence insights
    print(f"\n🎯 REFEREE INFLUENCE INSIGHTS:")
    
    # High-impact scenarios
    high_card_risk = (df['referee_high_card_risk'] == 1).sum()
    home_boost = (df['referee_home_advantage_boost'] == 1).sum()
    away_boost = (df['referee_away_advantage_boost'] == 1).sum()
    
    print(f"  • High card risk matches: {high_card_risk} ({high_card_risk/len(df)*100:.1f}%)")
    print(f"  • Home-biased referee matches: {home_boost} ({home_boost/len(df)*100:.1f}%)")
    print(f"  • Away-biased referee matches: {away_boost} ({away_boost/len(df)*100:.1f}%)")
    
    # Impact score distribution
    avg_home_impact = df['referee_home_impact_score'].mean()
    impact_range = df['referee_home_impact_score'].max() - df['referee_home_impact_score'].min()
    
    print(f"  • Average referee home impact: {avg_home_impact:.3f}")
    print(f"  • Referee impact range: {impact_range:.3f}")
    print(f"  • High disruption matches: {(df['referee_disruption_factor'] > 1.2).sum()} ({(df['referee_disruption_factor'] > 1.2).sum()/len(df)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    logger.info("🚀 Starting v4.1 Referee Features build...")
    df = build_v41_referee_dataset()
    if df is not None:
        logger.info("✅ v4.1 Referee Features build complete!")
    else:
        logger.error("❌ v4.1 build failed")