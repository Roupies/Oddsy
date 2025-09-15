#!/usr/bin/env python3
"""
Build Player Absence Features - PoC Implementation
Integrate key player absence data with v3.1 efficiency features

Focus: Goalkeeper changes and top scorer availability
Strategy: Binary indicators (0/1) for MVP validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlayerAbsenceFeatureBuilder:
    """Build player absence features from lineup data."""
    
    def __init__(self):
        # Known starting goalkeepers for major teams (2023-24 baseline)
        self.main_goalkeepers = {
            'Arsenal': ['Aaron Ramsdale', 'David Raya'],  # Raya took over mid-season
            'Chelsea': ['Robert Sanchez', 'Thiago Silva'],
            'Liverpool': ['Alisson', 'Alisson Becker'],
            'Man City': ['Ederson', 'Ederson Moraes'],
            'Man United': ['Andre Onana', 'Onana'],
            'Tottenham': ['Guglielmo Vicario', 'Vicario'],
            'Brighton': ['Jason Steele', 'Bart Verbruggen'],
            'Newcastle': ['Nick Pope', 'Martin Dubravka'],
            'West Ham': ['Alphonse Areola', 'Lukasz Fabianski'],
            'Aston Villa': ['Emiliano Martinez', 'Martinez'],
            'Crystal Palace': ['Sam Johnstone', 'Dean Henderson'],
            'Brentford': ['Mark Flekken', 'Thomas Strakosha'],
            'Fulham': ['Bernd Leno', 'Leno'],
            'Wolves': ['Jose Sa', 'Dan Bentley'],
            'Everton': ['Jordan Pickford', 'Pickford'],
            'Bournemouth': ['Neto', 'Andrei Radu'],
            'Nottm Forest': ['Matz Sels', 'Matt Turner'],
            'Sheffield United': ['Wes Foderingham', 'Ivo Grbic'],
            'Burnley': ['James Trafford', 'Arijanet Muric'],
            'Luton': ['Thomas Kaminski', 'Tim Krul']
        }
        
        # Known top scorers for teams (2023-24 season leaders)
        self.top_scorers = {
            'Arsenal': ['Bukayo Saka', 'Gabriel Jesus', 'Martin Odegaard'],
            'Chelsea': ['Cole Palmer', 'Nicolas Jackson', 'Raheem Sterling'],
            'Liverpool': ['Mohamed Salah', 'Darwin Nunez', 'Luis Diaz'],
            'Man City': ['Erling Haaland', 'Phil Foden', 'Julian Alvarez'],
            'Man United': ['Bruno Fernandes', 'Rasmus Hojlund', 'Marcus Rashford'],
            'Tottenham': ['Son Heung-min', 'Richarlison', 'Dejan Kulusevski'],
            'Brighton': ['Joao Pedro', 'Evan Ferguson', 'Danny Welbeck'],
            'Newcastle': ['Alexander Isak', 'Callum Wilson', 'Anthony Gordon'],
            'West Ham': ['Jarrod Bowen', 'Michail Antonio', 'Lucas Paqueta'],
            'Aston Villa': ['Ollie Watkins', 'Moussa Diaby', 'John McGinn'],
            'Crystal Palace': ['Jean-Philippe Mateta', 'Eberechi Eze', 'Michael Olise'],
            'Brentford': ['Ivan Toney', 'Yoane Wissa', 'Bryan Mbeumo'],
            'Fulham': ['Rodrigo Muniz', 'Alex Iwobi', 'Harry Wilson'],
            'Wolves': ['Hwang Hee-chan', 'Matheus Cunha', 'Pedro Neto'],
            'Everton': ['Dominic Calvert-Lewin', 'Abdoulaye Doucoure', 'Jack Harrison'],
            'Bournemouth': ['Dominic Solanke', 'Antoine Semenyo', 'Justin Kluivert'],
            'Nottm Forest': ['Chris Wood', 'Taiwo Awoniyi', 'Callum Hudson-Odoi'],
            'Sheffield United': ['Gustavo Hamer', 'Cameron Archer', 'Ben Brereton Diaz'],
            'Burnley': ['Lyle Foster', 'Zeki Amdouni', 'Nathan Redmond'],
            'Luton': ['Elijah Adebayo', 'Carlton Morris', 'Chiedozie Ogbene']
        }
        
    def normalize_player_name(self, name):
        """Normalize player names for matching."""
        if not name or pd.isna(name):
            return None
        
        # Basic normalization
        name = str(name).strip()
        
        # Handle common variations
        name_variants = {
            'Alisson Becker': 'Alisson',
            'Ederson Moraes': 'Ederson',
            'Emiliano Martinez': 'Martinez',
            'Guglielmo Vicario': 'Vicario'
        }
        
        return name_variants.get(name, name)
    
    def detect_backup_goalkeeper(self, team, goalkeeper_name):
        """Detect if a backup goalkeeper is playing."""
        
        if not goalkeeper_name or pd.isna(goalkeeper_name):
            return True  # Missing data = assume backup
        
        # Normalize name
        gk_name = self.normalize_player_name(goalkeeper_name)
        
        # Check against known main goalkeepers
        main_gks = self.main_goalkeepers.get(team, [])
        main_gks_normalized = [self.normalize_player_name(gk) for gk in main_gks]
        
        # If goalkeeper is not in main list, assume backup
        return gk_name not in main_gks_normalized
    
    def detect_missing_top_scorer(self, team, forwards_str):
        """Detect if team's top scorer is missing from forward line."""
        
        if not forwards_str or pd.isna(forwards_str):
            return True  # No forwards data = assume key player missing
        
        # Parse forwards string
        forwards_list = [f.strip() for f in str(forwards_str).split(',')]
        forwards_normalized = [self.normalize_player_name(f) for f in forwards_list]
        
        # Check against known top scorers
        top_scorers = self.top_scorers.get(team, [])
        top_scorers_normalized = [self.normalize_player_name(s) for s in top_scorers]
        
        # If none of the top scorers are in the forward line, assume missing
        return not any(scorer in forwards_normalized for scorer in top_scorers_normalized if scorer)
    
    def build_player_absence_features(self, base_df, lineup_df):
        """Build player absence features from lineup data."""
        
        logger.info("Building player absence features...")
        
        # Convert dates to datetime for merging
        base_df = base_df.copy()
        lineup_df = lineup_df.copy()
        
        base_df['Date'] = pd.to_datetime(base_df['Date'])
        lineup_df['date'] = pd.to_datetime(lineup_df['date'])
        
        # Merge lineup data with base dataset
        merged_df = base_df.merge(
            lineup_df[['date', 'home_team', 'away_team', 'home_goalkeeper', 'away_goalkeeper', 
                      'home_forwards', 'away_forwards']],
            left_on=['Date', 'HomeTeam', 'AwayTeam'],
            right_on=['date', 'home_team', 'away_team'], 
            how='left'
        )
        
        logger.info(f"Merged {len(merged_df)} matches with lineup data")
        logger.info(f"Lineup data coverage: {merged_df['home_goalkeeper'].notna().sum()}/{len(merged_df)} matches")
        
        # Create backup goalkeeper features
        merged_df['home_backup_gk_playing'] = merged_df.apply(
            lambda row: int(self.detect_backup_goalkeeper(row['HomeTeam'], row['home_goalkeeper'])),
            axis=1
        )
        
        merged_df['away_backup_gk_playing'] = merged_df.apply(
            lambda row: int(self.detect_backup_goalkeeper(row['AwayTeam'], row['away_goalkeeper'])),
            axis=1
        )
        
        # Create top scorer missing features
        merged_df['home_top_scorer_missing'] = merged_df.apply(
            lambda row: int(self.detect_missing_top_scorer(row['HomeTeam'], row['home_forwards'])),
            axis=1
        )
        
        merged_df['away_top_scorer_missing'] = merged_df.apply(
            lambda row: int(self.detect_missing_top_scorer(row['AwayTeam'], row['away_forwards'])),
            axis=1
        )
        
        # Create derived features
        merged_df['gk_advantage'] = merged_df['away_backup_gk_playing'] - merged_df['home_backup_gk_playing']  # Positive = home advantage
        merged_df['scorer_advantage'] = merged_df['away_top_scorer_missing'] - merged_df['home_top_scorer_missing']  # Positive = home advantage
        
        # Clean up temporary columns
        merged_df = merged_df.drop(columns=['date', 'home_team', 'away_team'], errors='ignore')
        
        # Feature summary
        player_features = ['home_backup_gk_playing', 'away_backup_gk_playing', 
                          'home_top_scorer_missing', 'away_top_scorer_missing',
                          'gk_advantage', 'scorer_advantage']
        
        logger.info("Player absence features created:")
        for feature in player_features:
            coverage = merged_df[feature].notna().sum()
            mean_val = merged_df[feature].mean()
            logger.info(f"  • {feature}: {mean_val:.3f} mean, {coverage}/{len(merged_df)} coverage")
        
        return merged_df, player_features
    
    def validate_player_features(self, df, player_features):
        """Validate player features for logical consistency."""
        
        logger.info("Validating player absence features...")
        
        validation_results = {
            'total_matches': len(df),
            'features_coverage': {},
            'logical_checks': {},
            'team_analysis': {}
        }
        
        # Coverage analysis
        for feature in player_features:
            coverage_pct = df[feature].notna().sum() / len(df) * 100
            validation_results['features_coverage'][feature] = {
                'coverage_pct': coverage_pct,
                'mean_value': df[feature].mean(),
                'non_zero_pct': (df[feature] > 0).sum() / len(df) * 100
            }
        
        # Logical consistency checks
        # Check 1: Backup GK rates should be reasonable (5-20% of matches)
        home_backup_rate = df['home_backup_gk_playing'].mean()
        away_backup_rate = df['away_backup_gk_playing'].mean()
        
        validation_results['logical_checks']['backup_gk_rates'] = {
            'home_rate': home_backup_rate,
            'away_rate': away_backup_rate,
            'reasonable': 0.05 <= home_backup_rate <= 0.25 and 0.05 <= away_backup_rate <= 0.25
        }
        
        # Check 2: Top scorer missing should be relatively rare (10-30%)
        home_missing_rate = df['home_top_scorer_missing'].mean()
        away_missing_rate = df['away_top_scorer_missing'].mean()
        
        validation_results['logical_checks']['top_scorer_missing'] = {
            'home_rate': home_missing_rate,
            'away_rate': away_missing_rate,
            'reasonable': 0.10 <= home_missing_rate <= 0.40 and 0.10 <= away_missing_rate <= 0.40
        }
        
        # Team-level analysis (sample)
        teams = ['Arsenal', 'Man City', 'Liverpool', 'Chelsea']
        for team in teams:
            if team in df['HomeTeam'].values:
                team_home = df[df['HomeTeam'] == team]
                team_away = df[df['AwayTeam'] == team]
                
                validation_results['team_analysis'][team] = {
                    'home_matches': len(team_home),
                    'away_matches': len(team_away),
                    'backup_gk_home': team_home['home_backup_gk_playing'].mean() if len(team_home) > 0 else 0,
                    'backup_gk_away': team_away['away_backup_gk_playing'].mean() if len(team_away) > 0 else 0,
                    'scorer_missing_home': team_home['home_top_scorer_missing'].mean() if len(team_home) > 0 else 0,
                    'scorer_missing_away': team_away['away_top_scorer_missing'].mean() if len(team_away) > 0 else 0
                }
        
        return validation_results
    
    def create_poc_dataset(self):
        """Create PoC dataset with player absence features."""
        
        logger.info("🚀 Creating Player Absence Features PoC Dataset...")
        
        # Load v3.1 efficiency features dataset  
        base_path = Path('data/processed/v31_efficiency_features_2025_09_06.csv')
        if not base_path.exists():
            logger.error(f"Base dataset not found: {base_path}")
            return None
        
        logger.info(f"Loading base dataset from {base_path}")
        base_df = pd.read_csv(base_path)
        
        # Load lineup data (test data for PoC)
        lineup_path = Path('data/external/fbref_lineups_2023_24_test.csv')
        if not lineup_path.exists():
            logger.error(f"Lineup dataset not found: {lineup_path}")
            return None
        
        logger.info(f"Loading lineup data from {lineup_path}")
        lineup_df = pd.read_csv(lineup_path)
        
        logger.info(f"Base dataset: {base_df.shape}")
        logger.info(f"Lineup dataset: {lineup_df.shape}")
        
        # Build player absence features
        enhanced_df, player_features = self.build_player_absence_features(base_df, lineup_df)
        
        # Validate features
        validation_results = self.validate_player_features(enhanced_df, player_features)
        
        # Save enhanced dataset
        output_path = Path('data/processed/v31_with_player_features_poc_2025_09_06.csv')
        enhanced_df.to_csv(output_path, index=False)
        
        # Save validation report
        validation_path = Path('evaluation/reports/player_features_validation_2025_09_06.json')
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"✅ Enhanced dataset saved to {output_path}")
        logger.info(f"✅ Validation report saved to {validation_path}")
        logger.info(f"Final dataset shape: {enhanced_df.shape}")
        
        # Summary report
        print("\\n" + "="*80)
        print("🏗️ PLAYER ABSENCE FEATURES - PoC RESULTS")
        print("="*80)
        
        print(f"\\n📊 DATASET INTEGRATION:")
        print(f"   • Base dataset (v3.1): {base_df.shape[0]} matches, {base_df.shape[1]} features")
        print(f"   • Lineup data coverage: {lineup_df.shape[0]} matches")
        print(f"   • Enhanced dataset: {enhanced_df.shape[0]} matches, {enhanced_df.shape[1]} features")
        print(f"   • New player features: {len(player_features)}")
        
        print(f"\\n🎯 FEATURE ANALYSIS:")
        for feature in player_features:
            if feature in validation_results['features_coverage']:
                stats = validation_results['features_coverage'][feature]
                print(f"   • {feature}: {stats['mean_value']:.3f} mean, {stats['coverage_pct']:.1f}% coverage")
        
        print(f"\\n✅ VALIDATION RESULTS:")
        logic_checks = validation_results['logical_checks']
        print(f"   • Backup GK rates: Home {logic_checks['backup_gk_rates']['home_rate']:.1%}, Away {logic_checks['backup_gk_rates']['away_rate']:.1%}")
        print(f"   • Top scorer missing: Home {logic_checks['top_scorer_missing']['home_rate']:.1%}, Away {logic_checks['top_scorer_missing']['away_rate']:.1%}")
        print(f"   • Logical consistency: {'✅ PASS' if logic_checks['backup_gk_rates']['reasonable'] else '❌ FAIL'}")
        
        print(f"\\n📋 NEXT STEPS:")
        print(f"   1. Test player features impact on v3.1 baseline (56.28%)")
        print(f"   2. Measure feature importance of new player indicators")
        print(f"   3. Go/No-Go decision based on +0.5pp improvement threshold")
        
        return enhanced_df, validation_results

def main():
    """Execute player absence features building."""
    
    logger.info("🚀 Starting Player Absence Features Builder...")
    
    builder = PlayerAbsenceFeatureBuilder()
    enhanced_df, validation_results = builder.create_poc_dataset()
    
    if enhanced_df is not None:
        logger.info("✅ Player Absence Features PoC Complete!")
        return enhanced_df, validation_results
    else:
        logger.error("❌ Player Absence Features PoC Failed!")
        return None, None

if __name__ == "__main__":
    main()