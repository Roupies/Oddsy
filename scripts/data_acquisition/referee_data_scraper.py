#!/usr/bin/env python3
"""
Referee Data Scraper - The Official Influence Factor
Scrape referee statistics and disciplinary patterns from Premier League data

Strategy: Build referee profiles showing disciplinary tendencies and biases
Focus: Cards per match, penalty decisions, home advantage patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import requests
from bs4 import BeautifulSoup
import time
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RefereeDataScraper:
    """Scrape and process referee statistics from various sources."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Common Premier League referees (2019-2024)
        self.known_referees = [
            'Michael Oliver', 'Anthony Taylor', 'Paul Tierney', 'Andre Marriner',
            'Mike Dean', 'Martin Atkinson', 'Craig Pawson', 'Jonathan Moss',
            'Chris Kavanagh', 'David Coote', 'Stuart Attwell', 'Peter Bankes',
            'Andy Madley', 'Darren England', 'Tony Harrington', 'John Brooks',
            'Graham Scott', 'Simon Hooper', 'Jarred Gillett', 'Michael Salisbury'
        ]
    
    def extract_referee_from_raw_data(self):
        """Extract referee information from our existing raw data files."""
        
        logger.info("Extracting referee data from existing raw files...")
        
        raw_files = [
            'data/raw/football_data_backup/football_data_2019_20.csv',
            'data/raw/football_data_backup/football_data_2020_21.csv', 
            'data/raw/football_data_backup/football_data_2021_22.csv',
            'data/raw/football_data_backup/football_data_2022_23.csv',
            'data/raw/football_data_backup/football_data_2023_24.csv'
        ]
        
        all_referee_data = []
        
        for file_path in raw_files:
            try:
                logger.info(f"Processing {file_path}")
                df = pd.read_csv(file_path)
                
                # Check if referee column exists
                if 'Referee' in df.columns:
                    # Extract relevant columns
                    referee_matches = df[['Date', 'HomeTeam', 'AwayTeam', 'Referee', 'FTR', 
                                         'HY', 'AY', 'HR', 'AR']].copy()
                    
                    # Add season info
                    season = file_path.split('_')[-2] + '_' + file_path.split('_')[-1].replace('.csv', '')
                    referee_matches['Season'] = season
                    
                    # Convert date
                    referee_matches['Date'] = pd.to_datetime(referee_matches['Date'], format='%d/%m/%Y', errors='coerce')
                    
                    all_referee_data.append(referee_matches)
                    
                else:
                    logger.warning(f"No Referee column in {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
        
        if all_referee_data:
            combined_data = pd.concat(all_referee_data, ignore_index=True)
            logger.info(f"Extracted referee data: {combined_data.shape}")
            return combined_data
        else:
            logger.error("No referee data extracted")
            return None
    
    def calculate_referee_statistics(self, referee_data):
        """Calculate comprehensive referee statistics and tendencies."""
        
        logger.info("Calculating referee statistics...")
        
        # Clean referee names and filter out invalid data
        referee_data = referee_data.dropna(subset=['Referee'])
        referee_data = referee_data[referee_data['Referee'].str.len() > 3]  # Filter very short names
        
        referee_stats = {}
        
        for referee in referee_data['Referee'].unique():
            ref_matches = referee_data[referee_data['Referee'] == referee].copy()
            
            if len(ref_matches) < 5:  # Skip referees with very few matches
                continue
            
            # Basic match statistics
            total_matches = len(ref_matches)
            
            # Result distribution
            home_wins = (ref_matches['FTR'] == 'H').sum()
            draws = (ref_matches['FTR'] == 'D').sum()
            away_wins = (ref_matches['FTR'] == 'A').sum()
            
            home_win_rate = home_wins / total_matches
            draw_rate = draws / total_matches
            away_win_rate = away_wins / total_matches
            
            # Card statistics
            total_yellow_cards = ref_matches['HY'].fillna(0).sum() + ref_matches['AY'].fillna(0).sum()
            total_red_cards = ref_matches['HR'].fillna(0).sum() + ref_matches['AR'].fillna(0).sum()
            
            yellow_cards_per_match = total_yellow_cards / total_matches
            red_cards_per_match = total_red_cards / total_matches
            
            # Home bias analysis
            home_yellow_cards = ref_matches['HY'].fillna(0).sum()
            away_yellow_cards = ref_matches['AY'].fillna(0).sum()
            home_red_cards = ref_matches['HR'].fillna(0).sum()
            away_red_cards = ref_matches['AR'].fillna(0).sum()
            
            # Calculate disciplinary bias (positive = more cards for away team)
            if total_yellow_cards > 0:
                yellow_bias = (away_yellow_cards - home_yellow_cards) / total_yellow_cards
            else:
                yellow_bias = 0
            
            if total_red_cards > 0:
                red_bias = (away_red_cards - home_red_cards) / total_red_cards
            else:
                red_bias = 0
            
            referee_stats[referee] = {
                'total_matches': total_matches,
                'home_win_rate': home_win_rate,
                'draw_rate': draw_rate,
                'away_win_rate': away_win_rate,
                'yellow_cards_per_match': yellow_cards_per_match,
                'red_cards_per_match': red_cards_per_match,
                'yellow_card_bias': yellow_bias,  # Positive = more cards for away
                'red_card_bias': red_bias,
                'seasons_active': ref_matches['Season'].nunique()
            }
        
        logger.info(f"Calculated stats for {len(referee_stats)} referees")
        return referee_stats
    
    def create_referee_database(self):
        """Create comprehensive referee database with statistics."""
        
        # Extract referee data from existing files
        referee_data = self.extract_referee_from_raw_data()
        
        if referee_data is None:
            return None
        
        # Calculate referee statistics
        referee_stats = self.calculate_referee_statistics(referee_data)
        
        # Convert to DataFrame for easier analysis
        stats_df = pd.DataFrame(referee_stats).T
        stats_df = stats_df.reset_index().rename(columns={'index': 'referee_name'})
        
        # Calculate league averages for normalization
        league_avg_home_rate = stats_df['home_win_rate'].mean()
        league_avg_yellow_per_match = stats_df['yellow_cards_per_match'].mean()
        league_avg_red_per_match = stats_df['red_cards_per_match'].mean()
        
        # Create normalized indices
        stats_df['home_bias_index'] = stats_df['home_win_rate'] / league_avg_home_rate
        stats_df['disciplinary_index'] = stats_df['yellow_cards_per_match'] / league_avg_yellow_per_match
        stats_df['severity_index'] = stats_df['red_cards_per_match'] / league_avg_red_per_match
        
        # Save referee database
        output_path = Path('data/external/referee_database_2025_09_07.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats_df.to_csv(output_path, index=False)
        logger.info(f"Saved referee database to {output_path}")
        
        # Save raw match data with referees
        referee_matches_path = Path('data/external/referee_matches_2025_09_07.csv')
        referee_data.to_csv(referee_matches_path, index=False)
        logger.info(f"Saved referee match data to {referee_matches_path}")
        
        return stats_df, referee_data
    
    def generate_referee_analysis_report(self, stats_df):
        """Generate comprehensive referee analysis report."""
        
        print("\\n" + "="*80)
        print("⚖️ REFEREE DATABASE ANALYSIS")
        print("="*80)
        
        print(f"\\n📊 DATABASE OVERVIEW:")
        print(f"   • Total referees: {len(stats_df)}")
        print(f"   • Total matches covered: {stats_df['total_matches'].sum()}")
        print(f"   • Average matches per referee: {stats_df['total_matches'].mean():.1f}")
        print(f"   • Most active referee: {stats_df.loc[stats_df['total_matches'].idxmax(), 'referee_name']} ({stats_df['total_matches'].max()} matches)")
        
        print(f"\\n🏠 HOME ADVANTAGE PATTERNS:")
        league_home_rate = stats_df['home_win_rate'].mean()
        print(f"   • League average home win rate: {league_home_rate:.1%}")
        
        # Most biased referees
        most_home_biased = stats_df.nlargest(3, 'home_bias_index')
        most_away_biased = stats_df.nsmallest(3, 'home_bias_index')
        
        print(f"   • Most home-biased referees:")
        for _, ref in most_home_biased.iterrows():
            print(f"     - {ref['referee_name']}: {ref['home_win_rate']:.1%} (index: {ref['home_bias_index']:.2f})")
        
        print(f"   • Most away-biased referees:")
        for _, ref in most_away_biased.iterrows():
            print(f"     - {ref['referee_name']}: {ref['home_win_rate']:.1%} (index: {ref['home_bias_index']:.2f})")
        
        print(f"\\n📋 DISCIPLINARY PATTERNS:")
        league_yellow_avg = stats_df['yellow_cards_per_match'].mean()
        league_red_avg = stats_df['red_cards_per_match'].mean()
        
        print(f"   • League average yellow cards per match: {league_yellow_avg:.2f}")
        print(f"   • League average red cards per match: {league_red_avg:.2f}")
        
        # Most strict referees
        most_strict = stats_df.nlargest(3, 'disciplinary_index')
        most_lenient = stats_df.nsmallest(3, 'disciplinary_index')
        
        print(f"   • Strictest referees:")
        for _, ref in most_strict.iterrows():
            print(f"     - {ref['referee_name']}: {ref['yellow_cards_per_match']:.2f} cards/match (index: {ref['disciplinary_index']:.2f})")
        
        print(f"   • Most lenient referees:")
        for _, ref in most_lenient.iterrows():
            print(f"     - {ref['referee_name']}: {ref['yellow_cards_per_match']:.2f} cards/match (index: {ref['disciplinary_index']:.2f})")
        
        print(f"\\n🎯 PREDICTIVE INSIGHTS:")
        
        # Identify extreme referees that could impact predictions
        high_variance_refs = stats_df[
            (stats_df['home_bias_index'] > 1.2) | 
            (stats_df['home_bias_index'] < 0.8) |
            (stats_df['disciplinary_index'] > 1.3) |
            (stats_df['disciplinary_index'] < 0.7)
        ]
        
        print(f"   • High-impact referees: {len(high_variance_refs)} ({len(high_variance_refs)/len(stats_df)*100:.1f}%)")
        print(f"   • These referees show significant deviation from league averages")
        print(f"   • Their patterns could provide predictive value")
        
        # Feature engineering potential
        print(f"\\n🚀 FEATURE ENGINEERING POTENTIAL:")
        print(f"   • referee_disciplinary_index: Yellow cards tendency vs league avg")
        print(f"   • referee_home_bias_index: Home win rate vs league avg")
        print(f"   • referee_severity_index: Red card tendency vs league avg")
        print(f"   • referee_card_bias: Home vs away disciplinary balance")
        
        return high_variance_refs

def main():
    """Execute referee data scraping and analysis."""
    
    logger.info("🚀 Starting Referee Data Collection...")
    
    scraper = RefereeDataScraper()
    
    # Create referee database
    result = scraper.create_referee_database()
    
    if result:
        stats_df, referee_data = result
        
        # Generate analysis report
        high_impact_refs = scraper.generate_referee_analysis_report(stats_df)
        
        logger.info("✅ Referee Data Collection Complete!")
        return stats_df, referee_data, high_impact_refs
    else:
        logger.error("❌ Referee Data Collection Failed!")
        return None, None, None

if __name__ == "__main__":
    main()