#!/usr/bin/env python3
"""
UNDERSTAT xG DATA SCRAPER
Professional-grade scraper for Expected Goals data
Covers Premier League 2019-2025 seasons for v2.0 development

IMPORTANT: This scraper is for educational/research purposes only
Please respect Understat's terms of service and rate limits
"""
import requests
import json
import pandas as pd
import time
import sys
import os
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

class UnderstatScraper:
    """
    Professional Understat scraper with rate limiting and error handling
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://understat.com"
        self.rate_limit = 2  # seconds between requests
        
    def get_season_data(self, season):
        """
        Get all Premier League matches for a specific season
        """
        logger = setup_logging()
        logger.info(f"🔍 Scraping season {season}...")
        
        # Understat season format: 2019 for 2019-20 season
        season_year = int(season.split('-')[0])
        
        url = f"{self.base_url}/league/EPL/{season_year}"
        
        try:
            time.sleep(self.rate_limit)  # Rate limiting
            response = self.session.get(url)
            response.raise_for_status()
            
            # Extract JSON data from JavaScript
            # Understat embeds data in script tags like: var datesData = JSON.parse('...')
            html_content = response.text
            
            # Find the matches data - multiple possible patterns
            patterns = [
                r"var datesData = JSON\.parse\('(.+?)'\);",
                r"JSON\.parse\('([^']+)'\)",
            ]
            
            json_str = None
            for pattern in patterns:
                match = re.search(pattern, html_content)
                if match:
                    json_str = match.group(1)
                    logger.info(f"✅ Found data with pattern: {pattern}")
                    break
            
            if not json_str:
                logger.error(f"Could not find matches data for season {season}")
                return None
            
            # The data is hex-encoded, decode it
            try:
                # Convert hex escape sequences to actual characters
                decoded_bytes = bytes.fromhex(json_str.replace('\\x', ''))
                decoded_str = decoded_bytes.decode('utf-8')
                matches_data = json.loads(decoded_str)
            except:
                # Fallback to unicode escape decoding
                json_str = json_str.encode().decode('unicode_escape')
                matches_data = json.loads(json_str)
            
            logger.info(f"✅ Found {len(matches_data)} matches for season {season}")
            return matches_data
            
        except Exception as e:
            logger.error(f"❌ Error scraping season {season}: {e}")
            return None
    
    def process_match_data(self, matches_data, season):
        """
        Process raw match data into structured format
        """
        logger = setup_logging()
        logger.info(f"📊 Processing {len(matches_data)} matches for season {season}")
        
        processed_matches = []
        
        for match in matches_data:
            try:
                # Extract key information
                match_info = {
                    'Date': match['datetime'],
                    'Season': season,
                    'HomeTeam': match['h']['title'],
                    'AwayTeam': match['a']['title'], 
                    'HomeGoals': int(match['goals']['h']),
                    'AwayGoals': int(match['goals']['a']),
                    'HomeXG': float(match['xG']['h']) if match['xG']['h'] else 0.0,
                    'AwayXG': float(match['xG']['a']) if match['xG']['a'] else 0.0,
                    'UnderstatID': match['id']
                }
                
                # Determine result
                if match_info['HomeGoals'] > match_info['AwayGoals']:
                    match_info['FullTimeResult'] = 'H'
                elif match_info['HomeGoals'] < match_info['AwayGoals']:
                    match_info['FullTimeResult'] = 'A'
                else:
                    match_info['FullTimeResult'] = 'D'
                
                # Calculate xG differences and ratios
                match_info['XG_Diff'] = match_info['HomeXG'] - match_info['AwayXG']
                match_info['Total_XG'] = match_info['HomeXG'] + match_info['AwayXG']
                
                # Goals vs xG efficiency
                home_efficiency = match_info['HomeGoals'] / match_info['HomeXG'] if match_info['HomeXG'] > 0 else 0
                away_efficiency = match_info['AwayGoals'] / match_info['AwayXG'] if match_info['AwayXG'] > 0 else 0
                
                match_info['Home_GoalsVsXG'] = home_efficiency
                match_info['Away_GoalsVsXG'] = away_efficiency
                
                processed_matches.append(match_info)
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing match: {e}")
                continue
        
        logger.info(f"✅ Processed {len(processed_matches)} matches successfully")
        return processed_matches
    
    def scrape_all_seasons(self, seasons):
        """
        Scrape all specified seasons
        """
        logger = setup_logging()
        logger.info(f"🚀 Starting comprehensive scrape for seasons: {seasons}")
        
        all_matches = []
        failed_seasons = []
        
        for season in seasons:
            logger.info(f"\n📅 SEASON {season}")
            logger.info("-" * 40)
            
            # Get raw data
            matches_data = self.get_season_data(season)
            
            if matches_data is None:
                failed_seasons.append(season)
                continue
            
            # Process data
            processed_matches = self.process_match_data(matches_data, season)
            
            if processed_matches:
                all_matches.extend(processed_matches)
                logger.info(f"✅ Season {season}: {len(processed_matches)} matches added")
            else:
                failed_seasons.append(season)
                logger.error(f"❌ Season {season}: No matches processed")
        
        logger.info(f"\n🏁 SCRAPING COMPLETE")
        logger.info(f"Total matches scraped: {len(all_matches)}")
        logger.info(f"Successful seasons: {len(seasons) - len(failed_seasons)}/{len(seasons)}")
        
        if failed_seasons:
            logger.warning(f"Failed seasons: {failed_seasons}")
        
        return all_matches, failed_seasons
    
    def save_data(self, matches_data, filename=None):
        """
        Save scraped data to CSV
        """
        logger = setup_logging()
        
        if not matches_data:
            logger.error("❌ No data to save")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(matches_data)
        
        # Sort by date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
            filename = f"understat_xg_data_{timestamp}.csv"
        
        # Save to data directory
        os.makedirs('data/external', exist_ok=True)
        filepath = f"data/external/{filename}"
        
        df.to_csv(filepath, index=False)
        
        logger.info(f"💾 Data saved: {filepath}")
        logger.info(f"📊 Shape: {df.shape}")
        logger.info(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Basic statistics
        logger.info(f"\n📈 XG STATISTICS:")
        logger.info(f"Average Home xG: {df['HomeXG'].mean():.2f}")
        logger.info(f"Average Away xG: {df['AwayXG'].mean():.2f}")
        logger.info(f"Average Total xG: {df['Total_XG'].mean():.2f}")
        logger.info(f"Highest xG match: {df['Total_XG'].max():.2f}")
        
        return filepath

def main():
    """
    Main scraping pipeline for Premier League xG data
    """
    logger = setup_logging()
    logger.info("🎯 UNDERSTAT xG DATA SCRAPER")
    logger.info("=" * 60)
    logger.info("Scraping Premier League Expected Goals data")
    logger.info("Educational/Research use - Respect rate limits")
    logger.info("=" * 60)
    
    # Define seasons to scrape (matching our existing data)
    seasons_to_scrape = [
        '2019-2020',
        '2020-2021', 
        '2021-2022',
        '2022-2023',
        '2023-2024',
        '2024-2025'
    ]
    
    logger.info(f"Target seasons: {seasons_to_scrape}")
    
    # Initialize scraper
    scraper = UnderstatScraper()
    
    # Scrape all seasons
    all_matches, failed_seasons = scraper.scrape_all_seasons(seasons_to_scrape)
    
    if not all_matches:
        logger.error("❌ No data scraped - exiting")
        return False
    
    # Save data
    saved_file = scraper.save_data(all_matches)
    
    if saved_file:
        logger.info(f"\n🎉 SCRAPING SUCCESS")
        logger.info(f"File saved: {saved_file}")
        logger.info(f"Matches scraped: {len(all_matches)}")
        
        if failed_seasons:
            logger.warning(f"Note: Some seasons failed: {failed_seasons}")
            logger.info("You may need to re-run for failed seasons")
        
        logger.info(f"\n🔄 NEXT STEPS:")
        logger.info("1. Review scraped data quality")
        logger.info("2. Merge with existing Premier League dataset")
        logger.info("3. Engineer xG features for v2.0 model")
        logger.info("4. Train and evaluate xG-enhanced model")
        
        return True
    else:
        logger.error("❌ Failed to save data")
        return False

def test_single_season():
    """
    Test function to scrape just one season
    """
    logger = setup_logging()
    logger.info("🧪 TESTING SINGLE SEASON SCRAPE")
    
    scraper = UnderstatScraper()
    
    # Test with 2023-2024 season
    test_season = '2023-2024'
    matches_data = scraper.get_season_data(test_season)
    
    if matches_data:
        processed = scraper.process_match_data(matches_data, test_season)
        
        if processed:
            # Save test data
            test_file = scraper.save_data(processed, f"understat_test_{test_season.replace('-', '_')}.csv")
            logger.info(f"✅ Test successful: {test_file}")
            return True
    
    logger.error("❌ Test failed")
    return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Understat xG Data Scraper')
    parser.add_argument('--test', action='store_true', help='Test with single season')
    parser.add_argument('--season', type=str, help='Scrape specific season (e.g., 2023-2024)')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            success = test_single_season()
        elif args.season:
            logger = setup_logging()
            scraper = UnderstatScraper()
            matches_data = scraper.get_season_data(args.season)
            if matches_data:
                processed = scraper.process_match_data(matches_data, args.season)
                saved_file = scraper.save_data(processed)
                success = saved_file is not None
            else:
                success = False
        else:
            success = main()
        
        if success:
            print("✅ Scraping completed successfully")
        else:
            print("❌ Scraping failed")
            
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)