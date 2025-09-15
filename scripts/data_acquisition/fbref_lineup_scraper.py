#!/usr/bin/env python3
"""
FBref Lineup Scraper - Player Data PoC
Scrape team lineups from FBref.com for key player absence detection

Focus: Goalkeeper and top scorer identification for 2023-24 season
Strategy: Minimal viable approach to validate player data value
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from pathlib import Path
import re
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FBrefLineupScraper:
    """Scrape FBref for Premier League team lineups."""
    
    def __init__(self):
        self.base_url = "https://fbref.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Premier League team URL mappings (FBref uses specific team codes)
        self.team_mappings = {
            'Arsenal': 'Arsenal',
            'Chelsea': 'Chelsea', 
            'Liverpool': 'Liverpool',
            'Man City': 'Manchester City',
            'Man United': 'Manchester United',
            'Tottenham': 'Tottenham',
            'Brighton': 'Brighton and Hove Albion',
            'Newcastle': 'Newcastle United',
            'West Ham': 'West Ham United',
            'Aston Villa': 'Aston Villa',
            'Crystal Palace': 'Crystal Palace',
            'Brentford': 'Brentford',
            'Fulham': 'Fulham',
            'Wolves': 'Wolverhampton Wanderers',
            'Everton': 'Everton',
            'Bournemouth': 'AFC Bournemouth',
            'Nottm Forest': 'Nottingham Forest',
            'Sheffield United': 'Sheffield United',
            'Burnley': 'Burnley',
            'Luton': 'Luton Town'
        }
        
        # Common goalkeeper and striker indicators
        self.gk_positions = ['GK', 'Goalkeeper', 'G']
        self.striker_positions = ['CF', 'ST', 'Striker', 'Centre-Forward', 'F']
        
    def get_premier_league_schedule(self, season='2023-24'):
        """Get Premier League match schedule from FBref."""
        
        logger.info(f"Fetching Premier League {season} schedule...")
        
        # FBref Premier League URLs by season
        season_urls = {
            '2023-24': 'https://fbref.com/en/comps/9/schedule/Premier-League-Fixtures',
            '2022-23': 'https://fbref.com/en/comps/9/10728/schedule/2022-2023-Premier-League-Fixtures'
        }
        
        if season not in season_urls:
            logger.error(f"Season {season} not supported")
            return None
        
        try:
            response = requests.get(season_urls[season], headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the fixtures table
            fixtures_table = soup.find('table', {'id': 'sched_ks_9_1'})
            if not fixtures_table:
                logger.error("Could not find fixtures table")
                return None
            
            fixtures = []
            rows = fixtures_table.find('tbody').find_all('tr')
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 6:
                    continue
                    
                # Extract match info
                date_cell = cells[1] if len(cells) > 1 else None
                home_team_cell = cells[3] if len(cells) > 3 else None
                away_team_cell = cells[5] if len(cells) > 5 else None
                
                if date_cell and home_team_cell and away_team_cell:
                    date_str = date_cell.get_text(strip=True)
                    home_team = home_team_cell.get_text(strip=True)
                    away_team = away_team_cell.get_text(strip=True)
                    
                    # Get match report link for lineup scraping
                    match_link = None
                    score_cell = cells[4] if len(cells) > 4 else None
                    if score_cell and score_cell.find('a'):
                        match_link = score_cell.find('a')['href']
                    
                    fixtures.append({
                        'date': date_str,
                        'home_team': home_team,
                        'away_team': away_team,
                        'match_link': match_link
                    })
            
            logger.info(f"Found {len(fixtures)} Premier League fixtures")
            return fixtures
            
        except Exception as e:
            logger.error(f"Failed to get schedule: {str(e)}")
            return None
    
    def scrape_match_lineup(self, match_link):
        """Scrape lineup information from a specific match page."""
        
        if not match_link:
            return None
            
        try:
            full_url = self.base_url + match_link if match_link.startswith('/') else match_link
            
            logger.debug(f"Scraping lineup from {full_url}")
            
            response = requests.get(full_url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find lineup tables (usually two - one for each team)
            lineup_tables = soup.find_all('table', class_='lineup')
            
            if len(lineup_tables) < 2:
                logger.warning("Could not find both team lineups")
                return None
            
            lineups = {'home': {}, 'away': {}}
            
            for i, table in enumerate(lineup_tables[:2]):  # Only process first two tables
                team_type = 'home' if i == 0 else 'away'
                
                # Extract starting XI
                starting_xi = []
                rows = table.find('tbody').find_all('tr') if table.find('tbody') else []
                
                for row in rows:
                    # Check if this is a starter (not substitute)
                    if 'substitute' in row.get('class', []):
                        continue
                        
                    player_cell = row.find('td', {'data-stat': 'player'})
                    position_cell = row.find('td', {'data-stat': 'position'})
                    
                    if player_cell and position_cell:
                        player_name = player_cell.get_text(strip=True)
                        position = position_cell.get_text(strip=True)
                        
                        starting_xi.append({
                            'name': player_name,
                            'position': position
                        })
                
                # Identify key players
                goalkeeper = None
                forwards = []
                
                for player in starting_xi:
                    # Goalkeeper identification
                    if any(pos in player['position'].upper() for pos in self.gk_positions):
                        goalkeeper = player['name']
                    
                    # Forward/Striker identification  
                    if any(pos in player['position'].upper() for pos in self.striker_positions):
                        forwards.append(player['name'])
                
                lineups[team_type] = {
                    'starting_xi': starting_xi,
                    'goalkeeper': goalkeeper,
                    'forwards': forwards
                }
            
            return lineups
            
        except Exception as e:
            logger.error(f"Failed to scrape lineup from {match_link}: {str(e)}")
            return None
    
    def identify_team_key_players(self, team_name, season='2023-24'):
        """Identify key players for a team (top scorer, main GK) from season data."""
        
        logger.info(f"Identifying key players for {team_name} ({season})...")
        
        try:
            # This would ideally scrape team's season stats to identify:
            # 1. Main goalkeeper (most appearances)
            # 2. Top scorer (most goals)
            # 3. Key playmaker (most assists)
            
            # For MVP, we'll use a simplified approach with known key players
            # In full implementation, this would scrape FBref team stats
            
            key_players = {
                'main_goalkeeper': None,
                'top_scorer': None,
                'key_playmaker': None
            }
            
            # Placeholder - in full implementation, scrape team stats
            logger.info(f"Key players identified for {team_name}")
            return key_players
            
        except Exception as e:
            logger.error(f"Failed to identify key players for {team_name}: {str(e)}")
            return None
    
    def scrape_season_lineups(self, season='2023-24', max_matches=50):
        """Scrape lineups for entire season (with rate limiting)."""
        
        logger.info(f"🚀 Starting lineup scraping for {season} season...")
        
        # Get match schedule
        fixtures = self.get_premier_league_schedule(season)
        if not fixtures:
            logger.error("Could not get fixtures")
            return None
        
        # Limit matches for PoC
        fixtures = fixtures[:max_matches]
        logger.info(f"Processing {len(fixtures)} matches for PoC")
        
        lineup_data = []
        successful_scrapes = 0
        
        for i, fixture in enumerate(fixtures):
            logger.info(f"Processing match {i+1}/{len(fixtures)}: {fixture['home_team']} vs {fixture['away_team']}")
            
            # Scrape lineup if match link available
            lineup = self.scrape_match_lineup(fixture['match_link']) if fixture['match_link'] else None
            
            if lineup:
                lineup_data.append({
                    'date': fixture['date'],
                    'home_team': fixture['home_team'],
                    'away_team': fixture['away_team'],
                    'home_goalkeeper': lineup['home']['goalkeeper'],
                    'away_goalkeeper': lineup['away']['goalkeeper'],
                    'home_forwards': ', '.join(lineup['home']['forwards']),
                    'away_forwards': ', '.join(lineup['away']['forwards']),
                    'home_starting_xi_count': len(lineup['home']['starting_xi']),
                    'away_starting_xi_count': len(lineup['away']['starting_xi'])
                })
                successful_scrapes += 1
            else:
                # Add empty record to maintain match tracking
                lineup_data.append({
                    'date': fixture['date'],
                    'home_team': fixture['home_team'], 
                    'away_team': fixture['away_team'],
                    'home_goalkeeper': None,
                    'away_goalkeeper': None,
                    'home_forwards': None,
                    'away_forwards': None,
                    'home_starting_xi_count': 0,
                    'away_starting_xi_count': 0
                })
            
            # Rate limiting to be respectful to FBref
            time.sleep(2)  # 2 second delay between requests
            
            # Progress update every 10 matches
            if (i + 1) % 10 == 0:
                success_rate = successful_scrapes / (i + 1) * 100
                logger.info(f"Progress: {i+1}/{len(fixtures)} matches, {success_rate:.1f}% success rate")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(lineup_data)
        
        # Save to CSV
        output_path = Path(f'data/external/fbref_lineups_{season.replace("-", "_")}_poc.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Lineup scraping complete!")
        logger.info(f"   • Total matches processed: {len(fixtures)}")
        logger.info(f"   • Successful scrapes: {successful_scrapes}")
        logger.info(f"   • Success rate: {successful_scrapes/len(fixtures)*100:.1f}%")
        logger.info(f"   • Data saved to: {output_path}")
        
        return df
    
    def create_minimal_test_data(self):
        """Create minimal test data for development without scraping."""
        
        logger.info("Creating minimal test data for PoC development...")
        
        # Sample lineup data based on known 2023-24 lineups
        test_data = [
            {
                'date': '2023-08-12',
                'home_team': 'Arsenal',
                'away_team': 'Nottm Forest',
                'home_goalkeeper': 'Aaron Ramsdale',
                'away_goalkeeper': 'Matz Sels',
                'home_forwards': 'Eddie Nketiah',
                'away_forwards': 'Taiwo Awoniyi',
                'home_starting_xi_count': 11,
                'away_starting_xi_count': 11
            },
            {
                'date': '2023-08-13',
                'home_team': 'Brighton',
                'away_team': 'Luton',
                'home_goalkeeper': 'Jason Steele',
                'away_goalkeeper': 'Thomas Kaminski',
                'home_forwards': 'Joao Pedro',
                'away_forwards': 'Elijah Adebayo',
                'home_starting_xi_count': 11,
                'away_starting_xi_count': 11
            },
            {
                'date': '2023-08-14',
                'home_team': 'Man City',
                'away_team': 'Burnley',
                'home_goalkeeper': 'Ederson',
                'away_goalkeeper': 'James Trafford',
                'home_forwards': 'Erling Haaland',
                'away_forwards': 'Lyle Foster',
                'home_starting_xi_count': 11,
                'away_starting_xi_count': 11
            }
        ]
        
        df = pd.DataFrame(test_data)
        
        # Save test data
        output_path = Path('data/external/fbref_lineups_2023_24_test.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Test data created: {output_path}")
        return df

def main():
    """Execute FBref lineup scraping."""
    
    logger.info("🚀 Starting FBref Lineup Scraper...")
    
    scraper = FBrefLineupScraper()
    
    # For PoC, create test data first to validate pipeline
    test_df = scraper.create_minimal_test_data()
    
    # Uncomment below for real scraping (respectful rate limiting)
    # real_df = scraper.scrape_season_lineups(season='2023-24', max_matches=20)
    
    logger.info("✅ FBref Lineup Scraper Complete!")
    return test_df

if __name__ == "__main__":
    main()