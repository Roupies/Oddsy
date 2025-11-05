#!/usr/bin/env python3
"""
Fixture Service for Real Kickoff Times
=====================================
Reads EPL_25_26_Full_Calendar.csv and provides real fixture data
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytz

logger = logging.getLogger(__name__)

class FixtureService:
    """Service to handle fixture data from calendar CSV"""
    
    def __init__(self, calendar_path: Optional[str] = None):
        if calendar_path:
            self.calendar_path = Path(calendar_path)
        else:
            # Default path relative to project root
            self.calendar_path = Path(__file__).parent.parent.parent / "data" / "EPL_25_26_Full_Calendar.csv"
        
        self._fixtures_cache: Dict[int, List[Dict]] = {}
        self._load_fixtures()
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team names for consistent matching"""
        # Handle common variations
        name_mapping = {
            "Spurs": "Tottenham Hotspur",
            "Tottenham": "Tottenham Hotspur", 
            "Man City": "Manchester City",
            "Man Utd": "Manchester United",
            "Nott'm Forest": "Nottingham Forest",
            "Brighton": "Brighton and Hove Albion",
        }
        
        normalized = name_mapping.get(team_name, team_name)
        return normalized.strip()
    
    def _parse_kickoff_time(self, date_str: str) -> datetime:
        """Parse DD/MM/YYYY HH:MM format to UTC datetime"""
        try:
            # Parse the DD/MM/YYYY HH:MM format
            naive_dt = datetime.strptime(date_str.strip(), "%d/%m/%Y %H:%M")
            
            # Assume GMT timezone for EPL matches
            gmt = pytz.timezone('GMT')
            gmt_dt = gmt.localize(naive_dt)
            
            # Convert to UTC
            utc_dt = gmt_dt.astimezone(pytz.UTC)
            
            return utc_dt
            
        except ValueError as e:
            logger.error(f"Failed to parse date '{date_str}': {e}")
            # Return a default time if parsing fails
            return datetime(2025, 11, 8, 15, 0, tzinfo=pytz.UTC)
    
    def _load_fixtures(self) -> None:
        """Load fixtures from CSV file"""
        if not self.calendar_path.exists():
            logger.error(f"Calendar file not found: {self.calendar_path}")
            return
        
        try:
            with open(self.calendar_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    round_num = int(row['Round Number'])
                    
                    if round_num not in self._fixtures_cache:
                        self._fixtures_cache[round_num] = []
                    
                    # Parse kickoff time
                    kickoff_utc = self._parse_kickoff_time(row['Date'])
                    
                    fixture = {
                        'match_number': int(row['Match Number']),
                        'round': round_num,
                        'home_team': self._normalize_team_name(row['Home Team']),
                        'away_team': self._normalize_team_name(row['Away Team']),
                        'location': row['Location'],
                        'kickoff_utc': kickoff_utc.isoformat(),
                        'kickoff_local': row['Date'],  # Keep original for reference
                        'result': row.get('Result', '').strip()
                    }
                    
                    self._fixtures_cache[round_num].append(fixture)
            
            logger.info(f"Loaded fixtures for {len(self._fixtures_cache)} gameweeks")
            
        except Exception as e:
            logger.error(f"Failed to load fixtures: {e}")
    
    def get_fixtures_for_gameweek(self, gameweek: int) -> List[Dict]:
        """Get all fixtures for a specific gameweek"""
        return self._fixtures_cache.get(gameweek, [])
    
    def get_fixture_by_teams(self, gameweek: int, home_team: str, away_team: str) -> Optional[Dict]:
        """Get specific fixture by teams"""
        fixtures = self.get_fixtures_for_gameweek(gameweek)
        
        # Normalize input team names
        home_normalized = self._normalize_team_name(home_team)
        away_normalized = self._normalize_team_name(away_team)
        
        for fixture in fixtures:
            if (fixture['home_team'] == home_normalized and 
                fixture['away_team'] == away_normalized):
                return fixture
        
        return None
    
    def get_available_gameweeks(self) -> List[int]:
        """Get list of available gameweeks"""
        return sorted(self._fixtures_cache.keys())

# Global service instance
_fixture_service: Optional[FixtureService] = None

def get_fixture_service() -> FixtureService:
    """Get or create the global fixture service instance"""
    global _fixture_service
    if _fixture_service is None:
        _fixture_service = FixtureService()
    return _fixture_service