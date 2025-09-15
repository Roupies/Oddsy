#!/usr/bin/env python3
"""
Team Initialization Module for EPL 2025-26

Handles proper initialization of team features, especially for promoted teams
(Leeds, Sunderland, Burnley) based on historical performance and Championship data.
"""

import pandas as pd
import numpy as np
from datetime import datetime

class TeamInitializer:
    """Initialize team states for EPL 2025-26 season"""
    
    def __init__(self):
        # Promoted teams with their expected Elo initialization
        self.promoted_teams = {
            'Leeds': {
                'elo_init': 1591,  # Strong Championship winner + historical EPL
                'form_confidence': 0.65,  # Good Championship form
                'xg_efficiency_init': 0.55
            },
            'Sunderland': {
                'elo_init': 1398,  # Lower Championship finish
                'form_confidence': 0.45,  # Moderate form
                'xg_efficiency_init': 0.45
            },
            'Burnley': {
                'elo_init': 1520,  # Recent EPL experience + immediate return
                'form_confidence': 0.60,  # Bounced back quickly
                'xg_efficiency_init': 0.52
            }
        }
        
        # EPL average Elo for reference
        self.epl_average_elo = 1500
        
    def get_team_initial_state(self, team_name, historical_data):
        """Get initial state for a team based on historical data"""
        
        if team_name in self.promoted_teams:
            return self._initialize_promoted_team(team_name, historical_data)
        else:
            return self._initialize_established_team(team_name, historical_data)
            
    def _initialize_promoted_team(self, team_name, historical_data):
        """Initialize promoted team with Championship-based estimates"""
        team_config = self.promoted_teams[team_name]
        
        # Get any historical EPL data if available
        historical_matches = historical_data[
            (historical_data['HomeTeam'] == team_name) |
            (historical_data['AwayTeam'] == team_name)
        ]
        
        if len(historical_matches) > 0:
            # Use most recent EPL data if available
            recent_matches = historical_matches.tail(10)
            base_features = self._extract_features_from_matches(team_name, recent_matches)
        else:
            # Use Championship-based initialization
            base_features = {
                'elo_rating': team_config['elo_init'],
                'form_rating': team_config['form_confidence'],
                'xg_efficiency': team_config['xg_efficiency_init'],
                'goals_scored_avg': 1.2,  # Conservative estimate
                'goals_conceded_avg': 1.1
            }
        
        return {
            'team': team_name,
            'status': 'promoted',
            'initialization_source': 'championship_estimate',
            'features': base_features,
            'confidence_level': team_config['form_confidence']
        }
        
    def _initialize_established_team(self, team_name, historical_data):
        """Initialize established EPL team from recent historical data"""
        
        # Get recent matches (last season)
        team_matches = historical_data[
            ((historical_data['HomeTeam'] == team_name) |
             (historical_data['AwayTeam'] == team_name)) &
            (historical_data['Season'] == '2024-2025')  # Most recent season
        ]
        
        if len(team_matches) == 0:
            # Fallback to any available data
            team_matches = historical_data[
                (historical_data['HomeTeam'] == team_name) |
                (historical_data['AwayTeam'] == team_name)
            ].tail(20)  # Last 20 matches
        
        features = self._extract_features_from_matches(team_name, team_matches)
        
        return {
            'team': team_name,
            'status': 'established', 
            'initialization_source': 'historical_data',
            'features': features,
            'confidence_level': 0.8  # High confidence for established teams
        }
        
    def _extract_features_from_matches(self, team_name, matches):
        """Extract key features from team's historical matches"""
        
        if len(matches) == 0:
            return self._get_default_features()
            
        # Separate home and away matches
        home_matches = matches[matches['HomeTeam'] == team_name]
        away_matches = matches[matches['AwayTeam'] == team_name]
        
        # Calculate basic stats
        total_matches = len(matches)
        
        # Form calculation (simplified)
        form_values = []
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team_name:
                form_val = match.get('form_diff_normalized', 0.5)
                if form_val > 0.5:  # Home advantage considered
                    form_values.append(form_val)
                else:
                    form_values.append(1 - form_val)  # Flip for away disadvantage
            else:
                form_val = match.get('form_diff_normalized', 0.5)
                form_values.append(1 - form_val)  # Away team perspective
        
        avg_form = np.mean(form_values) if form_values else 0.5
        
        # Elo estimation (simplified)
        elo_values = []
        for _, match in matches.iterrows():
            if match['HomeTeam'] == team_name:
                elo_diff = match.get('elo_diff_normalized', 0.5)
                elo_values.append(self.epl_average_elo + (elo_diff - 0.5) * 200)
            else:
                elo_diff = match.get('elo_diff_normalized', 0.5) 
                elo_values.append(self.epl_average_elo - (elo_diff - 0.5) * 200)
        
        avg_elo = np.mean(elo_values) if elo_values else self.epl_average_elo
        
        # xG efficiency
        xg_eff_home = matches['home_xg_eff_10'].mean() if 'home_xg_eff_10' in matches else 0.5
        xg_eff_away = matches['away_xg_eff_10'].mean() if 'away_xg_eff_10' in matches else 0.5
        
        return {
            'elo_rating': max(1200, min(1800, avg_elo)),  # Reasonable bounds
            'form_rating': max(0.2, min(0.8, avg_form)),
            'xg_efficiency': (xg_eff_home + xg_eff_away) / 2,
            'matches_analyzed': total_matches,
            'goals_scored_avg': 1.3,  # Default estimates
            'goals_conceded_avg': 1.2
        }
        
    def _get_default_features(self):
        """Default feature values when no data available"""
        return {
            'elo_rating': self.epl_average_elo,
            'form_rating': 0.5,
            'xg_efficiency': 0.5,
            'matches_analyzed': 0,
            'goals_scored_avg': 1.25,
            'goals_conceded_avg': 1.25
        }
        
    def initialize_all_teams(self, historical_data):
        """Initialize all EPL teams for 2025-26 season"""
        
        # Get unique teams from latest season or calendar
        latest_season = historical_data[
            historical_data['Season'] == historical_data['Season'].max()
        ]
        
        all_teams = set(
            list(latest_season['HomeTeam'].unique()) +
            list(latest_season['AwayTeam'].unique())
        )
        
        team_states = {}
        
        print("🏟️  Initializing EPL 2025-26 teams...")
        
        for team in sorted(all_teams):
            team_state = self.get_team_initial_state(team, historical_data)
            team_states[team] = team_state
            
            status_emoji = "🆕" if team_state['status'] == 'promoted' else "🏛️"
            confidence = team_state['confidence_level']
            elo = team_state['features']['elo_rating']
            
            print(f"   {status_emoji} {team:<15} | Elo: {elo:4.0f} | Confidence: {confidence:.2f}")
            
        print(f"\n✅ Initialized {len(team_states)} teams")
        print(f"   - Promoted: {len([t for t in team_states.values() if t['status'] == 'promoted'])}")
        print(f"   - Established: {len([t for t in team_states.values() if t['status'] == 'established'])}")
        
        return team_states
        
    def create_match_features(self, home_team, away_team, team_states, gameweek=1):
        """Create feature vector for a specific match"""
        
        home_state = team_states.get(home_team, {'features': self._get_default_features()})
        away_state = team_states.get(away_team, {'features': self._get_default_features()})
        
        home_features = home_state['features']
        away_features = away_state['features']
        
        # Calculate relative features (home vs away)
        elo_diff = (home_features['elo_rating'] - away_features['elo_rating']) / 400
        elo_diff_normalized = max(0, min(1, (elo_diff + 1) / 2))  # Normalize to [0,1]
        
        form_diff = home_features['form_rating'] - away_features['form_rating']
        form_diff_normalized = max(0, min(1, (form_diff + 1) / 2))
        
        # Create complete feature vector
        features = {
            'form_diff_normalized': form_diff_normalized,
            'elo_diff_normalized': elo_diff_normalized,
            'h2h_score': 0.5,  # Neutral for new season
            'matchday_normalized': (gameweek - 1) / 37,  # Normalize gameweek
            'shots_diff_normalized': 0.5,  # Neutral start
            'corners_diff_normalized': 0.5,  # Neutral start  
            'market_entropy_norm': 0.7,  # Higher uncertainty for new season
            'home_xg_eff_10': home_features['xg_efficiency'],
            'away_xg_eff_10': away_features['xg_efficiency'],
            'away_goals_sum_5': away_features['goals_scored_avg'] * 5  # 5-match proxy
        }
        
        # Add metadata
        match_info = {
            'home_team': home_team,
            'away_team': away_team,
            'home_status': home_state.get('status', 'established'),
            'away_status': away_state.get('status', 'established'),
            'involves_promoted': (
                home_state.get('status') == 'promoted' or 
                away_state.get('status') == 'promoted'
            ),
            'gameweek': gameweek
        }
        
        return features, match_info