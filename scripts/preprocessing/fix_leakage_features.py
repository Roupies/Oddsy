import pandas as pd
import numpy as np
from datetime import datetime

class OddsyFixedFeatureEngineering:
    """
    Version corrigée sans data leakage + nouvelles features
    """
    
    def __init__(self):
        self.df = None
        
    def load_data(self, filepath):
        """Load cleaned dataset"""
        print("🔧 FIXING DATA LEAKAGE + ADDING NEW FEATURES")
        print("="*50)
        self.df = pd.read_csv(filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        print(f"Loaded {len(self.df)} matches from {self.df['Date'].min()} to {self.df['Date'].max()}")
        return self
    
    def add_temporal_features(self):
        """Add temporal features (matchday, season period)"""
        print("\n📅 ADDING TEMPORAL FEATURES...")
        
        # Matchday normalized (0-1)
        self.df['matchday_normalized'] = (self.df['MatchWeek'] - 1) / 37  # 38 matchdays in EPL
        
        # Season period (early/mid/late)  
        self.df['season_period'] = self.df['MatchWeek'].apply(lambda x: 
            'early' if x <= 12 else 'mid' if x <= 26 else 'late'
        )
        
        # Encode season period as numeric
        period_map = {'early': 0, 'mid': 0.5, 'late': 1}
        self.df['season_period_numeric'] = self.df['season_period'].map(period_map)
        
        print(f"✅ Added matchday_normalized: [{self.df['matchday_normalized'].min():.2f}, {self.df['matchday_normalized'].max():.2f}]")
        print(f"✅ Added season_period_numeric: early={len(self.df[self.df['season_period']=='early'])}, mid={len(self.df[self.df['season_period']=='mid'])}, late={len(self.df[self.df['season_period']=='late'])}")
        
        return self
    
    def calculate_rolling_match_stats(self, window=5):
        """Calculate rolling averages for match stats (NO LEAKAGE)"""
        print(f"\n📊 CALCULATING ROLLING MATCH STATS (window={window}, NO LEAKAGE)...")
        
        # Initialize lists
        home_shots_avg = []
        away_shots_avg = []
        home_corners_avg = []
        away_corners_avg = []
        home_possession_proxy = []
        away_possession_proxy = []
        
        for idx, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam'] 
            current_date = row['Date']
            
            # Get rolling stats BEFORE current match (no leakage)
            def get_team_rolling_stats(team, exclude_date):
                # Home matches for this team
                home_matches = self.df[
                    (self.df['HomeTeam'] == team) & 
                    (self.df['Date'] < exclude_date)
                ].tail(window//2 + 1)
                
                # Away matches for this team
                away_matches = self.df[
                    (self.df['AwayTeam'] == team) & 
                    (self.df['Date'] < exclude_date)  
                ].tail(window//2 + 1)
                
                # Combine and get most recent
                all_matches = pd.concat([home_matches, away_matches]).sort_values('Date').tail(window)
                
                if len(all_matches) == 0:
                    return 0, 0, 0.5  # neutral defaults
                
                # Calculate averages
                shots = []
                corners = []
                possession_proxies = []
                
                for _, match in all_matches.iterrows():
                    if match['HomeTeam'] == team:  # Team was home
                        shots.append(match['HomeTeamShots'])
                        corners.append(match['HomeTeamCorners'])
                        # Possession proxy: shots ratio
                        total_shots = match['HomeTeamShots'] + match['AwayTeamShots']
                        poss_proxy = match['HomeTeamShots'] / total_shots if total_shots > 0 else 0.5
                        possession_proxies.append(poss_proxy)
                    else:  # Team was away
                        shots.append(match['AwayTeamShots'])
                        corners.append(match['AwayTeamCorners'])
                        # Possession proxy: shots ratio  
                        total_shots = match['HomeTeamShots'] + match['AwayTeamShots']
                        poss_proxy = match['AwayTeamShots'] / total_shots if total_shots > 0 else 0.5
                        possession_proxies.append(poss_proxy)
                
                avg_shots = np.mean(shots) if shots else 0
                avg_corners = np.mean(corners) if corners else 0
                avg_poss_proxy = np.mean(possession_proxies) if possession_proxies else 0.5
                
                return avg_shots, avg_corners, avg_poss_proxy
            
            # Get stats for both teams
            h_shots, h_corners, h_poss = get_team_rolling_stats(home_team, current_date)
            a_shots, a_corners, a_poss = get_team_rolling_stats(away_team, current_date)
            
            home_shots_avg.append(h_shots)
            away_shots_avg.append(a_shots)
            home_corners_avg.append(h_corners)
            away_corners_avg.append(a_corners)
            home_possession_proxy.append(h_poss)
            away_possession_proxy.append(a_poss)
            
            if idx % 500 == 0:
                print(f"  Processed {idx+1} matches...")
        
        # Add to dataframe
        self.df['home_shots_avg'] = home_shots_avg
        self.df['away_shots_avg'] = away_shots_avg
        self.df['home_corners_avg'] = home_corners_avg
        self.df['away_corners_avg'] = away_corners_avg
        self.df['home_possession_proxy'] = home_possession_proxy
        self.df['away_possession_proxy'] = away_possession_proxy
        
        # Calculate differences (normalized)
        max_shots = max(self.df['home_shots_avg'].max(), self.df['away_shots_avg'].max())
        max_corners = max(self.df['home_corners_avg'].max(), self.df['away_corners_avg'].max())
        
        self.df['shots_diff_normalized'] = ((self.df['home_shots_avg'] - self.df['away_shots_avg']) / max_shots + 1) / 2
        self.df['corners_diff_normalized'] = ((self.df['home_corners_avg'] - self.df['away_corners_avg']) / max_corners + 1) / 2
        self.df['possession_proxy_diff'] = self.df['home_possession_proxy'] - self.df['away_possession_proxy']
        self.df['possession_proxy_diff_normalized'] = (self.df['possession_proxy_diff'] + 1) / 2
        
        print(f"✅ Rolling match stats calculated (NO LEAKAGE)")
        print(f"   Shots avg range: [{min(home_shots_avg + away_shots_avg):.1f}, {max(home_shots_avg + away_shots_avg):.1f}]")
        print(f"   Corners avg range: [{min(home_corners_avg + away_corners_avg):.1f}, {max(home_corners_avg + away_corners_avg):.1f}]")
        print(f"   Possession proxy range: [{min(home_possession_proxy + away_possession_proxy):.3f}, {max(home_possession_proxy + away_possession_proxy):.3f}]")
        
        return self
    
    def add_league_position_features(self):
        """Add league position/points difference features"""
        print(f"\n🏆 ADDING LEAGUE POSITION FEATURES...")
        
        # Calculate running league table position for each match
        points_diff = []
        position_diff = []
        
        for idx, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            current_date = row['Date']
            current_season = row['Season']
            
            # Get all matches before current date in same season
            season_matches_before = self.df[
                (self.df['Season'] == current_season) &
                (self.df['Date'] < current_date)
            ]
            
            if len(season_matches_before) == 0:
                # Start of season - neutral
                points_diff.append(0)
                position_diff.append(0.5)
                continue
                
            # Calculate points table before this match
            teams_points = {}
            all_teams = pd.concat([season_matches_before['HomeTeam'], season_matches_before['AwayTeam']]).unique()
            
            for team in all_teams:
                teams_points[team] = 0
                
            for _, match in season_matches_before.iterrows():
                if match['FullTimeResult'] == 'H':
                    teams_points[match['HomeTeam']] += 3
                elif match['FullTimeResult'] == 'A':
                    teams_points[match['AwayTeam']] += 3
                else:  # Draw
                    teams_points[match['HomeTeam']] += 1
                    teams_points[match['AwayTeam']] += 1
            
            # Points difference
            home_points = teams_points.get(home_team, 0)
            away_points = teams_points.get(away_team, 0)
            points_diff.append(home_points - away_points)
            
            # Position difference (higher points = lower position number)
            sorted_teams = sorted(teams_points.items(), key=lambda x: x[1], reverse=True)
            positions = {team: pos+1 for pos, (team, pts) in enumerate(sorted_teams)}
            
            home_pos = positions.get(home_team, 10)  # Mid-table default
            away_pos = positions.get(away_team, 10)
            
            # Normalize position difference (away_pos - home_pos because lower number = better)
            pos_diff_raw = away_pos - home_pos  # Positive if home team better positioned
            position_diff.append((pos_diff_raw + 20) / 40)  # Normalize to 0-1
            
            if idx % 500 == 0:
                print(f"  Processed league positions for {idx+1} matches...")
        
        self.df['points_diff'] = points_diff
        self.df['position_diff_normalized'] = position_diff
        
        # Normalize points difference
        max_points_diff = max(abs(min(points_diff)), max(points_diff))
        self.df['points_diff_normalized'] = [(p + max_points_diff) / (2 * max_points_diff) for p in points_diff]
        
        print(f"✅ League position features added")
        print(f"   Points diff range: [{min(points_diff)}, {max(points_diff)}]")
        print(f"   Position diff range: [{min(position_diff):.3f}, {max(position_diff):.3f}]")
        
        return self
    
    def combine_with_existing_features(self, processed_file):
        """Combine with existing processed features"""
        print(f"\n🔗 COMBINING WITH EXISTING FEATURES...")
        
        # Load existing processed features
        existing_df = pd.read_csv(processed_file)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        
        # Merge on Date, HomeTeam, AwayTeam
        merged_df = self.df.merge(
            existing_df[['Date', 'HomeTeam', 'AwayTeam', 'home_form', 'away_form', 
                        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'home_advantage']],
            on=['Date', 'HomeTeam', 'AwayTeam'],
            how='left'
        )
        
        print(f"✅ Merged with existing features: {merged_df.shape}")
        
        return merged_df
    
    def save_enhanced_dataset(self, output_path):
        """Save enhanced dataset with all features"""
        print(f"\n💾 SAVING ENHANCED DATASET...")
        
        # Select final features (NO LEAKAGE)
        final_features = [
            # Core identification
            'Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult',
            
            # Original engineered (0-1)
            'home_form', 'away_form', 'form_diff_normalized',
            'elo_diff_normalized', 'h2h_score', 'home_advantage',
            
            # New temporal features
            'matchday_normalized', 'season_period_numeric',
            
            # New rolling stats (NO LEAKAGE)
            'shots_diff_normalized', 'corners_diff_normalized',
            'possession_proxy_diff_normalized',
            
            # League position features  
            'points_diff_normalized', 'position_diff_normalized'
        ]
        
        # Check which features exist
        available_features = [f for f in final_features if f in self.df.columns]
        final_df = self.df[available_features].copy()
        
        final_df.to_csv(output_path, index=False)
        
        print(f"✅ Enhanced dataset saved: {output_path}")
        print(f"   Shape: {final_df.shape}")
        
        # Count ML features
        ml_features = [f for f in available_features if any(x in f for x in ['form', 'elo', 'h2h', 'advantage', 'normalized', 'numeric', 'proxy'])]
        print(f"   ML features: {len(ml_features)}")
        print(f"   Features: {ml_features}")
        
        return final_df

# Execute
if __name__ == "__main__":
    print("🚀 ODDSY ENHANCED FEATURE ENGINEERING (NO DATA LEAKAGE)")
    
    fe = OddsyFixedFeatureEngineering()
    fe.load_data('/Users/maxime/Desktop/Oddsy/data/cleaned/premier_league_2019_2024_cleaned.csv')
    fe.add_temporal_features()
    fe.calculate_rolling_match_stats(window=5)
    fe.add_league_position_features()
    
    # Combine with existing good features
    enhanced_df = fe.combine_with_existing_features('/Users/maxime/Desktop/Oddsy/data/processed/premier_league_2019_2024_processed.csv')
    fe.df = enhanced_df
    
    # Save enhanced dataset
    final_df = fe.save_enhanced_dataset('/Users/maxime/Desktop/Oddsy/data/processed/premier_league_2019_2024_enhanced.csv')
    
    print(f"\n🎉 ENHANCED FEATURE ENGINEERING COMPLETE!")
    print(f"💡 Possession approximated via shots ratio")
    print(f"🛡️  No data leakage - all features calculated BEFORE match")
    print(f"📈 Ready for improved ML model!")