import pandas as pd
import numpy as np
from datetime import datetime

class OddsyFeatureEngineering:
    """
    Feature Engineering pipeline pour Oddsy
    Génère les features contractées 0-1 pour prédiction H/D/A
    """
    
    def __init__(self, initial_elo=1500, k_factor=32, k_new_team=50):
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.k_new_team = k_new_team
        self.elo_ratings = {}
        
    def load_data(self, filepath):
        """Load cleaned dataset"""
        print("=== LOADING ODDSY DATA ===")
        self.df = pd.read_csv(filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        print(f"Loaded {len(self.df)} matches from {self.df['Date'].min()} to {self.df['Date'].max()}")
        return self
    
    def initialize_elo_ratings(self):
        """Initialize Elo ratings for all teams"""
        print("=== INITIALIZING ELO RATINGS ===")
        all_teams = pd.concat([self.df['HomeTeam'], self.df['AwayTeam']]).unique()
        
        # Check team history to identify new teams (less data)
        team_match_counts = {}
        for team in all_teams:
            home_matches = len(self.df[self.df['HomeTeam'] == team])
            away_matches = len(self.df[self.df['AwayTeam'] == team])
            team_match_counts[team] = home_matches + away_matches
        
        # Initialize Elo ratings
        for team in all_teams:
            if team_match_counts[team] < 76:  # Less than 2 seasons worth
                self.elo_ratings[team] = self.initial_elo - 100  # Lower for new teams
                print(f"{team}: {team_match_counts[team]} matches -> Elo {self.elo_ratings[team]} (new team)")
            else:
                self.elo_ratings[team] = self.initial_elo
                print(f"{team}: {team_match_counts[team]} matches -> Elo {self.elo_ratings[team]}")
        
        return self
    
    def calculate_elo_update(self, home_team, away_team, result, home_elo, away_elo):
        """Calculate Elo rating updates after a match"""
        # Expected scores
        expected_home = 1 / (1 + 10**((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home
        
        # Actual scores
        if result == 'H':
            actual_home, actual_away = 1, 0
        elif result == 'A':
            actual_home, actual_away = 0, 1
        else:  # Draw
            actual_home, actual_away = 0.5, 0.5
        
        # K-factor (adaptive for new teams)
        team_match_counts = {}
        for team in [home_team, away_team]:
            home_matches = len(self.df[self.df['HomeTeam'] == team])
            away_matches = len(self.df[self.df['AwayTeam'] == team])
            team_match_counts[team] = home_matches + away_matches
        
        k_home = self.k_new_team if team_match_counts[home_team] < 76 else self.k_factor
        k_away = self.k_new_team if team_match_counts[away_team] < 76 else self.k_factor
        
        # Updates
        home_change = k_home * (actual_home - expected_home)
        away_change = k_away * (actual_away - expected_away)
        
        return home_change, away_change
    
    def build_elo_history(self):
        """Build complete Elo history for all matches"""
        print("=== BUILDING ELO HISTORY ===")
        
        elo_history = []
        
        for idx, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            result = row['FullTimeResult']
            
            # Get current Elo ratings
            home_elo = self.elo_ratings[home_team]
            away_elo = self.elo_ratings[away_team]
            
            # Store pre-match Elo
            elo_history.append({
                'match_idx': idx,
                'home_elo_before': home_elo,
                'away_elo_before': away_elo,
                'elo_diff': home_elo - away_elo
            })
            
            # Calculate updates
            home_change, away_change = self.calculate_elo_update(
                home_team, away_team, result, home_elo, away_elo
            )
            
            # Update ratings
            self.elo_ratings[home_team] += home_change
            self.elo_ratings[away_team] += away_change
            
            if idx % 500 == 0:
                print(f"Processed {idx+1} matches...")
        
        # Convert to DataFrame and merge
        elo_df = pd.DataFrame(elo_history)
        self.df = self.df.merge(elo_df, left_index=True, right_on='match_idx', how='left')
        
        print(f"✅ Elo history built for {len(self.df)} matches")
        return self
    
    def calculate_rolling_stats(self, window=5):
        """Calculate rolling statistics for form"""
        print(f"=== CALCULATING ROLLING STATS (window={window}) ===")
        
        # Initialize lists to store rolling stats
        home_form = []
        away_form = []
        
        for idx, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            current_date = row['Date']
            
            # Get last N matches before current date for each team
            def get_recent_form(team, exclude_date):
                # Home matches
                home_matches = self.df[
                    (self.df['HomeTeam'] == team) & 
                    (self.df['Date'] < exclude_date)
                ].tail(window//2 + 1)
                
                # Away matches  
                away_matches = self.df[
                    (self.df['AwayTeam'] == team) & 
                    (self.df['Date'] < exclude_date)
                ].tail(window//2 + 1)
                
                # Combine and get most recent
                all_matches = pd.concat([home_matches, away_matches]).sort_values('Date').tail(window)
                
                if len(all_matches) == 0:
                    return 0.5  # Neutral for new teams
                
                # Calculate points earned
                points = 0
                for _, match in all_matches.iterrows():
                    if match['HomeTeam'] == team:
                        if match['FullTimeResult'] == 'H':
                            points += 3
                        elif match['FullTimeResult'] == 'D':
                            points += 1
                    else:  # Away team
                        if match['FullTimeResult'] == 'A':
                            points += 3
                        elif match['FullTimeResult'] == 'D':
                            points += 1
                
                # Convert to form score (0-1)
                max_points = len(all_matches) * 3
                return points / max_points if max_points > 0 else 0.5
            
            home_form_score = get_recent_form(home_team, current_date)
            away_form_score = get_recent_form(away_team, current_date)
            
            home_form.append(home_form_score)
            away_form.append(away_form_score)
            
            if idx % 500 == 0:
                print(f"Processed rolling stats for {idx+1} matches...")
        
        # Add to dataframe
        self.df['home_form'] = home_form
        self.df['away_form'] = away_form
        self.df['form_diff'] = self.df['home_form'] - self.df['away_form']
        
        print(f"✅ Rolling stats calculated")
        print(f"Home form range: {self.df['home_form'].min():.3f} to {self.df['home_form'].max():.3f}")
        print(f"Away form range: {self.df['away_form'].min():.3f} to {self.df['away_form'].max():.3f}")
        
        return self
    
    def calculate_h2h_history(self, window=5):
        """Calculate head-to-head historical performance"""
        print(f"=== CALCULATING H2H HISTORY (window={window}) ===")
        
        h2h_scores = []
        
        for idx, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            current_date = row['Date']
            
            # Find previous H2H matches
            h2h_matches = self.df[
                (((self.df['HomeTeam'] == home_team) & (self.df['AwayTeam'] == away_team)) |
                 ((self.df['HomeTeam'] == away_team) & (self.df['AwayTeam'] == home_team))) &
                (self.df['Date'] < current_date)
            ].sort_values('Date').tail(window)
            
            if len(h2h_matches) == 0:
                h2h_score = 0.5  # Neutral if no history
            else:
                # Calculate home team's performance in H2H
                home_points = 0
                for _, match in h2h_matches.iterrows():
                    if match['HomeTeam'] == home_team:
                        if match['FullTimeResult'] == 'H':
                            home_points += 3
                        elif match['FullTimeResult'] == 'D':
                            home_points += 1
                    else:  # home_team was away
                        if match['FullTimeResult'] == 'A':
                            home_points += 3
                        elif match['FullTimeResult'] == 'D':
                            home_points += 1
                
                max_points = len(h2h_matches) * 3
                h2h_score = home_points / max_points
            
            h2h_scores.append(h2h_score)
            
            if idx % 500 == 0:
                print(f"Processed H2H for {idx+1} matches...")
        
        self.df['h2h_score'] = h2h_scores
        
        print(f"✅ H2H history calculated")
        print(f"H2H score range: {self.df['h2h_score'].min():.3f} to {self.df['h2h_score'].max():.3f}")
        
        return self
    
    def generate_final_features(self):
        """Generate final contracted features (0-1 scale)"""
        print("=== GENERATING FINAL CONTRACTED FEATURES ===")
        
        # Normalize Elo difference to 0-1 scale
        # Typical Elo differences range from -800 to +800
        self.df['elo_diff_normalized'] = (self.df['elo_diff'] + 800) / 1600
        self.df['elo_diff_normalized'] = self.df['elo_diff_normalized'].clip(0, 1)
        
        # Normalize form_diff (-1 to +1) to (0 to 1)
        self.df['form_diff_normalized'] = (self.df['form_diff'] + 1) / 2
        
        # Add home advantage (simple binary feature)
        self.df['home_advantage'] = 1  # Always 1 for home team
        
        # Create feature summary
        feature_cols = [
            'home_form', 'away_form', 'form_diff_normalized', 
            'elo_diff_normalized', 'h2h_score', 'home_advantage'
        ]
        
        print("Final feature ranges:")
        for col in feature_cols:
            if col in self.df.columns:
                print(f"{col:20s}: {self.df[col].min():.3f} to {self.df[col].max():.3f}")
        
        return self
    
    def save_processed_data(self, output_path):
        """Save processed dataset with all features"""
        print("=== SAVING PROCESSED DATA ===")
        
        # Select key columns for final dataset
        key_columns = [
            'Date', 'Season', 'HomeTeam', 'AwayTeam',
            'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'FullTimeResult',
            'home_form', 'away_form', 'form_diff_normalized',
            'home_elo_before', 'away_elo_before', 'elo_diff_normalized',
            'h2h_score', 'home_advantage',
            'HomeTeamShots', 'AwayTeamShots', 'HomeTeamCorners', 'AwayTeamCorners'
        ]
        
        # Filter existing columns
        available_cols = [col for col in key_columns if col in self.df.columns]
        final_df = self.df[available_cols].copy()
        
        # Save
        final_df.to_csv(output_path, index=False)
        print(f"✅ Processed dataset saved to: {output_path}")
        print(f"Shape: {final_df.shape}")
        print(f"Features for ML: {[col for col in available_cols if 'form' in col or 'elo' in col or 'h2h' in col or 'home_advantage' in col]}")
        
        return final_df

# Main execution
if __name__ == "__main__":
    print("🚀 ODDSY FEATURE ENGINEERING PIPELINE 🚀")
    
    # Initialize pipeline
    fe = OddsyFeatureEngineering()
    
    # Process data
    fe.load_data('/Users/maxime/Desktop/Oddsy/data/cleaned/premier_league_2019_2024_cleaned.csv')
    fe.initialize_elo_ratings()
    fe.build_elo_history()
    fe.calculate_rolling_stats(window=5)
    fe.calculate_h2h_history(window=5)
    fe.generate_final_features()
    
    # Save processed data
    processed_df = fe.save_processed_data('/Users/maxime/Desktop/Oddsy/data/processed/premier_league_2019_2024_processed.csv')
    
    print("\n🎉 FEATURE ENGINEERING COMPLETE!")
    print("Ready for ML model training!")