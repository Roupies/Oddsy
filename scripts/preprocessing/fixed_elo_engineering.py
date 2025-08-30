import pandas as pd
import numpy as np
from datetime import datetime

class OddsyFixedEloEngineering:
    """
    Feature Engineering avec Elo CONTINU (pas de reset entre saisons)
    """
    
    def __init__(self, initial_elo=1500, k_factor=32):
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.elo_ratings = {}
        
    def load_data(self, filepath):
        """Load cleaned dataset"""
        print("🔧 ODDSY ELO CONTINU - CORRECTION DU BUG")
        print("="*60)
        self.df = pd.read_csv(filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        print(f"Loaded {len(self.df)} matches from {self.df['Date'].min()} to {self.df['Date'].max()}")
        return self
    
    def initialize_elo_ratings(self):
        """Initialize Elo ratings ONCE at start - NO RESET between seasons"""
        print("\n⚡ INITIALISATION ELO (UNE SEULE FOIS)")
        print("-" * 40)
        
        # Get all unique teams across ALL seasons
        all_teams = pd.concat([self.df['HomeTeam'], self.df['AwayTeam']]).unique()
        
        # Initialize ALL teams to same starting Elo
        # (Dans la vraie vie, on utiliserait l'Elo de fin 2018-19, mais on n'a pas ces données)
        for team in all_teams:
            self.elo_ratings[team] = self.initial_elo
            
        print(f"✅ {len(all_teams)} équipes initialisées à Elo {self.initial_elo}")
        print("📝 Note: En production, on utiliserait l'Elo de fin de saison précédente")
        
        return self
    
    def calculate_elo_update(self, home_team, away_team, result, home_elo, away_elo):
        """Calculate Elo rating updates after a match"""
        # Expected scores (formule standard Elo)
        expected_home = 1 / (1 + 10**((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home
        
        # Actual scores
        if result == 'H':
            actual_home, actual_away = 1, 0
        elif result == 'A':
            actual_home, actual_away = 0, 1
        else:  # Draw
            actual_home, actual_away = 0.5, 0.5
        
        # Updates (K-factor constant pour MVP)
        home_change = self.k_factor * (actual_home - expected_home)
        away_change = self.k_factor * (actual_away - expected_away)
        
        return home_change, away_change
    
    def build_elo_history(self):
        """Build CONTINUOUS Elo history - NO RESET between seasons"""
        print(f"\n📈 CALCUL ELO CONTINU (K={self.k_factor})")
        print("-" * 40)
        
        elo_history = []
        season_transitions = []
        
        for idx, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            result = row['FullTimeResult']
            current_season = row['Season']
            
            # Track season changes (pour debug)
            if idx > 0 and self.df.iloc[idx-1]['Season'] != current_season:
                prev_season = self.df.iloc[idx-1]['Season']
                print(f"  🔄 Passage {prev_season} → {current_season}")
                print(f"     Exemples Elo: Liverpool={self.elo_ratings.get('Liverpool', 'N/A'):.0f}, "
                      f"Man City={self.elo_ratings.get('Man City', 'N/A'):.0f}")
                season_transitions.append({
                    'from_season': prev_season,
                    'to_season': current_season,
                    'match_idx': idx
                })
            
            # Get current Elo ratings (CONTINU, pas de reset!)
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
            
            # Update ratings (PERSISTENT across seasons)
            self.elo_ratings[home_team] += home_change
            self.elo_ratings[away_team] += away_change
            
            if idx % 500 == 0:
                print(f"  Processed {idx+1} matches...")
        
        # Convert to DataFrame and merge
        elo_df = pd.DataFrame(elo_history)
        self.df = self.df.merge(elo_df, left_index=True, right_on='match_idx', how='left')
        
        print(f"✅ Elo continu calculé pour {len(self.df)} matches")
        
        # Show final Elo ratings as example
        print(f"\n📊 EXEMPLES ELO FINAL (après {len(self.df)} matchs):")
        sample_teams = ['Liverpool', 'Man City', 'Arsenal', 'Norwich', 'Sheffield United']
        for team in sample_teams:
            if team in self.elo_ratings:
                print(f"   {team:15s}: {self.elo_ratings[team]:.0f}")
        
        return self
    
    def generate_final_features(self):
        """Generate final contracted features (0-1 scale)"""
        print(f"\n🎯 GÉNÉRATION FEATURES FINALES")
        print("-" * 40)
        
        # Normalize Elo difference to 0-1 scale
        elo_diff_min = self.df['elo_diff'].min()
        elo_diff_max = self.df['elo_diff'].max()
        elo_diff_range = elo_diff_max - elo_diff_min
        
        self.df['elo_diff_normalized'] = (self.df['elo_diff'] - elo_diff_min) / elo_diff_range
        
        print(f"✅ Elo diff normalisé:")
        print(f"   Range brut: [{elo_diff_min:.0f}, {elo_diff_max:.0f}]")
        print(f"   Range normalisé: [{self.df['elo_diff_normalized'].min():.3f}, {self.df['elo_diff_normalized'].max():.3f}]")
        
        return self
    
    def add_other_features(self):
        """Add other features (simplified for MVP)"""
        print(f"\n⚡ AJOUT AUTRES FEATURES (version rapide)")
        print("-" * 40)
        
        # Simplified form (5-match rolling average of points)
        home_form = []
        away_form = []
        
        for idx, row in self.df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            current_date = row['Date']
            
            def get_recent_points(team, exclude_date, window=5):
                recent_matches = self.df[
                    ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
                    (self.df['Date'] < exclude_date)
                ].tail(window)
                
                if len(recent_matches) == 0:
                    return 1.5  # Neutral (1.5/3 = 0.5)
                
                points = 0
                for _, match in recent_matches.iterrows():
                    if match['HomeTeam'] == team:
                        if match['FullTimeResult'] == 'H': points += 3
                        elif match['FullTimeResult'] == 'D': points += 1
                    else:
                        if match['FullTimeResult'] == 'A': points += 3
                        elif match['FullTimeResult'] == 'D': points += 1
                
                return points / len(recent_matches) if len(recent_matches) > 0 else 1.5
            
            h_form = get_recent_points(home_team, current_date)
            a_form = get_recent_points(away_team, current_date)
            
            home_form.append(h_form / 3)  # Normalize to 0-1
            away_form.append(a_form / 3)
        
        self.df['home_form'] = home_form
        self.df['away_form'] = away_form
        self.df['form_diff_normalized'] = (np.array(home_form) - np.array(away_form) + 1) / 2
        
        # Simple features
        self.df['home_advantage'] = 1  # Always 1 for home team
        self.df['h2h_score'] = 0.5     # Neutral for MVP (can be improved)
        
        print(f"✅ Features ajoutées:")
        print(f"   home_form: [{min(home_form):.3f}, {max(home_form):.3f}]")
        print(f"   away_form: [{min(away_form):.3f}, {max(away_form):.3f}]")
        
        return self
    
    def save_corrected_data(self, output_path):
        """Save corrected dataset with proper Elo"""
        print(f"\n💾 SAUVEGARDE DATASET CORRIGÉ")
        print("-" * 40)
        
        # Key columns for final dataset
        final_columns = [
            'Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult',
            'home_elo_before', 'away_elo_before', 'elo_diff_normalized',
            'home_form', 'away_form', 'form_diff_normalized',
            'h2h_score', 'home_advantage'
        ]
        
        available_cols = [col for col in final_columns if col in self.df.columns]
        final_df = self.df[available_cols].copy()
        
        final_df.to_csv(output_path, index=False)
        
        print(f"✅ Dataset sauvé: {output_path}")
        print(f"   Shape: {final_df.shape}")
        print(f"   Features ML: {len([c for c in available_cols if c not in ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult']])}")
        
        return final_df

# Execute
if __name__ == "__main__":
    print("🚀 CORRECTION ELO ODDSY - Version Continue")
    
    fe = OddsyFixedEloEngineering(k_factor=32)  # K constant pour MVP
    
    fe.load_data('/Users/maxime/Desktop/Oddsy/data/cleaned/premier_league_2019_2024_cleaned.csv')
    fe.initialize_elo_ratings()
    fe.build_elo_history()
    fe.generate_final_features()
    fe.add_other_features()
    
    # Save corrected dataset
    corrected_df = fe.save_corrected_data('/Users/maxime/Desktop/Oddsy/data/processed/premier_league_2019_2024_corrected_elo.csv')
    
    print(f"\n🎉 CORRECTION ELO TERMINÉE!")
    print(f"🔬 Test avec Liverpool vs Norwich maintenant...")
    
    # Quick test Liverpool vs Norwich
    liverpool_norwich = corrected_df[
        (corrected_df['HomeTeam'] == 'Liverpool') & 
        (corrected_df['AwayTeam'] == 'Norwich') &
        (corrected_df['Date'] >= '2019-08-01') & 
        (corrected_df['Date'] <= '2019-09-01')
    ]
    
    if len(liverpool_norwich) > 0:
        match = liverpool_norwich.iloc[0]
        print(f"   📊 Liverpool vs Norwich (2019-08-09):")
        print(f"      Elo Liverpool: {match['home_elo_before']:.0f}")
        print(f"      Elo Norwich:   {match['away_elo_before']:.0f}")
        print(f"      elo_diff_normalized: {match['elo_diff_normalized']:.3f}")
        print(f"      {'✅ CORRIGÉ!' if match['elo_diff_normalized'] > 0.55 else '❌ Encore un problème'}")
    else:
        print(f"   ⚠️ Match Liverpool vs Norwich non trouvé")