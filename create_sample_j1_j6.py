"""
Création Échantillon J1-J6 EPL 2025-26 Réaliste
===============================================
Crée un échantillon réaliste des 6 premières journées EPL 2025-26
pour démontrer la jointure avec E0 (14).csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_realistic_j1_j6_sample():
    """Crée échantillon réaliste J1-J6 avec vraies équipes EPL 2025-26"""
    
    print("🔄 Création échantillon J1-J6 EPL 2025-26...")
    
    # Équipes EPL 2025-26 (20 équipes)
    teams = [
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
        'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
        'Nottm Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolverhampton'
    ]
    
    # Dates J1-J6 (approximatives saison 2025-26)
    j1_date = pd.to_datetime('2025-08-17')
    matchdays = []
    
    for round_num in range(1, 7):  # J1 à J6
        matchday_date = j1_date + timedelta(days=(round_num-1)*7)
        
        # 10 matchs par journée (20 équipes / 2)
        available_teams = teams.copy()
        np.random.shuffle(available_teams)
        
        for match_idx in range(10):
            if len(available_teams) < 2:
                break
                
            home_team = available_teams.pop()
            away_team = available_teams.pop()
            
            # Statistiques réalistes basées sur EPL patterns
            home_shots = np.random.randint(8, 22)
            away_shots = np.random.randint(6, 18)
            
            home_sot = min(home_shots, np.random.randint(3, home_shots//2 + 3))
            away_sot = min(away_shots, np.random.randint(2, away_shots//2 + 2))
            
            home_corners = np.random.randint(2, 12)
            away_corners = np.random.randint(1, 10)
            
            # xG réaliste (corrélé avec shots)
            home_xg = round(np.random.uniform(0.5, 3.5) * (home_shots/15), 2)
            away_xg = round(np.random.uniform(0.4, 3.0) * (away_shots/15), 2)
            
            match_data = {
                'Date': (matchday_date + timedelta(days=np.random.randint(0, 3))).strftime('%Y-%m-%d'),
                'Round': round_num,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'H_Shots': home_shots,
                'A_Shots': away_shots,
                'H_SoT': home_sot,
                'A_SoT': away_sot,
                'H_Corner': home_corners,
                'A_Corner': away_corners,
                'H_xG': home_xg,
                'A_xG': away_xg
            }
            
            matchdays.append(match_data)
    
    df = pd.DataFrame(matchdays)
    
    # Sauvegarder
    output_path = "data/fbref/epl_2025_26_stats_J1_J6.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Échantillon créé: {len(df)} matchs J1-J6")
    print(f"💾 Sauvegardé: {output_path}")
    
    # Stats de couverture
    coverage = {
        "matches": len(df),
        "shots_coverage": 100.0,  # Échantillon complet
        "sot_coverage": 100.0,
        "corners_coverage": 100.0,
        "xg_coverage": 100.0,
    }
    print("📊 Couverture:", coverage)
    
    # Afficher échantillon
    print(f"\n📊 ÉCHANTILLON J1-J6:")
    for round_num in range(1, 4):  # Afficher J1-J3
        round_matches = df[df['Round'] == round_num]
        print(f"\n🏆 J{round_num} ({len(round_matches)} matchs):")
        for _, match in round_matches.head(3).iterrows():
            print(f"   {match['HomeTeam']} vs {match['AwayTeam']}")
            print(f"   Shots: {match['H_Shots']}-{match['A_Shots']} | Corners: {match['H_Corner']}-{match['A_Corner']} | xG: {match['H_xG']}-{match['A_xG']}")
    
    return output_path

if __name__ == "__main__":
    create_realistic_j1_j6_sample()