#!/usr/bin/env python3
"""
Extract Championship Initialization Data
----------------------------------------
Extrait données Championship 2024-25 pour initialiser équipes promues en EPL 2025-26.
Crée fichier temporaire avec état final (Elo, form, H2H) pour signal supplémentaire.

JAMAIS intégré au training set - seulement pour initialisation features.

Usage:
    python extract_championship_initialization.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Équipes promues EPL 2025-26
PROMOTED_TEAMS = {
    'Leeds': ['Leeds', 'Leeds United'],
    'Leicester': ['Leicester', 'Leicester City'],
    'Ipswich': ['Ipswich', 'Ipswich Town'],
    'Southampton': ['Southampton'],  # Relégué puis repromu
    'Sunderland': ['Sunderland']     # Promu Championship
}

def load_championship_2024_25():
    """Charge données Championship 2024-25"""
    
    champ_file = Path("data/raw/Championship 2024 2025.csv")
    
    if not champ_file.exists():
        print(f"❌ Fichier Championship 2024-25 non trouvé: {champ_file}")
        return None
    
    print(f"📊 Chargement Championship 2024-25...")
    
    # Charger avec structure football-data.co.uk
    df = pd.read_csv(champ_file, encoding='utf-8-sig')
    print(f"   Total lignes: {len(df)}")
    
    # Filtrer seulement matches avec résultat (joués)
    df_played = df[df['FTR'].notna()].copy()
    print(f"✅ {len(df_played)} matches Championship joués")
    
    # Conversion format standard
    df_played['Date'] = pd.to_datetime(df_played['Date'], dayfirst=True)
    df_played['Season'] = '2024-2025'
    
    return df_played

def filter_promoted_teams_matches(df_champ):
    """Filtre matches impliquant équipes promues"""
    
    print("🎯 Filtrage matches équipes promues...")
    
    # Créer liste de tous les noms possibles
    all_promoted_names = []
    for team_variants in PROMOTED_TEAMS.values():
        all_promoted_names.extend(team_variants)
    
    # Filtrer matches avec au moins une équipe promue
    df_promoted = df_champ[
        (df_champ['HomeTeam'].isin(all_promoted_names)) |
        (df_champ['AwayTeam'].isin(all_promoted_names))
    ].copy()
    
    print(f"✅ {len(df_promoted)} matches avec équipes promues")
    
    # Statistiques par équipe
    for main_name, variants in PROMOTED_TEAMS.items():
        team_matches = df_promoted[
            (df_promoted['HomeTeam'].isin(variants)) |
            (df_promoted['AwayTeam'].isin(variants))
        ]
        print(f"   {main_name}: {len(team_matches)} matches")
    
    return df_promoted

def calculate_final_elo_ratings(df_promoted):
    """Calcule ratings Elo finaux des équipes promues"""
    
    print("📊 Calcul ratings Elo finaux équipes promues...")
    
    # Trier par date pour traitement séquentiel
    df_sorted = df_promoted.sort_values('Date').copy()
    
    # Initialiser Elo pour toutes les équipes (Championship niveau)
    all_teams = list(set(df_sorted['HomeTeam'].tolist() + df_sorted['AwayTeam'].tolist()))
    elo_ratings = {team: 1400 for team in all_teams}  # Championship = niveau plus bas qu'EPL (1500)
    
    # Historique évolution Elo
    elo_history = {team: [] for team in all_teams}
    
    for idx, match in df_sorted.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        result = match['FTR']
        
        # Ratings avant match
        home_elo_before = elo_ratings[home_team]
        away_elo_before = elo_ratings[away_team]
        
        # Mettre à jour Elo
        new_home_elo, new_away_elo = update_elo_ratings(
            home_elo_before, away_elo_before, result, k=32
        )
        
        elo_ratings[home_team] = new_home_elo
        elo_ratings[away_team] = new_away_elo
        
        # Enregistrer évolution
        elo_history[home_team].append({
            'date': match['Date'],
            'elo': new_home_elo,
            'opponent': away_team,
            'result': result,
            'home': True
        })
        elo_history[away_team].append({
            'date': match['Date'],
            'elo': new_away_elo,
            'opponent': home_team,
            'result': 'H' if result == 'A' else ('A' if result == 'H' else 'D'),
            'home': False
        })
    
    # Extraire Elo finaux équipes promues
    final_elos = {}
    for main_name, variants in PROMOTED_TEAMS.items():
        for variant in variants:
            if variant in elo_ratings:
                final_elos[main_name] = {
                    'final_elo': elo_ratings[variant],
                    'team_name_champ': variant,
                    'matches_played': len([h for h in elo_history[variant]]),
                    'elo_evolution': elo_history[variant][-10:]  # 10 derniers pour tendance
                }
                break
    
    print(f"✅ Elo finaux calculés pour {len(final_elos)} équipes promues")
    for team, data in final_elos.items():
        print(f"   {team}: {data['final_elo']:.0f} (après {data['matches_played']} matches)")
    
    return final_elos

def update_elo_ratings(home_elo, away_elo, result, k=32):
    """Met à jour ratings Elo après match"""
    
    # Probabilités attendues
    prob_home = 1 / (1 + 10**((away_elo - home_elo) / 400))
    prob_away = 1 - prob_home
    
    # Résultats réels
    if result == 'H':
        score_home = 1.0
        score_away = 0.0
    elif result == 'A':
        score_home = 0.0
        score_away = 1.0
    else:  # Draw
        score_home = 0.5
        score_away = 0.5
    
    # Nouveaux ratings
    new_home_elo = home_elo + k * (score_home - prob_home)
    new_away_elo = away_elo + k * (score_away - prob_away)
    
    return new_home_elo, new_away_elo

def calculate_final_form(df_promoted):
    """Calcule forme finale (5 et 10 derniers matches) équipes promues"""
    
    print("📈 Calcul forme finale équipes promues...")
    
    df_sorted = df_promoted.sort_values('Date').copy()
    
    final_forms = {}
    
    for main_name, variants in PROMOTED_TEAMS.items():
        
        # Trouver tous les matches de cette équipe
        team_matches = []
        
        for variant in variants:
            home_matches = df_sorted[df_sorted['HomeTeam'] == variant].copy()
            away_matches = df_sorted[df_sorted['AwayTeam'] == variant].copy()
            
            # Standardiser format résultats
            for idx, match in home_matches.iterrows():
                result = match['FTR']
                if result == 'H':
                    points = 3
                elif result == 'D':
                    points = 1
                else:
                    points = 0
                
                team_matches.append({
                    'date': match['Date'],
                    'opponent': match['AwayTeam'],
                    'home': True,
                    'result': result,
                    'points': points,
                    'goals_for': match['FTHG'],
                    'goals_against': match['FTAG']
                })
            
            for idx, match in away_matches.iterrows():
                result = match['FTR']
                if result == 'A':
                    points = 3
                elif result == 'D':
                    points = 1
                else:
                    points = 0
                
                team_matches.append({
                    'date': match['Date'],
                    'opponent': match['HomeTeam'],
                    'home': False,
                    'result': 'H' if result == 'A' else ('A' if result == 'H' else 'D'),
                    'points': points,
                    'goals_for': match['FTAG'],
                    'goals_against': match['FTHG']
                })
        
        if team_matches:
            # Trier par date
            team_matches = sorted(team_matches, key=lambda x: x['date'])
            
            # Calculer formes finales
            form_5 = calculate_team_form([m['points'] for m in team_matches[-5:]])
            form_10 = calculate_team_form([m['points'] for m in team_matches[-10:]])
            
            # Stats supplémentaires
            goals_scored_5 = sum(m['goals_for'] for m in team_matches[-5:])
            goals_conceded_5 = sum(m['goals_against'] for m in team_matches[-5:])
            
            final_forms[main_name] = {
                'form_5_normalized': form_5,
                'form_10_normalized': form_10,
                'goals_scored_last_5': goals_scored_5,
                'goals_conceded_last_5': goals_conceded_5,
                'total_matches': len(team_matches),
                'last_5_results': [m['result'] for m in team_matches[-5:]],
                'last_match_date': team_matches[-1]['date'].strftime('%Y-%m-%d')
            }
    
    print(f"✅ Forme finale calculée pour {len(final_forms)} équipes")
    for team, data in final_forms.items():
        results = ''.join(data['last_5_results'])
        print(f"   {team}: Form {data['form_5_normalized']:.3f} (derniers: {results})")
    
    return final_forms

def calculate_team_form(points_list):
    """Calcule forme équipe normalisée [0,1]"""
    if not points_list:
        return 0.5  # Neutre
    
    # Points moyens normalisés [0,1]
    points = sum(points_list) / len(points_list) / 3.0
    return points

def calculate_h2h_vs_epl_teams(df_promoted):
    """Calcule H2H des équipes promues vs équipes EPL (historique)"""
    
    print("🏟️ Calcul H2H équipes promues vs équipes EPL...")
    
    # Équipes EPL qui peuvent avoir joué vs équipes promues (relégations passées)
    potential_epl_teams = [
        'Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United',
        'Tottenham', 'Newcastle', 'West Ham', 'Brighton', 'Crystal Palace',
        'Brentford', 'Fulham', 'Everton', 'Aston Villa', 'Bournemouth',
        'Wolves', 'Sheffield United', 'Burnley'  # Certaines peuvent avoir été reléguées
    ]
    
    h2h_data = {}
    
    for main_name, variants in PROMOTED_TEAMS.items():
        
        team_h2h = {}
        
        for variant in variants:
            # Chercher matches vs équipes EPL potentielles
            for epl_team in potential_epl_teams:
                
                home_vs_epl = df_promoted[
                    (df_promoted['HomeTeam'] == variant) & 
                    (df_promoted['AwayTeam'] == epl_team)
                ]
                
                away_vs_epl = df_promoted[
                    (df_promoted['AwayTeam'] == variant) & 
                    (df_promoted['HomeTeam'] == epl_team)
                ]
                
                if len(home_vs_epl) > 0 or len(away_vs_epl) > 0:
                    # Calculer bilan H2H
                    total_matches = len(home_vs_epl) + len(away_vs_epl)
                    wins = 0
                    draws = 0
                    
                    # Matches à domicile
                    for _, match in home_vs_epl.iterrows():
                        if match['FTR'] == 'H':
                            wins += 1
                        elif match['FTR'] == 'D':
                            draws += 1
                    
                    # Matches à l'extérieur  
                    for _, match in away_vs_epl.iterrows():
                        if match['FTR'] == 'A':
                            wins += 1
                        elif match['FTR'] == 'D':
                            draws += 1
                    
                    # Score H2H normalisé
                    points = wins * 3 + draws * 1
                    max_points = total_matches * 3
                    h2h_score = points / max_points if max_points > 0 else 0.5
                    
                    team_h2h[epl_team] = {
                        'matches': total_matches,
                        'wins': wins,
                        'draws': draws,
                        'losses': total_matches - wins - draws,
                        'h2h_score': h2h_score
                    }
        
        h2h_data[main_name] = team_h2h
    
    print(f"✅ H2H calculé pour {len(h2h_data)} équipes promues")
    for team, h2h in h2h_data.items():
        if h2h:
            print(f"   {team}: H2H vs {len(h2h)} équipes EPL")
    
    return h2h_data

def save_initialization_data(final_elos, final_forms, h2h_data):
    """Sauvegarde données d'initialisation dans fichier temporaire"""
    
    print("💾 Sauvegarde données d'initialisation...")
    
    # Combiner toutes les données
    initialization_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'purpose': 'Initialization data for promoted teams EPL 2025-26',
            'source': 'Championship 2024-25 final state',
            'warning': 'NEVER use in training set - initialization only'
        },
        'promoted_teams': list(PROMOTED_TEAMS.keys()),
        'elo_ratings': final_elos,
        'team_forms': final_forms,
        'h2h_vs_epl': h2h_data
    }
    
    # Calculer données agrégées pour résumé
    summary = {
        'teams_with_elo': len(final_elos),
        'teams_with_form': len(final_forms),
        'teams_with_h2h': len([t for t in h2h_data.values() if t]),
        'avg_final_elo': np.mean([data['final_elo'] for data in final_elos.values()]),
        'avg_form_5': np.mean([data['form_5_normalized'] for data in final_forms.values()])
    }
    
    initialization_data['summary'] = summary
    
    # Sauvegarder fichier temporaire
    output_dir = Path("temp/championship_initialization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_file = output_dir / f"promoted_teams_initialization_{timestamp}.json"
    
    with open(init_file, 'w') as f:
        json.dump(initialization_data, f, indent=2, default=str)
    
    print(f"✅ Données d'initialisation: {init_file}")
    
    # Créer aussi version CSV pour facilité d'usage
    csv_data = []
    for team in PROMOTED_TEAMS.keys():
        row = {
            'team': team,
            'final_elo': final_elos.get(team, {}).get('final_elo', 1400),
            'form_5': final_forms.get(team, {}).get('form_5_normalized', 0.5),
            'form_10': final_forms.get(team, {}).get('form_10_normalized', 0.5),
            'goals_scored_5': final_forms.get(team, {}).get('goals_scored_last_5', 0),
            'total_champ_matches': final_forms.get(team, {}).get('total_matches', 0),
            'h2h_matches': len(h2h_data.get(team, {}))
        }
        csv_data.append(row)
    
    df_summary = pd.DataFrame(csv_data)
    csv_file = output_dir / f"promoted_teams_summary_{timestamp}.csv"
    df_summary.to_csv(csv_file, index=False)
    
    print(f"📊 Résumé CSV: {csv_file}")
    print(f"\n📋 RÉSUMÉ INITIALISATION:")
    print(df_summary.to_string(index=False))
    
    return init_file, csv_file

def main():
    print("🏆 EXTRACT CHAMPIONSHIP INITIALIZATION DATA")
    print("=" * 50)
    print("Extraction données Championship 2024-25 pour initialisation équipes promues")
    print("⚠️  JAMAIS intégré au training set - seulement initialisation")
    print()
    
    # Phase 1: Charger Championship 2024-25
    print("📊 PHASE 1: Chargement Championship 2024-25")
    df_champ = load_championship_2024_25()
    if df_champ is None:
        return 1
    
    # Phase 2: Filtrer équipes promues
    print("\n🎯 PHASE 2: Filtrage équipes promues")
    df_promoted = filter_promoted_teams_matches(df_champ)
    
    # Phase 3: Calculer Elo finaux
    print("\n📊 PHASE 3: Calcul Elo finaux")
    final_elos = calculate_final_elo_ratings(df_promoted)
    
    # Phase 4: Calculer forme finale
    print("\n📈 PHASE 4: Calcul forme finale")
    final_forms = calculate_final_form(df_promoted)
    
    # Phase 5: Calculer H2H vs EPL
    print("\n🏟️ PHASE 5: Calcul H2H vs équipes EPL")
    h2h_data = calculate_h2h_vs_epl_teams(df_promoted)
    
    # Phase 6: Sauvegarde
    print("\n💾 PHASE 6: Sauvegarde données d'initialisation")
    init_file, csv_file = save_initialization_data(final_elos, final_forms, h2h_data)
    
    # Résumé final
    print(f"\n🎉 EXTRACTION COMPLÈTE!")
    print(f"   Données d'initialisation: {init_file}")
    print(f"   Résumé équipes promues: {csv_file}")
    print(f"   🎯 PRÊT pour initialisation features EPL 2025-26")
    
    return 0

if __name__ == "__main__":
    exit(main())