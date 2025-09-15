#!/usr/bin/env python3
"""
Oddsy Season State Initializer
------------------------------
Génère état initial pour nouvelle saison avec carry-over robuste.
Calcule Elo, H2H, form et autres métriques depuis l'historique.

Usage:
    # Préparer état pour 2025-26 basé sur historique complet
    python initialize_season_state.py \
        --data data/processed/v13_xg_safe_features.csv \
        --target_season "2025-2026" \
        --out_state results/season_state_2025_26.json \
        --elo_k 32 \
        --form_window 5

    # Analyser état existant
    python initialize_season_state.py \
        --data data/processed/v13_xg_safe_features.csv \
        --analyze_season "2024-2025" \
        --out_state results/season_analysis_2024_25.json
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_elo_rating(winner_elo, loser_elo, k_factor=32, draw=False):
    """
    Calcule nouveaux ratings Elo après un match.
    
    Args:
        winner_elo, loser_elo: ratings actuels
        k_factor: facteur d'ajustement (32 standard)
        draw: True si match nul
    
    Returns:
        tuple: (nouveau_winner_elo, nouveau_loser_elo)
    """
    # Expected scores
    expected_winner = 1 / (1 + 10**((loser_elo - winner_elo) / 400))
    expected_loser = 1 - expected_winner
    
    if draw:
        # Match nul = 0.5 points chacun
        actual_winner = 0.5
        actual_loser = 0.5
    else:
        # Victoire/défaite = 1/0 points
        actual_winner = 1.0
        actual_loser = 0.0
    
    # Nouveaux ratings
    new_winner_elo = winner_elo + k_factor * (actual_winner - expected_winner)
    new_loser_elo = loser_elo + k_factor * (actual_loser - expected_loser)
    
    return new_winner_elo, new_loser_elo

def process_match_for_elo(elo_ratings, home_team, away_team, result, k_factor=32):
    """Traite un match pour mise à jour Elo"""
    home_elo = elo_ratings.get(home_team, 1500)  # Elo initial 1500
    away_elo = elo_ratings.get(away_team, 1500)
    
    if result == 'H':
        # Home win
        new_home_elo, new_away_elo = calculate_elo_rating(home_elo, away_elo, k_factor, draw=False)
    elif result == 'A':
        # Away win 
        new_away_elo, new_home_elo = calculate_elo_rating(away_elo, home_elo, k_factor, draw=False)
    else:  # 'D'
        # Draw
        new_home_elo, new_away_elo = calculate_elo_rating(home_elo, away_elo, k_factor, draw=True)
    
    elo_ratings[home_team] = new_home_elo
    elo_ratings[away_team] = new_away_elo
    
    return elo_ratings

def calculate_h2h_history(df, lookback_matches=10):
    """Calcule historique H2H entre équipes (derniers N matches)"""
    h2h_stats = defaultdict(lambda: defaultdict(list))
    
    # Trier par date
    df_sorted = df.sort_values('Date')
    
    for _, row in df_sorted.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        result = row['FullTimeResult']
        
        # Clé H2H (ordre alphabétique pour cohérence)
        teams = tuple(sorted([home_team, away_team]))
        match_key = f"{home_team}_vs_{away_team}"
        
        # Score H2H du point de vue home
        if result == 'H':
            h2h_score = 1.0  # Home wins
        elif result == 'A':
            h2h_score = 0.0  # Away wins  
        else:
            h2h_score = 0.5  # Draw
        
        # Ajouter à l'historique (limité à lookback_matches)
        h2h_stats[teams][match_key].append(h2h_score)
        if len(h2h_stats[teams][match_key]) > lookback_matches:
            h2h_stats[teams][match_key].pop(0)  # Remove oldest
    
    return h2h_stats

def calculate_form(df, form_window=5):
    """Calcule form récente de chaque équipe (derniers N matches)"""
    team_form = defaultdict(lambda: deque(maxlen=form_window))
    team_points = defaultdict(float)
    
    df_sorted = df.sort_values('Date')
    
    for _, row in df_sorted.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        result = row['FullTimeResult']
        
        # Points pour Home team
        if result == 'H':
            home_points = 3
            away_points = 0
        elif result == 'A':
            home_points = 0
            away_points = 3
        else:  # Draw
            home_points = 1
            away_points = 1
        
        # Ajouter aux deques (auto-limite à form_window)
        team_form[home_team].append(home_points)
        team_form[away_team].append(away_points)
        
        # Calculer moyenne form actuelle
        team_points[home_team] = np.mean(list(team_form[home_team])) if len(team_form[home_team]) > 0 else 1.5
        team_points[away_team] = np.mean(list(team_form[away_team])) if len(team_form[away_team]) > 0 else 1.5
    
    return dict(team_points)

def get_season_end_stats(df, season):
    """Récupère stats de fin de saison spécifique"""
    season_df = df[df['Season'] == season].sort_values('Date')
    
    if len(season_df) == 0:
        return None
    
    last_date = season_df['Date'].iloc[-1]
    teams = set(season_df['HomeTeam']).union(set(season_df['AwayTeam']))
    
    return {
        "season": season,
        "last_date": last_date,
        "n_matches": len(season_df),
        "teams": sorted(list(teams))
    }

def initialize_season_state(df, target_season=None, analyze_season=None, 
                          elo_k=32, form_window=5, h2h_lookback=10, verbose=True):
    """
    Initialise état pour nouvelle saison ou analyse saison existante.
    
    Args:
        df: DataFrame avec matches
        target_season: Saison cible pour initialisation (ex: "2025-2026")
        analyze_season: Saison existante à analyser (ex: "2024-2025")
        elo_k: K-factor pour Elo
        form_window: Fenêtre pour calcul form
        h2h_lookback: Nombre matches H2H à conserver
    """
    
    if target_season:
        # Mode: Initialiser nouvel état
        if verbose:
            print(f"🎯 Initialisation état pour saison {target_season}")
        
        # Utiliser tout l'historique disponible pour calculer état final
        historical_data = df.sort_values('Date')
        reference_season = "all_history"
        
    elif analyze_season:
        # Mode: Analyser saison existante
        if verbose:
            print(f"🔍 Analyse état saison {analyze_season}")
        
        # Utiliser données jusqu'à la fin de cette saison
        historical_data = df[df['Season'] <= analyze_season].sort_values('Date')
        reference_season = analyze_season
        
    else:
        raise ValueError("Doit spécifier target_season OU analyze_season")
    
    if len(historical_data) == 0:
        raise ValueError(f"Pas de données historiques trouvées")
    
    # 1. Calculer Elo ratings finaux
    if verbose:
        print("   Calcul Elo ratings...")
    elo_ratings = {}
    for _, row in historical_data.iterrows():
        elo_ratings = process_match_for_elo(
            elo_ratings, row['HomeTeam'], row['AwayTeam'], 
            row['FullTimeResult'], elo_k
        )
    
    # 2. Calculer form récente
    if verbose:
        print("   Calcul form récente...")
    form_ratings = calculate_form(historical_data, form_window)
    
    # 3. Calculer H2H history
    if verbose:
        print("   Calcul historique H2H...")
    h2h_history = calculate_h2h_history(historical_data, h2h_lookback)
    
    # 4. Statistiques générales
    all_teams = set(historical_data['HomeTeam']).union(set(historical_data['AwayTeam']))
    
    if target_season:
        last_season_data = df[df['Season'] == df['Season'].max()]
        season_info = {
            "target_season": target_season,
            "initialized_from": "full_history",
            "last_historical_season": df['Season'].max(),
            "historical_matches": len(historical_data),
            "last_match_date": historical_data['Date'].iloc[-1]
        }
    else:
        season_stats = get_season_end_stats(df, analyze_season)
        season_info = {
            "analyzed_season": analyze_season,
            "season_stats": season_stats,
            "historical_matches": len(historical_data),
            "cutoff_date": historical_data['Date'].iloc[-1]
        }
    
    # Construire état complet
    state = {
        "metadata": {
            **season_info,
            "teams": sorted(list(all_teams)),
            "n_teams": len(all_teams),
            "elo_k_factor": elo_k,
            "form_window": form_window,
            "h2h_lookback": h2h_lookback,
            "timestamp": datetime.now().isoformat()
        },
        "elo_ratings": elo_ratings,
        "form_ratings": form_ratings,
        "h2h_history": {
            # Convert defaultdict to regular dict for JSON serialization
            str(teams): {
                match_key: scores for match_key, scores in matches.items()
            } for teams, matches in h2h_history.items()
        },
        "team_stats": {}
    }
    
    # Statistiques par équipe
    for team in all_teams:
        team_matches = historical_data[
            (historical_data['HomeTeam'] == team) | (historical_data['AwayTeam'] == team)
        ]
        
        state["team_stats"][team] = {
            "elo_rating": elo_ratings.get(team, 1500),
            "form_rating": form_ratings.get(team, 1.5),
            "total_matches": len(team_matches),
            "home_matches": len(team_matches[team_matches['HomeTeam'] == team]),
            "away_matches": len(team_matches[team_matches['AwayTeam'] == team])
        }
    
    if verbose:
        print(f"✅ État initialisé pour {len(all_teams)} équipes")
        print(f"   Elo moyen: {np.mean(list(elo_ratings.values())):.1f}")
        print(f"   Form moyenne: {np.mean(list(form_ratings.values())):.2f}")
        print(f"   Matches H2H: {sum(len(matches) for teams_matches in h2h_history.values() for matches in teams_matches.values())}")
    
    return state

def save_season_state(state, output_path):
    """Sauvegarde état saison en JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    print(f"💾 État saison sauvegardé: {output_path}")

def generate_state_report(state, output_dir):
    """Génère rapport texte de l'état"""
    output_path = Path(output_dir) / "state_report.txt"
    
    with open(output_path, 'w') as f:
        f.write("Season State Report\n")
        f.write("=" * 30 + "\n\n")
        
        meta = state["metadata"]
        f.write(f"Metadata:\n")
        for key, value in meta.items():
            if key != "teams":  # Skip teams list (too long)
                f.write(f"  {key}: {value}\n")
        f.write(f"\n")
        
        # Top 10 Elo
        elo_ratings = state["elo_ratings"]
        top_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:10]
        f.write("Top 10 Elo Ratings:\n")
        for i, (team, rating) in enumerate(top_elo, 1):
            f.write(f"  {i:2d}. {team:<20} {rating:6.1f}\n")
        f.write("\n")
        
        # Top 10 Form
        form_ratings = state["form_ratings"]
        top_form = sorted(form_ratings.items(), key=lambda x: x[1], reverse=True)[:10]
        f.write("Top 10 Form Ratings:\n")
        for i, (team, rating) in enumerate(top_form, 1):
            f.write(f"  {i:2d}. {team:<20} {rating:4.2f}\n")
        f.write("\n")
    
    print(f"📄 Rapport état généré: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Oddsy Season State Initializer")
    parser.add_argument("--data", required=True, help="Dataset CSV path")
    
    # Mode selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target_season", help="Saison cible pour initialisation (ex: '2025-2026')")
    group.add_argument("--analyze_season", help="Saison existante à analyser (ex: '2024-2025')")
    
    parser.add_argument("--out_state", required=True, help="Fichier JSON état de sortie")
    parser.add_argument("--elo_k", type=int, default=32, help="K-factor Elo")
    parser.add_argument("--form_window", type=int, default=5, help="Fenêtre calcul form")
    parser.add_argument("--h2h_lookback", type=int, default=10, help="Matches H2H à conserver")
    parser.add_argument("--verbose", action="store_true", default=True, help="Mode verbose")
    
    args = parser.parse_args()
    
    print("🔧 Oddsy Season State Initializer")
    print(f"   Data: {args.data}")
    if args.target_season:
        print(f"   Target Season: {args.target_season}")
    if args.analyze_season:
        print(f"   Analyze Season: {args.analyze_season}")
    print(f"   Output: {args.out_state}")
    
    # Load data
    try:
        df = pd.read_csv(args.data)
        print(f"✅ Data loaded: {len(df)} matches")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    # Initialize state
    try:
        state = initialize_season_state(
            df, 
            target_season=args.target_season,
            analyze_season=args.analyze_season,
            elo_k=args.elo_k,
            form_window=args.form_window,
            h2h_lookback=args.h2h_lookback,
            verbose=args.verbose
        )
        
        save_season_state(state, args.out_state)
        
        # Generate report in same directory as state file
        output_dir = Path(args.out_state).parent
        generate_state_report(state, output_dir)
        
        print("🎉 Initialisation état terminée avec succès!")
        return 0
        
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())