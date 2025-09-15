#!/usr/bin/env python3
"""
Build Enhanced v2.3 Features - Integration Complete EPL 2025-26
-------------------------------------------------------------
Fusionne EPL 2025-26 complet avec données xG enrichies et recalcule 
toutes les features v2.3 pour dataset unifié.

Pipeline:
1. Charge EPL 2025-26 depuis data/raw/Championship 2025 2026.csv 
2. Fusionne avec données xG enrichies 
3. Forge features xG, odds, shots/corners
4. Calcule features v2.3 (Elo, form, H2H, etc.)
5. Agrège avec dataset historique v13_xg_safe_features.csv

Usage:
    python build_enhanced_v23_features.py --output data/processed/v14_enhanced_features.csv
"""

import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_epl_2025_26_raw():
    """Charge données EPL 2025-26 depuis fichier raw"""
    
    raw_file = Path("data/raw/Championship 2025 2026.csv")
    
    if not raw_file.exists():
        print(f"❌ Fichier EPL 2025-26 non trouvé: {raw_file}")
        return None
    
    print(f"📊 Chargement EPL 2025-26 depuis {raw_file}...")
    
    # Charger avec structure football-data.co.uk
    df = pd.read_csv(raw_file, encoding='utf-8-sig')
    print(f"✅ {len(df)} matches EPL 2025-26 chargés")
    
    # Conversion format standard
    df_processed = pd.DataFrame({
        'Date': pd.to_datetime(df['Date'], dayfirst=True).dt.strftime('%Y-%m-%d'),
        'Season': '2025-2026',
        'HomeTeam': df['HomeTeam'],
        'AwayTeam': df['AwayTeam'],
        'FTHG': df['FTHG'],
        'FTAG': df['FTAG'],
        'FullTimeResult': df['FTR'],
        'HS': df['HS'],
        'AS': df['AS'], 
        'HST': df['HST'],
        'AST': df['AST'],
        'HC': df['HC'],
        'AC': df['AC'],
        # Odds betting
        'B365H': df['B365H'],
        'B365D': df['B365D'], 
        'B365A': df['B365A'],
        # Market averages
        'AvgH': df['AvgH'],
        'AvgD': df['AvgD'],
        'AvgA': df['AvgA']
    })
    
    # Filtrer seulement matches joués (avec résultat)
    df_played = df_processed[df_processed['FullTimeResult'].notna()].copy()
    print(f"📋 {len(df_played)} matches joués avec résultats")
    
    return df_played

def load_enhanced_xg_data():
    """Charge données xG enrichies créées précédemment"""
    
    xg_file = Path("data/enhanced/xg_enhanced_data_2025_26.csv")
    
    if not xg_file.exists():
        print(f"❌ Données xG enrichies non trouvées: {xg_file}")
        return None
    
    print(f"📊 Chargement données xG enrichies...")
    df_xg = pd.read_csv(xg_file)
    
    # Standardiser format dates
    df_xg['date'] = pd.to_datetime(df_xg['date']).dt.strftime('%Y-%m-%d')
    
    print(f"✅ {len(df_xg)} matches avec données xG chargés")
    return df_xg

def merge_epl_with_xg(df_epl, df_xg):
    """Fusionne données EPL 2025-26 avec données xG enrichies"""
    
    print("🔗 Fusion EPL 2025-26 avec données xG...")
    
    # Préparer merge sur date + équipes
    df_epl['merge_key'] = df_epl['Date'] + "_" + df_epl['HomeTeam'] + "_" + df_epl['AwayTeam']
    df_xg['merge_key'] = df_xg['date'] + "_" + df_xg['home_team'] + "_" + df_xg['away_team']
    
    # Merger avec données xG
    df_merged = df_epl.merge(
        df_xg[['merge_key', 'home_xg', 'away_xg', 'source']], 
        on='merge_key', 
        how='left'
    )
    
    # Stats fusion
    matches_with_xg = df_merged['home_xg'].notna().sum()
    print(f"✅ Fusion complète: {len(df_merged)} matches")
    print(f"   Avec xG: {matches_with_xg}")
    print(f"   Sans xG: {len(df_merged) - matches_with_xg}")
    
    # Estimer xG manquante si nécessaire
    missing_xg = df_merged['home_xg'].isna()
    if missing_xg.sum() > 0:
        print(f"⚠️  Estimation xG pour {missing_xg.sum()} matches...")
        
        df_merged.loc[missing_xg, 'home_xg'] = estimate_xg_from_stats(
            df_merged.loc[missing_xg, 'HS'],
            df_merged.loc[missing_xg, 'HST'],
            df_merged.loc[missing_xg, 'HC']
        )
        
        df_merged.loc[missing_xg, 'away_xg'] = estimate_xg_from_stats(
            df_merged.loc[missing_xg, 'AS'],
            df_merged.loc[missing_xg, 'AST'],
            df_merged.loc[missing_xg, 'AC']
        )
        
        df_merged.loc[missing_xg, 'source'] = 'estimated_from_shots'
    
    return df_merged

def estimate_xg_from_stats(shots, shots_on_target, corners):
    """Estime xG depuis statistiques shots/corners"""
    
    shots = pd.to_numeric(shots, errors='coerce').fillna(0)
    shots_on_target = pd.to_numeric(shots_on_target, errors='coerce').fillna(0)
    corners = pd.to_numeric(corners, errors='coerce').fillna(0)
    
    # Modèle empirique simple
    estimated_xg = (shots * 0.11 + 
                   shots_on_target * 0.04 + 
                   corners * 0.04)
    
    # Cap réaliste
    estimated_xg = np.clip(estimated_xg, 0.1, 4.0)
    
    return estimated_xg

def calculate_market_features(df):
    """Calcule features market entropy et probabilities"""
    
    print("💰 Calcul features market...")
    
    # Market entropy (incertitude du marché)
    def calculate_entropy(h_odds, d_odds, a_odds):
        # Convertir odds en probabilités
        h_prob = 1 / h_odds
        d_prob = 1 / d_odds  
        a_prob = 1 / a_odds
        
        # Normaliser 
        total = h_prob + d_prob + a_prob
        h_prob /= total
        d_prob /= total
        a_prob /= total
        
        # Entropy = -sum(p * log(p))
        entropy = -(h_prob * np.log(h_prob) + 
                   d_prob * np.log(d_prob) + 
                   a_prob * np.log(a_prob))
        
        return entropy
    
    # Utiliser odds moyennes pour robustesse
    df['market_entropy'] = df.apply(lambda x: calculate_entropy(
        x['AvgH'], x['AvgD'], x['AvgA']
    ), axis=1)
    
    # Normaliser entropy [0,1]
    max_entropy = np.log(3)  # Entropy maximale pour 3 classes équiprobables
    df['market_entropy_norm'] = df['market_entropy'] / max_entropy
    
    print(f"   Market entropy: min={df['market_entropy_norm'].min():.3f}, max={df['market_entropy_norm'].max():.3f}")
    
    return df

def calculate_team_stats_rolling(df):
    """Calcule statistiques roulantes par équipe (Elo, form, xG efficiency)"""
    
    print("📈 Calcul statistiques roulantes par équipe...")
    
    # Copie pour éviter modifications
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Listes équipes uniques
    all_teams = list(set(df['HomeTeam'].tolist() + df['AwayTeam'].tolist()))
    print(f"   Équipes détectées: {len(all_teams)}")
    
    # Initialiser features
    features_cols = [
        'home_form_5', 'away_form_5',
        'home_elo', 'away_elo',
        'home_xg_eff_5', 'away_xg_eff_5',
        'home_xg_eff_10', 'away_xg_eff_10',
        'home_goals_sum_5', 'away_goals_sum_5'
    ]
    
    for col in features_cols:
        df[col] = 0.5  # Valeur neutre par défaut
    
    # Initialiser Elo pour toutes les équipes
    elo_ratings = {team: 1500 for team in all_teams}
    
    # Historiques par équipe
    team_histories = {team: {
        'results': [],
        'goals_scored': [],
        'goals_conceded': [],
        'xg_for': [],
        'xg_against': []
    } for team in all_teams}
    
    for idx, match in df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        home_goals = match['FTHG']
        away_goals = match['FTAG']
        home_xg = match['home_xg']
        away_xg = match['away_xg']
        result = match['FullTimeResult']
        
        # AVANT le match: calculer features depuis historique
        
        # 1. Form (5 derniers matches)
        home_form = calculate_team_form(team_histories[home_team]['results'], window=5)
        away_form = calculate_team_form(team_histories[away_team]['results'], window=5)
        
        df.at[idx, 'home_form_5'] = home_form
        df.at[idx, 'away_form_5'] = away_form
        
        # 2. Elo ratings
        df.at[idx, 'home_elo'] = elo_ratings[home_team]
        df.at[idx, 'away_elo'] = elo_ratings[away_team]
        
        # 3. xG efficiency (5 et 10 derniers matches)
        home_xg_eff_5 = calculate_xg_efficiency(
            team_histories[home_team]['goals_scored'][-5:],
            team_histories[home_team]['xg_for'][-5:]
        )
        away_xg_eff_5 = calculate_xg_efficiency(
            team_histories[away_team]['goals_scored'][-5:],
            team_histories[away_team]['xg_for'][-5:]
        )
        
        home_xg_eff_10 = calculate_xg_efficiency(
            team_histories[home_team]['goals_scored'][-10:],
            team_histories[home_team]['xg_for'][-10:]
        )
        away_xg_eff_10 = calculate_xg_efficiency(
            team_histories[away_team]['goals_scored'][-10:],
            team_histories[away_team]['xg_for'][-10:]
        )
        
        df.at[idx, 'home_xg_eff_5'] = home_xg_eff_5
        df.at[idx, 'away_xg_eff_5'] = away_xg_eff_5
        df.at[idx, 'home_xg_eff_10'] = home_xg_eff_10
        df.at[idx, 'away_xg_eff_10'] = away_xg_eff_10
        
        # 4. Goals sum (5 derniers matches)
        home_goals_sum_5 = sum(team_histories[home_team]['goals_scored'][-5:])
        away_goals_sum_5 = sum(team_histories[away_team]['goals_scored'][-5:])
        
        df.at[idx, 'home_goals_sum_5'] = home_goals_sum_5
        df.at[idx, 'away_goals_sum_5'] = away_goals_sum_5
        
        # APRÈS le match: mettre à jour historiques
        
        # Résultat pour chaque équipe
        if result == 'H':
            home_result = 3  # Win
            away_result = 0  # Loss
        elif result == 'A':
            home_result = 0  # Loss
            away_result = 3  # Win
        else:  # Draw
            home_result = 1
            away_result = 1
        
        # Mettre à jour historiques
        team_histories[home_team]['results'].append(home_result)
        team_histories[home_team]['goals_scored'].append(home_goals)
        team_histories[home_team]['goals_conceded'].append(away_goals)
        team_histories[home_team]['xg_for'].append(home_xg)
        team_histories[home_team]['xg_against'].append(away_xg)
        
        team_histories[away_team]['results'].append(away_result)
        team_histories[away_team]['goals_scored'].append(away_goals)
        team_histories[away_team]['goals_conceded'].append(home_goals)
        team_histories[away_team]['xg_for'].append(away_xg)
        team_histories[away_team]['xg_against'].append(home_xg)
        
        # Mettre à jour Elo ratings
        elo_ratings[home_team], elo_ratings[away_team] = update_elo_ratings(
            elo_ratings[home_team], elo_ratings[away_team], result
        )
    
    print(f"   Statistiques calculées pour {len(df)} matches")
    return df

def calculate_team_form(results, window=5):
    """Calcule forme équipe sur fenêtre donnée"""
    if not results:
        return 0.5  # Neutre
    
    recent_results = results[-window:]
    if not recent_results:
        return 0.5
    
    # Points moyens normalisés [0,1]
    points = sum(recent_results) / len(recent_results) / 3.0
    return points

def calculate_xg_efficiency(goals, xg_values):
    """Calcule efficacité xG (goals/xG)"""
    if not goals or not xg_values or len(goals) != len(xg_values):
        return 0.5  # Neutre
    
    total_goals = sum(goals)
    total_xg = sum(xg_values)
    
    if total_xg == 0:
        return 0.5
    
    efficiency = total_goals / total_xg
    # Normaliser autour de 1.0 (efficiency parfaite)
    # Clip entre 0.1 et 2.0, puis map vers [0,1]
    efficiency = np.clip(efficiency, 0.1, 2.0)
    normalized = (efficiency - 0.1) / (2.0 - 0.1)
    
    return normalized

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

def calculate_h2h_features(df):
    """Calcule features Head-to-Head"""
    
    print("🏟️ Calcul features Head-to-Head...")
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    df['h2h_score'] = 0.5  # Neutre par défaut
    
    # Historique H2H par paire d'équipes
    h2h_history = {}
    
    for idx, match in df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        result = match['FullTimeResult']
        
        # Clé unique pour la paire (ordre important)
        h2h_key = f"{home_team}_vs_{away_team}"
        reverse_key = f"{away_team}_vs_{home_team}"
        
        # AVANT le match: calculer H2H score
        home_h2h_results = h2h_history.get(h2h_key, [])
        away_h2h_results = h2h_history.get(reverse_key, [])
        
        if home_h2h_results or away_h2h_results:
            # Combiner historiques (home favorisé légèrement)
            all_h2h = home_h2h_results + [-r for r in away_h2h_results]
            
            if all_h2h:
                h2h_score = sum(all_h2h) / len(all_h2h) / 3.0  # Normaliser [0,1]
                h2h_score = np.clip(h2h_score + 0.5, 0, 1)  # Centrer sur 0.5
                df.at[idx, 'h2h_score'] = h2h_score
        
        # APRÈS le match: mettre à jour historique
        if h2h_key not in h2h_history:
            h2h_history[h2h_key] = []
        
        # Points pour équipe domicile dans ce H2H
        if result == 'H':
            h2h_points = 3
        elif result == 'D':
            h2h_points = 1
        else:
            h2h_points = 0
        
        h2h_history[h2h_key].append(h2h_points)
    
    print(f"   H2H calculé pour {len(df)} matches")
    return df

def calculate_normalized_features(df):
    """Calcule features normalisées et différentielles"""
    
    print("📏 Calcul features normalisées...")
    
    # Features différentielles
    df['form_diff'] = df['home_form_5'] - df['away_form_5']
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    df['shots_diff'] = df['HS'] - df['AS']
    df['corners_diff'] = df['HC'] - df['AC']
    
    # Matchday (supposé séquentiel par saison)
    df['matchday'] = df.groupby('Season').cumcount() + 1
    
    # Normalisation [0,1] 
    def normalize_feature(series, center_on_zero=True):
        if series.std() == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        
        normalized = (series - series.min()) / (series.max() - series.min())
        
        if center_on_zero:
            # Pour features différentielles, centrer sur 0.5
            return normalized
        else:
            return normalized
    
    df['form_diff_normalized'] = normalize_feature(df['form_diff'])
    df['elo_diff_normalized'] = normalize_feature(df['elo_diff'])
    df['shots_diff_normalized'] = normalize_feature(df['shots_diff'])
    df['corners_diff_normalized'] = normalize_feature(df['corners_diff'])
    df['matchday_normalized'] = normalize_feature(df['matchday'], center_on_zero=False)
    
    print(f"   Features normalisées calculées")
    return df

def load_historical_features():
    """Charge dataset historique v13_xg_safe_features.csv"""
    
    historical_file = Path("data/processed/v13_xg_safe_features.csv")
    
    if not historical_file.exists():
        print(f"❌ Dataset historique non trouvé: {historical_file}")
        return None
    
    print(f"📊 Chargement dataset historique...")
    df_historical = pd.read_csv(historical_file)
    
    print(f"✅ {len(df_historical)} matches historiques chargés")
    print(f"   Colonnes: {list(df_historical.columns)}")
    
    return df_historical

def combine_with_historical(df_new, df_historical):
    """Combine nouveau dataset avec données historiques"""
    
    print("🔗 Combinaison avec données historiques...")
    
    # Colonnes communes pour la fusion
    common_columns = [
        'Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult',
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm', 'home_xg_eff_5', 'away_xg_eff_5',
        'home_xg_eff_10', 'away_xg_eff_10', 'home_goals_sum_5', 'away_goals_sum_5'
    ]
    
    # Vérifier colonnes disponibles
    missing_new = [col for col in common_columns if col not in df_new.columns]
    missing_historical = [col for col in common_columns if col not in df_historical.columns]
    
    if missing_new:
        print(f"⚠️  Colonnes manquantes dans nouveau dataset: {missing_new}")
    if missing_historical:
        print(f"⚠️  Colonnes manquantes dans historique: {missing_historical}")
    
    # Prendre seulement colonnes disponibles
    available_columns = [col for col in common_columns 
                        if col in df_new.columns and col in df_historical.columns]
    
    print(f"   Colonnes communes: {len(available_columns)}")
    
    # Combiner datasets avec normalisation des dates
    df_historical_subset = df_historical[available_columns].copy()
    df_new_subset = df_new[available_columns].copy()
    
    # Normaliser format des dates pour éviter erreur de tri
    df_historical_subset['Date'] = pd.to_datetime(df_historical_subset['Date']).dt.strftime('%Y-%m-%d')
    df_new_subset['Date'] = pd.to_datetime(df_new_subset['Date']).dt.strftime('%Y-%m-%d')
    
    df_combined = pd.concat([df_historical_subset, df_new_subset], ignore_index=True)
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)
    
    print(f"✅ Dataset combiné: {len(df_combined)} matches")
    print(f"   Historique: {len(df_historical_subset)}")
    print(f"   Nouveau: {len(df_new_subset)}")
    
    return df_combined

def save_enhanced_features(df, output_path):
    """Sauvegarde features enrichies v2.3"""
    
    print(f"💾 Sauvegarde dataset enrichi vers {output_path}...")
    
    # Créer répertoire si nécessaire
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder CSV
    df.to_csv(output_path, index=False)
    
    # Rapport de qualité
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_matches': len(df),
            'seasons': sorted(df['Season'].unique().tolist()),
            'date_range': [df['Date'].min(), df['Date'].max()],
            'teams': len(set(df['HomeTeam'].tolist() + df['AwayTeam'].tolist()))
        },
        'features_info': {
            'total_columns': len(df.columns),
            'feature_columns': [col for col in df.columns if col not in ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult']],
            'v23_production_features': [
                'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
                'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
                'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
            ]
        },
        'data_quality': {
            'missing_values': df.isnull().sum().sum(),
            'complete_cases': len(df.dropna()),
            'target_distribution': df['FullTimeResult'].value_counts().to_dict()
        }
    }
    
    report_path = output_path.parent / f"{output_path.stem}_quality_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Dataset sauvegardé: {output_path}")
    print(f"📊 Rapport qualité: {report_path}")
    print(f"   {len(df)} matches, {len(df.columns)} colonnes")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Build Enhanced v2.3 Features")
    parser.add_argument("--output", default="data/processed/v14_enhanced_features.csv", 
                       help="Fichier de sortie")
    
    args = parser.parse_args()
    
    print("🚀 BUILD ENHANCED V2.3 FEATURES")
    print("=" * 50)
    print("Integration EPL 2025-26 + données xG enrichies")
    print()
    
    # Phase 1: Charger données EPL 2025-26
    print("📊 PHASE 1: Chargement EPL 2025-26")
    df_epl = load_epl_2025_26_raw()
    if df_epl is None:
        return 1
    
    # Phase 2: Charger données xG enrichies
    print("\n📊 PHASE 2: Chargement données xG enrichies")
    df_xg = load_enhanced_xg_data()
    if df_xg is None:
        print("⚠️  Pas de données xG - continue sans")
        df_merged = df_epl.copy()
        # Estimer xG pour tous
        df_merged['home_xg'] = estimate_xg_from_stats(df_merged['HS'], df_merged['HST'], df_merged['HC'])
        df_merged['away_xg'] = estimate_xg_from_stats(df_merged['AS'], df_merged['AST'], df_merged['AC'])
        df_merged['source'] = 'estimated_all'
    else:
        # Phase 3: Fusion EPL + xG
        print("\n🔗 PHASE 3: Fusion EPL + données xG")
        df_merged = merge_epl_with_xg(df_epl, df_xg)
    
    # Phase 4: Calcul features market
    print("\n💰 PHASE 4: Calcul features market")
    df_merged = calculate_market_features(df_merged)
    
    # Phase 5: Calcul statistiques roulantes
    print("\n📈 PHASE 5: Calcul statistiques roulantes par équipe")
    df_merged = calculate_team_stats_rolling(df_merged)
    
    # Phase 6: Calcul H2H
    print("\n🏟️ PHASE 6: Calcul Head-to-Head")
    df_merged = calculate_h2h_features(df_merged)
    
    # Phase 7: Features normalisées
    print("\n📏 PHASE 7: Normalisation features")
    df_merged = calculate_normalized_features(df_merged)
    
    # Phase 8: Combiner avec historique
    print("\n🔗 PHASE 8: Combinaison avec données historiques")
    df_historical = load_historical_features()
    
    if df_historical is not None:
        df_final = combine_with_historical(df_merged, df_historical)
    else:
        print("⚠️  Pas de données historiques - utilise seulement nouveau dataset")
        df_final = df_merged
    
    # Phase 9: Sauvegarde
    print("\n💾 PHASE 9: Sauvegarde")
    output_path = save_enhanced_features(df_final, args.output)
    
    # Résumé final
    print(f"\n🎉 SUCCESS!")
    print(f"   Dataset enrichi créé: {output_path}")
    print(f"   Total matches: {len(df_final)}")
    print(f"   Saisons: {', '.join(sorted(df_final['Season'].unique()))}")
    print(f"   Features disponibles: {len(df_final.columns)}")
    
    return 0

if __name__ == "__main__":
    exit(main())