#!/usr/bin/env python3
"""
Fetch xG Data - Stratégie Hybride Optimale
------------------------------------------
Récupère données xG pour EPL 2025-26 + utilise datasets existants pour Championship.

Stratégie:
- EPL 2025-26: UnderstatAPI (vraies données xG) 
- Championship 2024-25: Dataset existant + estimation xG si nécessaire

Usage:
    python fetch_xg_hybrid_strategy.py --output data/enhanced/
"""

import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def fetch_epl_2025_26_from_understat():
    """Récupère données EPL 2025-26 avec xG depuis UnderstatAPI"""
    
    try:
        from understatapi import UnderstatClient
        client = UnderstatClient()
        epl = client.league('EPL')
        
        print("📊 Récupération EPL 2025-26 depuis UnderstatAPI...")
        
        # Récupérer données saison courante (format "2026" = saison 2025-26)
        matches = epl.get_match_data('2026')
        
        if not matches:
            return {'status': 'failed', 'error': 'No EPL 2025-26 data found'}
        
        print(f"✅ {len(matches)} matches EPL 2025-26 récupérés")
        
        # Convertir en DataFrame format cohérent
        processed_matches = []
        
        for match in matches:
            # Extraire infos de base
            home_team = match['h']['title'] if isinstance(match['h'], dict) else str(match['h'])
            away_team = match['a']['title'] if isinstance(match['a'], dict) else str(match['a'])
            
            # Extraire xG (données critiques!)
            xg_data = match.get('xG', {})
            if xg_data and isinstance(xg_data, dict):
                try:
                    home_xg = float(xg_data.get('h', 0)) if xg_data.get('h') is not None else 0
                    away_xg = float(xg_data.get('a', 0)) if xg_data.get('a') is not None else 0
                except (ValueError, TypeError):
                    home_xg = 0
                    away_xg = 0
            else:
                home_xg = 0
                away_xg = 0
            
            # Extraire résultat si disponible
            goals = match.get('goals', {})
            if goals and isinstance(goals, dict):
                try:
                    home_goals = int(goals.get('h')) if goals.get('h') is not None else None
                    away_goals = int(goals.get('a')) if goals.get('a') is not None else None
                except (ValueError, TypeError):
                    home_goals = None
                    away_goals = None
            else:
                home_goals = None
                away_goals = None
            
            # Déterminer résultat
            if home_goals is not None and away_goals is not None:
                if home_goals > away_goals:
                    result = 'H'
                elif away_goals > home_goals:
                    result = 'A'
                else:
                    result = 'D'
            else:
                result = None  # Match pas encore joué
            
            processed_match = {
                'date': match.get('datetime', ''),
                'season': '2025-2026',
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'result': result,
                'home_xg': home_xg,
                'away_xg': away_xg,
                'source': 'understat_api',
                'match_id': match.get('id')
            }
            
            processed_matches.append(processed_match)
        
        df_epl = pd.DataFrame(processed_matches)
        
        # Statistiques
        played_matches = len(df_epl[df_epl['result'].notna()])
        matches_with_xg = len(df_epl[df_epl['home_xg'] > 0])
        
        return {
            'status': 'success',
            'data': df_epl,
            'stats': {
                'total_matches': len(df_epl),
                'played_matches': played_matches,
                'matches_with_xg': matches_with_xg,
                'xg_coverage': matches_with_xg / len(df_epl) if len(df_epl) > 0 else 0
            }
        }
        
    except ImportError:
        return {'status': 'failed', 'error': 'UnderstatAPI not installed'}
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

def load_championship_2024_25_data():
    """Charge données Championship 2024-25 depuis dataset existant"""
    
    try:
        champ_file = Path("data/raw/Championship 2024 2025.csv")
        
        if not champ_file.exists():
            return {'status': 'failed', 'error': f'Championship file not found: {champ_file}'}
        
        print("📊 Chargement Championship 2024-25 depuis dataset local...")
        
        # Charger avec structure football-data.co.uk
        df_champ = pd.read_csv(champ_file)
        print(f"✅ {len(df_champ)} matches Championship chargés")
        
        # Filtrer seulement équipes promues (optimisation)
        promoted_teams = ['Sunderland', 'Leeds', 'Sheffield United', 'Ipswich', 'Southampton']
        
        df_promoted = df_champ[
            (df_champ['HomeTeam'].isin(promoted_teams)) | 
            (df_champ['AwayTeam'].isin(promoted_teams))
        ].copy()
        
        print(f"📋 {len(df_promoted)} matches avec équipes promues")
        
        # Convertir au format standard
        df_promoted['season'] = '2024-2025'
        df_promoted['home_team'] = df_promoted['HomeTeam']
        df_promoted['away_team'] = df_promoted['AwayTeam'] 
        df_promoted['home_goals'] = df_promoted['FTHG']
        df_promoted['away_goals'] = df_promoted['FTAG']
        df_promoted['result'] = df_promoted['FTR']
        df_promoted['date'] = pd.to_datetime(df_promoted['Date'], dayfirst=True).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Estimer xG depuis shots/corners (approximation)
        df_promoted['home_xg'] = estimate_xg_from_stats(
            df_promoted.get('HS', 0), 
            df_promoted.get('HST', 0),
            df_promoted.get('HC', 0)
        )
        df_promoted['away_xg'] = estimate_xg_from_stats(
            df_promoted.get('AS', 0),
            df_promoted.get('AST', 0), 
            df_promoted.get('AC', 0)
        )
        
        df_promoted['source'] = 'championship_dataset_estimated_xg'
        
        # Colonnes finales
        final_columns = ['date', 'season', 'home_team', 'away_team', 'home_goals', 
                        'away_goals', 'result', 'home_xg', 'away_xg', 'source']
        
        df_final = df_promoted[final_columns].copy()
        
        return {
            'status': 'success',
            'data': df_final,
            'stats': {
                'total_matches': len(df_final),
                'promoted_teams_matches': len(df_final),
                'xg_estimated': True,
                'source': 'local_dataset'
            }
        }
        
    except Exception as e:
        return {'status': 'failed', 'error': str(e)}

def estimate_xg_from_stats(shots, shots_on_target, corners):
    """Estime xG basique depuis statistiques disponibles"""
    
    # Modèle simple basé sur recherche empirique
    # Shots: ~0.11 xG par shot
    # Shots on target: bonus +0.04  
    # Corners: ~0.04 xG par corner
    
    shots = pd.to_numeric(shots, errors='coerce').fillna(0)
    shots_on_target = pd.to_numeric(shots_on_target, errors='coerce').fillna(0) 
    corners = pd.to_numeric(corners, errors='coerce').fillna(0)
    
    estimated_xg = (shots * 0.11 + 
                   shots_on_target * 0.04 + 
                   corners * 0.04)
    
    # Cap réaliste: 0.1 - 4.0 xG par match
    estimated_xg = np.clip(estimated_xg, 0.1, 4.0)
    
    return estimated_xg

def merge_all_xg_data(epl_result, champ_result):
    """Fusionne toutes les données xG en dataset unifié"""
    
    all_dataframes = []
    merge_stats = {
        'sources_used': [],
        'total_matches': 0,
        'real_xg_matches': 0,
        'estimated_xg_matches': 0
    }
    
    # EPL 2025-26 (priorité)
    if epl_result['status'] == 'success':
        df_epl = epl_result['data']
        all_dataframes.append(df_epl)
        merge_stats['sources_used'].append('understat_epl_2025_26')
        merge_stats['real_xg_matches'] += len(df_epl)
        print(f"✅ EPL 2025-26 ajouté: {len(df_epl)} matches avec vraies xG")
    else:
        print(f"❌ EPL 2025-26 échoué: {epl_result['error']}")
    
    # Championship 2024-25 
    if champ_result['status'] == 'success':
        df_champ = champ_result['data']
        all_dataframes.append(df_champ)
        merge_stats['sources_used'].append('championship_dataset_2024_25')
        merge_stats['estimated_xg_matches'] += len(df_champ)
        print(f"✅ Championship 2024-25 ajouté: {len(df_champ)} matches avec xG estimées")
    else:
        print(f"❌ Championship 2024-25 échoué: {champ_result['error']}")
    
    if not all_dataframes:
        return {'status': 'failed', 'error': 'No data sources succeeded'}
    
    # Fusionner tout
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    
    merge_stats['total_matches'] = len(df_combined)
    
    return {
        'status': 'success',
        'data': df_combined,
        'stats': merge_stats
    }

def save_enhanced_data(combined_result, output_dir):
    """Sauvegarde données enrichies xG"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if combined_result['status'] != 'success':
        print(f"❌ Pas de données à sauvegarder: {combined_result['error']}")
        return
    
    df = combined_result['data']
    stats = combined_result['stats']
    
    # 1. Dataset principal avec xG
    main_file = output_path / "xg_enhanced_data_2025_26.csv"
    df.to_csv(main_file, index=False)
    print(f"💾 Dataset principal: {main_file}")
    
    # 2. Rapport qualité
    quality_report = {
        'timestamp': datetime.now().isoformat(),
        'summary': stats,
        'data_quality': {
            'total_matches': len(df),
            'epl_2025_26_matches': len(df[df['season'] == '2025-2026']),
            'championship_2024_25_matches': len(df[df['season'] == '2024-2025']),
            'real_xg_coverage': stats['real_xg_matches'] / stats['total_matches'] if stats['total_matches'] > 0 else 0,
            'estimated_xg_coverage': stats['estimated_xg_matches'] / stats['total_matches'] if stats['total_matches'] > 0 else 0
        },
        'recommendations': {
            'use_for_features_v23': stats['real_xg_matches'] > 0,
            'quality_level': 'high' if stats['real_xg_matches'] > 300 else 'medium',
            'ready_for_production': True
        }
    }
    
    report_file = output_path / "xg_data_quality_report.json"
    with open(report_file, 'w') as f:
        json.dump(quality_report, f, indent=2)
    print(f"📊 Rapport qualité: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Fetch xG Data - Stratégie Hybride")
    parser.add_argument("--output", default="data/enhanced/", help="Répertoire de sortie")
    parser.add_argument("--verbose", action="store_true", default=True, help="Mode verbose")
    
    args = parser.parse_args()
    
    print("🚀 Récupération xG - Stratégie Hybride Optimale")
    print("=" * 55)
    print("EPL 2025-26: UnderstatAPI (vraies xG)")
    print("Championship 2024-25: Dataset local (xG estimées)")
    print()
    
    # Phase 1: EPL 2025-26 depuis UnderstatAPI
    epl_result = fetch_epl_2025_26_from_understat()
    
    # Phase 2: Championship 2024-25 depuis dataset local
    champ_result = load_championship_2024_25_data()
    
    # Phase 3: Fusion et sauvegarde
    print("\n🔗 Fusion des données...")
    combined_result = merge_all_xg_data(epl_result, champ_result)
    
    # Phase 4: Sauvegarde
    save_enhanced_data(combined_result, args.output)
    
    # Résumé final
    if combined_result['status'] == 'success':
        stats = combined_result['stats']
        print(f"\n🎉 SUCCESS!")
        print(f"   Total matches: {stats['total_matches']}")
        print(f"   Vraies xG: {stats['real_xg_matches']}")
        print(f"   xG estimées: {stats['estimated_xg_matches']}")
        print(f"   Sources: {', '.join(stats['sources_used'])}")
        return 0
    else:
        print(f"\n❌ ÉCHEC: {combined_result['error']}")
        return 1

if __name__ == "__main__":
    exit(main())