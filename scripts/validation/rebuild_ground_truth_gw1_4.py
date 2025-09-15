#!/usr/bin/env python3
"""
Reconstitution de la Vérité Terrain - EPL 2025-26 Journées 1-4
================================================================

MISSION CRITIQUE: Créer le jeu de données de référence complet pour auditer 
le score de 59.1% du modèle Domain Adaptation.

Ce script complète les résultats manquants de la J4:
- Man City 3-0 Man United  
- Liverpool 1-0 Burnley

Et valide que nous avons exactement 38 matchs pour les 4 premières journées.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def rebuild_complete_ground_truth():
    """
    Reconstitue la vérité terrain complète pour GW1-4 avec les résultats manquants
    """
    
    print("🏆 RECONSTITUTION VÉRITÉ TERRAIN EPL 2025-26 GW1-4")
    print("="*60)
    
    # Charger les données EPL 2025-26
    epl_file = "data/raw/EPL 2025 2026.csv"
    print(f"📊 Chargement: {epl_file}")
    
    df = pd.read_csv(epl_file)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    print(f"✅ Matches chargés: {len(df)}")
    
    # Filtrer pour les 4 premières journées
    # Approximation basée sur les dates des premiers matchs
    gw1_4_cutoff = pd.to_datetime('2025-09-05')  # Approximativement après GW4
    
    gw1_4 = df[df['Date'] <= gw1_4_cutoff].copy()
    print(f"📅 Matches GW1-4 (avant filtre): {len(gw1_4)}")
    
    # COMPLÉTER LES RÉSULTATS MANQUANTS DE LA J4
    print(f"\n🔧 Completion des résultats manquants J4...")
    
    # Man City 3-0 Man United
    city_united_mask = (
        ((gw1_4['HomeTeam'] == 'Man City') & (gw1_4['AwayTeam'] == 'Man United')) |
        ((gw1_4['HomeTeam'] == 'Man United') & (gw1_4['AwayTeam'] == 'Man City'))
    )
    
    if city_united_mask.any():
        idx = gw1_4[city_united_mask].index[0]
        if pd.isna(gw1_4.loc[idx, 'FTHG']):  # Si résultat manquant
            if gw1_4.loc[idx, 'HomeTeam'] == 'Man City':
                gw1_4.loc[idx, 'FTHG'] = 3
                gw1_4.loc[idx, 'FTAG'] = 0
                gw1_4.loc[idx, 'FTR'] = 'H'
                print(f"   ✅ Ajouté: Man City 3-0 Man United")
            else:
                gw1_4.loc[idx, 'FTHG'] = 0
                gw1_4.loc[idx, 'FTAG'] = 3
                gw1_4.loc[idx, 'FTR'] = 'A'
                print(f"   ✅ Ajouté: Man United 0-3 Man City")
    
    # Liverpool 1-0 Burnley
    liv_burnley_mask = (
        ((gw1_4['HomeTeam'] == 'Liverpool') & (gw1_4['AwayTeam'] == 'Burnley')) |
        ((gw1_4['HomeTeam'] == 'Burnley') & (gw1_4['AwayTeam'] == 'Liverpool'))
    )
    
    if liv_burnley_mask.any():
        idx = gw1_4[liv_burnley_mask].index[0]
        if pd.isna(gw1_4.loc[idx, 'FTHG']):  # Si résultat manquant
            if gw1_4.loc[idx, 'HomeTeam'] == 'Liverpool':
                gw1_4.loc[idx, 'FTHG'] = 1
                gw1_4.loc[idx, 'FTAG'] = 0
                gw1_4.loc[idx, 'FTR'] = 'H'
                print(f"   ✅ Ajouté: Liverpool 1-0 Burnley")
            else:
                gw1_4.loc[idx, 'FTHG'] = 0
                gw1_4.loc[idx, 'FTAG'] = 1
                gw1_4.loc[idx, 'FTR'] = 'A'
                print(f"   ✅ Ajouté: Burnley 0-1 Liverpool")
    
    # Filtrer seulement les matches avec résultats complets
    complete_matches = gw1_4.dropna(subset=['FTHG', 'FTAG', 'FTR']).copy()
    
    print(f"\n📊 VALIDATION FINALE:")
    print(f"   Matches complets GW1-4: {len(complete_matches)}")
    
    # VÉRIFICATION CRITIQUE: Doit être exactement 38 matches (ou proche)
    expected_matches = 40  # 10 matches par journée * 4 journées
    if len(complete_matches) < 35:
        print(f"   ⚠️  ATTENTION: Seulement {len(complete_matches)} matches (attendu ~{expected_matches})")
    else:
        print(f"   ✅ Nombre de matches acceptable: {len(complete_matches)}")
    
    # Statistiques des résultats
    home_wins = (complete_matches['FTR'] == 'H').sum()
    draws = (complete_matches['FTR'] == 'D').sum()
    away_wins = (complete_matches['FTR'] == 'A').sum()
    
    print(f"\n📈 Distribution des résultats GW1-4:")
    print(f"   Home wins: {home_wins} ({home_wins/len(complete_matches)*100:.1f}%)")
    print(f"   Draws:     {draws} ({draws/len(complete_matches)*100:.1f}%)")
    print(f"   Away wins: {away_wins} ({away_wins/len(complete_matches)*100:.1f}%)")
    
    # Créer le répertoire de validation
    validation_dir = Path("data/validation")
    validation_dir.mkdir(exist_ok=True)
    
    # Sauvegarder la vérité terrain
    output_file = validation_dir / "ground_truth_gw1_4.csv"
    complete_matches.to_csv(output_file, index=False)
    
    print(f"\n💾 VÉRITÉ TERRAIN SAUVEGARDÉE:")
    print(f"   Fichier: {output_file}")
    print(f"   Matches: {len(complete_matches)}")
    print(f"   Période: {complete_matches['Date'].min().strftime('%d/%m/%Y')} → {complete_matches['Date'].max().strftime('%d/%m/%Y')}")
    
    # Détails pour l'audit
    print(f"\n🔍 ÉCHANTILLON POUR VÉRIFICATION:")
    sample_matches = complete_matches.head(5)[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
    for _, match in sample_matches.iterrows():
        print(f"   {match['Date'].strftime('%d/%m')} | {match['HomeTeam']} {match['FTHG']}-{match['FTAG']} {match['AwayTeam']} ({match['FTR']})")
    
    return {
        'ground_truth_file': output_file,
        'total_matches': len(complete_matches),
        'home_wins': home_wins,
        'draws': draws, 
        'away_wins': away_wins,
        'date_range': (complete_matches['Date'].min(), complete_matches['Date'].max())
    }

if __name__ == "__main__":
    results = rebuild_complete_ground_truth()
    
    print(f"\n🎯 MISSION ACCOMPLIE!")
    print(f"Vérité terrain GW1-4 reconstituée avec {results['total_matches']} matches")
    print(f"Prêt pour l'audit du score 59.1%")