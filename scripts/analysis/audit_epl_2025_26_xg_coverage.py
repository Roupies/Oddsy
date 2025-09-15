#!/usr/bin/env python3
"""
Audit EPL 2025-26 xG Coverage - Identification Matches Sans xG
------------------------------------------------------------
Analyse précise des matches EPL 2025-26 joués vs données xG disponibles.
Identifie exactement quels matches manquent de vraies xG pour recherche ciblée.

Usage:
    python audit_epl_2025_26_xg_coverage.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_epl_2025_26_played():
    """Charge matches EPL 2025-26 joués avec résultats"""
    
    epl_file = Path("data/raw/EPL 2025 2026.csv")
    
    if not epl_file.exists():
        print(f"❌ Fichier EPL 2025-26 non trouvé: {epl_file}")
        return None
    
    print(f"📊 Chargement matches EPL 2025-26...")
    
    # Charger avec structure football-data.co.uk
    df = pd.read_csv(epl_file, encoding='utf-8-sig')
    print(f"   Total lignes: {len(df)}")
    
    # Filtrer seulement matches avec résultat (joués)
    df_played = df[df['FTR'].notna()].copy()
    print(f"✅ {len(df_played)} matches EPL joués trouvés")
    
    # Conversion format standard
    df_played['Date'] = pd.to_datetime(df_played['Date'], dayfirst=True).dt.strftime('%Y-%m-%d')
    df_played['Season'] = '2025-2026'
    
    # Colonnes essentielles
    df_result = df_played[['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()
    
    return df_result

def load_xg_enhanced_data():
    """Charge données xG enrichies disponibles"""
    
    xg_file = Path("data/enhanced/xg_enhanced_data_2025_26.csv")
    
    if not xg_file.exists():
        print(f"❌ Données xG enrichies non trouvées: {xg_file}")
        return None
    
    print(f"📊 Chargement données xG enrichies...")
    df_xg = pd.read_csv(xg_file)
    
    # Filtrer saison 2025-2026 uniquement
    df_xg_2025 = df_xg[df_xg['season'] == '2025-2026'].copy()
    
    # Filtrer seulement matches avec vraies xG (> 0)
    df_xg_real = df_xg_2025[
        (df_xg_2025['home_xg'] > 0) | (df_xg_2025['away_xg'] > 0)
    ].copy()
    
    # Standardiser dates
    df_xg_real['date'] = pd.to_datetime(df_xg_real['date']).dt.strftime('%Y-%m-%d')
    
    print(f"✅ {len(df_xg_real)} matches 2025-26 avec vraies xG")
    
    return df_xg_real

def standardize_team_names():
    """Mapping noms d'équipes EPL vs UnderstatAPI"""
    
    # Mapping équipes (EPL filename -> UnderstatAPI name)
    team_mapping = {
        # Équipes EPL standards
        'Arsenal': 'Arsenal',
        'Aston Villa': 'Aston Villa',
        'Brighton': 'Brighton',
        'Burnley': 'Burnley',
        'Chelsea': 'Chelsea',
        'Crystal Palace': 'Crystal Palace',
        'Everton': 'Everton',
        'Fulham': 'Fulham',
        'Liverpool': 'Liverpool',
        'Manchester City': 'Manchester City',
        'Manchester United': 'Manchester United',
        'Newcastle': 'Newcastle United',
        'Nottingham Forest': 'Nottingham Forest',
        'Sheffield United': 'Sheffield United',
        'Tottenham': 'Tottenham',
        'West Ham': 'West Ham',
        'Wolves': 'Wolverhampton Wanderers',
        'Brentford': 'Brentford',
        'Bournemouth': 'Bournemouth',
        
        # Équipes promues 2025-26 (à vérifier dans UnderstatAPI)
        'Leeds': 'Leeds United',
        'Leicester': 'Leicester City', 
        'Ipswich': 'Ipswich Town',
        'Southampton': 'Southampton',
        'Sunderland': 'Sunderland',  # Promu - peut manquer dans UnderstatAPI
        
        # Autres variations possibles
        'Man City': 'Manchester City',
        'Man United': 'Manchester United',
        'Spurs': 'Tottenham'
    }
    
    return team_mapping

def cross_reference_matches(df_epl, df_xg):
    """Cross-référence matches EPL vs données xG disponibles"""
    
    print("🔍 Cross-référence matches EPL vs xG...")
    
    team_mapping = standardize_team_names()
    
    # Préparer listes pour audit
    matches_with_xg = []
    matches_without_xg = []
    
    for idx, match in df_epl.iterrows():
        date = match['Date']
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Mapper noms équipes vers format UnderstatAPI
        home_mapped = team_mapping.get(home_team, home_team)
        away_mapped = team_mapping.get(away_team, away_team)
        
        # Chercher match correspondant dans données xG
        xg_match = df_xg[
            (df_xg['date'] == date) &
            (df_xg['home_team'] == home_mapped) &
            (df_xg['away_team'] == away_mapped)
        ]
        
        match_info = {
            'date': date,
            'home_team_epl': home_team,
            'away_team_epl': away_team,
            'home_team_mapped': home_mapped,
            'away_team_mapped': away_mapped,
            'home_goals': match['FTHG'],
            'away_goals': match['FTAG'],
            'result': match['FTR']
        }
        
        if len(xg_match) > 0:
            # Match trouvé avec xG
            xg_row = xg_match.iloc[0]
            match_info.update({
                'has_xg': True,
                'home_xg': xg_row['home_xg'],
                'away_xg': xg_row['away_xg'],
                'xg_source': xg_row['source'],
                'match_id': xg_row.get('match_id', None)
            })
            matches_with_xg.append(match_info)
        else:
            # Match sans xG
            match_info.update({
                'has_xg': False,
                'home_xg': None,
                'away_xg': None,
                'xg_source': None,
                'match_id': None
            })
            matches_without_xg.append(match_info)
    
    return matches_with_xg, matches_without_xg

def analyze_missing_patterns(matches_without_xg):
    """Analyse patterns dans matches sans xG pour cibler recherche"""
    
    if not matches_without_xg:
        return
    
    print(f"\n🔎 ANALYSE MATCHES SANS XG ({len(matches_without_xg)} matches)")
    print("=" * 60)
    
    # Équipes impliquées dans matches sans xG
    teams_missing = set()
    for match in matches_without_xg:
        teams_missing.add(match['home_team_epl'])
        teams_missing.add(match['away_team_epl'])
    
    print(f"Équipes impliquées dans matches sans xG:")
    for team in sorted(teams_missing):
        count = sum(1 for m in matches_without_xg 
                   if m['home_team_epl'] == team or m['away_team_epl'] == team)
        print(f"  - {team}: {count} match(es)")
    
    # Dates des matches manquants
    print(f"\nDates des matches sans xG:")
    dates_missing = sorted(set(m['date'] for m in matches_without_xg))
    for date in dates_missing:
        matches_on_date = [m for m in matches_without_xg if m['date'] == date]
        print(f"  - {date}: {len(matches_on_date)} match(es)")
    
    return teams_missing, dates_missing

def generate_search_recommendations(matches_without_xg, teams_missing):
    """Génère recommandations de recherche pour xG manquantes"""
    
    if not matches_without_xg:
        return
    
    print(f"\n💡 RECOMMANDATIONS DE RECHERCHE")
    print("=" * 40)
    
    # 1. Équipes promues problématiques
    promoted_teams = ['Sunderland', 'Leeds', 'Ipswich', 'Southampton', 'Leicester']
    promoted_missing = teams_missing.intersection(promoted_teams)
    
    if promoted_missing:
        print(f"🎯 PRIORITÉ 1: Équipes promues manquantes")
        for team in promoted_missing:
            print(f"   → {team}: Probablement pas encore dans UnderstatAPI")
            print(f"     Solutions: FBRef, Football-Data, SoccerData")
    
    # 2. Dates récentes vs anciennes
    recent_dates = [m['date'] for m in matches_without_xg if m['date'] >= '2025-08-01']
    old_dates = [m['date'] for m in matches_without_xg if m['date'] < '2025-08-01']
    
    if recent_dates:
        print(f"\n🎯 PRIORITÉ 2: Matches récents sans xG ({len(recent_dates)})")
        print(f"   Solutions: Vérifier délai mise à jour UnderstatAPI")
    
    if old_dates:
        print(f"\n🎯 PRIORITÉ 3: Matches anciens sans xG ({len(old_dates)})")
        print(f"   Solutions: Problème mapping équipes ou saison")
    
    # 3. Actions concrètes
    print(f"\n📋 ACTIONS CONCRÈTES:")
    print(f"1. Tester autres endpoints UnderstatAPI (seasons, match details)")
    print(f"2. Vérifier SoccerData pour équipes promues")
    print(f"3. Scraper FBRef pour matches spécifiques")
    print(f"4. Chercher datasets EPL 2025-26 avec xG complets")

def save_audit_results(matches_with_xg, matches_without_xg):
    """Sauvegarde résultats audit pour référence"""
    
    output_dir = Path("results/xg_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Matches avec xG
    if matches_with_xg:
        df_with_xg = pd.DataFrame(matches_with_xg)
        with_xg_file = output_dir / f"matches_with_xg_{timestamp}.csv"
        df_with_xg.to_csv(with_xg_file, index=False)
        print(f"💾 Matches avec xG: {with_xg_file}")
    
    # Matches sans xG (priorité pour recherche)
    if matches_without_xg:
        df_without_xg = pd.DataFrame(matches_without_xg)
        without_xg_file = output_dir / f"matches_WITHOUT_xg_{timestamp}.csv"
        df_without_xg.to_csv(without_xg_file, index=False)
        print(f"💾 Matches SANS xG: {without_xg_file}")
        print(f"   👆 FICHIER PRIORITÉ POUR RECHERCHE")
    
    # Rapport sommaire
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_epl_matches': len(matches_with_xg) + len(matches_without_xg),
        'matches_with_xg': len(matches_with_xg),
        'matches_without_xg': len(matches_without_xg),
        'xg_coverage_percent': len(matches_with_xg) / (len(matches_with_xg) + len(matches_without_xg)) * 100,
        'priority_for_search': len(matches_without_xg) > 0
    }
    
    summary_file = output_dir / f"xg_coverage_summary_{timestamp}.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📊 Rapport sommaire: {summary_file}")

def main():
    print("🔍 AUDIT EPL 2025-26 xG COVERAGE")
    print("=" * 45)
    print("Identification précise matches sans xG pour recherche ciblée")
    print()
    
    # Phase 1: Charger matches EPL 2025-26 joués
    print("📊 PHASE 1: Matches EPL 2025-26 joués")
    df_epl = load_epl_2025_26_played()
    if df_epl is None:
        return 1
    
    # Phase 2: Charger données xG disponibles
    print("\n📊 PHASE 2: Données xG disponibles")
    df_xg = load_xg_enhanced_data()
    if df_xg is None:
        print("⚠️  Aucune donnée xG - tous les matches à rechercher")
        # Créer liste complète comme sans xG
        matches_without_xg = []
        for idx, match in df_epl.iterrows():
            match_info = {
                'date': match['Date'],
                'home_team_epl': match['HomeTeam'],
                'away_team_epl': match['AwayTeam'],
                'home_goals': match['FTHG'],
                'away_goals': match['FTAG'],
                'result': match['FTR'],
                'has_xg': False
            }
            matches_without_xg.append(match_info)
        matches_with_xg = []
    else:
        # Phase 3: Cross-référence
        print("\n🔍 PHASE 3: Cross-référence matches")
        matches_with_xg, matches_without_xg = cross_reference_matches(df_epl, df_xg)
    
    # Phase 4: Analyse patterns manquants
    print(f"\n📈 PHASE 4: Analyse résultats")
    total_matches = len(matches_with_xg) + len(matches_without_xg)
    xg_coverage = len(matches_with_xg) / total_matches * 100 if total_matches > 0 else 0
    
    print(f"✅ Matches avec xG: {len(matches_with_xg)}")
    print(f"❌ Matches sans xG: {len(matches_without_xg)}")
    print(f"📊 Couverture xG: {xg_coverage:.1f}%")
    
    if matches_without_xg:
        teams_missing, dates_missing = analyze_missing_patterns(matches_without_xg)
        generate_search_recommendations(matches_without_xg, teams_missing)
    else:
        print("\n🎉 PARFAIT! Tous les matches EPL 2025-26 ont des xG!")
    
    # Phase 5: Sauvegarde
    print(f"\n💾 PHASE 5: Sauvegarde résultats")
    save_audit_results(matches_with_xg, matches_without_xg)
    
    # Résumé final
    print(f"\n{'🎉' if len(matches_without_xg) == 0 else '⚠️'} AUDIT TERMINÉ")
    if matches_without_xg:
        print(f"👉 {len(matches_without_xg)} matches à rechercher activement")
        print(f"📁 Voir results/xg_audit/ pour détails")
    else:
        print("✅ Toutes les xG disponibles - prêt pour intégration")
    
    return 0

if __name__ == "__main__":
    exit(main())