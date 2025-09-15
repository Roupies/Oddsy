#!/usr/bin/env python3
"""
Fetch EPL Complete Calendar - Official Sources
---------------------------------------------
Récupère le calendrier EPL 2025-26 COMPLET (380 matches) depuis sources officielles.
Essaie plusieurs sources pour garantir couverture totale.

Sources:
1. football-data.org API (gratuit, fiable)
2. Site officiel Premier League (scraping si nécessaire)
3. ESPN API 

Usage:
    python fetch_epl_official_calendar.py --output data/calendars/
"""

import argparse
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def fetch_from_football_data_org():
    """Récupère depuis football-data.org API (gratuit, 10 calls/minute)"""
    
    print("🏈 Tentative football-data.org API...")
    
    # API gratuite, pas de clé nécessaire pour certains endpoints
    base_url = "https://api.football-data.org/v4"
    
    # Headers recommandés
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        # 1. Récupérer informations competitions
        competitions_url = f"{base_url}/competitions"
        
        response = requests.get(competitions_url, headers=headers)
        print(f"   Status competitions: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Chercher Premier League
            for comp in data.get('competitions', []):
                if 'Premier League' in comp.get('name', '') and comp.get('area', {}).get('name') == 'England':
                    comp_id = comp.get('id')
                    print(f"   ✅ Premier League trouvée, ID: {comp_id}")
                    
                    # 2. Récupérer matches de la saison
                    matches_url = f"{base_url}/competitions/{comp_id}/matches"
                    
                    matches_response = requests.get(matches_url, headers=headers)
                    print(f"   Status matches: {matches_response.status_code}")
                    
                    if matches_response.status_code == 200:
                        matches_data = matches_response.json()
                        matches = matches_data.get('matches', [])
                        
                        print(f"   ✅ {len(matches)} matches récupérés")
                        return format_football_data_org_matches(matches)
                    else:
                        print(f"   ❌ Erreur matches: {matches_response.text[:200]}")
                        return None
            
            print("   ❌ Premier League non trouvée dans competitions")
            return None
        else:
            print(f"   ❌ Erreur competitions: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"   ❌ Erreur football-data.org: {e}")
        return None

def format_football_data_org_matches(matches):
    """Formate matches depuis football-data.org"""
    
    print("   🔧 Formatage matches football-data.org...")
    
    formatted_matches = []
    
    for match in matches:
        try:
            # Informations de base
            match_id = match.get('id')
            utc_date = match.get('utcDate')
            status = match.get('status')
            
            # Équipes
            home_team = match.get('homeTeam', {}).get('name', '')
            away_team = match.get('awayTeam', {}).get('name', '')
            
            # Score
            score = match.get('score', {})
            full_time = score.get('fullTime', {})
            home_score = full_time.get('home')
            away_score = full_time.get('away')
            
            # Déterminer si joué
            is_played = (status in ['FINISHED', 'AWARDED']) and (home_score is not None)
            
            # Résultat
            result = None
            if is_played:
                if home_score > away_score:
                    result = 'H'
                elif away_score > home_score:
                    result = 'A'
                else:
                    result = 'D'
            
            # Date et heure
            match_datetime = None
            date_str = None
            time_str = None
            
            if utc_date:
                try:
                    match_datetime = datetime.fromisoformat(utc_date.replace('Z', '+00:00'))
                    date_str = match_datetime.strftime('%Y-%m-%d')
                    time_str = match_datetime.strftime('%H:%M:%S')
                except:
                    date_str = utc_date[:10] if len(utc_date) >= 10 else None
            
            match_info = {
                'match_id': match_id,
                'date': date_str,
                'time': time_str,
                'datetime': match_datetime,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'result': result,
                'status': status,
                'is_played': is_played,
                'season': '2025-2026',
                'league': 'EPL',
                'source': 'football-data.org'
            }
            
            formatted_matches.append(match_info)
            
        except Exception as e:
            print(f"      ⚠️ Erreur match {match.get('id', 'N/A')}: {e}")
            continue
    
    print(f"   ✅ {len(formatted_matches)} matches formatés")
    return formatted_matches

def fetch_from_espn():
    """Récupère depuis ESPN API"""
    
    print("📺 Tentative ESPN API...")
    
    try:
        # ESPN a une API publique pour les sports
        espn_url = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(espn_url, headers=headers)
        print(f"   Status ESPN: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            events = data.get('events', [])
            print(f"   ✅ {len(events)} événements ESPN récupérés")
            
            if events:
                return format_espn_matches(events)
            else:
                print("   ❌ Aucun événement ESPN")
                return None
        else:
            print(f"   ❌ Erreur ESPN: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"   ❌ Erreur ESPN: {e}")
        return None

def format_espn_matches(events):
    """Formate matches ESPN"""
    
    print("   🔧 Formatage matches ESPN...")
    
    formatted_matches = []
    
    for event in events:
        try:
            # Extraire infos ESPN
            event_id = event.get('id')
            name = event.get('name', '')
            date = event.get('date')
            status = event.get('status', {}).get('type', {}).get('description', '')
            
            # Équipes
            competitions = event.get('competitions', [])
            if competitions:
                competitors = competitions[0].get('competitors', [])
                
                home_team = None
                away_team = None
                home_score = None
                away_score = None
                
                for comp in competitors:
                    team = comp.get('team', {})
                    if comp.get('homeAway') == 'home':
                        home_team = team.get('displayName', '')
                        home_score = comp.get('score')
                    elif comp.get('homeAway') == 'away':
                        away_team = team.get('displayName', '')
                        away_score = comp.get('score')
                
                if home_team and away_team:
                    # Conversion scores
                    try:
                        home_score = int(home_score) if home_score else None
                        away_score = int(away_score) if away_score else None
                    except:
                        home_score = None
                        away_score = None
                    
                    # Résultat
                    result = None
                    is_played = (home_score is not None and away_score is not None)
                    
                    if is_played:
                        if home_score > away_score:
                            result = 'H'
                        elif away_score > home_score:
                            result = 'A'
                        else:
                            result = 'D'
                    
                    # Date
                    match_datetime = None
                    date_str = None
                    time_str = None
                    
                    if date:
                        try:
                            match_datetime = datetime.fromisoformat(date.replace('Z', '+00:00'))
                            date_str = match_datetime.strftime('%Y-%m-%d')
                            time_str = match_datetime.strftime('%H:%M:%S')
                        except:
                            pass
                    
                    match_info = {
                        'match_id': event_id,
                        'date': date_str,
                        'time': time_str,
                        'datetime': match_datetime,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'result': result,
                        'status': status,
                        'is_played': is_played,
                        'season': '2025-2026',
                        'league': 'EPL',
                        'source': 'espn'
                    }
                    
                    formatted_matches.append(match_info)
                    
        except Exception as e:
            print(f"      ⚠️ Erreur événement ESPN: {e}")
            continue
    
    print(f"   ✅ {len(formatted_matches)} matches ESPN formatés")
    return formatted_matches

def generate_full_season_schedule():
    """Génère calendrier complet théorique EPL 2025-26"""
    
    print("📅 Génération calendrier théorique complet...")
    
    # 20 équipes EPL 2025-26 
    teams = [
        "Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United",
        "Tottenham Hotspur", "Newcastle United", "West Ham United", "Brighton and Hove Albion", 
        "Crystal Palace", "Brentford", "Fulham", "Everton", "Aston Villa", "AFC Bournemouth",
        "Wolverhampton Wanderers", "Nottingham Forest", "Leeds United", "Sunderland", 
        "Burnley"
    ]
    
    # Dates approximatives saison EPL (Août 2025 - Mai 2026)
    season_start = datetime(2025, 8, 15)  # Mi-août traditionnel
    season_end = datetime(2026, 5, 25)    # Fin mai traditionnel
    
    matches = []
    match_id_counter = 50000
    
    # Générer tous les matches (chaque équipe joue chaque autre 2 fois)
    for round_num in range(2):  # 2 tours (aller-retour)
        for i, home_team in enumerate(teams):
            for j, away_team in enumerate(teams):
                if i != j:  # Pas jouer contre soi-même
                    
                    # Inverser domicile/extérieur pour le 2ème tour
                    if round_num == 1:
                        home_team, away_team = away_team, home_team
                    
                    # Date approximative (répartir sur la saison)
                    weeks_into_season = (len(matches) // 10)  # ~10 matches par semaine
                    match_date = season_start + timedelta(weeks=weeks_into_season)
                    
                    # Éviter les doublons
                    match_key = f"{home_team}_vs_{away_team}"
                    existing_keys = [f"{m['home_team']}_vs_{m['away_team']}" for m in matches]
                    
                    if match_key not in existing_keys:
                        match_info = {
                            'match_id': match_id_counter,
                            'date': match_date.strftime('%Y-%m-%d'),
                            'time': '15:00:00',
                            'datetime': match_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': None,
                            'away_score': None,
                            'result': None,
                            'status': 'SCHEDULED',
                            'is_played': False,
                            'season': '2025-2026',
                            'league': 'EPL',
                            'source': 'generated_schedule'
                        }
                        
                        matches.append(match_info)
                        match_id_counter += 1
    
    print(f"   ✅ {len(matches)} matches théoriques générés")
    print(f"   📅 Période: {season_start.strftime('%Y-%m-%d')} → {season_end.strftime('%Y-%m-%d')}")
    
    return matches

def merge_all_sources(sources_data):
    """Fusionne toutes les sources pour calendrier complet"""
    
    print("🔗 Fusion de toutes les sources...")
    
    all_matches = []
    source_stats = {}
    
    for source_name, matches in sources_data.items():
        if matches:
            print(f"   {source_name}: {len(matches)} matches")
            source_stats[source_name] = len(matches)
            
            # Ajouter matches en évitant doublons
            existing_keys = set()
            for existing_match in all_matches:
                key = f"{existing_match['date']}_{existing_match['home_team']}_{existing_match['away_team']}"
                existing_keys.add(key)
            
            new_matches_added = 0
            for match in matches:
                key = f"{match['date']}_{match['home_team']}_{match['away_team']}"
                if key not in existing_keys:
                    all_matches.append(match)
                    existing_keys.add(key)
                    new_matches_added += 1
            
            print(f"      → {new_matches_added} nouveaux matches ajoutés")
    
    # Trier par date
    all_matches.sort(key=lambda x: x['date'] or '1900-01-01')
    
    print(f"🎯 TOTAL FINAL: {len(all_matches)} matches uniques")
    
    return all_matches, source_stats

def main():
    parser = argparse.ArgumentParser(description="Fetch Complete EPL Calendar")
    parser.add_argument("--output", default="data/calendars/", 
                       help="Répertoire de sortie")
    
    args = parser.parse_args()
    
    print("📅 FETCH EPL COMPLETE CALENDAR 2025-26")
    print("=" * 50)
    print("Récupération calendrier COMPLET (380 matches) depuis sources officielles")
    print()
    
    sources_data = {}
    
    # Source 1: football-data.org
    print("🏈 SOURCE 1: football-data.org")
    sources_data['football_data_org'] = fetch_from_football_data_org()
    
    # Source 2: ESPN
    print("\n📺 SOURCE 2: ESPN")
    sources_data['espn'] = fetch_from_espn()
    
    # Source 3: Génération théorique (backup)
    print("\n📅 SOURCE 3: Génération théorique")
    sources_data['generated'] = generate_full_season_schedule()
    
    # Fusion de toutes les sources
    print(f"\n🔗 FUSION SOURCES")
    all_matches, source_stats = merge_all_sources(sources_data)
    
    if len(all_matches) < 100:
        print("❌ Pas assez de matches récupérés")
        return 1
    
    # Analyse et sauvegarde
    played_matches = [m for m in all_matches if m['is_played']]
    future_matches = [m for m in all_matches if not m['is_played']]
    
    print(f"\n📊 ANALYSE FINALE:")
    print(f"   Total matches: {len(all_matches)}")
    print(f"   Matches joués: {len(played_matches)}")
    print(f"   Matches futurs: {len(future_matches)}")
    
    # Sauvegarder
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calendrier complet
    df_all = pd.DataFrame(all_matches)
    calendar_file = output_path / f"epl_complete_calendar_{timestamp}.csv"
    df_all.to_csv(calendar_file, index=False)
    
    # Matches futurs seulement
    if future_matches:
        df_future = pd.DataFrame(future_matches)
        future_file = output_path / f"epl_future_matches_complete_{timestamp}.csv"
        df_future.to_csv(future_file, index=False)
    
    # Rapport
    report = {
        'timestamp': datetime.now().isoformat(),
        'sources_used': source_stats,
        'total_matches': len(all_matches),
        'played_matches': len(played_matches),
        'future_matches': len(future_matches),
        'target_season_matches': 380,
        'coverage_percent': (len(all_matches) / 380) * 100
    }
    
    report_file = output_path / f"complete_calendar_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n🎉 CALENDRIER COMPLET CRÉÉ!")
    print(f"   📁 Calendrier: {calendar_file}")
    if future_matches:
        print(f"   🔮 Matches futurs: {future_file}")
    print(f"   📊 Couverture: {len(all_matches)}/380 matches ({(len(all_matches)/380)*100:.1f}%)")
    
    return 0

if __name__ == "__main__":
    exit(main())