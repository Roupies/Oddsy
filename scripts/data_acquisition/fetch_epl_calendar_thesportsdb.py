#!/usr/bin/env python3
"""
Fetch EPL Calendar 2025-26 - TheSportsDB API
--------------------------------------------
Récupère le calendrier complet EPL 2025-26 via TheSportsDB API
pour préparer les prédictions des matches futurs.

Usage:
    python fetch_epl_calendar_thesportsdb.py --output data/calendars/
"""

import argparse
import requests
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def fetch_epl_league_info():
    """Récupère infos league EPL depuis TheSportsDB"""
    
    print("🔍 Utilisation ID EPL connu...")
    
    # Premier League ID connu dans TheSportsDB
    epl_league_id = "4328"
    
    # Vérifier que la league existe
    url = f"https://www.thesportsdb.com/api/v1/json/3/lookupleague.php?id={epl_league_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'leagues' in data and data['leagues']:
            league = data['leagues'][0]
            print(f"✅ EPL confirmée: {league.get('strLeague', 'Premier League')}")
            print(f"   ID: {epl_league_id}")
            print(f"   Pays: {league.get('strCountry', 'England')}")
            print(f"   Sport: {league.get('strSport', 'Soccer')}")
            return epl_league_id
        else:
            print("❌ League EPL non confirmée")
            return None
        
    except Exception as e:
        print(f"❌ Erreur vérification league: {e}")
        # Retourner ID quand même car c'est l'ID standard
        print("ℹ️  Utilisation ID standard EPL")
        return epl_league_id

def fetch_epl_2025_26_schedule(league_id):
    """Récupère calendrier EPL 2025-26 complet avec tous les formats"""
    
    print(f"📅 Récupération calendrier EPL saison 2025-2026...")
    
    # Essayer différents formats de saison
    season_formats = [
        "2025-2026",  # Format standard
        "2025",       # Format année
        "2024-2025",  # Saison précédente (au cas où)
        "25-26",      # Format court
        "2526"        # Format compact
    ]
    
    all_events = []
    
    for season_format in season_formats:
        print(f"   Tentative format: {season_format}")
        
        url = f"https://www.thesportsdb.com/api/v1/json/3/eventsseason.php?id={league_id}&s={season_format}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'events' in data and data['events']:
                events = data['events']
                print(f"   ✅ {len(events)} événements trouvés avec format {season_format}")
                
                # Filtrer pour éviter les doublons
                new_events = []
                existing_ids = {event.get('idEvent') for event in all_events}
                
                for event in events:
                    if event.get('idEvent') not in existing_ids:
                        new_events.append(event)
                
                all_events.extend(new_events)
                print(f"   📝 {len(new_events)} nouveaux événements ajoutés")
            else:
                print(f"   ❌ Aucun événement avec format {season_format}")
                
        except Exception as e:
            print(f"   ❌ Erreur avec format {season_format}: {e}")
            continue
    
    if all_events:
        print(f"🎯 TOTAL: {len(all_events)} événements EPL récupérés")
        
        # Vérifier si on a une saison complète (380 matches attendus)
        if len(all_events) < 300:  # Seuil minimal pour saison complète
            print("⚠️  Nombre de matches insuffisant pour saison complète")
            print("🔄 Tentative récupération par équipes...")
            
            # Essayer de récupérer par équipes
            team_events = fetch_events_by_teams(league_id)
            if team_events:
                # Fusionner avec événements existants
                existing_ids = {event.get('idEvent') for event in all_events}
                for event in team_events:
                    if event.get('idEvent') not in existing_ids:
                        all_events.append(event)
                        
                print(f"🔗 FINAL: {len(all_events)} événements après récupération par équipes")
        
        return all_events
    else:
        print("❌ Aucun événement récupéré avec tous les formats")
        return None

def fetch_events_by_teams(league_id):
    """Récupère événements par équipes pour compléter le calendrier"""
    
    print("🔄 Récupération par équipes EPL...")
    
    # Équipes EPL 2025-26 connues
    epl_teams = [
        "Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United",
        "Tottenham", "Newcastle", "West Ham", "Brighton", "Crystal Palace",
        "Brentford", "Fulham", "Everton", "Aston Villa", "Bournemouth",
        "Wolves", "Nottingham Forest", "Leeds", "Sunderland", "Burnley"
    ]
    
    all_team_events = []
    
    # Essayer quelques équipes principales seulement (pour éviter trop d'API calls)
    key_teams = ["Arsenal", "Liverpool", "Manchester City", "Chelsea", "Leeds"]
    
    for team in key_teams:
        print(f"   Recherche matches {team}...")
        
        # API recherche par équipe
        search_url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={team.replace(' ', '%20')}"
        
        try:
            response = requests.get(search_url)
            response.raise_for_status()
            data = response.json()
            
            if 'teams' in data and data['teams']:
                team_info = data['teams'][0]
                team_id = team_info.get('idTeam')
                
                if team_id:
                    # Récupérer événements de cette équipe pour 2025
                    events_url = f"https://www.thesportsdb.com/api/v1/json/3/eventslast.php?id={team_id}"
                    
                    resp = requests.get(events_url)
                    resp.raise_for_status()
                    events_data = resp.json()
                    
                    if 'results' in events_data and events_data['results']:
                        team_events = events_data['results']
                        
                        # Filtrer pour EPL 2025-26 seulement
                        epl_events = [e for e in team_events 
                                     if e.get('strLeague') == 'English Premier League' 
                                     and '2025' in str(e.get('dateEvent', ''))]
                        
                        all_team_events.extend(epl_events)
                        print(f"      {len(epl_events)} matches EPL 2025 trouvés")
                    
        except Exception as e:
            print(f"      ❌ Erreur équipe {team}: {e}")
            continue
    
    print(f"✅ {len(all_team_events)} événements récupérés par équipes")
    return all_team_events

def process_epl_events(events):
    """Traite événements EPL en format standardisé"""
    
    print("🔧 Traitement événements EPL...")
    
    processed_matches = []
    
    for event in events:
        try:
            # Informations de base
            event_id = event.get('idEvent')
            date_str = event.get('dateEvent')
            time_str = event.get('strTime', '00:00:00')
            
            # Équipes
            home_team = event.get('strHomeTeam', '')
            away_team = event.get('strAwayTeam', '')
            
            # Scores (si joué)
            home_score = event.get('intHomeScore')
            away_score = event.get('intAwayScore')
            
            # Statut match
            status = event.get('strStatus', '')
            
            # Déterminer si joué
            is_played = (home_score is not None and away_score is not None)
            
            # Construire datetime
            if date_str and time_str:
                try:
                    datetime_str = f"{date_str} {time_str}"
                    match_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                except:
                    # Format de temps alternatif
                    try:
                        match_datetime = datetime.strptime(date_str, '%Y-%m-%d')
                    except:
                        match_datetime = None
            else:
                match_datetime = None
            
            # Déterminer résultat
            result = None
            if is_played:
                if int(home_score) > int(away_score):
                    result = 'H'
                elif int(away_score) > int(home_score):
                    result = 'A'
                else:
                    result = 'D'
            
            match_info = {
                'event_id': event_id,
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
                'league': 'EPL'
            }
            
            processed_matches.append(match_info)
            
        except Exception as e:
            print(f"   ⚠️  Erreur traitement événement {event.get('idEvent', 'N/A')}: {e}")
            continue
    
    print(f"✅ {len(processed_matches)} matches traités")
    return processed_matches

def analyze_calendar_coverage(matches):
    """Analyse couverture calendrier"""
    
    print("📊 Analyse couverture calendrier...")
    
    df = pd.DataFrame(matches)
    
    # Statistiques générales
    total_matches = len(df)
    played_matches = len(df[df['is_played'] == True])
    future_matches = len(df[df['is_played'] == False])
    
    print(f"   Total matches: {total_matches}")
    print(f"   Matches joués: {played_matches}")
    print(f"   Matches futurs: {future_matches}")
    
    # Équipes uniques
    all_teams = set(df['home_team'].tolist() + df['away_team'].tolist())
    print(f"   Équipes détectées: {len(all_teams)}")
    
    # Équipes promues identifiées
    promoted_teams = ['Leeds', 'Leicester', 'Ipswich', 'Southampton', 'Sunderland']
    promoted_found = [team for team in all_teams if any(promo in team for promo in promoted_teams)]
    
    if promoted_found:
        print(f"   Équipes promues trouvées: {promoted_found}")
    
    # Distribution temporelle
    if played_matches > 0:
        played_df = df[df['is_played'] == True].copy()
        if not played_df.empty and played_df['date'].notna().any():
            date_range = f"{played_df['date'].min()} → {played_df['date'].max()}"
            print(f"   Période matches joués: {date_range}")
    
    if future_matches > 0:
        future_df = df[df['is_played'] == False].copy()
        if not future_df.empty and future_df['date'].notna().any():
            next_matches = future_df.sort_values('date').head(5)
            print(f"   Prochains matches:")
            for _, match in next_matches.iterrows():
                print(f"      {match['date']}: {match['home_team']} vs {match['away_team']}")
    
    return {
        'total_matches': total_matches,
        'played_matches': played_matches,
        'future_matches': future_matches,
        'teams_count': len(all_teams),
        'teams_list': sorted(all_teams),
        'promoted_teams_found': promoted_found
    }

def save_calendar_data(matches, stats, output_dir):
    """Sauvegarde données calendrier"""
    
    print("💾 Sauvegarde calendrier EPL 2025-26...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Calendrier complet
    df_calendar = pd.DataFrame(matches)
    calendar_file = output_path / f"epl_2025_26_calendar_{timestamp}.csv"
    df_calendar.to_csv(calendar_file, index=False)
    print(f"   📅 Calendrier complet: {calendar_file}")
    
    # 2. Matches futurs seulement (pour prédictions)
    df_future = df_calendar[df_calendar['is_played'] == False].copy()
    if not df_future.empty:
        future_file = output_path / f"epl_2025_26_future_matches_{timestamp}.csv"
        df_future.to_csv(future_file, index=False)
        print(f"   🔮 Matches futurs: {future_file}")
        print(f"      {len(df_future)} matches à prédire")
    
    # 3. Rapport statistiques
    report = {
        'timestamp': datetime.now().isoformat(),
        'source': 'TheSportsDB API',
        'season': '2025-2026',
        'league': 'EPL',
        'statistics': stats,
        'files_generated': {
            'calendar_complete': str(calendar_file),
            'future_matches': str(future_file) if not df_future.empty else None
        }
    }
    
    report_file = output_path / f"calendar_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"   📊 Rapport: {report_file}")
    
    return calendar_file, future_file if not df_future.empty else None

def main():
    parser = argparse.ArgumentParser(description="Fetch EPL Calendar 2025-26")
    parser.add_argument("--output", default="data/calendars/", 
                       help="Répertoire de sortie")
    
    args = parser.parse_args()
    
    print("📅 FETCH EPL CALENDAR 2025-26 - THESPORTSDB")
    print("=" * 50)
    print("Récupération calendrier complet pour prédictions")
    print()
    
    # Phase 1: Trouver league EPL
    print("🔍 PHASE 1: Recherche league EPL")
    league_id = fetch_epl_league_info()
    if not league_id:
        print("❌ Impossible de trouver league EPL")
        return 1
    
    # Phase 2: Récupérer calendrier 2025-26
    print(f"\n📅 PHASE 2: Récupération calendrier saison 2025-26")
    events = fetch_epl_2025_26_schedule(league_id)
    if not events:
        print("❌ Impossible de récupérer calendrier")
        return 1
    
    # Phase 3: Traitement événements
    print(f"\n🔧 PHASE 3: Traitement événements")
    matches = process_epl_events(events)
    if not matches:
        print("❌ Aucun match traité")
        return 1
    
    # Phase 4: Analyse couverture
    print(f"\n📊 PHASE 4: Analyse couverture")
    stats = analyze_calendar_coverage(matches)
    
    # Phase 5: Sauvegarde
    print(f"\n💾 PHASE 5: Sauvegarde")
    calendar_file, future_file = save_calendar_data(matches, stats, args.output)
    
    # Résumé final
    print(f"\n🎉 CALENDRIER EPL 2025-26 RÉCUPÉRÉ!")
    print(f"   📁 Calendrier: {calendar_file}")
    if future_file:
        print(f"   🔮 Matches futurs: {future_file}")
    print(f"   📊 {stats['total_matches']} matches, {stats['future_matches']} à prédire")
    print(f"   ⚽ {stats['teams_count']} équipes, {len(stats['promoted_teams_found'])} promues identifiées")
    
    return 0

if __name__ == "__main__":
    exit(main())