#!/usr/bin/env python3
"""
Recherche xG Manquantes - Sources Alternatives
----------------------------------------------
Recherche active pour les 6 matches EPL 2025-26 sans xG identifiés.
Teste différentes sources et endpoints pour compléter la couverture.

Usage:
    python search_missing_xg.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Matches ciblés à rechercher
MISSING_MATCHES = [
    {'date': '2025-08-17', 'home': "Nott'm Forest", 'away': 'Brentford'},
    {'date': '2025-08-18', 'home': 'Leeds', 'away': 'Everton'},
    {'date': '2025-08-23', 'home': 'Arsenal', 'away': 'Leeds'},
    {'date': '2025-08-24', 'home': 'Crystal Palace', 'away': "Nott'm Forest"},
    {'date': '2025-08-30', 'home': 'Leeds', 'away': 'Newcastle'},
    {'date': '2025-08-31', 'home': "Nott'm Forest", 'away': 'West Ham'}
]

def test_understat_alternative_names():
    """Teste différentes variations de noms d'équipes UnderstatAPI"""
    
    print("🔍 Test variations noms équipes UnderstatAPI...")
    
    team_variations = {
        'Leeds': ['Leeds United', 'Leeds', 'Leeds Utd'],
        "Nott'm Forest": ['Nottingham Forest', "Nott'm Forest", 'Nottm Forest', 'Forest'],
        'Crystal Palace': ['Crystal Palace', 'Palace'],
        'Newcastle': ['Newcastle United', 'Newcastle', 'Newcastle Utd']
    }
    
    try:
        from understatapi import UnderstatClient
        client = UnderstatClient()
        epl = client.league('EPL')
        
        print("✅ UnderstatClient connecté")
        
        # Récupérer toutes les données saison 2026
        matches = epl.get_match_data('2026')
        print(f"   Total matches récupérés: {len(matches) if matches else 0}")
        
        if not matches:
            print("❌ Aucun match récupéré")
            return []
        
        found_matches = []
        
        for missing in MISSING_MATCHES:
            target_date = missing['date']
            home_team = missing['home']
            away_team = missing['away']
            
            print(f"\n🎯 Recherche: {home_team} vs {away_team} ({target_date})")
            
            # Tester toutes les variations
            home_variations = team_variations.get(home_team, [home_team])
            away_variations = team_variations.get(away_team, [away_team])
            
            match_found = False
            
            for match in matches:
                match_date = match.get('datetime', '')
                if target_date in match_date:
                    
                    match_home = match.get('h', {})
                    match_away = match.get('a', {})
                    
                    if isinstance(match_home, dict):
                        match_home_name = match_home.get('title', '')
                    else:
                        match_home_name = str(match_home)
                    
                    if isinstance(match_away, dict):
                        match_away_name = match_away.get('title', '')
                    else:
                        match_away_name = str(match_away)
                    
                    # Vérifier si noms correspondent
                    if (any(var in match_home_name for var in home_variations) and 
                        any(var in match_away_name for var in away_variations)):
                        
                        # Extraire xG
                        xg_data = match.get('xG', {})
                        if xg_data and isinstance(xg_data, dict):
                            home_xg = float(xg_data.get('h', 0)) if xg_data.get('h') is not None else 0
                            away_xg = float(xg_data.get('a', 0)) if xg_data.get('a') is not None else 0
                            
                            if home_xg > 0 or away_xg > 0:
                                print(f"   ✅ TROUVÉ! {match_home_name} vs {match_away_name}")
                                print(f"      xG: {home_xg} - {away_xg}")
                                
                                found_matches.append({
                                    'date': target_date,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'home_team_understat': match_home_name,
                                    'away_team_understat': match_away_name,
                                    'home_xg': home_xg,
                                    'away_xg': away_xg,
                                    'source': 'understat_alternative_search',
                                    'match_id': match.get('id')
                                })
                                match_found = True
                                break
            
            if not match_found:
                print(f"   ❌ Pas trouvé")
        
        return found_matches
        
    except ImportError:
        print("❌ UnderstatAPI non disponible")
        return []
    except Exception as e:
        print(f"❌ Erreur UnderstatAPI: {e}")
        return []

def test_soccerdata_source():
    """Teste SoccerData comme source alternative"""
    
    print("\n🔍 Test SoccerData comme source alternative...")
    
    try:
        import soccerdata as sd
        
        print("✅ SoccerData disponible")
        
        # Essayer EPL 2025-26
        epl = sd.FBref('EPL', seasons='2025-26')
        
        print("   Tentative récupération matches EPL 2025-26...")
        schedule = epl.read_schedule()
        
        if schedule is not None and not schedule.empty:
            print(f"   ✅ {len(schedule)} matches dans schedule")
            
            # Chercher matches manquants
            found_matches = []
            
            for missing in MISSING_MATCHES:
                target_date = missing['date']
                home_team = missing['home']
                away_team = missing['away']
                
                print(f"\n🎯 Recherche SoccerData: {home_team} vs {away_team}")
                
                # Note: SoccerData peut avoir des noms différents
                # Cette partie nécessiterait un mapping plus avancé
                
            return found_matches
        else:
            print("   ❌ Pas de données schedule récupérées")
            return []
            
    except ImportError:
        print("❌ SoccerData non installé")
        return []
    except Exception as e:
        print(f"❌ Erreur SoccerData: {e}")
        return []

def search_fbref_manual():
    """Suggestions pour recherche manuelle FBRef"""
    
    print("\n💡 RECHERCHE MANUELLE FBREF RECOMMANDÉE")
    print("=" * 50)
    
    print("🌐 URLs cibles FBRef pour matches manquants:")
    
    base_fbref = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    print(f"📋 Page principale: {base_fbref}")
    
    for missing in MISSING_MATCHES:
        date = missing['date']
        home = missing['home']
        away = missing['away']
        
        print(f"\n📅 {date}: {home} vs {away}")
        print(f"   → Chercher dans fixtures FBRef du {date}")
        
        # URLs spécifiques si disponibles
        if 'Leeds' in home or 'Leeds' in away:
            print(f"   → Équipe promue Leeds - vérifier couverture FBRef")
        if 'Forest' in home or 'Forest' in away:
            print(f"   → Nottingham Forest - équipe EPL établie")

def manual_xg_entry():
    """Interface pour saisie manuelle xG si trouvées"""
    
    print("\n📝 SAISIE MANUELLE XG (si trouvées via FBRef/autres)")
    print("=" * 60)
    
    manual_data = []
    
    for missing in MISSING_MATCHES:
        print(f"\n📅 {missing['date']}: {missing['home']} vs {missing['away']}")
        print("   Si vous avez trouvé les xG pour ce match:")
        print("   → Ajouter manuellement au fichier enhanced data")
        
        # Template pour ajout manuel
        template = {
            'date': missing['date'],
            'season': '2025-2026',
            'home_team': missing['home'],
            'away_team': missing['away'],
            'home_xg': 'À_REMPLIR',
            'away_xg': 'À_REMPLIR',
            'source': 'manual_entry_fbref'
        }
        
        manual_data.append(template)
    
    # Sauvegarder template
    template_file = Path("results/xg_audit/manual_xg_template.json")
    template_file.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(template_file, 'w') as f:
        json.dump(manual_data, f, indent=2)
    
    print(f"\n💾 Template sauvegardé: {template_file}")
    print("   → Compléter les xG trouvées manuellement")

def save_found_xg(found_matches):
    """Sauvegarde xG trouvées automatiquement"""
    
    if not found_matches:
        print("\n❌ Aucune xG supplémentaire trouvée automatiquement")
        return
    
    print(f"\n💾 Sauvegarde {len(found_matches)} xG trouvées...")
    
    df_found = pd.DataFrame(found_matches)
    
    output_file = Path("results/xg_audit/found_missing_xg.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df_found.to_csv(output_file, index=False)
    
    print(f"✅ xG trouvées sauvegardées: {output_file}")
    
    # Afficher résumé
    for match in found_matches:
        print(f"   ✅ {match['home_team']} vs {match['away_team']}: {match['home_xg']} - {match['away_xg']}")

def main():
    print("🔍 RECHERCHE xG MANQUANTES - SOURCES ALTERNATIVES")
    print("=" * 55)
    print(f"Target: {len(MISSING_MATCHES)} matches EPL 2025-26 sans xG")
    print()
    
    all_found = []
    
    # Stratégie 1: Variations noms UnderstatAPI
    print("📊 STRATÉGIE 1: Variations noms UnderstatAPI")
    found_understat = test_understat_alternative_names()
    all_found.extend(found_understat)
    
    # Stratégie 2: SoccerData
    print("\n📊 STRATÉGIE 2: SoccerData")
    found_soccerdata = test_soccerdata_source()
    all_found.extend(found_soccerdata)
    
    # Stratégie 3: Recommandations manuelles
    search_fbref_manual()
    manual_xg_entry()
    
    # Sauvegarde résultats
    if all_found:
        save_found_xg(all_found)
    
    # Bilan final
    remaining = len(MISSING_MATCHES) - len(all_found)
    
    print(f"\n📊 BILAN RECHERCHE:")
    print(f"   ✅ xG trouvées automatiquement: {len(all_found)}")
    print(f"   ❌ xG encore manquantes: {remaining}")
    
    if remaining > 0:
        print(f"\n👉 ACTIONS SUIVANTES:")
        print(f"   1. Recherche manuelle FBRef pour {remaining} matches")
        print(f"   2. Compléter template manual_xg_template.json")
        print(f"   3. Procéder avec {24 + len(all_found)} matches validés")
    else:
        print(f"\n🎉 PARFAIT! Toutes les xG récupérées!")
    
    return len(all_found)

if __name__ == "__main__":
    exit(main())