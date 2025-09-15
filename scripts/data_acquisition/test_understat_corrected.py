#!/usr/bin/env python3
"""
Test UnderstatAPI (Corrigé) pour Coverage EPL 2025-26 et Championship 2024-25
-----------------------------------------------------------------------------
Évalue la disponibilité et qualité des données xG depuis UnderstatClient.

Usage:
    python test_understat_corrected.py --output test_results/understat_coverage.json
"""

import argparse
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_understat_coverage():
    """Test complet de la couverture UnderstatAPI avec UnderstatClient"""
    
    try:
        # Import correct
        from understatapi import UnderstatClient
        client = UnderstatClient()
        print("✅ UnderstatClient créé avec succès")
    except ImportError as e:
        print(f"❌ UnderstatAPI non disponible: {e}")
        return {
            'status': 'failed',
            'error': 'UnderstatAPI not available',
            'epl_2025_26': {'status': 'failed', 'error': 'API not available'},
            'championship_2024_25': {'status': 'failed', 'error': 'API not available'},
            'overall_assessment': {'status': 'failed', 'error': 'API not available'}
        }
    except Exception as e:
        print(f"❌ Erreur initialisation UnderstatClient: {e}")
        return {
            'status': 'failed', 
            'error': str(e),
            'epl_2025_26': {'status': 'failed', 'error': str(e)},
            'championship_2024_25': {'status': 'failed', 'error': str(e)},
            'overall_assessment': {'status': 'failed', 'error': str(e)}
        }

    results = {
        'timestamp': datetime.now().isoformat(),
        'api_name': 'UnderstatAPI (UnderstatClient)',
        'epl_2025_26': {},
        'championship_2024_25': {},
        'overall_assessment': {}
    }

    # Test 1: EPL 2025-26 (Priorité critique)
    print("\n🔍 Test EPL 2025-26...")
    try:
        # Tester différentes années pour EPL
        test_years = [2025, 2024, 2023]  # 2025 puis fallback
        epl_success = False
        
        for year in test_years:
            try:
                print(f"   Tentative EPL année {year}...")
                
                # Understat utilise généralement 'EPL' pour Premier League
                league_data = client.league(league='EPL', year=year)
                
                if league_data and hasattr(league_data, 'get_match_data'):
                    matches = league_data.get_match_data()
                    
                    if matches and len(matches) > 0:
                        epl_success = True
                        print(f"   ✅ Succès EPL {year}! {len(matches)} matches trouvés")
                        
                        # Analyser structure données
                        sample_match = matches[0] if matches else {}
                        
                        results['epl_2025_26'] = {
                            'status': 'success',
                            'year_found': year,
                            'total_matches': len(matches),
                            'sample_match_keys': list(sample_match.keys()) if isinstance(sample_match, dict) else [],
                            'xg_available': check_xg_availability(sample_match),
                            'data_sample': str(sample_match)[:500]  # Tronqué pour JSON
                        }
                        
                        # Si c'est 2025, c'est parfait
                        if year == 2025:
                            break
                        
                except Exception as e:
                    print(f"   ❌ Échec EPL {year}: {e}")
                    continue
        
        if not epl_success:
            results['epl_2025_26'] = {
                'status': 'failed',
                'error': 'No EPL data found for any test year',
                'years_tested': test_years
            }
            
    except Exception as e:
        print(f"❌ Erreur générale EPL test: {e}")
        results['epl_2025_26'] = {
            'status': 'failed',
            'error': str(e)
        }

    # Test 2: Championship (plus difficile, peut ne pas être disponible)
    print("\n🔍 Test Championship...")
    try:
        # Tester différents noms pour Championship
        test_leagues = ['Championship', 'EFL_Championship', 'Second_Division']
        test_years = [2025, 2024, 2023]
        champ_success = False
        
        for league_name in test_leagues:
            for year in test_years:
                try:
                    print(f"   Tentative {league_name} {year}...")
                    
                    league_data = client.league(league=league_name, year=year)
                    
                    if league_data and hasattr(league_data, 'get_match_data'):
                        matches = league_data.get_match_data()
                        
                        if matches and len(matches) > 0:
                            champ_success = True
                            print(f"   ✅ Succès {league_name} {year}! {len(matches)} matches")
                            
                            # Chercher équipes promues connues
                            promoted_teams_found = find_promoted_teams_in_data(matches)
                            
                            results['championship_2024_25'] = {
                                'status': 'success',
                                'league_name': league_name,
                                'year_found': year,
                                'total_matches': len(matches),
                                'promoted_teams_found': promoted_teams_found,
                                'xg_available': check_xg_availability(matches[0]) if matches else False,
                                'data_sample': str(matches[0])[:300] if matches else 'No data'
                            }
                            break
                            
                except Exception as e:
                    print(f"   ❌ Échec {league_name} {year}: {e}")
                    continue
                    
            if champ_success:
                break
        
        if not champ_success:
            results['championship_2024_25'] = {
                'status': 'failed',
                'error': 'Championship data not available on Understat',
                'note': 'Understat primarily focuses on top-tier leagues',
                'fallback_required': True
            }
            
    except Exception as e:
        print(f"❌ Erreur générale Championship test: {e}")
        results['championship_2024_25'] = {
            'status': 'failed',
            'error': str(e)
        }

    # Assessment global
    epl_ok = results['epl_2025_26'].get('status') == 'success'
    champ_ok = results['championship_2024_25'].get('status') == 'success'
    
    if epl_ok and champ_ok:
        overall_status = 'excellent'
        recommendation = 'UnderstatAPI couvre tous nos besoins - source principale recommandée'
    elif epl_ok and not champ_ok:
        overall_status = 'good'  
        recommendation = 'UnderstatAPI excellent pour EPL, chercher alternative pour Championship'
    elif not epl_ok:
        overall_status = 'poor'
        recommendation = 'UnderstatAPI insuffisant - tester SoccerData comme alternative'
    else:
        overall_status = 'unknown'
        recommendation = 'Résultats ambigus - investigation supplémentaire requise'
    
    results['overall_assessment'] = {
        'status': overall_status,
        'epl_coverage': epl_ok,
        'championship_coverage': champ_ok,
        'recommendation': recommendation,
        'priority_for_project': 'high' if epl_ok else 'low'
    }

    return results

def check_xg_availability(data_sample):
    """Vérifie si les données xG sont disponibles dans l'échantillon"""
    if not data_sample:
        return False
    
    data_str = str(data_sample).lower()
    xg_indicators = ['xg', 'expected_goals', 'npxg', 'xg_home', 'xg_away']
    
    return any(indicator in data_str for indicator in xg_indicators)

def find_promoted_teams_in_data(matches):
    """Cherche les équipes promues connues dans les données"""
    promoted_teams = ['Sunderland', 'Leeds', 'Sheffield United', 'Ipswich', 'Southampton']
    teams_found = []
    
    matches_str = str(matches).lower()
    
    for team in promoted_teams:
        if team.lower() in matches_str:
            teams_found.append(team)
    
    return teams_found

def save_results(results, output_path):
    """Sauvegarde les résultats du test"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Résultats sauvegardés: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Test UnderstatAPI Coverage (Corrected)")
    parser.add_argument("--output", default="test_results/understat_coverage_corrected.json",
                       help="Fichier de sortie JSON")
    
    args = parser.parse_args()
    
    print("🧪 Test UnderstatClient - Coverage EPL 2025-26 & Championship 2024-25")
    print("=" * 75)
    
    # Run test
    results = test_understat_coverage()
    
    # Save results
    save_results(results, args.output)
    
    # Summary
    print(f"\n📊 RÉSUMÉ TEST UNDERSTAT (CORRIGÉ)")
    print("=" * 35)
    
    epl_status = results['epl_2025_26'].get('status', 'unknown')
    champ_status = results['championship_2024_25'].get('status', 'unknown')
    overall = results['overall_assessment'].get('status', 'unknown')
    
    print(f"EPL Coverage: {epl_status}")
    print(f"Championship Coverage: {champ_status}")
    print(f"Overall Assessment: {overall}")
    print(f"Recommendation: {results['overall_assessment'].get('recommendation', 'N/A')}")
    
    return 0 if overall in ['excellent', 'good'] else 1

if __name__ == "__main__":
    exit(main())