#!/usr/bin/env python3
"""
Test UnderstatAPI pour Coverage EPL 2025-26 et Championship 2024-25
------------------------------------------------------------------
Évalue la disponibilité et qualité des données xG depuis UnderstatAPI.

Usage:
    python test_understat_api.py --output test_results/understat_coverage.json
"""

import argparse
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_understat_coverage():
    """Test complet de la couverture UnderstatAPI"""
    
    try:
        # Tentative d'import
        from understatapi import UnderstatApi
        api = UnderstatApi()
        print("✅ UnderstatAPI importé avec succès")
    except ImportError as e:
        print("❌ UnderstatAPI non installé. Installation requise:")
        print("   pip install understatapi")
        return {
            'status': 'failed',
            'error': 'UnderstatAPI not installed',
            'recommendation': 'Install with: pip install understatapi',
            'epl_2025_26': {'status': 'failed', 'error': 'API not available'},
            'championship_2024_25': {'status': 'failed', 'error': 'API not available'},
            'overall_assessment': {'status': 'failed', 'error': 'API not available'}
        }
    except Exception as e:
        print(f"❌ Erreur initialisation UnderstatAPI: {e}")
        return {
            'status': 'failed', 
            'error': str(e)
        }

    results = {
        'timestamp': datetime.now().isoformat(),
        'api_name': 'UnderstatAPI',
        'epl_2025_26': {},
        'championship_2024_25': {},
        'overall_assessment': {}
    }

    # Test 1: EPL 2025-26 (Priorité critique)
    print("\n🔍 Test EPL 2025-26...")
    try:
        # Tester différentes façons d'accéder aux données EPL
        test_leagues = ['EPL', 'Premier League', 'English Premier League']
        epl_success = False
        
        for league_name in test_leagues:
            try:
                print(f"   Tentative avec league='{league_name}'...")
                
                # Test récupération matches 2025
                matches_2025 = api.get_league_matches(league_name, 2025)
                
                if matches_2025 and len(matches_2025) > 0:
                    epl_success = True
                    print(f"   ✅ Succès! {len(matches_2025)} matches trouvés")
                    
                    # Analyser qualité données
                    sample_match = matches_2025[0] if matches_2025 else {}
                    
                    results['epl_2025_26'] = {
                        'status': 'success',
                        'league_name_used': league_name,
                        'total_matches': len(matches_2025),
                        'sample_match_keys': list(sample_match.keys()) if sample_match else [],
                        'xg_available': 'xG' in str(sample_match) or 'expected_goals' in str(sample_match),
                        'data_sample': sample_match
                    }
                    break
                    
            except Exception as e:
                print(f"   ❌ Échec avec '{league_name}': {e}")
                continue
        
        if not epl_success:
            results['epl_2025_26'] = {
                'status': 'failed',
                'error': 'No working league name found for EPL 2025-26'
            }
            
    except Exception as e:
        print(f"❌ Erreur générale EPL test: {e}")
        results['epl_2025_26'] = {
            'status': 'failed',
            'error': str(e)
        }

    # Test 2: Championship 2024-25 (Important pour équipes promues)
    print("\n🔍 Test Championship 2024-25...")
    try:
        test_champ_names = ['Championship', 'EFL Championship', 'English Championship', 'Second Division']
        champ_success = False
        
        for champ_name in test_champ_names:
            try:
                print(f"   Tentative avec league='{champ_name}'...")
                
                # Test année 2024 et 2025 (parfois décalage dans APIs)
                for year in [2024, 2025]:
                    try:
                        matches_champ = api.get_league_matches(champ_name, year)
                        
                        if matches_champ and len(matches_champ) > 0:
                            champ_success = True
                            print(f"   ✅ Succès! {len(matches_champ)} matches Championship trouvés (année {year})")
                            
                            # Vérifier si on trouve les équipes promues
                            promoted_teams = ['Sunderland', 'Leeds', 'Sheffield United', 'Ipswich']
                            teams_found = []
                            
                            for match in matches_champ[:10]:  # Check first 10 matches
                                for team in promoted_teams:
                                    if team in str(match):
                                        teams_found.append(team)
                            
                            results['championship_2024_25'] = {
                                'status': 'success',
                                'league_name_used': champ_name,
                                'year_used': year,
                                'total_matches': len(matches_champ),
                                'promoted_teams_found': list(set(teams_found)),
                                'sample_match': matches_champ[0] if matches_champ else {},
                                'xg_available': 'xG' in str(matches_champ[0]) if matches_champ else False
                            }
                            break
                    except:
                        continue
                        
                if champ_success:
                    break
                    
            except Exception as e:
                print(f"   ❌ Échec avec '{champ_name}': {e}")
                continue
        
        if not champ_success:
            results['championship_2024_25'] = {
                'status': 'failed',
                'error': 'No working league name found for Championship 2024-25',
                'impact': 'Will need fallback source for promoted teams history'
            }
            
    except Exception as e:
        print(f"❌ Erreur générale Championship test: {e}")
        results['championship_2024_25'] = {
            'status': 'failed',
            'error': str(e)
        }

    # Test 3: Rate limits et capacités API
    print("\n🔍 Test capacités API...")
    try:
        # Test autres fonctions API disponibles
        api_functions = [func for func in dir(api) if not func.startswith('_')]
        
        results['api_capabilities'] = {
            'available_functions': api_functions,
            'estimated_rate_limit': 'Unknown - requires testing',
            'bulk_data_support': 'get_league_matches' in api_functions
        }
        
    except Exception as e:
        results['api_capabilities'] = {
            'error': str(e)
        }

    # Assessment global
    epl_ok = results['epl_2025_26'].get('status') == 'success'
    champ_ok = results['championship_2024_25'].get('status') == 'success'
    
    if epl_ok and champ_ok:
        overall_status = 'excellent'
        recommendation = 'UnderstatAPI couvre toutes nos besoins - utilisation recommandée'
    elif epl_ok and not champ_ok:
        overall_status = 'good'
        recommendation = 'UnderstatAPI parfait pour EPL, fallback requis pour Championship'
    elif not epl_ok:
        overall_status = 'poor'
        recommendation = 'UnderstatAPI insuffisant - utiliser SoccerData comme source primaire'
    else:
        overall_status = 'unknown'
        recommendation = 'Tests incomplets - réévaluation requise'
    
    results['overall_assessment'] = {
        'status': overall_status,
        'epl_coverage': epl_ok,
        'championship_coverage': champ_ok,
        'recommendation': recommendation,
        'priority_for_project': 'high' if epl_ok else 'low'
    }

    return results

def analyze_xg_quality(matches_sample):
    """Analyse la qualité des données xG dans un échantillon"""
    if not matches_sample:
        return {'status': 'no_data'}
    
    analysis = {
        'total_matches_analyzed': len(matches_sample),
        'xg_fields_found': [],
        'data_completeness': 0,
        'sample_values': {}
    }
    
    # Chercher champs xG dans premier match
    first_match = matches_sample[0] if matches_sample else {}
    
    potential_xg_fields = [
        'xG', 'expected_goals', 'home_xg', 'away_xg', 
        'xg_home', 'xg_away', 'npxG', 'deep'
    ]
    
    for field in potential_xg_fields:
        if field in str(first_match).lower():
            analysis['xg_fields_found'].append(field)
    
    # Calculer complétude
    complete_matches = 0
    for match in matches_sample:
        if any(field in str(match).lower() for field in potential_xg_fields):
            complete_matches += 1
    
    analysis['data_completeness'] = complete_matches / len(matches_sample) if matches_sample else 0
    analysis['sample_values'] = first_match
    
    return analysis

def save_results(results, output_path):
    """Sauvegarde les résultats du test"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Résultats sauvegardés: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Test UnderstatAPI Coverage")
    parser.add_argument("--output", default="test_results/understat_coverage.json", 
                       help="Fichier de sortie JSON")
    parser.add_argument("--verbose", action="store_true", help="Mode verbose")
    
    args = parser.parse_args()
    
    print("🧪 Test UnderstatAPI - Coverage EPL 2025-26 & Championship 2024-25")
    print("=" * 70)
    
    # Run test
    results = test_understat_coverage()
    
    # Save results
    save_results(results, args.output)
    
    # Summary
    print(f"\n📊 RÉSUMÉ TEST UNDERSTATAPI")
    print("=" * 30)
    
    epl_status = results['epl_2025_26'].get('status', 'unknown')
    champ_status = results['championship_2024_25'].get('status', 'unknown') 
    overall = results['overall_assessment'].get('status', 'unknown')
    
    print(f"EPL 2025-26: {epl_status}")
    print(f"Championship 2024-25: {champ_status}")
    print(f"Assessment global: {overall}")
    print(f"Recommandation: {results['overall_assessment'].get('recommendation', 'N/A')}")
    
    return 0 if overall in ['excellent', 'good'] else 1

if __name__ == "__main__":
    exit(main())