#!/usr/bin/env python3
"""
Test Simple UnderstatAPI - Focus sur EPL
----------------------------------------
Test rapide et efficace de la couverture UnderstatAPI.
"""

import json
from datetime import datetime
from pathlib import Path

def test_understat_simple():
    """Test simple et direct d'UnderstatAPI"""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'api_name': 'UnderstatClient',
        'tests': {}
    }
    
    try:
        from understatapi import UnderstatClient
        client = UnderstatClient()
        print("✅ UnderstatClient importé et créé")
        
        # Test EPL
        print("\n🔍 Test EPL...")
        try:
            epl = client.league('EPL')
            print(f"✅ EPL endpoint créé: {type(epl)}")
            
            # Explorer méthodes disponibles
            methods = [m for m in dir(epl) if not m.startswith('_')]
            print(f"   Méthodes disponibles: {methods}")
            
            results['tests']['epl'] = {
                'status': 'endpoint_created',
                'endpoint_type': str(type(epl)),
                'available_methods': methods
            }
            
            # Test récupération données
            for method_name in ['get_match_data', 'get_season_data', 'matches', 'teams']:
                if hasattr(epl, method_name):
                    try:
                        print(f"   Tentative {method_name}()...")
                        data = getattr(epl, method_name)()
                        
                        if data:
                            print(f"   ✅ {method_name}: {len(data)} éléments")
                            
                            # Analyser premier élément
                            sample = data[0] if data else None
                            if sample:
                                if isinstance(sample, dict):
                                    print(f"      Clés: {list(sample.keys())[:5]}...")
                                    # Chercher xG
                                    xg_found = any('xg' in str(k).lower() for k in sample.keys())
                                    print(f"      xG data: {xg_found}")
                                
                                results['tests']['epl'][method_name] = {
                                    'status': 'success',
                                    'data_count': len(data),
                                    'sample_keys': list(sample.keys()) if isinstance(sample, dict) else None,
                                    'xg_available': xg_found if isinstance(sample, dict) else None,
                                    'sample_data': str(sample)[:200]
                                }
                            break  # On a trouvé des données, c'est suffisant
                        else:
                            print(f"   ❌ {method_name}: pas de données")
                            
                    except Exception as e:
                        print(f"   ❌ {method_name}: erreur {e}")
                        results['tests']['epl'][method_name] = {'status': 'error', 'error': str(e)}
            
        except Exception as e:
            print(f"❌ Erreur EPL: {e}")
            results['tests']['epl'] = {'status': 'failed', 'error': str(e)}
        
        # Test Championship (probablement pas disponible)
        print("\n🔍 Test Championship...")
        championship_available = False
        for league_name in ['Championship', 'EFL', 'Championship2']:
            try:
                champ = client.league(league_name)
                print(f"   ✅ {league_name} endpoint créé")
                championship_available = True
                break
            except Exception as e:
                print(f"   ❌ {league_name}: {e}")
        
        results['tests']['championship'] = {
            'status': 'available' if championship_available else 'not_available',
            'note': 'Understat focuses on top-tier leagues'
        }
        
        # Assessment global
        epl_working = 'epl' in results['tests'] and results['tests']['epl'].get('status') != 'failed'
        
        if epl_working:
            overall_status = 'good'
            recommendation = 'UnderstatAPI utilisable pour EPL - tester données réelles'
        else:
            overall_status = 'poor'
            recommendation = 'UnderstatAPI non opérationnel - utiliser SoccerData'
            
        results['assessment'] = {
            'status': overall_status,
            'epl_working': epl_working,
            'championship_available': championship_available,
            'recommendation': recommendation
        }
        
    except ImportError:
        print("❌ UnderstatAPI non installé")
        results = {
            'status': 'failed',
            'error': 'UnderstatAPI not installed',
            'assessment': {'status': 'failed', 'recommendation': 'Install UnderstatAPI or use alternative'}
        }
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        results['assessment'] = {'status': 'failed', 'error': str(e)}
    
    return results

def main():
    print("🧪 Test Simple UnderstatAPI")
    print("=" * 30)
    
    results = test_understat_simple()
    
    # Sauvegarder
    output_path = Path("test_results/understat_simple_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Résultats: {output_path}")
    
    # Résumé
    assessment = results.get('assessment', {})
    print(f"\n📊 RÉSUMÉ:")
    print(f"Status: {assessment.get('status', 'unknown')}")
    print(f"Recommandation: {assessment.get('recommendation', 'N/A')}")
    
    return 0 if assessment.get('status') in ['good', 'excellent'] else 1

if __name__ == "__main__":
    exit(main())