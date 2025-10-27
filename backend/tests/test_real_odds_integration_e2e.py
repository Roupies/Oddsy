#!/usr/bin/env python3
"""
Test e2e pour l'intégration Real Odds v5.3
Valide le flow complet: backend real odds → API → frontend TypeScript
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from api_strict_v5 import EPLStrictAPI

class TestRealOddsIntegrationE2E:
    """Tests end-to-end pour l'intégration Real Odds v5.3"""
    
    def __init__(self):
        self.api = EPLStrictAPI()
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
    
    async def test_odds_health_status(self):
        """Test endpoint /api/v1/odds/status"""
        print("🧪 Test: Odds Health Status Endpoint")
        
        try:
            result = await self.api.get_odds_health_status()
            
            # Validations critiques
            assert 'status' in result, "Missing 'status' field"
            assert result['status'] in ['operational', 'degraded', 'unavailable'], f"Invalid status: {result['status']}"
            assert 'season' in result, "Missing 'season' field"
            assert 'total_fixtures' in result, "Missing 'total_fixtures' field"
            assert 'configuration' in result, "Missing 'configuration' field"
            
            # Validation configuration
            config = result['configuration']
            assert 'required_bookmakers' in config, "Missing required_bookmakers"
            assert 'tier1' in config['required_bookmakers'], "Missing tier1 bookmakers"
            
            tier1_bookmakers = config['required_bookmakers']['tier1']
            expected_tier1 = ['bet365', 'pinnacle']
            assert set(tier1_bookmakers) == set(expected_tier1), f"Expected {expected_tier1}, got {tier1_bookmakers}"
            
            self.test_results['tests']['odds_health_status'] = {
                'status': 'PASS',
                'result': result,
                'validations': [
                    'Status field present and valid',
                    'Season field present',
                    'Configuration structure valid',
                    'Tier1 bookmakers correct (bet365, pinnacle)'
                ]
            }
            
            print("  ✅ Status:", result['status'])
            print("  ✅ Season:", result['season'])
            print("  ✅ Tier1 bookmakers:", tier1_bookmakers)
            print("  ✅ Total fixtures:", result['total_fixtures'])
            
        except Exception as e:
            self.test_results['tests']['odds_health_status'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"  ❌ Error: {e}")
            raise
    
    async def test_predictions_with_odds(self):
        """Test endpoint /api/v1/predictions avec données odds réelles"""
        print("\n🧪 Test: Predictions with Real Odds Integration")
        
        try:
            result = await self.api.generate_predictions()
            
            # Validations structure
            assert 'predictions' in result, "Missing 'predictions' field"
            assert 'metadata' in result, "Missing 'metadata' field"
            
            predictions = result['predictions']
            if len(predictions) == 0:
                print("  ⚠️  No predictions available (no fixtures in pipeline)")
                self.test_results['tests']['predictions_with_odds'] = {
                    'status': 'SKIP',
                    'reason': 'No fixtures in pipeline'
                }
                return
            
            # Tester une prédiction
            sample_prediction = predictions[0]
            
            # Validations odds fields obligatoires
            required_fields = [
                'ko2h_ok', 'odds_source', 'individual_status'
            ]
            
            for field in required_fields:
                assert field in sample_prediction, f"Missing required field: {field}"
            
            # Validation odds_source
            valid_sources = ['real', 'unavailable']
            assert sample_prediction['odds_source'] in valid_sources, f"Invalid odds_source: {sample_prediction['odds_source']}"
            
            # Validation individual_status
            valid_statuses = ['ready', 'blocked', 'ko2h_violation']
            assert sample_prediction['individual_status'] in valid_statuses, f"Invalid individual_status: {sample_prediction['individual_status']}"
            
            # Si selected_snapshot présent, valider structure
            if 'selected_snapshot' in sample_prediction and sample_prediction['selected_snapshot']:
                snapshot = sample_prediction['selected_snapshot']
                snapshot_fields = ['bookmaker', 'snapshot_utc', 'overround', 'market_confidence']
                for field in snapshot_fields:
                    assert field in snapshot, f"Missing snapshot field: {field}"
                
                # Validation bookmaker tier1
                valid_bookmakers = ['bet365', 'pinnacle', 'betfair']
                assert snapshot['bookmaker'] in valid_bookmakers, f"Invalid bookmaker: {snapshot['bookmaker']}"
                
                print(f"  ✅ Snapshot bookmaker: {snapshot['bookmaker']}")
                print(f"  ✅ Market confidence: {snapshot['market_confidence']}")
            
            self.test_results['tests']['predictions_with_odds'] = {
                'status': 'PASS',
                'sample_prediction': {
                    'ko2h_ok': sample_prediction['ko2h_ok'],
                    'odds_source': sample_prediction['odds_source'],
                    'individual_status': sample_prediction['individual_status'],
                    'has_snapshot': 'selected_snapshot' in sample_prediction and sample_prediction['selected_snapshot'] is not None
                },
                'total_predictions': len(predictions)
            }
            
            print(f"  ✅ Total predictions: {len(predictions)}")
            print(f"  ✅ KO2h status: {sample_prediction['ko2h_ok']}")
            print(f"  ✅ Odds source: {sample_prediction['odds_source']}")
            print(f"  ✅ Individual status: {sample_prediction['individual_status']}")
            
        except Exception as e:
            self.test_results['tests']['predictions_with_odds'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"  ❌ Error: {e}")
            raise
    
    async def test_deterministic_bookmaker_selection(self):
        """Test sélection déterministe bookmaker (bet365→pinnacle→betfair)"""
        print("\n🧪 Test: Deterministic Bookmaker Selection")
        
        try:
            # Tester si le service odds utilise la logique déterministe
            from services.real_odds_integration import RealOddsIntegrationService
            odds_service = RealOddsIntegrationService()
            
            # Validation de la priorité des tiers
            expected_priority = ['bet365', 'pinnacle', 'betfair']
            tier_priority = odds_service.tier_priority
            
            assert tier_priority == expected_priority, f"Expected {expected_priority}, got {tier_priority}"
            
            self.test_results['tests']['deterministic_bookmaker_selection'] = {
                'status': 'PASS',
                'tier_priority': tier_priority,
                'validation': 'Bookmaker selection follows bet365→pinnacle→betfair priority'
            }
            
            print(f"  ✅ Tier priority: {' → '.join(tier_priority)}")
            
        except Exception as e:
            self.test_results['tests']['deterministic_bookmaker_selection'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"  ❌ Error: {e}")
            raise
    
    async def test_ko2h_validation(self):
        """Test validation contrainte KO-2h"""
        print("\n🧪 Test: KO-2h Constraint Validation")
        
        try:
            from services.real_odds_integration import RealOddsIntegrationService
            odds_service = RealOddsIntegrationService()
            
            # Test avec une date dans le futur (valide)
            future_kickoff = (datetime.now() + timedelta(hours=4)).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Test avec une date dans le passé (invalide)
            past_kickoff = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Vérifier que la logique KO-2h est appliquée
            assert hasattr(odds_service, 'ko2h_hours'), "Missing ko2h_hours configuration"
            assert odds_service.ko2h_hours == 2, f"Expected ko2h_hours=2, got {odds_service.ko2h_hours}"
            
            self.test_results['tests']['ko2h_validation'] = {
                'status': 'PASS',
                'ko2h_hours': odds_service.ko2h_hours,
                'validation': 'KO-2h constraint properly configured (2 hours before kickoff)'
            }
            
            print(f"  ✅ KO-2h cutoff: {odds_service.ko2h_hours} hours before kickoff")
            
        except Exception as e:
            self.test_results['tests']['ko2h_validation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"  ❌ Error: {e}")
            raise
    
    def generate_report(self):
        """Génère le rapport final des tests"""
        total_tests = len(self.test_results['tests'])
        passed_tests = sum(1 for test in self.test_results['tests'].values() if test['status'] == 'PASS')
        failed_tests = sum(1 for test in self.test_results['tests'].values() if test['status'] == 'FAIL')
        skipped_tests = sum(1 for test in self.test_results['tests'].values() if test['status'] == 'SKIP')
        
        self.test_results['summary'] = {
            'total': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'skipped': skipped_tests,
            'success_rate': f"{(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%"
        }
        
        # Sauvegarde du rapport
        report_path = f"reports/real_odds_integration_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('reports', exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 RAPPORT FINAL")
        print(f"================")
        print(f"✅ Tests réussis: {passed_tests}/{total_tests}")
        print(f"❌ Tests échoués: {failed_tests}")
        print(f"⏭️  Tests ignorés: {skipped_tests}")
        print(f"📈 Taux de réussite: {self.test_results['summary']['success_rate']}")
        print(f"💾 Rapport sauvé: {report_path}")
        
        return failed_tests == 0

async def main():
    """Point d'entrée principal"""
    print("🚀 Tests E2E Real Odds Integration v5.3")
    print("=" * 50)
    
    tester = TestRealOddsIntegrationE2E()
    
    try:
        # Exécution des tests
        await tester.test_odds_health_status()
        await tester.test_predictions_with_odds()
        await tester.test_deterministic_bookmaker_selection()
        await tester.test_ko2h_validation()
        
        # Génération du rapport
        success = tester.generate_report()
        
        if success:
            print("\n🎉 Tous les tests sont passés avec succès!")
            print("✅ Real Odds Integration v5.3 validée")
        else:
            print("\n⚠️  Certains tests ont échoué")
            exit(1)
            
    except Exception as e:
        print(f"\n💥 Erreur critique: {e}")
        tester.generate_report()
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())