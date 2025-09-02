#!/usr/bin/env python3
"""
test_period_validator.py

VALIDATION AUTOMATIQUE DES PÉRIODES DE TEST

OBJECTIF: Empêcher DÉFINITIVEMENT les erreurs de data leakage temporel
- Valider que période de test est POSTÉRIEURE à l'entraînement du modèle
- Vérifier qu'il n'y a AUCUN overlap entre train/test
- Alerter si période de test semble suspecte

USAGE: Import dans tous les scripts de test
"""

import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import json
import os

class TestPeriodValidator:
    """Validateur anti-data-leakage pour périodes de test"""
    
    # PERIODS CRITIQUES CONNUES (à ne jamais utiliser pour test)
    KNOWN_TRAINING_PERIODS = {
        'v2.1_model': {
            'train_end': '2024-05-19',  # Modèle v2.1 entraîné jusqu'à mai 2024
            'description': 'Modèle clean_xg_model_traditional_baseline trained on data up to 2024-05-19'
        },
        'v13_model': {
            'train_end': '2023-05-28',  # Modèle v13 entraîné jusqu'à mai 2023
            'description': 'Modèle v13_production trained on data up to 2023-05-28'
        }
    }
    
    def __init__(self):
        self.validation_log = []
    
    def validate_test_period(self, 
                           test_start: str, 
                           test_end: str,
                           model_info: Optional[Dict] = None,
                           model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Valide que la période de test est SAFE (pas de data leakage)
        
        Returns:
            Dict avec validation_passed (bool) et détails
        """
        
        test_start_date = pd.to_datetime(test_start)
        test_end_date = pd.to_datetime(test_end)
        today = pd.Timestamp.now()
        
        validation_result = {
            'validation_passed': True,
            'test_period': f"{test_start} to {test_end}",
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # 1. VALIDATION BASIQUE
        if test_start_date >= test_end_date:
            validation_result['errors'].append(
                f"❌ ERREUR: test_start ({test_start}) >= test_end ({test_end})"
            )
            validation_result['validation_passed'] = False
        
        # 2. VALIDATION PÉRIODE FUTURE
        if test_end_date > today:
            validation_result['warnings'].append(
                f"⚠️ WARNING: Test period ends in future ({test_end} > today). "
                f"Make sure this data actually exists!"
            )
        
        # 3. VALIDATION MODÈLE SPÉCIFIQUE
        if model_name and model_name in self.KNOWN_TRAINING_PERIODS:
            model_train_end = pd.to_datetime(self.KNOWN_TRAINING_PERIODS[model_name]['train_end'])
            
            if test_start_date <= model_train_end:
                validation_result['errors'].append(
                    f"🚨 DATA LEAKAGE DÉTECTÉ: Test start ({test_start}) <= "
                    f"model training end ({self.KNOWN_TRAINING_PERIODS[model_name]['train_end']}) "
                    f"for {model_name}"
                )
                validation_result['validation_passed'] = False
                
                # Recommandation automatique
                safe_start = (model_train_end + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
                validation_result['recommendations'].append(
                    f"💡 RECOMMANDATION: Utiliser test_start >= {safe_start} "
                    f"(1 mois après fin training pour éviter overlap)"
                )
        
        # 4. VALIDATION PÉRIODE 2024-2025 (pour nos modèles actuels)
        current_season_start = pd.to_datetime('2024-08-01')
        if test_start_date < current_season_start:
            validation_result['warnings'].append(
                f"⚠️ WARNING: Test period starts before 2024-08-01 season. "
                f"Ensure this is not overlapping with training data used for current models."
            )
        
        # 5. VALIDATION LONGUEUR PÉRIODE
        period_days = (test_end_date - test_start_date).days
        if period_days < 30:
            validation_result['warnings'].append(
                f"⚠️ WARNING: Test period is only {period_days} days. "
                f"Consider longer period for more robust evaluation."
            )
        elif period_days > 400:
            validation_result['warnings'].append(
                f"⚠️ WARNING: Test period is {period_days} days (>1 year). "
                f"This might span multiple seasons with different dynamics."
            )
        
        # Log de validation
        self.validation_log.append({
            'timestamp': datetime.now().isoformat(),
            'test_period': validation_result['test_period'],
            'passed': validation_result['validation_passed'],
            'model_name': model_name
        })
        
        return validation_result
    
    def get_safe_test_period_2024_2025(self) -> Tuple[str, str]:
        """
        Retourne une période de test SAFE pour 2024-2025
        (garantie sans data leakage pour nos modèles actuels)
        """
        # Période safe: Août 2024 - Mai 2025 (saison 2024-2025)
        return ('2024-08-01', '2025-05-31')
    
    def print_validation_report(self, validation_result: Dict[str, Any]) -> None:
        """Affiche un rapport de validation lisible"""
        
        print("🔒 VALIDATION PÉRIODE DE TEST")
        print("="*50)
        print(f"📅 Période: {validation_result['test_period']}")
        print(f"✅ Status: {'PASS' if validation_result['validation_passed'] else '❌ FAIL'}")
        
        if validation_result['errors']:
            print("\n🚨 ERREURS CRITIQUES:")
            for error in validation_result['errors']:
                print(f"  {error}")
        
        if validation_result['warnings']:
            print("\n⚠️ WARNINGS:")
            for warning in validation_result['warnings']:
                print(f"  {warning}")
        
        if validation_result['recommendations']:
            print("\n💡 RECOMMANDATIONS:")
            for rec in validation_result['recommendations']:
                print(f"  {rec}")
        
        if not validation_result['validation_passed']:
            print(f"\n🛑 TEST BLOQUÉ - PÉRIODE NON VALIDE!")
            print(f"🔧 Corrigez les erreurs avant de continuer.")
        else:
            print(f"\n✅ TEST AUTORISÉ - PÉRIODE VALIDÉE")

# FONCTION HELPER POUR USAGE FACILE
def validate_test_period_safe(test_start: str, test_end: str, 
                            model_name: Optional[str] = None,
                            auto_print: bool = True) -> bool:
    """
    Fonction helper pour validation rapide
    Returns True si safe, False sinon
    """
    validator = TestPeriodValidator()
    result = validator.validate_test_period(test_start, test_end, model_name=model_name)
    
    if auto_print:
        validator.print_validation_report(result)
    
    return result['validation_passed']

# DÉCORATEUR POUR PROTÉGER LES FONCTIONS DE TEST
def require_safe_test_period(model_name: Optional[str] = None):
    """Décorateur pour forcer validation période avant test"""
    def decorator(func):
        def wrapper(*args, test_start: str, test_end: str, **kwargs):
            if not validate_test_period_safe(test_start, test_end, model_name, auto_print=True):
                raise ValueError("🚨 DATA LEAKAGE DÉTECTÉ - Test bloqué par validation automatique!")
            return func(*args, test_start=test_start, test_end=test_end, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test du validateur
    validator = TestPeriodValidator()
    
    print("🧪 TESTS DU VALIDATEUR:")
    print()
    
    # Test 1: Période invalide (data leakage)
    print("1. Test avec data leakage (2023-2024 vs modèle v13):")
    result1 = validator.validate_test_period('2023-08-01', '2024-05-31', model_name='v13_model')
    validator.print_validation_report(result1)
    print()
    
    # Test 2: Période safe (2024-2025)
    print("2. Test avec période safe (2024-2025):")
    result2 = validator.validate_test_period('2024-08-01', '2025-05-31', model_name='v2.1_model')
    validator.print_validation_report(result2)
    print()
    
    # Test 3: Période recommandée
    safe_start, safe_end = validator.get_safe_test_period_2024_2025()
    print(f"3. Période recommandée safe: {safe_start} to {safe_end}")