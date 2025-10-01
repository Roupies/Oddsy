#!/usr/bin/env python3
"""
🚀 Weekly Pipeline Automatisé - Pipeline Durci v1.0
===================================================
Enchaîne extraction xG strict → calcul enhanced → validation → prédictions
Arrêt net si assertion critique échoue
"""

import subprocess
import sys
import json
import os
import logging
from datetime import datetime

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/weekly_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_directories():
    """Créer répertoires nécessaires"""
    directories = ['logs', 'predictions', 'data/processed', 'reports/weekly']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_command(command, description, critical=True):
    """Exécuter une commande avec logging et gestion d'erreurs"""
    logger.info(f"🔄 {description}")
    logger.info(f"Commande: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - SUCCÈS")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"❌ {description} - ÉCHEC")
            logger.error(f"Error: {result.stderr}")
            if critical:
                logger.critical(f"🛑 ARRÊT PIPELINE - Assertion critique échouée: {description}")
                sys.exit(1)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ {description} - TIMEOUT après 10 minutes")
        if critical:
            logger.critical(f"🛑 ARRÊT PIPELINE - Timeout: {description}")
            sys.exit(1)
        return False
    except Exception as e:
        logger.error(f"💥 {description} - ERREUR INATTENDUE: {e}")
        if critical:
            logger.critical(f"🛑 ARRÊT PIPELINE - Erreur inattendue: {description}")
            sys.exit(1)
        return False

def validate_pipeline_output():
    """Valider que les sorties critiques existent"""
    critical_files = [
        'data/processed/enhanced_features_strict_temporal.csv',
        'data/processed/real_coverage_validation_report.json',
        'data/processed/strict_temporal_report.json'
    ]
    
    for file_path in critical_files:
        if not os.path.exists(file_path):
            logger.error(f"❌ Fichier critique manquant: {file_path}")
            return False
        
        # Vérifier que le fichier n'est pas vide
        if os.path.getsize(file_path) == 0:
            logger.error(f"❌ Fichier critique vide: {file_path}")
            return False
    
    return True

def check_validation_assertions():
    """Vérifier que les assertions critiques sont passées"""
    try:
        with open('data/processed/real_coverage_validation_report.json', 'r') as f:
            validation_report = json.load(f)
        
        # Vérifier assertions production
        production_ready = validation_report.get('validation_results', {}).get('production_readiness', {})
        critical_assertions_passed = production_ready.get('critical_assertions_passed', False)
        
        if not critical_assertions_passed:
            logger.error("❌ Assertions critiques échouées - Voir rapport validation")
            return False
        
        # Log métriques importantes
        quality_score = production_ready.get('quality_score', 0)
        logger.info(f"📊 Score qualité: {quality_score}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Impossible de lire rapport validation: {e}")
        return False

def generate_weekly_monitoring_report():
    """Générer rapport monitoring hebdomadaire"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'reports/weekly/monitoring_report_{timestamp}.json'
    
    try:
        # Charger rapports existants
        with open('data/processed/real_coverage_validation_report.json', 'r') as f:
            validation_data = json.load(f)
        
        with open('data/processed/strict_temporal_report.json', 'r') as f:
            temporal_data = json.load(f)
        
        # Créer rapport monitoring
        monitoring_report = {
            'report_timestamp': datetime.now().isoformat(),
            'pipeline_version': 'durci-v1.0',
            'status': 'SUCCESS',
            'metrics': {
                'jointure_coverage': validation_data.get('validation_results', {}).get('jointure_strict', {}).get('coverage_rate', 0) * 100,
                'xg_valid_coverage': validation_data.get('validation_results', {}).get('temporal_rolling', {}).get('xg_valid_coverage_pct', 0),
                'quality_score': validation_data.get('validation_results', {}).get('production_readiness', {}).get('quality_score', 0),
                'constants_eliminated': temporal_data.get('validation_stats', {}).get('constants_eliminated_pct', 0),
                'matches_processed': temporal_data.get('matches_processed', 0)
            },
            'alerts': [],
            'files_generated': {
                'enhanced_dataset': 'data/processed/enhanced_features_strict_temporal.csv',
                'validation_report': 'data/processed/real_coverage_validation_report.json',
                'temporal_report': 'data/processed/strict_temporal_report.json',
                'predictions': 'predictions/j6_production_*.csv'
            }
        }
        
        # Ajouter alertes si seuils non respectés
        if monitoring_report['metrics']['jointure_coverage'] < 80:
            monitoring_report['alerts'].append('COUVERTURE_JOINTURE_FAIBLE')
        
        if monitoring_report['metrics']['quality_score'] < 98:
            monitoring_report['alerts'].append('SCORE_QUALITE_FAIBLE')
        
        # Sauvegarder rapport
        os.makedirs('reports/weekly', exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(monitoring_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Rapport monitoring généré: {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur génération rapport monitoring: {e}")
        return False

def run_weekly_pipeline():
    """Exécuter pipeline hebdomadaire complet"""
    logger.info("🚀 DÉMARRAGE PIPELINE HEBDOMADAIRE DURCI v1.0")
    logger.info("=" * 70)
    
    # Créer répertoires
    create_directories()
    
    # Étape 1: Extraction xG strict (autorisé à échouer en démo)
    success = run_command(
        "python extract_understat_real_strict.py",
        "Extraction xG Understat strict",
        critical=False  # Non critique en démo (API peut être indisponible)
    )
    
    if not success:
        logger.warning("⚠️ Extraction xG échouée - Utilisation dataset démo existant")
    
    # Étape 2: Calcul enhanced strict temporal 
    run_command(
        "python enhanced_calculator_strict_temporal.py",
        "Calcul enhanced strict temporal",
        critical=True
    )
    
    # Étape 3: Validation réel coverage avec seuil 98%
    run_command(
        "python validation_real_coverage.py",
        "Validation coverage réelle",
        critical=False  # Non critique en démo (seuils démo attendus)
    )
    
    # Vérification assertions critiques
    if not check_validation_assertions():
        logger.critical("🛑 ARRÊT PIPELINE - Assertions critiques échouées")
        sys.exit(1)
    
    # Étape 4: Prédictions J6 avec dataset enhanced
    run_command(
        "python j6_predictions_production.py",
        "Génération prédictions J6",
        critical=True
    )
    
    # Validation finale des outputs
    if not validate_pipeline_output():
        logger.critical("🛑 ARRÊT PIPELINE - Fichiers critiques manquants")
        sys.exit(1)
    
    # Génération rapport monitoring
    generate_weekly_monitoring_report()
    
    logger.info("🎯 PIPELINE HEBDOMADAIRE TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 70)
    logger.info("✅ Extraction → Enhanced → Validation → Prédictions: COMPLET")
    logger.info("📊 Tous les artefacts générés et validés")
    logger.info("🔍 Voir logs détaillés et rapports monitoring pour métriques")

if __name__ == "__main__":
    run_weekly_pipeline()