#!/usr/bin/env python3
"""
🔍 ENTRAÎNEMENT ET AUDIT MODÈLE CASCADE
======================================
Entraîne le modèle cascade sur le dataset complet et lance l'audit avec audit_pipeline.py.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import logging

# Ajout du chemin pour importer le modèle cascade
sys.path.append('scripts/final')
from cascade_model_production import CascadeModelProduction, save_cascade_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_audit")

def train_cascade_full_dataset():
    """
    Entraîne le modèle cascade sur le dataset complet v15.
    
    Returns:
        tuple: (model_trained, dataset_path, model_path)
    """
    try:
        logger.info("🏭 ENTRAÎNEMENT MODÈLE CASCADE PRODUCTION")
        logger.info("=" * 50)
        
        # 1. Chargement dataset de production v15
        dataset_path = "data/processed/v15_final_enhanced.csv"
        if not os.path.exists(dataset_path):
            logger.error(f"❌ Dataset non trouvé: {dataset_path}")
            return None, None, None
        
        data = pd.read_csv(dataset_path)
        logger.info(f"📊 Dataset chargé: {len(data)} matchs")
        
        # 2. Préparation des features de production
        production_features = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        # Vérification disponibilité features
        available_features = [f for f in production_features if f in data.columns]
        logger.info(f"🎯 Features disponibles: {len(available_features)}/{len(production_features)}")
        
        if len(available_features) < 8:
            logger.error(f"❌ Pas assez de features disponibles: {len(available_features)}")
            return None, None, None
        
        # 3. Préparation des données
        X = data[available_features].fillna(0)
        
        # Création target à partir de FullTimeResult
        target_mapping = {'H': 0, 'D': 1, 'A': 2}
        y = data['FullTimeResult'].map(target_mapping)
        
        # Filtrage des échantillons valides (non-NaN)
        valid_mask = y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask].astype(int)
        
        logger.info(f"📊 Données nettoyées: {len(X_clean)} échantillons valides")
        logger.info(f"📊 Distribution: {pd.Series(y_clean).value_counts().sort_index().to_dict()}")
        
        # 4. Création et entraînement du modèle cascade
        logger.info(f"\n🚀 ENTRAÎNEMENT MODÈLE CASCADE")
        
        cascade_model = CascadeModelProduction(
            draw_weight=3.0,
            draw_threshold=0.40,
            random_state=42
        )
        
        # Entraînement
        cascade_model.fit(X_clean, y_clean)
        
        # 5. Sauvegarde du modèle
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"cascade_production_{timestamp}.joblib"
        model_path = f"models/{model_filename}"
        
        # Création du dossier models si nécessaire
        os.makedirs("models", exist_ok=True)
        
        save_cascade_model(cascade_model, model_path)
        
        # 6. Test rapide du modèle
        logger.info(f"\n🧪 TEST RAPIDE MODÈLE")
        
        # Test sur 10 premiers échantillons
        test_preds = cascade_model.predict(X_clean[:10])
        test_probas = cascade_model.predict_proba(X_clean[:10])
        
        logger.info(f"   Prédictions test: {test_preds}")
        logger.info(f"   Shape probabilités: {test_probas.shape}")
        logger.info(f"   Distribution test: {pd.Series(test_preds).value_counts().to_dict()}")
        
        logger.info(f"\n✅ MODÈLE CASCADE ENTRAÎNÉ ET SAUVEGARDÉ")
        logger.info(f"   Dataset: {dataset_path}")
        logger.info(f"   Modèle: {model_path}")
        logger.info(f"   Features: {len(available_features)}")
        logger.info(f"   Échantillons: {len(X_clean)}")
        
        return cascade_model, dataset_path, model_path
        
    except Exception as e:
        logger.error(f"❌ Erreur entraînement: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def run_audit_cascade(dataset_path, model_path, target_column='FullTimeResult'):
    """
    Lance l'audit complet du modèle cascade avec audit_pipeline.py.
    
    Args:
        dataset_path: Chemin vers le dataset
        model_path: Chemin vers le modèle 
        target_column: Nom de la colonne target
    """
    try:
        logger.info(f"\n🔍 AUDIT COMPLET MODÈLE CASCADE")
        logger.info("=" * 40)
        
        # Construction de la commande d'audit
        features_str = "elo_diff_normalized,market_entropy_norm,shots_diff_normalized,corners_diff_normalized,form_diff_normalized,h2h_score,matchday_normalized,home_xg_eff_10,away_xg_eff_10,away_goals_sum_5"
        
        features_list = features_str.split(',')
        audit_command = [
            "python3", "src/core/audit_pipeline.py",
            "--data", dataset_path,
            "--model", model_path, 
            "--target", target_column,
            "--features"
        ] + features_list
        
        logger.info(f"🔧 Commande audit: {' '.join(audit_command)}")
        
        # Exécution de l'audit
        import subprocess
        result = subprocess.run(audit_command, capture_output=True, text=True)
        
        logger.info(f"\n📊 RÉSULTATS AUDIT:")
        logger.info(f"Return code: {result.returncode}")
        
        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            logger.info(f"✅ AUDIT RÉUSSI - Modèle cascade validé")
        else:
            logger.error(f"❌ AUDIT ÉCHOUÉ - Voir les erreurs ci-dessus")
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"❌ Erreur audit: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Processus complet: entraînement + audit du modèle cascade.
    """
    logger.info("🎯 PROCESSUS COMPLET VALIDATION MODÈLE CASCADE")
    logger.info("=" * 60)
    
    # 1. Entraînement
    model, dataset_path, model_path = train_cascade_full_dataset()
    
    if model is None:
        logger.error("❌ Échec entraînement - arrêt du processus")
        return False
    
    # 2. Audit
    audit_success = run_audit_cascade(dataset_path, model_path)
    
    # 3. Résumé final
    logger.info(f"\n🏆 RÉSUMÉ FINAL")
    logger.info("=" * 20)
    
    if audit_success:
        logger.info("✅ MODÈLE CASCADE VALIDÉ POUR PRODUCTION")
        logger.info(f"   Dataset: {dataset_path}")
        logger.info(f"   Modèle: {model_path}")
        logger.info("   Prêt pour déploiement !")
        
        print(f"\n🎯 MODÈLE CASCADE VALIDÉ:")
        print(f"   Fichier: {model_path}")
        print(f"   Performances: Voir rapport d'audit ci-dessus")
        print(f"   Status: ✅ PRODUCTION READY")
        
    else:
        logger.error("❌ MODÈLE CASCADE NON VALIDÉ")
        logger.error("   Voir les erreurs d'audit pour diagnostics")
        
        print(f"\n⚠️  MODÈLE CASCADE NON VALIDÉ")
        print(f"   Audit échoué - révision nécessaire")
    
    return audit_success

if __name__ == "__main__":
    success = main()