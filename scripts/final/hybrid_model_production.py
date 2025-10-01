#!/usr/bin/env python3
"""
🚀 MODÈLE HYBRIDE PRODUCTION
===========================
Pipeline adaptatif qui combine :
- Cascade optimisé pour J1-J4 (début saison)
- Baseline v2.3 pour reste de saison

Switch automatique basé sur matchday_normalized.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import joblib
import logging

# Import modèle cascade
import sys
sys.path.append('scripts/final')
from cascade_model_production import CascadeModelProduction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_prod")

class HybridModelProduction(BaseEstimator, ClassifierMixin):
    """
    Modèle hybride adaptatif pour production.
    
    Architecture:
    - J1-J4 (matchday_normalized <= 0.15): Cascade spécialisé
    - J5+ (matchday_normalized > 0.15): RandomForest baseline
    
    Paramètres:
    - early_season_threshold: 0.15 (≈ J1-J4 sur 38 journées)
    - cascade_params: Paramètres optimaux cascade
    - baseline_params: Paramètres baseline
    """
    
    def __init__(self, 
                 early_season_threshold=0.15,
                 cascade_draw_weight=3.0,
                 cascade_draw_threshold=0.35,
                 cascade_calibration_factor=0.85,
                 random_state=42):
        
        self.early_season_threshold = early_season_threshold
        self.cascade_draw_weight = cascade_draw_weight
        self.cascade_draw_threshold = cascade_draw_threshold
        self.cascade_calibration_factor = cascade_calibration_factor
        self.random_state = random_state
        
        # Modèles internes
        self.cascade_model = None
        self.baseline_model = None
        
        # Classes pour compatibilité sklearn
        self.classes_ = np.array(['H', 'D', 'A'])
        self.n_classes_ = 3
        
    def fit(self, X, y):
        """
        Entraînement hybride sur dataset complet.
        
        Args:
            X: Features incluant obligatoirement 'matchday_normalized'
            y: Target (int ou string)
        """
        # Validation sklearn
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32)
        
        # Conversion DataFrame si nécessaire pour accès colonnes
        if isinstance(X, np.ndarray):
            # Reconstruction des noms de colonnes (approximation)
            feature_names = [
                'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
                'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
                'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
            ]
            X_df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
        else:
            X_df = X.copy()
        
        # Vérification presence matchday_normalized
        if 'matchday_normalized' not in X_df.columns:
            raise ValueError("Feature 'matchday_normalized' requise pour modèle hybride")
        
        matchday_col = X_df['matchday_normalized']
        
        logger.info(f"🚀 ENTRAÎNEMENT MODÈLE HYBRIDE")
        logger.info(f"   Dataset: {len(X)} échantillons")
        logger.info(f"   Seuil early season: {self.early_season_threshold}")
        
        # Split early season vs rest
        early_mask = matchday_col <= self.early_season_threshold
        rest_mask = ~early_mask
        
        early_count = early_mask.sum()
        rest_count = rest_mask.sum()
        
        logger.info(f"   Early season (J1-J4): {early_count} échantillons")
        logger.info(f"   Reste saison (J5+): {rest_count} échantillons")\n        \n        # 1. Entraînement Cascade sur tout le dataset (mais optimisé early)\n        logger.info(f\"   🎯 Entraînement Cascade (spécialisé early season)\")\n        \n        self.cascade_model = CascadeModelProduction(\n            draw_weight=self.cascade_draw_weight,\n            draw_threshold=self.cascade_draw_threshold,\n            calibration_factor=self.cascade_calibration_factor,\n            random_state=self.random_state\n        )\n        \n        # Pondération: plus de poids sur early season\n        sample_weights = np.ones(len(X))\n        sample_weights[early_mask] = 2.0  # Double poids early season\n        \n        # Note: CascadeModelProduction ne supporte pas sample_weight, \n        # donc on entraîne sur dataset complet\n        self.cascade_model.fit(X, y)\n        \n        # 2. Entraînement Baseline sur tout le dataset\n        logger.info(f\"   📊 Entraînement Baseline (général)\")\n        \n        self.baseline_model = RandomForestClassifier(\n            n_estimators=200,\n            max_depth=15,\n            min_samples_leaf=3,\n            class_weight=\"balanced\",\n            random_state=self.random_state\n        )\n        \n        self.baseline_model.fit(X, y)\n        \n        # Stockage des paramètres\n        self.n_features_in_ = X.shape[1]\n        \n        logger.info(f\"✅ Modèle hybride entraîné\")\n        \n        return self\n    \n    def predict(self, X):\n        \"\"\"\n        Prédiction hybride avec switch automatique.\n        \n        Args:\n            X: Features incluant 'matchday_normalized'\n            \n        Returns:\n            array: Prédictions ['H', 'D', 'A']\n        \"\"\"\n        # Validation sklearn\n        X = check_array(X, accept_sparse=False, dtype=np.float32)\n        \n        # Conversion DataFrame si nécessaire\n        if isinstance(X, np.ndarray):\n            feature_names = [\n                'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',\n                'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',\n                'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'\n            ]\n            X_df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])\n        else:\n            X_df = X.copy()\n        \n        if 'matchday_normalized' not in X_df.columns:\n            raise ValueError(\"Feature 'matchday_normalized' requise pour prédiction hybride\")\n        \n        matchday_col = X_df['matchday_normalized']\n        \n        # Switch selon phase de saison\n        early_mask = matchday_col <= self.early_season_threshold\n        \n        # Initialisation prédictions\n        predictions = np.full(len(X), 'H', dtype=object)\n        \n        # Prédictions early season (cascade)\n        if early_mask.sum() > 0:\n            early_preds = self.cascade_model.predict(X[early_mask])\n            predictions[early_mask] = early_preds\n        \n        # Prédictions reste saison (baseline)\n        rest_mask = ~early_mask\n        if rest_mask.sum() > 0:\n            rest_preds = self.baseline_model.predict(X[rest_mask])\n            # Conversion int vers string si nécessaire\n            if hasattr(rest_preds[0], 'dtype') and np.issubdtype(rest_preds.dtype, np.integer):\n                rest_preds_str = pd.Series(rest_preds).map({0: 'H', 1: 'D', 2: 'A'}).values\n            else:\n                rest_preds_str = rest_preds\n            predictions[rest_mask] = rest_preds_str\n        \n        return predictions\n    \n    def predict_proba(self, X):\n        \"\"\"\n        Probabilités hybrides avec switch automatique.\n        \n        Returns:\n            array: Probabilités [P(H), P(D), P(A)]\n        \"\"\"\n        X = check_array(X, accept_sparse=False, dtype=np.float32)\n        \n        # Conversion DataFrame\n        if isinstance(X, np.ndarray):\n            feature_names = [\n                'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',\n                'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',\n                'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'\n            ]\n            X_df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])\n        else:\n            X_df = X.copy()\n        \n        matchday_col = X_df['matchday_normalized']\n        early_mask = matchday_col <= self.early_season_threshold\n        \n        # Initialisation probabilités\n        n_samples = len(X)\n        probas = np.zeros((n_samples, 3))  # [H, D, A]\n        \n        # Probabilités early season (cascade)\n        if early_mask.sum() > 0:\n            early_probas = self.cascade_model.predict_proba(X[early_mask])\n            probas[early_mask] = early_probas\n        \n        # Probabilités reste saison (baseline)\n        rest_mask = ~early_mask\n        if rest_mask.sum() > 0:\n            rest_probas = self.baseline_model.predict_proba(X[rest_mask])\n            # Réorganisation si baseline prédit en format différent\n            if rest_probas.shape[1] == 3:\n                # Supposer ordre [classe0, classe1, classe2] = [H, D, A] ou [0, 1, 2]\n                # Ajustement selon classes_ baseline\n                baseline_classes = getattr(self.baseline_model, 'classes_', [0, 1, 2])\n                if len(baseline_classes) == 3:\n                    # Mapping vers ordre [H, D, A]\n                    class_to_idx = {}\n                    for i, cls in enumerate(baseline_classes):\n                        if cls == 0 or cls == 'H': class_to_idx['H'] = i\n                        elif cls == 1 or cls == 'D': class_to_idx['D'] = i \n                        elif cls == 2 or cls == 'A': class_to_idx['A'] = i\n                    \n                    # Réarrangement\n                    if len(class_to_idx) == 3:\n                        reordered_probas = np.zeros_like(rest_probas)\n                        reordered_probas[:, 0] = rest_probas[:, class_to_idx['H']]  # H\n                        reordered_probas[:, 1] = rest_probas[:, class_to_idx['D']]  # D\n                        reordered_probas[:, 2] = rest_probas[:, class_to_idx['A']]  # A\n                        probas[rest_mask] = reordered_probas\n                    else:\n                        probas[rest_mask] = rest_probas\n                else:\n                    probas[rest_mask] = rest_probas\n            else:\n                # Fallback uniforme si problème format\n                probas[rest_mask] = 1/3\n        \n        # Normalisation sécurité\n        probas = probas / probas.sum(axis=1, keepdims=True)\n        \n        return probas\n    \n    def get_model_info(self, X):\n        \"\"\"\n        Information sur quel modèle est utilisé pour chaque échantillon.\n        \n        Returns:\n            dict: Statistiques d'utilisation des modèles\n        \"\"\"\n        if isinstance(X, np.ndarray):\n            feature_names = [\n                'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',\n                'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',\n                'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'\n            ]\n            X_df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])\n        else:\n            X_df = X.copy()\n        \n        matchday_col = X_df['matchday_normalized']\n        early_mask = matchday_col <= self.early_season_threshold\n        \n        return {\n            'total_samples': len(X),\n            'cascade_samples': early_mask.sum(),\n            'baseline_samples': (~early_mask).sum(),\n            'cascade_ratio': early_mask.mean(),\n            'early_season_threshold': self.early_season_threshold\n        }\n    \n    def get_params(self, deep=True):\n        \"\"\"Paramètres du modèle pour sklearn.\"\"\"\n        return {\n            'early_season_threshold': self.early_season_threshold,\n            'cascade_draw_weight': self.cascade_draw_weight,\n            'cascade_draw_threshold': self.cascade_draw_threshold,\n            'cascade_calibration_factor': self.cascade_calibration_factor,\n            'random_state': self.random_state\n        }\n    \n    def set_params(self, **params):\n        \"\"\"Définition des paramètres pour sklearn.\"\"\"\n        for key, value in params.items():\n            setattr(self, key, value)\n        return self\n\ndef create_hybrid_model_production():\n    \"\"\"\n    Créateur du modèle hybride de production.\n    \n    Returns:\n        HybridModelProduction: Modèle configuré avec paramètres optimaux\n    \"\"\"\n    logger.info(\"🚀 Création modèle hybride production\")\n    \n    model = HybridModelProduction(\n        early_season_threshold=0.15,  # ≈ J1-J4\n        cascade_draw_weight=3.0,\n        cascade_draw_threshold=0.35,\n        cascade_calibration_factor=0.85,\n        random_state=42\n    )\n    \n    logger.info(f\"   Seuil early season: {model.early_season_threshold}\")\n    logger.info(f\"   Paramètres cascade: weight={model.cascade_draw_weight}, threshold={model.cascade_draw_threshold}\")\n    \n    return model\n\ndef save_hybrid_model(model, filepath):\n    \"\"\"\n    Sauvegarde du modèle hybride.\n    \n    Args:\n        model: Modèle hybride entraîné\n        filepath: Chemin de sauvegarde\n    \"\"\"\n    try:\n        joblib.dump(model, filepath)\n        logger.info(f\"✅ Modèle hybride sauvegardé: {filepath}\")\n        \n        # Métadonnées\n        metadata = {\n            'model_type': 'HybridModelProduction',\n            'early_season_threshold': model.early_season_threshold,\n            'cascade_params': {\n                'draw_weight': model.cascade_draw_weight,\n                'draw_threshold': model.cascade_draw_threshold,\n                'calibration_factor': model.cascade_calibration_factor\n            },\n            'random_state': model.random_state,\n            'classes': model.classes_.tolist(),\n            'n_features': getattr(model, 'n_features_in_', None)\n        }\n        \n        metadata_path = filepath.replace('.joblib', '_metadata.json')\n        import json\n        with open(metadata_path, 'w') as f:\n            json.dump(metadata, f, indent=2)\n        \n        logger.info(f\"✅ Métadonnées sauvegardées: {metadata_path}\")\n        \n    except Exception as e:\n        logger.error(f\"❌ Erreur sauvegarde: {e}\")\n\nif __name__ == \"__main__\":\n    # Test rapide du modèle hybride\n    logger.info(\"🧪 Test modèle hybride production\")\n    \n    # Création modèle\n    model = create_hybrid_model_production()\n    \n    # Test avec données fictives incluant matchday_normalized\n    np.random.seed(42)\n    X_test = np.random.randn(20, 10)\n    # Simulation matchdays: early (0.1) et late (0.8)\n    X_test[:10, 6] = 0.1  # Early season\n    X_test[10:, 6] = 0.8  # Late season\n    \n    y_test = np.random.choice([0, 1, 2], size=20)\n    \n    # Entraînement\n    model.fit(X_test, y_test)\n    \n    # Prédictions\n    preds = model.predict(X_test)\n    probas = model.predict_proba(X_test)\n    info = model.get_model_info(X_test)\n    \n    logger.info(f\"✅ Test réussi - Prédictions: {preds[:5]}\")\n    logger.info(f\"✅ Test réussi - Probabilités shape: {probas.shape}\")\n    logger.info(f\"✅ Test réussi - Info modèles: {info}\")\n    \n    print(\"🚀 Modèle hybride production prêt pour audit !\")"