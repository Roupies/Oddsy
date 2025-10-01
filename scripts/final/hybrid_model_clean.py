#!/usr/bin/env python3
"""
🚀 MODÈLE HYBRIDE PRODUCTION - VERSION CLEAN
============================================
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
import sys

# Import modèle cascade
sys.path.append('scripts/final')
from cascade_model_production import CascadeModelProduction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_clean")

class HybridModelClean(BaseEstimator, ClassifierMixin):
    """
    Modèle hybride adaptatif pour production.
    
    Architecture:
    - J1-J4 (matchday_normalized <= 0.15): Cascade spécialisé
    - J5+ (matchday_normalized > 0.15): RandomForest baseline
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
        
        # Features standard
        self.feature_names = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        # Classes pour compatibilité sklearn
        self.classes_ = np.array(['H', 'D', 'A'])
        self.n_classes_ = 3
        
    def _prepare_dataframe(self, X):
        """Conversion array vers DataFrame avec noms de colonnes."""
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        else:
            return X.copy()
    
    def fit(self, X, y):
        """Entraînement hybride sur dataset complet."""
        # Validation sklearn
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32)
        
        # Conversion DataFrame
        X_df = self._prepare_dataframe(X)
        
        # Vérification matchday_normalized
        if 'matchday_normalized' not in X_df.columns:
            raise ValueError("Feature 'matchday_normalized' requise pour modèle hybride")
        
        matchday_col = X_df['matchday_normalized']
        
        logger.info("🚀 ENTRAÎNEMENT MODÈLE HYBRIDE")
        logger.info(f"   Dataset: {len(X)} échantillons")
        logger.info(f"   Seuil early season: {self.early_season_threshold}")
        
        # Split early season vs rest
        early_mask = matchday_col <= self.early_season_threshold
        early_count = early_mask.sum()
        rest_count = (~early_mask).sum()
        
        logger.info(f"   Early season (J1-J4): {early_count} échantillons")
        logger.info(f"   Reste saison (J5+): {rest_count} échantillons")
        
        # 1. Entraînement Cascade
        logger.info("   🎯 Entraînement Cascade (spécialisé early season)")
        
        self.cascade_model = CascadeModelProduction(
            draw_weight=self.cascade_draw_weight,
            draw_threshold=self.cascade_draw_threshold,
            calibration_factor=self.cascade_calibration_factor,
            random_state=self.random_state
        )
        
        self.cascade_model.fit(X, y)
        
        # 2. Entraînement Baseline
        logger.info("   📊 Entraînement Baseline (général)")
        
        self.baseline_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=self.random_state
        )
        
        self.baseline_model.fit(X, y)
        
        # Stockage des paramètres
        self.n_features_in_ = X.shape[1]
        
        logger.info("✅ Modèle hybride entraîné")
        
        return self
    
    def predict(self, X):
        """Prédiction hybride avec switch automatique."""
        # Validation sklearn
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        # Conversion DataFrame
        X_df = self._prepare_dataframe(X)
        
        if 'matchday_normalized' not in X_df.columns:
            raise ValueError("Feature 'matchday_normalized' requise pour prédiction hybride")
        
        matchday_col = X_df['matchday_normalized']
        
        # Switch selon phase de saison
        early_mask = matchday_col <= self.early_season_threshold
        
        # Initialisation prédictions
        predictions = np.full(len(X), 'H', dtype=object)
        
        # Prédictions early season (cascade)
        if early_mask.sum() > 0:
            early_preds = self.cascade_model.predict(X[early_mask])
            predictions[early_mask] = early_preds
        
        # Prédictions reste saison (baseline)
        rest_mask = ~early_mask
        if rest_mask.sum() > 0:
            rest_preds = self.baseline_model.predict(X[rest_mask])
            # Conversion int vers string si nécessaire
            if len(rest_preds) > 0:
                if hasattr(rest_preds[0], 'dtype') and np.issubdtype(rest_preds.dtype, np.integer):
                    rest_preds_str = pd.Series(rest_preds).map({0: 'H', 1: 'D', 2: 'A'}).values
                else:
                    rest_preds_str = rest_preds
                predictions[rest_mask] = rest_preds_str
        
        return predictions
    
    def predict_proba(self, X):
        """Probabilités hybrides avec switch automatique."""
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        # Conversion DataFrame
        X_df = self._prepare_dataframe(X)
        matchday_col = X_df['matchday_normalized']
        early_mask = matchday_col <= self.early_season_threshold
        
        # Initialisation probabilités
        n_samples = len(X)
        probas = np.zeros((n_samples, 3))  # [H, D, A]
        
        # Probabilités early season (cascade)
        if early_mask.sum() > 0:
            early_probas = self.cascade_model.predict_proba(X[early_mask])
            probas[early_mask] = early_probas
        
        # Probabilités reste saison (baseline)
        rest_mask = ~early_mask
        if rest_mask.sum() > 0:
            rest_probas = self.baseline_model.predict_proba(X[rest_mask])
            
            # Gestion ordre des classes baseline
            baseline_classes = getattr(self.baseline_model, 'classes_', [0, 1, 2])
            if len(baseline_classes) == 3 and rest_probas.shape[1] == 3:
                # Mapping vers ordre [H, D, A]
                class_mapping = {}
                for i, cls in enumerate(baseline_classes):
                    if cls == 0 or cls == 'H':
                        class_mapping[0] = i  # H
                    elif cls == 1 or cls == 'D':
                        class_mapping[1] = i  # D
                    elif cls == 2 or cls == 'A':
                        class_mapping[2] = i  # A
                
                if len(class_mapping) == 3:
                    # Réarrangement
                    reordered_probas = np.zeros_like(rest_probas)
                    reordered_probas[:, 0] = rest_probas[:, class_mapping[0]]  # H
                    reordered_probas[:, 1] = rest_probas[:, class_mapping[1]]  # D
                    reordered_probas[:, 2] = rest_probas[:, class_mapping[2]]  # A
                    probas[rest_mask] = reordered_probas
                else:
                    probas[rest_mask] = rest_probas
            else:
                probas[rest_mask] = rest_probas
        
        # Normalisation sécurité
        row_sums = probas.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Éviter division par zéro
        probas = probas / row_sums
        
        return probas
    
    def get_model_info(self, X):
        """Information sur quel modèle est utilisé pour chaque échantillon."""
        X_df = self._prepare_dataframe(X)
        matchday_col = X_df['matchday_normalized']
        early_mask = matchday_col <= self.early_season_threshold
        
        return {
            'total_samples': len(X),
            'cascade_samples': early_mask.sum(),
            'baseline_samples': (~early_mask).sum(),
            'cascade_ratio': early_mask.mean(),
            'early_season_threshold': self.early_season_threshold
        }
    
    def get_params(self, deep=True):
        """Paramètres du modèle pour sklearn."""
        return {
            'early_season_threshold': self.early_season_threshold,
            'cascade_draw_weight': self.cascade_draw_weight,
            'cascade_draw_threshold': self.cascade_draw_threshold,
            'cascade_calibration_factor': self.cascade_calibration_factor,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Définition des paramètres pour sklearn."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

def create_hybrid_model():
    """Créateur du modèle hybride de production."""
    logger.info("🚀 Création modèle hybride production")
    
    model = HybridModelClean(
        early_season_threshold=0.15,  # ≈ J1-J4
        cascade_draw_weight=3.0,
        cascade_draw_threshold=0.35,
        cascade_calibration_factor=0.85,
        random_state=42
    )
    
    logger.info(f"   Seuil early season: {model.early_season_threshold}")
    logger.info(f"   Paramètres cascade: weight={model.cascade_draw_weight}, threshold={model.cascade_draw_threshold}")
    
    return model

if __name__ == "__main__":
    # Test rapide du modèle hybride
    logger.info("🧪 Test modèle hybride clean")
    
    # Création modèle
    model = create_hybrid_model()
    
    # Test avec données fictives incluant matchday_normalized
    np.random.seed(42)
    X_test = np.random.randn(20, 10)
    # Simulation matchdays: early (0.1) et late (0.8)
    X_test[:10, 6] = 0.1  # Early season
    X_test[10:, 6] = 0.8  # Late season
    
    y_test = np.random.choice([0, 1, 2], size=20)
    
    # Entraînement
    model.fit(X_test, y_test)
    
    # Prédictions
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)
    info = model.get_model_info(X_test)
    
    logger.info(f"✅ Test réussi - Prédictions: {preds[:5]}")
    logger.info(f"✅ Test réussi - Probabilités shape: {probas.shape}")
    logger.info(f"✅ Test réussi - Info modèles: {info}")
    
    print("🚀 Modèle hybride clean prêt pour audit !")