#!/usr/bin/env python3
"""
🏆 MODÈLE CASCADE PRODUCTION
===========================
Modèle cascade optimisé pour production avec seuil 0.40.
Compatible avec l'infrastructure d'audit existante.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cascade_prod")

class CascadeModelProduction(BaseEstimator, ClassifierMixin):
    """
    Modèle cascade optimisé pour production.
    
    Architecture:
    1. Draw Forest: Détecte les matchs nuls avec class_weight et seuil calibré
    2. Home/Away Forest: Prédit H vs A pour les non-nuls avec équilibrage
    
    Paramètres optimisés:
    - draw_weight: 3.0
    - draw_threshold: 0.40 
    - homeaway_balance: True
    """
    
    def __init__(self, draw_weight=3.0, draw_threshold=0.35, calibration_factor=0.85, random_state=42):
        self.draw_weight = draw_weight
        self.draw_threshold = draw_threshold
        self.calibration_factor = calibration_factor
        self.random_state = random_state
        
        # Initialisation des modèles
        self.clf_draw = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_leaf=5,
            class_weight={0: 1, 1: self.draw_weight}, 
            random_state=self.random_state
        )
        
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=150, 
            class_weight="balanced",
            random_state=self.random_state
        )
        
        # Classes pour compatibilité sklearn
        self.classes_ = np.array(['H', 'D', 'A'])
        self.n_classes_ = 3
        
    def fit(self, X, y):
        """
        Entraînement du modèle cascade.
        
        Args:
            X: Features (DataFrame ou array)
            y: Target (int ou string - sera converti en string H/D/A)
        """
        # Validation sklearn
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32)
        
        # Conversion du target vers format string si nécessaire
        if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.integer):
            y_str = pd.Series(y).map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = pd.Series(y)
        
        logger.info(f"🔧 Entraînement cascade sur {len(X)} échantillons")
        
        # 1. Entraînement Draw Forest
        y_draw = (y_str == 'D').astype(int)
        draw_distribution = y_draw.value_counts(normalize=True)
        logger.info(f"   Distribution draws: {draw_distribution[1]:.1%} draws")
        
        self.clf_draw.fit(X, y_draw)
        
        # 2. Entraînement Home/Away Forest
        mask_notdraw = y_str != 'D'
        n_notdraw = mask_notdraw.sum()
        
        if n_notdraw > 5:
            X_notdraw = X[mask_notdraw]
            y_homeaway = y_str[mask_notdraw].map({'H': 1, 'A': 0})
            
            # Filtrage des NaN potentiels
            valid_homeaway = y_homeaway.notna()
            if valid_homeaway.sum() > 5:
                self.clf_homeaway.fit(X_notdraw[valid_homeaway], y_homeaway[valid_homeaway])
                logger.info(f"   Home/Away entraîné sur {valid_homeaway.sum()} échantillons")
            else:
                logger.warning(f"⚠️  Pas assez d'échantillons Home/Away valides")
        else:
            logger.warning(f"⚠️  Pas assez d'échantillons non-draws ({n_notdraw})")
        
        # Stockage des paramètres d'entraînement
        self.n_features_in_ = X.shape[1]
        
        return self
    
    def predict(self, X):
        """
        Prédiction cascade avec seuil optimisé.
        
        Args:
            X: Features (DataFrame ou array)
            
        Returns:
            array: Prédictions ['H', 'D', 'A']
        """
        # Validation sklearn
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        # 1. Prédiction draws
        draw_proba_output = self.clf_draw.predict_proba(X)
        
        # Gestion robuste des probabilités
        if draw_proba_output.shape[1] == 1:
            # Une seule classe détectée
            single_class = self.clf_draw.classes_[0]
            draw_proba = np.full(len(X), single_class)
        else:
            draw_proba = draw_proba_output[:, 1]
        
        # Application du seuil calibré avec facteur de calibration
        calibrated_proba = draw_proba * self.calibration_factor
        is_draw = calibrated_proba >= self.draw_threshold
        
        # 2. Initialisation des prédictions
        predictions = np.full(len(X), 'D', dtype=object)
        
        # 3. Prédiction Home/Away pour les non-draws
        mask_notdraw = ~is_draw
        if mask_notdraw.sum() > 0 and hasattr(self.clf_homeaway, 'predict'):
            try:
                homeaway_pred = self.clf_homeaway.predict(X[mask_notdraw])
                predictions[mask_notdraw] = np.where(homeaway_pred == 1, 'H', 'A')
            except Exception as e:
                logger.warning(f"⚠️  Erreur Home/Away prediction: {e}")
                # Fallback: prédire Home par défaut
                predictions[mask_notdraw] = 'H'
        
        return predictions
    
    def predict_proba(self, X):
        """
        Probabilités des classes pour compatibilité sklearn.
        
        Returns:
            array: Probabilités [P(H), P(D), P(A)]
        """
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        # Prédictions détaillées
        draw_proba_output = self.clf_draw.predict_proba(X)
        
        if draw_proba_output.shape[1] == 1:
            draw_proba = np.full(len(X), self.clf_draw.classes_[0])
        else:
            draw_proba = draw_proba_output[:, 1]
        
        is_draw = draw_proba >= self.draw_threshold
        
        # Initialisation des probabilités
        n_samples = len(X)
        probas = np.zeros((n_samples, 3))  # [H, D, A]
        
        # Probabilités draws
        probas[:, 1] = np.where(is_draw, 0.8, draw_proba * 0.5)  # Ajustement heuristique
        
        # Probabilités Home/Away pour non-draws
        mask_notdraw = ~is_draw
        if mask_notdraw.sum() > 0 and hasattr(self.clf_homeaway, 'predict_proba'):
            try:
                homeaway_proba = self.clf_homeaway.predict_proba(X[mask_notdraw])
                if homeaway_proba.shape[1] == 2:
                    probas[mask_notdraw, 2] = homeaway_proba[:, 0]  # Away
                    probas[mask_notdraw, 0] = homeaway_proba[:, 1]  # Home
            except:
                # Fallback uniforme
                probas[mask_notdraw, 0] = 0.6  # Home par défaut
                probas[mask_notdraw, 2] = 0.4  # Away
        
        # Normalisation des probabilités
        probas = probas / probas.sum(axis=1, keepdims=True)
        
        return probas
    
    def get_params(self, deep=True):
        """Paramètres du modèle pour sklearn."""
        return {
            'draw_weight': self.draw_weight,
            'draw_threshold': self.draw_threshold,
            'calibration_factor': self.calibration_factor,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Définition des paramètres pour sklearn."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

def create_cascade_model_production():
    """
    Créateur du modèle cascade de production.
    
    Returns:
        CascadeModelProduction: Modèle configuré avec paramètres optimaux
    """
    logger.info("🏭 Création modèle cascade production")
    
    model = CascadeModelProduction(
        draw_weight=3.0,
        draw_threshold=0.35,
        calibration_factor=0.85,
        random_state=42
    )
    
    logger.info(f"   Paramètres: draw_weight={model.draw_weight}, threshold={model.draw_threshold}")
    
    return model

def save_cascade_model(model, filepath):
    """
    Sauvegarde du modèle cascade.
    
    Args:
        model: Modèle entraîné
        filepath: Chemin de sauvegarde
    """
    try:
        joblib.dump(model, filepath)
        logger.info(f"✅ Modèle cascade sauvegardé: {filepath}")
        
        # Métadonnées
        metadata = {
            'model_type': 'CascadeModelProduction',
            'draw_weight': model.draw_weight,
            'draw_threshold': model.draw_threshold,
            'random_state': model.random_state,
            'classes': model.classes_.tolist(),
            'n_features': getattr(model, 'n_features_in_', None)
        }
        
        metadata_path = filepath.replace('.joblib', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Métadonnées sauvegardées: {metadata_path}")
        
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde: {e}")

def load_cascade_model(filepath):
    """
    Chargement du modèle cascade.
    
    Args:
        filepath: Chemin du modèle
        
    Returns:
        CascadeModelProduction: Modèle chargé
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"✅ Modèle cascade chargé: {filepath}")
        return model
    except Exception as e:
        logger.error(f"❌ Erreur chargement: {e}")
        return None

if __name__ == "__main__":
    # Test rapide du modèle
    logger.info("🧪 Test modèle cascade production")
    
    # Création modèle
    model = create_cascade_model_production()
    
    # Test avec données fictives
    X_test = np.random.randn(10, 10)
    y_test = np.array(['H', 'D', 'A', 'H', 'A', 'D', 'H', 'H', 'A', 'D'])
    
    # Entraînement
    model.fit(X_test, y_test)
    
    # Prédiction
    preds = model.predict(X_test[:5])
    probas = model.predict_proba(X_test[:5])
    
    logger.info(f"✅ Test réussi - Prédictions: {preds}")
    logger.info(f"✅ Test réussi - Probabilités: {probas.shape}")
    
    print("🏆 Modèle cascade production prêt pour audit !")