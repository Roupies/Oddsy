#!/usr/bin/env python3
"""
🔧 TEST CALIBRATION MODÈLE CASCADE
===============================
Test différents seuils et class weights pour équilibrer le modèle cascade.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calibration")

class CascadeModelCalibrated:
    def __init__(self, draw_weight=2.5, draw_threshold=0.25, homeaway_balance=True):
        self.clf_draw = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight={0: 1, 1: draw_weight}, random_state=42
        )
        
        # Class weight pour Home/Away
        if homeaway_balance:
            homeaway_weights = "balanced"
        else:
            homeaway_weights = None
            
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight=homeaway_weights
        )
        self.draw_threshold = draw_threshold
    
    def fit(self, X, y):
        # Conversion vers classes string
        if y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y.copy()
        
        # 1. Draw Forest
        y_draw = (y_str == 'D').astype(int)
        self.clf_draw.fit(X, y_draw)
        
        # 2. Home/Away Forest
        mask_notdraw = y_str != 'D'
        if mask_notdraw.sum() > 5:
            X_notdraw = X[mask_notdraw]
            y_homeaway = y_str[mask_notdraw].map({'H': 1, 'A': 0})
            valid_homeaway = y_homeaway.notna()
            if valid_homeaway.sum() > 5:
                self.clf_homeaway.fit(X_notdraw[valid_homeaway], y_homeaway[valid_homeaway])
    
    def predict(self, X):
        # 1. Prédiction draws avec seuil calibré
        draw_proba_output = self.clf_draw.predict_proba(X)
        
        if draw_proba_output.shape[1] == 1:
            single_class = self.clf_draw.classes_[0]
            draw_proba = np.full(len(X), single_class)
        else:
            draw_proba = draw_proba_output[:, 1]
        
        is_draw = draw_proba >= self.draw_threshold
        
        # 2. Prédiction Home/Away
        predictions = np.full(len(X), 'D')
        
        mask_notdraw = ~is_draw
        if mask_notdraw.sum() > 0 and hasattr(self, 'clf_homeaway'):
            try:
                homeaway_pred = self.clf_homeaway.predict(X[mask_notdraw])
                predictions[mask_notdraw] = np.where(homeaway_pred == 1, 'H', 'A')
            except:
                predictions[mask_notdraw] = 'H'
        
        return predictions

def test_calibration():
    try:
        logger.info("🔧 TEST CALIBRATION CASCADE")
        logger.info("=" * 40)
        
        # Chargement données
        auto_data = pd.read_csv("data/processed/v_auto_update_20250916_110247.csv")
        
        # Filtre et préparation
        auto_data['Date'] = pd.to_datetime(auto_data['Date'])
        epl_2025_26 = auto_data[auto_data['Date'] >= '2025-08-01'].copy().sort_values('Date')
        
        train_data = auto_data[auto_data['Date'] < '2025-08-01'].copy()
        test_40 = epl_2025_26.head(40).copy()
        
        # Création target
        def create_target_from_result(df):
            df_copy = df.copy()
            mask_no_target = df_copy['target'].isna()
            if mask_no_target.any():
                df_copy.loc[mask_no_target, 'target'] = df_copy.loc[mask_no_target, 'FullTimeResult'].map({
                    'H': 0, 'D': 1, 'A': 2
                })
            return df_copy
        
        train_data = create_target_from_result(train_data)
        test_40 = create_target_from_result(test_40)
        
        # Features
        feature_cols = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        X_train = train_data[feature_cols].fillna(0)
        X_test = test_40[feature_cols].fillna(0)
        y_train = train_data['target']
        y_test = test_40['target']
        
        # Filtrage NaN
        train_valid = y_train.notna()
        test_valid = y_test.notna()
        
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]
        test_40 = test_40[test_valid]
        
        y_test_str = y_test.map({0: 'H', 1: 'D', 2: 'A'})
        
        logger.info(f"📊 Données: Train {len(X_train)}, Test {len(X_test)}")
        
        # Test différentes calibrations
        configs = [
            {"draw_weight": 3.0, "draw_threshold": 0.20, "homeaway_balance": True, "name": "Seuil 0.20 + Balance"},
            {"draw_weight": 4.0, "draw_threshold": 0.25, "homeaway_balance": True, "name": "Seuil 0.25 + Balance"},
            {"draw_weight": 5.0, "draw_threshold": 0.30, "homeaway_balance": True, "name": "Seuil 0.30 + Balance"},
            {"draw_weight": 3.0, "draw_threshold": 0.35, "homeaway_balance": False, "name": "Seuil 0.35 + NoBalance"},
        ]
        
        results = []
        
        for config in configs:
            logger.info(f"\n🧪 Test: {config['name']}")
            
            model = CascadeModelCalibrated(
                draw_weight=config['draw_weight'],
                draw_threshold=config['draw_threshold'], 
                homeaway_balance=config['homeaway_balance']
            )
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            accuracy = accuracy_score(y_test_str, preds) * 100
            
            # Distribution
            dist = pd.Series(preds).value_counts(normalize=True).sort_index() * 100
            draws_predicted = (preds == 'D').sum()
            
            # Performance par classe
            perf_by_class = {}
            for outcome in ['H', 'D', 'A']:
                mask = y_test_str == outcome
                if mask.sum() > 0:
                    correct = (preds[mask] == outcome).sum()
                    total = mask.sum()
                    perf_by_class[outcome] = f"{correct}/{total} ({correct/total*100:.1f}%)"
            
            result = {
                'name': config['name'],
                'accuracy': accuracy,
                'draws_predicted': draws_predicted,
                'dist_H': dist.get('H', 0),
                'dist_D': dist.get('D', 0), 
                'dist_A': dist.get('A', 0),
                'perf_by_class': perf_by_class
            }
            results.append(result)
            
            logger.info(f"   Accuracy: {accuracy:.1f}%")
            logger.info(f"   Draws: {draws_predicted}/40 ({draws_predicted/40*100:.1f}%)")
            logger.info(f"   Distribution: H={dist.get('H', 0):.1f}%, D={dist.get('D', 0):.1f}%, A={dist.get('A', 0):.1f}%")
            logger.info(f"   Performance: {perf_by_class}")
        
        # Résumé comparatif
        logger.info(f"\n🏆 RÉSUMÉ COMPARATIF")
        logger.info(f"=" * 50)
        
        for result in results:
            logger.info(f"{result['name']:<20}: {result['accuracy']:.1f}% accuracy, {result['draws_predicted']} draws")
        
        # Meilleur équilibre
        best_balanced = max(results, key=lambda x: x['accuracy'] if x['draws_predicted'] > 0 else 0)
        logger.info(f"\n🎯 MEILLEUR ÉQUILIBRE: {best_balanced['name']}")
        logger.info(f"   {best_balanced['accuracy']:.1f}% accuracy avec {best_balanced['draws_predicted']} draws détectés")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_calibration()