#!/usr/bin/env python3
"""
🎯 TEST ACCURACY GLOBALE 40 MATCHS J1-J4
=====================================
Test complet sur 40 matchs EPL 2025-26 avec accuracy globale.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("accuracy_40")

class CascadeModel:
    def __init__(self, draw_weight=3.0, draw_threshold=0.27):
        self.clf_draw = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight={0: 1, 1: draw_weight}, random_state=42
        )
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
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
            # Filtrage des NaN dans y_homeaway
            valid_homeaway = y_homeaway.notna()
            if valid_homeaway.sum() > 5:
                self.clf_homeaway.fit(X_notdraw[valid_homeaway], y_homeaway[valid_homeaway])
    
    def predict(self, X):
        # 1. Prédiction draws
        draw_proba_output = self.clf_draw.predict_proba(X)
        
        # Gestion du cas où le modèle n'a qu'une seule classe
        if draw_proba_output.shape[1] == 1:
            # Si une seule classe, probabilité draw = 0 ou 1 selon la classe
            single_class = self.clf_draw.classes_[0]
            draw_proba = np.full(len(X), single_class)
        else:
            draw_proba = draw_proba_output[:, 1]
        
        is_draw = draw_proba >= self.draw_threshold
        
        # 2. Prédiction Home/Away pour non-draws
        predictions = np.full(len(X), 'D')  # Par défaut draw
        
        # Pour les non-draws prédits
        mask_notdraw = ~is_draw
        if mask_notdraw.sum() > 0 and hasattr(self, 'clf_homeaway'):
            try:
                homeaway_pred = self.clf_homeaway.predict(X[mask_notdraw])
                predictions[mask_notdraw] = np.where(homeaway_pred == 1, 'H', 'A')
            except:
                # Fallback: prédire Home par défaut
                predictions[mask_notdraw] = 'H'
        
        return predictions

def test_accuracy_40():
    try:
        logger.info("🎯 TEST ACCURACY GLOBALE 40 MATCHS J1-J4")
        logger.info("=" * 50)
        
        # Chargement données
        auto_data = pd.read_csv("data/processed/v_auto_update_20250916_110247.csv")
        logger.info(f"📊 Dataset auto chargé: {len(auto_data)} matchs")
        
        # Filtre 2025-26
        auto_data['Date'] = pd.to_datetime(auto_data['Date'])
        epl_2025_26 = auto_data[auto_data['Date'] >= '2025-08-01'].copy().sort_values('Date')
        logger.info(f"📅 EPL 2025-26: {len(epl_2025_26)} matchs")
        
        # Split train/test
        train_data = auto_data[auto_data['Date'] < '2025-08-01'].copy()
        test_40 = epl_2025_26.head(40).copy()
        
        logger.info(f"📊 Train: {len(train_data)}, Test: {len(test_40)} matchs")
        logger.info(f"📅 Test période: {test_40['Date'].min()} à {test_40['Date'].max()}")
        
        # Features communes disponibles
        feature_cols = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        # Vérification features
        available_features = [f for f in feature_cols if f in auto_data.columns]
        logger.info(f"🎯 Features disponibles: {len(available_features)}")
        
        if len(available_features) < 8:
            logger.warning(f"⚠️  Seulement {len(available_features)} features disponibles")
            
        # Création target à partir de FullTimeResult pour données historiques
        def create_target_from_result(df):
            df_copy = df.copy()
            # Pour les données historiques sans target, créer à partir de FullTimeResult
            mask_no_target = df_copy['target'].isna()
            if mask_no_target.any():
                df_copy.loc[mask_no_target, 'target'] = df_copy.loc[mask_no_target, 'FullTimeResult'].map({
                    'H': 0, 'D': 1, 'A': 2
                })
            return df_copy
        
        train_data = create_target_from_result(train_data)
        test_40 = create_target_from_result(test_40)
        
        # Préparation données
        X_train = train_data[available_features].fillna(0)
        X_test = test_40[available_features].fillna(0)
        
        y_train = train_data['target']
        y_test = test_40['target']
        
        # Filtrage des NaN restants dans les targets
        train_valid_mask = y_train.notna()
        test_valid_mask = y_test.notna()
        
        X_train = X_train[train_valid_mask]
        y_train = y_train[train_valid_mask]
        
        X_test = X_test[test_valid_mask]
        y_test = y_test[test_valid_mask]
        test_40 = test_40[test_valid_mask]
        
        logger.info(f"📊 Après création target et filtrage: Train {len(X_train)}, Test {len(X_test)}")
        
        # Conversion target en string pour cohérence
        y_train_str = y_train.map({0: 'H', 1: 'D', 2: 'A'})
        y_test_str = y_test.map({0: 'H', 1: 'D', 2: 'A'})
        
        logger.info(f"✅ Données préparées, test sur {len(test_40)} matchs")
        
        # 1. Modèle Baseline (RandomForest standard)
        baseline_model = RandomForestClassifier(n_estimators=150, random_state=42)
        baseline_model.fit(X_train, y_train_str)
        baseline_preds = baseline_model.predict(X_test)
        
        # 2. Modèle Cascade
        cascade_model = CascadeModel(draw_weight=2.5, draw_threshold=0.40)
        cascade_model.fit(X_train, y_train)
        cascade_preds = cascade_model.predict(X_test)
        
        # CALCUL ACCURACY
        baseline_accuracy = accuracy_score(y_test_str, baseline_preds) * 100
        cascade_accuracy = accuracy_score(y_test_str, cascade_preds) * 100
        
        logger.info(f"\n🏆 RÉSULTATS ACCURACY SUR 40 MATCHS")
        logger.info(f"=" * 40)
        logger.info(f"📊 Baseline: {baseline_accuracy:.1f}%")
        logger.info(f"📊 Cascade:  {cascade_accuracy:.1f}%")
        logger.info(f"📈 Différence: {cascade_accuracy - baseline_accuracy:+.1f}pp")
        
        # DISTRIBUTION PRÉDICTIONS
        baseline_dist = pd.Series(baseline_preds).value_counts(normalize=True).sort_index() * 100
        cascade_dist = pd.Series(cascade_preds).value_counts(normalize=True).sort_index() * 100
        
        logger.info(f"\n📊 DISTRIBUTIONS PRÉDITES:")
        logger.info(f"   Baseline: H={baseline_dist.get('H', 0):.1f}%, D={baseline_dist.get('D', 0):.1f}%, A={baseline_dist.get('A', 0):.1f}%")
        logger.info(f"   Cascade:  H={cascade_dist.get('H', 0):.1f}%, D={cascade_dist.get('D', 0):.1f}%, A={cascade_dist.get('A', 0):.1f}%")
        
        # DÉTECTION DRAWS
        baseline_draws = (baseline_preds == 'D').sum()
        cascade_draws = (cascade_preds == 'D').sum()
        
        logger.info(f"\n🎯 DÉTECTION DRAWS:")
        logger.info(f"   Baseline: {baseline_draws}/40 draws prédits ({baseline_draws/40*100:.1f}%)")
        logger.info(f"   Cascade:  {cascade_draws}/40 draws prédits ({cascade_draws/40*100:.1f}%)")
        
        # PERFORMANCE PAR CLASSE
        logger.info(f"\n📈 PERFORMANCE PAR CLASSE RÉELLE:")
        
        for outcome in ['H', 'D', 'A']:
            mask = y_test_str == outcome
            if mask.sum() > 0:
                actual_count = mask.sum()
                baseline_correct = (baseline_preds[mask] == outcome).sum()
                cascade_correct = (cascade_preds[mask] == outcome).sum()
                
                baseline_rate = baseline_correct / actual_count * 100
                cascade_rate = cascade_correct / actual_count * 100
                
                logger.info(f"   {outcome} ({actual_count} vrais): Baseline {baseline_rate:.1f}%, Cascade {cascade_rate:.1f}%")
        
        # ÉCHANTILLON PRÉDICTIONS
        logger.info(f"\n🔍 ÉCHANTILLON PRÉDICTIONS (8 premiers):")
        for i in range(min(8, len(test_40))):
            row = test_40.iloc[i]
            actual = y_test_str.iloc[i]
            baseline_pred = baseline_preds[i]
            cascade_pred = cascade_preds[i]
            
            baseline_mark = "✓" if baseline_pred == actual else "✗"
            cascade_mark = "✓" if cascade_pred == actual else "✗"
            
            logger.info(f"   {row['Date'].strftime('%Y-%m-%d')}: {row['HomeTeam']} vs {row['AwayTeam']}")
            logger.info(f"      Réel: {actual} | Baseline: {baseline_pred} {baseline_mark} | Cascade: {cascade_pred} {cascade_mark}")
        
        print(f"\n🎯 RÉSUMÉ FINAL 40 MATCHS:")
        print(f"   Baseline: {baseline_accuracy:.1f}% accuracy, {baseline_draws} draws")
        print(f"   Cascade:  {cascade_accuracy:.1f}% accuracy, {cascade_draws} draws") 
        print(f"   Amélioration: {cascade_accuracy - baseline_accuracy:+.1f}pp accuracy, +{cascade_draws - baseline_draws} draws")
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'cascade_accuracy': cascade_accuracy,
            'accuracy_diff': cascade_accuracy - baseline_accuracy,
            'baseline_draws': baseline_draws,
            'cascade_draws': cascade_draws
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_accuracy_40()
    
    if not result:
        print("❌ Échec test accuracy 40 matchs")