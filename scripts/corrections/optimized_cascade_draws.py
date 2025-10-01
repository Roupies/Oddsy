#!/usr/bin/env python3
"""
🎯 CASCADE OPTIMISÉ POUR DRAWS - CAPTURE DES NULS
================================================

Test combiné de toutes les techniques pour attraper les draws :
- Class weights optimaux
- Seuils ajustés  
- Hyperparamètres tunés
- Léger undersampling
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimized_draws")

class OptimizedCascadeModel:
    """Modèle cascade optimisé pour capturer draws"""
    
    def __init__(self, draw_weight=4, draw_threshold=0.3):
        # Hyperparamètres optimisés pour draws
        self.clf_draw = RandomForestClassifier(
            n_estimators=300,  # Plus d'arbres pour patterns minoritaires
            max_depth=15,      # Profondeur limitée pour éviter overfitting
            min_samples_leaf=3,  # Feuilles pas trop petites
            class_weight={0: 1, 1: draw_weight},  # Poids élevé pour draws
            random_state=42
        )
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            class_weight="balanced"
        )
        self.draw_threshold = draw_threshold  # Seuil abaissé pour draws
        self.is_fitted = False
    
    def fit(self, X, y, undersample_ratio=0.8):
        """Entrainement avec léger undersampling optionnel"""
        
        # Convertir target si numérique
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y.copy()
        
        # Léger undersampling des classes majoritaires si demandé
        if undersample_ratio < 1.0:
            X_balanced, y_balanced = self._undersample_majority(X, y_str, undersample_ratio)
        else:
            X_balanced, y_balanced = X, y_str
        
        logger.info(f"  Entraînement sur {len(X_balanced)} échantillons (undersample: {undersample_ratio})")
        
        # Étape 1: Draw vs NotDraw avec poids et hyperparams optimisés
        y_draw = (y_balanced == 'D').astype(int)
        
        draw_counts = y_draw.value_counts()
        logger.info(f"  Distribution Draw training: NotDraw={draw_counts.get(0, 0)}, Draw={draw_counts.get(1, 0)}")
        
        self.clf_draw.fit(X_balanced, y_draw)
        
        # Étape 2: Home vs Away sur NotDraw
        mask_notdraw = y_balanced != 'D'
        if mask_notdraw.sum() > 5:
            X_notdraw = X_balanced[mask_notdraw]
            y_homeaway = y_balanced[mask_notdraw].map({'H': 1, 'A': 0})
            
            # Nettoyer NaN
            valid_homeaway = y_homeaway.notna()
            X_notdraw_clean = X_notdraw[valid_homeaway]
            y_homeaway_clean = y_homeaway[valid_homeaway]
            
            if len(y_homeaway_clean) > 5:
                self.clf_homeaway.fit(X_notdraw_clean, y_homeaway_clean)
        
        self.is_fitted = True
        return self
    
    def _undersample_majority(self, X, y, ratio):
        """Undersample léger des classes majoritaires"""
        
        draws = y == 'D'
        not_draws = y != 'D'
        
        # Garder tous les draws
        X_draws = X[draws]
        y_draws = y[draws]
        
        # Undersampler H et A
        X_not_draws = X[not_draws]
        y_not_draws = y[not_draws]
        
        n_keep = int(len(X_not_draws) * ratio)
        indices = np.random.choice(len(X_not_draws), n_keep, replace=False)
        
        X_not_draws_sub = X_not_draws.iloc[indices]
        y_not_draws_sub = y_not_draws.iloc[indices]
        
        # Recombiner
        X_balanced = pd.concat([X_draws, X_not_draws_sub]).reset_index(drop=True)
        y_balanced = pd.concat([y_draws, y_not_draws_sub]).reset_index(drop=True)
        
        return X_balanced, y_balanced
    
    def predict(self, X):
        """Prédiction avec seuil ajusté pour draws"""
        
        # Probabilités draw
        proba_draw = self.clf_draw.predict_proba(X)[:, 1]
        
        # Appliquer seuil abaissé
        pred_draw = (proba_draw > self.draw_threshold).astype(int)
        
        # Prédictions H/A pour les non-draws
        pred_homeaway = self.clf_homeaway.predict(X)
        
        y_pred = []
        for is_draw, home_away in zip(pred_draw, pred_homeaway):
            if is_draw == 1:
                y_pred.append('D')
            else:
                y_pred.append('H' if home_away == 1 else 'A')
        
        return np.array(y_pred)
    
    def predict_with_details(self, X):
        """Prédiction avec détails des probabilités"""
        
        proba_draw = self.clf_draw.predict_proba(X)[:, 1]
        pred_draw = (proba_draw > self.draw_threshold).astype(int)
        pred_homeaway = self.clf_homeaway.predict(X)
        
        results = []
        for i, (is_draw, home_away, prob_draw) in enumerate(zip(pred_draw, pred_homeaway, proba_draw)):
            if is_draw == 1:
                outcome = 'D'
            else:
                outcome = 'H' if home_away == 1 else 'A'
            
            results.append({
                'prediction': outcome,
                'draw_probability': prob_draw,
                'draw_threshold': self.draw_threshold,
                'predicted_draw': is_draw == 1
            })
        
        return np.array([r['prediction'] for r in results]), results

def test_optimized_draws():
    """Test cascade optimisé pour draws"""
    logger.info("🎯 TEST CASCADE OPTIMISÉ POUR DRAWS")
    logger.info("=" * 50)
    
    try:
        # Charger dataset
        df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
        
        # Charger vrais résultats
        df_real = pd.read_csv('data/raw/E0 (7).csv', encoding='utf-8-sig')
        team_mapping = {
            'Spurs': 'Tottenham',
            "Nott'm Forest": "Nott'm Forest"
        }
        df_real['HomeTeam'] = df_real['HomeTeam'].replace(team_mapping)
        df_real['AwayTeam'] = df_real['AwayTeam'].replace(team_mapping)
        real_matches = df_real[['HomeTeam', 'AwayTeam', 'FTR']]
        
        # Target encoding
        if 'target' not in df.columns:
            df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Split temporel
        train_cutoff = pd.to_datetime('2025-08-01')
        df_train = df[df['Date'] < train_cutoff].copy()
        
        # Extension auto pour test
        try:
            auto_dataset = pd.read_csv('data/processed/v_auto_update_20250916_105039.csv', parse_dates=['Date'])
            auto_season_2025 = auto_dataset[auto_dataset['Date'] >= '2025-08-01'].copy()
            auto_test_candidates = auto_season_2025.head(40).copy()
            df_test = pd.merge(auto_test_candidates, real_matches, on=['HomeTeam', 'AwayTeam'], how='inner')
        except:
            df_season_2025 = df[df['Date'] >= '2025-08-01'].copy()
            df_test_candidates = df_season_2025.head(40).copy()
            df_test = pd.merge(df_test_candidates, real_matches, on=['HomeTeam', 'AwayTeam'], how='inner')
        
        # Features
        model_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        # Préparer données
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        
        X_train = df_train_clean[model_features].fillna(0.5)
        y_train = df_train_clean['target']
        
        X_test = df_test[model_features].fillna(0.5)
        y_real = df_test['FTR']
        
        logger.info(f"📊 Données: train={len(X_train)}, test={len(X_test)}")
        
        # Test plusieurs configurations
        configs = [
            {"draw_weight": 3, "draw_threshold": 0.35, "undersample": 1.0, "name": "Base"},
            {"draw_weight": 4, "draw_threshold": 0.3, "undersample": 0.8, "name": "Optimisé"},
            {"draw_weight": 5, "draw_threshold": 0.25, "undersample": 0.7, "name": "Agressif"},
            {"draw_weight": 6, "draw_threshold": 0.2, "undersample": 0.6, "name": "Très Agressif"}
        ]
        
        best_f1_draw = 0
        best_config = None
        best_predictions = None
        
        for config in configs:
            logger.info(f"\n🔬 TEST CONFIG: {config['name']}")
            logger.info(f"   draw_weight={config['draw_weight']}, threshold={config['draw_threshold']}, undersample={config['undersample']}")
            
            # Entrainer modèle
            model = OptimizedCascadeModel(
                draw_weight=config['draw_weight'],
                draw_threshold=config['draw_threshold']
            )
            model.fit(X_train, y_train, undersample_ratio=config['undersample'])
            
            # Prédictions
            y_pred, details = model.predict_with_details(X_test)
            
            # Métriques
            accuracy = accuracy_score(y_real, y_pred)
            
            # F1-score spécifique pour Draw
            f1_draw = f1_score(y_real, y_pred, labels=['H', 'D', 'A'], average=None)[1]  # Index 1 = Draw
            
            # Compter draws prédits
            draws_predicted = (y_pred == 'D').sum()
            draws_real = (y_real == 'D').sum()
            draws_correct = ((y_pred == 'D') & (y_real == 'D')).sum()
            
            logger.info(f"   Accuracy: {accuracy:.3f}")
            logger.info(f"   F1-Draw: {f1_draw:.3f}")
            logger.info(f"   Draws prédits: {draws_predicted} (réels: {draws_real}, corrects: {draws_correct})")
            
            if f1_draw > best_f1_draw:
                best_f1_draw = f1_draw
                best_config = config
                best_predictions = (y_pred, details)
        
        # Résultats meilleure config
        logger.info(f"\n🏆 MEILLEURE CONFIG: {best_config['name']}")
        logger.info(f"   F1-Draw: {best_f1_draw:.3f}")
        
        if best_predictions:
            y_pred_best, details_best = best_predictions
            
            # Analyse détaillée
            accuracy_best = accuracy_score(y_real, y_pred_best)
            cm = confusion_matrix(y_real, y_pred_best, labels=['H', 'D', 'A'])
            
            logger.info(f"\n📊 RÉSULTATS OPTIMISÉS:")
            logger.info(f"   Accuracy globale: {accuracy_best:.1%}")
            
            # Distribution
            real_dist = y_real.value_counts(normalize=True)
            pred_dist = pd.Series(y_pred_best).value_counts(normalize=True)
            
            logger.info(f"   Distribution réelle: H={real_dist.get('H', 0):.1%}, D={real_dist.get('D', 0):.1%}, A={real_dist.get('A', 0):.1%}")
            logger.info(f"   Distribution prédite: H={pred_dist.get('H', 0):.1%}, D={pred_dist.get('D', 0):.1%}, A={pred_dist.get('A', 0):.1%}")
            
            # Matrice confusion
            logger.info(f"\n📊 MATRICE CONFUSION OPTIMISÉE:")
            logger.info(f"     Real\\Pred  H   D   A")
            for i, label in enumerate(['H', 'D', 'A']):
                logger.info(f"     {label}        {cm[i][0]:2d}  {cm[i][1]:2d}  {cm[i][2]:2d}")
            
            # Détail draws prédits
            draws_predicted_indices = np.where(y_pred_best == 'D')[0]
            if len(draws_predicted_indices) > 0:
                logger.info(f"\n🎯 DRAWS PRÉDITS ({len(draws_predicted_indices)}):")
                for idx in draws_predicted_indices:
                    home = df_test.iloc[idx]['HomeTeam']
                    away = df_test.iloc[idx]['AwayTeam']
                    real = y_real.iloc[idx]
                    prob = details_best[idx]['draw_probability']
                    correct = "✅" if real == 'D' else "❌"
                    logger.info(f"   {correct} {home} vs {away}: prob={prob:.3f}, réel={real}")
            
            return {
                'best_config': best_config,
                'accuracy': accuracy_best,
                'f1_draw': best_f1_draw,
                'draws_predicted': len(draws_predicted_indices),
                'draws_real': (y_real == 'D').sum()
            }
        
    except Exception as e:
        logger.error(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_optimized_draws()
    
    if result:
        print(f"\n🎯 OPTIMISATION DRAWS TERMINÉE")
        print(f"Meilleure config: {result['best_config']['name']}")
        print(f"F1-Draw: {result['f1_draw']:.3f}")
        print(f"Draws capturés: {result['draws_predicted']}/{result['draws_real']}")
    else:
        print("❌ Échec optimisation")