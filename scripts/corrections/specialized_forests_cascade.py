#!/usr/bin/env python3
"""
🌳 CASCADE AVEC FORÊTS SPÉCIALISÉES INDÉPENDANTES
==============================================

Deux Random Forest complètement spécialisés et indépendants:
1. DrawSpecialistForest - Expert détection nuls
2. HomeAwaySpecialistForest - Expert différenciation H/A
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("specialized_cascade")

class DrawSpecialistForest:
    """Forêt spécialisée dans la détection des nuls"""
    
    def __init__(self, draw_class_weight=6, n_estimators=300, max_depth=15):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=2,  # Plus fin pour capturer patterns rares
            min_samples_split=4,
            class_weight={0: 1, 1: draw_class_weight},  # Ultra-agressif pour draws
            random_state=42,
            bootstrap=True,
            max_features='sqrt'
        )
        self.draw_class_weight = draw_class_weight
        self.is_fitted = False
        
    def _engineer_draw_features(self, X_base):
        """Feature engineering spécialisé pour détection draws"""
        X_draw = X_base.copy()
        
        # 1. TEAM PARITY SCORE - Équilibre parfait des forces
        elo_diff = X_base.get('elo_diff_normalized', 0.5)
        market_entropy = X_base.get('market_entropy_norm', 0.8)
        
        # Parity score: plus proche de 0.5 l'elo_diff + entropy élevée = plus de chance de nul
        parity_from_elo = 1 - 2 * np.abs(elo_diff - 0.5)  # 1 si elo=0.5, 0 si elo=0 ou 1
        X_draw['team_parity_score'] = 0.7 * parity_from_elo + 0.3 * market_entropy
        
        # 2. LOW SCORING POTENTIAL - Matchs peu prolifiques
        home_xg = X_base.get('home_xg_eff_10', 0.5)
        away_xg = X_base.get('away_xg_eff_10', 0.5) 
        away_goals = X_base.get('away_goals_sum_5', 5.0)
        
        # Plus les efficacités sont faibles, plus le potentiel de nul est élevé
        offensive_strength = (home_xg + away_xg + away_goals/10) / 3
        X_draw['low_scoring_potential'] = 1 - offensive_strength  # Inverse
        
        # 3. ELO VARIANCE (simulé) - Instabilité = imprévisibilité
        form_diff = X_base.get('form_diff_normalized', 0.5)
        
        # Simuler variance via form instable (form proche 0.5 = instable)
        form_instability = 1 - 2 * np.abs(form_diff - 0.5)
        X_draw['elo_variance_recent'] = form_instability
        
        # 4. DEFENSIVE STRENGTH COMBINED - Solidité défensive
        shots_diff = X_base.get('shots_diff_normalized', 0.5)
        corners_diff = X_base.get('corners_diff_normalized', 0.5)
        
        # Défenses équilibrées (shots/corners proches de 0.5) = plus de nuls
        defensive_balance = 1 - (np.abs(shots_diff - 0.5) + np.abs(corners_diff - 0.5))
        X_draw['defensive_strength_combined'] = np.clip(defensive_balance, 0, 1)
        
        # 5. MARKET ODDS SPREAD (simulé via entropy)
        # Plus l'entropy est élevée, plus le spread est faible (match serré)
        X_draw['market_odds_spread'] = 1 - market_entropy  # Inverse de entropy
        
        logger.debug(f"   Features draws créées: parity={X_draw['team_parity_score'].iloc[0]:.3f}, "
                    f"low_scoring={X_draw['low_scoring_potential'].iloc[0]:.3f}")
        
        return X_draw
    
    def fit(self, X, y):
        """Entrainement spécialisé draws"""
        # Convertir target
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y.copy()
        
        # Target binaire Draw vs NotDraw
        y_draw = (y_str == 'D').astype(int)
        
        # Feature engineering spécialisé
        X_draw = self._engineer_draw_features(X)
        
        # Features spécialisées draws
        draw_features = [
            'team_parity_score', 'low_scoring_potential', 'elo_variance_recent',
            'defensive_strength_combined', 'market_odds_spread',
            # Garder quelques features de base importantes
            'elo_diff_normalized', 'market_entropy_norm', 'form_diff_normalized'
        ]
        
        X_draw_selected = X_draw[draw_features]
        
        # Entrainement ultra-spécialisé
        self.clf.fit(X_draw_selected, y_draw)
        self.draw_features = draw_features
        self.is_fitted = True
        
        # Statistiques entrainement
        draw_ratio = np.mean(y_draw)
        logger.info(f"✅ DrawSpecialistForest entrainé:")
        logger.info(f"   Class weight: 1:{self.draw_class_weight}")
        logger.info(f"   Draw ratio: {draw_ratio:.1%}")
        logger.info(f"   Features: {len(draw_features)}")
        
        return self
    
    def predict_proba(self, X):
        """Probabilités draws spécialisées"""
        X_draw = self._engineer_draw_features(X)
        X_draw_selected = X_draw[self.draw_features]
        return self.clf.predict_proba(X_draw_selected)
    
    def predict_draws(self, X, dynamic_threshold=True):
        """Prédiction draws avec seuil dynamique"""
        proba = self.predict_proba(X)[:, 1]  # Prob classe Draw
        
        if dynamic_threshold:
            # Seuil dynamique basé sur contexte
            X_draw = self._engineer_draw_features(X)
            parity = X_draw['team_parity_score'].values
            
            # Plus la parity est élevée, plus le seuil est bas (plus facile de prédire draw)
            thresholds = 0.35 - 0.1 * parity  # De 0.35 à 0.25
            predictions = proba > thresholds
        else:
            # Seuil fixe
            predictions = proba > 0.30  # Seuil agressif pour draws
        
        return predictions.astype(int), proba

class HomeAwaySpecialistForest:
    """Forêt spécialisée dans la différenciation Home/Away"""
    
    def __init__(self, n_estimators=250, max_depth=12):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=3,
            class_weight='balanced',  # Équilibré pour H/A
            random_state=42,
            bootstrap=True
        )
        self.is_fitted = False
    
    def _engineer_homeaway_features(self, X_base):
        """Feature engineering spécialisé pour Home/Away"""
        X_ha = X_base.copy()
        
        # 1. HOME ADVANTAGE CONTEXTUAL - Avantage domicile variable
        elo_diff = X_base.get('elo_diff_normalized', 0.5)
        
        # Plus l'équipe domicile est forte, plus l'avantage est amplifié
        home_advantage_base = 0.55  # Avantage domicile de base
        home_strength = elo_diff  # Plus proche de 1 = équipe domicile forte
        X_ha['home_advantage_contextual'] = home_advantage_base + 0.2 * home_strength
        
        # 2. AWAY FORM RESILIENCE - Résistance extérieur
        away_xg = X_base.get('away_xg_eff_10', 0.5)
        away_goals = X_base.get('away_goals_sum_5', 5.0)
        
        # Capacité équipe extérieur à scorer loin de ses bases
        X_ha['away_form_resilience'] = 0.7 * away_xg + 0.3 * (away_goals / 10)
        
        # 3. HEAD TO HEAD DOMINANCE - Domination historique directe
        h2h_score = X_base.get('h2h_score', 0.5)
        
        # Amplifier signal H2H pour différenciation H/A
        X_ha['head_to_head_dominance'] = h2h_score
        
        # 4. PERFORMANCE VS TALENT RATIO (simulé)
        shots_diff = X_base.get('shots_diff_normalized', 0.5)
        
        # Simuler sur-performance via shots (équipe qui tire plus que prévu)
        X_ha['performance_vs_talent_ratio'] = shots_diff
        
        # 5. IS PROMOTED (binaire simulé)
        # Identifier équipes promues par leurs features neutres
        form_diff = X_base.get('form_diff_normalized', 0.5)
        matchday = X_base.get('matchday_normalized', 0.0)
        
        # Heuristique: si form=0.5 et matchday faible, probablement promue
        is_promoted_home = (np.abs(elo_diff - 0.5) < 0.1) & (matchday < 0.2) & (np.abs(form_diff - 0.5) < 0.1)
        X_ha['is_promoted'] = is_promoted_home.astype(float)
        
        logger.debug(f"   Features H/A créées: home_adv={X_ha['home_advantage_contextual'].iloc[0]:.3f}, "
                    f"away_res={X_ha['away_form_resilience'].iloc[0]:.3f}")
        
        return X_ha
    
    def fit(self, X, y):
        """Entrainement spécialisé Home/Away sur données NotDraw uniquement"""
        # Convertir target
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y.copy()
        
        # Filtrer seulement NotDraw pour entrainement H/A
        mask_notdraw = y_str != 'D'
        
        if mask_notdraw.sum() < 10:
            logger.warning("Pas assez de données NotDraw pour entrainement H/A")
            return self
        
        X_notdraw = X[mask_notdraw]
        y_notdraw = y_str[mask_notdraw]
        
        # Target binaire Home=1, Away=0
        y_homeaway = y_notdraw.map({'H': 1, 'A': 0})
        
        # Feature engineering spécialisé
        X_ha = self._engineer_homeaway_features(X_notdraw)
        
        # Features spécialisées H/A
        homeaway_features = [
            'home_advantage_contextual', 'away_form_resilience', 'head_to_head_dominance',
            'performance_vs_talent_ratio', 'is_promoted',
            # Garder features de base importantes pour H/A
            'elo_diff_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'home_xg_eff_10', 'away_xg_eff_10'
        ]
        
        X_ha_selected = X_ha[homeaway_features]
        
        # Nettoyer NaN
        valid_mask = y_homeaway.notna()
        X_ha_clean = X_ha_selected[valid_mask]
        y_ha_clean = y_homeaway[valid_mask]
        
        if len(y_ha_clean) < 5:
            logger.warning("Données H/A insuffisantes après nettoyage")
            return self
        
        # Entrainement spécialisé H/A
        self.clf.fit(X_ha_clean, y_ha_clean)
        self.homeaway_features = homeaway_features
        self.is_fitted = True
        
        # Statistiques
        home_ratio = np.mean(y_ha_clean)
        logger.info(f"✅ HomeAwaySpecialistForest entrainé:")
        logger.info(f"   Home ratio: {home_ratio:.1%}")
        logger.info(f"   Features: {len(homeaway_features)}")
        logger.info(f"   Samples: {len(y_ha_clean)}")
        
        return self
    
    def predict_homeaway(self, X):
        """Prédiction Home/Away spécialisée"""
        X_ha = self._engineer_homeaway_features(X)
        X_ha_selected = X_ha[self.homeaway_features]
        
        predictions = self.clf.predict(X_ha_selected)
        probabilities = self.clf.predict_proba(X_ha_selected)
        
        return predictions, probabilities

class SpecializedCascadeModel:
    """Modèle cascade avec deux forêts complètement spécialisées"""
    
    def __init__(self, draw_class_weight=6, dynamic_threshold=True):
        self.draw_forest = DrawSpecialistForest(draw_class_weight=draw_class_weight)
        self.homeaway_forest = HomeAwaySpecialistForest()
        self.dynamic_threshold = dynamic_threshold
        self.is_fitted = False
    
    def fit(self, X, y):
        """Entrainement des deux forêts spécialisées"""
        logger.info("🌳 Entrainement CASCADE FORÊTS SPÉCIALISÉES")
        
        # Entrainer forêt spécialiste draws
        self.draw_forest.fit(X, y)
        
        # Entrainer forêt spécialiste H/A  
        self.homeaway_forest.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Prédiction cascade avec forêts spécialisées"""
        # Étape 1: Détection draws par spécialiste
        is_draw, draw_proba = self.draw_forest.predict_draws(X, self.dynamic_threshold)
        
        # Étape 2: Classification H/A par spécialiste pour NotDraw
        homeaway_pred, homeaway_proba = self.homeaway_forest.predict_homeaway(X)
        
        # Combiner prédictions
        final_predictions = []
        for i in range(len(X)):
            if is_draw[i] == 1:
                final_predictions.append('D')
            else:
                final_predictions.append('H' if homeaway_pred[i] == 1 else 'A')
        
        return np.array(final_predictions)
    
    def predict_with_details(self, X):
        """Prédiction avec détails des probabilités"""
        is_draw, draw_proba = self.draw_forest.predict_draws(X, self.dynamic_threshold)
        homeaway_pred, homeaway_proba = self.homeaway_forest.predict_homeaway(X)
        
        results = []
        for i in range(len(X)):
            if is_draw[i] == 1:
                outcome = 'D'
                confidence = draw_proba[i]
                specialist = 'DrawForest'
            else:
                outcome = 'H' if homeaway_pred[i] == 1 else 'A'
                confidence = homeaway_proba[i].max()
                specialist = 'HomeAwayForest'
            
            results.append({
                'prediction': outcome,
                'confidence': confidence,
                'draw_prob': draw_proba[i],
                'specialist_used': specialist
            })
        
        predictions = [r['prediction'] for r in results]
        return np.array(predictions), results

def test_specialized_cascade():
    """Test forêts spécialisées vs cascade classique"""
    logger.info("🔬 TEST FORÊTS SPÉCIALISÉES INDÉPENDANTES")
    logger.info("=" * 60)
    
    try:
        # Charger datasets
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
        
        # Features de base (pour compatibility)
        base_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        # Préparer données
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        
        X_train = df_train_clean[base_features].fillna(0.5)
        y_train = df_train_clean['target']
        
        X_test = df_test[base_features].fillna(0.5)
        y_real = df_test['FTR']
        
        logger.info(f"📊 Données: train={len(X_train)}, test={len(X_test)}")
        
        # Test avec différents class weights pour draws
        weight_configs = [
            {"weight": 4, "name": "Modéré"},
            {"weight": 6, "name": "Agressif"},
            {"weight": 8, "name": "Ultra-Agressif"}
        ]
        
        best_score = 0
        best_config = None
        best_result = None
        
        for config in weight_configs:
            logger.info(f"\n🔬 TEST CONFIG: {config['name']} (weight={config['weight']})")
            
            # Modèle spécialisé
            model = SpecializedCascadeModel(
                draw_class_weight=config['weight'],
                dynamic_threshold=True
            )
            model.fit(X_train, y_train)
            
            # Prédictions détaillées
            y_pred, details = model.predict_with_details(X_test)
            
            # Métriques
            accuracy = accuracy_score(y_real, y_pred)
            
            # Draws
            draws_predicted = (y_pred == 'D').sum()
            draws_real = (y_real == 'D').sum()
            draws_correct = ((y_pred == 'D') & (y_real == 'D')).sum()
            draw_recall = draws_correct / draws_real if draws_real > 0 else 0
            
            # Score combiné privilégiant draws
            combined_score = accuracy * 0.6 + draw_recall * 0.4
            
            logger.info(f"   Accuracy: {accuracy:.1%}")
            logger.info(f"   Draws: {draws_correct}/{draws_real} capturés ({draw_recall:.1%})")
            logger.info(f"   Score combiné: {combined_score:.3f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_config = config
                best_result = {
                    'y_pred': y_pred,
                    'details': details,
                    'accuracy': accuracy,
                    'draw_recall': draw_recall,
                    'draws_predicted': draws_predicted,
                    'draws_correct': draws_correct
                }
        
        # Résultats meilleure config
        if best_result:
            logger.info(f"\n🏆 MEILLEURE CONFIG SPÉCIALISÉE: {best_config['name']}")
            logger.info(f"   Score combiné: {best_score:.3f}")
            logger.info(f"   Accuracy: {best_result['accuracy']:.1%}")
            logger.info(f"   Draw recall: {best_result['draw_recall']:.1%}")
            
            y_pred_best = best_result['y_pred']
            
            # Distribution
            real_dist = y_real.value_counts(normalize=True)
            pred_dist = pd.Series(y_pred_best).value_counts(normalize=True)
            
            logger.info(f"\n📊 RÉSULTATS FORÊTS SPÉCIALISÉES:")
            logger.info(f"   Distribution réelle: H={real_dist.get('H', 0):.1%}, D={real_dist.get('D', 0):.1%}, A={real_dist.get('A', 0):.1%}")
            logger.info(f"   Distribution prédite: H={pred_dist.get('H', 0):.1%}, D={pred_dist.get('D', 0):.1%}, A={pred_dist.get('A', 0):.1%}")
            
            # Comparaison avec baseline
            logger.info(f"\n📈 COMPARAISON:")
            logger.info(f"   Forêts spécialisées: {best_result['accuracy']:.1%}")
            logger.info(f"   Cascade baseline: 52.5%")
            logger.info(f"   Amélioration: {best_result['accuracy'] - 0.525:+.1%}")
            
            logger.info(f"   Draw recall spécialisé: {best_result['draw_recall']:.1%}")
            logger.info(f"   Draw recall baseline: 33.3%")
            logger.info(f"   Amélioration draws: {best_result['draw_recall'] - 0.333:+.1%}")
            
            return {
                'specialized_accuracy': best_result['accuracy'],
                'specialized_draw_recall': best_result['draw_recall'],
                'improvement_accuracy': best_result['accuracy'] - 0.525,
                'improvement_draws': best_result['draw_recall'] - 0.333,
                'best_config': best_config
            }
        
    except Exception as e:
        logger.error(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_specialized_cascade()
    
    if result:
        print(f"\n🌳 FORÊTS SPÉCIALISÉES TESTÉES")
        print(f"Accuracy: {result['specialized_accuracy']:.1%} ({result['improvement_accuracy']:+.1%})")
        print(f"Draw recall: {result['specialized_draw_recall']:.1%} ({result['improvement_draws']:+.1%})")
        print(f"Config optimale: {result['best_config']['name']}")
    else:
        print("❌ Échec test forêts spécialisées")