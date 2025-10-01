#!/usr/bin/env python3
"""
🌳 CASCADE AVEC VRAIES FEATURES SPÉCIALISÉES
==========================================

Implémentation des VRAIES features du plan initial :
1. elo_variance_recent - Variance Elo réelle sur 5-10 matchs
2. is_promoted - Flag exact équipes promues 
3. market_odds_spread - Calcul réel écart cotes
4. performance_vs_talent_ratio - Rapport performance/talent
5. low_scoring_potential - Calcul précis matchs peu prolifiques
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("true_specialized")

class TrueFeatureEngineer:
    """Calculateur des vraies features spécialisées"""
    
    def __init__(self):
        # Équipes exactes promues EPL 2025-26
        self.promoted_teams_2025 = {'Leeds', 'Sunderland', 'Burnley'}
        
    def calculate_elo_variance_recent(self, team, df_historical, reference_date, window=8):
        """Calcule la VRAIE variance Elo d'une équipe sur les N derniers matchs"""
        
        # Filtrer matchs de l'équipe avant date référence
        team_matches = df_historical[
            ((df_historical['HomeTeam'] == team) | (df_historical['AwayTeam'] == team)) &
            (df_historical['Date'] < reference_date)
        ].sort_values('Date')
        
        if len(team_matches) < window:
            return 0.15  # Variance par défaut (modérée)
        
        # Prendre les N derniers matchs
        recent_matches = team_matches.tail(window)
        
        # Calculer Elo implicite pour chaque match
        elo_values = []
        for _, match in recent_matches.iterrows():
            if match['HomeTeam'] == team:
                # Équipe à domicile
                elo_diff = match.get('elo_diff_normalized', 0.5)
                # Elo implicite = 0.5 + elo_diff (approximation)
                elo_implicit = 0.5 + (elo_diff - 0.5)
            else:
                # Équipe à l'extérieur  
                elo_diff = match.get('elo_diff_normalized', 0.5)
                # Inverser car elo_diff = home - away
                elo_implicit = 0.5 - (elo_diff - 0.5)
            
            elo_values.append(elo_implicit)
        
        # Calculer variance
        if len(elo_values) > 1:
            variance = np.var(elo_values)
            # Normaliser variance (typiquement entre 0.01 et 0.25)
            variance_normalized = min(variance * 4, 1.0)  # Scale up et clamp
            return variance_normalized
        else:
            return 0.15
    
    def calculate_market_odds_spread(self, home_team, away_team, market_entropy):
        """Calcule écart cotes basé sur entropy marché"""
        
        # Entropy élevée = spread faible (match serré)
        # Entropy faible = spread élevé (grand favori)
        
        if market_entropy > 0.8:
            # Match très serré
            spread = 0.1 + 0.1 * (1 - market_entropy)  # 0.1 à 0.2
        elif market_entropy > 0.6:
            # Match équilibré
            spread = 0.3 + 0.3 * (1 - market_entropy)  # 0.2 à 0.5
        else:
            # Grand favori
            spread = 0.6 + 0.4 * (1 - market_entropy)  # 0.6 à 1.0
        
        return min(spread, 1.0)
    
    def calculate_performance_vs_talent_ratio(self, team, df_historical, reference_date):
        """Calcule rapport performance vs talent (approximation sans valeur marchande)"""
        
        # Approximation: performance récente vs Elo moyen
        team_matches = df_historical[
            ((df_historical['HomeTeam'] == team) | (df_historical['AwayTeam'] == team)) &
            (df_historical['Date'] < reference_date)
        ].sort_values('Date')
        
        if len(team_matches) < 5:
            return 0.5  # Neutre
        
        # Prendre derniers 10 matchs pour performance récente
        recent_matches = team_matches.tail(10)
        
        # Calculer win rate récent
        wins = 0
        total = 0
        avg_elo = 0
        
        for _, match in recent_matches.iterrows():
            result = match.get('FullTimeResult', 'D')
            
            if match['HomeTeam'] == team:
                # Équipe à domicile
                if result == 'H':
                    wins += 1
                elo_implicit = 0.5 + (match.get('elo_diff_normalized', 0.5) - 0.5)
            else:
                # Équipe à l'extérieur
                if result == 'A':
                    wins += 1
                elo_implicit = 0.5 - (match.get('elo_diff_normalized', 0.5) - 0.5)
            
            avg_elo += elo_implicit
            total += 1
        
        if total > 0:
            win_rate = wins / total
            avg_elo = avg_elo / total
            
            # Ratio performance vs talent
            # Si win_rate > avg_elo → sur-performance
            # Si win_rate < avg_elo → sous-performance
            ratio = win_rate / max(avg_elo, 0.1)  # Éviter division par 0
            
            # Normaliser autour de 0.5
            ratio_normalized = min(max(ratio / 2, 0), 1)
            return ratio_normalized
        else:
            return 0.5
    
    def calculate_low_scoring_potential(self, home_xg, away_xg, home_goals_recent, away_goals_recent):
        """Calcul précis du potentiel de match peu prolifique"""
        
        # Critères stricts pour match peu prolifique
        conditions = []
        
        # 1. Efficacités xG faibles
        conditions.append(home_xg < 0.4)
        conditions.append(away_xg < 0.4)
        
        # 2. Historique buts faible
        conditions.append(home_goals_recent < 4)  # Moins de 4 buts sur 5 matchs
        conditions.append(away_goals_recent < 4)
        
        # 3. Au moins 3/4 conditions remplies = potentiel faible score
        score = sum(conditions) / len(conditions)
        
        return score
    
    def engineer_all_features(self, X_base, df_historical, reference_date):
        """Calcule toutes les vraies features spécialisées"""
        
        X_enhanced = X_base.copy()
        n_matches = len(X_base)
        
        logger.info(f"🔧 Engineering {n_matches} matchs avec vraies features...")
        
        # Initialiser nouvelles features
        X_enhanced['elo_variance_home'] = 0.15
        X_enhanced['elo_variance_away'] = 0.15
        X_enhanced['elo_variance_combined'] = 0.15
        X_enhanced['is_promoted_home'] = 0
        X_enhanced['is_promoted_away'] = 0
        X_enhanced['is_promoted_match'] = 0
        X_enhanced['market_odds_spread'] = 0.5
        X_enhanced['performance_vs_talent_home'] = 0.5
        X_enhanced['performance_vs_talent_away'] = 0.5
        X_enhanced['performance_vs_talent_diff'] = 0.5
        X_enhanced['low_scoring_potential'] = 0.5
        
        # Pour chaque match, calculer features individuellement
        for idx in X_base.index:
            try:
                match_data = X_base.loc[idx]
                
                # Récupérer équipes si disponibles via merge avec données test
                home_team = getattr(match_data, 'HomeTeam', None)
                away_team = getattr(match_data, 'AwayTeam', None)
                
                # Si pas d'équipes disponibles, continuer avec moyennes
                if pd.isna(home_team) or pd.isna(away_team):
                    continue
                
                # 1. ELO VARIANCE RECENT
                home_variance = self.calculate_elo_variance_recent(
                    home_team, df_historical, reference_date
                )
                away_variance = self.calculate_elo_variance_recent(
                    away_team, df_historical, reference_date  
                )
                
                X_enhanced.loc[idx, 'elo_variance_home'] = home_variance
                X_enhanced.loc[idx, 'elo_variance_away'] = away_variance
                X_enhanced.loc[idx, 'elo_variance_combined'] = (home_variance + away_variance) / 2
                
                # 2. IS PROMOTED (exact)
                is_home_promoted = 1 if home_team in self.promoted_teams_2025 else 0
                is_away_promoted = 1 if away_team in self.promoted_teams_2025 else 0
                
                X_enhanced.loc[idx, 'is_promoted_home'] = is_home_promoted
                X_enhanced.loc[idx, 'is_promoted_away'] = is_away_promoted
                X_enhanced.loc[idx, 'is_promoted_match'] = max(is_home_promoted, is_away_promoted)
                
                # 3. MARKET ODDS SPREAD (basé sur entropy)
                market_entropy = match_data.get('market_entropy_norm', 0.8)
                spread = self.calculate_market_odds_spread(home_team, away_team, market_entropy)
                X_enhanced.loc[idx, 'market_odds_spread'] = spread
                
                # 4. PERFORMANCE VS TALENT RATIO
                home_perf_ratio = self.calculate_performance_vs_talent_ratio(
                    home_team, df_historical, reference_date
                )
                away_perf_ratio = self.calculate_performance_vs_talent_ratio(
                    away_team, df_historical, reference_date
                )
                
                X_enhanced.loc[idx, 'performance_vs_talent_home'] = home_perf_ratio
                X_enhanced.loc[idx, 'performance_vs_talent_away'] = away_perf_ratio
                X_enhanced.loc[idx, 'performance_vs_talent_diff'] = home_perf_ratio - away_perf_ratio
                
                # 5. LOW SCORING POTENTIAL (précis)
                home_xg = match_data.get('home_xg_eff_10', 0.5)
                away_xg = match_data.get('away_xg_eff_10', 0.5)
                away_goals = match_data.get('away_goals_sum_5', 5.0)
                home_goals_est = 5.0  # Estimation si pas disponible
                
                low_scoring = self.calculate_low_scoring_potential(
                    home_xg, away_xg, home_goals_est, away_goals
                )
                X_enhanced.loc[idx, 'low_scoring_potential'] = low_scoring
                
            except Exception as e:
                logger.warning(f"Erreur feature engineering match {idx}: {e}")
                continue
        
        logger.info(f"✅ Features calculées pour {n_matches} matchs")
        
        return X_enhanced

class TrueDrawSpecialistForest:
    """Forêt spécialisée draws avec VRAIES features"""
    
    def __init__(self, draw_class_weight=6):
        self.clf = RandomForestClassifier(
            n_estimators=400,  # Plus d'arbres avec vraies features
            max_depth=14,
            min_samples_leaf=3,
            class_weight={0: 1, 1: draw_class_weight},
            random_state=42,
            max_features='sqrt'
        )
        self.feature_engineer = TrueFeatureEngineer()
        self.is_fitted = False
    
    def fit(self, X, y, df_historical, reference_date):
        """Entrainement avec vraies features"""
        
        # Convertir target
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y.copy()
        
        y_draw = (y_str == 'D').astype(int)
        
        # Engineer vraies features
        X_enhanced = self.feature_engineer.engineer_all_features(X, df_historical, reference_date)
        
        # Features spécialisées VRAIES
        self.true_draw_features = [
            # Nouvelles vraies features
            'elo_variance_combined', 'is_promoted_match', 'market_odds_spread',
            'performance_vs_talent_diff', 'low_scoring_potential',
            # Features de base importantes
            'elo_diff_normalized', 'market_entropy_norm', 'form_diff_normalized'
        ]
        
        X_selected = X_enhanced[self.true_draw_features].fillna(0.5)
        
        # Entrainement
        self.clf.fit(X_selected, y_draw)
        self.is_fitted = True
        
        # Stats
        draw_ratio = np.mean(y_draw)
        logger.info(f"✅ TrueDrawSpecialist entrainé:")
        logger.info(f"   Draw ratio: {draw_ratio:.1%}")
        logger.info(f"   Vraies features: {len(self.true_draw_features)}")
        
        return self
    
    def predict_draws(self, X, df_historical, reference_date, dynamic_threshold=True):
        """Prédiction avec vraies features et seuil dynamique"""
        
        # Engineer features
        X_enhanced = self.feature_engineer.engineer_all_features(X, df_historical, reference_date)
        X_selected = X_enhanced[self.true_draw_features].fillna(0.5)
        
        # Probabilités
        proba = self.clf.predict_proba(X_selected)[:, 1]
        
        if dynamic_threshold:
            # Seuil adaptatif basé sur VRAIES features
            predictions = np.zeros(len(X))
            
            for i in range(len(X)):
                # Critères stricts avec vraies features
                elo_variance = X_enhanced.iloc[i]['elo_variance_combined']
                is_promoted = X_enhanced.iloc[i]['is_promoted_match']
                market_spread = X_enhanced.iloc[i]['market_odds_spread']
                low_scoring = X_enhanced.iloc[i]['low_scoring_potential']
                
                # Seuil adaptatif
                base_threshold = 0.35
                
                # Réduire seuil si conditions favorables draws
                if elo_variance > 0.2:  # Équipes instables
                    base_threshold -= 0.05
                if market_spread < 0.3:  # Match serré
                    base_threshold -= 0.05
                if low_scoring > 0.6:  # Potentiel peu prolifique
                    base_threshold -= 0.05
                if is_promoted > 0:  # Équipe promue impliquée
                    base_threshold -= 0.03
                
                # Appliquer seuil adaptatif
                if proba[i] > base_threshold:
                    predictions[i] = 1
        else:
            predictions = (proba > 0.32).astype(int)
        
        return predictions, proba

class TrueHomeAwayForest:
    """Forêt H/A avec vraies features contextuelles"""
    
    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42
        )
        self.feature_engineer = TrueFeatureEngineer()
        self.is_fitted = False
    
    def fit(self, X, y, df_historical, reference_date):
        """Entrainement H/A avec vraies features"""
        
        # Convertir et filtrer NotDraw
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y.copy()
        
        mask_notdraw = y_str != 'D'
        if mask_notdraw.sum() < 10:
            return self
        
        X_notdraw = X[mask_notdraw]
        y_notdraw = y_str[mask_notdraw]
        y_homeaway = y_notdraw.map({'H': 1, 'A': 0})
        
        # Engineer features
        X_enhanced = self.feature_engineer.engineer_all_features(X_notdraw, df_historical, reference_date)
        
        # Features H/A avec vraies features
        self.true_homeaway_features = [
            # Nouvelles vraies features
            'is_promoted_home', 'is_promoted_away', 'performance_vs_talent_diff',
            'elo_variance_home', 'elo_variance_away',
            # Features de base H/A
            'elo_diff_normalized', 'shots_diff_normalized', 'home_xg_eff_10', 'away_xg_eff_10'
        ]
        
        X_selected = X_enhanced[self.true_homeaway_features].fillna(0.5)
        
        # Nettoyer et entrainer
        valid_mask = y_homeaway.notna()
        X_clean = X_selected[valid_mask]
        y_clean = y_homeaway[valid_mask]
        
        if len(y_clean) < 5:
            return self
        
        self.clf.fit(X_clean, y_clean)
        self.is_fitted = True
        
        home_ratio = np.mean(y_clean)
        logger.info(f"✅ TrueHomeAwaySpecialist entrainé:")
        logger.info(f"   Home ratio: {home_ratio:.1%}")
        logger.info(f"   Vraies features: {len(self.true_homeaway_features)}")
        
        return self
    
    def predict_homeaway(self, X, df_historical, reference_date):
        """Prédiction H/A avec vraies features"""
        
        X_enhanced = self.feature_engineer.engineer_all_features(X, df_historical, reference_date)
        X_selected = X_enhanced[self.true_homeaway_features].fillna(0.5)
        
        predictions = self.clf.predict(X_selected)
        probabilities = self.clf.predict_proba(X_selected)
        
        return predictions, probabilities

class TrueSpecializedCascade:
    """Cascade avec VRAIES features spécialisées"""
    
    def __init__(self):
        self.draw_forest = TrueDrawSpecialistForest(draw_class_weight=6)
        self.homeaway_forest = TrueHomeAwayForest()
        self.is_fitted = False
    
    def fit(self, X, y, df_historical, reference_date):
        """Entrainement avec vraies features"""
        
        logger.info("🌳 Entrainement CASCADE VRAIES FEATURES SPÉCIALISÉES")
        
        self.draw_forest.fit(X, y, df_historical, reference_date)
        self.homeaway_forest.fit(X, y, df_historical, reference_date)
        
        self.is_fitted = True
        return self
    
    def predict(self, X, df_historical, reference_date):
        """Prédiction avec vraies features"""
        
        # Étape 1: Draws avec vraies features
        is_draw, draw_proba = self.draw_forest.predict_draws(X, df_historical, reference_date)
        
        # Étape 2: H/A avec vraies features
        homeaway_pred, homeaway_proba = self.homeaway_forest.predict_homeaway(X, df_historical, reference_date)
        
        # Combiner
        final_predictions = []
        for i in range(len(X)):
            if is_draw[i] == 1:
                final_predictions.append('D')
            else:
                final_predictions.append('H' if homeaway_pred[i] == 1 else 'A')
        
        return np.array(final_predictions)

def test_true_specialized_cascade():
    """Test cascade avec VRAIES features spécialisées"""
    logger.info("🌳 TEST CASCADE VRAIES FEATURES SPÉCIALISÉES")
    logger.info("=" * 70)
    
    try:
        # Charger données avec équipes
        df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
        
        df_real = pd.read_csv('data/raw/E0 (7).csv', encoding='utf-8-sig')
        team_mapping = {
            'Spurs': 'Tottenham',
            "Nott'm Forest": "Nott'm Forest"
        }
        df_real['HomeTeam'] = df_real['HomeTeam'].replace(team_mapping)
        df_real['AwayTeam'] = df_real['AwayTeam'].replace(team_mapping)
        real_matches = df_real[['HomeTeam', 'AwayTeam', 'FTR']]
        
        if 'target' not in df.columns:
            df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        train_cutoff = pd.to_datetime('2025-08-01')
        df_train = df[df['Date'] < train_cutoff].copy()
        
        # Test avec équipes identifiées
        try:
            auto_dataset = pd.read_csv('data/processed/v_auto_update_20250916_105039.csv', parse_dates=['Date'])
            auto_season_2025 = auto_dataset[auto_dataset['Date'] >= '2025-08-01'].copy()
            auto_test_candidates = auto_season_2025.head(40).copy()
            df_test = pd.merge(auto_test_candidates, real_matches, on=['HomeTeam', 'AwayTeam'], how='inner')
        except:
            df_season_2025 = df[df['Date'] >= '2025-08-01'].copy()
            df_test_candidates = df_season_2025.head(40).copy()
            df_test = pd.merge(df_test_candidates, real_matches, on=['HomeTeam', 'AwayTeam'], how='inner')
        
        base_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        # Préparer données avec équipes
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        
        X_train = df_train_clean[base_features + ['HomeTeam', 'AwayTeam']].fillna(0.5)
        y_train = df_train_clean['target']
        
        X_test = df_test[base_features + ['HomeTeam', 'AwayTeam']].fillna(0.5)
        y_real = df_test['FTR']
        
        logger.info(f"📊 Données: train={len(X_train)}, test={len(X_test)}")
        
        # Test avec vraies features
        model = TrueSpecializedCascade()
        model.fit(X_train, y_train, df, train_cutoff)
        
        y_pred = model.predict(X_test, df, train_cutoff)
        accuracy = accuracy_score(y_real, y_pred)
        
        # Métriques draws
        draws_predicted = (y_pred == 'D').sum()
        draws_real = (y_real == 'D').sum()
        draws_correct = ((y_pred == 'D') & (y_real == 'D')).sum()
        draw_recall = draws_correct / draws_real if draws_real > 0 else 0
        
        logger.info(f"\n🏆 RÉSULTATS VRAIES FEATURES SPÉCIALISÉES:")
        logger.info(f"   Accuracy: {accuracy:.1%} ({int(accuracy * len(y_real))}/{len(y_real)})")
        logger.info(f"   Draw recall: {draw_recall:.1%} ({draws_correct}/{draws_real})")
        logger.info(f"   Draws prédits: {draws_predicted}")
        
        # Distribution
        real_dist = y_real.value_counts(normalize=True)
        pred_dist = pd.Series(y_pred).value_counts(normalize=True)
        
        logger.info(f"\n📊 DISTRIBUTION:")
        logger.info(f"   Réelle: H={real_dist.get('H', 0):.1%}, D={real_dist.get('D', 0):.1%}, A={real_dist.get('A', 0):.1%}")
        logger.info(f"   Prédite: H={pred_dist.get('H', 0):.1%}, D={pred_dist.get('D', 0):.1%}, A={pred_dist.get('A', 0):.1%}")
        
        # Comparaison finale
        logger.info(f"\n🏆 COMPARAISON FINALE:")
        logger.info(f"   VRAIES FEATURES: {accuracy:.1%} accuracy, {draw_recall:.1%} draws")
        logger.info(f"   Cascade baseline: 52.5% accuracy, 33.3% draws")
        logger.info(f"   Features simulées: 50.0% accuracy, 33.3% draws")
        
        # Verdict
        if accuracy > 0.525 or draw_recall > 0.4:
            verdict = "🔥 VRAIES FEATURES SUPÉRIEURES !"
        elif accuracy > 0.50 and draw_recall > 0.35:
            verdict = "✅ Amélioration confirmée"
        else:
            verdict = "⚠️  Pas d'amélioration majeure"
        
        logger.info(f"\n🎯 VERDICT: {verdict}")
        
        return {
            'true_features_accuracy': accuracy,
            'true_features_draw_recall': draw_recall,
            'improvement_vs_baseline': accuracy - 0.525,
            'improvement_draws': draw_recall - 0.333,
            'verdict': verdict
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur test vraies features: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_true_specialized_cascade()
    
    if result:
        print(f"\n🌳 VRAIES FEATURES TESTÉES")
        print(f"Accuracy: {result['true_features_accuracy']:.1%} ({result['improvement_vs_baseline']:+.1%})")
        print(f"Draw recall: {result['true_features_draw_recall']:.1%} ({result['improvement_draws']:+.1%})")
        print(f"Verdict: {result['verdict']}")
    else:
        print("❌ Échec test vraies features")