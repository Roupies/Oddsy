#!/usr/bin/env python3
"""
🎯 TEST CASCADE AVEC ELO RÉALISTE - EXPÉRIENCE
============================================

Test cascade équilibré avec les 10 features MAIS Elo basé sur 
vraies données fin 2024-25 au lieu de 0.5 neutre.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("real_elo_cascade")

class BalancedCascadeModel:
    """Modèle cascade équilibré"""
    
    def __init__(self, draw_weight=3, draw_threshold=0.35, calibration_factor=0.85):
        self.clf_draw = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=4,
            class_weight={0: 1, 1: draw_weight},
            random_state=42
        )
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            class_weight="balanced"
        )
        self.draw_threshold = draw_threshold
        self.calibration_factor = calibration_factor
        self.is_fitted = False
    
    def fit(self, X, y):
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y.copy()
        
        y_draw = (y_str == 'D').astype(int)
        self.clf_draw.fit(X, y_draw)
        
        mask_notdraw = y_str != 'D'
        if mask_notdraw.sum() > 5:
            X_notdraw = X[mask_notdraw]
            y_homeaway = y_str[mask_notdraw].map({'H': 1, 'A': 0})
            valid_homeaway = y_homeaway.notna()
            X_notdraw_clean = X_notdraw[valid_homeaway]
            y_homeaway_clean = y_homeaway[valid_homeaway]
            
            if len(y_homeaway_clean) > 5:
                self.clf_homeaway.fit(X_notdraw_clean, y_homeaway_clean)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        proba_draw = self.clf_draw.predict_proba(X)[:, 1]
        
        calibrated_threshold = self.draw_threshold + (1 - self.calibration_factor) * 0.1
        pred_draw = (proba_draw > calibrated_threshold).astype(int)
        
        # Limitation par percentile
        target_draw_ratio = 0.25
        n_draws_target = int(len(X) * target_draw_ratio)
        
        if pred_draw.sum() > n_draws_target:
            top_draw_indices = np.argsort(proba_draw)[-n_draws_target:]
            pred_draw_filtered = np.zeros_like(pred_draw)
            pred_draw_filtered[top_draw_indices] = 1
            pred_draw = pred_draw_filtered
        
        pred_homeaway = self.clf_homeaway.predict(X)
        
        y_pred = []
        for is_draw, home_away in zip(pred_draw, pred_homeaway):
            if is_draw == 1:
                y_pred.append('D')
            else:
                y_pred.append('H' if home_away == 1 else 'A')
        
        return np.array(y_pred)

def get_real_elo_from_historical(team, df_baseline, cutoff_date):
    """Récupère le vrai Elo d'une équipe fin 2024-25"""
    
    # Équipes promues - Elo estimé Championship
    promoted_elo = {
        'Leeds': 0.58,        # Fort en Championship
        'Sunderland': 0.42,   # Moyen Championship  
        'Burnley': 0.52       # Correct Championship
    }
    
    if team in promoted_elo:
        return promoted_elo[team]
    
    # Équipes historiques - dernier Elo connu 2024-25
    team_matches = df_baseline[
        ((df_baseline['HomeTeam'] == team) | (df_baseline['AwayTeam'] == team)) &
        (df_baseline['Date'] <= cutoff_date)
    ]
    
    if len(team_matches) > 0:
        last_match = team_matches.tail(1).iloc[0]
        
        # Si équipe à domicile, prendre elo_diff_normalized directement
        # Si équipe extérieur, inverser
        if last_match['HomeTeam'] == team:
            # Domicile: elo_diff = elo_home - elo_away, on veut elo_home
            base_elo = last_match['elo_diff_normalized']
            return min(max(base_elo, 0.1), 0.9)  # Clamper
        else:
            # Extérieur: elo_diff = elo_home - elo_away, on veut elo_away
            base_elo = 1 - last_match['elo_diff_normalized']  # Inverser
            return min(max(base_elo, 0.1), 0.9)  # Clamper
    
    # Défaut si pas trouvé
    return 0.5

def test_real_elo_cascade():
    """Test cascade avec Elo réaliste"""
    logger.info("🎯 TEST CASCADE AVEC ELO RÉALISTE")
    logger.info("=" * 50)
    
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
        historical_cutoff = pd.to_datetime('2025-05-31')  # Fin saison 2024-25
        
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
        
        logger.info(f"📊 Données: train={len(df_train)}, test={len(df_test)}")
        
        # === CORRECTION ELO RÉALISTE ===
        logger.info("🔧 Correction Elo avec données historiques 2024-25...")
        
        elo_corrections = {}
        for i, row in df_test.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Récupérer vrais Elos fin 2024-25
            home_elo = get_real_elo_from_historical(home_team, df, historical_cutoff)
            away_elo = get_real_elo_from_historical(away_team, df, historical_cutoff)
            
            # Calculer différence Elo réaliste
            elo_diff_real = home_elo - away_elo + 0.1  # +0.1 avantage domicile
            elo_diff_normalized = min(max(elo_diff_real, 0), 1)  # Normaliser 0-1
            
            elo_corrections[i] = {
                'home_elo': home_elo,
                'away_elo': away_elo, 
                'elo_diff_real': elo_diff_normalized
            }
            
            # Corriger dans df_test
            df_test.at[i, 'elo_diff_normalized'] = elo_diff_normalized
        
        logger.info(f"✅ Elo corrigé pour {len(elo_corrections)} matchs")
        
        # Exemple corrections
        logger.info("🔍 EXEMPLES CORRECTIONS ELO:")
        for i, (idx, correction) in enumerate(list(elo_corrections.items())[:5]):
            home = df_test.iloc[idx]['HomeTeam']
            away = df_test.iloc[idx]['AwayTeam']
            elo_diff = correction['elo_diff_real']
            logger.info(f"   {home} vs {away}: Elo diff = {elo_diff:.3f}")
        
        # Features modèle (ordre exact)
        model_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        # Préparer données
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        
        X_train = df_train_clean[model_features].fillna(0.5)
        y_train = df_train_clean['target']
        
        X_test = df_test[model_features].fillna(0.5)
        y_real = df_test['FTR']
        
        # Test modèle avec Elo réaliste
        logger.info("⚙️  Test cascade avec Elo réaliste...")
        
        model = BalancedCascadeModel(draw_weight=3, draw_threshold=0.35, calibration_factor=0.85)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_real, y_pred)
        
        # Résultats
        logger.info(f"\n🏆 RÉSULTATS AVEC ELO RÉALISTE")
        logger.info(f"   Accuracy: {accuracy:.1%} ({int(accuracy * len(y_real))}/{len(y_real)})")
        
        # Distribution
        real_dist = y_real.value_counts(normalize=True)
        pred_dist = pd.Series(y_pred).value_counts(normalize=True)
        
        logger.info(f"   Réel: H={real_dist.get('H', 0):.1%}, D={real_dist.get('D', 0):.1%}, A={real_dist.get('A', 0):.1%}")
        logger.info(f"   Prédit: H={pred_dist.get('H', 0):.1%}, D={pred_dist.get('D', 0):.1%}, A={pred_dist.get('A', 0):.1%}")
        
        # Draws
        draws_predicted = (y_pred == 'D').sum()
        draws_correct = ((y_pred == 'D') & (y_real == 'D')).sum()
        draws_real = (y_real == 'D').sum()
        
        logger.info(f"   Draws: {draws_correct}/{draws_real} capturés ({draws_correct/draws_real*100 if draws_real > 0 else 0:.1f}%)")
        
        # Comparaison
        logger.info(f"\n📈 COMPARAISON:")
        logger.info(f"   Avec Elo réaliste: {accuracy:.1%}")
        logger.info(f"   Cascade baseline (Elo 0.5): 52.5%")
        logger.info(f"   Amélioration: {accuracy - 0.525:+.1%}")
        
        # Stats Elo utilisées
        elo_values_used = X_test['elo_diff_normalized'].values
        logger.info(f"\n📊 ELO RÉALISTE UTILISÉ:")
        logger.info(f"   Moyenne: {np.mean(elo_values_used):.3f}")
        logger.info(f"   Écart-type: {np.std(elo_values_used):.3f}")
        logger.info(f"   Min-Max: {np.min(elo_values_used):.3f} - {np.max(elo_values_used):.3f}")
        
        return {
            'accuracy_real_elo': accuracy,
            'improvement': accuracy - 0.525,
            'draws_captured': draws_correct,
            'draws_total': draws_real,
            'elo_stats': {
                'mean': np.mean(elo_values_used),
                'std': np.std(elo_values_used),
                'min': np.min(elo_values_used),
                'max': np.max(elo_values_used)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_real_elo_cascade()
    
    if result:
        print(f"\n🎯 AVEC ELO RÉALISTE: {result['accuracy_real_elo']:.1%}")
        print(f"Amélioration: {result['improvement']:+.1%}")
        print(f"Draws: {result['draws_captured']}/{result['draws_total']}")
    else:
        print("❌ Échec test")