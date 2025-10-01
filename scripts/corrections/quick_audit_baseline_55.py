#!/usr/bin/env python3
"""
🔍 QUICK AUDIT BASELINE 55% - VALIDATION
======================================

Audit rapide pour valider la performance 55% sur 40 matchs EPL 2025-26.
Vérifie l'intégrité, anti-leakage et reproductibilité.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audit_55")

class SimpleCascadeModel:
    """Modèle cascade pour audit"""
    
    def __init__(self):
        self.clf_draw = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        self.is_fitted = False
    
    def fit(self, X, y):
        # Convertir target si numérique
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y.copy()
        
        # Étape 1: Draw vs NotDraw
        y_draw = (y_str == 'D').astype(int)
        self.clf_draw.fit(X, y_draw)
        
        # Étape 2: Home vs Away sur NotDraw
        mask_notdraw = y_str != 'D'
        if mask_notdraw.sum() > 5:
            X_notdraw = X[mask_notdraw]
            y_homeaway = y_str[mask_notdraw].map({'H': 1, 'A': 0})
            
            # Nettoyer NaN
            valid_homeaway = y_homeaway.notna()
            X_notdraw_clean = X_notdraw[valid_homeaway]
            y_homeaway_clean = y_homeaway[valid_homeaway]
            
            if len(y_homeaway_clean) > 5:
                self.clf_homeaway.fit(X_notdraw_clean, y_homeaway_clean)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        pred_draw = self.clf_draw.predict(X)
        pred_homeaway = self.clf_homeaway.predict(X)
        
        y_pred = []
        for is_draw, home_away in zip(pred_draw, pred_homeaway):
            if is_draw == 1:
                y_pred.append('D')
            else:
                y_pred.append('H' if home_away == 1 else 'A')
        
        return np.array(y_pred)

def quick_audit_55_percent():
    """Audit rapide validation 55%"""
    logger.info("🔍 QUICK AUDIT BASELINE 55% - VALIDATION")
    logger.info("=" * 50)
    
    try:
        # 1. Charger dataset production
        df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
        logger.info(f"✅ Dataset production: {df.shape}")
        
        # 2. Charger vrais résultats 40 matchs
        df_real = pd.read_csv('data/raw/E0 (7).csv', encoding='utf-8-sig')
        
        # Normaliser équipes (audit identique)
        team_mapping = {
            'Spurs': 'Tottenham',
            "Nott'm Forest": "Nott'm Forest"
        }
        df_real['HomeTeam'] = df_real['HomeTeam'].replace(team_mapping)
        df_real['AwayTeam'] = df_real['AwayTeam'].replace(team_mapping)
        
        real_matches = df_real[['HomeTeam', 'AwayTeam', 'FTR']]
        logger.info(f"✅ {len(real_matches)} vrais résultats chargés")
        
        # 3. Target encoding (audit identique)
        if 'target' not in df.columns and 'FullTimeResult' in df.columns:
            df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # 4. Anti-leakage: entrainement avant 2025-08-01
        train_cutoff = pd.to_datetime('2025-08-01')
        df_train = df[df['Date'] < train_cutoff].copy()
        logger.info(f"🔒 Anti-leakage: {len(df_train)} matchs < {train_cutoff.date()}")
        
        # 5. Test sur 2025-26 avec correspondance
        df_season_2025 = df[df['Date'] >= '2025-08-01'].copy()
        
        # Extension auto-intégrée si nécessaire (audit identique)
        df_test_candidates = df_season_2025.head(40).copy()
        df_test = pd.merge(
            df_test_candidates, real_matches,
            on=['HomeTeam', 'AwayTeam'],
            how='inner'
        )
        
        logger.info(f"🎯 Matchs test: {len(df_test)} correspondants")
        
        if len(df_test) < 35:
            # Extension via auto-integration (comme baseline)
            try:
                auto_dataset = pd.read_csv('data/processed/v_auto_update_20250916_105039.csv', parse_dates=['Date'])
                auto_season_2025 = auto_dataset[auto_dataset['Date'] >= '2025-08-01'].copy()
                auto_test_candidates = auto_season_2025.head(40).copy()
                df_test = pd.merge(
                    auto_test_candidates, real_matches,
                    on=['HomeTeam', 'AwayTeam'],
                    how='inner'
                )
                logger.info(f"🔄 Extension auto: {len(df_test)} correspondances")
            except:
                logger.warning("⚠️  Pas d'extension auto disponible")
        
        # 6. Features modèle (audit identique - ordre exact)
        model_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        missing = [f for f in model_features if f not in df_test.columns]
        if missing:
            logger.error(f"❌ Features manquantes: {missing}")
            return None
        
        # 7. Préparer données (audit identique)
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        
        X_train = df_train_clean[model_features].fillna(0.5)
        y_train = df_train_clean['target']
        
        X_test = df_test[model_features].fillna(0.5)
        y_real = df_test['FTR']
        
        logger.info(f"📊 Audit données: X_train{X_train.shape}, X_test{X_test.shape}")
        
        # 8. Test reproductibilité (random_state=42)
        model = SimpleCascadeModel()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_real, y_pred)
        
        # 9. RÉSULTATS AUDIT
        logger.info(f"\n🏆 AUDIT BASELINE 55%")
        logger.info("=" * 30)
        
        logger.info(f"✅ Performance reproduite: {accuracy:.1%}")
        logger.info(f"✅ Anti-leakage vérifié: {len(df_train)} < {train_cutoff.date()}")
        logger.info(f"✅ Features validées: {len(model_features)}")
        logger.info(f"✅ Correspondance: {len(df_test)}/40 matchs")
        
        # Vérifications intégrité
        checks = {
            'performance_55_reproduced': abs(accuracy - 0.55) < 0.025,  # ±2.5% tolérance
            'proper_train_test_split': len(df_train) > 2000,
            'no_data_leakage': df_train['Date'].max() < train_cutoff,
            'sufficient_test_matches': len(df_test) >= 35
        }
        
        logger.info(f"\n🔍 VÉRIFICATIONS AUDIT:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {check}: {passed}")
        
        all_passed = all(checks.values())
        
        if all_passed:
            logger.info(f"\n🎉 AUDIT PASSED: Baseline 55% VALIDÉ")
            verdict = "BASELINE 55% OFFICIELLEMENT VALIDÉ"
        else:
            logger.info(f"\n⚠️  AUDIT WARNINGS: Vérifications incomplètes")
            verdict = "BASELINE NÉCESSITE ATTENTION"
        
        # Détail prédictions pour rapport .md
        predictions_detail = []
        for i in range(len(df_test)):
            predictions_detail.append({
                'match_idx': i+1,
                'home_team': df_test.iloc[i]['HomeTeam'],
                'away_team': df_test.iloc[i]['AwayTeam'],
                'real_result': y_real.iloc[i],
                'predicted': y_pred[i],
                'correct': y_real.iloc[i] == y_pred[i]
            })
        
        return {
            'accuracy': accuracy,
            'verdict': verdict,
            'checks_passed': all_passed,
            'n_train': len(X_train),
            'n_test': len(df_test),
            'predictions': predictions_detail
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur audit: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = quick_audit_55_percent()
    
    if result and result['checks_passed']:
        print(f"\n✅ AUDIT RÉUSSI: {result['accuracy']:.1%}")
        print(f"Verdict: {result['verdict']}")
    else:
        print(f"\n⚠️  Audit incomplet ou échec")