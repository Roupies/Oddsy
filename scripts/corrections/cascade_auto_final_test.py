#!/usr/bin/env python3
"""
🎯 TEST FINAL SYSTÈME AUTO-INTÉGRATION COMPLET
==========================================

Test définitif du système d'auto-intégration avec pipeline cascade
sur plus de matchs disponibles avec le nouveau dataset.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cascade_auto_final")

class SimpleCascadeModel:
    """Modèle cascade fonctionnel Draw vs NotDraw -> Home vs Away"""
    
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
            self.clf_homeaway.fit(X_notdraw, y_homeaway)
        
        self.is_fitted = True
        logger.info(f"✅ Cascade entrainé: {np.mean(y_draw):.1%} Draw, {mask_notdraw.sum()} NotDraw")
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

def test_auto_integration_system():
    """Test final système auto-intégration"""
    logger.info("🚀 TEST FINAL SYSTÈME AUTO-INTÉGRATION")
    logger.info("=" * 60)
    
    try:
        # 1. Charger dataset auto-intégré
        auto_dataset = pd.read_csv('data/processed/v_auto_update_20250916_105039.csv', parse_dates=['Date'])
        logger.info(f"📂 Dataset auto-intégré: {auto_dataset.shape}")
        logger.info(f"📅 Période: {auto_dataset['Date'].min()} → {auto_dataset['Date'].max()}")
        
        # 2. Charger dataset baseline pour comparaison
        baseline_dataset = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
        
        # 3. Ajouter target encoding aux deux datasets
        for df in [auto_dataset, baseline_dataset]:
            if 'target' not in df.columns and 'FullTimeResult' in df.columns:
                df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # 4. Charger vrais résultats - prendre ce qui est disponible
        try:
            real_results = pd.read_csv('data/raw/E0 (7).csv', encoding='utf-8-sig')
            real_results['HomeTeam'] = real_results['HomeTeam'].replace({
                'Spurs': 'Tottenham', 
                "Nott'm Forest": "Nott'm Forest"
            })
            real_results['AwayTeam'] = real_results['AwayTeam'].replace({
                'Spurs': 'Tottenham', 
                "Nott'm Forest": "Nott'm Forest"
            })
            logger.info(f"📊 {len(real_results)} vrais résultats chargés")
        except:
            logger.warning("⚠️  Utilisation des résultats du dataset auto-intégré")
            real_results = None
        
        # 5. Test avec matchs disponibles dans dataset auto-intégré
        cutoff = pd.to_datetime('2025-08-01')
        
        # Données entrainement
        train_auto = auto_dataset[auto_dataset['Date'] < cutoff].copy()
        train_baseline = baseline_dataset[baseline_dataset['Date'] < cutoff].copy()
        
        # Données test - tous les matchs 2025-26 disponibles
        test_auto = auto_dataset[auto_dataset['Date'] >= cutoff].copy()
        test_baseline = baseline_dataset[baseline_dataset['Date'] >= cutoff].copy()
        
        logger.info(f"📊 Comparaison datasets:")
        logger.info(f"   Auto-intégré: {len(train_auto)} train, {len(test_auto)} test")
        logger.info(f"   Baseline: {len(train_baseline)} train, {len(test_baseline)} test")
        
        # Features modèle
        features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized', 
            'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        # Test cascade sur baseline (contrôle)
        logger.info("\n🎯 TEST CONTRÔLE - BASELINE")
        X_train_base = train_baseline[features].fillna(0.5)
        y_train_base = train_baseline['target'].dropna()
        valid_mask_base = train_baseline['target'].notna()
        X_train_base = X_train_base[valid_mask_base]
        
        X_test_base = test_baseline[features].fillna(0.5)
        y_test_base = test_baseline['FullTimeResult']
        
        model_baseline = SimpleCascadeModel()
        model_baseline.fit(X_train_base, y_train_base)
        pred_baseline = model_baseline.predict(X_test_base)
        acc_baseline = accuracy_score(y_test_base, pred_baseline)
        logger.info(f"Baseline cascade: {acc_baseline:.1%} sur {len(y_test_base)} matchs")
        
        # Test cascade sur auto-intégré (principal)
        logger.info("\n🎯 TEST PRINCIPAL - AUTO-INTÉGRÉ")
        
        # Vérifier données entrainement
        logger.info(f"Train auto target stats: {train_auto['target'].value_counts().to_dict()}")
        
        X_train_auto = train_auto[features].fillna(0.5)
        y_train_auto = train_auto['target'].dropna()
        valid_mask_auto = train_auto['target'].notna()
        X_train_auto = X_train_auto[valid_mask_auto]
        
        logger.info(f"Données entrainement auto: {len(X_train_auto)} échantillons valides")
        
        if len(X_train_auto) < 100:
            logger.error("❌ Pas assez de données entrainement valides")
            return False
        
        X_test_auto = test_auto[features].fillna(0.5)
        y_test_auto = test_auto['FullTimeResult']
        
        model_auto = SimpleCascadeModel()
        model_auto.fit(X_train_auto, y_train_auto)
        pred_auto = model_auto.predict(X_test_auto)
        acc_auto = accuracy_score(y_test_auto, pred_auto)
        
        # 6. RÉSULTATS COMPARATIFS
        logger.info(f"\n🏆 RÉSULTATS FINAUX SYSTÈME AUTO-INTÉGRATION")
        logger.info("=" * 70)
        
        logger.info(f"📈 PERFORMANCES:")
        logger.info(f"   Baseline (contrôle): {acc_baseline:.1%} sur {len(y_test_base)} matchs")
        logger.info(f"   Auto-intégré: {acc_auto:.1%} sur {len(y_test_auto)} matchs")
        
        diff = acc_auto - acc_baseline
        logger.info(f"   Différence: {diff:+.1%}")
        
        # Distribution prédictions auto-intégré
        pred_dist = pd.Series(pred_auto).value_counts(normalize=True)
        real_dist = pd.Series(y_test_auto).value_counts(normalize=True)
        
        logger.info(f"\n📊 DISTRIBUTION MATCHS AUTO-INTÉGRÉS ({len(y_test_auto)} matchs):")
        logger.info(f"   Réel: H={real_dist.get('H', 0):.1%}, D={real_dist.get('D', 0):.1%}, A={real_dist.get('A', 0):.1%}")
        logger.info(f"   Prédit: H={pred_dist.get('H', 0):.1%}, D={pred_dist.get('D', 0):.1%}, A={pred_dist.get('A', 0):.1%}")
        
        # Matrice confusion
        cm = confusion_matrix(y_test_auto, pred_auto, labels=['H', 'D', 'A'])
        logger.info(f"\n📊 MATRICE CONFUSION AUTO-INTÉGRÉ:")
        logger.info(f"     Real\\Pred  H   D   A")
        for i, label in enumerate(['H', 'D', 'A']):
            logger.info(f"     {label}        {cm[i][0]:2d}  {cm[i][1]:2d}  {cm[i][2]:2d}")
        
        # 7. VALIDATION SYSTÈME
        logger.info(f"\n🔍 VALIDATION SYSTÈME AUTO-INTÉGRATION:")
        
        success_criteria = {
            'performance_acceptable': acc_auto >= 0.45,
            'plus_de_matchs_test': len(y_test_auto) > len(y_test_base),
            'pas_de_degradation_majeure': diff >= -0.05,
            'system_fonctionnel': acc_auto > 0.35
        }
        
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {criterion}: {passed}")
        
        overall_success = all(success_criteria.values())
        
        if overall_success:
            logger.info(f"\n🎉 SYSTÈME AUTO-INTÉGRATION: ✅ VALIDÉ")
            logger.info(f"🚀 Performance: {acc_auto:.1%} sur {len(y_test_auto)} matchs")
            logger.info(f"📈 {len(y_test_auto) - len(y_test_base)} matchs supplémentaires disponibles")
            verdict = "SYSTÈME PRODUCTION READY"
        else:
            logger.info(f"\n⚠️  SYSTÈME AUTO-INTÉGRATION: PARTIELLEMENT VALIDÉ")
            verdict = "SYSTÈME NÉCESSITE AJUSTEMENTS"
        
        logger.info(f"🏆 VERDICT: {verdict}")
        
        return {
            'auto_accuracy': acc_auto,
            'baseline_accuracy': acc_baseline,
            'auto_matches': len(y_test_auto),
            'baseline_matches': len(y_test_base),
            'verdict': verdict,
            'success': overall_success
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    result = test_auto_integration_system()
    
    if result and result['success']:
        print(f"\n🎉 SYSTÈME AUTO-INTÉGRATION VALIDÉ !")
        print(f"Performance: {result['auto_accuracy']:.1%}")
        print(f"Matchs disponibles: {result['auto_matches']}")
    else:
        print(f"\n⚠️  Système nécessite ajustements")

if __name__ == "__main__":
    main()