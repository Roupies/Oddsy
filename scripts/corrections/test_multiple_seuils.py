#!/usr/bin/env python3
"""
🎯 TEST MULTIPLES SEUILS CASCADE
==============================
Test différents seuils pour trouver l'équilibre optimal accuracy/draws.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi_seuils")

def test_multiple_seuils():
    try:
        logger.info("🎯 TEST MULTIPLES SEUILS CASCADE")
        logger.info("=" * 40)
        
        # Chargement données
        auto_data = pd.read_csv("data/processed/v_auto_update_20250916_110247.csv")
        auto_data['Date'] = pd.to_datetime(auto_data['Date'])
        
        # Splits
        train_data = auto_data[auto_data['Date'] < '2025-08-01'].copy()
        test_data = auto_data[auto_data['Date'] >= '2025-08-01'].head(40).copy()
        
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
        test_data = create_target_from_result(test_data)
        
        # Features
        feature_cols = [
            'elo_diff_normalized', 'market_entropy_norm', 'shots_diff_normalized',
            'corners_diff_normalized', 'form_diff_normalized', 'h2h_score',
            'matchday_normalized', 'home_xg_eff_10', 'away_xg_eff_10', 'away_goals_sum_5'
        ]
        
        X_train = train_data[feature_cols].fillna(0)
        X_test = test_data[feature_cols].fillna(0)
        y_train = train_data['target']
        y_test = test_data['target']
        
        # Filtrage NaN
        train_valid = y_train.notna()
        test_valid = y_test.notna()
        
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]
        test_data = test_data[test_valid]
        
        y_train_str = y_train.map({0: 'H', 1: 'D', 2: 'A'})
        y_test_str = y_test.map({0: 'H', 1: 'D', 2: 'A'})
        
        logger.info(f"📊 Données: Train {len(X_train)}, Test {len(X_test)}")
        logger.info(f"📊 Vrais draws dans test: {(y_test_str == 'D').sum()}/40")
        
        # 1. MODÈLE BASELINE POUR RÉFÉRENCE
        baseline_model = RandomForestClassifier(n_estimators=150, random_state=42)
        baseline_model.fit(X_train, y_train_str)
        baseline_preds = baseline_model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test_str, baseline_preds) * 100
        baseline_draws = (baseline_preds == 'D').sum()
        
        logger.info(f"📊 Baseline référence: {baseline_accuracy:.1f}% accuracy, {baseline_draws} draws")
        
        # 2. ENTRAÎNEMENT MODÈLES POUR CASCADE
        logger.info(f"\n🔧 ENTRAÎNEMENT MODÈLES CASCADE")
        
        # Draw model
        y_draw_train = (y_train_str == 'D').astype(int)
        draw_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight={0: 1, 1: 3.0}, random_state=42
        )
        draw_model.fit(X_train, y_draw_train)
        
        # Home/Away model
        mask_notdraw_train = y_train_str != 'D'
        X_homeaway = X_train[mask_notdraw_train]
        y_homeaway = y_train_str[mask_notdraw_train].map({'H': 1, 'A': 0})
        
        homeaway_model = RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )
        homeaway_model.fit(X_homeaway, y_homeaway)
        
        # Probabilités draw pour tous les tests
        draw_proba = draw_model.predict_proba(X_test)[:, 1]
        logger.info(f"   Proba Draw range: {draw_proba.min():.3f} - {draw_proba.max():.3f}")
        
        # 3. TEST DIFFÉRENTS SEUILS
        seuils_a_tester = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40]
        resultats = []
        
        logger.info(f"\n🧪 TEST SEUILS CASCADE")
        logger.info(f"=" * 50)
        
        for seuil in seuils_a_tester:
            # Prédiction cascade avec ce seuil
            is_draw = draw_proba >= seuil
            cascade_preds = np.full(len(X_test), 'D', dtype=object)
            
            # Pour les non-draws
            mask_notdraw = ~is_draw
            if mask_notdraw.sum() > 0:
                homeaway_pred = homeaway_model.predict(X_test[mask_notdraw])
                cascade_preds[mask_notdraw] = np.where(homeaway_pred == 1, 'H', 'A')
            
            # Métriques
            cascade_accuracy = accuracy_score(y_test_str, cascade_preds) * 100
            cascade_draws = (cascade_preds == 'D').sum()
            
            # Distribution
            cascade_dist = pd.Series(cascade_preds).value_counts(normalize=True).sort_index() * 100
            
            # Performance par classe
            perf_draws = None
            vrais_draws = (y_test_str == 'D').sum()
            if vrais_draws > 0:
                draws_correct = ((cascade_preds == 'D') & (y_test_str == 'D')).sum()
                perf_draws = f"{draws_correct}/{vrais_draws}"
            
            # Stockage résultat
            resultat = {
                'seuil': seuil,
                'accuracy': cascade_accuracy,
                'draws_detected': cascade_draws,
                'accuracy_vs_baseline': cascade_accuracy - baseline_accuracy,
                'dist_H': cascade_dist.get('H', 0),
                'dist_D': cascade_dist.get('D', 0),
                'dist_A': cascade_dist.get('A', 0),
                'perf_draws': perf_draws
            }
            resultats.append(resultat)
            
            # Log résultat
            logger.info(f"Seuil {seuil:.2f}: {cascade_accuracy:.1f}% acc ({cascade_accuracy - baseline_accuracy:+.1f}pp), {cascade_draws} draws, H/D/A={cascade_dist.get('H', 0):.0f}/{cascade_dist.get('D', 0):.0f}/{cascade_dist.get('A', 0):.0f}%")
        
        # 4. ANALYSE COMPARATIVE
        logger.info(f"\n🏆 ANALYSE COMPARATIVE")
        logger.info(f"=" * 40)
        
        # Meilleur accuracy
        best_accuracy = max(resultats, key=lambda x: x['accuracy'])
        logger.info(f"🎯 Meilleur accuracy: Seuil {best_accuracy['seuil']:.2f} = {best_accuracy['accuracy']:.1f}%")
        
        # Meilleur équilibre (accuracy > baseline ET draws > 0)
        balanced_results = [r for r in resultats if r['accuracy'] >= baseline_accuracy and r['draws_detected'] > 0]
        if balanced_results:
            best_balanced = max(balanced_results, key=lambda x: x['accuracy'])
            logger.info(f"🎯 Meilleur équilibre: Seuil {best_balanced['seuil']:.2f} = {best_balanced['accuracy']:.1f}% avec {best_balanced['draws_detected']} draws")
        else:
            logger.info(f"⚠️  Aucun seuil n'améliore le baseline tout en détectant des draws")
        
        # Recommandation basée sur draws (3-8 draws idéal)
        ideal_draws = [r for r in resultats if 3 <= r['draws_detected'] <= 8]
        if ideal_draws:
            best_ideal = max(ideal_draws, key=lambda x: x['accuracy'])
            logger.info(f"🎯 Meilleur 3-8 draws: Seuil {best_ideal['seuil']:.2f} = {best_ideal['accuracy']:.1f}% avec {best_ideal['draws_detected']} draws")
        
        # 5. DÉTAIL MEILLEUR SEUIL
        if balanced_results:
            meilleur = best_balanced
        elif ideal_draws:
            meilleur = best_ideal
        else:
            meilleur = best_accuracy
        
        logger.info(f"\n🔍 DÉTAIL SEUIL RECOMMANDÉ {meilleur['seuil']:.2f}:")
        logger.info(f"   Accuracy: {meilleur['accuracy']:.1f}% ({meilleur['accuracy_vs_baseline']:+.1f}pp vs baseline)")
        logger.info(f"   Draws détectés: {meilleur['draws_detected']}/40")
        logger.info(f"   Distribution: H={meilleur['dist_H']:.0f}%, D={meilleur['dist_D']:.0f}%, A={meilleur['dist_A']:.0f}%")
        logger.info(f"   Performance draws: {meilleur['perf_draws']}")
        
        # Tableau de résumé
        print(f"\n📊 TABLEAU RÉSUMÉ:")
        print(f"{'Seuil':<6} {'Accuracy':<8} {'Δ vs Base':<9} {'Draws':<6} {'Distribution H/D/A':<16}")
        print(f"{'='*6} {'='*8} {'='*9} {'='*6} {'='*16}")
        for r in resultats:
            print(f"{r['seuil']:<6.2f} {r['accuracy']:<8.1f} {r['accuracy_vs_baseline']:<+9.1f} {r['draws_detected']:<6} {r['dist_H']:.0f}/{r['dist_D']:.0f}/{r['dist_A']:.0f}")
        
        print(f"\n🎯 RECOMMANDATION: Seuil {meilleur['seuil']:.2f}")
        print(f"   {meilleur['accuracy']:.1f}% accuracy ({meilleur['accuracy_vs_baseline']:+.1f}pp), {meilleur['draws_detected']} draws détectés")
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'resultats': resultats,
            'recommandation': meilleur
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_multiple_seuils()