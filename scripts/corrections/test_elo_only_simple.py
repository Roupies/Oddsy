#!/usr/bin/env python3
"""
🎯 TEST ELO SEULEMENT - EXPÉRIENCE SIMPLE
======================================

Test rapide avec seulement Elo comme feature pour voir l'impact.
Modèle simple sur les 40 matchs.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("elo_only_test")

def test_elo_only():
    """Test simple avec Elo seulement"""
    logger.info("🎯 TEST ELO SEULEMENT - 40 MATCHS")
    logger.info("=" * 40)
    
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
        
        # SEULEMENT ELO comme feature
        elo_feature = ['elo_diff_normalized']
        
        # Préparer données
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        
        X_train = df_train_clean[elo_feature].fillna(0.5)
        y_train = df_train_clean['target']
        
        X_test = df_test[elo_feature].fillna(0.5)
        y_real = df_test['FTR']
        
        logger.info(f"📊 Données: train={len(X_train)}, test={len(X_test)}")
        logger.info(f"🎯 Feature unique: {elo_feature[0]}")
        
        # Modèle simple Random Forest
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight="balanced"
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_real, y_pred)
        
        # Résultats
        logger.info(f"\n🏆 RÉSULTATS ELO SEULEMENT")
        logger.info(f"   Accuracy: {accuracy:.1%} ({int(accuracy * len(y_real))}/{len(y_real)})")
        
        # Distribution
        real_dist = y_real.value_counts(normalize=True)
        pred_dist = pd.Series(y_pred).value_counts(normalize=True)
        
        logger.info(f"   Réel: H={real_dist.get('H', 0):.1%}, D={real_dist.get('D', 0):.1%}, A={real_dist.get('A', 0):.1%}")
        logger.info(f"   Prédit: H={pred_dist.get('H', 0):.1%}, D={pred_dist.get('D', 0):.1%}, A={pred_dist.get('A', 0):.1%}")
        
        # Valeurs Elo test
        elo_values = X_test['elo_diff_normalized'].values
        logger.info(f"\n📊 VALEURS ELO TEST:")
        logger.info(f"   Moyenne: {np.mean(elo_values):.3f}")
        logger.info(f"   Min-Max: {np.min(elo_values):.3f} - {np.max(elo_values):.3f}")
        logger.info(f"   Écart-type: {np.std(elo_values):.3f}")
        
        # Quelques exemples
        logger.info(f"\n🔍 EXEMPLES MATCHS:")
        for i in range(min(5, len(df_test))):
            home = df_test.iloc[i]['HomeTeam']
            away = df_test.iloc[i]['AwayTeam']
            elo_val = elo_values[i]
            real = y_real.iloc[i]
            pred = y_pred[i]
            correct = "✅" if real == pred else "❌"
            
            logger.info(f"   {correct} {home} vs {away}: Elo={elo_val:.3f}, réel={real}, prédit={pred}")
        
        # Comparaison avec baseline
        logger.info(f"\n📈 COMPARAISON:")
        logger.info(f"   Elo seul: {accuracy:.1%}")
        logger.info(f"   Cascade 10 features: 52.5%")
        logger.info(f"   Différence: {accuracy - 0.525:+.1%}")
        
        return {
            'accuracy': accuracy,
            'elo_performance': accuracy,
            'baseline_comparison': accuracy - 0.525,
            'n_test': len(y_real)
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_elo_only()
    
    if result:
        print(f"\n🎯 ELO SEULEMENT: {result['elo_performance']:.1%}")
        print(f"vs Cascade: {result['baseline_comparison']:+.1%}")
    else:
        print("❌ Échec test")