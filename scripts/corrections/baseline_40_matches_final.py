#!/usr/bin/env python3
"""
🎯 BASELINE OFFICIELLE - 40 MATCHS EPL 2025-26
==========================================

Établit la baseline officielle sur les 40 premiers matchs EPL 2025-26 J1-J4
avec modèle cascade pour toutes futures comparaisons.

RÉFÉRENCE ABSOLUE du projet Oddsy.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baseline_40_matches")

class CascadeModelBaseline:
    """Modèle cascade baseline officiel"""
    
    def __init__(self):
        self.clf_draw = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        self.is_fitted = False
        self.training_stats = {}
    
    def fit(self, X, y):
        """Entrainement cascade avec stats détaillées"""
        # Convertir target si numérique
        if hasattr(y.iloc[0], 'dtype') and np.issubdtype(y.iloc[0], np.integer):
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        elif y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y.copy()
        
        # Statistiques entrainement
        self.training_stats = {
            'total_matches': len(y_str),
            'distribution': y_str.value_counts().to_dict(),
            'draw_percentage': np.mean(y_str == 'D') * 100,
            'home_percentage': np.mean(y_str == 'H') * 100,
            'away_percentage': np.mean(y_str == 'A') * 100
        }
        
        # Étape 1: Draw vs NotDraw
        y_draw = (y_str == 'D').astype(int)
        self.clf_draw.fit(X, y_draw)
        
        # Étape 2: Home vs Away sur NotDraw
        mask_notdraw = y_str != 'D'
        self.training_stats['notdraw_samples'] = mask_notdraw.sum()
        
        if mask_notdraw.sum() > 5:
            X_notdraw = X[mask_notdraw]
            y_homeaway = y_str[mask_notdraw].map({'H': 1, 'A': 0})
            self.clf_homeaway.fit(X_notdraw, y_homeaway)
        
        self.is_fitted = True
        logger.info(f"✅ Baseline cascade entrainé:")
        logger.info(f"   Total: {self.training_stats['total_matches']} matchs")
        logger.info(f"   Draw: {self.training_stats['draw_percentage']:.1f}%")
        logger.info(f"   Home: {self.training_stats['home_percentage']:.1f}%") 
        logger.info(f"   Away: {self.training_stats['away_percentage']:.1f}%")
        logger.info(f"   NotDraw samples: {self.training_stats['notdraw_samples']}")
        
        return self
    
    def predict(self, X):
        """Prédiction cascade"""
        pred_draw = self.clf_draw.predict(X)
        pred_homeaway = self.clf_homeaway.predict(X)
        
        y_pred = []
        for is_draw, home_away in zip(pred_draw, pred_homeaway):
            if is_draw == 1:
                y_pred.append('D')
            else:
                y_pred.append('H' if home_away == 1 else 'A')
        
        return np.array(y_pred)
    
    def get_training_stats(self):
        """Retourne statistiques entrainement"""
        return self.training_stats

def load_40_real_matches():
    """Charge les 40 vrais résultats EPL 2025-26 de référence"""
    df_real = pd.read_csv('data/raw/E0 (7).csv', encoding='utf-8-sig')
    
    # Normalisation équipes (cohérence avec dataset)
    team_mapping = {
        'Man United': 'Man United',
        'Spurs': 'Tottenham',
        "Nott'm Forest": "Nott'm Forest"
    }
    
    df_real['HomeTeam'] = df_real['HomeTeam'].replace(team_mapping)
    df_real['AwayTeam'] = df_real['AwayTeam'].replace(team_mapping)
    
    logger.info(f"✅ {len(df_real)} vrais résultats EPL 2025-26 chargés")
    return df_real[['Date', 'HomeTeam', 'AwayTeam', 'FTR']]

def establish_40_matches_baseline():
    """Établit baseline officielle sur 40 matchs EPL 2025-26"""
    logger.info("🎯 ÉTABLISSEMENT BASELINE OFFICIELLE - 40 MATCHS EPL 2025-26")
    logger.info("=" * 80)
    
    try:
        # 1. Charger dataset production
        dataset_path = 'data/processed/v15_final_enhanced.csv'
        df = pd.read_csv(dataset_path, parse_dates=['Date'])
        logger.info(f"📂 Dataset production: {df.shape}")
        
        # 2. Charger 40 vrais résultats
        real_matches = load_40_real_matches()
        
        # 3. Créer target encoding
        if 'target' not in df.columns and 'FullTimeResult' in df.columns:
            df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
            logger.info("✅ Target encoding appliqué")
        
        # 4. Données entrainement (anti-leakage strict)
        train_cutoff = pd.to_datetime('2025-08-01')
        df_train = df[df['Date'] < train_cutoff].copy()
        
        # Nettoyer données entrainement
        valid_mask = df_train['target'].notna()
        df_train_clean = df_train[valid_mask].copy()
        
        logger.info(f"🔒 Entrainement: {len(df_train_clean)} matchs avant {train_cutoff.date()}")
        
        # 5. Données test - Étendre dataset avec 40 matchs complets
        # D'abord essayer avec les données actuelles
        df_test_current = df[df['Date'] >= train_cutoff].head(40).copy()
        
        # Correspondance avec vrais résultats
        df_test = pd.merge(
            df_test_current, real_matches,
            on=['HomeTeam', 'AwayTeam'], 
            how='inner'
        )
        
        logger.info(f"🎯 Matchs test correspondants: {len(df_test)}")
        
        if len(df_test) < 35:
            logger.warning(f"⚠️  Seulement {len(df_test)} correspondances, extension nécessaire")
            
            # Utiliser le système d'auto-intégration pour avoir les 40 complets
            logger.info("🔄 Extension via système auto-intégration...")
            
            import subprocess
            result = subprocess.run([
                'python3', 'scripts/auto_update/match_updater.py',
                '--base-dataset', dataset_path,
                '--new-csv', 'data/raw/E0 (7).csv'
            ], capture_output=True, text=True)
            
            # Charger dataset étendu
            extended_files = [f for f in Path('data/processed/').glob('v_auto_update_*.csv')]
            if extended_files:
                latest_file = max(extended_files, key=lambda x: x.stat().st_mtime)
                df_extended = pd.read_csv(latest_file, parse_dates=['Date'])
                
                # Target encoding pour dataset étendu
                if 'FullTimeResult' in df_extended.columns:
                    df_extended['target'] = df_extended['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
                
                # Nouvelles données test
                df_train = df_extended[df_extended['Date'] < train_cutoff].copy()
                df_train_clean = df_train[df_train['target'].notna()].copy()
                
                df_test_extended = df_extended[df_extended['Date'] >= train_cutoff].head(40).copy()
                df_test = pd.merge(
                    df_test_extended, real_matches,
                    on=['HomeTeam', 'AwayTeam'],
                    how='inner'
                )
                
                logger.info(f"📈 Dataset étendu: {len(df_test)} correspondances")
        
        # 6. Features modèle cascade (ordre exact v2.3)
        model_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        # Vérifier features disponibles
        missing = [f for f in model_features if f not in df_test.columns]
        if missing:
            logger.error(f"❌ Features manquantes: {missing}")
            return None
        
        logger.info(f"✅ {len(model_features)} features modèle disponibles")
        
        # 7. Préparer données finales
        X_train = df_train_clean[model_features].fillna(0.5)
        y_train = df_train_clean['target']
        
        X_test = df_test[model_features].fillna(0.5)
        y_real = df_test['FTR']  # Vrais résultats
        
        logger.info(f"📊 Données finales: X_train{X_train.shape}, X_test{X_test.shape}")
        
        # 8. ENTRAINEMENT MODÈLE BASELINE
        logger.info("⚙️  Entrainement modèle cascade baseline...")
        baseline_model = CascadeModelBaseline()
        baseline_model.fit(X_train, y_train)
        
        # 9. TEST SUR 40 MATCHS OFFICIELS
        logger.info("🎯 Test sur 40 matchs EPL 2025-26 J1-J4...")
        y_pred = baseline_model.predict(X_test)
        baseline_accuracy = accuracy_score(y_real, y_pred)
        
        # 10. RÉSULTATS BASELINE OFFICIELLE
        logger.info(f"\n🏆 BASELINE OFFICIELLE - 40 MATCHS EPL 2025-26")
        logger.info("=" * 70)
        
        # Performance
        logger.info(f"🎯 PERFORMANCE BASELINE OFFICIELLE:")
        logger.info(f"   Accuracy: {baseline_accuracy:.3f} ({baseline_accuracy*100:.1f}%)")
        logger.info(f"   Dataset: {dataset_path}")
        logger.info(f"   Modèle: Cascade Draw vs NotDraw → Home vs Away")
        logger.info(f"   Features: {len(model_features)} features v2.3")
        logger.info(f"   Entrainement: {len(X_train)} matchs historiques")
        logger.info(f"   Test: {len(df_test)} matchs EPL 2025-26 J1-J4")
        
        # Distribution réelle
        real_dist = pd.Series(y_real).value_counts(normalize=True)
        logger.info(f"\n📊 DISTRIBUTION RÉELLE (baseline reference):")
        logger.info(f"   Home: {real_dist.get('H', 0):.1%}")
        logger.info(f"   Draw: {real_dist.get('D', 0):.1%}")
        logger.info(f"   Away: {real_dist.get('A', 0):.1%}")
        
        # Distribution prédictions
        pred_dist = pd.Series(y_pred).value_counts(normalize=True)
        logger.info(f"\n🔮 DISTRIBUTION PRÉDICTIONS (baseline):")
        logger.info(f"   Home: {pred_dist.get('H', 0):.1%}")
        logger.info(f"   Draw: {pred_dist.get('D', 0):.1%}")
        logger.info(f"   Away: {pred_dist.get('A', 0):.1%}")
        
        # Matrice confusion détaillée
        cm = confusion_matrix(y_real, y_pred, labels=['H', 'D', 'A'])
        logger.info(f"\n📊 MATRICE CONFUSION BASELINE:")
        logger.info(f"     Real\\Pred  H   D   A    Total")
        for i, label in enumerate(['H', 'D', 'A']):
            total = cm[i].sum()
            accuracy_class = cm[i][i] / total if total > 0 else 0
            logger.info(f"     {label}        {cm[i][0]:2d}  {cm[i][1]:2d}  {cm[i][2]:2d}    {total:2d} ({accuracy_class:.1%})")
        
        # Rapport classification complet
        logger.info(f"\n📈 RAPPORT CLASSIFICATION DÉTAILLÉ:")
        report = classification_report(y_real, y_pred, target_names=['Home', 'Draw', 'Away'], output_dict=True)
        for class_name in ['Home', 'Draw', 'Away']:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            logger.info(f"   {class_name:4s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}")
        
        # Comparaison benchmarks
        logger.info(f"\n📈 COMPARAISON BENCHMARKS BASELINE:")
        random_baseline = 0.333
        majority_baseline = real_dist.get('H', 0.436)  # Toujours prédire Home
        
        vs_random = baseline_accuracy - random_baseline
        vs_majority = baseline_accuracy - majority_baseline
        
        logger.info(f"   vs Random (33.3%): {vs_random:+.1%}")
        logger.info(f"   vs Majority Class: {vs_majority:+.1%}")
        
        if baseline_accuracy >= 0.55:
            performance_level = "🔥 EXCELLENT"
        elif baseline_accuracy >= 0.50:
            performance_level = "✅ BON"
        elif baseline_accuracy >= 0.45:
            performance_level = "⚠️  ACCEPTABLE"
        else:
            performance_level = "❌ INSUFFISANT"
        
        logger.info(f"   Niveau: {performance_level}")
        
        # 11. SAUVEGARDE BASELINE OFFICIELLE
        baseline_metadata = {
            'timestamp': datetime.now().isoformat(),
            'baseline_type': 'CASCADE_40_MATCHES_EPL_2025_26',
            'accuracy': float(baseline_accuracy),
            'dataset_path': dataset_path,
            'model_type': 'Cascade RandomForest (Draw vs NotDraw → Home vs Away)',
            'features_count': len(model_features),
            'features_used': model_features,
            'training_matches': len(X_train),
            'test_matches': len(df_test),
            'test_period': f"{df_test['Date'].min()} to {df_test['Date'].max()}",
            'real_distribution': {
                'Home': float(real_dist.get('H', 0)),
                'Draw': float(real_dist.get('D', 0)),
                'Away': float(real_dist.get('A', 0))
            },
            'predicted_distribution': {
                'Home': float(pred_dist.get('H', 0)),
                'Draw': float(pred_dist.get('D', 0)),
                'Away': float(pred_dist.get('A', 0))
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'training_stats': baseline_model.get_training_stats(),
            'benchmark_comparisons': {
                'vs_random_33pct': float(vs_random),
                'vs_majority_class': float(vs_majority)
            },
            'performance_level': performance_level,
            'notes': 'Baseline officielle établie sur 40 premiers matchs EPL 2025-26 J1-J4'
        }
        
        # Sauvegarder métadonnées baseline
        baseline_path = f'evaluation/baseline_40_matches_epl_2025_26.json'
        Path(baseline_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(baseline_path, 'w', encoding='utf-8') as f:
            json.dump(baseline_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 BASELINE OFFICIELLE SAUVEGARDÉE:")
        logger.info(f"   Métadonnées: {baseline_path}")
        
        # 12. RÉSUMÉ EXÉCUTIF
        logger.info(f"\n📋 RÉSUMÉ EXÉCUTIF BASELINE:")
        logger.info(f"🏆 Performance officielle: {baseline_accuracy:.1%}")
        logger.info(f"🎯 Référence absolue: 40 matchs EPL 2025-26 J1-J4")
        logger.info(f"⚙️  Modèle: Cascade RandomForest") 
        logger.info(f"📊 Niveau: {performance_level}")
        logger.info(f"✅ Baseline établie pour futures comparaisons")
        
        return {
            'accuracy': baseline_accuracy,
            'test_matches': len(df_test),
            'metadata_path': baseline_path,
            'performance_level': performance_level,
            'model': baseline_model
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur établissement baseline: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Établissement baseline officielle"""
    from pathlib import Path
    
    result = establish_40_matches_baseline()
    
    if result:
        print(f"\n🎉 BASELINE OFFICIELLE ÉTABLIE !")
        print(f"Performance: {result['accuracy']:.1%}")
        print(f"Référence: {result['test_matches']} matchs EPL 2025-26")
        print(f"Métadonnées: {result['metadata_path']}")
        print(f"Niveau: {result['performance_level']}")
    else:
        print(f"❌ Échec établissement baseline")

if __name__ == "__main__":
    main()