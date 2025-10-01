#!/usr/bin/env python3
"""
🎯 CASCADE TEST SIMPLIFIÉ - 40 MATCHS RÉELS EPL 2025-26
====================================================

Test cascade simplifié mais CORRECT sur les 40 vrais matchs.
Anti-leakage standard (pas ultra-strict) pour avoir des résultats concrets.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleCascadeModel:
    """Modèle cascade simplifié Draw vs NotDraw -> Home vs Away"""
    
    def __init__(self):
        self.clf_draw = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        self.clf_homeaway = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        self.is_fitted = False
    
    def fit(self, X, y):
        """Entrainement cascade"""
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
        if mask_notdraw.sum() > 5:  # Minimum 5 échantillons
            X_notdraw = X[mask_notdraw]
            y_homeaway = y_str[mask_notdraw].map({'H': 1, 'A': 0})
            self.clf_homeaway.fit(X_notdraw, y_homeaway)
        
        self.is_fitted = True
        print(f"✅ Cascade entrainé: {np.mean(y_draw):.1%} Draw, {mask_notdraw.sum()} NotDraw")
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

def load_40_real_matches():
    """Charge les 40 vrais résultats"""
    df_real = pd.read_csv('data/raw/E0 (7).csv', encoding='utf-8-sig')
    
    # Normaliser équipes
    team_mapping = {
        'Spurs': 'Tottenham',
        "Nott'm Forest": "Nott'm Forest"
    }
    df_real['HomeTeam'] = df_real['HomeTeam'].replace(team_mapping)
    df_real['AwayTeam'] = df_real['AwayTeam'].replace(team_mapping)
    
    print(f"✅ {len(df_real)} vrais résultats chargés")
    return df_real[['HomeTeam', 'AwayTeam', 'FTR']]

def simple_test_cascade(dataset_path, dataset_name, real_matches):
    """Test cascade simplifié"""
    print(f"\n🎯 CASCADE SIMPLIFIÉ - {dataset_name.upper()}")
    print("=" * 50)
    
    try:
        # Charger dataset
        df = pd.read_csv(dataset_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"Dataset: {df.shape}")
        
        # Créer target si nécessaire
        if 'target' not in df.columns and 'FullTimeResult' in df.columns:
            df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Anti-leakage simple: entrainement avant 2025-08-01
        train_cutoff = pd.to_datetime('2025-08-01')
        df_train = df[df['Date'] < train_cutoff].copy()
        
        # Test: matchs 2025-26 correspondant aux 40 réels  
        df_season_2025 = df[df['Date'] >= '2025-08-01'].head(40).copy()
        
        # Correspondance avec vrais résultats
        df_test = pd.merge(
            df_season_2025, real_matches,
            on=['HomeTeam', 'AwayTeam'],
            how='inner'
        )
        
        print(f"Entrainement: {len(df_train)} matchs avant {train_cutoff.date()}")
        print(f"Test: {len(df_test)} matchs correspondants")
        
        if len(df_train) < 100:
            print("❌ Pas assez de données entrainement")
            return None
        
        if len(df_test) < 25:
            print("❌ Correspondance insuffisante")
            return None
        
        # Features modèle
        features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        missing = [f for f in features if f not in df_test.columns]
        if missing:
            print(f"❌ Features manquantes: {missing}")
            return None
        
        # Données
        X_train = df_train[features].fillna(0.5)
        y_train = df_train['target'].dropna()
        
        # Aligner X_train avec y_train
        valid_indices = df_train['target'].notna()
        X_train = X_train[valid_indices]
        
        X_test = df_test[features].fillna(0.5)
        y_real = df_test['FTR']
        
        print(f"Données: X_train{X_train.shape}, X_test{X_test.shape}")
        
        if len(X_train) < 50:
            print("❌ Données entrainement insuffisantes après nettoyage")
            return None
        
        # Entrainement cascade
        model = SimpleCascadeModel()
        model.fit(X_train, y_train)
        
        # Test
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_real, y_pred)
        
        return {
            'accuracy': accuracy,
            'y_real': y_real,
            'y_pred': y_pred,
            'n_train': len(X_train),
            'n_test': len(df_test)
        }
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Test cascade simplifié sur 40 matchs"""
    print("🚀 TEST CASCADE SIMPLIFIÉ - 40 MATCHS EPL 2025-26")
    print("=" * 60)
    
    # Charger vrais résultats  
    real_matches = load_40_real_matches()
    
    # Datasets à tester
    datasets = {
        'v15_baseline': 'data/processed/v15_final_enhanced.csv',
        'v16_contextuelles': 'data/processed/v16_contextual_features_20250915_171540.csv',
        'v_auto_integrated': 'data/processed/v_auto_update_20250916_105039.csv'
    }
    
    results = {}
    
    # Tests
    for name, path in datasets.items():
        result = simple_test_cascade(path, name, real_matches)
        results[name] = result
    
    # Comparaison
    print(f"\n🏆 RÉSULTATS CASCADE - 40 MATCHS RÉELS")
    print("=" * 50)
    
    # Distribution réelle
    real_dist = real_matches['FTR'].value_counts(normalize=True)
    print(f"📈 Distribution réelle:")
    print(f"   Home: {real_dist.get('H', 0):.1%}")
    print(f"   Draw: {real_dist.get('D', 0):.1%}")  
    print(f"   Away: {real_dist.get('A', 0):.1%}")
    
    best_acc = 0
    best_approach = None
    
    for approach, result in results.items():
        if result:
            acc = result['accuracy']
            y_real = result['y_real']
            y_pred = result['y_pred']
            
            print(f"\n🎯 CASCADE {approach.upper()}")
            print(f"   Accuracy: {acc:.3f} ({acc*100:.1f}%)")
            print(f"   Entrainement: {result['n_train']} matchs")
            print(f"   Test: {result['n_test']} matchs")
            
            # Distribution prédictions
            pred_dist = pd.Series(y_pred).value_counts(normalize=True)
            print(f"   Prédictions: H={pred_dist.get('H', 0):.1%}, D={pred_dist.get('D', 0):.1%}, A={pred_dist.get('A', 0):.1%}")
            
            # Matrice confusion
            cm = confusion_matrix(y_real, y_pred, labels=['H', 'D', 'A'])
            print(f"   Matrice confusion:")
            print(f"     Real\\Pred  H   D   A")
            for i, label in enumerate(['H', 'D', 'A']):
                print(f"     {label}        {cm[i][0]:2d}  {cm[i][1]:2d}  {cm[i][2]:2d}")
            
            if acc > best_acc:
                best_acc = acc
                best_approach = approach
        else:
            print(f"\n❌ CASCADE {approach.upper()}: Échec")
    
    # Recommandation finale
    print(f"\n📋 RECOMMANDATION CASCADE")
    print("=" * 30)
    
    if best_approach:
        print(f"🏆 Meilleur: {best_approach} ({best_acc:.1%})")
        
        # Comparaison avec modèle unique (43.3%)
        baseline_single = 0.433
        improvement = best_acc - baseline_single
        
        if improvement >= 0.05:
            print(f"🔥 AMÉLIORATION MAJEURE: +{improvement:.1%}")
            decision = "ADOPTER CASCADE"
        elif improvement >= 0.02:
            print(f"✅ Bonne amélioration: +{improvement:.1%}")
            decision = "ADOPTER CASCADE"
        elif improvement > 0:
            print(f"⚠️  Amélioration légère: +{improvement:.1%}")
            decision = "ENVISAGER CASCADE"
        else:
            print(f"❌ Performance inférieure: {improvement:.1%}")
            decision = "RESTER MODÈLE UNIQUE"
        
        print(f"🎯 Décision: {decision}")
        
        return best_approach, best_acc, decision
    else:
        print("❌ Aucun test valide")
        return None, None, "ÉCHEC"

if __name__ == "__main__":
    main()