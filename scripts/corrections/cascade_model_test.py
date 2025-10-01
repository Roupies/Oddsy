#!/usr/bin/env python3
"""
🔄 PIPELINE CASCADE GÉNÉRIQUE - TEST V15 vs V16
=============================================

Modèle cascade:
1. Classifieur Draw vs NotDraw (binaire)
2. Classifieur Home vs Away sur NotDraw uniquement (binaire)
3. Assemblage → prédiction finale H/D/A

Test sur 40 vrais matchs EPL 2025-26 J1-J4
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime

class CascadeModel:
    """Modèle cascade Draw vs NotDraw -> Home vs Away"""
    
    def __init__(self, clf_draw=None, clf_homeaway=None):
        # Classifieur Draw vs NotDraw
        self.clf_draw = clf_draw or RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        
        # Classifieur Home vs Away (sur NotDraw uniquement)
        self.clf_homeaway = clf_homeaway or RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        
        self.is_fitted = False
    
    def fit(self, X, y):
        """Entrainement cascade"""
        print("🔄 Entrainement modèle cascade...")
        
        # Convertir target numérique vers string si nécessaire
        if y.dtype == 'int64':
            y_str = y.map({0: 'H', 1: 'D', 2: 'A'})
        else:
            y_str = y
        
        # Étape 1: Créer labels binaires Draw vs NotDraw
        y_draw = (y_str == 'D').astype(int)
        print(f"  Draw distribution: {np.mean(y_draw):.1%}")
        
        # Entrainement classifieur Draw
        self.clf_draw.fit(X, y_draw)
        
        # Étape 2: Entrainer Home vs Away sur NotDraw uniquement
        mask_notdraw = y_str != 'D'
        if mask_notdraw.sum() > 0:
            X_notdraw = X[mask_notdraw]
            y_homeaway = y_str[mask_notdraw].map({'H': 1, 'A': 0})
            print(f"  Home/Away sur {mask_notdraw.sum()} matchs NotDraw")
            print(f"  Home distribution: {np.mean(y_homeaway):.1%}")
            
            self.clf_homeaway.fit(X_notdraw, y_homeaway)
        else:
            print("  ⚠️  Aucun match NotDraw pour entrainer Home/Away")
        
        self.is_fitted = True
        print("✅ Modèle cascade entrainé")
        return self
    
    def predict(self, X):
        """Prédiction cascade"""
        if not self.is_fitted:
            raise ValueError("Modèle non entrainé")
        
        # Étape 1: Prédire Draw vs NotDraw
        pred_draw = self.clf_draw.predict(X)
        
        # Étape 2: Prédire Home vs Away pour tous (même si Draw prédit)
        pred_homeaway = self.clf_homeaway.predict(X)
        
        # Étape 3: Assemblage final
        y_pred = []
        for i, (is_draw, home_prob) in enumerate(zip(pred_draw, pred_homeaway)):
            if is_draw == 1:
                y_pred.append('D')
            else:
                y_pred.append('H' if home_prob == 1 else 'A')
        
        return np.array(y_pred)
    
    def predict_proba(self, X):
        """Probabilités cascade (approximation)"""
        if not self.is_fitted:
            raise ValueError("Modèle non entrainé")
        
        # Probabilités Draw vs NotDraw
        prob_draw = self.clf_draw.predict_proba(X)[:, 1]  # P(Draw)
        prob_notdraw = 1 - prob_draw
        
        # Probabilités Home vs Away
        prob_homeaway = self.clf_homeaway.predict_proba(X)
        prob_home_given_notdraw = prob_homeaway[:, 1]  # P(Home|NotDraw)
        prob_away_given_notdraw = prob_homeaway[:, 0]  # P(Away|NotDraw)
        
        # Probabilités finales
        prob_final = np.zeros((X.shape[0], 3))  # [Home, Draw, Away]
        prob_final[:, 0] = prob_notdraw * prob_home_given_notdraw  # P(Home)
        prob_final[:, 1] = prob_draw                               # P(Draw)
        prob_final[:, 2] = prob_notdraw * prob_away_given_notdraw  # P(Away)
        
        return prob_final

def load_real_results():
    """Charge 40 vrais résultats EPL 2025-26"""
    df_real = pd.read_csv('data/raw/E0 (7).csv', encoding='utf-8-sig')
    
    # Normaliser noms équipes
    team_mapping = {
        'Man United': 'Man United',
        'Spurs': 'Tottenham',
        "Nott'm Forest": "Nott'm Forest"
    }
    
    df_real['HomeTeam'] = df_real['HomeTeam'].replace(team_mapping)
    df_real['AwayTeam'] = df_real['AwayTeam'].replace(team_mapping)
    
    return df_real[['HomeTeam', 'AwayTeam', 'FTR']]

def test_cascade_on_dataset(dataset_path, dataset_name, real_results):
    """Test modèle cascade sur un dataset"""
    print(f"\n🧪 TEST CASCADE - {dataset_name.upper()}")
    print("=" * 60)
    
    try:
        # Charger dataset
        df = pd.read_csv(dataset_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"Dataset: {df.shape}")
        print(f"Période: {df['Date'].min()} → {df['Date'].max()}")
        
        # Target encoding
        if 'target' not in df.columns and 'FullTimeResult' in df.columns:
            df['target'] = df['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Features modèle v2.3 (ordre exact)
        model_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        # Vérifier features
        missing = [f for f in model_features if f not in df.columns]
        if missing:
            print(f"❌ Features manquantes: {missing}")
            return None
        
        # Données historiques pour entrainement (avant 2025-08-01)
        df_train = df[df['Date'] < '2025-08-01'].copy()
        
        # Données test (2025-26) matchées avec vrais résultats
        df_test = df[df['Date'] >= '2025-08-01'].head(30).copy()
        
        df_merged = pd.merge(
            df_test, real_results,
            on=['HomeTeam', 'AwayTeam'],
            how='inner'
        )
        
        print(f"Entrainement: {len(df_train)} matchs historiques")
        print(f"Test: {len(df_merged)} matchs réels EPL 2025-26")
        
        if len(df_train) < 100:
            print("❌ Pas assez de données d'entrainement")
            return None
        
        if len(df_merged) < 20:
            print("❌ Pas assez de matchs test correspondants")
            return None
        
        # Préparer données entrainement
        X_train = df_train[model_features].fillna(0.5)
        y_train = df_train['target']
        
        # Préparer données test
        X_test = df_merged[model_features].fillna(0.5)
        y_real = df_merged['FTR']  # Vrais résultats
        
        print(f"Shape: X_train{X_train.shape}, X_test{X_test.shape}")
        
        # Entrainer modèle cascade
        model = CascadeModel()
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_real, y_pred)
        
        return {
            'accuracy': accuracy,
            'y_real': y_real,
            'y_pred': y_pred,
            'n_train': len(df_train),
            'n_test': len(df_merged),
            'model': model
        }
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_cascade_results(results, real_results):
    """Compare résultats cascade vs modèle unique"""
    print(f"\n🔄 COMPARAISON MODÈLES CASCADE")
    print("=" * 60)
    
    # Distribution réelle
    real_dist = real_results['FTR'].value_counts(normalize=True)
    print(f"\n📈 Distribution réelle (30 matchs):")
    print(f"   Home: {real_dist.get('H', 0):.1%}")
    print(f"   Draw: {real_dist.get('D', 0):.1%}")
    print(f"   Away: {real_dist.get('A', 0):.1%}")
    
    best_acc = 0
    best_approach = None
    
    # Résultats par approche
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
            print(f"   Matrice confusion:")
            cm = confusion_matrix(y_real, y_pred, labels=['H', 'D', 'A'])
            print(f"     Real\\Pred  H   D   A")
            for i, label in enumerate(['H', 'D', 'A']):
                print(f"     {label}        {cm[i][0]:2d}  {cm[i][1]:2d}  {cm[i][2]:2d}")
            
            if acc > best_acc:
                best_acc = acc
                best_approach = approach
        else:
            print(f"\n❌ CASCADE {approach.upper()}: Échec")
    
    # Recommandation
    print(f"\n🏆 RÉSULTAT CASCADE")
    print("=" * 30)
    
    if best_approach:
        print(f"🥇 Meilleur cascade: {best_approach}")
        print(f"📊 Performance: {best_acc:.1%}")
        
        # Comparaison avec modèle unique précédent (43.3%)
        baseline_acc = 0.433
        improvement = best_acc - baseline_acc
        
        if improvement > 0.02:
            print(f"🔥 AMÉLIORATION significative: +{improvement:.1%}")
            recommendation = "ADOPTER CASCADE"
        elif improvement > 0:
            print(f"✅ Amélioration légère: +{improvement:.1%}")
            recommendation = "ENVISAGER CASCADE"
        else:
            print(f"❌ Performance inférieure: {improvement:.1%}")
            recommendation = "GARDER MODÈLE UNIQUE"
        
        print(f"🎯 Recommandation: {recommendation}")
        
        return best_approach, best_acc, recommendation
    else:
        print("❌ Aucun modèle cascade valide")
        return None, None, "ÉCHEC CASCADE"

def main():
    """Pipeline test cascade"""
    print("🚀 TEST PIPELINE CASCADE - V15 vs V16")
    print("=" * 50)
    
    # Charger vrais résultats
    real_results = load_real_results()
    print(f"✅ {len(real_results)} vrais résultats chargés")
    
    # Datasets à tester
    datasets = {
        'v15_baseline': 'data/processed/v15_final_enhanced.csv',
        'v16_contextuelles': 'data/processed/v16_contextual_features_20250915_171540.csv'
    }
    
    results = {}
    
    # Test cascade sur chaque dataset
    for name, path in datasets.items():
        result = test_cascade_on_dataset(path, name, real_results)
        results[name] = result
    
    # Comparaison
    best_approach, best_acc, recommendation = compare_cascade_results(results, real_results)
    
    # Résumé final
    print(f"\n📋 RÉSUMÉ CASCADE")
    print("=" * 30)
    print(f"✅ Test sur 30 vrais matchs EPL 2025-26")
    print(f"🔄 Modèle: Draw vs NotDraw → Home vs Away")
    print(f"🏆 Meilleur: {best_approach} ({best_acc:.1%})")
    print(f"💡 Décision: {recommendation}")
    
    return best_approach, recommendation

if __name__ == "__main__":
    main()