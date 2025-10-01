#!/usr/bin/env python3
"""
🎯 COMPARAISON APPROCHES XG - J1-J4 (40 MATCHS)
============================================

Test complet pour décider quelle approche garder:
1. Approche actuelle (dataset v16 avec features contextuelles)  
2. Approche corrigée (rolling production réaliste)

Test sur exactement 40 premiers matchs EPL 2025-26.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime

def load_and_test_approach(dataset_path, approach_name, model_path):
    """Test une approche sur les 40 premiers matchs"""
    print(f"\n🧪 TEST APPROCHE: {approach_name}")
    print("=" * 60)
    
    try:
        # Charger dataset
        df = pd.read_csv(dataset_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"Dataset chargé: {df.shape}")
        print(f"Colonnes: {len(df.columns)}")
        
        # Filtrer exactement 40 premiers matchs 2025-26
        df_season_2025 = df[df['Date'] >= '2025-08-01'].copy()
        df_test = df_season_2025.head(40).copy()
        
        print(f"Matchs saison 2025-26 disponibles: {len(df_season_2025)}")
        print(f"Test sur: {len(df_test)} matchs")
        print(f"Période: {df_test['Date'].min()} → {df_test['Date'].max()}")
        
        if len(df_test) < 40:
            print(f"⚠️  Seulement {len(df_test)} matchs disponibles (<40)")
            return None, None, df_test
        
        # Ajouter target si manquant
        if 'target' not in df_test.columns and 'FullTimeResult' in df_test.columns:
            df_test['target'] = df_test['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Features modèle v2.3 (ordre exact)
        model_features = [
            'form_diff_normalized',
            'elo_diff_normalized', 
            'h2h_score',
            'matchday_normalized',
            'shots_diff_normalized',
            'corners_diff_normalized',
            'market_entropy_norm',
            'home_xg_eff_10',
            'away_goals_sum_5',
            'away_xg_eff_10'
        ]
        
        # Vérifier features disponibles
        available_features = [f for f in model_features if f in df_test.columns]
        missing_features = [f for f in model_features if f not in df_test.columns]
        
        if missing_features:
            print(f"❌ Features manquantes: {missing_features}")
            return None, None, df_test
        
        print(f"✅ Features disponibles: {len(available_features)}/10")
        
        # Préparer données
        X_test = df_test[available_features].fillna(0.5)
        y_test = df_test['target']
        
        print(f"Shape test: X{X_test.shape}, y{y_test.shape}")
        
        # Charger et tester modèle
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, y_pred, df_test
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None, None, None

def compare_results(results):
    """Compare les résultats des différentes approches"""
    print(f"\n📊 COMPARAISON RÉSULTATS FINAUX")
    print("=" * 60)
    
    for approach_name, (accuracy, y_pred, df_test) in results.items():
        if accuracy is not None:
            print(f"\n🎯 {approach_name.upper()}")
            print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            # Distribution
            y_test = df_test['target']
            print(f"Distribution réelle: H={np.mean(y_test==0):.1%}, D={np.mean(y_test==1):.1%}, A={np.mean(y_test==2):.1%}")
            print(f"Distribution prédite: H={np.mean(y_pred==0):.1%}, D={np.mean(y_pred==1):.1%}, A={np.mean(y_pred==2):.1%}")
            
            # Performance par classe
            report = classification_report(y_test, y_pred, target_names=['Home', 'Draw', 'Away'], output_dict=True)
            print(f"F1-Score: Home={report['Home']['f1-score']:.2f}, Draw={report['Draw']['f1-score']:.2f}, Away={report['Away']['f1-score']:.2f}")
        else:
            print(f"\n❌ {approach_name.upper()}: Test échoué")
    
    # Recommandation
    print(f"\n🏆 RECOMMANDATION")
    print("=" * 30)
    
    valid_results = {k: v for k, v in results.items() if v[0] is not None}
    if valid_results:
        best_approach = max(valid_results.items(), key=lambda x: x[1][0])
        best_name, (best_accuracy, _, _) = best_approach
        
        print(f"Meilleure approche: {best_name}")
        print(f"Performance: {best_accuracy:.1%}")
        
        if best_accuracy >= 0.47:
            print("✅ Performance satisfaisante (≥47%)")
        elif best_accuracy >= 0.42:
            print("⚠️  Performance acceptable (≥42% baseline)")  
        else:
            print("❌ Performance sous baseline (<42%)")
            
        return best_name, best_accuracy
    else:
        print("❌ Aucun test valide")
        return None, None

def main():
    """Pipeline principal comparaison"""
    print("🚀 COMPARAISON APPROCHES XG - TEST J1-J4 (40 MATCHS)")
    print("=" * 70)
    
    model_path = 'models/v23_retrained_2025_09_11_154613.joblib'
    
    # Tests à effectuer
    tests = {
        'v16_contextuelles': 'data/processed/v16_contextual_features_20250915_171540.csv',
        'v17_rolling_correct': 'data/processed/v17_xg_corrected_20250916_100539.csv',
        'v15_baseline': 'data/processed/v15_final_enhanced.csv'
    }
    
    results = {}
    
    # Exécuter tests
    for approach_name, dataset_path in tests.items():
        try:
            accuracy, y_pred, df_test = load_and_test_approach(
                dataset_path, approach_name, model_path
            )
            results[approach_name] = (accuracy, y_pred, df_test)
        except Exception as e:
            print(f"❌ Erreur test {approach_name}: {e}")
            results[approach_name] = (None, None, None)
    
    # Comparer résultats
    best_approach, best_accuracy = compare_results(results)
    
    # Résumé final
    print(f"\n📋 DÉCISION FINALE")
    print("=" * 30)
    
    if best_approach:
        print(f"👑 Approche recommandée: {best_approach}")
        print(f"📈 Performance validée: {best_accuracy:.1%}")
        
        if 'v17_rolling_correct' in best_approach:
            print("✅ Utiliser méthodologie production réaliste")
            print("   • Équipes promues: valeur neutre 0.5")
            print("   • Équipes existantes: rolling 10 matchs 2024-25")
            recommended_dataset = 'data/processed/v17_xg_corrected_20250916_100539.csv'
        elif 'v16_contextuelles' in best_approach:
            print("⚠️  Garder features contextuelles temporairement")
            print("   • Performance supérieure mais features non validées")
            recommended_dataset = 'data/processed/v16_contextual_features_20250915_171540.csv'
        else:
            print("🔄 Revenir au baseline v15")
            recommended_dataset = 'data/processed/v15_final_enhanced.csv'
        
        print(f"\n📂 Dataset recommandé: {recommended_dataset}")
        
        return recommended_dataset
    else:
        print("❌ Aucune approche satisfaisante")
        return None

if __name__ == "__main__":
    main()