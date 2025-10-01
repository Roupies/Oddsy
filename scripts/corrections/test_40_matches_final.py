#!/usr/bin/env python3
"""
🎯 TEST FINAL - 40 PREMIERS MATCHS J1-J4 EPL 2025-26
================================================

Test définitif sur les 40 vrais premiers matchs de la saison.
Comparaison des 3 approches pour décision finale.

Données : E0 (7).csv - 40 matchs complets J1-J4
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime

def load_real_results():
    """Charge les 40 vrais résultats EPL 2025-26"""
    print("📊 Chargement des 40 vrais résultats EPL 2025-26...")
    
    # Charger fichier E0 avec vrais résultats
    df_real = pd.read_csv('data/raw/E0 (7).csv', encoding='utf-8-sig')
    
    print(f"Matchs chargés: {len(df_real)}")
    print(f"Colonnes: {list(df_real.columns)[:10]}...")
    
    # Normaliser noms équipes pour correspondance avec dataset
    team_mapping = {
        'Man United': 'Man United',
        'Man City': 'Man City', 
        'Spurs': 'Tottenham',
        "Nott'm Forest": "Nott'm Forest"
    }
    
    df_real['HomeTeam'] = df_real['HomeTeam'].replace(team_mapping)
    df_real['AwayTeam'] = df_real['AwayTeam'].replace(team_mapping)
    
    # Convertir résultats
    result_map = {'H': 0, 'D': 1, 'A': 2}
    df_real['target_real'] = df_real['FTR'].map(result_map)
    
    print("✅ 40 vrais résultats chargés")
    return df_real[['HomeTeam', 'AwayTeam', 'FTR', 'target_real']]

def test_approach_on_real_matches(dataset_path, approach_name, real_results):
    """Test une approche sur les 40 vrais matchs"""
    print(f"\n🧪 TEST {approach_name.upper()}")
    print("=" * 50)
    
    try:
        # Charger dataset
        df = pd.read_csv(dataset_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filtrer saison 2025-26 
        df_season = df[df['Date'] >= '2025-08-01'].copy()
        print(f"Matchs 2025-26 dans dataset: {len(df_season)}")
        
        if len(df_season) < 30:
            print("❌ Pas assez de matchs 2025-26")
            return None
        
        # Prendre premiers matchs disponibles (devrait correspondre aux 40 vrais)
        df_test = df_season.head(len(real_results)).copy()
        print(f"Matchs de test: {len(df_test)}")
        
        # Ajouter target si manquant
        if 'target' not in df_test.columns:
            df_test['target'] = df_test['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Merger avec vrais résultats pour validation
        df_merged = pd.merge(
            df_test, real_results, 
            on=['HomeTeam', 'AwayTeam'], 
            how='inner'
        )
        
        print(f"Matchs correspondants: {len(df_merged)}")
        if len(df_merged) < 30:
            print("⚠️  Correspondance insuffisante avec vrais résultats")
            return None
        
        # Features modèle (ordre exact)
        model_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
            'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
            'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        # Vérifier features
        missing_features = [f for f in model_features if f not in df_merged.columns]
        if missing_features:
            print(f"❌ Features manquantes: {missing_features}")
            return None
        
        # Préparer données
        X_test = df_merged[model_features].fillna(0.5)
        y_real = df_merged['target_real']  # Vrais résultats
        
        print(f"Shape test: X{X_test.shape}, y_real{y_real.shape}")
        
        # Tester avec modèle
        model = joblib.load('models/v23_retrained_2025_09_11_154613.joblib')
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_real, y_pred)
        
        return {
            'accuracy': accuracy,
            'y_real': y_real,
            'y_pred': y_pred,
            'matches_tested': len(df_merged)
        }
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_final_results(results, real_results):
    """Compare finale des résultats"""
    print(f"\n🏆 COMPARAISON FINALE - 40 MATCHS RÉELS EPL 2025-26")
    print("=" * 70)
    
    # Stats générales vrais résultats
    real_dist = real_results['target_real'].value_counts(normalize=True)
    print(f"\n📈 Distribution réelle J1-J4:")
    print(f"   Home: {real_dist.get(0, 0):.1%}")
    print(f"   Draw: {real_dist.get(1, 0):.1%}") 
    print(f"   Away: {real_dist.get(2, 0):.1%}")
    
    # Résultats par approche
    best_accuracy = 0
    best_approach = None
    
    for approach_name, result in results.items():
        if result is not None:
            acc = result['accuracy']
            y_real = result['y_real']
            y_pred = result['y_pred']
            n_matches = result['matches_tested']
            
            print(f"\n🎯 {approach_name.upper()}")
            print(f"   Accuracy: {acc:.3f} ({acc*100:.1f}%)")
            print(f"   Matchs testés: {n_matches}")
            
            # Distribution prédictions
            pred_dist = pd.Series(y_pred).value_counts(normalize=True)
            print(f"   Prédictions: H={pred_dist.get(0, 0):.1%}, D={pred_dist.get(1, 0):.1%}, A={pred_dist.get(2, 0):.1%}")
            
            # F1-scores
            try:
                report = classification_report(y_real, y_pred, target_names=['Home', 'Draw', 'Away'], output_dict=True)
                f1_home = report['Home']['f1-score']
                f1_draw = report['Draw']['f1-score'] 
                f1_away = report['Away']['f1-score']
                f1_macro = report['macro avg']['f1-score']
                print(f"   F1-Score: H={f1_home:.2f}, D={f1_draw:.2f}, A={f1_away:.2f}, Macro={f1_macro:.2f}")
            except:
                print("   F1-Score: Calcul échoué")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_approach = approach_name
        else:
            print(f"\n❌ {approach_name.upper()}: Échec")
    
    # Recommandation finale
    print(f"\n👑 RECOMMANDATION FINALE")
    print("=" * 40)
    
    if best_approach:
        print(f"🥇 Meilleure approche: {best_approach}")
        print(f"📊 Performance: {best_accuracy:.1%}")
        
        # Évaluation performance
        if best_accuracy >= 0.50:
            print("🔥 EXCELLENT - Performance > 50%")
            recommendation = "ADOPTER"
        elif best_accuracy >= 0.47:
            print("✅ BON - Performance ≥ 47%")
            recommendation = "ADOPTER"
        elif best_accuracy >= 0.425:
            print("⚠️  ACCEPTABLE - Performance ≥ baseline 42.5%")
            recommendation = "ACCEPTABLE"
        else:
            print("❌ INSUFFISANT - Performance < baseline")
            recommendation = "REJETER"
        
        print(f"🎯 Décision: {recommendation}")
        
        # Dataset recommandé
        dataset_mapping = {
            'v17_rolling_correct': 'data/processed/v17_xg_corrected_20250916_100539.csv',
            'v16_contextuelles': 'data/processed/v16_contextual_features_20250915_171540.csv', 
            'v15_baseline': 'data/processed/v15_final_enhanced.csv'
        }
        
        recommended_dataset = dataset_mapping.get(best_approach, 'Unknown')
        print(f"📂 Dataset: {recommended_dataset}")
        
        return best_approach, best_accuracy, recommended_dataset
    else:
        print("❌ Aucune approche valide")
        return None, None, None

def main():
    """Pipeline test final"""
    print("🚀 TEST FINAL - 40 PREMIERS MATCHS J1-J4 EPL 2025-26")
    print("=" * 60)
    
    try:
        # Charger vrais résultats
        real_results = load_real_results()
        
        # Tests à effectuer
        approaches = {
            'v17_rolling_correct': 'data/processed/v17_xg_corrected_20250916_100539.csv',
            'v16_contextuelles': 'data/processed/v16_contextual_features_20250915_171540.csv',
            'v15_baseline': 'data/processed/v15_final_enhanced.csv'
        }
        
        results = {}
        
        # Tester chaque approche
        for approach_name, dataset_path in approaches.items():
            result = test_approach_on_real_matches(dataset_path, approach_name, real_results)
            results[approach_name] = result
        
        # Comparaison finale
        best_approach, best_accuracy, recommended_dataset = compare_final_results(results, real_results)
        
        # Résumé
        print(f"\n📋 RÉSUMÉ EXÉCUTIF")
        print("=" * 30)
        print(f"✅ Test sur 40 vrais matchs EPL 2025-26 J1-J4")
        print(f"🏆 Meilleure approche: {best_approach}")
        print(f"📈 Performance validée: {best_accuracy:.1%}")
        print(f"📂 Dataset recommandé: {recommended_dataset}")
        
        return best_approach, recommended_dataset
        
    except Exception as e:
        print(f"❌ Erreur globale: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()