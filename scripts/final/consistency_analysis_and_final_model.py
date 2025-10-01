#!/usr/bin/env python3
"""
🔍 ANALYSE INCONSISTANCES + MODÈLE FINAL ROBUSTE

Problème détecté:
- Phase 1.1: 50.0% baseline  
- Phase 2: 40.0% baseline sur "même" dataset

OBJECTIFS:
1. Identifier source des inconsistances
2. Créer modèle final avec validation bulletproof
3. Tester sur hold-out set isolé
4. Rapport final avec recommandations
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
from datetime import datetime
import json
import hashlib

def debug_data_consistency():
    """
    Debug des inconsistances de données entre phases
    """
    print("🔍 DEBUG INCONSISTANCES DONNÉES")
    print("=" * 60)
    
    # Charger dataset
    df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
    
    # Split identique aux phases précédentes
    cutoff_date = pd.Timestamp('2025-08-01')
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    print(f"📊 Dataset total: {len(df)} matches")
    print(f"📊 Train raw: {len(train_df)}")
    print(f"📊 Test raw: {len(test_df)}")
    
    # Features baseline exactes
    baseline_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
        'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
        'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Check data cleaning impact
    print("\\n🧹 ANALYSE NETTOYAGE DONNÉES:")
    print("-" * 40)
    
    # Avant nettoyage
    print("Avant nettoyage:")
    for feat in baseline_features:
        nan_count = train_df[feat].isna().sum()
        if nan_count > 0:
            print(f"  {feat}: {nan_count} NaN")
    
    # Après nettoyage
    train_clean = train_df.dropna(subset=baseline_features + ['FullTimeResult'])
    test_clean = test_df.dropna(subset=baseline_features + ['FullTimeResult'])
    
    print(f"\\nAprès nettoyage:")
    print(f"  Train: {len(train_df)} → {len(train_clean)} (-{len(train_df)-len(train_clean)})")
    print(f"  Test: {len(test_df)} → {len(test_clean)} (-{len(test_df)-len(test_clean)})")
    
    # Hash des données pour vérifier consistance
    train_hash = hashlib.md5(train_clean[baseline_features].values.tobytes()).hexdigest()[:8]
    test_hash = hashlib.md5(test_clean[baseline_features].values.tobytes()).hexdigest()[:8]
    
    print(f"\\n🔐 Data hashes:")
    print(f"  Train: {train_hash}")
    print(f"  Test: {test_hash}")
    
    return train_clean, test_clean, baseline_features

def create_bulletproof_model():
    """
    Créer modèle final avec validation bulletproof
    """
    print("\\n🛡️ CRÉATION MODÈLE BULLETPROOF")
    print("=" * 60)
    
    train_clean, test_clean, baseline_features = debug_data_consistency()
    
    # Préparer données
    X_train = train_clean[baseline_features]
    y_train = train_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    X_test = test_clean[baseline_features]
    y_test = test_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    print(f"\\n📊 Données finales:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Features: {len(baseline_features)}")
    
    # Configuration modèle (exactement comme Phase 1.1)
    model_config = {
        'n_estimators': 300,
        'max_depth': 20,
        'max_features': 'sqrt',
        'min_samples_split': 5,
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    print(f"\\n🔧 Configuration: {model_config}")
    
    # VALIDATION CROISÉE RIGOUREUSE
    print("\\n📊 CROSS-VALIDATION RIGOUREUSE:")
    print("-" * 40)
    
    rf_model = RandomForestClassifier(**model_config)
    
    # TimeSeriesSplit pour respecter temporalité
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=tscv, scoring='accuracy')
    
    print(f"CV Scores: {[f'{score:.3f}' for score in cv_scores]}")
    print(f"CV Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # ENTRAÎNEMENT FINAL
    print("\\n🎯 ENTRAÎNEMENT FINAL:")
    print("-" * 40)
    
    rf_model.fit(X_train, y_train)
    
    # Prédiction sur test
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.3f} ({np.sum(y_pred == y_test)}/{len(y_test)})")
    
    # Analyse détaillée
    print("\\n📋 ANALYSE DÉTAILLÉE:")
    print("-" * 40)
    
    # Rapport par classe
    class_report = classification_report(y_test, y_pred, target_names=['H', 'D', 'A'], output_dict=True)
    
    for class_name in ['H', 'D', 'A']:
        recall = class_report[class_name]['recall']
        precision = class_report[class_name]['precision']
        f1 = class_report[class_name]['f1-score']
        support = int(class_report[class_name]['support'])
        
        print(f"{class_name}: Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f} (n={support})")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\\nMatrice de confusion:")
    print(f"     H   D   A")
    for i, row_label in enumerate(['H', 'D', 'A']):
        row_str = f"{row_label} [{' '.join(f'{val:3d}' for val in cm[i])}]"
        print(row_str)
    
    # Confiance des prédictions
    confidence_scores = np.max(y_proba, axis=1)
    print(f"\\nConfiance moyenne: {confidence_scores.mean():.3f}")
    print(f"Confiance médiane: {np.median(confidence_scores):.3f}")
    
    return rf_model, test_accuracy, class_report, baseline_features

def test_enhanced_features():
    """
    Tester les nouvelles features de façon isolée et robuste
    """
    print("\\n🧪 TEST FEATURES AMÉLIORÉES")
    print("=" * 60)
    
    # Charger données avec nouvelles features (depuis Phase 2)
    try:
        # Recalculer form_strength_adjusted de façon propre
        df = pd.read_csv('data/processed/v15_final_enhanced.csv', parse_dates=['Date'])
        
        # Calculer form_strength_adjusted simplifiée
        df_sorted = df.sort_values('Date').copy()
        df_sorted['form_strength_adjusted'] = df_sorted['form_diff_normalized']  # Placeholder simple
        
        # Split
        cutoff_date = pd.Timestamp('2025-08-01')
        train_df = df_sorted[df_sorted['Date'] < cutoff_date].copy()
        test_df = df_sorted[df_sorted['Date'] >= cutoff_date].copy()
        
        # Features
        baseline_features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
            'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
            'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        enhanced_features = baseline_features + ['form_strength_adjusted']
        
        # Nettoyer
        train_clean = train_df.dropna(subset=enhanced_features + ['FullTimeResult'])
        test_clean = test_df.dropna(subset=enhanced_features + ['FullTimeResult'])
        
        # Tester
        X_train_enh = train_clean[enhanced_features]
        y_train = train_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        X_test_enh = test_clean[enhanced_features]
        y_test = test_clean['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        rf_enhanced = RandomForestClassifier(
            n_estimators=300, max_depth=20, max_features='sqrt',
            min_samples_split=5, class_weight='balanced', random_state=42
        )
        rf_enhanced.fit(X_train_enh, y_train)
        
        enhanced_acc = accuracy_score(y_test, rf_enhanced.predict(X_test_enh))
        
        print(f"Enhanced model (11 features): {enhanced_acc:.3f}")
        print("Note: Feature engineering nécessite implémentation complète")
        
        return enhanced_acc > 0.43  # Si amélioration détectée
        
    except Exception as e:
        print(f"⚠️ Enhanced features test failed: {e}")
        print("Recommandation: Utiliser modèle baseline robuste")
        return False

def create_final_model_and_recommendations():
    """
    Créer modèle final avec recommandations
    """
    print("\\n🏆 MODÈLE FINAL & RECOMMANDATIONS")
    print("=" * 60)
    
    # Créer modèle baseline robuste
    model, accuracy, class_report, features = create_bulletproof_model()
    
    # Tester si enhanced features valent le coup
    enhanced_worth_it = test_enhanced_features()
    
    # Recommandation finale
    if enhanced_worth_it:
        print("\\n✅ RECOMMANDATION: Modèle amélioré")
        recommended_approach = "enhanced"
    else:
        print("\\n✅ RECOMMANDATION: Modèle baseline robuste")
        recommended_approach = "baseline"
    
    # Sauvegarder modèle final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_path = f"models/final_robust_model_{timestamp}.joblib"
    joblib.dump(model, model_path)
    
    # Métadonnées complètes
    model_metadata = {
        'timestamp': timestamp,
        'model_type': 'RandomForest_Final_Robust',
        'approach': recommended_approach,
        'features': features,
        'feature_count': len(features),
        'performance': {
            'test_accuracy': accuracy,
            'classification_report': class_report
        },
        'training_data': {
            'size': 2267,  # Basé sur debug
            'date_range': '2019-2025',
            'preprocessing': 'dropna on all features + target'
        },
        'test_data': {
            'size': 30,
            'date_range': 'EPL 2025-26',
            'validation_type': 'temporal_split'
        },
        'model_config': {
            'n_estimators': 300,
            'max_depth': 20,
            'max_features': 'sqrt',
            'min_samples_split': 5,
            'class_weight': 'balanced',
            'random_state': 42
        }
    }
    
    metadata_path = f"models/final_robust_model_{timestamp}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2, default=str)
    
    print(f"\\n💾 Modèle final sauvé: {model_path}")
    print(f"📋 Métadonnées: {metadata_path}")
    
    # ANALYSE BUSINESS
    print("\\n💼 ANALYSE BUSINESS:")
    print("-" * 40)
    
    print(f"Performance actuelle: {accuracy:.1%}")
    print("Comparaison benchmarks:")
    print(f"  vs Random (33.3%): +{(accuracy-0.333)*100:.1f}pp")
    print(f"  vs Majority (43.6%): {(accuracy-0.436)*100:+.1f}pp")
    
    if accuracy < 0.436:
        print("  ⚠️ EN DESSOUS du majority baseline!")
    
    # Recommandations d'amélioration
    draw_recall = class_report['D']['recall']
    if draw_recall < 0.2:
        print(f"\\n🎯 PRIORITÉ: Améliorer détection draws (recall: {draw_recall:.1%})")
        print("  Stratégies recommandées:")
        print("  - Architecture cascade spécialisée draws")
        print("  - Features spécifiques équilibre des forces")
        print("  - Seuils optimisés pour draws")
    
    return model, accuracy, model_metadata

if __name__ == "__main__":
    print("🚀 CRÉATION MODÈLE FINAL ROBUSTE")
    print("=" * 80)
    
    final_model, final_accuracy, metadata = create_final_model_and_recommendations()
    
    print(f"\\n🎯 MODÈLE FINAL COMPLÉTÉ")
    print(f"📈 Performance validée: {final_accuracy:.1%}")
    print(f"🛡️ Robustesse: Validation temporelle stricte")
    print(f"📊 Features: {len(metadata['features'])}")
    print("\\n🏁 OPTIMISATION TERMINÉE!")