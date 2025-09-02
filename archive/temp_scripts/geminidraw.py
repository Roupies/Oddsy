#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Amélioration des Nuls (v1.6) avec SMOTE
Objectif: Augmenter drastiquement le F1-score pour la classe "Draw"
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.over_sampling import SMOTE
import json
import os

print("🎯 v1.6 - Amélioration des Nuls avec SMOTE")
print("="*60)

# --- 1. CHARGEMENT DES DONNÉES (MÉTHODOLOGIE PROPRE) ---
def load_clean_data():
    """Charge les données avec le split temporel propre."""
    df = pd.read_csv('data/processed/v13_xg_safe_features.csv', parse_dates=['Date'])
    
    # 10 features qui ont bien performé
    optimal_features = [
        'form_diff_normalized', 'elo_diff_normalized', 'h2h_score',
        'matchday_normalized', 'shots_diff_normalized', 'corners_diff_normalized',
        'market_entropy_norm', 'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
    ]
    
    # Split temporel par saison
    train_data = df[df['Date'] <= '2024-05-19'].copy().dropna(subset=optimal_features)
    test_data = df[df['Date'] >= '2024-08-16'].copy().dropna(subset=optimal_features)

    X_train = train_data[optimal_features].fillna(0.5)
    X_test = test_data[optimal_features].fillna(0.5)
    y_train = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
    
    print(f"📊 Données prêtes: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_clean_data()

# --- 2. ENTRAÎNEMENT DU MODÈLE DE BASE (SANS SMOTE) ---
print("\n--- Étape 1: Performance du Modèle de Base (v1.5) ---")

# Utilisation des meilleurs hyperparamètres trouvés précédemment
baseline_params = {
    'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 10,
    'min_samples_split': 15, 'max_features': 'log2', 'class_weight': 'balanced',
    'random_state': 42, 'n_jobs': -1
}

baseline_model = RandomForestClassifier(**baseline_params)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
baseline_f1_draw = f1_score(y_test, y_pred_baseline, average='macro', labels=[1])

print(f"🎯 Accuracy (Baseline): {baseline_accuracy:.4f}")
print("Classification Report (Baseline):")
print(classification_report(y_test, y_pred_baseline, target_names=['Home', 'Draw', 'Away']))


# --- 3. APPLICATION DE SMOTE (LA MAGIE OPÈRE ICI) ---
print("\n--- Étape 2: Création de Données d'Entraînement Enrichies avec SMOTE ---")

# SMOTE ne s'applique QUE sur les données d'entraînement pour éviter toute fuite de données
smote = SMOTE(random_state=42, k_neighbors=5)
print(f"Distribution originale (Train): {y_train.value_counts().to_dict()}")

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Distribution après SMOTE (Train): {y_train_smote.value_counts().to_dict()}")
print("✅ SMOTE a créé des exemples synthétiques pour équilibrer les classes.")


# --- 4. ENTRAÎNEMENT DU MODÈLE AMÉLIORÉ (AVEC SMOTE) ---
print("\n--- Étape 3: Performance du Modèle Amélioré avec Données SMOTE ---")

# On utilise un nouveau modèle avec les mêmes excellents hyperparamètres
smote_model = RandomForestClassifier(**baseline_params)

# On l'entraîne sur les données enrichies
smote_model.fit(X_train_smote, y_train_smote)
y_pred_smote = smote_model.predict(X_test)
smote_accuracy = accuracy_score(y_test, y_pred_smote)
smote_f1_draw = f1_score(y_test, y_pred_smote, average='macro', labels=[1])

print(f"🎯 Accuracy (SMOTE): {smote_accuracy:.4f}")
print("Classification Report (SMOTE):")
print(classification_report(y_test, y_pred_smote, target_names=['Home', 'Draw', 'Away']))

# --- 5. COMPARAISON FINALE ---
print("\n🏆🏆🏆 COMPARAISON FINALE 🏆🏆🏆")
print("="*60)
print(f"| Métrique          | Baseline (v1.5) | Modèle SMOTE (v1.6) | Amélioration     |")
print(f"|-------------------|-----------------|---------------------|------------------|")
print(f"| Accuracy Générale | {baseline_accuracy:15.2%} | {smote_accuracy:19.2%} | {smote_accuracy - baseline_accuracy:+.2%}            |")
print(f"| F1-Score (Draws)  | {baseline_f1_draw:15.2%} | {smote_f1_draw:19.2%} | {smote_f1_draw - baseline_f1_draw:+.2%} 🔥🔥🔥 |")
print("="*60)

if smote_f1_draw > baseline_f1_draw * 1.5 and smote_f1_draw > 0.20:
    print("\n🎉 Victoire Totale ! Le F1-score pour les nuls a explosé. C'est une avancée majeure.")
else:
    print("\n🤔 Résultat mitigé. SMOTE a aidé, mais il y a encore du travail pour maîtriser les nuls.")
