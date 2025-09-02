#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 Benchmark RAPIDE - 10 Features Nettoyées + Gestion Draws
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import json, os

print("🎯 Benchmark RAPIDE - 10 Features Optimales + Draws")
print("="*50)

# ----------------------
# Charger données
df = pd.read_csv('data/processed/v13_xg_safe_features.csv', parse_dates=['Date'])

# 10 features optimales
optimal_features = [
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

# Split temporel rapide
valid_data = df.dropna(subset=optimal_features)
train_data = valid_data[valid_data['Date'] <= '2024-05-19'].copy()
test_data = valid_data[valid_data['Date'] >= '2024-08-16'].copy()

X_train = train_data[optimal_features].fillna(0.5).values
X_test = test_data[optimal_features].fillna(0.5).values
y_train = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values

print(f"📊 Données: {len(df)} matches, Train: {len(train_data)}, Test: {len(test_data)}")
print(f"✅ Données prêtes: X_train {X_train.shape}, X_test {X_test.shape}")

# ----------------------
# Fonction utilitaire
def benchmark_model(model, X_train, y_train, X_test, y_test, name):
    start_time = datetime.now()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Home','Draw','Away'])
    cm = confusion_matrix(y_test, y_pred)
    time_taken = (datetime.now() - start_time).total_seconds()
    print(f"\n{name} - Accuracy: {score:.4f} ({score*100:.2f}%), Temps: {time_taken:.0f}s")
    print(report)
    return score, time_taken

# ----------------------
# 🌲 RandomForest pondéré + calibration
print("🌲 RANDOMFOREST - Pondéré + Calibration")
rf_params = {
    'n_estimators': [200, 300],
    'max_depth': [20, None],
    'min_samples_split': [5, 10],
    'max_features': ['sqrt'],
    'class_weight': ['balanced']
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
tscv = TimeSeriesSplit(n_splits=3)

rf_grid = GridSearchCV(rf_base, rf_params, cv=tscv, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

rf_best = CalibratedClassifierCV(rf_grid.best_estimator_, method='isotonic', cv=3)
rf_score, rf_time = benchmark_model(rf_best, X_train, y_train, X_test, y_test, "RandomForest Pondéré+Calibration")

# ----------------------
# ⚡ XGBoost pondéré Draw
print("⚡ XGBOOST - Scale_pos_weight Draw")
unique, counts = np.unique(y_train, return_counts=True)
class_counts = dict(zip(unique, counts))
draw_weight = (class_counts[0] + class_counts[2]) / class_counts[1]

xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    scale_pos_weight=draw_weight,
    n_jobs=-1,
    eval_metric='mlogloss',
    verbosity=0
)

xgb_best = CalibratedClassifierCV(xgb_model, method='isotonic', cv=3)
xgb_score, xgb_time = benchmark_model(xgb_best, X_train, y_train, X_test, y_test, "XGBoost Pondéré Draw")

# ----------------------
# 💡 LightGBM baseline
print("�� LIGHTGBM - Baseline")
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
    force_row_wise=True,
    feature_name='auto'
)

lgb_best = CalibratedClassifierCV(lgb_model, method='isotonic', cv=3)
lgb_score, lgb_time = benchmark_model(lgb_best, X_train, y_train, X_test, y_test, "LightGBM Baseline")

# ----------------------
# 🏆 Résultats finaux
results = {'RandomForest': rf_score, 'XGBoost': xgb_score, 'LightGBM': lgb_score}
times = {'RandomForest': rf_time, 'XGBoost': xgb_time, 'LightGBM': lgb_time}

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\n🏆 CLASSEMENT FINAL:")
for i, (algo, score) in enumerate(sorted_results, 1):
    print(f"{i}. {algo:12}: {score:.4f} ({score*100:.2f}%) - {times[algo]:.0f}s")

best_algo, best_score = sorted_results[0]
print(f"\n🥇 GAGNANT: {best_algo} avec {best_score:.1%}")

# ----------------------
# Graphique rapide
fig, ax = plt.subplots(figsize=(10,6))
algos = list(results.keys())
scores = list(results.values())
colors = ['gold' if s == max(scores) else 'lightblue' for s in scores]
bars = ax.bar(algos, scores, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Accuracy')
ax.set_title('Benchmark Final - 10 Features Nettoyées + Draws')
ax.set_ylim(0.45, 0.58)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{score:.1%}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()

# ----------------------
# Sauvegarde résultats
timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
os.makedirs('reports', exist_ok=True)
output_file = f'reports/fast_benchmark_draws_{timestamp}.json'

final_results = {
    'timestamp': timestamp,
    'experiment': 'fast_algorithm_benchmark_draws',
    'features_count': 10,
    'results': {algo: float(score) for algo, score in results.items()},
    'training_times': {algo: float(time) for algo, time in times.items()},
    'best_algorithm': best_algo,
    'best_score': float(best_score)
}

with open(output_file, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\n✅ Résultats sauvegardés: {output_file}")

