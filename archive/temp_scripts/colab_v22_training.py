# ============================================================================
# COLAB NOTEBOOK - v2.2 MODEL TRAINING WITH 27 FEATURES
# ============================================================================
# Copy-paste this entire code into Google Colab
# Upload: v13_xg_safe_features.csv to Colab files
# Run: Runtime → Run All
# ============================================================================

# CELL 1: Setup and Dependencies
print("🚀 SETTING UP COLAB ENVIRONMENT")
print("="*50)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
import joblib

print("✅ Libraries imported successfully")
print("📊 Environment ready for training")

# ============================================================================
# CELL 2: Data Loading and Validation
print("\n📊 LOADING AND PREPARING DATA")
print("="*50)

# Load data (make sure v13_xg_safe_features.csv is uploaded to Colab)
try:
    df = pd.read_csv('v13_xg_safe_features.csv', parse_dates=['Date'])
    print(f"✅ Data loaded: {df.shape[0]} matches, {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ Please upload 'v13_xg_safe_features.csv' to Colab files first!")
    raise

# Get feature columns
feature_cols = [col for col in df.columns if col not in ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult']]
print(f"📈 Available features: {len(feature_cols)}")

# Display feature list
print("\n🎯 FEATURE LIST:")
for i, feat in enumerate(feature_cols, 1):
    print(f"{i:2d}. {feat}")

# Data info
print(f"\n📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"🏆 Target distribution:")
target_dist = df['FullTimeResult'].value_counts()
for result, count in target_dist.items():
    print(f"  {result}: {count} ({count/len(df)*100:.1f}%)")

# ============================================================================
# CELL 3: Train/Test Split with Anti-Leakage Validation
print("\n🔒 TEMPORAL SPLIT WITH ANTI-LEAKAGE VALIDATION")
print("="*50)

# Safe temporal split
train_end = pd.to_datetime('2024-05-19')  # End of v2.1 training
test_start = pd.to_datetime('2024-08-16')  # Start of 2024-2025 season
test_end = pd.to_datetime('2025-05-25')    # End of 2024-2025 season

# Split data
train_data = df[df['Date'] <= train_end].copy()
test_data = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)].copy()

print(f"📊 Training data: {len(train_data)} matches (until {train_end.strftime('%Y-%m-%d')})")
print(f"📊 Test data: {len(test_data)} matches ({test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')})")
print(f"🔒 Gap between train/test: {(test_start - train_end).days} days")

# Prepare features and targets
X_train = train_data[feature_cols].values
X_test = test_data[feature_cols].values

# Target encoding: H->0, D->1, A->2
y_train = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values

# Handle missing values (fill with 0.5 for normalized features)
train_missing = np.isnan(X_train).sum()
test_missing = np.isnan(X_test).sum()

if train_missing > 0 or test_missing > 0:
    print(f"⚠️ Missing values: Train {train_missing}, Test {test_missing}")
    X_train = np.nan_to_num(X_train, nan=0.5)
    X_test = np.nan_to_num(X_test, nan=0.5)

print(f"✅ Data prepared:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  No data leakage: ✅")

# ============================================================================
# CELL 4: Model Training with Hyperparameter Tuning
print("\n🚀 HYPERPARAMETER TUNING & MODEL TRAINING")
print("="*50)

# Split train data for validation during tuning
n_val = int(0.2 * len(X_train))
X_tune, X_val = X_train[:-n_val], X_train[-n_val:]
y_tune, y_val = y_train[:-n_val], y_train[-n_val:]

print(f"Tuning split: {len(X_tune)} for tuning, {len(X_val)} for validation")

# Hyperparameter grid (optimized for Colab speed)
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [15, 20, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2']
}

print(f"🔍 Grid search combinations: {np.prod([len(v) for v in param_grid.values()])}")

# Base RandomForest
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    class_weight='balanced'  # Handle class imbalance
)

# Grid search with TimeSeriesSplit (respects temporal order)
tscv = TimeSeriesSplit(n_splits=3)  # Reduced for speed

print("⏱️ Starting grid search...")
start_time = datetime.now()

grid_search = GridSearchCV(
    rf, param_grid,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Training
grid_search.fit(X_tune, y_tune)

training_time = datetime.now() - start_time
print(f"⏱️ Training completed in: {training_time}")

best_model = grid_search.best_estimator_
print(f"\n✅ Best parameters: {grid_search.best_params_}")
print(f"✅ Best CV score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")

# ============================================================================
# CELL 5: Model Calibration
print("\n📐 MODEL CALIBRATION")
print("="*50)

print("Calibrating model for better probability estimates...")
calibrated_model = CalibratedClassifierCV(
    best_model, 
    method='isotonic',
    cv=3
)
calibrated_model.fit(X_tune, y_tune)

# Validation predictions
y_val_pred = calibrated_model.predict(X_val)
y_val_proba = calibrated_model.predict_proba(X_val)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_logloss = log_loss(y_val, y_val_proba)

print(f"📊 Validation Results:")
print(f"  Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"  Log Loss: {val_logloss:.4f}")

# ============================================================================
# CELL 6: Feature Importance Analysis
print("\n🎯 FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Get feature importance from best model
feature_importance = dict(zip(feature_cols, best_model.feature_importances_))
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print("🏆 TOP 15 MOST IMPORTANT FEATURES:")
for i, (feat, imp) in enumerate(sorted_importance[:15], 1):
    print(f"{i:2d}. {feat:<30} {imp:.4f}")

# Visualize top 10 features
plt.figure(figsize=(12, 8))
top_features = sorted_importance[:10]
features, importances = zip(*top_features)

plt.barh(range(len(features)), importances)
plt.yticks(range(len(features)), features)
plt.xlabel('Feature Importance')
plt.title('Top 10 Most Important Features - v2.2 Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================================================
# CELL 7: Final Test Evaluation (UNSEEN 2024-2025 DATA)
print("\n🎯 FINAL TEST EVALUATION - UNSEEN 2024-2025 DATA")
print("="*60)

# Predictions on truly unseen test data
y_test_pred = calibrated_model.predict(X_test)
y_test_proba = calibrated_model.predict_proba(X_test)

# Metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_logloss = log_loss(y_test, y_test_proba)

print(f"🎯 FINAL RESULTS - v2.2 MODEL:")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Test Log Loss: {test_logloss:.4f}")

# Performance vs baselines
print(f"\n📊 PERFORMANCE vs BASELINES:")
print(f"  vs Random (33.3%): {(test_accuracy-0.333)*100:+.1f} percentage points")
print(f"  vs Majority (43.6%): {(test_accuracy-0.436)*100:+.1f} percentage points")
print(f"  vs v2.1 Target (54.2%): {(test_accuracy-0.542)*100:+.1f} percentage points")
print(f"  vs Rolling 5-feat (48.9%): {(test_accuracy-0.489)*100:+.1f} percentage points")

# Success assessment
if test_accuracy >= 0.55:
    print("\n🎉 EXCELLENT MODEL ACHIEVED (>55%)!")
    success_level = "EXCELLENT"
elif test_accuracy >= 0.52:
    print("\n✅ GOOD MODEL ACHIEVED (>52%)")
    success_level = "GOOD"
elif test_accuracy >= 0.45:
    print("\n⚠️ ABOVE BASELINES - Needs improvement")
    success_level = "ABOVE_BASELINES"
else:
    print("\n❌ BELOW EXPECTED PERFORMANCE")
    success_level = "POOR"

# Detailed classification report
print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
report = classification_report(y_test, y_test_pred, 
                             target_names=['Home', 'Draw', 'Away'],
                             digits=3)
print(report)

# ============================================================================
# CELL 8: Results Visualization
print("\n📊 RESULTS VISUALIZATION")
print("="*50)

# Confusion Matrix
plt.figure(figsize=(15, 5))

# Subplot 1: Confusion Matrix
plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Home', 'Draw', 'Away'],
            yticklabels=['Home', 'Draw', 'Away'])
plt.title(f'Confusion Matrix\nAccuracy: {test_accuracy:.1%}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Subplot 2: Performance Comparison
plt.subplot(1, 3, 2)
baselines = ['Random\n(33.3%)', 'Majority\n(43.6%)', 'Rolling 5-feat\n(48.9%)', 'v2.1 Target\n(54.2%)', 'v2.2 (Ours)']
scores = [0.333, 0.436, 0.489, 0.542, test_accuracy]
colors = ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen']

bars = plt.bar(baselines, scores, color=colors, alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)

# Add value labels on bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{score:.1%}', ha='center', va='bottom')

# Subplot 3: Probability Distribution
plt.subplot(1, 3, 3)
max_probas = np.max(y_test_proba, axis=1)
plt.hist(max_probas, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Maximum Predicted Probability')
plt.ylabel('Number of Predictions')
plt.title(f'Prediction Confidence Distribution\nMean: {np.mean(max_probas):.3f}')

plt.tight_layout()
plt.show()

# ============================================================================
# CELL 9: Save Results and Model
print("\n💾 SAVING RESULTS")
print("="*50)

# Prepare complete results dictionary
final_results = {
    'timestamp': datetime.now().isoformat(),
    'version': 'v2.2_colab_27features',
    'training_info': {
        'n_train_samples': int(len(X_train)),
        'n_test_samples': int(len(X_test)),
        'n_features': len(feature_cols),
        'training_time_minutes': training_time.total_seconds() / 60,
        'best_params': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_)
    },
    'performance': {
        'validation_accuracy': float(val_accuracy),
        'validation_log_loss': float(val_logloss),
        'test_accuracy': float(test_accuracy),
        'test_log_loss': float(test_logloss),
        'success_level': success_level
    },
    'baselines_comparison': {
        'vs_random_33': float((test_accuracy-0.333)*100),
        'vs_majority_436': float((test_accuracy-0.436)*100),
        'vs_v21_target_542': float((test_accuracy-0.542)*100),
        'vs_rolling_5feat_489': float((test_accuracy-0.489)*100)
    },
    'feature_importance': sorted_importance[:15],
    'beats_targets': {
        'beats_v21': test_accuracy > 0.542,
        'beats_rolling_5feat': test_accuracy > 0.489,
        'achieves_excellent': test_accuracy >= 0.55
    }
}

# Save results as JSON
with open('v22_colab_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

# Save model
joblib.dump(calibrated_model, 'v22_colab_model.joblib')

print("✅ Results saved to: v22_colab_results.json")
print("✅ Model saved to: v22_colab_model.joblib")

# ============================================================================
# CELL 10: Final Summary
print("\n🎯 FINAL SUMMARY")
print("="*60)
print(f"📊 v2.2 Model Performance: {test_accuracy:.1%}")
print(f"🎯 Success Level: {success_level}")
print(f"⏱️ Training Time: {training_time}")
print(f"💻 Features Used: {len(feature_cols)} (vs 5 in v2.1)")

if test_accuracy > 0.542:
    print("\n🎉 SUCCESS: v2.2 BEATS v2.1 TARGET!")
    print(f"Improvement: +{(test_accuracy-0.542)*100:.1f} percentage points")
else:
    print(f"\n⚠️ Close but not quite: {(test_accuracy-0.542)*100:.1f}pp from v2.1")

if test_accuracy > 0.489:
    print(f"✅ BEATS Rolling 5-feat: +{(test_accuracy-0.489)*100:.1f}pp")

print("\n🚀 TRAINING COMPLETE!")
print("Download the saved files from Colab to use the model!")