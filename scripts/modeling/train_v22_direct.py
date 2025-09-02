#!/usr/bin/env python3
"""
train_v22_direct.py

ÉTAPE 2 SIMPLIFIÉE: Entraînement direct modèle v2.2 sur features existantes

STRATÉGIE EFFICACE:
- Utiliser directement les 27 features de v13_xg_safe_features.csv
- Pas besoin de recalculer rolling - déjà fait!
- Focus sur hyperparameter tuning pour maximiser performance

USAGE:
python scripts/modeling/train_v22_direct.py \
    --data data/processed/v13_xg_safe_features.csv \
    --output_model models/v22_direct_27features.joblib \
    --output_report results/v22_direct_training_report.json

TARGET: Dépasser v2.1 (54.2%) et rolling 5-features (48.9%)
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# ML imports
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging
from scripts.analysis.test_period_validator import validate_test_period_safe

class V22DirectTrainer:
    """Entraîneur modèle v2.2 direct sur features existantes"""
    
    def __init__(self):
        self.logger = setup_logging()
        
    def load_and_split_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Charge et split les données avec validation anti-leakage"""
        self.logger.info("📊 LOADING AND SPLITTING DATA")
        self.logger.info("="*50)
        
        # Load data
        df = pd.read_csv(data_file, parse_dates=['Date'])
        
        # Get feature columns (tout sauf metadata et target)
        feature_cols = [col for col in df.columns if col not in ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult']]
        
        self.logger.info(f"Total features available: {len(feature_cols)}")
        self.logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        self.logger.info(f"Total matches: {len(df)}")
        
        # Split temporel safe
        train_end = pd.to_datetime('2024-05-19')  # Fin training v2.1
        test_start = pd.to_datetime('2024-08-16')  # Début saison 2024-2025
        test_end = pd.to_datetime('2025-05-25')   # Fin saison 2024-2025
        
        # Validation période
        if not validate_test_period_safe(test_start.strftime('%Y-%m-%d'), 
                                       test_end.strftime('%Y-%m-%d'), 
                                       'v2.1_model', auto_print=False):
            raise ValueError("🚨 Période de test invalide!")
        
        # Split
        train_data = df[df['Date'] <= train_end].copy()
        test_data = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)].copy()
        
        self.logger.info(f"📊 Train: {len(train_data)} matches (until {train_end.strftime('%Y-%m-%d')})")
        self.logger.info(f"📊 Test: {len(test_data)} matches ({test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')})")
        
        # Prepare features and targets
        X_train = train_data[feature_cols].values
        X_test = test_data[feature_cols].values
        
        # Target encoding: H->0, D->1, A->2
        y_train = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
        y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
        
        # Check for missing values
        train_missing = np.isnan(X_train).sum()
        test_missing = np.isnan(X_test).sum()
        
        if train_missing > 0 or test_missing > 0:
            self.logger.warning(f"⚠️ Missing values detected: Train {train_missing}, Test {test_missing}")
            # Fill avec 0.5 (valeur neutre pour features normalisées)
            X_train = np.nan_to_num(X_train, nan=0.5)
            X_test = np.nan_to_num(X_test, nan=0.5)
        
        self.logger.info(f"✅ Data prepared:")
        self.logger.info(f"  X_train shape: {X_train.shape}")
        self.logger.info(f"  X_test shape: {X_test.shape}")
        self.logger.info(f"  Feature list: {feature_cols[:5]}...{feature_cols[-2:]}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_and_tune_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           feature_names: List[str]) -> Tuple[Any, Dict]:
        """Entraîne avec hyperparameter tuning intensif"""
        self.logger.info("🚀 HYPERPARAMETER TUNING & TRAINING")
        self.logger.info("="*50)
        
        # Split train/validation pour tuning
        n_val = int(0.2 * len(X_train))  # 20% pour validation
        X_tune, X_val = X_train[:-n_val], X_train[-n_val:]
        y_tune, y_val = y_train[:-n_val], y_train[-n_val:]
        
        self.logger.info(f"Tuning split: {len(X_tune)} tune, {len(X_val)} validation")
        
        # Hyperparameter grid élargi
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4, 6],
            'max_features': ['sqrt', 'log2', 0.8],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        # Base model
        rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )
        
        # Grid search avec TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        self.logger.info(f"🔍 Grid search: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])} combinations...")
        
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Training
        grid_search.fit(X_tune, y_tune)
        
        best_model = grid_search.best_estimator_
        self.logger.info(f"✅ Best params: {grid_search.best_params_}")
        self.logger.info(f"✅ Best CV score: {grid_search.best_score_:.4f}")
        
        # Model calibration
        self.logger.info("📐 Calibrating model for better probabilities...")
        calibrated_model = CalibratedClassifierCV(
            best_model, 
            method='isotonic',
            cv=3
        )
        calibrated_model.fit(X_tune, y_tune)
        
        # Validation predictions
        y_val_pred = calibrated_model.predict(X_val)
        y_val_proba = calibrated_model.predict_proba(X_val)
        
        # Metrics
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_logloss = log_loss(y_val, y_val_proba)
        val_report = classification_report(y_val, y_val_pred, 
                                         target_names=['Home', 'Draw', 'Away'],
                                         output_dict=True)
        
        # Feature importance
        feature_importance = dict(zip(feature_names, best_model.feature_importances_))
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"🎯 Validation Results:")
        self.logger.info(f"  Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        self.logger.info(f"  Log Loss: {val_logloss:.4f}")
        self.logger.info(f"🏆 Top 5 features:")
        for i, (feat, imp) in enumerate(sorted_importance[:5], 1):
            self.logger.info(f"  {i}. {feat}: {imp:.4f}")
        
        training_results = {
            'model_type': 'CalibratedClassifierCV(RandomForestClassifier)',
            'best_params': grid_search.best_params_,
            'best_cv_score': float(grid_search.best_score_),
            'validation_accuracy': float(val_accuracy),
            'validation_log_loss': float(val_logloss),
            'classification_report': val_report,
            'feature_importance': sorted_importance,
            'n_features': len(feature_names)
        }
        
        return calibrated_model, training_results
    
    def evaluate_on_test(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Évaluation finale sur vraies données non vues"""
        self.logger.info("🎯 FINAL TEST EVALUATION (UNSEEN 2024-2025 DATA)")
        self.logger.info("="*50)
        
        # Predictions
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        
        # Metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_logloss = log_loss(y_test, y_test_proba)
        test_report = classification_report(y_test, y_test_pred, 
                                          target_names=['Home', 'Draw', 'Away'],
                                          output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        self.logger.info(f"🎯 FINAL TEST RESULTS:")
        self.logger.info(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        self.logger.info(f"  Log Loss: {test_logloss:.4f}")
        
        # Performance vs baselines
        self.logger.info(f"📊 vs Baselines:")
        self.logger.info(f"  Random (33.3%): {(test_accuracy-0.333)*100:+.1f}pp")
        self.logger.info(f"  Majority (43.6%): {(test_accuracy-0.436)*100:+.1f}pp")
        self.logger.info(f"  v2.1 target (54.2%): {(test_accuracy-0.542)*100:+.1f}pp")
        self.logger.info(f"  Rolling 5-feat (48.9%): {(test_accuracy-0.489)*100:+.1f}pp")
        
        # Target assessment
        if test_accuracy >= 0.55:
            self.logger.info("🎉 EXCELLENT MODEL achieved (>55%)!")
        elif test_accuracy >= 0.52:
            self.logger.info("✅ GOOD MODEL achieved (>52%)")
        elif test_accuracy >= 0.45:
            self.logger.info("⚠️ Above baselines but needs improvement")
        else:
            self.logger.info("❌ Below expected performance")
        
        test_results = {
            'test_accuracy': float(test_accuracy),
            'test_log_loss': float(test_logloss),
            'classification_report': test_report,
            'confusion_matrix': cm.tolist(),
            'n_test_samples': int(len(y_test)),
            'baselines_comparison': {
                'random_33': float((test_accuracy-0.333)*100),
                'majority_436': float((test_accuracy-0.436)*100),
                'v21_target_542': float((test_accuracy-0.542)*100),
                'rolling_5feat_489': float((test_accuracy-0.489)*100)
            }
        }
        
        return test_results
    
    def run_full_training(self, data_file: str) -> Tuple[Any, Dict]:
        """Pipeline complet d'entraînement v2.2"""
        self.logger.info("🚀 v2.2 DIRECT TRAINING PIPELINE")
        self.logger.info("="*60)
        
        # 1. Load and split data
        X_train, X_test, y_train, y_test, feature_names = self.load_and_split_data(data_file)
        
        # 2. Train and tune model
        model, training_results = self.train_and_tune_model(X_train, y_train, feature_names)
        
        # 3. Final test evaluation
        test_results = self.evaluate_on_test(model, X_test, y_test)
        
        # 4. Complete results
        results = {
            'timestamp': datetime.now().isoformat(),
            'version': 'v2.2_direct_27features',
            'data_file': data_file,
            'feature_names': feature_names,
            'training_results': training_results,
            'test_results': test_results,
            'summary': {
                'final_test_accuracy': test_results['test_accuracy'],
                'target_achieved': test_results['test_accuracy'] >= 0.55,
                'beats_v21': test_results['test_accuracy'] > 0.542,
                'beats_rolling_5feat': test_results['test_accuracy'] > 0.489
            }
        }
        
        return model, results

def main():
    parser = argparse.ArgumentParser(description='Train v2.2 direct on existing features')
    parser.add_argument('--data', required=True, help='Data file path')
    parser.add_argument('--output_model', required=True, help='Output model path')
    parser.add_argument('--output_report', required=True, help='Output report JSON')
    
    args = parser.parse_args()
    
    # Train model
    trainer = V22DirectTrainer()
    model, results = trainer.run_full_training(args.data)
    
    # Save model and results
    joblib.dump(model, args.output_model)
    trainer.logger.info(f"✅ Model saved: {args.output_model}")
    
    with open(args.output_report, 'w') as f:
        json.dump(results, f, indent=2)
    trainer.logger.info(f"✅ Report saved: {args.output_report}")
    
    # Final status
    accuracy = results['summary']['final_test_accuracy']
    target_met = results['summary']['target_achieved']
    
    trainer.logger.info("🎯 TRAINING COMPLETE!")
    trainer.logger.info(f"Final accuracy: {accuracy:.1%}")
    trainer.logger.info(f"Target (>55%): {'✅ ACHIEVED' if target_met else '❌ NOT MET'}")
    
    return 0 if target_met else 1

if __name__ == "__main__":
    sys.exit(main())