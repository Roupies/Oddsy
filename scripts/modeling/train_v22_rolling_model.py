#!/usr/bin/env python3
"""
train_v22_rolling_model.py

ÉTAPE 2: Entraînement modèle v2.2 avec TOUTES les features rolling (25)

OBJECTIF:
- Exploiter la richesse du pipeline rolling (25 features vs 5)
- Surpasser v2.1 batch (54.2%) et rolling 5-features (48.9%)
- Atteindre performance "excellent" (>55%)

USAGE:
python scripts/modeling/train_v22_rolling_model.py \
    --data data/processed/v13_complete_with_dates.csv \
    --config config/rolling_v22_full_features.json \
    --output_model models/v22_rolling_full_features.joblib \
    --output_report results/v22_training_report.json

FEATURES v2.2 (27):
- Toutes les features Elo, Form, Goals, xG, Shots, Corners
- Features individuelles (home/away) + différences + normalisées
- H2H score + contexte temporel (matchday)
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
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

# Import rolling components
from scripts.analysis.initialize_historical_state import HistoricalStateBuilder, StateManager
from scripts.analysis.rolling_simulation_engine import FeatureBuilder, DynamicNormalizer
from scripts.analysis.test_period_validator import validate_test_period_safe

class V22RollingModelTrainer:
    """Entraîneur modèle v2.2 avec features rolling complètes"""
    
    def __init__(self, config_file: str):
        self.logger = setup_logging()
        self.config = self._load_config(config_file)
        self.feature_builder = None
        
    def _load_config(self, config_file: str) -> Dict:
        """Charge la configuration v2.2"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.logger.info(f"✅ Configuration loaded: {config['version']}")
        self.logger.info(f"Expected features: {len(config['expected_features_list'])}")
        self.logger.info(f"Target performance: {config['target_performance']}")
        
        return config
    
    def generate_rolling_dataset(self, data_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Génère le dataset complet avec features rolling"""
        self.logger.info("🔬 GENERATING ROLLING DATASET WITH ALL FEATURES")
        self.logger.info("="*60)
        
        # Load data
        df = pd.read_csv(data_file, parse_dates=['Date'])
        
        # Split train/test selon config
        train_start = pd.to_datetime(self.config['training_config']['train_start'])
        train_end = pd.to_datetime(self.config['training_config']['train_end'])
        test_start = pd.to_datetime(self.config['training_config']['test_start'])
        test_end = pd.to_datetime(self.config['training_config']['test_end'])
        
        # Validation période de test
        if not validate_test_period_safe(test_start.strftime('%Y-%m-%d'), 
                                       test_end.strftime('%Y-%m-%d'), 
                                       'v2.1_model', auto_print=False):
            raise ValueError("🚨 Période de test invalide détectée!")
        
        # Train data (pour building features)
        train_data = df[(df['Date'] >= train_start) & (df['Date'] <= train_end)].copy()
        test_data = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)].copy()
        
        self.logger.info(f"📊 Train period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} ({len(train_data)} matches)")
        self.logger.info(f"📊 Test period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')} ({len(test_data)} matches)")
        
        # Initialize rolling state manager pour training
        self.logger.info("🏗️ BUILDING ROLLING STATE FOR TRAINING PERIOD")
        builder = HistoricalStateBuilder(data_file)
        
        # Build state jusqu'à la fin du training
        state_manager = builder.build_initial_state(train_end.strftime('%Y-%m-%d'))
        
        # Initialize feature builder avec toutes les features
        self.feature_builder = FeatureBuilder(
            feature_config=self.config['feature_config'],
            use_dynamic_normalization=self.config['normalization_config']['dynamic_normalization'],
            normalization_method=self.config['normalization_config']['method']
        )
        
        # Generate features pour train data
        self.logger.info("🔬 GENERATING TRAINING FEATURES")
        X_train, feature_names = self._generate_features_for_dataset(train_data, state_manager, "TRAIN")
        y_train = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
        
        # Generate features pour test data
        self.logger.info("🔬 GENERATING TEST FEATURES")
        X_test, _ = self._generate_features_for_dataset(test_data, state_manager, "TEST")
        y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
        
        self.logger.info(f"✅ Dataset generated:")
        self.logger.info(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        self.logger.info(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        self.logger.info(f"  Feature names: {len(feature_names)}")
        
        # Combine pour training
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.hstack([y_train, y_test])
        
        return X_combined, y_combined, feature_names
    
    def _generate_features_for_dataset(self, data: pd.DataFrame, 
                                     state_manager: StateManager, 
                                     phase: str) -> Tuple[np.ndarray, List[str]]:
        """Génère features rolling pour un dataset"""
        features_list = []
        feature_names = None
        
        for idx, row in data.iterrows():
            # Générer toutes les features rolling
            features = self.feature_builder.build_match_features(
                row['HomeTeam'], 
                row['AwayTeam'], 
                state_manager,
                row.to_dict()
            )
            
            if feature_names is None:
                feature_names = list(features.keys())
                self.logger.info(f"  {phase} features ({len(feature_names)}): {feature_names[:5]}...{feature_names[-2:]}")
            
            # Ordre exact des features
            feature_vector = [features.get(name, 0.0) for name in feature_names]
            features_list.append(feature_vector)
            
            # Update state avec résultats réels (pour rolling simulation)
            if phase == "TRAIN":
                # Simuler mise à jour état pour training avec la méthode existante
                state_manager.process_match(row)
            
            if (idx + 1) % 200 == 0:
                self.logger.info(f"    Generated {phase} features for {idx+1} matches...")
        
        return np.array(features_list), feature_names
    
    def train_model(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[Any, Dict]:
        """Entraîne le modèle v2.2 avec hyperparameter tuning"""
        self.logger.info("🚀 TRAINING v2.2 MODEL WITH HYPERPARAMETER TUNING")
        self.logger.info("="*60)
        
        # Split pour validation (dernière partie pour test)
        n_train = int(0.8 * len(X))  # 80% pour training
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        self.logger.info(f"📊 Training split: {len(X_train)} train, {len(X_val)} validation")
        
        # Hyperparameter grid pour RandomForest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Base model
        rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Important pour classes déséquilibrées
        )
        
        # Grid search avec TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        self.logger.info("🔍 Grid search avec TimeSeriesSplit...")
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Entraînement avec grid search
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        self.logger.info(f"✅ Best params: {grid_search.best_params_}")
        self.logger.info(f"✅ Best CV score: {grid_search.best_score_:.4f}")
        
        # Calibration du modèle
        self.logger.info("📐 Calibrating model...")
        calibrated_model = CalibratedClassifierCV(
            best_model, 
            method='isotonic',
            cv=3
        )
        calibrated_model.fit(X_train, y_train)
        
        # Prédictions sur validation
        y_pred = calibrated_model.predict(X_val)
        y_proba = calibrated_model.predict_proba(X_val)
        
        # Métriques
        accuracy = accuracy_score(y_val, y_pred)
        logloss = log_loss(y_val, y_proba)
        report = classification_report(y_val, y_pred, 
                                     target_names=['Home', 'Draw', 'Away'],
                                     output_dict=True)
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, best_model.feature_importances_))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_importance = []
        
        training_results = {
            'model_type': 'CalibratedClassifierCV(RandomForestClassifier)',
            'best_params': grid_search.best_params_,
            'best_cv_score': float(grid_search.best_score_),
            'validation_accuracy': float(accuracy),
            'validation_log_loss': float(logloss),
            'classification_report': report,
            'feature_importance': sorted_importance[:15],  # Top 15 features
            'n_features': len(feature_names),
            'feature_names': feature_names
        }
        
        self.logger.info(f"🎯 Validation Results:")
        self.logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"  Log Loss: {logloss:.4f}")
        
        return calibrated_model, training_results
    
    def run_full_training(self, data_file: str) -> Tuple[Any, Dict]:
        """Exécute l'entraînement complet v2.2"""
        self.logger.info("🚀 RUNNING FULL v2.2 MODEL TRAINING")
        self.logger.info("="*60)
        
        # 1. Generate rolling dataset
        X, y, feature_names = self.generate_rolling_dataset(data_file)
        
        # 2. Train model
        model, training_results = self.train_model(X, y, feature_names)
        
        # 3. Complete results
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'training_results': training_results,
            'dataset_info': {
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[1]),
                'feature_names': feature_names,
                'class_distribution': {
                    'Home': int(np.sum(y == 0)),
                    'Draw': int(np.sum(y == 1)), 
                    'Away': int(np.sum(y == 2))
                }
            }
        }
        
        return model, results

def main():
    parser = argparse.ArgumentParser(description='Train v2.2 rolling model with full features')
    parser.add_argument('--data', required=True, help='Data file path')
    parser.add_argument('--config', required=True, help='v2.2 config file path')
    parser.add_argument('--output_model', required=True, help='Output model path')
    parser.add_argument('--output_report', required=True, help='Output training report JSON')
    
    args = parser.parse_args()
    
    # Train model
    trainer = V22RollingModelTrainer(args.config)
    model, results = trainer.run_full_training(args.data)
    
    # Save model
    joblib.dump(model, args.output_model)
    trainer.logger.info(f"✅ Model saved to: {args.output_model}")
    
    # Save results
    with open(args.output_report, 'w') as f:
        json.dump(results, f, indent=2)
    
    trainer.logger.info(f"✅ Training report saved to: {args.output_report}")
    
    # Final summary
    accuracy = results['training_results']['validation_accuracy']
    target_met = accuracy >= 0.55
    
    trainer.logger.info("🎯 TRAINING COMPLETE!")
    trainer.logger.info(f"Final validation accuracy: {accuracy:.1%}")
    trainer.logger.info(f"Target (>55%): {'✅ MET' if target_met else '❌ NOT MET'}")
    
    return 0 if target_met else 1

if __name__ == "__main__":
    sys.exit(main())