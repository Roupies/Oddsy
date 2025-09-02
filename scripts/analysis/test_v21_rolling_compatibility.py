#!/usr/bin/env python3
"""
test_v21_rolling_compatibility.py

ÉTAPE 1: Validation pipeline rolling compatibilité modèle v2.1

OBJECTIF:
- Configurer pipeline rolling pour générer exactement les 5 features du modèle v2.1
- Valider que les performances restent stables (54.2% ± 0.5%)
- Prouver que rolling peut remplacer batch directement

USAGE:
python scripts/analysis/test_v21_rolling_compatibility.py \
    --data data/processed/v13_complete_with_dates.csv \
    --model models/clean_xg_model_traditional_baseline_2025_08_31_235028.joblib \
    --config config/rolling_v21_compatible.json \
    --test_season 2023-2024 \
    --output results/v21_rolling_compatibility_test.json

FEATURES V2.1 (5):
1. elo_diff_normalized
2. form_diff_normalized  
3. h2h_score
4. matchday_normalized
5. corners_diff_normalized
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
from sklearn.metrics import accuracy_score, classification_report, log_loss

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

# Import rolling components
from initialize_historical_state import HistoricalStateBuilder, StateManager
from rolling_simulation_engine import FeatureBuilder, DynamicNormalizer

class V21CompatibilityTester:
    """Testeur de compatibilité rolling pipeline avec modèle v2.1"""
    
    def __init__(self, config_file: str):
        self.logger = setup_logging()
        self.config = self._load_config(config_file)
        self.feature_builder = None
        self.model = None
        
    def _load_config(self, config_file: str) -> Dict:
        """Charge la configuration v2.1"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.logger.info(f"✅ Configuration loaded: {config['version']}")
        self.logger.info(f"Target model: {config['target_model']}")
        self.logger.info(f"Expected features: {config['expected_features']}")
        
        return config
    
    def load_model(self) -> None:
        """Charge le modèle v2.1"""
        model_path = self.config['target_model']
        self.model = joblib.load(model_path)
        
        self.logger.info(f"✅ Model loaded: {type(self.model).__name__}")
        if hasattr(self.model, 'feature_names_in_'):
            self.logger.info(f"Model expects {len(self.model.feature_names_in_)} features:")
            for i, feat in enumerate(self.model.feature_names_in_, 1):
                self.logger.info(f"  {i}. {feat}")
    
    def initialize_state_manager(self, data_file: str, season_start: str) -> StateManager:
        """Initialize le state manager rolling"""
        self.logger.info("🏗️ INITIALIZING ROLLING STATE MANAGER")
        
        builder = HistoricalStateBuilder(data_file)
        state_manager = builder.build_initial_state(season_start)
        
        # Initialize feature builder with v2.1 compatible config
        self.feature_builder = FeatureBuilder(
            feature_config=self.config['feature_config'],
            use_dynamic_normalization=self.config['normalization_config']['dynamic_normalization'],
            normalization_method=self.config['normalization_config']['method']
        )
        
        self.logger.info("✅ State manager and feature builder initialized")
        return state_manager
    
    def test_feature_generation(self, state_manager: StateManager, 
                              test_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Test la génération des 5 features v2.1 compatibles"""
        self.logger.info("🔬 TESTING V2.1 FEATURE GENERATION")
        
        features_list = []
        feature_names = self.config['exact_features_required']
        
        for idx, row in test_data.iterrows():
            # Utiliser la méthode spécialisée v2.1
            features = self.feature_builder.build_v21_compatible_features(
                row['HomeTeam'], 
                row['AwayTeam'], 
                state_manager,
                row.to_dict()
            )
            
            # Valider que les features correspondent exactement
            assert set(features.keys()) == set(feature_names), \
                f"Feature mismatch: got {set(features.keys())}, expected {set(feature_names)}"
            
            # Ordre exact requis par le modèle
            feature_vector = [features[name] for name in feature_names]
            features_list.append(feature_vector)
            
            if idx % 50 == 0:
                self.logger.info(f"  Generated features for {idx+1} matches...")
        
        X_features = np.array(features_list)
        self.logger.info(f"✅ Generated {X_features.shape[0]} feature vectors with {X_features.shape[1]} features each")
        
        return X_features, feature_names
    
    def run_compatibility_test(self, data_file: str, season_start: str, 
                             season_end: str) -> Dict[str, Any]:
        """Exécute le test complet de compatibilité"""
        self.logger.info("🎯 RUNNING V2.1 COMPATIBILITY TEST")
        self.logger.info("="*60)
        
        # 1. Load model
        self.load_model()
        
        # 2. Load and split data
        df = pd.read_csv(data_file, parse_dates=['Date'])
        
        # Split historique (pour state) vs test
        season_start_date = pd.to_datetime(season_start)
        season_end_date = pd.to_datetime(season_end)
        
        test_data = df[(df['Date'] >= season_start_date) & (df['Date'] <= season_end_date)].copy()
        
        self.logger.info(f"📊 Test data: {len(test_data)} matches ({season_start} to {season_end})")
        
        # 3. Initialize rolling state manager
        state_manager = self.initialize_state_manager(data_file, season_start)
        
        # 4. Generate features using rolling pipeline
        X_rolling, feature_names = self.test_feature_generation(state_manager, test_data)
        
        # 5. Get true labels
        y_true = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
        
        # 6. Make predictions
        self.logger.info("🎯 MAKING PREDICTIONS WITH V2.1 MODEL")
        y_pred = self.model.predict(X_rolling)
        y_proba = self.model.predict_proba(X_rolling)
        
        # 7. Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        logloss = log_loss(y_true, y_proba)
        
        self.logger.info(f"🎯 RESULTS:")
        self.logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"  Log Loss: {logloss:.4f}")
        
        # 8. Validate against target
        target_min = self.config['validation_target']['accuracy_min']
        target_max = self.config['validation_target']['accuracy_max']
        
        success = target_min <= accuracy <= target_max
        self.logger.info(f"🎯 VALIDATION: {'✅ PASS' if success else '❌ FAIL'}")
        self.logger.info(f"  Target range: {target_min:.1%} - {target_max:.1%}")
        self.logger.info(f"  Actual: {accuracy:.1%}")
        
        # 9. Detailed report
        report = classification_report(y_true, y_pred, 
                                     target_names=['Home', 'Draw', 'Away'],
                                     output_dict=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'test_period': f"{season_start} to {season_end}",
            'n_matches': len(test_data),
            'feature_names': feature_names,
            'accuracy': float(accuracy),
            'log_loss': float(logloss),
            'target_validation': {
                'passed': success,
                'target_range': [target_min, target_max],
                'actual_accuracy': float(accuracy)
            },
            'classification_report': report,
            'sample_features': {
                'first_match': X_rolling[0].tolist(),
                'feature_stats': {
                    'mean': np.mean(X_rolling, axis=0).tolist(),
                    'std': np.std(X_rolling, axis=0).tolist()
                }
            }
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Test rolling pipeline compatibility with v2.1 model')
    parser.add_argument('--data', required=True, help='Data file path')
    parser.add_argument('--model', required=True, help='v2.1 Model file path') 
    parser.add_argument('--config', required=True, help='v2.1 config file path')
    parser.add_argument('--season_start', required=True, help='Test season start (YYYY-MM-DD)')
    parser.add_argument('--season_end', required=True, help='Test season end (YYYY-MM-DD)')
    parser.add_argument('--output', required=True, help='Output results JSON')
    
    args = parser.parse_args()
    
    # Run compatibility test
    tester = V21CompatibilityTester(args.config)
    results = tester.run_compatibility_test(
        args.data, 
        args.season_start, 
        args.season_end
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    tester.logger.info(f"✅ Results saved to: {args.output}")
    
    # Return success code
    success = results['target_validation']['passed']
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())