#!/usr/bin/env python3
"""
train_v22_fast.py

VERSION RAPIDE - Pas de grid search, hyperparamètres optimaux fixes

OBJECTIF: Entraîner v2.2 en <2 minutes avec performance maximale
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# ML imports
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, log_loss

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Fast v2.2 training')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_model', required=True)
    parser.add_argument('--output_report', required=True)
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("🚀 FAST v2.2 TRAINING")
    
    # Load data
    df = pd.read_csv(args.data, parse_dates=['Date'])
    feature_cols = [col for col in df.columns if col not in ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FullTimeResult']]
    
    # Split
    train_data = df[df['Date'] <= '2024-05-19'].copy()
    test_data = df[(df['Date'] >= '2024-08-16') & (df['Date'] <= '2025-05-25')].copy()
    
    X_train = train_data[feature_cols].fillna(0.5).values
    X_test = test_data[feature_cols].fillna(0.5).values
    y_train = train_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    y_test = test_data['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2}).values
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Optimal hyperparameters (no search needed)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Training model...")
    rf.fit(X_train, y_train)
    
    # Calibration
    logger.info("Calibrating...")
    model = CalibratedClassifierCV(rf, method='isotonic', cv=3)
    model.fit(X_train, y_train)
    
    # Test predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Results
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)
    
    logger.info(f"🎯 RESULTS:")
    logger.info(f"Accuracy: {accuracy:.1%}")
    logger.info(f"vs v2.1 (54.2%): {(accuracy-0.542)*100:+.1f}pp")
    logger.info(f"vs Rolling 5-feat (48.9%): {(accuracy-0.489)*100:+.1f}pp")
    
    # Feature importance
    importance = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    logger.info("Top 5 features:")
    for i, (feat, imp) in enumerate(importance[:5], 1):
        logger.info(f"  {i}. {feat}: {imp:.3f}")
    
    # Save
    joblib.dump(model, args.output_model)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'feature_importance': importance[:15],
        'beats_v21': accuracy > 0.542,
        'beats_rolling': accuracy > 0.489
    }
    
    with open(args.output_report, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Model saved: {args.output_model}")
    logger.info(f"✅ Report saved: {args.output_report}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())