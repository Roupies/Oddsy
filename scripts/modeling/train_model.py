import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import json
import joblib
from datetime import datetime
import os

def load_ml_data():
    """Load prepared ML data"""
    print("Loading ML data...")
    
    # Load features and target
    X = pd.read_csv('data/processed/X_features.csv')
    y = pd.read_csv('data/processed/y_target.csv')['target']
    
    # Load configuration
    with open('config/features.json', 'r') as f:
        feature_names = json.load(f)
    
    with open('config/target_mapping.json', 'r') as f:
        target_info = json.load(f)
    
    print(f"Features loaded: {X.shape}")
    print(f"Target loaded: {y.shape}")
    
    return X, y, feature_names, target_info

def create_time_series_splits(X, y, n_splits=5):
    """Create time series cross-validation splits"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"Time Series CV with {n_splits} splits:")
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        train_size = len(train_idx)
        val_size = len(val_idx)
        print(f"Fold {i+1}: Train={train_size}, Val={val_size}")
    
    return tscv

def train_random_forest(X, y, target_info):
    """Train Random Forest with Time Series Cross-Validation"""
    print("\n=== Training Random Forest ===")
    
    # Random Forest parameters
    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
        'class_weight': 'balanced'  # Handle class imbalance
    }
    
    print(f"RF Parameters: {rf_params}")
    
    # Create model
    rf_model = RandomForestClassifier(**rf_params)
    
    # Time Series Cross-Validation
    tscv = create_time_series_splits(X, y, n_splits=5)
    
    print("\nPerforming Time Series Cross-Validation...")
    cv_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
    
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model on all data
    print("\nTraining final model on complete dataset...")
    rf_model.fit(X, y)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Most Important Features:")
    for idx, row in feature_importance.head().iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Training accuracy
    train_predictions = rf_model.predict(X)
    train_accuracy = accuracy_score(y, train_predictions)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    
    # Save model and results
    timestamp = datetime.now().strftime("%Y_%m_%d")
    model_filename = f"models/random_forest_{timestamp}.joblib"
    
    os.makedirs('models/', exist_ok=True)
    joblib.dump(rf_model, model_filename)
    
    # Save feature importance
    feature_importance.to_csv(f"models/feature_importance_{timestamp}.csv", index=False)
    
    # Save training results
    training_results = {
        'model_type': 'RandomForestClassifier',
        'parameters': rf_params,
        'cv_scores': cv_scores.tolist(),
        'mean_cv_accuracy': float(cv_scores.mean()),
        'std_cv_accuracy': float(cv_scores.std()),
        'train_accuracy': float(train_accuracy),
        'feature_importance': feature_importance.to_dict('records'),
        'training_date': timestamp,
        'n_samples': len(X),
        'n_features': len(X.columns)
    }
    
    with open(f"models/training_results_{timestamp}.json", 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print(f"\nModel saved: {model_filename}")
    print(f"Feature importance saved: models/feature_importance_{timestamp}.csv")
    print(f"Training results saved: models/training_results_{timestamp}.json")
    
    return rf_model, training_results

def evaluate_baselines(y, target_info):
    """Calculate baseline accuracies for comparison"""
    print("\n=== Baseline Comparisons ===")
    
    # Random prediction (33.33%)
    random_accuracy = 1/3
    print(f"Random prediction: {random_accuracy:.3f}")
    
    # Majority class (Home wins)
    majority_class = y.mode()[0]  # Most frequent class
    majority_accuracy = (y == majority_class).mean()
    class_name = target_info['reverse_mapping'][str(majority_class)]
    print(f"Always predict {class_name}: {majority_accuracy:.3f}")
    
    # Weighted random by distribution
    class_distribution = y.value_counts(normalize=True)
    weighted_random = (class_distribution ** 2).sum()
    print(f"Weighted random: {weighted_random:.3f}")
    
    baselines = {
        'random': random_accuracy,
        'majority_class': majority_accuracy,
        'weighted_random': weighted_random
    }
    
    return baselines

if __name__ == "__main__":
    # Load data
    X, y, feature_names, target_info = load_ml_data()
    
    # Calculate baselines
    baselines = evaluate_baselines(y, target_info)
    
    # Train model
    model, results = train_random_forest(X, y, target_info)
    
    print(f"\n=== Training Complete ===")
    print(f"Model CV Accuracy: {results['mean_cv_accuracy']:.3f}")
    print(f"Beat Random (33.3%): {'✓' if results['mean_cv_accuracy'] > 0.333 else '✗'}")
    print(f"Beat Majority Class ({baselines['majority_class']:.3f}): {'✓' if results['mean_cv_accuracy'] > baselines['majority_class'] else '✗'}")
    print(f"Target: > 50% accuracy: {'✓' if results['mean_cv_accuracy'] > 0.5 else '✗'}")
    print(f"Model ready for evaluation!")